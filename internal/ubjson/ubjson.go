package ubjson

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// UBJSON marker constants
const (
	markerObjectStart   = '{'
	markerObjectEnd     = '}'
	markerArrayStart    = '['
	markerArrayEnd      = ']'
	markerTypeMarker    = '$'
	markerCountMarker   = '#'
	markerString        = 'S'
	markerInt8          = 'i'
	markerUint8         = 'U'
	markerInt16         = 'I'
	markerInt32         = 'l'
	markerInt64         = 'L'
	markerFloat32       = 'd'
	markerFloat64       = 'D'
	markerTrue          = 'T'
	markerFalse         = 'F'
	markerNull          = 'Z'
	markerNoop          = 'N'
	markerHighPrecision = 'H'
)

// TokenKind identifies the type of a decoded UBJSON token.
type TokenKind int

const (
	// TokObjectStart marks the beginning of an object (map).
	TokObjectStart TokenKind = iota
	// TokObjectEnd marks the end of an object.
	TokObjectEnd
	// TokArrayStart marks the beginning of an array.
	TokArrayStart
	// TokArrayEnd marks the end of an array.
	TokArrayEnd
	// TokKey is the key string of the next object field.
	TokKey
	// TokString is a string value.
	TokString
	// TokInt8 is an int8 value.
	TokInt8
	// TokUint8 is a uint8 value.
	TokUint8
	// TokInt16 is an int16 value.
	TokInt16
	// TokInt32 is an int32 value.
	TokInt32
	// TokInt64 is an int64 value.
	TokInt64
	// TokFloat32 is a float32 value.
	TokFloat32
	// TokFloat64 is a float64 value.
	TokFloat64
	// TokBool is a boolean value.
	TokBool
	// TokNull is a null value.
	TokNull
	// TokNoop is a no-op marker (ignored in object keys).
	TokNoop
	// TokInt32Slice is an optimized int32 array (bulk decoded).
	TokInt32Slice
	// TokUint8Slice is an optimized uint8 array (bulk decoded).
	TokUint8Slice
	// TokFloat32Slice is an optimized float32 array (bulk decoded).
	TokFloat32Slice
	// TokInt64Slice is an optimized int64 array (bulk decoded).
	TokInt64Slice
	// TokFloat64Slice is an optimized float64 array (bulk decoded).
	TokFloat64Slice
)

// Token holds a decoded UBJSON value.
type Token struct {
	Kind     TokenKind
	StrVal   string
	I8Val    int8
	U8Val    uint8
	I16Val   int16
	I32Val   int32
	I64Val   int64
	F32Val   float32
	F64Val   float64
	BoolVal  bool
	I32Slice []int32
	U8Slice  []uint8
	F32Slice []float32
	I64Slice []int64
	F64Slice []float64
}

// Decoder is a pull-based UBJSON decoder optimized for XGBoost model files.
// It reads UBJSON tokens sequentially from an io.Reader.
type Decoder struct {
	r          io.Reader
	buf        [8]byte
	peekedByte byte   // byte read ahead for format detection
	hasPeek    bool   // whether peekedByte is valid
	objDepth   int    // nesting depth of objects (for key detection)
	arrayDepth int    // nesting depth of arrays (for XGBoost value disambiguation)
	expectKey  bool   // next string in object context should be a key
	xgbMode    bool   // true when inside an XGBoost-style counted object
	xgbStack   []bool // stack tracking which object levels are XGBoost format
	streamPos  int    // position in stream for debugging
}

// NewDecoder creates a new UBJSON decoder reading from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

func (d *Decoder) readByte() (byte, error) {
	d.streamPos++
	if d.hasPeek {
		b := d.peekedByte
		d.hasPeek = false
		fmt.Printf("DEBUG readByte[%d]: using peeked byte 0x%02x ('%c')\n", d.streamPos, b, b)
		return b, nil
	}
	_, err := io.ReadFull(d.r, d.buf[:1])
	if err != nil {
		return 0, err
	}
	fmt.Printf("DEBUG readByte[%d]: read new byte 0x%02x ('%c')\n", d.streamPos, d.buf[0], d.buf[0])
	return d.buf[0], nil
}

// peekByte reads the next byte without consuming it.
// Stores in d.peekedByte so readByte() and readInt32/readFloat32 etc can use it.
// Subsequent calls to peekByte return the same byte.
func (d *Decoder) peekByte() (byte, error) {
	if d.hasPeek {
		return d.peekedByte, nil
	}
	_, err := io.ReadFull(d.r, d.buf[:1])
	if err != nil {
		return 0, err
	}
	d.peekedByte = d.buf[0]
	d.hasPeek = true
	return d.peekedByte, nil
}

func (d *Decoder) readFull(n int) ([]byte, error) {
	buf := make([]byte, n)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return nil, err
	}
	return buf, nil
}

// readInt32Fixed reads 4 bytes for an int32 value, handling the peek buffer correctly.
func (d *Decoder) readInt32Fixed() ([4]byte, error) {
	var buf [4]byte
	// Case 1: Peek buffer has data
	if d.hasPeek {
		buf[0] = d.peekedByte
		d.hasPeek = false
		// Read remaining 3 bytes directly from stream
		if _, err := io.ReadFull(d.r, buf[1:4]); err != nil {
			return buf, err
		}
		return buf, nil
	}

	// Case 2: Standard read via readFull helper
	data, err := d.readFull(4)
	if err != nil {
		return buf, err
	}
	copy(buf[:], data)
	return buf, nil
}

func (d *Decoder) readInt8() (int8, error) {
	b, err := d.readByte()
	if err != nil {
		return 0, err
	}
	return int8(b), nil
}

func (d *Decoder) readUint8() (uint8, error) {
	return d.readByte()
}

func (d *Decoder) readInt16() (int16, error) {
	_, err := io.ReadFull(d.r, d.buf[:2])
	if err != nil {
		return 0, err
	}
	return int16(binary.BigEndian.Uint16(d.buf[:2])), nil
}

func (d *Decoder) readInt32() (int32, error) {
	// Handle peeked byte first
	if d.hasPeek {
		d.buf[0] = d.peekedByte
		d.hasPeek = false
		_, err := io.ReadFull(d.r, d.buf[1:4])
		if err != nil {
			return 0, err
		}
	} else {
		_, err := io.ReadFull(d.r, d.buf[:4])
		if err != nil {
			return 0, err
		}
	}
	return int32(binary.BigEndian.Uint32(d.buf[:4])), nil
}

func (d *Decoder) readInt64() (int64, error) {
	// Handle peeked byte first
	if d.hasPeek {
		d.buf[0] = d.peekedByte
		d.hasPeek = false
		_, err := io.ReadFull(d.r, d.buf[1:8])
		if err != nil {
			return 0, err
		}
	} else {
		_, err := io.ReadFull(d.r, d.buf[:8])
		if err != nil {
			return 0, err
		}
	}
	result := int64(binary.BigEndian.Uint64(d.buf[:8]))
	return result, nil
}

func (d *Decoder) readFloat32() (float32, error) {
	// Handle peeked byte first
	if d.hasPeek {
		d.buf[0] = d.peekedByte
		d.hasPeek = false
		_, err := io.ReadFull(d.r, d.buf[1:4])
		if err != nil {
			return 0, err
		}
	} else {
		_, err := io.ReadFull(d.r, d.buf[:4])
		if err != nil {
			return 0, err
		}
	}
	bits := binary.BigEndian.Uint32(d.buf[:4])
	return math.Float32frombits(bits), nil
}

func (d *Decoder) readFloat64() (float64, error) {
	// Handle peeked byte first
	if d.hasPeek {
		d.buf[0] = d.peekedByte
		d.hasPeek = false
		_, err := io.ReadFull(d.r, d.buf[1:8])
		if err != nil {
			return 0, err
		}
	} else {
		_, err := io.ReadFull(d.r, d.buf[:8])
		if err != nil {
			return 0, err
		}
	}
	bits := binary.BigEndian.Uint64(d.buf[:8])
	return math.Float64frombits(bits), nil
}

// readCount reads a UBJSON element count (int8, int16, int32, or int64 depending on marker).
func (d *Decoder) readCount() (int64, error) {
	marker, err := d.readByte()
	if err != nil {
		return 0, err
	}
	switch marker {
	case markerInt8:
		v, err := d.readInt8()
		return int64(v), err
	case markerUint8:
		v, err := d.readUint8()
		return int64(v), err
	case markerInt16:
		v, err := d.readInt16()
		return int64(v), err
	case markerInt32:
		v, err := d.readInt32()
		return int64(v), err
	case markerInt64:
		return d.readInt64()
	}
	return 0, fmt.Errorf("ubjson: invalid count marker %q", marker)
}

// readCountWithMarker reads an integer value for a given marker byte.
// Unlike readCount, the marker byte is already consumed and passed as parameter.
func (d *Decoder) readCountWithMarker(marker byte) (int64, error) {
	switch marker {
	case markerInt8:
		v, err := d.readInt8()
		return int64(v), err
	case markerUint8:
		v, err := d.readUint8()
		return int64(v), err
	case markerInt16:
		v, err := d.readInt16()
		return int64(v), err
	case markerInt32:
		v, err := d.readInt32()
		return int64(v), err
	case markerInt64:
		return d.readInt64()
	}
	return 0, fmt.Errorf("ubjson: invalid int marker %q (0x%02x)", marker, marker)
}

// readString reads a length-prefixed string.
// Per UBJSON spec, the length is an int8 directly after the 'S' marker.
func (d *Decoder) readString() (string, error) {
	length, err := d.readInt8()
	if err != nil {
		return "", err
	}
	if length < 0 {
		return "", fmt.Errorf("ubjson: negative string length %d", length)
	}
	if length == 0 {
		return "", nil
	}
	buf, err := d.readFull(int(length))
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

// readTypedArray reads an optimized typed array: [$][type][#][count][raw_bytes...]
// This is the performance-critical path for XGBoost models.
func (d *Decoder) readTypedArray() (Token, error) {
	typeMarker, err := d.readByte()
	if err != nil {
		return Token{}, err
	}

	// Read count marker '#'
	countMarker, err := d.readByte()
	if err != nil {
		return Token{}, err
	}
	if countMarker != markerCountMarker {
		return Token{}, fmt.Errorf("ubjson: expected '#' after type marker, got %q", countMarker)
	}

	count, err := d.readCount()
	if err != nil {
		return Token{}, err
	}
	if count < 0 {
		return Token{}, fmt.Errorf("ubjson: negative array count %d", count)
	}

	switch typeMarker {
	case markerInt32:
		return d.readInt32Array(count)
	case markerUint8:
		return d.readUint8Array(count)
	case markerFloat32:
		return d.readFloat32Array(count)
	case markerInt64:
		return d.readInt64Array(count)
	case markerFloat64:
		return d.readFloat64Array(count)
	case markerInt8:
		return d.readInt8Array(count)
	case markerInt16:
		return d.readInt16Array(count)
	default:
		return Token{}, fmt.Errorf("ubjson: unsupported typed array type %q", typeMarker)
	}
}

func (d *Decoder) readInt8Array(count int64) (Token, error) {
	slice := make([]int8, count)
	for i := int64(0); i < count; i++ {
		v, err := d.readInt8()
		if err != nil {
			return Token{}, err
		}
		slice[i] = v
	}
	// Convert to int32 for consistency
	i32slice := make([]int32, count)
	for i, v := range slice {
		i32slice[i] = int32(v)
	}
	return Token{Kind: TokInt32Slice, I32Slice: i32slice}, nil
}

func (d *Decoder) readUint8Array(count int64) (Token, error) {
	slice := make([]uint8, count)
	for i := int64(0); i < count; i++ {
		v, err := d.readUint8()
		if err != nil {
			return Token{}, err
		}
		slice[i] = v
	}
	return Token{Kind: TokUint8Slice, U8Slice: slice}, nil
}

func (d *Decoder) readInt16Array(count int64) (Token, error) {
	slice := make([]int16, count)
	for i := int64(0); i < count; i++ {
		v, err := d.readInt16()
		if err != nil {
			return Token{}, err
		}
		slice[i] = v
	}
	// Convert to int32
	i32slice := make([]int32, count)
	for i, v := range slice {
		i32slice[i] = int32(v)
	}
	return Token{Kind: TokInt32Slice, I32Slice: i32slice}, nil
}

func (d *Decoder) readInt32Array(count int64) (Token, error) {
	slice := make([]int32, count)
	buf := make([]byte, count*4)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return Token{}, err
	}
	for i := int64(0); i < count; i++ {
		slice[i] = int32(binary.BigEndian.Uint32(buf[i*4 : (i+1)*4]))
	}
	return Token{Kind: TokInt32Slice, I32Slice: slice}, nil
}

func (d *Decoder) readInt64Array(count int64) (Token, error) {
	slice := make([]int64, count)
	buf := make([]byte, count*8)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return Token{}, err
	}
	for i := int64(0); i < count; i++ {
		slice[i] = int64(binary.BigEndian.Uint64(buf[i*8 : (i+1)*8]))
	}
	return Token{Kind: TokInt64Slice, I64Slice: slice}, nil
}

func (d *Decoder) readFloat32Array(count int64) (Token, error) {
	slice := make([]float32, count)
	buf := make([]byte, count*4)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return Token{}, err
	}
	for i := int64(0); i < count; i++ {
		bits := binary.BigEndian.Uint32(buf[i*4 : (i+1)*4])
		slice[i] = math.Float32frombits(bits)
	}
	return Token{Kind: TokFloat32Slice, F32Slice: slice}, nil
}

func (d *Decoder) readFloat64Array(count int64) (Token, error) {
	slice := make([]float64, count)
	buf := make([]byte, count*8)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return Token{}, err
	}
	for i := int64(0); i < count; i++ {
		bits := binary.BigEndian.Uint64(buf[i*8 : (i+1)*8])
		slice[i] = math.Float64frombits(bits)
	}
	return Token{Kind: TokFloat64Slice, F64Slice: slice}, nil
}

// readValue reads a single non-optimized UBJSON value.
func (d *Decoder) readValue(marker byte) (Token, error) {
	switch marker {
	case markerString:
		s, err := d.readString()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokString, StrVal: s}, nil
	case markerInt8:
		v, err := d.readInt8()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokInt8, I8Val: v}, nil
	case markerUint8:
		v, err := d.readUint8()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokUint8, U8Val: v}, nil
	case markerInt16:
		v, err := d.readInt16()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokInt16, I16Val: v}, nil
	case markerInt32:
		// In XGBoost format, 'l' is ambiguous - it can be:
		// 1. An int32 value (l <4-byte value>)
		// 2. A string length prefix (l <4-byte length> <bytes>)
		// We disambiguate by peeking at the next byte after the 4 length/value bytes.
		// If it's '$' or another type marker, it's int32. Otherwise, check if the
		// bytes form a plausible string length by looking at what follows.
		if d.xgbMode && !d.expectKey {
			// Read the 4 bytes that follow 'l' using the helper that handles peek buffer
			fourBytes, err := d.readInt32Fixed()
			if err != nil {
				return Token{}, err
			}
			// Peek at the next byte to determine if this is a String length
			peek, peekErr := d.peekByte()
			fmt.Printf("DEBUG markerInt32: fourBytes=%v, peek=0x%02x(%q), peekErr=%v\n", fourBytes, peek, string(peek), peekErr)
			if peekErr == nil {
				// If the next byte is a known marker (not data), treat as int32
				if peek == markerTypeMarker || peek == markerObjectStart ||
					peek == markerObjectEnd || peek == markerArrayStart ||
					peek == markerArrayEnd {
					// This is an int32 value
					fmt.Printf("DEBUG markerInt32: returning TokInt32(val=%d)\n", int32(binary.BigEndian.Uint32(fourBytes[:])))
					return Token{Kind: TokInt32, I32Val: int32(binary.BigEndian.Uint32(fourBytes[:]))}, nil
				}
				// Otherwise, check if the bytes form a valid string length
				// by seeing if we can read that many bytes as valid data
				length := int(binary.BigEndian.Uint32(fourBytes[:]))
				if length > 0 && length < 10000 {
					// Try to peek more bytes to validate
					// For now, just check if the first byte looks like text
					if (peek >= 'a' && peek <= 'z') || (peek >= 'A' && peek <= 'Z') ||
						(peek >= '0' && peek <= '9') || peek == '_' || peek == '.' {
						// Looks like string data - this is a string length
						// Read the rest of the string (length-1 bytes after the peeked one)
						rest := length - 1
						var buf []byte
						if rest > 0 {
							restBuf, err := d.readFull(rest)
							if err != nil {
								return Token{}, err
							}
							buf = append([]byte{peek}, restBuf...)
						} else {
							buf = []byte{peek}
						}
						d.hasPeek = false // consumed the peek
						return Token{Kind: TokString, StrVal: string(buf)}, nil
					}
				}
				// Fallback: treat as int32
				fmt.Printf("DEBUG markerInt32: fallback - peek=0x%02x not a marker, not text\n", peek)
				d.peekedByte = peek
				d.hasPeek = true
			}
			fmt.Printf("DEBUG markerInt32: returning TokInt32(val=%d), peekErr=%v\n", int32(binary.BigEndian.Uint32(fourBytes[:])), peekErr)
			return Token{Kind: TokInt32, I32Val: int32(binary.BigEndian.Uint32(fourBytes[:]))}, nil
		}
		v, err := d.readInt32()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokInt32, I32Val: v}, nil
	case markerInt64:
		v, err := d.readInt64()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokInt64, I64Val: v}, nil
	case markerFloat32:
		v, err := d.readFloat32()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokFloat32, F32Val: v}, nil
	case markerFloat64:
		v, err := d.readFloat64()
		if err != nil {
			return Token{}, err
		}
		return Token{Kind: TokFloat64, F64Val: v}, nil
	case markerTrue:
		return Token{Kind: TokBool, BoolVal: true}, nil
	case markerFalse:
		return Token{Kind: TokBool, BoolVal: false}, nil
	case markerNull:
		return Token{Kind: TokNull}, nil
	case markerNoop:
		return Token{Kind: TokNoop}, nil
	case markerHighPrecision:
		// High-precision is followed by a full string token (S + length + bytes)
		marker2, err := d.readByte()
		if err != nil {
			return Token{}, err
		}
		if marker2 != markerString {
			return Token{}, fmt.Errorf("ubjson: expected string after high-precision marker, got %q", marker2)
		}
		s, err := d.readString()
		if err != nil {
			return Token{}, err
		}
		// Treat high-precision as string (XGBoost uses this rarely)
		return Token{Kind: TokString, StrVal: s}, nil
	case markerObjectStart:
		return Token{Kind: TokObjectStart}, nil
	case markerObjectEnd:
		return Token{Kind: TokObjectEnd}, nil
	case markerArrayStart:
		return Token{Kind: TokArrayStart}, nil
	case markerArrayEnd:
		return Token{Kind: TokArrayEnd}, nil
	default:
		return Token{}, fmt.Errorf("ubjson: unknown marker %q (0x%02x)", marker, marker)
	}
}

// Next reads and returns the next UBJSON token from the stream.
// It handles optimized typed arrays automatically by returning Tok*Slice tokens.
// It also tracks object nesting context and returns TokKey for strings inside objects.
// Returns io.EOF when the stream is exhausted.
func (d *Decoder) Next() (Token, error) {
	fmt.Printf("DEBUG Next: CALLED, hasPeek=%v, peekedByte=0x%02x\n", d.hasPeek, d.peekedByte)
	marker, err := d.readByte()
	if err != nil {
		return Token{}, err
	}
	fmt.Printf("DEBUG Next: marker=0x%02x ('%c'), xgbMode=%v, expectKey=%v, hasPeek=%v\n", marker, marker, d.xgbMode, d.expectKey, d.hasPeek)

	// Check for optimized type/count container: [$]
	if marker == markerTypeMarker {
		tok, err := d.readTypedArray()
		if err != nil {
			return tok, err
		}
		// Apply state tracking for optimized arrays (they are values, not keys)
		d.expectKey = d.objDepth > 0
		return tok, nil
	}

	// In XGBoost mode, when expecting a key, the marker could be:
	// 1. 'l' - string length prefix or typed array '$l' (peeking ahead)
	// 2. '$' - start of typed array (unoptimized form)
	// 3. Other markers for string/int/float values
	// Check for typed arrays that start with type markers before
	// treating them as XGBoost string length prefixes.
	if d.xgbMode && d.expectKey {
		// Peek ahead to check for typed array: '$' followed by type marker
		if marker == markerTypeMarker || marker == markerInt32 || marker == markerFloat32 {
			// Peek at the next byte to determine if this is a typed array
			peek, peekErr := d.peekByte()
			if peekErr == nil && peek == markerTypeMarker {
				// This is a typed array in the form: $ <type> # <count>
				// Read the '$' that we already consumed, then read the array
				tok, err := d.readTypedArray()
				if err != nil {
					return tok, err
				}
				d.expectKey = d.objDepth > 0
				return tok, nil
			}
		}
	}

	// XGBoost mode: when expecting a key, the marker is a type indicator
	// for a raw length prefix (l <4 bytes>, i <1 byte>, etc.)
	// NOTE: When NOT expecting a key (i.e., reading a value), we fall through
	// to readValue which handles 'l' via readString() for string values.
	if d.xgbMode && d.expectKey {
		if marker == markerObjectEnd {
			// Object end: handle directly since XGBoost branch bypasses readValue
			tok, err := d.readValue(marker)
			if err != nil {
				return tok, err
			}
			d.objDepth--
			// Pop xgb stack and restore mode for enclosing object
			if len(d.xgbStack) > 0 {
				poppedMode := d.xgbStack[len(d.xgbStack)-1]
				d.xgbStack = d.xgbStack[:len(d.xgbStack)-1]
				d.xgbMode = poppedMode
			}
			// After exiting object: expect a key only if inside an object (not an array)
			// When arrayDepth > 0, we're inside an array and should NOT expect keys
			fmt.Printf("DEBUG ObjectEnd: objDepth=%d, arrayDepth=%d\n", d.objDepth, d.arrayDepth)
			if d.arrayDepth > 0 {
				d.expectKey = false
				fmt.Println("DEBUG ObjectEnd: setting expectKey=false (inside array)")
			} else if d.objDepth > 0 {
				d.expectKey = true
				fmt.Println("DEBUG ObjectEnd: setting expectKey=true (inside object)")
			} else {
				d.expectKey = false
				fmt.Println("DEBUG ObjectEnd: setting expectKey=false (at root)")
			}
			return tok, nil
		}
		if marker == markerArrayEnd {
			// Array end: handle directly since XGBoost branch bypasses readValue
			fmt.Printf("DEBUG ArrayEnd (xgb): marker=0x%02x, objDepth=%d, arrayDepth=%d\n", marker, d.objDepth, d.arrayDepth)
			tok, err := d.readValue(marker)
			if err != nil {
				return tok, err
			}
			d.arrayDepth--
			// Pop xgb stack (matching the push in TokArrayStart)
			if len(d.xgbStack) > 0 {
				d.xgbStack = d.xgbStack[:len(d.xgbStack)-1]
			}
			// After exiting array, if we're in an object context, expect a key next
			if d.objDepth > 0 && d.arrayDepth == 0 {
				d.expectKey = true
			} else {
				d.expectKey = false
			}
			fmt.Printf("DEBUG ArrayEnd (xgb): after handling, expectKey=%v\n", d.expectKey)
			return tok, nil
		}
		length, readErr := d.readCountWithMarker(marker)
		if readErr != nil {
			return Token{}, fmt.Errorf("reading XGBoost key length (marker=0x%02x): %w", marker, readErr)
		}
		buf, readErr := d.readFull(int(length))
		if readErr != nil {
			return Token{}, fmt.Errorf("reading XGBoost key bytes (length=%d): %w", length, readErr)
		}
		d.expectKey = false
		return Token{Kind: TokKey, StrVal: string(buf)}, nil
	}

	tok, err := d.readValue(marker)
	if err != nil {
		return tok, err
	}
	fmt.Printf("DEBUG Next: after readValue, tok.Kind=%d, hasPeek=%v, peekedByte=0x%02x\n", tok.Kind, d.hasPeek, d.peekedByte)

	// Track object nesting
	switch tok.Kind {
	case TokObjectStart:
		d.objDepth++
		d.expectKey = true
		// XGBoost UBJSON format detection: {L <8-byte count>
		// XGBoost writes object counts as raw int64 without '#' marker.
		// Keys/values are <int-marker><length><bytes> without 'S' marker.
		// XGBoost format can appear at any nesting level - detect it when we see it.
		next, peekErr := d.peekByte()
		if peekErr == nil && next == markerInt64 {
			d.readByte() // consume the 'L'
			if _, readErr := d.readInt64(); readErr != nil {
				return Token{}, fmt.Errorf("reading XGBoost object count: %w", readErr)
			}
			// Detected XGBoost format
			// Push current mode to stack, then set mode to true
			fmt.Printf("DEBUG ObjectStart(XGB): push xgbMode=%v, set xgbMode=true, stack was %v\n", d.xgbMode, d.xgbStack)
			d.xgbStack = append(d.xgbStack, d.xgbMode)
			d.xgbMode = true
			d.expectKey = true
		} else {
			// Standard object: push current xgbMode to stack, set to false
			// When we exit this object, xgbMode will be restored from stack
			fmt.Printf("DEBUG ObjectStart(std): push xgbMode=%v, set xgbMode=false, stack was %v\n", d.xgbMode, d.xgbStack)
			d.xgbStack = append(d.xgbStack, d.xgbMode)
			d.xgbMode = false
		}
	case TokObjectEnd:
		fmt.Printf("DEBUG ObjectEnd: objDepth=%d, xgbMode=%v, stack=%v\n", d.objDepth, d.xgbMode, d.xgbStack)
		d.objDepth--
		d.expectKey = d.objDepth > 0
		// Pop xgb stack and restore mode for the enclosing object
		if len(d.xgbStack) > 0 {
			poppedMode := d.xgbStack[len(d.xgbStack)-1]
			d.xgbStack = d.xgbStack[:len(d.xgbStack)-1]
			fmt.Printf("DEBUG ObjectEnd: pop %v, restore xgbMode to %v, stack now %v\n", poppedMode, poppedMode, d.xgbStack)
			d.xgbMode = poppedMode
		} else {
			fmt.Printf("DEBUG ObjectEnd: stack empty!\n")
		}
	case TokArrayStart:
		d.arrayDepth++
		d.xgbStack = append(d.xgbStack, false) // arrays are never XGBoost
	case TokArrayEnd:
		d.arrayDepth--
		// Pop xgb stack (matching the push in TokArrayStart)
		if len(d.xgbStack) > 0 {
			d.xgbStack = d.xgbStack[:len(d.xgbStack)-1]
		}
		// After exiting an array, if we're in an object context, expect a key next
		if d.objDepth > 0 && d.arrayDepth == 0 {
			d.expectKey = true
		}
	case TokString:
		if d.expectKey {
			tok.Kind = TokKey
			d.expectKey = false
		} else {
			// String value consumed, next token in object should be a key
			d.expectKey = d.objDepth > 0
		}
	default:
		// After any non-container value in an object, expect a key next.
		// Note: typed arrays ($l, $d, etc.) bypass readValue via readTypedArray and
		// don't use arrayDepth, so we must always set expectKey inside objects.
		if d.objDepth > 0 {
			d.expectKey = true
		}
	}

	return tok, nil
}

// SkipValue skips an entire UBJSON value (object, array, or primitive).
// This is useful for ignoring unknown fields during parsing.
func (d *Decoder) SkipValue() error {
	tok, err := d.Next()
	if err != nil {
		return err
	}
	switch tok.Kind {
	case TokObjectStart:
		return d.skipObject()
	case TokArrayStart:
		return d.skipArray()
	}
	// XGBoost mode: int tokens are string length prefixes, skip the bytes
	if d.xgbMode {
		var length int64
		switch tok.Kind {
		case TokInt8:
			length = int64(tok.I8Val)
		case TokUint8:
			length = int64(tok.U8Val)
		case TokInt16:
			length = int64(tok.I16Val)
		case TokInt32:
			length = int64(tok.I32Val)
		case TokInt64:
			length = tok.I64Val
		default:
			return nil // non-int primitives need no extra skipping
		}
		_, err := d.readFull(int(length))
		return err
	}
	// Primitive tokens need no additional skipping.
	return nil
}

func (d *Decoder) skipObject() error {
	for {
		tok, err := d.Next()
		if err != nil {
			return err
		}
		switch tok.Kind {
		case TokObjectEnd:
			return nil
		case TokKey:
			// Skip the value after the key.
			if err := d.SkipValue(); err != nil {
				return err
			}
		default:
			return fmt.Errorf("ubjson: expected object key or end, got %v", tok.Kind)
		}
	}
}

func (d *Decoder) skipArray() error {
	for {
		tok, err := d.Next()
		if err != nil {
			return err
		}
		switch tok.Kind {
		case TokArrayEnd:
			return nil
		default:
			// For non-optimized tokens, if it's a container start we need to skip recursively.
			if tok.Kind == TokObjectStart {
				if err := d.skipObject(); err != nil {
					return err
				}
			} else if tok.Kind == TokArrayStart {
				if err := d.skipArray(); err != nil {
					return err
				}
			}
			// Primitive and optimized array tokens are already fully consumed.
		}
	}
}

// ExpectObjectStart reads the next token and verifies it is an object start.
func (d *Decoder) ExpectObjectStart() error {
	tok, err := d.Next()
	if err != nil {
		return err
	}
	if tok.Kind != TokObjectStart {
		return fmt.Errorf("ubjson: expected object start, got %v", tok.Kind)
	}
	return nil
}

// ExpectArrayStart reads the next token and verifies it is an array start.
func (d *Decoder) ExpectArrayStart() error {
	tok, err := d.Next()
	if err != nil {
		return err
	}
	if tok.Kind != TokArrayStart {
		return fmt.Errorf("ubjson: expected array start, got %v", tok.Kind)
	}
	return nil
}

// ExpectKey reads the next token and verifies it is a key, returning the key string.
func (d *Decoder) ExpectKey() (string, error) {
	tok, err := d.Next()
	if err != nil {
		return "", err
	}
	if tok.Kind != TokKey {
		return "", fmt.Errorf("ubjson: expected object key, got %v", tok.Kind)
	}
	return tok.StrVal, nil
}

// ReadInt32 reads the next token as an int32 value.
func (d *Decoder) ReadInt32() (int32, error) {
	tok, err := d.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case TokInt8:
		return int32(tok.I8Val), nil
	case TokUint8:
		return int32(tok.U8Val), nil
	case TokInt16:
		return int32(tok.I16Val), nil
	case TokInt32:
		return tok.I32Val, nil
	case TokInt64:
		return int32(tok.I64Val), nil
	}
	return 0, fmt.Errorf("ubjson: expected integer, got %v", tok.Kind)
}

// ReadInt64 reads the next token as an int64 value.
func (d *Decoder) ReadInt64() (int64, error) {
	tok, err := d.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case TokInt8:
		return int64(tok.I8Val), nil
	case TokUint8:
		return int64(tok.U8Val), nil
	case TokInt16:
		return int64(tok.I16Val), nil
	case TokInt32:
		return int64(tok.I32Val), nil
	case TokInt64:
		return tok.I64Val, nil
	}
	return 0, fmt.Errorf("ubjson: expected integer, got %v", tok.Kind)
}

// ReadFloat32 reads the next token as a float32 value.
func (d *Decoder) ReadFloat32() (float32, error) {
	tok, err := d.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case TokFloat32:
		return tok.F32Val, nil
	case TokFloat64:
		return float32(tok.F64Val), nil
	case TokInt8:
		return float32(tok.I8Val), nil
	case TokUint8:
		return float32(tok.U8Val), nil
	case TokInt16:
		return float32(tok.I16Val), nil
	case TokInt32:
		return float32(tok.I32Val), nil
	case TokInt64:
		return float32(tok.I64Val), nil
	}
	return 0, fmt.Errorf("ubjson: expected number, got %v", tok.Kind)
}

// ReadFloat64 reads the next token as a float64 value.
func (d *Decoder) ReadFloat64() (float64, error) {
	tok, err := d.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case TokFloat32:
		return float64(tok.F32Val), nil
	case TokFloat64:
		return tok.F64Val, nil
	case TokInt8:
		return float64(tok.I8Val), nil
	case TokUint8:
		return float64(tok.U8Val), nil
	case TokInt16:
		return float64(tok.I16Val), nil
	case TokInt32:
		return float64(tok.I32Val), nil
	case TokInt64:
		return float64(tok.I64Val), nil
	}
	return 0, fmt.Errorf("ubjson: expected number, got %v", tok.Kind)
}

// ReadString reads the next token as a string value.
// In XGBoost mode, integer tokens represent string length prefixes —
// the method reads that many raw bytes from the stream.
func (d *Decoder) ReadString() (string, error) {
	tok, err := d.Next()
	if err != nil {
		return "", err
	}
	switch tok.Kind {
	case TokString:
		return tok.StrVal, nil
	// XGBoost mode: int tokens after keys represent string length prefixes
	case TokInt32:
		buf, err := d.readFull(int(tok.I32Val))
		if err != nil {
			return "", err
		}
		return string(buf), nil
	case TokInt8:
		buf, err := d.readFull(int(tok.I8Val))
		if err != nil {
			return "", err
		}
		return string(buf), nil
	case TokUint8:
		buf, err := d.readFull(int(tok.U8Val))
		if err != nil {
			return "", err
		}
		return string(buf), nil
	case TokInt16:
		buf, err := d.readFull(int(tok.I16Val))
		if err != nil {
			return "", err
		}
		return string(buf), nil
	case TokInt64:
		buf, err := d.readFull(int(tok.I64Val))
		if err != nil {
			return "", err
		}
		return string(buf), nil
	}
	return "", fmt.Errorf("ubjson: expected string, got %v", tok.Kind)
}

// ReadInt32Slice reads an optimized int32 array token.
func (d *Decoder) ReadInt32Slice() ([]int32, error) {
	tok, err := d.Next()
	if err != nil {
		return nil, err
	}
	if tok.Kind == TokInt32Slice {
		return tok.I32Slice, nil
	}
	return nil, fmt.Errorf("ubjson: expected int32 array, got %v", tok.Kind)
}

// ReadUint8Slice reads an optimized uint8 array token.
func (d *Decoder) ReadUint8Slice() ([]uint8, error) {
	tok, err := d.Next()
	if err != nil {
		return nil, err
	}
	if tok.Kind == TokUint8Slice {
		return tok.U8Slice, nil
	}
	return nil, fmt.Errorf("ubjson: expected uint8 array, got %v", tok.Kind)
}

// ReadFloat32Slice reads an optimized float32 array token.
func (d *Decoder) ReadFloat32Slice() ([]float32, error) {
	tok, err := d.Next()
	if err != nil {
		return nil, err
	}
	if tok.Kind == TokFloat32Slice {
		return tok.F32Slice, nil
	}
	return nil, fmt.Errorf("ubjson: expected float32 array, got %v", tok.Kind)
}

// ReadBytes reads exactly n raw bytes from the stream.
// This is useful for reading XGBoost string values where the length
// was already obtained from a Next() call returning TokInt32.
func (d *Decoder) ReadBytes(n int) ([]byte, error) {
	return d.readFull(n)
}
