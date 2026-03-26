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
	r         io.Reader
	buf       [8]byte
	objDepth  int  // nesting depth of objects (for key detection)
	expectKey bool // next string in object context should be a key
}

// NewDecoder creates a new UBJSON decoder reading from r.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{r: r}
}

func (d *Decoder) readByte() (byte, error) {
	_, err := io.ReadFull(d.r, d.buf[:1])
	if err != nil {
		return 0, err
	}
	return d.buf[0], nil
}

func (d *Decoder) readFull(n int) ([]byte, error) {
	buf := make([]byte, n)
	_, err := io.ReadFull(d.r, buf)
	if err != nil {
		return nil, err
	}
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
	_, err := io.ReadFull(d.r, d.buf[:4])
	if err != nil {
		return 0, err
	}
	return int32(binary.BigEndian.Uint32(d.buf[:4])), nil
}

func (d *Decoder) readInt64() (int64, error) {
	_, err := io.ReadFull(d.r, d.buf[:8])
	if err != nil {
		return 0, err
	}
	return int64(binary.BigEndian.Uint64(d.buf[:8])), nil
}

func (d *Decoder) readFloat32() (float32, error) {
	_, err := io.ReadFull(d.r, d.buf[:4])
	if err != nil {
		return 0, err
	}
	bits := binary.BigEndian.Uint32(d.buf[:4])
	return math.Float32frombits(bits), nil
}

func (d *Decoder) readFloat64() (float64, error) {
	_, err := io.ReadFull(d.r, d.buf[:8])
	if err != nil {
		return 0, err
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
	marker, err := d.readByte()
	if err != nil {
		return Token{}, err
	}

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

	tok, err := d.readValue(marker)
	if err != nil {
		return tok, err
	}

	// Track object nesting
	switch tok.Kind {
	case TokObjectStart:
		d.objDepth++
		d.expectKey = true
	case TokObjectEnd:
		d.objDepth--
		d.expectKey = d.objDepth > 0
	case TokString:
		if d.expectKey {
			tok.Kind = TokKey
			d.expectKey = false
		} else {
			// String value consumed, next token in object should be a key
			d.expectKey = d.objDepth > 0
		}
	default:
		// After any non-container value in an object, expect a key next
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
func (d *Decoder) ReadString() (string, error) {
	tok, err := d.Next()
	if err != nil {
		return "", err
	}
	if tok.Kind != TokString {
		return "", fmt.Errorf("ubjson: expected string, got %v", tok.Kind)
	}
	return tok.StrVal, nil
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
