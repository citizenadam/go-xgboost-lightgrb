package ubjson

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"testing"
)

func TestDecoderPrimitives(t *testing.T) {
	tests := []struct {
		name  string
		input []byte
		check func(t *testing.T, tok Token)
	}{
		{
			name:  "null",
			input: []byte{markerNull},
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokNull {
					t.Errorf("expected TokNull, got %v", tok.Kind)
				}
			},
		},
		{
			name:  "true",
			input: []byte{markerTrue},
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokBool || !tok.BoolVal {
					t.Errorf("expected true bool, got %v", tok)
				}
			},
		},
		{
			name:  "false",
			input: []byte{markerFalse},
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokBool || tok.BoolVal {
					t.Errorf("expected false bool, got %v", tok)
				}
			},
		},
		{
			name:  "int8",
			input: []byte{markerInt8, 0x80}, // -128
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokInt8 || tok.I8Val != -128 {
					t.Errorf("expected int8 -128, got %v", tok)
				}
			},
		},
		{
			name:  "uint8",
			input: []byte{markerUint8, 255},
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokUint8 || tok.U8Val != 255 {
					t.Errorf("expected uint8 255, got %v", tok)
				}
			},
		},
		{
			name:  "int16",
			input: []byte{markerInt16, 0x01, 0x00}, // 256
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokInt16 || tok.I16Val != 256 {
					t.Errorf("expected int16 256, got %v", tok)
				}
			},
		},
		{
			name: "int32",
			input: func() []byte {
				buf := make([]byte, 5)
				buf[0] = markerInt32
				binary.BigEndian.PutUint32(buf[1:], 1000000)
				return buf
			}(),
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokInt32 || tok.I32Val != 1000000 {
					t.Errorf("expected int32 1000000, got %v", tok)
				}
			},
		},
		{
			name: "int64",
			input: func() []byte {
				buf := make([]byte, 9)
				buf[0] = markerInt64
				binary.BigEndian.PutUint64(buf[1:], 1234567890)
				return buf
			}(),
			check: func(t *testing.T, tok Token) {
				if tok.Kind != TokInt64 || tok.I64Val != 1234567890 {
					t.Errorf("expected int64 1234567890, got %v", tok)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDecoder(bytes.NewReader(tt.input))
			tok, err := d.Next()
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.check(t, tok)
		})
	}
}

func TestDecoderString(t *testing.T) {
	// String: 'S' + length(int8) + bytes
	input := []byte{markerString, 5, 'h', 'e', 'l', 'l', 'o'}
	d := NewDecoder(bytes.NewReader(input))
	tok, err := d.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok.Kind != TokString || tok.StrVal != "hello" {
		t.Errorf("expected string 'hello', got %v", tok)
	}
}

func TestDecoderFloat32(t *testing.T) {
	// float32: 'd' + 4 bytes (big-endian IEEE 754)
	input := make([]byte, 5)
	input[0] = markerFloat32
	binary.BigEndian.PutUint32(input[1:], float32ToBits(3.14))

	d := NewDecoder(bytes.NewReader(input))
	tok, err := d.Next()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if tok.Kind != TokFloat32 {
		t.Errorf("expected TokFloat32, got %v", tok.Kind)
	}
	// Allow small float32 precision error
	if tok.F32Val < 3.13 || tok.F32Val > 3.15 {
		t.Errorf("expected float32 ~3.14, got %v", tok.F32Val)
	}
}

func float32ToBits(f float32) uint32 {
	return math.Float32bits(f)
}

func TestDecoderObject(t *testing.T) {
	// Object: { "name": "S" + "test", "value": int32 42 }
	var buf bytes.Buffer
	buf.WriteByte(markerObjectStart)
	// key "name"
	buf.WriteByte(markerString)
	buf.WriteByte(4)
	buf.WriteString("name")
	// value "test"
	buf.WriteByte(markerString)
	buf.WriteByte(4)
	buf.WriteString("test")
	// key "value"
	buf.WriteByte(markerString)
	buf.WriteByte(5)
	buf.WriteString("value")
	// value 42
	buf.WriteByte(markerInt32)
	valBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(valBuf, 42)
	buf.Write(valBuf)
	buf.WriteByte(markerObjectEnd)

	d := NewDecoder(&buf)

	// ObjectStart
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectStart {
		t.Fatalf("expected object start, got %v", tok.Kind)
	}

	// Key "name"
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokKey || tok.StrVal != "name" {
		t.Fatalf("expected key 'name', got %v", tok)
	}

	// Value "test"
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokString || tok.StrVal != "test" {
		t.Fatalf("expected string 'test', got %v", tok)
	}

	// Key "value"
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokKey || tok.StrVal != "value" {
		t.Fatalf("expected key 'value', got %v", tok)
	}

	// Value 42
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt32 || tok.I32Val != 42 {
		t.Fatalf("expected int32 42, got %v", tok)
	}

	// ObjectEnd
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectEnd {
		t.Fatalf("expected object end, got %v", tok.Kind)
	}
}

func TestDecoderArray(t *testing.T) {
	// Array: [1, 2, 3]
	var buf bytes.Buffer
	buf.WriteByte(markerArrayStart)
	buf.WriteByte(markerInt32)
	valBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(valBuf, 1)
	buf.Write(valBuf)
	buf.WriteByte(markerInt32)
	binary.BigEndian.PutUint32(valBuf, 2)
	buf.Write(valBuf)
	buf.WriteByte(markerInt32)
	binary.BigEndian.PutUint32(valBuf, 3)
	buf.Write(valBuf)
	buf.WriteByte(markerArrayEnd)

	d := NewDecoder(&buf)

	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokArrayStart {
		t.Fatalf("expected array start, got %v", tok.Kind)
	}

	for i := int32(1); i <= 3; i++ {
		tok, err = d.Next()
		if err != nil {
			t.Fatal(err)
		}
		if tok.Kind != TokInt32 || tok.I32Val != i {
			t.Fatalf("expected int32 %d, got %v", i, tok)
		}
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokArrayEnd {
		t.Fatalf("expected array end, got %v", tok.Kind)
	}
}

func TestDecoderTypedInt32Array(t *testing.T) {
	// Optimized int32 array: [$][l][#][count][raw_bytes]
	// [$][l][#][l][3][00 00 00 0A][00 00 00 14][00 00 00 1E] = [10, 20, 30]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)  // $
	buf.WriteByte(markerInt32)       // l
	buf.WriteByte(markerCountMarker) // #
	buf.WriteByte(markerInt32)       // l (count type)
	countBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(countBuf, 3)
	buf.Write(countBuf)

	// values: 10, 20, 30
	valBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(valBuf, 10)
	buf.Write(valBuf)
	binary.BigEndian.PutUint32(valBuf, 20)
	buf.Write(valBuf)
	binary.BigEndian.PutUint32(valBuf, 30)
	buf.Write(valBuf)

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt32Slice {
		t.Fatalf("expected int32 slice, got %v", tok.Kind)
	}
	if len(tok.I32Slice) != 3 {
		t.Fatalf("expected slice length 3, got %d", len(tok.I32Slice))
	}
	expected := []int32{10, 20, 30}
	for i, v := range expected {
		if tok.I32Slice[i] != v {
			t.Errorf("expected [%d] = %d, got %d", i, v, tok.I32Slice[i])
		}
	}
}

func TestDecoderTypedUint8Array(t *testing.T) {
	// Optimized uint8 array: [$][U][#][l][3][01][00][01] = [1, 0, 1]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)
	buf.WriteByte(markerUint8)
	buf.WriteByte(markerCountMarker)
	buf.WriteByte(markerInt32)
	countBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(countBuf, 3)
	buf.Write(countBuf)
	buf.WriteByte(1)
	buf.WriteByte(0)
	buf.WriteByte(1)

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokUint8Slice {
		t.Fatalf("expected uint8 slice, got %v", tok.Kind)
	}
	if len(tok.U8Slice) != 3 {
		t.Fatalf("expected slice length 3, got %d", len(tok.U8Slice))
	}
	expected := []uint8{1, 0, 1}
	for i, v := range expected {
		if tok.U8Slice[i] != v {
			t.Errorf("expected [%d] = %d, got %d", i, v, tok.U8Slice[i])
		}
	}
}

func TestDecoderTypedFloat32Array(t *testing.T) {
	// Optimized float32 array: [$][d][#][l][2][<float32 bytes>][<float32 bytes>]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)
	buf.WriteByte(markerFloat32)
	buf.WriteByte(markerCountMarker)
	buf.WriteByte(markerInt32)
	countBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(countBuf, 2)
	buf.Write(countBuf)

	// values: 1.5, 2.5
	valBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(valBuf, float32ToBits(1.5))
	buf.Write(valBuf)
	binary.BigEndian.PutUint32(valBuf, float32ToBits(2.5))
	buf.Write(valBuf)

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokFloat32Slice {
		t.Fatalf("expected float32 slice, got %v", tok.Kind)
	}
	if len(tok.F32Slice) != 2 {
		t.Fatalf("expected slice length 2, got %d", len(tok.F32Slice))
	}
	if tok.F32Slice[0] < 1.4 || tok.F32Slice[0] > 1.6 {
		t.Errorf("expected [0] ~1.5, got %v", tok.F32Slice[0])
	}
	if tok.F32Slice[1] < 2.4 || tok.F32Slice[1] > 2.6 {
		t.Errorf("expected [1] ~2.5, got %v", tok.F32Slice[1])
	}
}

func TestDecoderEOF(t *testing.T) {
	d := NewDecoder(bytes.NewReader(nil))
	_, err := d.Next()
	if err != io.EOF {
		t.Errorf("expected io.EOF, got %v", err)
	}
}

func TestDecoderNestedObject(t *testing.T) {
	// { "a": { "b": 1 } }
	var buf bytes.Buffer
	buf.WriteByte(markerObjectStart)

	// key "a"
	buf.WriteByte(markerString)
	buf.WriteByte(1)
	buf.WriteByte('a')
	// value: nested object
	buf.WriteByte(markerObjectStart)
	// key "b"
	buf.WriteByte(markerString)
	buf.WriteByte(1)
	buf.WriteByte('b')
	// value: 1
	buf.WriteByte(markerInt8)
	buf.WriteByte(1)
	buf.WriteByte(markerObjectEnd)

	buf.WriteByte(markerObjectEnd)

	d := NewDecoder(&buf)

	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectStart {
		t.Fatalf("expected object start, got %v", tok.Kind)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokKey || tok.StrVal != "a" {
		t.Fatalf("expected key 'a', got %v", tok)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectStart {
		t.Fatalf("expected nested object start, got %v", tok.Kind)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokKey || tok.StrVal != "b" {
		t.Fatalf("expected key 'b', got %v", tok)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt8 || tok.I8Val != 1 {
		t.Fatalf("expected int8 1, got %v", tok)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectEnd {
		t.Fatalf("expected nested object end, got %v", tok.Kind)
	}

	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokObjectEnd {
		t.Fatalf("expected outer object end, got %v", tok.Kind)
	}
}

func TestDecoderSkipValue(t *testing.T) {
	// { "skip": [1, 2, 3], "keep": "yes" }
	var buf bytes.Buffer
	buf.WriteByte(markerObjectStart)

	// key "skip"
	buf.WriteByte(markerString)
	buf.WriteByte(4)
	buf.WriteString("skip")
	// value: array [1, 2, 3]
	buf.WriteByte(markerArrayStart)
	for _, v := range []byte{1, 2, 3} {
		buf.WriteByte(markerInt8)
		buf.WriteByte(v)
	}
	buf.WriteByte(markerArrayEnd)

	// key "keep"
	buf.WriteByte(markerString)
	buf.WriteByte(4)
	buf.WriteString("keep")
	// value: "yes"
	buf.WriteByte(markerString)
	buf.WriteByte(3)
	buf.WriteString("yes")

	buf.WriteByte(markerObjectEnd)

	d := NewDecoder(&buf)

	// ObjectStart
	d.Next()

	// Key "skip"
	d.Next()
	// Skip the array value
	err := d.SkipValue()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Key "keep"
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokKey || tok.StrVal != "keep" {
		t.Fatalf("expected key 'keep', got %v", tok)
	}

	// Value "yes"
	tok, err = d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokString || tok.StrVal != "yes" {
		t.Fatalf("expected 'yes', got %v", tok)
	}
}

func TestDecoderTypedInt8Array(t *testing.T) {
	// Optimized int8 array: [$][i][#][l][3][FF][00][01] = [-1, 0, 1]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)
	buf.WriteByte(markerInt8)
	buf.WriteByte(markerCountMarker)
	buf.WriteByte(markerInt32)
	countBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(countBuf, 3)
	buf.Write(countBuf)
	buf.WriteByte(0xFF) // -1 as int8
	buf.WriteByte(0x00) // 0
	buf.WriteByte(0x01) // 1

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt32Slice {
		t.Fatalf("expected int32 slice, got %v", tok.Kind)
	}
	if len(tok.I32Slice) != 3 {
		t.Fatalf("expected slice length 3, got %d", len(tok.I32Slice))
	}
	expected := []int32{-1, 0, 1}
	for i, v := range expected {
		if tok.I32Slice[i] != v {
			t.Errorf("expected [%d] = %d, got %d", i, v, tok.I32Slice[i])
		}
	}
}

func TestDecoderTypedInt64Array(t *testing.T) {
	// Optimized int64 array: [$][L][#][l][2][<int64 bytes>][<int64 bytes>]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)
	buf.WriteByte(markerInt64)
	buf.WriteByte(markerCountMarker)
	buf.WriteByte(markerInt32)
	countBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(countBuf, 2)
	buf.Write(countBuf)

	valBuf := make([]byte, 8)
	binary.BigEndian.PutUint64(valBuf, 100)
	buf.Write(valBuf)
	binary.BigEndian.PutUint64(valBuf, 200)
	buf.Write(valBuf)

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt64Slice {
		t.Fatalf("expected int64 slice, got %v", tok.Kind)
	}
	if len(tok.I64Slice) != 2 {
		t.Fatalf("expected slice length 2, got %d", len(tok.I64Slice))
	}
	if tok.I64Slice[0] != 100 || tok.I64Slice[1] != 200 {
		t.Errorf("expected [100, 200], got %v", tok.I64Slice)
	}
}

func TestDecoderCountUint8(t *testing.T) {
	// Optimized int32 array with uint8 count: [$][l][#][U][3][...]
	var buf bytes.Buffer
	buf.WriteByte(markerTypeMarker)
	buf.WriteByte(markerInt32)
	buf.WriteByte(markerCountMarker)
	buf.WriteByte(markerUint8) // count type is uint8
	buf.WriteByte(2)           // count = 2

	valBuf := make([]byte, 4)
	binary.BigEndian.PutUint32(valBuf, 42)
	buf.Write(valBuf)
	binary.BigEndian.PutUint32(valBuf, 99)
	buf.Write(valBuf)

	d := NewDecoder(&buf)
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokInt32Slice {
		t.Fatalf("expected int32 slice, got %v", tok.Kind)
	}
	if len(tok.I32Slice) != 2 || tok.I32Slice[0] != 42 || tok.I32Slice[1] != 99 {
		t.Errorf("expected [42, 99], got %v", tok.I32Slice)
	}
}

func TestDecoderEmptyString(t *testing.T) {
	input := []byte{markerString, 0}
	d := NewDecoder(bytes.NewReader(input))
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokString || tok.StrVal != "" {
		t.Errorf("expected empty string, got %v", tok)
	}
}

func TestDecoderHighPrecision(t *testing.T) {
	// High-precision: 'H' + 'S' + length + bytes
	input := []byte{markerHighPrecision, markerString, 5, '1', '.', '2', '3', '4'}
	d := NewDecoder(bytes.NewReader(input))
	tok, err := d.Next()
	if err != nil {
		t.Fatal(err)
	}
	if tok.Kind != TokString || tok.StrVal != "1.234" {
		t.Errorf("expected '1.234', got %v", tok)
	}
}
