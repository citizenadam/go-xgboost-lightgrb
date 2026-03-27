package leaves

import (
	"bytes"
	"fmt"
	"github.com/citizenadam/go-xgboost-lightgrb/leaves/internal/ubjson"
	"testing"
)

func TestDebugFullDecode(t *testing.T) {
	data := createTestXGBoostUBJSON()
	dec := ubjson.NewDecoder(bytes.NewReader(data))

	fmt.Println("=== Full decode trace ===")
	for i := 0; i < 100; i++ {
		tok, err := dec.Next()
		if err != nil {
			fmt.Printf("token %d: error: %v\n", i, err)
			break
		}
		kindStr := ""
		switch tok.Kind {
		case ubjson.TokObjectStart:
			kindStr = "ObjectStart"
		case ubjson.TokObjectEnd:
			kindStr = "ObjectEnd"
		case ubjson.TokArrayStart:
			kindStr = "ArrayStart"
		case ubjson.TokArrayEnd:
			kindStr = "ArrayEnd"
		case ubjson.TokKey:
			kindStr = "Key"
		case ubjson.TokString:
			kindStr = "String"
		case ubjson.TokInt32:
			kindStr = "Int32"
		case ubjson.TokUint8:
			kindStr = "Uint8"
		case ubjson.TokInt32Slice:
			kindStr = "Int32Slice"
		case ubjson.TokUint8Slice:
			kindStr = "Uint8Slice"
		case ubjson.TokFloat32Slice:
			kindStr = "Float32Slice"
		default:
			kindStr = fmt.Sprintf("Kind%d", tok.Kind)
		}
		fmt.Printf("token %d: %s str=%q i32=%d\n", i, kindStr, tok.StrVal, tok.I32Val)
	}
}
