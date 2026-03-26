package util

import (
	"bufio"
	"os"
	"path/filepath"
	"testing"
)

func TestFirstNonZeroBit(t *testing.T) {
	const length = 10
	const size = 32
	bitset := make([]uint32, length)
	_, err := FirstNonZeroBit(bitset)
	if err == nil {
		t.Error("all zeros bitset should fail")
	}

	check := func(trueAnswer uint32) {
		pos, err := FirstNonZeroBit(bitset)
		if err != nil {
			t.Error(err.Error())
		}
		if pos != trueAnswer {
			t.Errorf("%d fail", trueAnswer)
		}
	}

	bitset[9] |= 1 << 31
	check(9*size + 31)

	bitset[3] |= 1 << 3
	check(3*size + 3)

	bitset[0] |= 1 << 7
	check(7)

	bitset[0] |= 1 << 0
	check(0)
}

func TestNumberOfSetBits(t *testing.T) {
	const length = 10
	bitset := make([]uint32, length)

	check := func(trueAnswer uint32) {
		if NumberOfSetBits(bitset) != trueAnswer {
			t.Errorf("%d fail", trueAnswer)
		}
	}

	bitset[9] |= 1 << 31
	check(1)

	bitset[3] |= 1 << 3
	check(2)

	bitset[0] |= 1 << 7
	check(3)

	bitset[0] |= 1 << 0
	check(4)
}

func TestReadParams(t *testing.T) {
	path := filepath.Join("..", "testdata", "model_simple.txt")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	// Читаем заголовок файла
	params, err := ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}

	trueMap := map[string]string{
		"version":                "v2",
		"num_class":              "1",
		"num_tree_per_iteration": "1",
		"label_index":            "0",
		"max_feature_idx":        "1",
		"objective":              "binary sigmoid:1",
		"feature_names":          "X1 X2",
		"feature_infos":          "[0:999] 1:0:3:100:-1",
		"tree_sizes":             "358 365",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}

	// Читаем первое дерево
	params, err = ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueMap = map[string]string{
		"Tree":           "0",
		"num_leaves":     "3",
		"num_cat":        "1",
		"split_feature":  "1 0",
		"split_gain":     "138.409 13.4409",
		"threshold":      "0 340.50000000000006",
		"decision_type":  "9 2",
		"left_child":     "-1 -2",
		"right_child":    "1 -3",
		"leaf_value":     "0.56697267424823339 0.3584987837673016 0.41213915936587919",
		"leaf_count":     "200 341 459",
		"internal_value": "0 -0.392018",
		"internal_count": "1000 800",
		"cat_boundaries": "0 4",
		"cat_threshold":  "0 0 0 16",
		"shrinkage":      "1",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}

	// Читаем второe дерево
	params, err = ReadParamsUntilBlank(bufReader)
	if err != nil {
		t.Fatal(err)
	}
	trueMap = map[string]string{
		"Tree":           "1",
		"num_leaves":     "3",
		"num_cat":        "1",
		"split_feature":  "1 0",
		"split_gain":     "118.043 10.5922",
		"threshold":      "0 340.50000000000006",
		"decision_type":  "9 2",
		"left_child":     "-1 -2",
		"right_child":    "1 -3",
		"leaf_value":     "0.12883103567558912 -0.063872842243335157 -0.016484332942214807",
		"leaf_count":     "200 341 459",
		"internal_value": "0 -0.349854",
		"internal_count": "1000 800",
		"cat_boundaries": "0 4",
		"cat_threshold":  "0 0 0 16",
		"shrinkage":      "0.1",
	}
	for key, val := range trueMap {
		if params[key] != val {
			t.Errorf("params[%s] != %s", key, val)
		}
	}
}

func TestConstructBitset(t *testing.T) {
	bitset := ConstructBitset([]int{0})

	check := func(trueAnswer []uint32) {
		if len(trueAnswer) != len(bitset) {
			t.Errorf("wrong length. expected %d, got %d", len(trueAnswer), len(bitset))
		}
		for i, v := range trueAnswer {
			if v != bitset[i] {
				t.Errorf("wrong %d-th value. expected %d, got %d", i, v, bitset[i])
			}
		}
	}

	check([]uint32{1})

	bitset = ConstructBitset([]int{33, 65, 105})
	check([]uint32{0, 2, 2, 512})

	bitset = ConstructBitset([]int{})
	check([]uint32{})
}

func TestSigmoidFloat64SliceInplace(t *testing.T) {
	vec := [...]float64{-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0}
	vecTrue := [...]float64{0.26894142, 0.37754067, 0.4378235, 0.5, 0.5621765, 0.62245933, 0.73105858}
	SigmoidFloat64SliceInplace(vec[:])
	err := AlmostEqualFloat64Slices(vec[:], vecTrue[:], 1e-8)
	if err != nil {
		t.Error(err.Error())
	}
}

func TestSoftmaxFloat64Slice(t *testing.T) {
	compare := func(vec []float64, vecTrue []float64) {
		res := make([]float64, len(vec))
		SoftmaxFloat64Slice(vec, res, 0)
		err := AlmostEqualFloat64Slices(res, vecTrue, 1e-8)
		if err != nil {
			t.Error(err.Error())
		}
	}

	compare(
		[]float64{0.25, 0.75},
		[]float64{0.37754067, 0.62245933},
	)

	compare(
		[]float64{0.0, 0.0},
		[]float64{0.5, 0.5},
	)

	compare(
		[]float64{1.0, 2.0, 3.0},
		[]float64{0.09003057, 0.24472847, 0.66524096},
	)

	compare(
		[]float64{10.0, 20.0, 30.0},
		[]float64{2.06106005e-09, 4.53978686e-05, 9.99954600e-01},
	)

	compare(
		[]float64{},
		[]float64{},
	)
}

func TestFindInBitsetUint32(t *testing.T) {
	// bit 0 is set
	if !FindInBitsetUint32(1, 0) {
		t.Error("bit 0 should be set")
	}
	// bit 5 is set
	if !FindInBitsetUint32(32, 5) {
		t.Error("bit 5 should be set")
	}
	// bit 0 is not set in value 2
	if FindInBitsetUint32(2, 0) {
		t.Error("bit 0 should not be set")
	}
	// pos >= 32 always returns false
	if FindInBitsetUint32(0xFFFFFFFF, 32) {
		t.Error("pos >= 32 should return false")
	}
	if FindInBitsetUint32(0xFFFFFFFF, 100) {
		t.Error("pos >= 32 should return false")
	}
}

func TestMinInt(t *testing.T) {
	if MinInt(1, 2) != 1 {
		t.Error("MinInt(1,2) should be 1")
	}
	if MinInt(2, 1) != 1 {
		t.Error("MinInt(2,1) should be 1")
	}
	if MinInt(5, 5) != 5 {
		t.Error("MinInt(5,5) should be 5")
	}
	if MinInt(-1, 0) != -1 {
		t.Error("MinInt(-1,0) should be -1")
	}
}

func TestStringParamsToInt(t *testing.T) {
	p := stringParams{"num": "42", "bad": "not_a_number", "neg": "-7"}

	v, err := p.ToInt("num")
	if err != nil {
		t.Fatal(err)
	}
	if v != 42 {
		t.Errorf("expected 42, got %d", v)
	}

	v, err = p.ToInt("neg")
	if err != nil {
		t.Fatal(err)
	}
	if v != -7 {
		t.Errorf("expected -7, got %d", v)
	}

	_, err = p.ToInt("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}

	_, err = p.ToInt("bad")
	if err == nil {
		t.Error("should fail for non-integer value")
	}
}

func TestStringParamsToString(t *testing.T) {
	p := stringParams{"name": "hello"}

	v, err := p.ToString("name")
	if err != nil {
		t.Fatal(err)
	}
	if v != "hello" {
		t.Errorf("expected 'hello', got '%s'", v)
	}

	_, err = p.ToString("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}
}

func TestStringParamsCompare(t *testing.T) {
	p := stringParams{"objective": "binary"}

	err := p.Compare("objective", "binary")
	if err != nil {
		t.Error("should match")
	}

	err = p.Compare("objective", "regression")
	if err == nil {
		t.Error("should fail for different value")
	}

	err = p.Compare("missing", "val")
	if err == nil {
		t.Error("should fail for missing key")
	}
}

func TestStringParamsToStrSlice(t *testing.T) {
	p := stringParams{"features": "X1 X2 X3"}

	v, err := p.ToStrSlice("features")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || v[0] != "X1" || v[1] != "X2" || v[2] != "X3" {
		t.Errorf("expected [X1 X2 X3], got %v", v)
	}

	_, err = p.ToStrSlice("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}
}

func TestStringParamsToFloat64Slice(t *testing.T) {
	p := stringParams{"values": "1.0 2.5 -0.3"}

	v, err := p.ToFloat64Slice("values")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || AlmostEqualFloat64(v[0], 1.0, 1e-10) == false ||
		AlmostEqualFloat64(v[1], 2.5, 1e-10) == false ||
		AlmostEqualFloat64(v[2], -0.3, 1e-10) == false {
		t.Errorf("expected [1.0 2.5 -0.3], got %v", v)
	}

	_, err = p.ToFloat64Slice("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}

	p2 := stringParams{"bad": "1.0 not_a_float"}
	_, err = p2.ToFloat64Slice("bad")
	if err == nil {
		t.Error("should fail for non-float value")
	}
}

func TestStringParamsToUint32Slice(t *testing.T) {
	p := stringParams{"indices": "0 5 100"}

	v, err := p.ToUint32Slice("indices")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || v[0] != 0 || v[1] != 5 || v[2] != 100 {
		t.Errorf("expected [0 5 100], got %v", v)
	}

	_, err = p.ToUint32Slice("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}

	p2 := stringParams{"bad": "0 not_a_number"}
	_, err = p2.ToUint32Slice("bad")
	if err == nil {
		t.Error("should fail for non-uint value")
	}
}

func TestStringParamsToInt32Slice(t *testing.T) {
	p := stringParams{"values": "0 -5 100"}

	v, err := p.ToInt32Slice("values")
	if err != nil {
		t.Fatal(err)
	}
	if len(v) != 3 || v[0] != 0 || v[1] != -5 || v[2] != 100 {
		t.Errorf("expected [0 -5 100], got %v", v)
	}

	_, err = p.ToInt32Slice("missing")
	if err == nil {
		t.Error("should fail for missing key")
	}

	p2 := stringParams{"bad": "0 not_a_number"}
	_, err = p2.ToInt32Slice("bad")
	if err == nil {
		t.Error("should fail for non-int value")
	}
}

func TestStringParamsContains(t *testing.T) {
	p := stringParams{"key": "value"}

	if !p.Contains("key") {
		t.Error("should contain 'key'")
	}
	if p.Contains("missing") {
		t.Error("should not contain 'missing'")
	}
}

func TestAlmostEqualFloat64(t *testing.T) {
	if !AlmostEqualFloat64(1.0, 1.0, 1e-10) {
		t.Error("equal values should be true")
	}
	if !AlmostEqualFloat64(1.0, 1.0000000001, 1e-9) {
		t.Error("close values should be true")
	}
	if AlmostEqualFloat64(1.0, 2.0, 1e-10) {
		t.Error("different values should be false")
	}
}

func TestNumMismatchedFloat64Slices(t *testing.T) {
	a := []float64{1.0, 2.0, 3.0, 4.0}
	b := []float64{1.0, 2.1, 3.0, 4.5}

	count, err := NumMismatchedFloat64Slices(a, b, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Errorf("expected 2 mismatches, got %d", count)
	}

	count, err = NumMismatchedFloat64Slices(a, b, 1.0)
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Errorf("expected 0 mismatches with large threshold, got %d", count)
	}

	_, err = NumMismatchedFloat64Slices([]float64{1.0}, []float64{1.0, 2.0}, 0.01)
	if err == nil {
		t.Error("should fail for different length slices")
	}
}

func TestFloat64FromBytes(t *testing.T) {
	// 0.5 in IEEE 754 little-endian: 0x000000000000E03F
	leBytes := []byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x3F}
	v := Float64FromBytes(leBytes, true)
	if v != 0.5 {
		t.Errorf("little-endian: expected 0.5, got %f", v)
	}

	// 0.5 in IEEE 754 big-endian: 0x3FE0000000000000
	beBytes := []byte{0x3F, 0xE0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}
	v = Float64FromBytes(beBytes, false)
	if v != 0.5 {
		t.Errorf("big-endian: expected 0.5, got %f", v)
	}
}
