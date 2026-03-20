package xgbin

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"reflect"
	"testing"
)

// Simple test struct for ReadStruct
type testStruct struct {
	A int32
	B float32
	C uint32
}

func TestReadStruct(t *testing.T) {
	// Create test data
	data := testStruct{A: 42, B: 3.14, C: 100}
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, data)

	// Read back
	var result testStruct
	reader := bufio.NewReader(&buf)
	err := ReadStruct(reader, &result)
	if err != nil {
		t.Fatalf("ReadStruct failed: %v", err)
	}

	if !reflect.DeepEqual(data, result) {
		t.Errorf("ReadStruct: expected %v, got %v", data, result)
	}
}

func TestReadStringValid(t *testing.T) {
	testString := "hello world"
	var buf bytes.Buffer
	// Write size as uint64
	binary.Write(&buf, binary.LittleEndian, uint64(len(testString)))
	// Write string bytes
	binary.Write(&buf, binary.LittleEndian, []byte(testString))

	reader := bufio.NewReader(&buf)
	result, err := ReadString(reader)
	if err != nil {
		t.Fatalf("ReadString failed: %v", err)
	}

	if result != testString {
		t.Errorf("ReadString: expected %q, got %q", testString, result)
	}
}

func TestReadStringEmpty(t *testing.T) {
	var buf bytes.Buffer
	// Write size 0
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	reader := bufio.NewReader(&buf)
	result, err := ReadString(reader)
	if err != nil {
		t.Fatalf("ReadString failed: %v", err)
	}

	if result != "" {
		t.Errorf("ReadString: expected empty string, got %q", result)
	}
}

func TestReadFloat32SliceValid(t *testing.T) {
	testData := []float32{1.1, 2.2, 3.3, 4.4}
	var buf bytes.Buffer
	// Write size as uint64
	binary.Write(&buf, binary.LittleEndian, uint64(len(testData)))
	// Write float data
	binary.Write(&buf, binary.LittleEndian, testData)

	reader := bufio.NewReader(&buf)
	result, err := ReadFloat32Slice(reader)
	if err != nil {
		t.Fatalf("ReadFloat32Slice failed: %v", err)
	}

	if !reflect.DeepEqual(testData, result) {
		t.Errorf("ReadFloat32Slice: expected %v, got %v", testData, result)
	}
}

func TestReadFloat32SliceEmpty(t *testing.T) {
	var buf bytes.Buffer
	// Write size 0
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	reader := bufio.NewReader(&buf)
	result, err := ReadFloat32Slice(reader)
	if err != nil {
		t.Fatalf("ReadFloat32Slice failed: %v", err)
	}

	if result != nil {
		t.Errorf("ReadFloat32Slice: expected nil slice, got %v", result)
	}
}

func TestReadInt32SliceValid(t *testing.T) {
	testData := []int32{10, 20, 30, 40}
	var buf bytes.Buffer
	// Write size as uint64
	binary.Write(&buf, binary.LittleEndian, uint64(len(testData)))
	// Write int data
	binary.Write(&buf, binary.LittleEndian, testData)

	reader := bufio.NewReader(&buf)
	result, err := ReadInt32Slice(reader)
	if err != nil {
		t.Fatalf("ReadInt32Slice failed: %v", err)
	}

	if !reflect.DeepEqual(testData, result) {
		t.Errorf("ReadInt32Slice: expected %v, got %v", testData, result)
	}
}

func TestReadInt32SliceEmpty(t *testing.T) {
	var buf bytes.Buffer
	// Write size 0
	binary.Write(&buf, binary.LittleEndian, uint64(0))

	reader := bufio.NewReader(&buf)
	result, err := ReadInt32Slice(reader)
	if err != nil {
		t.Fatalf("ReadInt32Slice failed: %v", err)
	}

	if result != nil {
		t.Errorf("ReadInt32Slice: expected nil slice, got %v", result)
	}
}

func TestReadModelHeader(t *testing.T) {
	// Create test ModelHeader binary data
	var buf bytes.Buffer

	// Write LearnerModelParam
	param := LearnerModelParam{
		BaseScore:          0.5,
		NumFeatures:        10,
		NumClass:           2,
		ContainExtraAttrs:  1,
		ContainEvalMetrics: 0,
	}
	binary.Write(&buf, binary.LittleEndian, param)

	// Write NameObj
	nameObj := "binary:logistic"
	binary.Write(&buf, binary.LittleEndian, uint64(len(nameObj)))
	binary.Write(&buf, binary.LittleEndian, []byte(nameObj))

	// Write NameGbm
	nameGbm := "gbtree"
	binary.Write(&buf, binary.LittleEndian, uint64(len(nameGbm)))
	binary.Write(&buf, binary.LittleEndian, []byte(nameGbm))

	reader := bufio.NewReader(&buf)
	result, err := ReadModelHeader(reader)
	if err != nil {
		t.Fatalf("ReadModelHeader failed: %v", err)
	}

	expected := &ModelHeader{
		Param:   param,
		NameObj: nameObj,
		NameGbm: nameGbm,
	}

	if !reflect.DeepEqual(expected, result) {
		t.Errorf("ReadModelHeader: expected %v, got %v", expected, result)
	}
}

func TestReadTreeModel(t *testing.T) {
	// Create test TreeModel binary data
	var buf bytes.Buffer

	// Write TreeParam
	param := TreeParam{
		NumRoots:       1,
		NumNodes:       2,
		NumDeleted:     0,
		MaxDepth:       1,
		NumFeature:     5,
		SizeLeafVector: 0,
	}
	binary.Write(&buf, binary.LittleEndian, param)

	// Write Nodes (2 nodes)
	node1 := Node{Parent: -1, CLeft: 1, CRight: 2, SIndex: 0, Info: 0.5}
	node2 := Node{Parent: 0, CLeft: -1, CRight: -1, SIndex: 0, Info: 1.0}
	binary.Write(&buf, binary.LittleEndian, node1)
	binary.Write(&buf, binary.LittleEndian, node2)

	// Write Stats (2 stats)
	stat1 := RTreeNodeStat{LossChg: 0.1, SumHess: 10.0, BaseWeight: 0.5, LeafChildCnt: 1}
	stat2 := RTreeNodeStat{LossChg: 0.0, SumHess: 5.0, BaseWeight: 1.0, LeafChildCnt: 0}
	binary.Write(&buf, binary.LittleEndian, stat1)
	binary.Write(&buf, binary.LittleEndian, stat2)

	reader := bufio.NewReader(&buf)
	result, err := ReadTreeModel(reader)
	if err != nil {
		t.Fatalf("ReadTreeModel failed: %v", err)
	}

	expected := &TreeModel{
		Param: param,
		Nodes: []Node{node1, node2},
		Stats: []RTreeNodeStat{stat1, stat2},
	}

	if !reflect.DeepEqual(expected, result) {
		t.Errorf("ReadTreeModel: expected %v, got %v", expected, result)
	}
}

func TestReadGBLinearModel(t *testing.T) {
	// Create test GBLinearModel binary data
	var buf bytes.Buffer

	// Write GBLinearModelParam
	param := GBLinearModelParam{
		NumFeature:     5,
		NumOutputGroup: 1,
	}
	binary.Write(&buf, binary.LittleEndian, param)

	// Write Weights
	weights := []float32{0.1, 0.2, 0.3, 0.4, 0.5}
	binary.Write(&buf, binary.LittleEndian, uint64(len(weights)))
	binary.Write(&buf, binary.LittleEndian, weights)

	reader := bufio.NewReader(&buf)
	result, err := ReadGBLinearModel(reader)
	if err != nil {
		t.Fatalf("ReadGBLinearModel failed: %v", err)
	}

	expected := &GBLinearModel{
		Param:   param,
		Weights: weights,
	}

	if !reflect.DeepEqual(expected, result) {
		t.Errorf("ReadGBLinearModel: expected %v, got %v", expected, result)
	}
}
