package pickle

import (
	"testing"
)

// Simple implementation of PythonClass for testing
type testClass struct {
	lastReduce Reduce
	lastBuild  Build
}

func (t *testClass) Reduce(reduce Reduce) error {
	t.lastReduce = reduce
	return nil
}

func (t *testClass) Build(build Build) error {
	t.lastBuild = build
	return nil
}

func TestToGlobal(t *testing.T) {
	// Test valid Global object
	globalObj := Global{Module: "test", Name: "TestClass"}
	global, err := toGlobal(globalObj, "test", "TestClass")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if global.Module != "test" || global.Name != "TestClass" {
		t.Errorf("Expected Global{Module:\"test\", Name:\"TestClass\"}, got %+v", global)
	}

	// Test mismatched module
	_, err = toGlobal(globalObj, "wrong", "TestClass")
	if err == nil {
		t.Error("Expected error for module mismatch")
	}

	// Test mismatched name
	_, err = toGlobal(globalObj, "test", "WrongName")
	if err == nil {
		t.Error("Expected error for name mismatch")
	}

	// Test invalid type
	_, err = toGlobal("not a global", "", "")
	if err == nil {
		t.Error("Expected error for non-Global type")
	}
}

func TestToTuple(t *testing.T) {
	// Test valid Tuple
	tupleObj := Tuple{1, "two", 3.0}
	tuple, err := toTuple(tupleObj, -1) // No length check
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(tuple) != 3 || tuple[0] != 1 || tuple[1] != "two" || tuple[2] != 3.0 {
		t.Errorf("Expected Tuple{1, \"two\", 3.0}, got %+v", tuple)
	}

	// Test valid Tuple with length check
	tuple, err = toTuple(tupleObj, 3)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(tuple) != 3 {
		t.Errorf("Expected tuple length 3, got %d", len(tuple))
	}

	// Test invalid length
	_, err = toTuple(tupleObj, 2)
	if err == nil {
		t.Error("Expected error for length mismatch")
	}

	// Test invalid type
	_, err = toTuple("not a tuple", -1)
	if err == nil {
		t.Error("Expected error for non-Tuple type")
	}
}

func TestToList(t *testing.T) {
	// Test valid List
	listObj := List{1, "two", 3.0}
	list, err := toList(listObj, -1) // No length check
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(list) != 3 || list[0] != 1 || list[1] != "two" || list[2] != 3.0 {
		t.Errorf("Expected List{1, \"two\", 3.0}, got %+v", list)
	}

	// Test valid List with length check
	list, err = toList(listObj, 3)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(list) != 3 {
		t.Errorf("Expected list length 3, got %d", len(list))
	}

	// Test invalid length
	_, err = toList(listObj, 2)
	if err == nil {
		t.Error("Expected error for length mismatch")
	}

	// Test invalid type
	_, err = toList("not a list", -1)
	if err == nil {
		t.Error("Expected error for non-List type")
	}
}

func TestToUnicode(t *testing.T) {
	// Test valid Unicode
	unicodeObj := Unicode{65, 66, 67}         // "ABC"
	unicode, err := toUnicode(unicodeObj, -1) // No length check
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(unicode) != 3 || unicode[0] != 65 || unicode[1] != 66 || unicode[2] != 67 {
		t.Errorf("Expected Unicode{65, 66, 67}, got %+v", unicode)
	}

	// Test valid Unicode with length check
	unicode, err = toUnicode(unicodeObj, 3)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if len(unicode) != 3 {
		t.Errorf("Expected unicode length 3, got %d", len(unicode))
	}

	// Test invalid length
	_, err = toUnicode(unicodeObj, 2)
	if err == nil {
		t.Error("Expected error for length mismatch")
	}

	// Test invalid type
	_, err = toUnicode("not unicode", -1)
	if err == nil {
		t.Error("Expected error for non-Unicode type")
	}
}

func TestToDict(t *testing.T) {
	// Test valid Dict
	dictObj := Dict{"key1": 1, "key2": "value"}
	dict, err := toDict(dictObj)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if dict["key1"] != 1 || dict["key2"] != "value" {
		t.Errorf("Expected Dict{\"key1\": 1, \"key2\": \"value\"}, got %+v", dict)
	}

	// Test invalid type
	_, err = toDict("not a dict")
	if err == nil {
		t.Error("Expected error for non-Dict type")
	}
}

func TestDictValueAndToInt(t *testing.T) {
	dict := Dict{"count": 42, "name": "test"}

	// Test value() with existing key
	value, err := dict.value("count")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if value != 42 {
		t.Errorf("Expected value 42, got %v", value)
	}

	// Test value() with non-existing key
	_, err = dict.value("nonexistent")
	if err == nil {
		t.Error("Expected error for nonexistent key")
	}

	// Test toInt() with existing int
	intValue, err := dict.toInt("count")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if intValue != 42 {
		t.Errorf("Expected int value 42, got %d", intValue)
	}

	// Test toInt() with non-int value
	_, err = dict.toInt("name")
	if err == nil {
		t.Error("Expected error for non-int value")
	}

	// Test toInt() with non-existing key
	_, err = dict.toInt("nonexistent")
	if err == nil {
		t.Error("Expected error for nonexistent key")
	}
}

func TestParseClass(t *testing.T) {
	// Create a test entity
	entity := &testClass{}

	// Test ParseClass with direct Reduce object
	reduceObj := Reduce{
		Callable: func() {},
		Args:     Tuple{1, 2, 3},
	}
	err := ParseClass(entity, reduceObj)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if entity.lastReduce.Callable == nil {
		t.Error("Expected Reduce.Callable to be set")
	}
	if len(entity.lastReduce.Args) != 3 {
		t.Errorf("Expected Reduce.Args length 3, got %d", len(entity.lastReduce.Args))
	}

	// Test ParseClass with Build object containing Reduce
	buildObj := Build{
		Object: reduceObj,
		Args:   Dict{"prop": "value"},
	}
	err = ParseClass(entity, buildObj)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	// Check that the Object field in Build contains our reduceObj
	if _, ok := entity.lastBuild.Object.(Reduce); !ok {
		t.Error("Expected Build.Object to be a Reduce")
	} else if entity.lastBuild.Object.(Reduce).Args == nil {
		t.Error("Expected Build.Object.Args to be set")
	} else if len(entity.lastBuild.Object.(Reduce).Args) != 3 {
		t.Errorf("Expected Build.Object.Args length 3, got %d", len(entity.lastBuild.Object.(Reduce).Args))
	}
	if entity.lastBuild.Args == nil {
		t.Error("Expected Build.Args to be set")
	}

	// Test ParseClass with invalid type
	err = ParseClass(entity, "invalid")
	if err == nil {
		t.Error("Expected error for invalid type")
	}

	// Test ParseClass with Build containing invalid Reduce
	invalidBuild := Build{
		Object: "not a reduce",
		Args:   nil,
	}
	err = ParseClass(entity, invalidBuild)
	if err == nil {
		t.Error("Expected error for invalid Reduce in Build")
	}
}
