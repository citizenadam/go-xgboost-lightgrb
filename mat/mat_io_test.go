package mat

import (
	"bufio"
	"bytes"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestDenseMatFromCsv_ValidData(t *testing.T) {
	csvData := "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
	reader := bufio.NewReader(strings.NewReader(csvData))

	mat, err := DenseMatFromCsv(reader, 0, false, ",", 0.0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows != 3 {
		t.Errorf("Expected 3 rows, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", mat.Cols)
	}

	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	if !reflect.DeepEqual(mat.Values, expected) {
		t.Errorf("Expected values %v, got %v", expected, mat.Values)
	}
}

func TestDenseMatFromCsv_EmptyValues(t *testing.T) {
	csvData := "1.0,,3.0\n4.0,5.0,\n,8.0,9.0"
	reader := bufio.NewReader(strings.NewReader(csvData))

	mat, err := DenseMatFromCsv(reader, 0, false, ",", -1.0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows != 3 {
		t.Errorf("Expected 3 rows, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", mat.Cols)
	}

	expected := []float64{1.0, -1.0, 3.0, 4.0, 5.0, -1.0, -1.0, 8.0, 9.0}
	if !reflect.DeepEqual(mat.Values, expected) {
		t.Errorf("Expected values %v, got %v", expected, mat.Values)
	}
}

func TestCSRMatFromLibsvm_ValidData(t *testing.T) {
	libsvmData := "0:1.0 1:2.0 2:3.0\n0:4.0 1:5.0 2:6.0\n0:7.0 1:8.0 2:9.0"
	reader := bufio.NewReader(strings.NewReader(libsvmData))

	mat, err := CSRMatFromLibsvm(reader, 0, false)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows() != 3 {
		t.Errorf("Expected 3 rows, got %d", mat.Rows())
	}
	// For CSRMat, we can't directly get Cols, but we can verify from data
	// All records have column indices 0,1,2 so we expect 3 columns

	expectedValues := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	if !reflect.DeepEqual(mat.Values, expectedValues) {
		t.Errorf("Expected values %v, got %v", expectedValues, mat.Values)
	}

	expectedColIndexes := []int{0, 1, 2, 0, 1, 2, 0, 1, 2}
	if !reflect.DeepEqual(mat.ColIndexes, expectedColIndexes) {
		t.Errorf("Expected col indexes %v, got %v", expectedColIndexes, mat.ColIndexes)
	}

	expectedRowHeaders := []int{0, 3, 6, 9}
	if !reflect.DeepEqual(mat.RowHeaders, expectedRowHeaders) {
		t.Errorf("Expected row headers %v, got %v", expectedRowHeaders, mat.RowHeaders)
	}
}

func TestRecordsToDenseMat_ValidRecords(t *testing.T) {
	mat := &DenseMat{}
	records := []libsvmRecord{
		{0, 1.0},
		{1, 2.0},
		{2, 3.0},
	}

	err := recordsToDenseMat(mat, records)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows != 1 {
		t.Errorf("Expected 1 row, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", mat.Cols)
	}

	expected := []float64{1.0, 2.0, 3.0}
	if !reflect.DeepEqual(mat.Values, expected) {
		t.Errorf("Expected values %v, got %v", expected, mat.Values)
	}
}

func TestRecordsToDenseMat_WrongColumnNumber(t *testing.T) {
	mat := &DenseMat{}
	records := []libsvmRecord{
		{0, 1.0},
		{2, 3.0}, // Missing column 1
	}

	err := recordsToDenseMat(mat, records)
	if err == nil {
		t.Fatal("Expected error for wrong column number")
	}

	if !strings.Contains(err.Error(), "wrong column number for dense matrix") {
		t.Errorf("Expected error about wrong column number, got: %v", err)
	}
}

func TestRecordsToCSRMat_ValidRecords(t *testing.T) {
	mat := &CSRMat{}
	// Initialize RowHeaders with 0 as done in CSRMatFromLibsvm
	mat.RowHeaders = append(mat.RowHeaders, 0)
	records := []libsvmRecord{
		{0, 1.0},
		{1, 2.0},
		{2, 3.0},
	}

	err := recordsToCSRMat(mat, records)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if len(mat.Values) != 3 {
		t.Errorf("Expected 3 values, got %d", len(mat.Values))
	}
	if len(mat.ColIndexes) != 3 {
		t.Errorf("Expected 3 column indexes, got %d", len(mat.ColIndexes))
	}
	// RowHeaders should have initial 0 plus one entry per row
	if len(mat.RowHeaders) != 2 {
		t.Errorf("Expected 2 row headers, got %d", len(mat.RowHeaders))
	}

	expectedValues := []float64{1.0, 2.0, 3.0}
	if !reflect.DeepEqual(mat.Values, expectedValues) {
		t.Errorf("Expected values %v, got %v", expectedValues, mat.Values)
	}

	expectedColIndexes := []int{0, 1, 2}
	if !reflect.DeepEqual(mat.ColIndexes, expectedColIndexes) {
		t.Errorf("Expected col indexes %v, got %v", expectedColIndexes, mat.ColIndexes)
	}

	// RowHeaders: [0, 3] where 0 is start, 3 is end of first (and only) row
	expectedRowHeaders := []int{0, 3}
	if !reflect.DeepEqual(mat.RowHeaders, expectedRowHeaders) {
		t.Errorf("Expected row headers %v, got %v", expectedRowHeaders, mat.RowHeaders)
	}
}

func TestDenseMat_WriteStr_ValidMatrix(t *testing.T) {
	mat := &DenseMat{
		Values: []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		Rows:   2,
		Cols:   3,
	}

	var buf bytes.Buffer
	err := mat.WriteStr(&buf, ",")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// WriteStr uses %.19g format which outputs minimal representation
	expected := "1,2,3\n4,5,6\n"
	if buf.String() != expected {
		t.Errorf("Expected output:\n%s\nGot:\n%s", expected, buf.String())
	}
}

func TestDenseMat_WriteStr_EmptyMatrix(t *testing.T) {
	mat := &DenseMat{
		Values: []float64{},
		Rows:   0,
		Cols:   0,
	}

	var buf bytes.Buffer
	err := mat.WriteStr(&buf, ",")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if buf.String() != "\n" {
		t.Errorf("Expected newline, got: %q", buf.String())
	}
}

func TestDenseMat_ToCsvFile_Roundtrip(t *testing.T) {
	original := &DenseMat{
		Values: []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
		Rows:   2,
		Cols:   3,
	}

	// Write to a temporary file
	tempFile := "test_matrix.csv"
	err := original.ToCsvFile(tempFile, ",")
	if err != nil {
		t.Fatalf("Unexpected error writing file: %v", err)
	}
	defer os.Remove(tempFile)

	// Read back from file
	reader, err := os.Open(tempFile)
	if err != nil {
		t.Fatalf("Unexpected error opening file: %v", err)
	}
	defer reader.Close()

	mat, err := DenseMatFromCsv(bufio.NewReader(reader), 0, false, ",", 0.0)
	if err != nil {
		t.Fatalf("Unexpected error reading file: %v", err)
	}

	if !reflect.DeepEqual(original.Values, mat.Values) {
		t.Errorf("Values don't match. Expected: %v, Got: %v", original.Values, mat.Values)
	}
	if original.Rows != mat.Rows {
		t.Errorf("Rows don't match. Expected: %d, Got: %d", original.Rows, mat.Rows)
	}
	if original.Cols != mat.Cols {
		t.Errorf("Cols don't match. Expected: %d, Got: %d", original.Cols, mat.Cols)
	}
}

func TestDenseMatFromLibsvm_ValidData(t *testing.T) {
	libsvmData := "0 0:1.0 1:2.0 2:3.0\n1 0:4.0 1:5.0 2:6.0\n0 0:7.0 1:8.0 2:9.0"
	reader := bufio.NewReader(strings.NewReader(libsvmData))

	mat, err := DenseMatFromLibsvm(reader, 0, true)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows != 3 {
		t.Errorf("Expected 3 rows, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", mat.Cols)
	}

	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}
	if !reflect.DeepEqual(mat.Values, expected) {
		t.Errorf("Expected values %v, got %v", expected, mat.Values)
	}
}

func TestReadFromCsv_LimitParameter(t *testing.T) {
	csvData := "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0"
	reader := bufio.NewReader(strings.NewReader(csvData))

	mat, err := DenseMatFromCsv(reader, 2, false, ",", 0.0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if mat.Rows != 2 {
		t.Errorf("Expected 2 rows due to limit, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Expected 3 columns, got %d", mat.Cols)
	}

	expected := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	if !reflect.DeepEqual(mat.Values, expected) {
		t.Errorf("Expected values %v, got %v", expected, mat.Values)
	}
}
