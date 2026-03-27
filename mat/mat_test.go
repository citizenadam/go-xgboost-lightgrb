package mat

import (
	"bufio"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/citizenadam/go-xgboost-lightgrb/leaves/util"
)

func TestDenseMatZero(t *testing.T) {
	mat := DenseMatZero(3, 4)
	if mat.Rows != 3 {
		t.Errorf("Rows: expected 3, got %d", mat.Rows)
	}
	if mat.Cols != 4 {
		t.Errorf("Cols: expected 4, got %d", mat.Cols)
	}
	if len(mat.Values) != 12 {
		t.Errorf("Values length: expected 12, got %d", len(mat.Values))
	}
	for i, v := range mat.Values {
		if v != 0.0 {
			t.Errorf("Values[%d]: expected 0.0, got %f", i, v)
		}
	}

	// Zero-size matrix
	mat2 := DenseMatZero(0, 0)
	if mat2.Rows != 0 || mat2.Cols != 0 || len(mat2.Values) != 0 {
		t.Error("zero-size matrix should have all zero fields")
	}
}

func TestDenseMatFromArray(t *testing.T) {
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	mat, err := DenseMatFromArray(values, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Rows != 2 {
		t.Errorf("Rows: expected 2, got %d", mat.Rows)
	}
	if mat.Cols != 3 {
		t.Errorf("Cols: expected 3, got %d", mat.Cols)
	}
	if err := util.AlmostEqualFloat64Slices(mat.Values, values, 1e-10); err != nil {
		t.Errorf("Values mismatch: %s", err)
	}

	// Verify values are a copy (not aliased)
	values[0] = 999.0
	if mat.Values[0] == 999.0 {
		t.Error("DenseMatFromArray should copy values, not alias them")
	}

	// Wrong dimensions
	_, err = DenseMatFromArray([]float64{1.0, 2.0}, 2, 3)
	if err == nil {
		t.Error("should fail for wrong dimensions")
	}
}

func TestCSRMatFromArray(t *testing.T) {
	values := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
	mat, err := CSRMatFromArray(values, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	// Check values
	if err := util.AlmostEqualFloat64Slices(mat.Values, values, 1e-10); err != nil {
		t.Errorf("Values mismatch: %s", err)
	}

	// Check row headers: [0, 3, 6] for 2 rows x 3 cols
	trueRowHeaders := []int{0, 3, 6}
	if !reflect.DeepEqual(mat.RowHeaders, trueRowHeaders) {
		t.Errorf("RowHeaders: expected %v, got %v", trueRowHeaders, mat.RowHeaders)
	}

	// Check col indexes: [0, 1, 2, 0, 1, 2] for dense 2x3
	trueColIndexes := []int{0, 1, 2, 0, 1, 2}
	if !reflect.DeepEqual(mat.ColIndexes, trueColIndexes) {
		t.Errorf("ColIndexes: expected %v, got %v", trueColIndexes, mat.ColIndexes)
	}

	// Rows() method
	if mat.Rows() != 2 {
		t.Errorf("Rows(): expected 2, got %d", mat.Rows())
	}

	// Wrong dimensions
	_, err = CSRMatFromArray([]float64{1.0, 2.0}, 2, 3)
	if err == nil {
		t.Error("should fail for wrong dimensions")
	}

	// Empty matrix
	empty := CSRMat{}
	if empty.Rows() != 0 {
		t.Errorf("empty CSRMat Rows(): expected 0, got %d", empty.Rows())
	}
}

func TestDenseMatFromLibsvm(t *testing.T) {
	path := filepath.Join("..", "testdata", "densemat.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	_, err = DenseMatFromLibsvm(bufReader, 0, false)
	if err == nil {
		t.Fatal("should fail because of first column")
	}

	// check reading correctness
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err := DenseMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Cols != 3 {
		t.Errorf("mat.Cols should be 3 (got %d)", mat.Cols)
	}
	if mat.Rows != 2 {
		t.Errorf("mat.Rows should be 2 (got %d)", mat.Rows)
	}
	trueValues := []float64{19.0, 45.3, 1e-6, 14.0, 0.0, 0.0}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	// check reading correctness with limit 1
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err = DenseMatFromLibsvm(bufReader, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if mat.Cols != 3 {
		t.Errorf("mat.Cols should be 3 (got %d)", mat.Cols)
	}
	if mat.Rows != 1 {
		t.Errorf("mat.Rows should be 1 (got %d)", mat.Rows)
	}
	trueValues = []float64{19.0, 45.3, 1e-6}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}
}

func TestCSRMatFromLibsvm(t *testing.T) {
	path := filepath.Join("..", "testdata", "csrmat.libsvm")
	reader, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	bufReader := bufio.NewReader(reader)

	_, err = CSRMatFromLibsvm(bufReader, 0, false)
	if err == nil {
		t.Fatal("should fail because of first column")
	}

	// check reading correctness
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err := CSRMatFromLibsvm(bufReader, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	trueValues := []float64{19.0, 45.3, 1e-6, 14.0, 0.0}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	trueRowHeaders := []int{0, 3, 5}
	if !reflect.DeepEqual(mat.RowHeaders, trueRowHeaders) {
		t.Error("mat.RowHeaders are incorrect")
	}

	trueColIndexes := []int{0, 10, 12, 4, 5}
	if !reflect.DeepEqual(mat.ColIndexes, trueColIndexes) {
		t.Error("mat.ColIndexes are incorrect")
	}

	// check reading correctness with limit 1
	reader.Seek(0, 0)
	bufReader = bufio.NewReader(reader)
	mat, err = CSRMatFromLibsvm(bufReader, 1, true)
	if err != nil {
		t.Fatal(err)
	}

	trueValues = []float64{19.0, 45.3, 1e-6}
	if err := util.AlmostEqualFloat64Slices(mat.Values, trueValues, 1e-10); err != nil {
		t.Errorf("mat.Values incorrect: %s", err.Error())
	}

	trueRowHeaders = []int{0, 3}
	if !reflect.DeepEqual(mat.RowHeaders, trueRowHeaders) {
		t.Error("mat.RowHeaders are incorrect")
	}

	trueColIndexes = []int{0, 10, 12}
	if !reflect.DeepEqual(mat.ColIndexes, trueColIndexes) {
		t.Error("mat.ColIndexes are incorrect")
	}
}
