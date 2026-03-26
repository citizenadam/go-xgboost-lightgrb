package transformation

import (
	"math"
	"testing"
)

func TestTransformRaw(t *testing.T) {
	tr := &TransformRaw{NumOutputGroups: 1}
	if tr.NOutputGroups() != 1 {
		t.Errorf("NOutputGroups: expected 1, got %d", tr.NOutputGroups())
	}
	if tr.Type() != Raw {
		t.Errorf("Type: expected Raw, got %d", tr.Type())
	}
	if tr.Name() != "raw" {
		t.Errorf("Name: expected 'raw', got '%s'", tr.Name())
	}

	// Test Transform: copy raw to output
	raw := []float64{1.5, 2.5, 3.5}
	output := make([]float64, 6)
	err := tr.Transform(raw, output, 0)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range raw {
		if output[i] != v {
			t.Errorf("output[%d]: expected %f, got %f", i, v, output[i])
		}
	}

	// Test with startIndex
	output2 := make([]float64, 6)
	err = tr.Transform(raw, output2, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range raw {
		if output2[3+i] != v {
			t.Errorf("output2[%d]: expected %f, got %f", 3+i, v, output2[3+i])
		}
	}

	// Empty slice
	err = tr.Transform([]float64{}, output, 0)
	if err != nil {
		t.Fatal(err)
	}
}

func TestTransformLogistic(t *testing.T) {
	tr := &TransformLogistic{}
	if tr.NOutputGroups() != 1 {
		t.Errorf("NOutputGroups: expected 1, got %d", tr.NOutputGroups())
	}
	if tr.Type() != Logistic {
		t.Errorf("Type: expected Logistic, got %d", tr.Type())
	}
	if tr.Name() != "logistic" {
		t.Errorf("Name: expected 'logistic', got '%s'", tr.Name())
	}

	// Test Transform: sigmoid of single value
	tests := []struct {
		input float64
		want  float64
	}{
		{0.0, 0.5},
		{1.0, 1.0 / (1.0 + math.Exp(-1.0))},
		{-1.0, 1.0 / (1.0 + math.Exp(1.0))},
	}
	for _, tc := range tests {
		output := make([]float64, 1)
		err := tr.Transform([]float64{tc.input}, output, 0)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(output[0]-tc.want) > 1e-10 {
			t.Errorf("sigmoid(%f): expected %f, got %f", tc.input, tc.want, output[0])
		}
	}

	// Error: wrong number of predictions
	output := make([]float64, 2)
	err := tr.Transform([]float64{1.0, 2.0}, output, 0)
	if err == nil {
		t.Error("expected error for len(rawPredictions) != 1")
	}
}

func TestTransformExponential(t *testing.T) {
	tr := &TransformExponential{}
	if tr.NOutputGroups() != 1 {
		t.Errorf("NOutputGroups: expected 1, got %d", tr.NOutputGroups())
	}
	if tr.Type() != Exponential {
		t.Errorf("Type: expected Exponential, got %d", tr.Type())
	}
	if tr.Name() != "exponential" {
		t.Errorf("Name: expected 'exponential', got '%s'", tr.Name())
	}

	// Test Transform: exp of single value
	tests := []struct {
		input float64
		want  float64
	}{
		{0.0, 1.0},
		{1.0, math.Exp(1.0)},
		{-1.0, math.Exp(-1.0)},
	}
	for _, tc := range tests {
		output := make([]float64, 1)
		err := tr.Transform([]float64{tc.input}, output, 0)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(output[0]-tc.want) > 1e-10 {
			t.Errorf("exp(%f): expected %f, got %f", tc.input, tc.want, output[0])
		}
	}

	// Error: wrong number of predictions
	output := make([]float64, 2)
	err := tr.Transform([]float64{1.0, 2.0}, output, 0)
	if err == nil {
		t.Error("expected error for len(rawPredictions) != 1")
	}
}

func TestTransformSoftmax(t *testing.T) {
	tr := &TransformSoftmax{NClasses: 3}
	if tr.NOutputGroups() != 3 {
		t.Errorf("NOutputGroups: expected 3, got %d", tr.NOutputGroups())
	}
	if tr.Type() != Softmax {
		t.Errorf("Type: expected Softmax, got %d", tr.Type())
	}
	if tr.Name() != "softmax" {
		t.Errorf("Name: expected 'softmax', got '%s'", tr.Name())
	}

	// Test Transform: softmax over 3 classes
	raw := []float64{1.0, 2.0, 3.0}
	output := make([]float64, 3)
	err := tr.Transform(raw, output, 0)
	if err != nil {
		t.Fatal(err)
	}

	// Verify probabilities sum to 1
	sum := 0.0
	for _, v := range output {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax probabilities don't sum to 1: got %f", sum)
	}

	// Verify ordering: higher input -> higher output probability
	if !(output[2] > output[1] && output[1] > output[0]) {
		t.Errorf("softmax ordering wrong: %v", output)
	}

	// Test with startIndex
	output2 := make([]float64, 5)
	err = tr.Transform(raw, output2, 2)
	if err != nil {
		t.Fatal(err)
	}
	sum = 0.0
	for i := 2; i < 5; i++ {
		sum += output2[i]
	}
	if math.Abs(sum-1.0) > 1e-10 {
		t.Errorf("softmax with startIndex doesn't sum to 1: got %f", sum)
	}

	// Error: wrong number of predictions
	output3 := make([]float64, 2)
	err = tr.Transform([]float64{1.0, 2.0}, output3, 0)
	if err == nil {
		t.Error("expected error for len(rawPredictions) != NClasses")
	}
}

func TestTransformLeafIndex(t *testing.T) {
	tr := &TransformLeafIndex{NumOutputGroups: 2}
	if tr.NOutputGroups() != 2 {
		t.Errorf("NOutputGroups: expected 2, got %d", tr.NOutputGroups())
	}
	if tr.Type() != LeafIndex {
		t.Errorf("Type: expected LeafIndex, got %d", tr.Type())
	}
	if tr.Name() != "leaf_index" {
		t.Errorf("Name: expected 'leaf_index', got '%s'", tr.Name())
	}

	// Test Transform: copy values directly
	raw := []float64{3.0, 7.0}
	output := make([]float64, 4)
	err := tr.Transform(raw, output, 0)
	if err != nil {
		t.Fatal(err)
	}
	if output[0] != 3.0 || output[1] != 7.0 {
		t.Errorf("expected [3.0, 7.0], got [%f, %f]", output[0], output[1])
	}

	// Test with startIndex
	output2 := make([]float64, 4)
	err = tr.Transform(raw, output2, 2)
	if err != nil {
		t.Fatal(err)
	}
	if output2[2] != 3.0 || output2[3] != 7.0 {
		t.Errorf("expected [_, _, 3.0, 7.0], got [%f, %f, %f, %f]", output2[0], output2[1], output2[2], output2[3])
	}

	// Empty slice
	output3 := make([]float64, 1)
	err = tr.Transform([]float64{}, output3, 0)
	if err != nil {
		t.Fatal(err)
	}
}
