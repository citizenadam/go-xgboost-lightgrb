package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

	"github.com/citizenadam/go-xgboost-lightgrb"
)

func main() {
	// Read the base64-encoded blob
	data, err := ioutil.ReadFile("testblob.blob")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading blob: %v\n", err)
		os.Exit(1)
	}

	// Decode base64
	raw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(data)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error decoding base64: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Decoded %d bytes\n", len(raw))
	fmt.Printf("First 8 bytes: %x\n", raw[:8])
	fmt.Printf("Starts with {L: %v\n\n", len(raw) >= 2 && raw[0] == '{' && raw[1] == 'L')

	// Load model using the library
	model, err := leaves.XGEnsembleFromUBJSON(bytes.NewReader(raw), false)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading model: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Model loaded successfully!\n")
	fmt.Printf("Number of estimators (trees): %d\n", model.NEstimators())
	fmt.Printf("Number of features: %d\n", model.NFeatures())
	fmt.Printf("Raw output groups: %d\n\n", model.NRawOutputGroups())

	// Use prediction probing to discover features and tree structure
	nFeatures := model.NFeatures()

	// Baseline prediction with all zeros
	baseline := make([]float64, 1)
	model.PredictDense(make([]float64, nFeatures), 1, 1, baseline, 0, 1)
	fmt.Printf("Baseline prediction (all zeros): %.6f\n\n", baseline[0])

	// Probe each feature
	fmt.Println("=== Features with Non-Zero Contribution ===")
	activeFeatures := []struct {
		idx  int
		diff float64
	}{}

	for i := 0; i < nFeatures; i++ {
		vec := make([]float64, nFeatures)
		vec[i] = 1.0
		pred := make([]float64, 1)
		model.PredictDense(vec, 1, 1, pred, 0, 1)
		diff := pred[0] - baseline[0]
		if diff != 0 {
			activeFeatures = append(activeFeatures, struct {
				idx  int
				diff float64
			}{i, diff})
		}
	}

	fmt.Printf("Total features: %d, Active: %d\n", nFeatures, len(activeFeatures))

	// Extract feature names from raw data
	featureNames := extractFeatureNames(raw)
	if len(featureNames) > 0 {
		fmt.Printf("\nFeature names found in model: %d\n", len(featureNames))
	}

	// Binary search for split thresholds
	fmt.Println("\n=== First Tree Branch Points ===")
	for idx, af := range activeFeatures {
		if idx >= 10 {
			break
		}

		low, high := -100.0, 100.0
		for j := 0; j < 40; j++ {
			mid := (low + high) / 2.0
			vecMid := make([]float64, nFeatures)
			vecMid[af.idx] = mid
			predMid := make([]float64, 1)
			model.PredictDense(vecMid, 1, 1, predMid, 0, 1)

			vecBelow := make([]float64, nFeatures)
			vecBelow[af.idx] = mid - 0.0001
			predBelow := make([]float64, 1)
			model.PredictDense(vecBelow, 1, 1, predBelow, 0, 1)

			if predMid[0] != predBelow[0] {
				high = mid
			} else {
				low = mid
			}
		}
		threshold := (low + high) / 2.0

		fname := fmt.Sprintf("feature[%d]", af.idx)
		if af.idx < len(featureNames) && featureNames[af.idx] != "" {
			fname = featureNames[af.idx]
		}
		fmt.Printf("  %s < %.4f (contribution: %+.6f)\n", fname, threshold, af.diff)
	}

	// Direct tree structure from raw bytes
	fmt.Println("\n=== First Tree Structure (from raw bytes) ===")
	analyzeFirstTree(raw, featureNames)
}

func extractFeatureNames(data []byte) []string {
	var names []string
	key := []byte("feature_names")

	for i := 0; i < len(data)-len(key); i++ {
		match := true
		for j := 0; j < len(key); j++ {
			if data[i+j] != key[j] {
				match = false
				break
			}
		}
		if !match {
			continue
		}
		pos := i + len(key)
		for pos < len(data) && pos < i+200 {
			if data[pos] != '[' {
				pos++
				continue
			}
			pos++
			for pos < len(data) && data[pos] != ']' {
				strLen, n := readUBStringLength(data, pos)
				if strLen < 0 || pos+n+strLen > len(data) {
					break
				}
				names = append(names, string(data[pos+n:pos+n+strLen]))
				pos += n + strLen
			}
			return names
		}
	}
	return names
}

func readUBStringLength(data []byte, pos int) (length int, bytesConsumed int) {
	if pos >= len(data) {
		return -1, 0
	}
	switch data[pos] {
	case 'l': // int32
		if pos+4 >= len(data) {
			return -1, 0
		}
		return int(data[pos+1])<<24 | int(data[pos+2])<<16 | int(data[pos+3])<<8 | int(data[pos+4]), 5
	case 'i': // int8
		if pos+1 >= len(data) {
			return -1, 0
		}
		return int(int8(data[pos+1])), 2
	case 'S': // standard UBJSON (1-byte length)
		if pos+1 >= len(data) {
			return -1, 0
		}
		return int(data[pos+1]), 2
	default:
		return -1, 0
	}
}

func analyzeFirstTree(data []byte, featureNames []string) {
	splitCond := findFloatArray(data, []byte("split_conditions"))
	splitIdx := findIntArray(data, []byte("split_indices"))
	leftChild := findIntArray(data, []byte("left_children"))
	rightChild := findIntArray(data, []byte("right_children"))
	baseWeights := findFloatArray(data, []byte("base_weights"))

	if len(splitCond) > 0 && len(splitIdx) > 0 {
		fmt.Printf("Tree has %d split nodes, %d leaf values\n", len(splitCond), len(baseWeights))
		fmt.Println("\nBranch points:")

		for i := 0; i < len(splitCond) && i < 20; i++ {
			fidx := 0
			if i < len(splitIdx) {
				fidx = int(splitIdx[i])
			}
			fname := fmt.Sprintf("feature[%d]", fidx)
			if fidx < len(featureNames) && featureNames[fidx] != "" {
				fname = featureNames[fidx]
			}

			left := -1
			right := -1
			if i < len(leftChild) {
				left = int(leftChild[i])
			}
			if i < len(rightChild) {
				right = int(rightChild[i])
			}

			fmt.Printf("  Node %2d: %s < %12.6f  -> left=%3d, right=%3d\n",
				i, fname, splitCond[i], left, right)
		}
	} else {
		fmt.Println("Could not find tree structure data")
	}
}

func findFloatArray(data, key []byte) []float32 {
	pos := searchKey(data, key)
	if pos < 0 {
		return nil
	}
	// Look for $d# pattern
	for pos < len(data) && pos < len(data)-5 {
		if data[pos] == '$' && pos+2 < len(data) && data[pos+1] == 'd' && data[pos+2] == '#' {
			return readTypedFloat32Array(data, pos+3)
		}
		pos++
	}
	return nil
}

func findIntArray(data, key []byte) []int32 {
	pos := searchKey(data, key)
	if pos < 0 {
		return nil
	}
	for pos < len(data) && pos < len(data)-5 {
		if data[pos] == '$' && pos+2 < len(data) && data[pos+1] == 'l' && data[pos+2] == '#' {
			return readTypedInt32Array(data, pos+3)
		}
		pos++
	}
	return nil
}

func searchKey(data, key []byte) int {
	for i := 0; i <= len(data)-len(key); i++ {
		match := true
		for j := 0; j < len(key); j++ {
			if data[i+j] != key[j] {
				match = false
				break
			}
		}
		if match {
			return i + len(key)
		}
	}
	return -1
}

func readTypedFloat32Array(data []byte, pos int) []float32 {
	count, n := readUBInt(data, pos)
	if count < 0 {
		return nil
	}
	pos += n
	result := make([]float32, count)
	for i := 0; i < int(count) && pos+4 <= len(data); i++ {
		bits := uint32(data[pos])<<24 | uint32(data[pos+1])<<16 | uint32(data[pos+2])<<8 | uint32(data[pos+3])
		result[i] = float32frombits(bits)
		pos += 4
	}
	return result
}

func readTypedInt32Array(data []byte, pos int) []int32 {
	count, n := readUBInt(data, pos)
	if count < 0 {
		return nil
	}
	pos += n
	result := make([]int32, count)
	for i := 0; i < int(count) && pos+4 <= len(data); i++ {
		result[i] = int32(data[pos])<<24 | int32(data[pos+1])<<16 | int32(data[pos+2])<<8 | int32(data[pos+3])
		pos += 4
	}
	return result
}

func readUBInt(data []byte, pos int) (value int64, bytesConsumed int) {
	if pos >= len(data) {
		return -1, 0
	}
	switch data[pos] {
	case 'l': // int32
		if pos+4 >= len(data) {
			return -1, 0
		}
		v := int64(data[pos+1])<<24 | int64(data[pos+2])<<16 | int64(data[pos+3])<<8 | int64(data[pos+4])
		return v, 5
	case 'i': // int8
		if pos+1 >= len(data) {
			return -1, 0
		}
		return int64(int8(data[pos+1])), 2
	case 'L': // int64
		if pos+8 >= len(data) {
			return -1, 0
		}
		v := int64(data[pos+1])<<56 | int64(data[pos+2])<<48 | int64(data[pos+3])<<40 | int64(data[pos+4])<<32 |
			int64(data[pos+5])<<24 | int64(data[pos+6])<<16 | int64(data[pos+7])<<8 | int64(data[pos+8])
		return v, 9
	default:
		return -1, 0
	}
}

func float32frombits(bits uint32) float32 {
	// IEEE 754 conversion using unsafe pointer cast alternative
	// Use math package for portable conversion
	return float32(float64(bits))
}
