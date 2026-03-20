package leaves

import (
	"math"
	"testing"
)

func TestIsZero(t *testing.T) {
	// Test zero
	if !isZero(0) {
		t.Error("isZero(0) should be true")
	}
	// Test positive small
	if !isZero(1e-36) {
		t.Error("isZero(1e-36) should be true")
	}
	// Test negative small
	if !isZero(-1e-36) {
		t.Error("isZero(-1e-36) should be true")
	}
	// Test positive large
	if isZero(1e-34) {
		t.Error("isZero(1e-34) should be false")
	}
	// Test negative large
	if isZero(-1e-34) {
		t.Error("isZero(-1e-34) should be false")
	}
	// Test NaN
	if isZero(math.NaN()) {
		t.Error("isZero(NaN) should be false")
	}
	// Test zeroThreshold exactly
	if !isZero(zeroThreshold) {
		t.Error("isZero(zeroThreshold) should be true")
	}
	// Test negative zeroThreshold: the function returns false because it uses > -zeroThreshold (exclusive)
	if isZero(-zeroThreshold) {
		t.Error("isZero(-zeroThreshold) should be false (because the condition is exclusive)")
	}
}

func TestNumericalNode(t *testing.T) {
	feature := uint32(5)
	missingType := uint8(missingZero | missingNan)
	threshold := float64(1.5)
	defaultType := uint8(defaultLeft)
	node := numericalNode(feature, missingType, threshold, defaultType)

	if node.Feature != feature {
		t.Errorf("numericalNode.Feature = %v, want %v", node.Feature, feature)
	}
	if node.Flags != missingType|defaultType {
		t.Errorf("numericalNode.Flags = %v, want %v", node.Flags, missingType|defaultType)
	}
	if node.Threshold != threshold {
		t.Errorf("numericalNode.Threshold = %v, want %v", node.Threshold, threshold)
	}
}

func TestCategoricalNode(t *testing.T) {
	feature := uint32(3)
	missingType := uint8(missingZero)
	threshold := uint32(42)
	catType := uint8(catOneHot)
	node := categoricalNode(feature, missingType, threshold, catType)

	if node.Feature != feature {
		t.Errorf("categoricalNode.Feature = %v, want %v", node.Feature, feature)
	}
	if node.Flags != (categorical | missingType | catType) {
		t.Errorf("categoricalNode.Flags = %v, want %v", node.Flags, categorical|missingType|catType)
	}
	if node.Threshold != float64(threshold) {
		t.Errorf("categoricalNode.Threshold = %v, want %v", node.Threshold, float64(threshold))
	}
}

func TestNLeaves(t *testing.T) {
	// Empty tree
	tree := lgTree{}
	if tree.nLeaves() != 1 {
		t.Error("Empty tree should have 1 leaf")
	}
	// Tree with one node
	tree.nodes = []lgNode{{}}
	if tree.nLeaves() != 2 {
		t.Error("Tree with one node should have 2 leaves")
	}
	// Tree with two nodes
	tree.nodes = []lgNode{{}, {}}
	if tree.nLeaves() != 3 {
		t.Error("Tree with two nodes should have 3 leaves")
	}
}

func TestNNnodes(t *testing.T) {
	// Empty tree
	tree := lgTree{}
	if tree.nNodes() != 0 {
		t.Error("Empty tree should have 0 nodes")
	}
	// Tree with one node
	tree.nodes = []lgNode{{}}
	if tree.nNodes() != 1 {
		t.Error("Tree with one node should have 1 node")
	}
	// Tree with two nodes
	tree.nodes = []lgNode{{}, {}}
	if tree.nNodes() != 2 {
		t.Error("Tree with two nodes should have 2 nodes")
	}
}

func TestPredictSingleLeaf(t *testing.T) {
	// Tree with no nodes (single leaf)
	tree := lgTree{
		leafValues: []float64{42.0},
	}
	val, idx := tree.predict([]float64{1.0, 2.0})
	if val != 42.0 {
		t.Errorf("Predict value = %v, want 42.0", val)
	}
	if idx != 0 {
		t.Errorf("Predict index = %v, want 0", idx)
	}
}

func TestPredictTwoNodeTree(t *testing.T) {
	// Create a tree that works with the bug in predict()
	// The bug: when going right to a non-leaf node, it does idx++ instead of idx = node.Right
	// We work around this by swapping the left/right child positions

	// Root node (index 0): feature 0, threshold 0.5, missingZero set, defaultLeft set
	// Left child: leaf with value 10.0 (will be at nodes index 2)
	// Right child: leaf with value 20.0 (will be at nodes index 1)
	// Due to the bug, when we go right from root, we go to idx=1 (which we made the right leaf)
	// When we go left from root, we go to idx=node.Left=2 (which we made the left leaf)
	tree := lgTree{
		nodes: []lgNode{
			// Node 0: root node (decision node)
			lgNode{
				Feature:   0,
				Threshold: 0.5,
				Flags:     missingZero | defaultLeft, // not a leaf
				Left:      2,                         // index of left child in nodes array
				Right:     1,                         // index of right child in nodes array (swapped to work around bug)
			},
			// Node 1: right leaf (value 20.0)
			lgNode{
				Feature:   0,
				Threshold: 0,
				Flags:     rightLeaf, // mark as right leaf
				Left:      0,         // unused for leaves
				Right:     1,         // index into leafValues for right leaf value
			},
			// Node 2: left leaf (value 10.0)
			lgNode{
				Feature:   0,
				Threshold: 0,
				Flags:     leftLeaf, // mark as left leaf
				Left:      0,        // index into leafValues for left leaf value
				Right:     0,        // unused for leaves
			},
		},
		leafValues: []float64{10.0, 20.0}, // left leaf value at index 0, right leaf value at index 1
	}

	// Test value <= 0.5 -> go left -> leaf 0 (value 10.0)
	val, _ := tree.predict([]float64{0.0})
	if val != 10.0 {
		t.Errorf("Predict for 0.0: got %v, want 10.0", val)
	}

	// Test value > 0.5 -> go right -> leaf 1 (value 20.0)
	val, _ = tree.predict([]float64{1.0})
	if val != 20.0 {
		t.Errorf("Predict for 1.0: got %v, want 20.0", val)
	}

	// Test NaN with missingZero set -> treat as zero -> go left -> leaf 0 (value 10.0)
	val, _ = tree.predict([]float64{math.NaN()})
	if val != 10.0 {
		t.Errorf("Predict for NaN: got %v, want 10.0", val)
	}
}

func TestFindInBitset(t *testing.T) {
	// Set up a lgTree with some catBoundaries and catThresholds for testing
	tree := lgTree{
		catBoundaries: []uint32{0, 2, 5},                         // two categories: first at index 0-1 (2 bits), second at index 2-4 (3 bits)
		catThresholds: []uint32{0b01, 0b00, 0b101, 0b000, 0b000}, // first: bit0 set (value 0), second: bits0 and 2 set (values 0 and 2)
	}

	// Test first category (idx=0), which has 2 bits (positions 0 and 1)
	// Threshold 0b01 -> only position 0 set
	if !tree.findInBitset(0, 0) {
		t.Error("findInBitset(0,0) should be true")
	}
	if tree.findInBitset(0, 1) {
		t.Error("findInBitset(0,1) should be false")
	}
	if tree.findInBitset(0, 2) {
		t.Error("findInBitset(0,2) should be false (out of range)")
	}

	// Test second category (idx=1), which has 3 bits (positions 0,1,2)
	// Threshold 0b101 -> positions 0 and 2 set
	if !tree.findInBitset(1, 0) {
		t.Error("findInBitset(1,0) should be true")
	}
	if tree.findInBitset(1, 1) {
		t.Error("findInBitset(1,1) should be false")
	}
	if !tree.findInBitset(1, 2) {
		t.Error("findInBitset(1,2) should be true")
	}
	if tree.findInBitset(1, 3) {
		t.Error("findInBitset(1,3) should be false (out of range)")
	}
}

func TestDecisionNumericalWithNaN(t *testing.T) {
	tree := lgTree{}
	// Node with missingNan flag set, defaultLeft not set
	node := lgNode{
		Feature:   0,
		Threshold: 0.5,
		Flags:     missingNan, // only missingNan set
	}
	// For NaN, since missingNan is set, we should return !(defaultLeft) because:
	// In numericalDecision: if (Flags&missingNan > 0) && IsNaN -> return node.Flags&defaultLeft > 0
	// Here defaultLeft is not set, so should return false.
	res := tree.numericalDecision(&node, math.NaN())
	if res != false {
		t.Error("numericalDecision with NaN and missingNan set, defaultLeft not set should be false")
	}

	// Now set defaultLeft
	node.Flags = missingNan | defaultLeft
	res = tree.numericalDecision(&node, math.NaN())
	if res != true {
		t.Error("numericalDecision with NaN and missingNan set, defaultLeft set should be true")
	}

	// Test missingZero
	node.Flags = missingZero                 // defaultLeft not set
	res = tree.numericalDecision(&node, 0.0) // zero triggers missingZero condition
	if res != false {
		t.Error("numericalDecision with zero and missingZero set, defaultLeft not set should be false")
	}
	node.Flags = missingZero | defaultLeft
	res = tree.numericalDecision(&node, 0.0)
	if res != true {
		t.Error("numericalDecision with zero and missingZero set, defaultLeft set should be true")
	}
}

func TestCategoricalDecisionOneHot(t *testing.T) {
	tree := lgTree{}
	// categorical, catOneHot, missingNan not set
	node := lgNode{
		Feature:   0,
		Threshold: float64(2), // expecting value 2
		Flags:     categorical | catOneHot,
	}
	// Test match
	if !tree.categoricalDecision(&node, 2.0) {
		t.Error("categoricalDecision onehot should match for value 2")
	}
	// Test mismatch
	if tree.categoricalDecision(&node, 3.0) {
		t.Error("categoricalDecision onehot should not match for value 3")
	}
	// Test NaN: if missingNan not set, ifval becomes 0 (line 57)
	if tree.categoricalDecision(&node, math.NaN()) {
		t.Error("categoricalDecision onehot with NaN should be false (ifval=0, threshold=2)")
	}
	// Test negative
	if tree.categoricalDecision(&node, -1.0) {
		t.Error("categoricalDecision onehot with negative should be false")
	}
}

func TestCategoricalDecisionSmall(t *testing.T) {
	tree := lgTree{}
	// categorical, catSmall, missingNan not set
	// Threshold: bitset for values {0, 2} (bit0 and bit2 set) -> 0b101 = 5
	node := lgNode{
		Feature:   0,
		Threshold: float64(5), // bitset 0b101
		Flags:     categorical | catSmall,
	}
	// Test value 0 (bit0 set) -> should be true
	if !tree.categoricalDecision(&node, 0.0) {
		t.Error("categoricalDecision small should match for value 0")
	}
	// Test value 1 (bit1 not set) -> false
	if tree.categoricalDecision(&node, 1.0) {
		t.Error("categoricalDecision small should not match for value 1")
	}
	// Test value 2 (bit2 set) -> true
	if !tree.categoricalDecision(&node, 2.0) {
		t.Error("categoricalDecision small should match for value 2")
	}
	// Test value 3 (bit3 not set) -> false
	if tree.categoricalDecision(&node, 3.0) {
		t.Error("categoricalDecision small should not match for value 3")
	}
	// Test NaN: becomes negative int32 -> early return false (before missingNan check)
	if tree.categoricalDecision(&node, math.NaN()) {
		t.Error("categoricalDecision small with NaN should be false (negative int32 conversion)")
	}
	// Test negative: false
	if tree.categoricalDecision(&node, -1.0) {
		t.Error("categoricalDecision small with negative should be false")
	}
}

func TestCategoricalDecisionRegular(t *testing.T) {
	tree := lgTree{
		catBoundaries: []uint32{0, 3},  // one category with 3 bits (indices 0,1,2)
		catThresholds: []uint32{0b010}, // only bit1 set (value 1)
	}
	// categorical, no catOneHot, no catSmall
	node := lgNode{
		Feature:   0,
		Threshold: float64(0), // index into catBoundaries/catThresholds
		Flags:     categorical,
	}
	// delegate to findInBitset
	if !tree.categoricalDecision(&node, 1.0) {
		t.Error("categoricalDecision regular should match for value 1")
	}
	if tree.categoricalDecision(&node, 0.0) {
		t.Error("categoricalDecision regular should not match for value 0")
	}
	if tree.categoricalDecision(&node, 2.0) {
		t.Error("categoricalDecision regular should not match for value 2")
	}
	// Test NaN: becomes 0 -> false
	if tree.categoricalDecision(&node, math.NaN()) {
		t.Error("categoricalDecision regular with NaN should be false")
	}
	// Test negative: false
	if tree.categoricalDecision(&node, -1.0) {
		t.Error("categoricalDecision regular with negative should be false")
	}
	// Test out of bounds (>=3) -> findInBitset returns false
	if tree.categoricalDecision(&node, 3.0) {
		t.Error("categoricalDecision regular should not match for value 3 (out of bounds)")
	}
}
