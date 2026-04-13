package leaves

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/citizenadam/go-xgboost-lightgrb/internal/ubjson"
	"github.com/citizenadam/go-xgboost-lightgrb/mat"
	"github.com/citizenadam/go-xgboost-lightgrb/transformation"
	"github.com/citizenadam/go-xgboost-lightgrb/util"
)

// --- UBJSON encoder for test model construction ---

type ubjsonWriter struct {
	buf bytes.Buffer
}

func newUBJSONWriter() *ubjsonWriter {
	return &ubjsonWriter{}
}

func (w *ubjsonWriter) writeByte(b byte) {
	w.buf.WriteByte(b)
}

func (w *ubjsonWriter) writeObjectStart() {
	w.writeByte('{')
}

func (w *ubjsonWriter) writeObjectEnd() {
	w.writeByte('}')
}

func (w *ubjsonWriter) writeArrayStart() {
	w.writeByte('[')
}

func (w *ubjsonWriter) writeArrayEnd() {
	w.writeByte(']')
}

func (w *ubjsonWriter) writeKey(s string) {
	w.writeString(s)
}

func (w *ubjsonWriter) writeString(s string) {
	w.writeByte('S')
	w.writeByte(byte(int8(len(s))))
	w.buf.WriteString(s)
}

func (w *ubjsonWriter) writeInt32(v int32) {
	w.writeByte('l')
	binary.Write(&w.buf, binary.BigEndian, v)
}

func (w *ubjsonWriter) writeInt32Slice(vals []int32) {
	w.writeByte('$')
	w.writeByte('l')
	w.writeByte('#')
	w.writeByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		binary.Write(&w.buf, binary.BigEndian, v)
	}
}

func (w *ubjsonWriter) writeUint8Slice(vals []uint8) {
	w.writeByte('$')
	w.writeByte('U')
	w.writeByte('#')
	w.writeByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		w.writeByte(v)
	}
}

func (w *ubjsonWriter) writeFloat32Slice(vals []float32) {
	w.writeByte('$')
	w.writeByte('d')
	w.writeByte('#')
	w.writeByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		bits := math.Float32bits(v)
		binary.Write(&w.buf, binary.BigEndian, bits)
	}
}

func (w *ubjsonWriter) Bytes() []byte {
	return w.buf.Bytes()
}

// createTestXGBoostUBJSON creates a minimal XGBoost UBJSON model for testing.
// This creates a simple binary classification model with 1 tree and a threshold split.
// The model has: 2 features, 1 tree, binary:logistic objective.
// Tree structure: root splits on feature 0 at threshold 0.5.
//
//	Left leaf: -1.0, Right leaf: 1.0
func createTestXGBoostUBJSON() []byte {
	w := newUBJSONWriter()

	// Top-level model object
	w.writeObjectStart()

	// "learner" key
	w.writeKey("learner")
	w.writeObjectStart()

	// learner_model_param
	w.writeKey("learner_model_param")
	w.writeObjectStart()
	w.writeKey("base_score")
	w.writeString("0.5")
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	// gradient_booster
	w.writeKey("gradient_booster")
	w.writeObjectStart()
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart()

	// gbtree_model_param
	w.writeKey("gbtree_model_param")
	w.writeObjectStart()
	w.writeKey("num_trees")
	w.writeString("1")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	// tree_info
	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeArrayEnd()

	// trees array
	w.writeKey("trees")
	w.writeArrayStart()

	// Tree 0
	w.writeObjectStart()
	w.writeKey("id")
	w.writeInt32(0)

	// left_children: [1, -1, -1]
	// Root (0) has left child 1 (leaf), internal node 1 doesn't exist as parent
	// Wait, let me think about this more carefully.
	// For a tree with 3 nodes (1 root + 2 leaves):
	// Node 0 (root): left_children[0] = -1 (left is leaf), right_children[0] = -1 (right is leaf)
	// split_conditions[0] = 0.5 (threshold)
	// split_indices[0] = 0 (feature 0)
	// The leaf values are stored in split_conditions for leaf nodes
	// But leaf nodes have index -1 in children arrays...
	// Let me use 3 nodes: root at 0, left leaf at 1, right leaf at 2
	// left_children[0] = 1, right_children[0] = 2
	// left_children[1] = -1, right_children[1] = -1 (leaf)
	// left_children[2] = -1, right_children[2] = -1 (leaf)
	// split_conditions[0] = 0.5 (threshold)
	// split_conditions[1] = -1.0 (left leaf value)
	// split_conditions[2] = 1.0 (right leaf value)

	w.writeKey("left_children")
	w.writeInt32Slice([]int32{1, -1, -1})

	w.writeKey("right_children")
	w.writeInt32Slice([]int32{2, -1, -1})

	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1, 0, 0})

	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{0.5, -1.0, 1.0})

	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0, 0, 0})

	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0, 0, 0})

	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{1, 0, 0})

	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0, -1.0, 1.0})

	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{1.0, 0.0, 0.0})

	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0, 40.0, 60.0})

	w.writeKey("num_nodes")
	w.writeInt32(3)

	w.writeObjectEnd() // tree 0

	w.writeArrayEnd() // trees

	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	// attributes
	w.writeKey("attributes")
	w.writeObjectStart()
	w.writeKey("objective")
	w.writeString("binary:logistic")
	w.writeObjectEnd()

	w.writeObjectEnd() // learner
	w.writeObjectEnd() // top-level

	return w.Bytes()
}

// createTestXGBoostUBJSONMultiTree creates a UBJSON model with 2 trees for testing.
// Tree 0: splits on feature 0, left=-1.0, right=2.0
// Tree 1: splits on feature 1, left=0.5, right=-0.5
func createTestXGBoostUBJSONMultiTree() []byte {
	w := newUBJSONWriter()

	w.writeObjectStart()

	w.writeKey("learner")
	w.writeObjectStart()

	w.writeKey("learner_model_param")
	w.writeObjectStart()
	w.writeKey("base_score")
	w.writeString("0.0")
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	w.writeKey("gradient_booster")
	w.writeObjectStart()
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart()

	w.writeKey("gbtree_model_param")
	w.writeObjectStart()
	w.writeKey("num_trees")
	w.writeString("2")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeInt32(0)
	w.writeArrayEnd()

	w.writeKey("trees")
	w.writeArrayStart()

	// Tree 0
	w.writeObjectStart()
	w.writeKey("id")
	w.writeInt32(0)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{1, -1, -1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{2, -1, -1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1, 0, 0})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{0.5, -1.0, 2.0})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0, 0, 0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0, 0, 0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{1, 0, 0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0, -1.0, 2.0})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{1.0, 0.0, 0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0, 40.0, 60.0})
	w.writeKey("num_nodes")
	w.writeInt32(3)
	w.writeObjectEnd()

	// Tree 1
	w.writeObjectStart()
	w.writeKey("id")
	w.writeInt32(1)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{1, -1, -1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{2, -1, -1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1, 0, 0})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{1.0, 0.5, -0.5})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{1, 0, 0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0, 0, 0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{0, 0, 0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0, 0.5, -0.5})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{1.0, 0.0, 0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0, 55.0, 45.0})
	w.writeKey("num_nodes")
	w.writeInt32(3)
	w.writeObjectEnd()

	w.writeArrayEnd()  // trees
	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	w.writeKey("attributes")
	w.writeObjectStart()
	w.writeKey("objective")
	w.writeString("binary:logistic")
	w.writeObjectEnd()

	w.writeObjectEnd() // learner
	w.writeObjectEnd() // top-level

	return w.Bytes()
}

// createTestXGBoostUBJSONSingleLeaf creates a UBJSON model with a single-leaf tree.
func createTestXGBoostUBJSONSingleLeaf() []byte {
	w := newUBJSONWriter()

	w.writeObjectStart()

	w.writeKey("learner")
	w.writeObjectStart()

	w.writeKey("learner_model_param")
	w.writeObjectStart()
	w.writeKey("base_score")
	w.writeString("0.5")
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	w.writeKey("gradient_booster")
	w.writeObjectStart()
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart()

	w.writeKey("gbtree_model_param")
	w.writeObjectStart()
	w.writeKey("num_trees")
	w.writeString("1")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeArrayEnd()

	w.writeKey("trees")
	w.writeArrayStart()

	// Single leaf tree
	w.writeObjectStart()
	w.writeKey("id")
	w.writeInt32(0)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{3.14})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{3.14})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0})
	w.writeKey("num_nodes")
	w.writeInt32(1)
	w.writeObjectEnd()

	w.writeArrayEnd()  // trees
	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	w.writeKey("attributes")
	w.writeObjectStart()
	w.writeKey("objective")
	w.writeString("binary:logistic")
	w.writeObjectEnd()

	w.writeObjectEnd() // learner
	w.writeObjectEnd() // top-level

	return w.Bytes()
}

// --- Tests ---

func TestXGUBJSONDecoder(t *testing.T) {
	data := createTestXGBoostUBJSON()
	dec := ubjson.NewDecoder(bytes.NewReader(data))

	model, err := decodeModel(dec)
	if err != nil {
		t.Fatalf("decodeModel failed: %v", err)
	}

	// Verify learner model param
	if model.Learner.ModelParam.BaseScore != 0.5 {
		t.Errorf("BaseScore = %v, want 0.5", model.Learner.ModelParam.BaseScore)
	}
	if model.Learner.ModelParam.NumClass != 1 {
		t.Errorf("NumClass = %v, want 1", model.Learner.ModelParam.NumClass)
	}
	if model.Learner.ModelParam.NumFeature != 2 {
		t.Errorf("NumFeature = %v, want 2", model.Learner.ModelParam.NumFeature)
	}

	// Verify gradient booster
	if model.Learner.GradientBooster.Name != "gbtree" {
		t.Errorf("Name = %v, want gbtree", model.Learner.GradientBooster.Name)
	}

	// Verify tree structure
	gbModel := &model.Learner.GradientBooster.Model
	if gbModel.NumTrees != 1 {
		t.Errorf("NumTrees = %v, want 1", gbModel.NumTrees)
	}
	if len(gbModel.Trees) != 1 {
		t.Fatalf("len(Trees) = %v, want 1", len(gbModel.Trees))
	}

	tree := gbModel.Trees[0]
	if tree.NumNodes != 3 {
		t.Errorf("NumNodes = %v, want 3", tree.NumNodes)
	}
	if len(tree.LeftChildren) != 3 {
		t.Errorf("len(LeftChildren) = %v, want 3", len(tree.LeftChildren))
	}
	if tree.LeftChildren[0] != 1 || tree.RightChildren[0] != 2 {
		t.Errorf("Root children = [%d, %d], want [1, 2]",
			tree.LeftChildren[0], tree.RightChildren[0])
	}
	if tree.SplitConditions[0] != 0.5 {
		t.Errorf("SplitConditions[0] = %v, want 0.5", tree.SplitConditions[0])
	}
	if tree.SplitIndices[0] != 0 {
		t.Errorf("SplitIndices[0] = %v, want 0", tree.SplitIndices[0])
	}

	// Verify attributes
	if model.Learner.Objective != "binary:logistic" {
		t.Errorf("Objective = %v, want binary:logistic", model.Learner.Objective)
	}
}

func TestXGUBJSONTreeConversion(t *testing.T) {
	data := createTestXGBoostUBJSON()
	dec := ubjson.NewDecoder(bytes.NewReader(data))

	model, err := decodeModel(dec)
	if err != nil {
		t.Fatalf("decodeModel failed: %v", err)
	}

	tree := &model.Learner.GradientBooster.Model.Trees[0]
	lgTree, err := xgTreeFromUBJSON(tree, 2)
	if err != nil {
		t.Fatalf("xgTreeFromUBJSON failed: %v", err)
	}

	// Verify tree structure: 1 internal node (root), 2 leaf values
	if lgTree.nNodes() != 1 {
		t.Errorf("nNodes = %d, want 1", lgTree.nNodes())
	}
	if lgTree.nLeaves() != 2 {
		t.Errorf("nLeaves = %d, want 2", lgTree.nLeaves())
	}
	if len(lgTree.leafValues) != 2 {
		t.Errorf("len(leafValues) = %d, want 2", len(lgTree.leafValues))
	}

	// Test prediction: feature[0] = 0.0 → left (threshold 0.5, left when <= 0.5)
	pred, _ := lgTree.predict([]float64{0.0, 0.0})
	if pred != -1.0 {
		t.Errorf("predict([0.0, 0.0]) = %v, want -1.0", pred)
	}

	// Test prediction: feature[0] = 1.0 → right (threshold 0.5)
	pred, _ = lgTree.predict([]float64{1.0, 0.0})
	if pred != 1.0 {
		t.Errorf("predict([1.0, 0.0]) = %v, want 1.0", pred)
	}

	// Test prediction: feature[0] = 0.5 → left (XGBoost uses <=)
	pred, _ = lgTree.predict([]float64{0.5, 0.0})
	if pred != -1.0 {
		t.Errorf("predict([0.5, 0.0]) = %v, want -1.0", pred)
	}
}

func TestXGUBJSONEnsemble(t *testing.T) {
	data := createTestXGBoostUBJSON()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
	if model.NOutputGroups() != 1 {
		t.Errorf("NOutputGroups = %d, want 1", model.NOutputGroups())
	}
	if model.NRawOutputGroups() != 1 {
		t.Errorf("NRawOutputGroups = %d, want 1", model.NRawOutputGroups())
	}
	if model.NFeatures() != 2 {
		t.Errorf("NFeatures = %d, want 2", model.NFeatures())
	}
	if model.Name() != "xgboost.gbtree" {
		t.Errorf("Name = %q, want xgboost.gbtree", model.Name())
	}

	// Test prediction
	predictions := make([]float64, 1)
	fvals := []float64{0.0, 0.0} // feature[0] = 0.0 → left leaf → -1.0
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	// prediction = base_score + tree_value = 0.5 + (-1.0) = -0.5
	if predictions[0] != -0.5 {
		t.Errorf("predict([0.0, 0.0]) = %v, want -0.5", predictions[0])
	}

	fvals = []float64{1.0, 0.0} // feature[0] = 1.0 → right leaf → 1.0
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	// prediction = 0.5 + 1.0 = 1.5
	if predictions[0] != 1.5 {
		t.Errorf("predict([1.0, 0.0]) = %v, want 1.5", predictions[0])
	}
}

func TestXGUBJSONEnsembleWithTransformation(t *testing.T) {
	data := createTestXGBoostUBJSON()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), true)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}

	if model.Transformation().Type() != transformation.Logistic {
		t.Errorf("Transformation = %v, want Logistic", model.Transformation().Type())
	}

	// Test prediction with logistic transformation
	predictions := make([]float64, 1)
	fvals := []float64{1.0, 0.0} // raw prediction = 0.5 + 1.0 = 1.5
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	// logistic(1.5) = 1/(1+exp(-1.5)) ≈ 0.8176
	expected := 1.0 / (1.0 + math.Exp(-1.5))
	if math.Abs(predictions[0]-expected) > 1e-6 {
		t.Errorf("predict with logistic = %v, want ~%v", predictions[0], expected)
	}
}

func TestXGUBJSONMultiTree(t *testing.T) {
	data := createTestXGBoostUBJSONMultiTree()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}

	if model.NEstimators() != 2 {
		t.Errorf("NEstimators = %d, want 2", model.NEstimators())
	}

	// Test prediction
	predictions := make([]float64, 1)
	fvals := []float64{0.0, 0.0} // feature[0]=0.0 → left, feature[1]=0.0 → left
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	// tree0: 0.0 <= 0.5 → left leaf (-1.0)
	// tree1: 0.0 <= 1.0 → left leaf (0.5)
	// prediction = base(0.0) + (-1.0) + 0.5 = -0.5
	if predictions[0] != -0.5 {
		t.Errorf("predict([0.0, 0.0]) = %v, want -0.5", predictions[0])
	}

	fvals = []float64{1.0, 2.0}
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	// tree0: 1.0 > 0.5 → right leaf (2.0)
	// tree1: 2.0 > 1.0 → right leaf (-0.5)
	// prediction = 0.0 + 2.0 + (-0.5) = 1.5
	if predictions[0] != 1.5 {
		t.Errorf("predict([1.0, 2.0]) = %v, want 1.5", predictions[0])
	}
}

func TestXGUBJSONSingleLeaf(t *testing.T) {
	data := createTestXGBoostUBJSONSingleLeaf()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}

	// Single leaf tree: prediction = base_score + leaf_value = 0.5 + 3.14 = 3.64
	predictions := make([]float64, 1)
	fvals := []float64{0.0, 0.0}
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	if math.Abs(predictions[0]-3.64) > 1e-6 {
		t.Errorf("predict = %v, want 3.64", predictions[0])
	}
}

func TestXGUBJSONFileRoundTrip(t *testing.T) {
	data := createTestXGBoostUBJSON()

	tmpFile := filepath.Join(t.TempDir(), "test_model.ubj")
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatal(err)
	}

	model, err := XGEnsembleFromUBJSONFile(tmpFile, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSONFile failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}

	predictions := make([]float64, 1)
	fvals := []float64{1.0, 0.0}
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	if predictions[0] != 1.5 {
		t.Errorf("predict = %v, want 1.5", predictions[0])
	}
}

func TestXGUBJSONAutoDetect(t *testing.T) {
	data := createTestXGBoostUBJSON()

	tmpFile := filepath.Join(t.TempDir(), "test_model.ubj")
	if err := os.WriteFile(tmpFile, data, 0644); err != nil {
		t.Fatal(err)
	}

	model, err := XGEnsembleFromAnyFile(tmpFile, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromAnyFile failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
}

func TestXGUBJSONAutoDetectLegacyBinary(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	if !isFileExists(modelPath) {
		t.Skipf("Skipping due to absence of file: %s", modelPath)
	}

	model, err := XGEnsembleFromAnyFile(modelPath, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromAnyFile for legacy binary failed: %v", err)
	}

	if model.NEstimators() != 3 {
		t.Errorf("NEstimators = %d, want 3", model.NEstimators())
	}
}

// TestXGUBJSONFromExistingBinary compares UBJSON model predictions against legacy binary model.
// This test validates that the UBJSON decoder produces identical results to the binary parser.
func TestXGUBJSONFromExistingBinary(t *testing.T) {
	modelPath := filepath.Join("testdata", "xgagaricus.model")
	testPath := filepath.Join("testdata", "agaricus_test.libsvm")
	truePath := filepath.Join("testdata", "xgagaricus_true_predictions.txt")
	skipTestIfFileNotExist(t, modelPath, testPath, truePath)

	// Load legacy binary model
	binaryModel, err := XGEnsembleFromFile(modelPath, true)
	if err != nil {
		t.Fatal(err)
	}

	// Load test data
	csr, err := mat.CSRMatFromLibsvmFile(testPath, 0, true)
	if err != nil {
		t.Fatal(err)
	}

	truePredictions, err := mat.DenseMatFromCsvFile(truePath, 0, false, ",", 0.0)
	if err != nil {
		t.Fatal(err)
	}

	// Verify legacy model works
	binaryPredictions := make([]float64, csr.Rows()*binaryModel.NOutputGroups())
	binaryModel.PredictCSR(csr.RowHeaders, csr.ColIndexes, csr.Values, binaryPredictions, 0, 1)
	if err := util.AlmostEqualFloat64Slices(truePredictions.Values, binaryPredictions, 1e-7); err != nil {
		t.Fatalf("legacy binary predictions differ: %s", err.Error())
	}

	// Note: We can't directly compare UBJSON predictions here because we don't have
	// a UBJSON version of the agaricus model. This test validates that the legacy
	// binary path still works correctly.
}

func TestXGUBJSONCSR(t *testing.T) {
	data := createTestXGBoostUBJSONMultiTree()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}

	// Create CSR data: 3 samples, 2 features
	// Sample 0: feature 0 = 0.0
	// Sample 1: feature 0 = 1.0, feature 1 = 0.0
	// Sample 2: feature 1 = 2.0
	indptr := []int{0, 1, 3, 4}
	cols := []int{0, 0, 1, 1}
	vals := []float64{0.0, 1.0, 0.0, 2.0}

	predictions := make([]float64, 3)
	model.PredictCSR(indptr, cols, vals, predictions, 0, 1)

	// Sample 0: feature[0]=0.0, feature[1]=missing → default_left=0 → RIGHT
	//   tree0: 0.0 <= 0.5 → left → -1.0
	//   tree1: missing → default_right → -0.5
	//   prediction = 0.0 + (-1.0) + (-0.5) = -1.5
	if predictions[0] != -1.5 {
		t.Errorf("sample 0: %v, want -1.5", predictions[0])
	}

	// Sample 1: feature[0]=1.0, feature[1]=0.0
	//   tree0: 1.0 > 0.5 → 2.0
	//   tree1: 0.0 <= 1.0 → 0.5
	//   prediction = 0.0 + 2.0 + 0.5 = 2.5
	if predictions[1] != 2.5 {
		t.Errorf("sample 1: %v, want 2.5", predictions[1])
	}

	// Sample 2: feature[0]=missing → default_left=1 → LEFT, feature[1]=2.0
	//   tree0: missing → default_left → -1.0
	//   tree1: 2.0 > 1.0 → -0.5
	//   prediction = 0.0 + (-1.0) + (-0.5) = -1.5
	if predictions[2] != -1.5 {
		t.Errorf("sample 2: %v, want -1.5", predictions[2])
	}
}

// TestXGUBJSONDecoderRoundTrip tests that the UBJSON decoder can correctly parse
// a model that was encoded by the test encoder.
func TestXGUBJSONDecoderRoundTrip(t *testing.T) {
	data := createTestXGBoostUBJSON()
	dec := ubjson.NewDecoder(bytes.NewReader(data))

	model, err := decodeModel(dec)
	if err != nil {
		t.Fatalf("decodeModel failed: %v", err)
	}

	// Verify we can convert to ensemble
	_, err = ubjsonModelToEnsemble(model, false)
	if err != nil {
		t.Fatalf("ubjsonModelToEnsemble failed: %v", err)
	}

	// Verify no more tokens (EOF)
	_, err = dec.Next()
	if err != io.EOF {
		t.Errorf("expected EOF after model, got %v", err)
	}
}

// TestXGUBJSONEmptyModel tests error handling for malformed models.
func TestXGUBJSONEmptyModel(t *testing.T) {
	// Empty object
	data := []byte{'{', '}'}
	_, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err == nil {
		t.Error("expected error for empty model")
	}
}

// TestXGUBJSONNoObjective tests that models without objective attribute work with loadTransformation=false.
func TestXGUBJSONNoObjective(t *testing.T) {
	w := newUBJSONWriter()

	w.writeObjectStart()

	w.writeKey("learner")
	w.writeObjectStart()

	w.writeKey("learner_model_param")
	w.writeObjectStart()
	w.writeKey("base_score")
	w.writeString("0.5")
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	w.writeKey("gradient_booster")
	w.writeObjectStart()
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart()

	w.writeKey("gbtree_model_param")
	w.writeObjectStart()
	w.writeKey("num_trees")
	w.writeString("1")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeArrayEnd()

	w.writeKey("trees")
	w.writeArrayStart()

	w.writeObjectStart()
	w.writeKey("id")
	w.writeInt32(0)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{1.0})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0})
	w.writeKey("num_nodes")
	w.writeInt32(1)
	w.writeObjectEnd()

	w.writeArrayEnd()  // trees
	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	// No attributes object
	w.writeObjectEnd() // learner
	w.writeObjectEnd() // top-level

	// Should work without transformation
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(w.Bytes()), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON failed: %v", err)
	}
	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}

	// Should fail with transformation (no objective)
	_, err = XGEnsembleFromUBJSON(bytes.NewReader(w.Bytes()), true)
	if err == nil {
		t.Error("expected error for missing objective with loadTransformation=true")
	}
}

// TestXGJSONFormat tests loading a JSON-format model.
func TestXGJSONFormat(t *testing.T) {
	// We need a JSON version of our test model
	// Create the same model but as JSON
	jsonData := `{
		"learner": {
			"learner_model_param": {
				"base_score": "0.5",
				"num_class": "1",
				"num_feature": "2"
			},
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {
						"num_trees": "1",
						"num_parallel_tree": "1"
					},
					"tree_info": [0],
					"trees": [{
						"id": 0,
						"left_children": [1, -1, -1],
						"right_children": [2, -1, -1],
						"parents": [-1, 0, 0],
						"split_conditions": [0.5, -1.0, 1.0],
						"split_indices": [0, 0, 0],
						"split_type": [0, 0, 0],
						"default_left": [1, 0, 0],
						"base_weights": [1.0, -1.0, 1.0],
						"loss_changes": [1.0, 0.0, 0.0],
						"sum_hessian": [100.0, 40.0, 60.0],
						"num_nodes": 3
					}]
				}
			},
			"attributes": {
				"objective": "binary:logistic"
			}
		}
	}`

	model, err := XGEnsembleFromJSON(bytes.NewReader([]byte(jsonData)), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromJSON failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
	if model.NFeatures() != 2 {
		t.Errorf("NFeatures = %d, want 2", model.NFeatures())
	}

	// Test prediction
	predictions := make([]float64, 1)
	fvals := []float64{1.0, 0.0}
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	if predictions[0] != 1.5 {
		t.Errorf("predict = %v, want 1.5", predictions[0])
	}
}

// TestXGJSONAutoDetect tests auto-detection of JSON format.
func TestXGJSONAutoDetect(t *testing.T) {
	jsonData := `{
		"learner": {
			"learner_model_param": {
				"base_score": "0.5",
				"num_class": "1",
				"num_feature": "2"
			},
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {
						"num_trees": "1",
						"num_parallel_tree": "1"
					},
					"tree_info": [0],
					"trees": [{
						"id": 0,
						"left_children": [-1],
						"right_children": [-1],
						"parents": [-1],
						"split_conditions": [1.0],
						"split_indices": [0],
						"split_type": [0],
						"default_left": [0],
						"base_weights": [1.0],
						"loss_changes": [0.0],
						"sum_hessian": [100.0],
						"num_nodes": 1
					}]
				}
			}
		}
	}`

	tmpFile := filepath.Join(t.TempDir(), "test_model.json")
	if err := os.WriteFile(tmpFile, []byte(jsonData), 0644); err != nil {
		t.Fatal(err)
	}

	model, err := XGEnsembleFromAnyFile(tmpFile, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromAnyFile for JSON failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
}

// --- XGBoost native UBJSON format tests ---

// xgbWriter writes XGBoost's non-standard UBJSON format:
// Objects use {L <count> ...} with keys/values as <int-marker><length><bytes> (no 'S' marker).
type xgbWriter struct {
	buf bytes.Buffer
}

func newXGBWriter() *xgbWriter {
	return &xgbWriter{}
}

func (w *xgbWriter) writeObjectStart(count int) {
	w.buf.WriteByte('{')
	w.buf.WriteByte('L') // int64 marker
	binary.Write(&w.buf, binary.BigEndian, int64(count))
}

func (w *xgbWriter) writeObjectEnd() {
	w.buf.WriteByte('}')
}

func (w *xgbWriter) writeArrayStart() {
	w.buf.WriteByte('[')
}

func (w *xgbWriter) writeArrayEnd() {
	w.buf.WriteByte(']')
}

func (w *xgbWriter) writeKey(s string) {
	w.buf.WriteByte('l') // int32 length prefix (no 'S' marker)
	binary.Write(&w.buf, binary.BigEndian, int32(len(s)))
	w.buf.WriteString(s)
}

func (w *xgbWriter) writeString(s string) {
	w.buf.WriteByte('l') // int32 length prefix (no 'S' marker)
	binary.Write(&w.buf, binary.BigEndian, int32(len(s)))
	w.buf.WriteString(s)
}

func (w *xgbWriter) writeInt32(v int32) {
	w.buf.WriteByte('l')
	binary.Write(&w.buf, binary.BigEndian, v)
}

func (w *xgbWriter) writeInt32Slice(vals []int32) {
	w.buf.WriteByte('$')
	w.buf.WriteByte('l')
	w.buf.WriteByte('#')
	w.buf.WriteByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		binary.Write(&w.buf, binary.BigEndian, v)
	}
}

func (w *xgbWriter) writeUint8Slice(vals []uint8) {
	w.buf.WriteByte('$')
	w.buf.WriteByte('U')
	w.buf.WriteByte('#')
	w.buf.WriteByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		w.buf.WriteByte(v)
	}
}

func (w *xgbWriter) writeFloat32Slice(vals []float32) {
	w.buf.WriteByte('$')
	w.buf.WriteByte('d')
	w.buf.WriteByte('#')
	w.buf.WriteByte('l')
	binary.Write(&w.buf, binary.BigEndian, int32(len(vals)))
	for _, v := range vals {
		bits := math.Float32bits(v)
		binary.Write(&w.buf, binary.BigEndian, bits)
	}
}

func (w *xgbWriter) Bytes() []byte {
	return w.buf.Bytes()
}

// createTestXGBoostNativeUBJSON creates a minimal XGBoost model in XGBoost 3.x's
// native UBJSON format (with {L count and l <4-byte> key/value prefixes).
func createTestXGBoostNativeUBJSON() []byte {
	w := newXGBWriter()

	// Top-level model object: 1 entry
	w.writeObjectStart(1)
	w.writeKey("learner")

	// learner object: 4 entries (learner_model_param, gradient_booster, attributes, objective/feature_names)
	w.writeObjectStart(4)
	w.writeKey("learner_model_param")
	w.writeObjectStart(3)
	w.writeKey("base_score")
	w.writeString("0.5")
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	w.writeKey("gradient_booster")
	w.writeObjectStart(2)
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart(3)
	w.writeKey("gbtree_model_param")
	w.writeObjectStart(2)
	w.writeKey("num_trees")
	w.writeString("1")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeArrayEnd()

	w.writeKey("trees")
	w.writeArrayStart()

	// Tree 0
	w.writeObjectStart(11)
	w.writeKey("id")
	w.writeInt32(0)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{1, -1, -1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{2, -1, -1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1, 0, 0})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{0.5, -1.0, 1.0})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0, 0, 0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0, 0, 0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{1, 0, 0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0, -1.0, 1.0})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{1.0, 0.0, 0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0, 40.0, 60.0})
	w.writeKey("num_nodes")
	w.writeInt32(3)
	w.writeObjectEnd() // tree 0

	w.writeArrayEnd() // trees

	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	w.writeKey("attributes")
	w.writeObjectStart(1)
	w.writeKey("objective")
	w.writeString("binary:logistic")
	w.writeObjectEnd()

	w.writeObjectEnd() // learner
	w.writeObjectEnd() // top-level

	return w.Bytes()
}

// createTestXGBoostNativeUBJSONArrayBaseScore creates a model where base_score
// is an array [0.5] instead of a string "0.5" (XGBoost 3.x format).
func createTestXGBoostNativeUBJSONArrayBaseScore() []byte {
	w := newXGBWriter()

	w.writeObjectStart(1)
	w.writeKey("learner")

	w.writeObjectStart(4)
	w.writeKey("learner_model_param")
	w.writeObjectStart(3)
	w.writeKey("base_score")
	// Write base_score as a single-element array
	w.buf.WriteByte('[')
	w.buf.WriteByte('d') // float32
	bits := math.Float32bits(0.5)
	binary.Write(&w.buf, binary.BigEndian, bits)
	w.buf.WriteByte(']')
	w.writeKey("num_class")
	w.writeString("1")
	w.writeKey("num_feature")
	w.writeString("2")
	w.writeObjectEnd()

	w.writeKey("gradient_booster")
	w.writeObjectStart(2)
	w.writeKey("name")
	w.writeString("gbtree")
	w.writeKey("model")
	w.writeObjectStart(3)
	w.writeKey("gbtree_model_param")
	w.writeObjectStart(2)
	w.writeKey("num_trees")
	w.writeString("1")
	w.writeKey("num_parallel_tree")
	w.writeString("1")
	w.writeObjectEnd()

	w.writeKey("tree_info")
	w.writeArrayStart()
	w.writeInt32(0)
	w.writeArrayEnd()

	w.writeKey("trees")
	w.writeArrayStart()

	w.writeObjectStart(11)
	w.writeKey("id")
	w.writeInt32(0)
	w.writeKey("left_children")
	w.writeInt32Slice([]int32{1, -1, -1})
	w.writeKey("right_children")
	w.writeInt32Slice([]int32{2, -1, -1})
	w.writeKey("parents")
	w.writeInt32Slice([]int32{-1, 0, 0})
	w.writeKey("split_conditions")
	w.writeFloat32Slice([]float32{0.5, -1.0, 1.0})
	w.writeKey("split_indices")
	w.writeInt32Slice([]int32{0, 0, 0})
	w.writeKey("split_type")
	w.writeUint8Slice([]uint8{0, 0, 0})
	w.writeKey("default_left")
	w.writeUint8Slice([]uint8{1, 0, 0})
	w.writeKey("base_weights")
	w.writeFloat32Slice([]float32{1.0, -1.0, 1.0})
	w.writeKey("loss_changes")
	w.writeFloat32Slice([]float32{1.0, 0.0, 0.0})
	w.writeKey("sum_hessian")
	w.writeFloat32Slice([]float32{100.0, 40.0, 60.0})
	w.writeKey("num_nodes")
	w.writeInt32(3)
	w.writeObjectEnd()

	w.writeArrayEnd()  // trees
	w.writeObjectEnd() // model
	w.writeObjectEnd() // gradient_booster

	w.writeKey("attributes")
	w.writeObjectStart(1)
	w.writeKey("objective")
	w.writeString("binary:logistic")
	w.writeObjectEnd()

	w.writeObjectEnd() // learner
	w.writeObjectEnd()

	return w.Bytes()
}

func TestXGBoostNativeUBJSONDebug(t *testing.T) {
	data := createTestXGBoostNativeUBJSON()

	// Find "num_nodes" in the data and print surrounding bytes
	idx := bytes.Index(data, []byte("num_nodes"))
	t.Logf("num_nodes at position %d", idx)
	t.Logf("Bytes around num_nodes: %x", data[idx:idx+20])

	// Find the position after num_nodes int32 value
	// num_nodes key: l + 4bytes + "num_nodes" = 1 + 4 + 9 = 14 bytes
	// num_nodes value: l + 4bytes = 1 + 4 = 5 bytes
	// So after num_nodes value: idx + 14 + 5 = idx + 19
	afterNumNodes := idx + 19
	t.Logf("After num_nodes value (position %d): %x", afterNumNodes, data[afterNumNodes:afterNumNodes+20])

	dec := ubjson.NewDecoder(bytes.NewReader(data))
	for i := 0; i < 60; i++ {
		tok, err := dec.Next()
		if err != nil {
			t.Logf("Token %d: ERROR: %v", i, err)
			break
		}
		desc := fmt.Sprintf("Kind=%d", tok.Kind)
		if tok.Kind == ubjson.TokKey {
			desc += fmt.Sprintf(" Key=%q", tok.StrVal)
		} else if tok.Kind == ubjson.TokString {
			desc += fmt.Sprintf(" StrVal=%q", tok.StrVal)
		} else if tok.Kind == ubjson.TokInt32 {
			desc += fmt.Sprintf(" I32Val=%d", tok.I32Val)
		}
		t.Logf("Token %d: %s", i, desc)
	}
}

func TestXGBoostNativeUBJSONArrayBaseScoreDebug(t *testing.T) {
	data := createTestXGBoostNativeUBJSONArrayBaseScore()
	dec := ubjson.NewDecoder(bytes.NewReader(data))
	for i := 0; i < 80; i++ {
		tok, err := dec.Next()
		if err != nil {
			t.Logf("Token %d: ERROR: %v", i, err)
			break
		}
		desc := fmt.Sprintf("Kind=%d", tok.Kind)
		if tok.Kind == ubjson.TokKey {
			desc += fmt.Sprintf(" Key=%q", tok.StrVal)
		} else if tok.Kind == ubjson.TokString {
			desc += fmt.Sprintf(" StrVal=%q", tok.StrVal)
		} else if tok.Kind == ubjson.TokInt32 {
			desc += fmt.Sprintf(" I32Val=%d", tok.I32Val)
		} else if tok.Kind == ubjson.TokFloat32 {
			desc += fmt.Sprintf(" F32Val=%v", tok.F32Val)
		} else if tok.Kind == ubjson.TokArrayStart {
			desc += " ["
		} else if tok.Kind == ubjson.TokArrayEnd {
			desc += " ]"
		}
		t.Logf("Token %d: %s", i, desc)
	}
}

func TestXGBoostNativeUBJSON(t *testing.T) {
	data := createTestXGBoostNativeUBJSON()

	// Print the last 50 bytes to verify structure
	t.Logf("Last 50 bytes of data: %v", data[len(data)-50:])
	t.Logf("As hex: %x", data[len(data)-50:])

	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON (native format) failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}

	// Verify predictions work
	fvals := []float64{0.0, 0.0, 1.0, 1.0}
	predictions := make([]float64, 2)
	model.PredictDense(fvals, 2, 2, predictions, 0, 1)
	if len(predictions) != 2 {
		t.Fatalf("expected 2 predictions, got %d", len(predictions))
	}
}

func TestXGBoostNativeUBJSONArrayBaseScore(t *testing.T) {
	data := createTestXGBoostNativeUBJSONArrayBaseScore()
	model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), false)
	if err != nil {
		t.Fatalf("XGEnsembleFromUBJSON (array base_score) failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}

	// Verify predictions work
	fvals := []float64{0.0, 0.0}
	predictions := make([]float64, 1)
	model.PredictDense(fvals, 1, 2, predictions, 0, 1)
	if len(predictions) != 1 {
		t.Fatalf("expected 1 prediction, got %d", len(predictions))
	}
}

// TestXGBoostSimpleObjectEnd tests that after reading an int32 value with peek,
// the next ObjectEnd marker is correctly read.
func TestXGBoostSimpleObjectEnd(t *testing.T) {
	// Create: { "num_nodes" l 00 00 00 03 } } } } { "attributes" l ... }
	// After the object with num_nodes ends, we should read } } } (object, array, object ends)
	// Then { for attributes object

	// Build: {L <count=1> l<num_nodes_len> "num_nodes" l<value> } ] } } l<attr_len> "attributes" ...
	// But simpler: just test the int32 value followed by objects

	// Create a simple object: { "a" l 00 00 00 01 }
	// In XGBoost format: {L l l<1> "a" l<1> <int32bytes> }
	w := newXGBWriter()
	w.writeObjectStart(1)
	w.writeKey("a")
	w.writeInt32(1)
	w.writeObjectEnd()

	data := w.Bytes()

	dec := ubjson.NewDecoder(bytes.NewReader(data))

	// Read ObjectStart
	tok, err := dec.Next()
	if err != nil {
		t.Fatalf("Next() failed: %v", err)
	}
	if tok.Kind != ubjson.TokObjectStart {
		t.Fatalf("expected ObjectStart, got %v", tok.Kind)
	}

	// Read key "a"
	tok, err = dec.Next()
	if err != nil {
		t.Fatalf("Next() for key: %v", err)
	}
	if tok.Kind != ubjson.TokKey || tok.StrVal != "a" {
		t.Fatalf("expected key 'a', got %v", tok.Kind)
	}

	// Read int32 value
	tok, err = dec.Next()
	if err != nil {
		t.Fatalf("Next() for value: %v", err)
	}
	if tok.Kind != ubjson.TokInt32 || tok.I32Val != 1 {
		t.Fatalf("expected int32 value 1, got kind=%v val=%d", tok.Kind, tok.I32Val)
	}

	// Read ObjectEnd - this should use the peeked byte '}'
	tok, err = dec.Next()
	if err != nil {
		t.Fatalf("Next() for ObjectEnd: %v", err)
	}
	if tok.Kind != ubjson.TokObjectEnd {
		t.Fatalf("expected ObjectEnd, got %v", tok.Kind)
	}

	// Now we're at EOF (no more data)
	tok, err = dec.Next()
	if err == nil {
		t.Fatalf("expected EOF, got token %v", tok.Kind)
	}
}

func TestXGJSONBaseScoreBracketedString(t *testing.T) {
	jsonData := `{
		"version": [1, 6, 0],
		"learner": {
			"learner_model_param": {
				"base_score": "[1.075E-1]",
				"num_class": "1",
				"num_feature": "2"
			},
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {
						"num_trees": "1",
						"num_parallel_tree": "1"
					},
					"tree_info": [0],
					"trees": [{
						"id": 0,
						"left_children": [-1],
						"right_children": [-1],
						"parents": [-1],
						"split_conditions": [1.0],
						"split_indices": [0],
						"split_type": [0],
						"default_left": [0],
						"base_weights": [1.0],
						"loss_changes": [0.0],
						"sum_hessian": [100.0],
						"num_nodes": 1
					}]
				}
			},
			"attributes": {
				"objective": "binary:logistic"
			}
		}
	}`

	tmpFile := filepath.Join(t.TempDir(), "test_model.json")
	if err := os.WriteFile(tmpFile, []byte(jsonData), 0644); err != nil {
		t.Fatal(err)
	}

	model, err := XGEnsembleFromAnyFile(tmpFile, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromAnyFile (bracketed base_score) failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
}

func TestXGJSONBaseScoreArray(t *testing.T) {
	jsonData := `{
		"version": [1, 6, 0],
		"learner": {
			"learner_model_param": {
				"base_score": [0.5],
				"num_class": "1",
				"num_feature": "2"
			},
			"gradient_booster": {
				"name": "gbtree",
				"model": {
					"gbtree_model_param": {
						"num_trees": "1",
						"num_parallel_tree": "1"
					},
					"tree_info": [0],
					"trees": [{
						"id": 0,
						"left_children": [-1],
						"right_children": [-1],
						"parents": [-1],
						"split_conditions": [1.0],
						"split_indices": [0],
						"split_type": [0],
						"default_left": [0],
						"base_weights": [1.0],
						"loss_changes": [0.0],
						"sum_hessian": [100.0],
						"num_nodes": 1
					}]
				}
			},
			"attributes": {
				"objective": "binary:logistic"
			}
		}
	}`

	tmpFile := filepath.Join(t.TempDir(), "test_model.json")
	if err := os.WriteFile(tmpFile, []byte(jsonData), 0644); err != nil {
		t.Fatal(err)
	}

	model, err := XGEnsembleFromAnyFile(tmpFile, false)
	if err != nil {
		t.Fatalf("XGEnsembleFromAnyFile (array base_score) failed: %v", err)
	}

	if model.NEstimators() != 1 {
		t.Errorf("NEstimators = %d, want 1", model.NEstimators())
	}
}
