package leaves

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/citizenadam/go-xgboost-lightgrb/internal/ubjson"
	"github.com/citizenadam/go-xgboost-lightgrb/transformation"
	"github.com/citizenadam/go-xgboost-lightgrb/util"
)

// ubjsonMarkerString is the UBJSON string type marker ('S').
// Used for auto-detection to distinguish UBJSON from JSON format.
const ubjsonMarkerString = 'S'

// ubjsonModel represents the top-level XGBoost UBJSON model structure.
type ubjsonModel struct {
	Learner ubjsonLearner
}

type ubjsonLearner struct {
	ModelParam      ubjsonLearnerModelParam
	GradientBooster ubjsonGradientBooster
	Attributes      map[string]string
	Objective       string // extracted from Attributes
}

type ubjsonLearnerModelParam struct {
	BaseScore  float64
	NumClass   int
	NumFeature int
}

type ubjsonGradientBooster struct {
	Name  string
	Model ubjsonGBTreeModel
}

type ubjsonGBTreeModel struct {
	NumTrees        int
	NumParallelTree int
	TreeInfo        []int32
	Trees           []ubjsonTree
}

type ubjsonTree struct {
	ID              int32
	LeftChildren    []int32
	RightChildren   []int32
	Parents         []int32
	SplitConditions []float32
	SplitIndices    []int32
	SplitType       []uint8
	DefaultLeft     []uint8
	BaseWeights     []float32
	LossChanges     []float32
	SumHessian      []float32
	NumNodes        int
	// Categorical support (v2.x)
	CategoriesSegments []int32
	CategoriesSizes    []int32
	CategoriesNodes    []int32
	Categories         []int32
}

// decodeModel reads an XGBoost UBJSON model from a decoder.
func decodeModel(dec *ubjson.Decoder) (*ubjsonModel, error) {
	model := &ubjsonModel{}
	err := dec.ExpectObjectStart()
	if err != nil {
		return nil, fmt.Errorf("expected model object: %w", err)
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return nil, err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return nil, fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "learner":
			if err := decodeLearner(dec, &model.Learner); err != nil {
				return nil, fmt.Errorf("decoding learner: %w", err)
			}
		default:
			if err := dec.SkipValue(); err != nil {
				return nil, fmt.Errorf("skipping key %q: %w", tok.StrVal, err)
			}
		}
	}
	return model, nil
}

func decodeLearner(dec *ubjson.Decoder, l *ubjsonLearner) error {
	if err := dec.ExpectObjectStart(); err != nil {
		return err
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "learner_model_param":
			if err := decodeLearnerModelParam(dec, &l.ModelParam); err != nil {
				return err
			}
		case "gradient_booster":
			if err := decodeGradientBooster(dec, &l.GradientBooster); err != nil {
				return err
			}
		case "attributes":
			attrs, err := decodeStringMap(dec)
			if err != nil {
				return err
			}
			l.Attributes = attrs
			if obj, ok := attrs["objective"]; ok {
				l.Objective = obj
			}
		case "feature_names":
			if err := dec.SkipValue(); err != nil {
				return err
			}
		case "feature_types":
			if err := dec.SkipValue(); err != nil {
				return err
			}
		default:
			if err := dec.SkipValue(); err != nil {
				return fmt.Errorf("skipping key %q: %w", tok.StrVal, err)
			}
		}
	}
	return nil
}

// readFloatFlexible reads a float64 from UBJSON, handling string, numeric,
// and single-element array formats (XGBoost 3.x wraps values in arrays).
func readFloatFlexible(dec *ubjson.Decoder) (float64, error) {
	tok, err := dec.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case ubjson.TokString:
		return strconv.ParseFloat(strings.Trim(tok.StrVal, "[]"), 64)
	case ubjson.TokFloat32:
		return float64(tok.F32Val), nil
	case ubjson.TokFloat64:
		return tok.F64Val, nil
	case ubjson.TokInt32:
		// In XGBoost mode, int32 after a string key is a string length prefix.
		buf, readErr := dec.ReadBytes(int(tok.I32Val))
		if readErr != nil {
			return 0, readErr
		}
		return strconv.ParseFloat(strings.Trim(string(buf), "[]"), 64)
	case ubjson.TokInt8:
		return float64(tok.I8Val), nil
	case ubjson.TokUint8:
		return float64(tok.U8Val), nil
	case ubjson.TokArrayStart:
		// XGBoost 3.x: [value] — read first element, skip rest
		inner, innerErr := dec.Next()
		if innerErr != nil {
			return 0, innerErr
		}
		var result float64
		switch inner.Kind {
		case ubjson.TokString:
			result, innerErr = strconv.ParseFloat(strings.Trim(inner.StrVal, "[]"), 64)
		case ubjson.TokFloat32:
			result = float64(inner.F32Val)
		case ubjson.TokFloat64:
			result = inner.F64Val
		case ubjson.TokInt32:
			result = float64(inner.I32Val)
		default:
			return 0, fmt.Errorf("unexpected array element type: %v", inner.Kind)
		}
		if innerErr != nil {
			return 0, innerErr
		}
		// Consume remaining elements and array end
		for {
			endTok, endErr := dec.Next()
			if endErr != nil {
				return 0, endErr
			}
			if endTok.Kind == ubjson.TokArrayEnd {
				break
			}
		}
		return result, nil
	}
	return 0, fmt.Errorf("unexpected type for float: %v", tok.Kind)
}

// readIntFlexible reads an int from UBJSON, handling string, numeric,
// and single-element array formats (XGBoost 3.x wraps values in arrays).
func readIntFlexible(dec *ubjson.Decoder) (int, error) {
	tok, err := dec.Next()
	if err != nil {
		return 0, err
	}
	switch tok.Kind {
	case ubjson.TokString:
		return strconv.Atoi(strings.Trim(tok.StrVal, "[]"))
	case ubjson.TokInt32:
		// In XGBoost mode, int32 after a string key is a string length prefix.
		// ReadString() handles this by reading tok.I32Val bytes.
		s, readErr := dec.ReadString()
		if readErr != nil {
			return 0, readErr
		}
		return strconv.Atoi(strings.Trim(s, "[]"))
	case ubjson.TokInt8:
		return int(tok.I8Val), nil
	case ubjson.TokUint8:
		return int(tok.U8Val), nil
	case ubjson.TokInt16:
		return int(tok.I16Val), nil
	case ubjson.TokInt64:
		return int(tok.I64Val), nil
	case ubjson.TokArrayStart:
		// XGBoost 3.x: [value] — read first element, consume end
		inner, innerErr := dec.Next()
		if innerErr != nil {
			return 0, innerErr
		}
		var result int
		switch inner.Kind {
		case ubjson.TokString:
			result, innerErr = strconv.Atoi(strings.Trim(inner.StrVal, "[]"))
		case ubjson.TokInt32:
			result = int(inner.I32Val)
		case ubjson.TokInt8:
			result = int(inner.I8Val)
		case ubjson.TokUint8:
			result = int(inner.U8Val)
		case ubjson.TokInt16:
			result = int(inner.I16Val)
		case ubjson.TokInt64:
			result = int(inner.I64Val)
		default:
			return 0, fmt.Errorf("unexpected array element type: %v", inner.Kind)
		}
		if innerErr != nil {
			return 0, innerErr
		}
		for {
			endTok, endErr := dec.Next()
			if endErr != nil {
				return 0, endErr
			}
			if endTok.Kind == ubjson.TokArrayEnd {
				break
			}
		}
		return result, nil
	}
	return 0, fmt.Errorf("unexpected type for int: %v", tok.Kind)
}

func decodeLearnerModelParam(dec *ubjson.Decoder, p *ubjsonLearnerModelParam) error {
	if err := dec.ExpectObjectStart(); err != nil {
		return err
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "base_score":
			v, err := readFloatFlexible(dec)
			if err != nil {
				return fmt.Errorf("parsing base_score: %w", err)
			}
			p.BaseScore = v
		case "num_class":
			v, err := readIntFlexible(dec)
			if err != nil {
				return fmt.Errorf("parsing num_class: %w", err)
			}
			p.NumClass = v
		case "num_feature":
			v, err := readIntFlexible(dec)
			if err != nil {
				return fmt.Errorf("parsing num_feature: %w", err)
			}
			p.NumFeature = v
		default:
			if err := dec.SkipValue(); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeGradientBooster(dec *ubjson.Decoder, gb *ubjsonGradientBooster) error {
	if err := dec.ExpectObjectStart(); err != nil {
		return err
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "name":
			name, err := dec.ReadString()
			if err != nil {
				return err
			}
			gb.Name = name
		case "model":
			if err := decodeGBTreeModel(dec, &gb.Model); err != nil {
				return err
			}
		default:
			if err := dec.SkipValue(); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeGBTreeModel(dec *ubjson.Decoder, m *ubjsonGBTreeModel) error {
	if err := dec.ExpectObjectStart(); err != nil {
		return err
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "gbtree_model_param":
			if err := decodeGBTreeModelParam(dec, m); err != nil {
				return err
			}
		case "tree_info":
			if err := dec.ExpectArrayStart(); err != nil {
				return err
			}
			for {
				tok2, err := dec.Next()
				if err != nil {
					return err
				}
				if tok2.Kind == ubjson.TokArrayEnd {
					break
				}
				if tok2.Kind == ubjson.TokInt32 {
					m.TreeInfo = append(m.TreeInfo, tok2.I32Val)
				}
			}
		case "trees":
			if err := decodeTrees(dec, m); err != nil {
				return err
			}
		default:
			if err := dec.SkipValue(); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeGBTreeModelParam(dec *ubjson.Decoder, m *ubjsonGBTreeModel) error {
	if err := dec.ExpectObjectStart(); err != nil {
		return err
	}

	for {
		tok, err := dec.Next()
		if err != nil {
			return err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "num_trees":
			s, err := dec.ReadString()
			if err != nil {
				return err
			}
			v, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("parsing num_trees %q: %w", s, err)
			}
			m.NumTrees = v
		case "num_parallel_tree":
			s, err := dec.ReadString()
			if err != nil {
				return err
			}
			v, err := strconv.Atoi(s)
			if err != nil {
				return fmt.Errorf("parsing num_parallel_tree %q: %w", s, err)
			}
			m.NumParallelTree = v
		default:
			if err := dec.SkipValue(); err != nil {
				return err
			}
		}
	}
	return nil
}

func decodeTrees(dec *ubjson.Decoder, m *ubjsonGBTreeModel) error {
	if err := dec.ExpectArrayStart(); err != nil {
		return fmt.Errorf("decodeTrees: expect array start: %w", err)
	}

	treeIdx := 0
	for {
		tok, err := dec.Next()
		if err != nil {
			return fmt.Errorf("decodeTrees: next token: %w", err)
		}
		fmt.Printf("DEBUG decodeTrees[%d]: got token kind=%d\n", treeIdx, int(tok.Kind))
		if tok.Kind == ubjson.TokArrayEnd {
			fmt.Printf("DEBUG decodeTrees: found array end, done\n")
			break
		}
		if tok.Kind != ubjson.TokObjectStart {
			return fmt.Errorf("decodeTrees: expected tree object, got %v (kind=%d)", tok.Kind, int(tok.Kind))
		}

		tree, err := decodeTree(dec)
		if err != nil {
			return err
		}
		fmt.Printf("DEBUG decodeTrees[%d]: decoded tree id=%d, numNodes=%d\n", treeIdx, tree.ID, tree.NumNodes)
		treeIdx++
		m.Trees = append(m.Trees, tree)
	}
	return nil
}

func decodeTree(dec *ubjson.Decoder) (ubjsonTree, error) {
	tree := ubjsonTree{}

	for {
		tok, err := dec.Next()
		if err != nil {
			return tree, fmt.Errorf("decodeTree: next token: %w", err)
		}
		fmt.Printf("DEBUG decodeTree: got token kind=%d\n", int(tok.Kind))
		if tok.Kind == ubjson.TokObjectEnd {
			fmt.Printf("DEBUG decodeTree: found object end, returning\n")
			break
		}
		if tok.Kind != ubjson.TokKey {
			return tree, fmt.Errorf("expected key, got %v", tok.Kind)
		}

		switch tok.StrVal {
		case "id":
			tree.ID, err = dec.ReadInt32()
			if err != nil {
				return tree, err
			}
		case "left_children":
			tree.LeftChildren, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "right_children":
			tree.RightChildren, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "parents":
			tree.Parents, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "split_conditions":
			tree.SplitConditions, err = dec.ReadFloat32Slice()
			if err != nil {
				return tree, err
			}
		case "split_indices":
			tree.SplitIndices, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "split_type":
			tree.SplitType, err = dec.ReadUint8Slice()
			if err != nil {
				return tree, err
			}
		case "default_left":
			tree.DefaultLeft, err = dec.ReadUint8Slice()
			if err != nil {
				return tree, err
			}
		case "base_weights":
			tree.BaseWeights, err = dec.ReadFloat32Slice()
			if err != nil {
				return tree, err
			}
		case "loss_changes":
			tree.LossChanges, err = dec.ReadFloat32Slice()
			if err != nil {
				return tree, err
			}
		case "sum_hessian":
			tree.SumHessian, err = dec.ReadFloat32Slice()
			if err != nil {
				return tree, err
			}
		case "categories_segments":
			tree.CategoriesSegments, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "categories_sizes":
			tree.CategoriesSizes, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "categories_nodes":
			tree.CategoriesNodes, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		case "categories":
			tree.Categories, err = dec.ReadInt32Slice()
			if err != nil {
				return tree, err
			}
		default:
			if err := dec.SkipValue(); err != nil {
				return tree, err
			}
		}
	}

	if len(tree.LeftChildren) > 0 {
		tree.NumNodes = len(tree.LeftChildren)
	}

	return tree, nil
}

func decodeStringMap(dec *ubjson.Decoder) (map[string]string, error) {
	if err := dec.ExpectObjectStart(); err != nil {
		return nil, err
	}

	m := make(map[string]string)
	for {
		tok, err := dec.Next()
		if err != nil {
			return nil, err
		}
		if tok.Kind == ubjson.TokObjectEnd {
			break
		}
		if tok.Kind != ubjson.TokKey {
			return nil, fmt.Errorf("expected key, got %v", tok.Kind)
		}
		key := tok.StrVal
		val, err := dec.ReadString()
		if err != nil {
			return nil, err
		}
		m[key] = val
	}
	return m, nil
}

// xgTreeFromUBJSON converts a UBJSON tree representation to the internal lgTree format.
func xgTreeFromUBJSON(ubjsonTree *ubjsonTree, numFeatures int) (lgTree, error) {
	t := lgTree{}

	if ubjsonTree.NumNodes == 0 {
		return t, fmt.Errorf("tree with zero number of nodes")
	}

	numNodes := ubjsonTree.NumNodes

	// Check for categorical splits
	hasCategorical := false
	if len(ubjsonTree.SplitType) > 0 {
		for _, st := range ubjsonTree.SplitType {
			if st == 1 {
				hasCategorical = true
				break
			}
		}
	}

	// Initialize categorical bitset storage
	if hasCategorical {
		t.catBoundaries = []uint32{0}
		t.catThresholds = []uint32{}
	}

	if numNodes == 1 {
		// Special case: constant value tree (single leaf)
		t.leafValues = append(t.leafValues, float64(ubjsonTree.SplitConditions[0]))
		return t, nil
	}

	// Build categorical category lookup
	catNodeMap := make(map[int32]int32) // node index → categories offset/size index
	if hasCategorical && len(ubjsonTree.CategoriesNodes) > 0 {
		for i, nodeIdx := range ubjsonTree.CategoriesNodes {
			catNodeMap[nodeIdx] = int32(i)
		}
	}

	createNode := func(nodeIdx int32) (lgNode, error) {
		node := lgNode{}
		missingType := uint8(missingNan)

		defaultType := uint8(0)
		if int(nodeIdx) < len(ubjsonTree.DefaultLeft) && ubjsonTree.DefaultLeft[nodeIdx] == 1 {
			defaultType = defaultLeft
		}

		feature := uint32(ubjsonTree.SplitIndices[nodeIdx])
		if int(feature) > numFeatures {
			return node, fmt.Errorf(
				"tree split feature %d exceeds num_feature %d",
				feature, numFeatures,
			)
		}

		isCategorical := false
		if int(nodeIdx) < len(ubjsonTree.SplitType) && ubjsonTree.SplitType[nodeIdx] == 1 {
			isCategorical = true
		}

		if isCategorical {
			// Categorical split
			catIdx := uint32(len(t.catBoundaries) - 1)
			catType := uint8(0)

			if catOff, ok := catNodeMap[nodeIdx]; ok {
				segStart := ubjsonTree.CategoriesSegments[catOff]
				catSize := ubjsonTree.CategoriesSizes[catOff]
				catValues := ubjsonTree.Categories[segStart : segStart+catSize]

				// Build bitset from category values
				bitset := constructCategoricalBitset(catValues)
				nBits := util.NumberOfSetBits(bitset)

				if nBits == 1 {
					// One-hot encoding optimization
					i, err := util.FirstNonZeroBit(bitset)
					if err != nil {
						return node, fmt.Errorf("finding first bit: %w", err)
					}
					catIdx = i
					catType = catOneHot
				} else if len(bitset) == 1 {
					// Small bitset (fits in single uint32)
					catIdx = bitset[0]
					catType = catSmall
				} else {
					// Large bitset
					catIdx = uint32(len(t.catBoundaries) - 1)
					t.catThresholds = append(t.catThresholds, bitset...)
					t.catBoundaries = append(t.catBoundaries, uint32(len(t.catThresholds)))
				}
				t.nCategorical++
			}

			node = categoricalNode(feature, missingType, catIdx, catType)
		} else {
			// Numerical split
			threshold := float64(ubjsonTree.SplitConditions[nodeIdx])
			node = numericalNode(feature, missingType, threshold, defaultType)
		}

		// Note: child processing (leaf detection, leaf value extraction) is handled
		// by the DFS traversal, NOT here. This function only sets up the node's own
		// split information (feature, threshold, categorical params).

		return node, nil
	}

	// DFS stack traversal (same as xgTreeFromTreeModel)
	origNodeIdxStack := make([]int32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, numNodes)
	t.nodes = make([]lgNode, 0, numNodes)

	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	visited[0] = true

	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		origIdx := origNodeIdxStack[len(origNodeIdxStack)-1]

		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			rightChild := ubjsonTree.RightChildren[origIdx]
			if rightChild < 0 {
				// Right child is a leaf in the original tree
				t.nodes[convIdx].Flags |= rightLeaf
				leafIdx := -rightChild
				t.nodes[convIdx].Right = uint32(len(t.leafValues))
				t.leafValues = append(t.leafValues, float64(ubjsonTree.SplitConditions[leafIdx]))
			} else if !visited[rightChild] {
				// Check if the child node is itself a leaf (both its children are -1)
				if ubjsonTree.LeftChildren[rightChild] < 0 {
					// Child is a leaf node
					t.nodes[convIdx].Flags |= rightLeaf
					t.nodes[convIdx].Right = uint32(len(t.leafValues))
					t.leafValues = append(t.leafValues, float64(ubjsonTree.SplitConditions[rightChild]))
					visited[rightChild] = true
				} else {
					// Child is an internal node
					node, err := createNode(rightChild)
					if err != nil {
						return t, err
					}
					t.nodes = append(t.nodes, node)
					convNewIdx := len(t.nodes) - 1
					convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
					origNodeIdxStack = append(origNodeIdxStack, rightChild)
					visited[rightChild] = true
					t.nodes[convIdx].Right = uint32(convNewIdx)
					continue
				}
			}
		}

		if t.nodes[convIdx].Flags&leftLeaf == 0 {
			leftChild := ubjsonTree.LeftChildren[origIdx]
			if leftChild < 0 {
				// Left child is a leaf in the original tree
				t.nodes[convIdx].Flags |= leftLeaf
				leafIdx := -leftChild
				t.nodes[convIdx].Left = uint32(len(t.leafValues))
				t.leafValues = append(t.leafValues, float64(ubjsonTree.SplitConditions[leafIdx]))
			} else if !visited[leftChild] {
				// Check if the child node is itself a leaf (both its children are -1)
				if ubjsonTree.LeftChildren[leftChild] < 0 {
					// Child is a leaf node
					t.nodes[convIdx].Flags |= leftLeaf
					t.nodes[convIdx].Left = uint32(len(t.leafValues))
					t.leafValues = append(t.leafValues, float64(ubjsonTree.SplitConditions[leftChild]))
					visited[leftChild] = true
				} else {
					// Child is an internal node
					node, err := createNode(leftChild)
					if err != nil {
						return t, err
					}
					t.nodes = append(t.nodes, node)
					convNewIdx := len(t.nodes) - 1
					convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
					origNodeIdxStack = append(origNodeIdxStack, leftChild)
					visited[leftChild] = true
					t.nodes[convIdx].Left = uint32(convNewIdx)
					continue
				}
			}
		}

		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}

	return t, nil
}

// constructCategoricalBitset creates a bitset from a list of categorical values.
func constructCategoricalBitset(values []int32) []uint32 {
	if len(values) == 0 {
		return nil
	}
	intValues := make([]int, len(values))
	for i, v := range values {
		intValues[i] = int(v)
	}
	return util.ConstructBitset(intValues)
}

// xgObjectiveToTransform maps XGBoost objective strings to transformation types.
func xgObjectiveToTransform(objective string, nOutputGroups int) (transformation.Transform, error) {
	switch objective {
	case "binary:logistic":
		return &transformation.TransformLogistic{}, nil
	case "reg:logistic":
		return &transformation.TransformLogistic{}, nil
	case "multi:softprob":
		return &transformation.TransformSoftmax{NClasses: nOutputGroups}, nil
	case "reg:squarederror", "reg:linear":
		return &transformation.TransformRaw{NumOutputGroups: nOutputGroups}, nil
	case "survival:cox":
		return &transformation.TransformExponential{}, nil
	default:
		return nil, fmt.Errorf("unknown objective function '%s'", objective)
	}
}

// XGEnsembleFromUBJSON reads an XGBoost model from a UBJSON stream.
// Works with models saved by XGBoost v1.6+ (gbtree, dart).
func XGEnsembleFromUBJSON(reader io.Reader, loadTransformation bool) (*Ensemble, error) {
	dec := ubjson.NewDecoder(reader)
	model, err := decodeModel(dec)
	if err != nil {
		return nil, fmt.Errorf("decoding UBJSON model: %w", err)
	}
	return ubjsonModelToEnsemble(model, loadTransformation)
}

// XGEnsembleFromUBJSONFile reads an XGBoost model from a UBJSON file.
func XGEnsembleFromUBJSONFile(filename string, loadTransformation bool) (*Ensemble, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return XGEnsembleFromUBJSON(f, loadTransformation)
}

// XGEnsembleFromJSON reads an XGBoost model from a JSON stream.
// This supports the JSON dump format exported by XGBoost's save_model().
func XGEnsembleFromJSON(reader io.Reader, loadTransformation bool) (*Ensemble, error) {
	// JSON models use the same structure as UBJSON but decoded via encoding/json.
	// We use a generic map-based decoder that mirrors the UBJSON decoder.
	dec := json.NewDecoder(reader)
	model, err := decodeJSONModel(dec)
	if err != nil {
		return nil, fmt.Errorf("decoding JSON model: %w", err)
	}
	return ubjsonModelToEnsemble(model, loadTransformation)
}

// XGEnsembleFromJSONFile reads an XGBoost model from a JSON file.
func XGEnsembleFromJSONFile(filename string, loadTransformation bool) (*Ensemble, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return XGEnsembleFromJSON(f, loadTransformation)
}

// XGEnsembleFromAnyFile reads an XGBoost model from file, auto-detecting
// whether it is legacy binary, JSON, or UBJSON format.
func XGEnsembleFromAnyFile(filename string, loadTransformation bool) (*Ensemble, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read entire file into memory for format detection and decoding
	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("reading model data: %w", err)
	}

	if len(data) == 0 {
		return nil, fmt.Errorf("empty model file")
	}

	if data[0] == '{' {
		// JSON or UBJSON format — try UBJSON first, fall back to JSON
		model, err := XGEnsembleFromUBJSON(bytes.NewReader(data), loadTransformation)
		if err == nil {
			return model, nil
		}
		return XGEnsembleFromJSON(bytes.NewReader(data), loadTransformation)
	}

	// Legacy binary format
	return XGEnsembleFromReader(bufio.NewReader(bytes.NewReader(data)), loadTransformation)
}

// ubjsonModelToEnsemble converts a parsed UBJSON model to a leaves Ensemble.
func ubjsonModelToEnsemble(model *ubjsonModel, loadTransformation bool) (*Ensemble, error) {
	e := &xgEnsemble{}

	gb := &model.Learner.GradientBooster
	switch gb.Name {
	case "gbtree":
		e.name = "xgboost.gbtree"
	case "dart":
		e.name = "xgboost.dart"
	default:
		return nil, fmt.Errorf("only 'gbtree' or 'dart' is supported (got %s)", gb.Name)
	}

	numFeature := model.Learner.ModelParam.NumFeature
	if numFeature == 0 {
		return nil, fmt.Errorf("zero number of features")
	}
	e.MaxFeatureIdx = numFeature - 1
	e.BaseScore = model.Learner.ModelParam.BaseScore

	gbModel := &gb.Model
	numTrees := gbModel.NumTrees
	if numTrees == 0 {
		return nil, fmt.Errorf("no trees in model")
	}

	nRawOutputGroups := 1
	if model.Learner.ModelParam.NumClass > 0 {
		nRawOutputGroups = model.Learner.ModelParam.NumClass
	}
	e.nRawOutputGroups = nRawOutputGroups

	// Set WeightDrop
	e.WeightDrop = make([]float64, numTrees)
	if gb.Name == "dart" {
		// DART models have weight_drop in the gradient_booster
		// For now, default to 1.0 (DART with UBJSON needs more investigation)
		for i := 0; i < numTrees; i++ {
			e.WeightDrop[i] = 1.0
		}
	} else {
		for i := 0; i < numTrees; i++ {
			e.WeightDrop[i] = 1.0
		}
	}

	// Set transformation
	var transform transformation.Transform
	transform = &transformation.TransformRaw{NumOutputGroups: e.nRawOutputGroups}
	if loadTransformation {
		var err error
		transform, err = xgObjectiveToTransform(model.Learner.Objective, e.nRawOutputGroups)
		if err != nil {
			return nil, err
		}
	}

	// Convert trees
	e.Trees = make([]lgTree, 0, numTrees)
	for i := 0; i < numTrees; i++ {
		tree, err := xgTreeFromUBJSON(&gbModel.Trees[i], numFeature)
		if err != nil {
			return nil, fmt.Errorf("error converting tree %d: %w", i, err)
		}
		e.Trees = append(e.Trees, tree)
	}

	return &Ensemble{e, transform}, nil
}

// --- JSON model decoder (mirrors UBJSON decoder but uses encoding/json) ---

// jsonModel, jsonLearner, etc. are intermediate representations for JSON decoding
// that get converted to the same ubjson* structs.

func decodeJSONModel(dec *json.Decoder) (*ubjsonModel, error) {
	var raw map[string]interface{}
	if err := dec.Decode(&raw); err != nil {
		return nil, err
	}
	return jsonModelFromMap(raw)
}

func jsonModelFromMap(raw map[string]interface{}) (*ubjsonModel, error) {
	model := &ubjsonModel{}
	learnerRaw, ok := raw["learner"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("missing 'learner' field")
	}

	if err := jsonLearnerFromMap(learnerRaw, &model.Learner); err != nil {
		return nil, err
	}
	return model, nil
}

func jsonLearnerFromMap(raw map[string]interface{}, l *ubjsonLearner) error {
	mpRaw, ok := raw["learner_model_param"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("missing 'learner_model_param'")
	}
	if err := jsonLearnerModelParamFromMap(mpRaw, &l.ModelParam); err != nil {
		return err
	}

	gbRaw, ok := raw["gradient_booster"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("missing 'gradient_booster'")
	}
	if err := jsonGradientBoosterFromMap(gbRaw, &l.GradientBooster); err != nil {
		return err
	}

	if attrsRaw, ok := raw["attributes"].(map[string]interface{}); ok {
		l.Attributes = make(map[string]string)
		for k, v := range attrsRaw {
			if s, ok := v.(string); ok {
				l.Attributes[k] = s
			}
		}
		if obj, ok := l.Attributes["objective"]; ok {
			l.Objective = obj
		}
	}

	return nil
}

func jsonLearnerModelParamFromMap(raw map[string]interface{}, p *ubjsonLearnerModelParam) error {
	p.BaseScore = jsonParseFloatFlexible(raw["base_score"])
	if nc, ok := raw["num_class"]; ok {
		p.NumClass = jsonParseIntFlexible(nc)
	}
	if nf, ok := raw["num_feature"]; ok {
		p.NumFeature = jsonParseIntFlexible(nf)
	}
	return nil
}

func jsonParseFloatFlexible(v interface{}) float64 {
	switch val := v.(type) {
	case string:
		s := strings.Trim(val, "[]")
		if f, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil {
			return f
		}
	case float64:
		return val
	case []interface{}:
		if len(val) > 0 {
			return jsonParseFloatFlexible(val[0])
		}
	}
	return 0
}

func jsonParseIntFlexible(v interface{}) int {
	switch val := v.(type) {
	case string:
		s := strings.Trim(val, "[]")
		if i, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			return i
		}
	case float64:
		return int(val)
	case []interface{}:
		if len(val) > 0 {
			return jsonParseIntFlexible(val[0])
		}
	}
	return 0
}

func jsonGradientBoosterFromMap(raw map[string]interface{}, gb *ubjsonGradientBooster) error {
	if name, ok := raw["name"].(string); ok {
		gb.Name = name
	}

	modelRaw, ok := raw["model"].(map[string]interface{})
	if !ok {
		return fmt.Errorf("missing 'model'")
	}
	return jsonGBTreeModelFromMap(modelRaw, &gb.Model)
}

func jsonGBTreeModelFromMap(raw map[string]interface{}, m *ubjsonGBTreeModel) error {
	if paramRaw, ok := raw["gbtree_model_param"].(map[string]interface{}); ok {
		if nt, ok := paramRaw["num_trees"].(string); ok {
			v, err := strconv.Atoi(nt)
			if err != nil {
				return fmt.Errorf("parsing num_trees: %w", err)
			}
			m.NumTrees = v
		}
		if npt, ok := paramRaw["num_parallel_tree"].(string); ok {
			v, err := strconv.Atoi(npt)
			if err != nil {
				return fmt.Errorf("parsing num_parallel_tree: %w", err)
			}
			m.NumParallelTree = v
		}
	}

	if tiRaw, ok := raw["tree_info"].([]interface{}); ok {
		m.TreeInfo = make([]int32, len(tiRaw))
		for i, v := range tiRaw {
			if f, ok := v.(float64); ok {
				m.TreeInfo[i] = int32(f)
			}
		}
	}

	if treesRaw, ok := raw["trees"].([]interface{}); ok {
		m.Trees = make([]ubjsonTree, 0, len(treesRaw))
		for _, tv := range treesRaw {
			treeMap, ok := tv.(map[string]interface{})
			if !ok {
				return fmt.Errorf("invalid tree object")
			}
			tree, err := jsonTreeFromMap(treeMap)
			if err != nil {
				return err
			}
			m.Trees = append(m.Trees, tree)
		}
	}

	return nil
}

func jsonTreeFromMap(raw map[string]interface{}) (ubjsonTree, error) {
	tree := ubjsonTree{}

	if id, ok := raw["id"].(float64); ok {
		tree.ID = int32(id)
	}

	tree.LeftChildren = jsonInt32Slice(raw["left_children"])
	tree.RightChildren = jsonInt32Slice(raw["right_children"])
	tree.Parents = jsonInt32Slice(raw["parents"])
	tree.SplitConditions = jsonFloat32Slice(raw["split_conditions"])
	tree.SplitIndices = jsonInt32Slice(raw["split_indices"])
	tree.SplitType = jsonUint8Slice(raw["split_type"])
	tree.DefaultLeft = jsonUint8Slice(raw["default_left"])
	tree.BaseWeights = jsonFloat32Slice(raw["base_weights"])
	tree.LossChanges = jsonFloat32Slice(raw["loss_changes"])
	tree.SumHessian = jsonFloat32Slice(raw["sum_hessian"])

	// Categorical support
	tree.CategoriesSegments = jsonInt32Slice(raw["categories_segments"])
	tree.CategoriesSizes = jsonInt32Slice(raw["categories_sizes"])
	tree.CategoriesNodes = jsonInt32Slice(raw["categories_nodes"])
	tree.Categories = jsonInt32Slice(raw["categories"])

	if len(tree.LeftChildren) > 0 {
		tree.NumNodes = len(tree.LeftChildren)
	}

	return tree, nil
}

func jsonInt32Slice(v interface{}) []int32 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	result := make([]int32, len(arr))
	for i, val := range arr {
		if f, ok := val.(float64); ok {
			result[i] = int32(f)
		}
	}
	return result
}

func jsonFloat32Slice(v interface{}) []float32 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	result := make([]float32, len(arr))
	for i, val := range arr {
		if f, ok := val.(float64); ok {
			result[i] = float32(f)
		}
	}
	return result
}

func jsonUint8Slice(v interface{}) []uint8 {
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	result := make([]uint8, len(arr))
	for i, val := range arr {
		if f, ok := val.(float64); ok {
			result[i] = uint8(f)
		}
	}
	return result
}
