package pickle

import (
	"encoding/binary"
	"fmt"

	"github.com/citizenadam/go-xgboost-lightgrb/util"
)

// SklearnNode represents tree node data structure
type SklearnNode struct {
	LeftChild            int
	RightChild           int
	Feature              int
	Threshold            float64
	Impurity             float64
	NNodeSamples         int
	WeightedNNodeSamples float64
}

// SklearnNodeFromBytes converts 56 raw bytes into SklearnNode struct
// The rule described in https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pyx#L70 (NODE_DTYPE)
// 'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples'],
// 'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp, np.float64]
func SklearnNodeFromBytes(bytes []byte) SklearnNode {
	offset := 0
	size := 8
	node := SklearnNode{}
	node.LeftChild = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.RightChild = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.Feature = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.Threshold = util.Float64FromBytes(bytes[offset:offset+size], true)
	offset += size
	node.Impurity = util.Float64FromBytes(bytes[offset:offset+size], true)
	offset += size
	node.NNodeSamples = int(binary.LittleEndian.Uint64(bytes[offset : offset+size]))
	offset += size
	node.WeightedNNodeSamples = util.Float64FromBytes(bytes[offset:offset+size], true)
	return node
}

// SklearnTree represents parsed tree struct
type SklearnTree struct {
	NOutputs int
	Classes  []int
	NNodes   int
	Nodes    []SklearnNode
	Values   []float64
}

func (t *SklearnTree) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "sklearn.tree._tree", "Tree")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("expected len(reduce.Args) = 3 (got %d)", len(reduce.Args))
	}

	var ok bool
	t.NOutputs, ok = reduce.Args[2].(int)
	if !ok {
		return fmt.Errorf("expected int (got %T)", reduce.Args[2])
	}

	arr := NumpyArrayRaw{}
	err = ParseClass(&arr, reduce.Args[1])
	if err != nil {
		return err
	}

	if len(arr.Shape) != 1 && arr.Shape[0] != t.NOutputs {
		return fmt.Errorf("expected 1 dim array with %d values (got: %v)", t.NOutputs, arr.Shape)
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, t.NOutputs)
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}
	return
}

func (t *SklearnTree) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	t.NNodes, err = dict.toInt("node_count")
	if err != nil {
		return
	}
	nodesObj, err := dict.value("nodes")
	if err != nil {
		return
	}
	arr := NumpyArrayRaw{}
	err = ParseClass(&arr, nodesObj)
	if err != nil {
		return
	}
	if arr.Type.Type != "V56" {
		return fmt.Errorf("expected arr.Type.Type = \"V56\" (got: %s)", arr.Type.Type)
	}
	err = arr.Data.Iterate(56, func(b []byte) error {
		t.Nodes = append(t.Nodes, SklearnNodeFromBytes(b))
		return nil
	})
	if err != nil {
		return
	}
	valuesObj, err := dict.value("values")
	if err != nil {
		return
	}
	arr = NumpyArrayRaw{}
	err = ParseClass(&arr, valuesObj)
	if err != nil {
		return
	}
	if arr.Type.Type != "f8" {
		return fmt.Errorf("expected ndtype \"f8\" (got: %#v)", arr.Type)
	}
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Values = append(t.Values, util.Float64FromBytes(b, arr.Type.LittleEndinan))
		return nil
	})
	if err != nil {
		return
	}
	return
}

type SklearnDecisionTreeRegressor struct {
	Tree        SklearnTree
	NClasses    int
	MaxFeatures int
	NOutputs    int
}

func (t *SklearnDecisionTreeRegressor) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.tree.tree", "DecisionTreeRegressor")
	if err != nil {
		return
	}
	return
}

func (t *SklearnDecisionTreeRegressor) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	nClassesRaw := NumpyScalarRaw{}
	nClassesObj, err := dict.value("n_classes_")
	if err != nil {
		return
	}
	err = ParseClass(&nClassesRaw, nClassesObj)
	if err != nil {
		return
	}
	if nClassesRaw.Type.Type != "i8" || !nClassesRaw.Type.LittleEndinan {
		return fmt.Errorf("expected little endian i8, got (%#v)", nClassesRaw.Type)
	}
	t.NClasses = int(binary.LittleEndian.Uint64(nClassesRaw.Data))
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NOutputs, err = dict.toInt("n_outputs_")
	if err != nil {
		return
	}

	treeObj, err := dict.value("tree_")
	if err != nil {
		return
	}
	err = ParseClass(&t.Tree, treeObj)
	if err != nil {
		return
	}
	return
}

type SklearnGradientBoosting struct {
	NClasses      int
	Classes       []int
	NEstimators   int
	MaxFeatures   int
	Estimators    []SklearnDecisionTreeRegressor
	LearningRate  float64
	InitEstimator SKlearnInitEstimator
}

func (t *SklearnGradientBoosting) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.ensemble.gradient_boosting", "GradientBoostingClassifier")
	if err != nil {
		return
	}
	return
}

func (t *SklearnGradientBoosting) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}
	t.NClasses, err = dict.toInt("n_classes_")
	if err != nil {
		return
	}

	t.LearningRate, err = dict.toFloat("learning_rate")
	if err != nil {
		return
	}

	obj, err := dict.value("loss")
	if err != nil {
		return
	}
	_ /* loss */, err = toUnicode(obj, -1)
	if err != nil {
		return
	}

	obj, err = dict.value("init_")
	if err != nil {
		return
	}
	err = ParseClass(&t.InitEstimator, obj)
	if err != nil {
		return
	}

	arr := NumpyArrayRaw{}
	classesObj, err := dict.value("classes_")
	if err != nil {
		return err
	}
	err = ParseClass(&arr, classesObj)
	if err != nil {
		return err
	}

	if len(arr.Shape) != 1 && arr.Shape[0] != t.NClasses {
		return fmt.Errorf("expected 1 dim array with %d values (got: %v)", t.NClasses, arr.Shape)
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, t.NClasses)
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NEstimators, err = dict.toInt("n_estimators")
	if err != nil {
		return
	}
	//  estimators_ : ndarray of DecisionTreeRegressor,\
	//  shape (n_estimators, ``loss_.K``)
	//  		The collection of fitted sub-estimators. ``loss_.K`` is 1 for binary
	//  		classification, otherwise n_classes.
	arr = NumpyArrayRaw{}
	obj, err = dict.value("estimators_")
	if err != nil {
		return
	}
	err = ParseClass(&arr, obj)
	if err != nil {
		return
	}
	adjNClasses := t.NClasses
	if t.NClasses == 2 {
		adjNClasses = 1
	}
	if len(arr.Shape) != 2 || arr.Shape[0] != t.NEstimators || arr.Shape[1] != adjNClasses {
		return fmt.Errorf("unexpected shape: %#v", arr.Shape)
	}
	if len(arr.DataList) != arr.Shape[0]*arr.Shape[1] {
		return fmt.Errorf("unexpected array list length")
	}
	t.Estimators = make([]SklearnDecisionTreeRegressor, arr.Shape[0]*arr.Shape[1])
	for i := range arr.DataList {
		err = ParseClass(&t.Estimators[i], arr.DataList[i])
		if err != nil {
			return
		}
	}
	return
}

// SklearnRandomForestClassifier represents sklearn's RandomForestClassifier
type SklearnRandomForestClassifier struct {
	NEstimators int
	MaxFeatures int
	NClasses    int
	Classes     []int
	Estimators  []SklearnDecisionTreeRegressor
	// For binary classification, n_outputs_ is 1
	NOutputs int
}

// Reduce handles the pickle REDUCE opcode for RandomForestClassifier
func (t *SklearnRandomForestClassifier) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.ensemble.forest", "RandomForestClassifier")
	if err != nil {
		return
	}
	return
}

// Build handles the pickle BUILD opcode for RandomForestClassifier
func (t *SklearnRandomForestClassifier) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}

	t.NEstimators, err = dict.toInt("n_estimators")
	if err != nil {
		return
	}
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NClasses, err = dict.toInt("n_classes_")
	if err != nil {
		return
	}
	t.NOutputs, err = dict.toInt("n_outputs_")
	if err != nil {
		return
	}

	// classes_: ndarray of shape (n_classes_,) or (n_classes_, n_outputs_)
	arr := NumpyArrayRaw{}
	classesObj, err := dict.value("classes_")
	if err != nil {
		return err
	}
	err = ParseClass(&arr, classesObj)
	if err != nil {
		return err
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, arr.Shape[0]*arr.Shape[1])
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}

	// estimators_: ndarray of shape (n_estimators, n_outputs_) or (n_estimators,)
	arr = NumpyArrayRaw{}
	obj, err := dict.value("estimators_")
	if err != nil {
		return
	}
	err = ParseClass(&arr, obj)
	if err != nil {
		return
	}

	// Handle different shapes: (n_estimators,) for single output, (n_estimators, n_outputs_) for multi-output
	totalTrees := arr.Shape[0]
	if len(arr.Shape) > 1 {
		totalTrees = arr.Shape[0] * arr.Shape[1]
	}

	t.Estimators = make([]SklearnDecisionTreeRegressor, 0, totalTrees)
	for i := range arr.DataList {
		t.Estimators = append(t.Estimators, SklearnDecisionTreeRegressor{})
		err = ParseClass(&t.Estimators[i], arr.DataList[i])
		if err != nil {
			return err
		}
	}

	return
}

// SklearnExtraTreesClassifier represents sklearn's ExtraTreesClassifier
// Same structure as RandomForestClassifier
type SklearnExtraTreesClassifier struct {
	NEstimators int
	MaxFeatures int
	NClasses    int
	Classes     []int
	Estimators  []SklearnDecisionTreeRegressor
	NOutputs    int
}

// Reduce handles the pickle REDUCE opcode for ExtraTreesClassifier
func (t *SklearnExtraTreesClassifier) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.ensemble.forest", "ExtraTreesClassifier")
	if err != nil {
		return
	}
	return
}

// Build handles the pickle BUILD opcode for ExtraTreesClassifier
// Identical to RandomForestClassifier
func (t *SklearnExtraTreesClassifier) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}

	t.NEstimators, err = dict.toInt("n_estimators")
	if err != nil {
		return
	}
	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}
	t.NClasses, err = dict.toInt("n_classes_")
	if err != nil {
		return
	}
	t.NOutputs, err = dict.toInt("n_outputs_")
	if err != nil {
		return
	}

	arr := NumpyArrayRaw{}
	classesObj, err := dict.value("classes_")
	if err != nil {
		return err
	}
	err = ParseClass(&arr, classesObj)
	if err != nil {
		return err
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, arr.Shape[0]*arr.Shape[1])
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}

	arr = NumpyArrayRaw{}
	obj, err := dict.value("estimators_")
	if err != nil {
		return
	}
	err = ParseClass(&arr, obj)
	if err != nil {
		return
	}

	totalTrees := arr.Shape[0]
	if len(arr.Shape) > 1 {
		totalTrees = arr.Shape[0] * arr.Shape[1]
	}

	t.Estimators = make([]SklearnDecisionTreeRegressor, 0, totalTrees)
	for i := range arr.DataList {
		t.Estimators = append(t.Estimators, SklearnDecisionTreeRegressor{})
		err = ParseClass(&t.Estimators[i], arr.DataList[i])
		if err != nil {
			return err
		}
	}

	return
}

// SklearnHistGradientBoostingClassifier represents sklearn's HistGradientBoostingClassifier
// Note: This uses a different internal structure than regular GradientBoostingClassifier
type SklearnHistGradientBoostingClassifier struct {
	NClasses         int
	Classes          []int
	NEstimators      int
	MaxFeatures      int
	NBins            int
	LearningRate     float64
	BinData          []byte // Compiled histogram data
	MissingThreshold float64
	// For regression, these fields are used instead
	IsClassifier bool
}

// Reduce handles the pickle REDUCE opcode for HistGradientBoostingClassifier
func (t *SklearnHistGradientBoostingClassifier) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	_, err = toGlobal(reduce.Args[0], "sklearn.ensemble._hist_gradient_boosting", "HistGradientBoostingClassifier")
	if err != nil {
		return
	}
	return
}

// Build handles the pickle BUILD opcode for HistGradientBoostingClassifier
func (t *SklearnHistGradientBoostingClassifier) Build(build Build) (err error) {
	dict, err := toDict(build.Args)
	if err != nil {
		return
	}

	t.NClasses, err = dict.toInt("n_classes_")
	if err != nil {
		return
	}

	t.LearningRate, err = dict.toFloat("learning_rate")
	if err != nil {
		return
	}

	t.NEstimators, err = dict.toInt("n_estimators")
	if err != nil {
		return
	}

	t.MaxFeatures, err = dict.toInt("max_features_")
	if err != nil {
		return
	}

	t.NBins, err = dict.toInt("n_bins_")
	if err != nil {
		return
	}

	// classes_: ndarray of shape (n_classes_,)
	arr := NumpyArrayRaw{}
	classesObj, err := dict.value("classes_")
	if err != nil {
		return err
	}
	err = ParseClass(&arr, classesObj)
	if err != nil {
		return err
	}
	if arr.Type.Type != "i8" || arr.Type.LittleEndinan != true {
		return fmt.Errorf("expected ndtype \"i8\" little endian (got: %#v)", arr.Type)
	}

	t.Classes = make([]int, 0, arr.Shape[0])
	err = arr.Data.Iterate(8, func(b []byte) error {
		t.Classes = append(t.Classes, int(binary.LittleEndian.Uint64(b)))
		return nil
	})
	if err != nil {
		return
	}

	// bin_data_: optional compiled histogram data
	obj, err := dict.value("bin_data_")
	if err != nil {
		// May not exist in all versions
		t.IsClassifier = true
		return nil
	}

	if arr, ok := obj.(List); ok && len(arr) >= 2 {
		t.IsClassifier = true
	}

	return
}

type SKlearnInitEstimator struct {
	Name  string
	Prior []float64
}

func (e *SKlearnInitEstimator) Reduce(reduce Reduce) (err error) {
	_, err = toGlobal(reduce.Callable, "copy_reg", "_reconstructor")
	if err != nil {
		return
	}
	if len(reduce.Args) != 3 {
		return fmt.Errorf("not expected tuple: %#v", reduce.Args)
	}
	classDesc, err := toGlobal(reduce.Args[0], "sklearn.ensemble.gradient_boosting", "")
	if err != nil {
		return
	}
	e.Name = classDesc.Name
	return
}

func (e *SKlearnInitEstimator) Build(build Build) (err error) {
	if e.Name == "LogOddsEstimator" {
		dict, err := toDict(build.Args)
		if err != nil {
			return err
		}
		priorObj, err := dict.value("prior")
		if err != nil {
			return err
		}
		numpyScalar := NumpyScalarRaw{}
		err = ParseClass(&numpyScalar, priorObj)
		if err != nil {
			return err
		}
		if numpyScalar.Type.Type != "f8" {
			return fmt.Errorf("expected f8, got (%#v)", numpyScalar.Type)
		}
		e.Prior = append(e.Prior, util.Float64FromBytes(numpyScalar.Data, numpyScalar.Type.LittleEndinan))
	} else if e.Name == "PriorProbabilityEstimator" {
		dict, err := toDict(build.Args)
		if err != nil {
			return err
		}
		priorObj, err := dict.value("priors")
		if err != nil {
			return err
		}
		numpyArray := NumpyArrayRaw{}
		err = ParseClass(&numpyArray, priorObj)
		if err != nil {
			return err
		}
		if numpyArray.Type.Type != "f8" {
			return fmt.Errorf("expected f8, got (%#v)", numpyArray.Type)
		}
		numpyArray.Data.Iterate(8, func(bytes []byte) error {
			e.Prior = append(e.Prior, util.Float64FromBytes(bytes, numpyArray.Type.LittleEndinan))
			return nil
		})
	} else {
		return fmt.Errorf("unknown init estimator class: %s", e.Name)
	}
	return
}

// SklearnLogisticRegression represents sklearn.linear_model.LogisticRegression
// Used as calibrator in CalibratedClassifierCV
type SklearnLogisticRegression struct {
	Coefficients [][]float64 // shape [n_classes, n_features] or [1, 1] for binary
	Intercepts   []float64   // shape [n_classes]
}

// Reduce implements PythonClass interface
func (t *SklearnLogisticRegression) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "sklearn.linear_model._logistic", "LogisticRegression")
	if err != nil {
		return err
	}
	return nil
}

// Build implements PythonClass interface
func (t *SklearnLogisticRegression) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}

	// Extract coef_ (coefficients)
	coefObj, err := dict.value("coef_")
	if err != nil {
		return fmt.Errorf("failed to get coef_: %w", err)
	}
	coefArray := NumpyArrayRaw{}
	if err := ParseClass(&coefArray, coefObj); err != nil {
		return fmt.Errorf("failed to parse coef_: %w", err)
	}

	// coef_ shape is [n_classes, n_features] or [1, 1] for binary
	if len(coefArray.Shape) != 2 {
		return fmt.Errorf("expected coef_ shape to be 2D, got %v", coefArray.Shape)
	}

	// Extract coefficients row by row
	t.Coefficients = make([][]float64, 0, coefArray.Shape[0])
	rowSize := coefArray.Shape[1]

	err = coefArray.Data.Iterate(8, func(b []byte) error {
		row := make([]float64, rowSize)
		for i := 0; i < rowSize; i++ {
			row[i] = util.Float64FromBytes(b[i*8:(i+1)*8], coefArray.Type.LittleEndinan)
		}
		t.Coefficients = append(t.Coefficients, row)
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to iterate coef_ data: %w", err)
	}

	// Extract intercept_ (bias)
	interceptObj, err := dict.value("intercept_")
	if err != nil {
		return fmt.Errorf("failed to get intercept_: %w", err)
	}
	interceptArray := NumpyArrayRaw{}
	if err := ParseClass(&interceptArray, interceptObj); err != nil {
		return fmt.Errorf("failed to parse intercept_: %w", err)
	}

	// intercept_ shape is [n_classes] for multiclass or [1] for binary
	if len(interceptArray.Shape) != 1 {
		return fmt.Errorf("expected intercept_ shape to be 1D, got %v", interceptArray.Shape)
	}

	t.Intercepts = make([]float64, interceptArray.Shape[0])
	err = interceptArray.Data.Iterate(8, func(b []byte) error {
		t.Intercepts = append(t.Intercepts, util.Float64FromBytes(b, interceptArray.Type.LittleEndinan))
		return nil
	})
	if err != nil {
		return fmt.Errorf("failed to iterate intercept_ data: %w", err)
	}

	return nil
}

// SklearnCalibratedClassifierCV represents sklearn.calibration.CalibratedClassifierCV
type SklearnCalibratedClassifierCV struct {
	BaseEstimator any                         // The wrapped classifier (XGBClassifier, LGBMClassifier, etc.)
	Method        string                      // 'sigmoid' or 'isotonic'
	Calibrators   []SklearnLogisticRegression // Fitted calibrators (one per fold)
	NClasses      int
}

// Reduce implements PythonClass interface
func (t *SklearnCalibratedClassifierCV) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "sklearn.calibration", "CalibratedClassifierCV")
	if err != nil {
		return err
	}
	return nil
}

// Build implements PythonClass interface
func (t *SklearnCalibratedClassifierCV) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}

	// Get method ('sigmoid' or 'isotonic')
	methodObj, err := dict.value("method")
	if err != nil {
		return fmt.Errorf("failed to get method: %w", err)
	}
	methodUnicode, err := toUnicode(methodObj, -1)
	if err != nil {
		return fmt.Errorf("failed to parse method: %w", err)
	}
	t.Method = string(methodUnicode)

	// Get base_estimator_ (the wrapped classifier)
	baseEstimatorObj, err := dict.value("base_estimator_")
	if err != nil {
		return fmt.Errorf("failed to get base_estimator_: %w", err)
	}
	t.BaseEstimator = baseEstimatorObj

	// Get calibrators_ (list of fitted calibrators)
	calibratorsObj, err := dict.value("calibrators_")
	if err != nil {
		return fmt.Errorf("failed to get calibrators_: %w", err)
	}

	// calibrators_ is a list of LogisticRegression objects (one per cv fold)
	calibratorsList, err := toList(calibratorsObj, -1)
	if err != nil {
		return fmt.Errorf("failed to parse calibrators_ as list: %w", err)
	}

	t.Calibrators = make([]SklearnLogisticRegression, 0, len(calibratorsList))
	for i, calObj := range calibratorsList {
		cal := SklearnLogisticRegression{}
		if err := ParseClass(&cal, calObj); err != nil {
			return fmt.Errorf("failed to parse calibrator %d: %w", i, err)
		}
		t.Calibrators = append(t.Calibrators, cal)
	}

	// Get n_classes_
	nClassesObj, err := dict.value("n_classes_")
	if err != nil {
		return fmt.Errorf("failed to get n_classes_: %w", err)
	}
	nClassesRaw := NumpyScalarRaw{}
	if err := ParseClass(&nClassesRaw, nClassesObj); err != nil {
		return fmt.Errorf("failed to parse n_classes_: %w", err)
	}
	if nClassesRaw.Type.Type != "i8" {
		return fmt.Errorf("expected i8 for n_classes_, got %v", nClassesRaw.Type.Type)
	}
	t.NClasses = int(util.Float64FromBytes(nClassesRaw.Data, nClassesRaw.Type.LittleEndinan))

	return nil
}

// SklearnXGBClassifier represents xgboost.XGBClassifier
type SklearnXGBClassifier struct {
	BoosterObj any // The underlying xgboost Booster
}

// Reduce implements PythonClass interface for XGBClassifier
func (t *SklearnXGBClassifier) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "xgboost", "XGBClassifier")
	if err != nil {
		return err
	}
	return nil
}

// Build implements PythonClass interface for XGBClassifier
func (t *SklearnXGBClassifier) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}

	// Get the booster_ attribute which contains the actual xgboost model
	boosterObj, err := dict.value("booster_")
	if err != nil {
		return fmt.Errorf("missing booster_ attribute in XGBClassifier: %w", err)
	}

	// Recursive parsing of the nested booster object
	booster := &SklearnXGBoostBooster{}
	if err := ParseClass(booster, boosterObj); err != nil {
		return fmt.Errorf("failed to recursively parse XGBoost booster: %w", err)
	}
	t.BoosterObj = booster

	return nil
}

// ParseXGBClassifier attempts to parse an XGBClassifier from a pickle object
// Returns the parsed XGBClassifier or an error if parsing fails
func ParseXGBClassifier(obj any) (*SklearnXGBClassifier, error) {
	xgb := &SklearnXGBClassifier{}
	if err := ParseClass(xgb, obj); err != nil {
		return nil, err
	}
	return xgb, nil
}

// SklearnLGBMClassifier represents lightgbm.LGBMClassifier
type SklearnLGBMClassifier struct {
	BoosterObj any // The underlying lightgbm Booster
}

// Reduce implements PythonClass interface for LGBMClassifier
func (t *SklearnLGBMClassifier) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "lightgbm", "LGBMClassifier")
	if err != nil {
		return err
	}
	return nil
}

// Build implements PythonClass interface for LGBMClassifier
func (t *SklearnLGBMClassifier) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}

	// Get the booster_ attribute which contains the actual lightgbm model
	boosterObj, err := dict.value("booster_")
	if err != nil {
		return fmt.Errorf("missing booster_ attribute in LGBMClassifier: %w", err)
	}

	// Recursive parsing of the nested booster object
	booster := &SklearnLightGBMBooster{}
	if err := ParseClass(booster, boosterObj); err != nil {
		return fmt.Errorf("failed to recursively parse LightGBM booster: %w", err)
	}
	t.BoosterObj = booster

	return nil
}

// ParseLGBMClassifier attempts to parse an LGBMClassifier from a pickle object
// Returns the parsed LGBMClassifier or an error if parsing fails
func ParseLGBMClassifier(obj any) (*SklearnLGBMClassifier, error) {
	lgbm := &SklearnLGBMClassifier{}
	if err := ParseClass(lgbm, obj); err != nil {
		return nil, err
	}
	return lgbm, nil
}

// SklearnXGBoostBooster parses the xgboost.core.Booster nested in XGBClassifier
type SklearnXGBoostBooster struct {
	Learner Dict
}

func (b *SklearnXGBoostBooster) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "xgboost.core", "Booster")
	return err
}

func (b *SklearnXGBoostBooster) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}
	b.Learner = dict
	return nil
}

// SklearnLightGBMBooster parses the lightgbm.basic.Booster nested in LGBMClassifier
type SklearnLightGBMBooster struct {
	Handle Dict
}

func (b *SklearnLightGBMBooster) Reduce(reduce Reduce) error {
	_, err := toGlobal(reduce.Callable, "lightgbm.basic", "Booster")
	return err
}

func (b *SklearnLightGBMBooster) Build(build Build) error {
	dict, err := toDict(build.Args)
	if err != nil {
		return err
	}
	b.Handle = dict
	return nil
}
