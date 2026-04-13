package leaves

import (
	"bufio"
	"bytes"
	"fmt"
	"os"

	"github.com/citizenadam/go-xgboost-lightgrb/internal/pickle"
	"github.com/citizenadam/go-xgboost-lightgrb/transformation"
)

func lgTreeFromSklearnDecisionTreeRegressor(tree pickle.SklearnDecisionTreeRegressor, scale float64, base float64) (lgTree, error) {
	t := lgTree{}
	// no support for categorical features in sklearn trees
	t.nCategorical = 0

	numLeaves := 0
	numNodes := 0
	for _, n := range tree.Tree.Nodes {
		if n.LeftChild < 0 {
			numLeaves++
		} else {
			numNodes++
		}
	}

	if numLeaves-1 != numNodes {
		return t, fmt.Errorf("unexpected number of leaves (%d) and nodes (%d)", numLeaves, numNodes)
	}

	if numNodes == 0 {
		// special case
		// we mimic decision rule but left and right childs lead to the same result
		t.nodes = make([]lgNode, 0, 1)
		node := numericalNode(0, 0, 0.0, 0)
		node.Flags |= leftLeaf
		node.Flags |= rightLeaf
		node.Left = uint32(len(t.leafValues))
		node.Right = uint32(len(t.leafValues))
		t.nodes = append(t.nodes, node)
		t.leafValues = append(t.leafValues, tree.Tree.Values[0]*scale+base)
		return t, nil
	}

	// Numerical only
	createNode := func(idx int) (lgNode, error) {
		node := lgNode{}
		refNode := &tree.Tree.Nodes[idx]
		missingType := uint8(0)
		defaultType := uint8(0)
		node = numericalNode(uint32(refNode.Feature), missingType, refNode.Threshold, defaultType)
		if tree.Tree.Nodes[refNode.LeftChild].LeftChild < 0 {
			node.Flags |= leftLeaf
			node.Left = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, tree.Tree.Values[refNode.LeftChild]*scale+base)
		}
		if tree.Tree.Nodes[refNode.RightChild].LeftChild < 0 {
			node.Flags |= rightLeaf
			node.Right = uint32(len(t.leafValues))
			t.leafValues = append(t.leafValues, tree.Tree.Values[refNode.RightChild]*scale+base)
		}
		return node, nil
	}

	origNodeIdxStack := make([]uint32, 0, numNodes)
	convNodeIdxStack := make([]uint32, 0, numNodes)
	visited := make([]bool, tree.Tree.NNodes)
	t.nodes = make([]lgNode, 0, numNodes)
	node, err := createNode(0)
	if err != nil {
		return t, err
	}
	t.nodes = append(t.nodes, node)
	origNodeIdxStack = append(origNodeIdxStack, 0)
	convNodeIdxStack = append(convNodeIdxStack, 0)
	for len(origNodeIdxStack) > 0 {
		convIdx := convNodeIdxStack[len(convNodeIdxStack)-1]
		if t.nodes[convIdx].Flags&rightLeaf == 0 {
			origIdx := tree.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].RightChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Right = uint32(convNewIdx)
				continue
			}
		}
		if t.nodes[convIdx].Flags&leftLeaf == 0 {
			origIdx := tree.Tree.Nodes[origNodeIdxStack[len(origNodeIdxStack)-1]].LeftChild
			if !visited[origIdx] {
				node, err := createNode(origIdx)
				if err != nil {
					return t, err
				}
				t.nodes = append(t.nodes, node)
				convNewIdx := len(t.nodes) - 1
				convNodeIdxStack = append(convNodeIdxStack, uint32(convNewIdx))
				origNodeIdxStack = append(origNodeIdxStack, uint32(origIdx))
				visited[origIdx] = true
				t.nodes[convIdx].Left = uint32(convNewIdx)
				continue
			}
		}
		origNodeIdxStack = origNodeIdxStack[:len(origNodeIdxStack)-1]
		convNodeIdxStack = convNodeIdxStack[:len(convNodeIdxStack)-1]
	}
	return t, nil
}

// SKEnsembleFromReader reads sklearn tree ensemble model from `reader`
func SKEnsembleFromReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	decoder := pickle.NewDecoder(reader)
	res, err := decoder.Decode()
	if err != nil {
		return nil, fmt.Errorf("error while decoding: %w", err)
	}

	// Try to parse as CalibratedClassifierCV first
	calibrated := &pickle.SklearnCalibratedClassifierCV{}
	if err := pickle.ParseClass(calibrated, res); err == nil {
		// Successfully parsed as CalibratedClassifierCV
		// ParseClass has populated the calibrated object, including Calibrators
		return parseCalibratedClassifierCV(calibrated, calibrated.Calibrators)
	}

	// Fall back to GradientBoostingClassifier (existing behavior)
	e := &lgEnsemble{name: "sklearn.ensemble.GradientBoostingClassifier"}
	gbdt := pickle.SklearnGradientBoosting{}
	err = pickle.ParseClass(&gbdt, res)
	if err != nil {
		return nil, fmt.Errorf("error while parsing gradient boosting class: %w", err)
	}

	e.nRawOutputGroups = gbdt.NClasses
	if e.nRawOutputGroups == 2 {
		e.nRawOutputGroups = 1
	}

	e.MaxFeatureIdx = gbdt.MaxFeatures - 1

	nTrees := gbdt.NEstimators
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	}

	if gbdt.NEstimators*e.nRawOutputGroups != len(gbdt.Estimators) {
		return nil, fmt.Errorf("unexpected number of trees (NEstimators = %d, nRawOutputGroups = %d, len(Estimatoers) = %d", gbdt.NEstimators, e.nRawOutputGroups, len(gbdt.Estimators))
	}

	scale := gbdt.LearningRate
	base := make([]float64, e.nRawOutputGroups)
	if gbdt.InitEstimator.Name == "LogOddsEstimator" {
		for i := 0; i < e.nRawOutputGroups; i++ {
			base[i] = gbdt.InitEstimator.Prior[0]
		}
	} else if gbdt.InitEstimator.Name == "PriorProbabilityEstimator" {
		if len(gbdt.InitEstimator.Prior) != len(base) {
			return nil, fmt.Errorf("len(gbdt.InitEstimator.Prior) != len(base)")
		}
		base = gbdt.InitEstimator.Prior
	} else {
		return nil, fmt.Errorf("unknown initial estimator \"%s\"", gbdt.InitEstimator.Name)
	}

	e.Trees = make([]lgTree, 0, gbdt.NEstimators*gbdt.NClasses)
	for i := 0; i < gbdt.NEstimators; i++ {
		for j := 0; j < e.nRawOutputGroups; j++ {
			treeNum := i*e.nRawOutputGroups + j
			tree, err := lgTreeFromSklearnDecisionTreeRegressor(gbdt.Estimators[treeNum], scale, base[j])
			if err != nil {
				return nil, fmt.Errorf("error while creating %d tree: %w", treeNum, err)
			}
			e.Trees = append(e.Trees, tree)
		}
		for k := range base {
			base[k] = 0.0
		}
	}
	return &Ensemble{e, &transformation.TransformRaw{NumOutputGroups: e.nRawOutputGroups}}, nil
}

// SKEnsembleFromFile reads sklearn tree ensemble model from pickle file
func SKEnsembleFromFile(filename string, loadTransformation bool) (*Ensemble, error) {
	reader, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	bufReader := bufio.NewReader(reader)
	return SKEnsembleFromReader(bufReader, loadTransformation)
}

// SKEnsembleFromPickleFile reads sklearn tree ensemble model from pickle file
// Supports pickle protocols 0-5
// This is an alias for SKEnsembleFromFile for API clarity
func SKEnsembleFromPickleFile(filename string, loadTransformation bool) (*Ensemble, error) {
	return SKEnsembleFromFile(filename, loadTransformation)
}

// SKEnsembleFromPickleReader reads sklearn tree ensemble model from pickle reader
// Supports pickle protocols 0-5
// This is an alias for SKEnsembleFromReader for API clarity
func SKEnsembleFromPickleReader(reader *bufio.Reader, loadTransformation bool) (*Ensemble, error) {
	return SKEnsembleFromReader(reader, loadTransformation)
}

// parseCalibratedClassifierCV parses a CalibratedClassifierCV wrapper
// and returns an Ensemble with the appropriate calibration transformation
func parseCalibratedClassifierCV(calibrated *pickle.SklearnCalibratedClassifierCV, calibrators []pickle.SklearnLogisticRegression) (*Ensemble, error) {
	if calibrated.Method != "sigmoid" {
		return nil, fmt.Errorf("unsupported calibration method: %s (only 'sigmoid' is supported)", calibrated.Method)
	}

	// Get the base estimator from the pickle object
	// The base estimator should be a Reduce or Build containing the actual model
	baseEstimator := calibrated.BaseEstimator

	// Try to parse as GradientBoostingClassifier
	gbdt := &pickle.SklearnGradientBoosting{}
	if err := pickle.ParseClass(gbdt, baseEstimator); err == nil {
		return parseGradientBoostingFromCalibrated(gbdt, calibrators)
	}

	// Try to parse as XGBClassifier (xgboost.XGBClassifier)
	xgbModel, err := pickle.ParseXGBClassifier(baseEstimator)
	if err == nil {
		return parseXGBClassifierFromCalibrated(xgbModel, calibrators)
	}

	// Try to parse as LGBMClassifier (lightgbm.LGBMClassifier)
	lgbmModel, err := pickle.ParseLGBMClassifier(baseEstimator)
	if err == nil {
		return parseLGBMClassifierFromCalibrated(lgbmModel, calibrators)
	}

	// If we can't identify the base estimator type, return an error with helpful message
	return nil, fmt.Errorf("unsupported base estimator type in CalibratedClassifierCV: %T (supported: GradientBoostingClassifier, XGBClassifier, LGBMClassifier)", baseEstimator)
}

// parseXGBClassifierFromCalibrated parses an XGBClassifier from inside a CalibratedClassifierCV
func parseXGBClassifierFromCalibrated(xgbModel *pickle.SklearnXGBClassifier, calibrators []pickle.SklearnLogisticRegression) (*Ensemble, error) {
	// The booster_ contains the actual xgboost Booster model
	// Now with recursive parsing, BoosterObj is *SklearnXGBoostBooster

	booster, ok := xgbModel.BoosterObj.(*pickle.SklearnXGBoostBooster)
	if !ok {
		return nil, fmt.Errorf("invalid booster type for XGBoost: %T", xgbModel.BoosterObj)
	}

	// Extract the model state - typically stored in "_learner" key
	rawLearner, ok := booster.Learner["_learner"]
	if !ok {
		return nil, fmt.Errorf("booster_ attribute missing _learner key")
	}

	modelData, ok := rawLearner.([]byte)
	if !ok {
		return nil, fmt.Errorf("_learner data is not a byte slice, got %T", rawLearner)
	}

	// Use bufio wrapper for binary parsing (required by XGEnsembleFromReader)
	bufReader := bufio.NewReader(bytes.NewReader(modelData))
	ensemble, err := XGEnsembleFromReader(bufReader, false)
	if err != nil {
		return nil, fmt.Errorf("XGEnsembleFromReader failed: %w", err)
	}

	// Build calibration transformation with ensemble averaging
	calTransform := buildCalibratedTransform(calibrators, ensemble.NRawOutputGroups())
	return &Ensemble{ensemble, calTransform}, nil
}

// parseLGBMClassifierFromCalibrated parses an LGBMClassifier from inside a CalibratedClassifierCV
func parseLGBMClassifierFromCalibrated(lgbmModel *pickle.SklearnLGBMClassifier, calibrators []pickle.SklearnLogisticRegression) (*Ensemble, error) {
	// The booster_ contains the actual lightgbm Booster model
	// Now with recursive parsing, BoosterObj is *SklearnLightGBMBooster

	booster, ok := lgbmModel.BoosterObj.(*pickle.SklearnLightGBMBooster)
	if !ok {
		return nil, fmt.Errorf("invalid booster type for LightGBM: %T", lgbmModel.BoosterObj)
	}

	// LightGBM typically stores model as JSON string in "handle" or "_handle" key
	var modelJSON []byte

	// Try "handle" key first
	if handle, ok := booster.Handle["handle"]; ok {
		if str, ok := handle.(string); ok {
			modelJSON = []byte(str)
		}
	}

	// If not found, try "_handle"
	if modelJSON == nil {
		if handle, ok := booster.Handle["_handle"]; ok {
			if str, ok := handle.(string); ok {
				modelJSON = []byte(str)
			}
		}
	}

	if len(modelJSON) == 0 {
		return nil, fmt.Errorf("no model data found in LightGBM booster handle")
	}

	// Use JSON parser for LightGBM (takes io.Reader, not bufio.Reader)
	ensemble, err := LGEnsembleFromJSON(bytes.NewReader(modelJSON), false)
	if err != nil {
		return nil, fmt.Errorf("LGEnsembleFromJSON failed: %w", err)
	}

	// Build calibration transformation with ensemble averaging
	calTransform := buildCalibratedTransform(calibrators, ensemble.NRawOutputGroups())
	return &Ensemble{ensemble, calTransform}, nil
}

// parseGradientBoostingFromCalibrated parses a GradientBoostingClassifier from inside a CalibratedClassifierCV
func parseGradientBoostingFromCalibrated(gbdt *pickle.SklearnGradientBoosting, calibrators []pickle.SklearnLogisticRegression) (*Ensemble, error) {
	e := &lgEnsemble{name: "sklearn.ensemble.GradientBoostingClassifier (calibrated)"}

	e.nRawOutputGroups = gbdt.NClasses
	if e.nRawOutputGroups == 2 {
		e.nRawOutputGroups = 1
	}

	e.MaxFeatureIdx = gbdt.MaxFeatures - 1

	nTrees := gbdt.NEstimators
	if nTrees == 0 {
		return nil, fmt.Errorf("no trees in file")
	}

	if gbdt.NEstimators*e.nRawOutputGroups != len(gbdt.Estimators) {
		return nil, fmt.Errorf("unexpected number of trees (NEstimators = %d, nRawOutputGroups = %d, len(Estimatoers) = %d", gbdt.NEstimators, e.nRawOutputGroups, len(gbdt.Estimators))
	}

	scale := gbdt.LearningRate
	base := make([]float64, e.nRawOutputGroups)
	if gbdt.InitEstimator.Name == "LogOddsEstimator" {
		for i := 0; i < e.nRawOutputGroups; i++ {
			base[i] = gbdt.InitEstimator.Prior[0]
		}
	} else if gbdt.InitEstimator.Name == "PriorProbabilityEstimator" {
		if len(gbdt.InitEstimator.Prior) != len(base) {
			return nil, fmt.Errorf("len(gbdt.InitEstimator.Prior) != len(base)")
		}
		base = gbdt.InitEstimator.Prior
	} else {
		return nil, fmt.Errorf("unknown initial estimator \"%s\"", gbdt.InitEstimator.Name)
	}

	e.Trees = make([]lgTree, 0, gbdt.NEstimators*gbdt.NClasses)
	for i := 0; i < gbdt.NEstimators; i++ {
		for j := 0; j < e.nRawOutputGroups; j++ {
			treeNum := i*e.nRawOutputGroups + j
			tree, err := lgTreeFromSklearnDecisionTreeRegressor(gbdt.Estimators[treeNum], scale, base[j])
			if err != nil {
				return nil, fmt.Errorf("error while creating %d tree: %w", treeNum, err)
			}
			e.Trees = append(e.Trees, tree)
		}
		for k := range base {
			base[k] = 0.0
		}
	}

	// Build calibration transformation with ensemble averaging
	calTransform := buildCalibratedTransform(calibrators, e.nRawOutputGroups)
	return &Ensemble{e, calTransform}, nil
}

// buildCalibratedTransform creates a calibrated sigmoid transformation from calibrator parameters
func buildCalibratedTransform(calibrators []pickle.SklearnLogisticRegression, nOutputGroups int) transformation.Transform {
	calParams := make([]transformation.CalibratorParams, len(calibrators))
	for i, cal := range calibrators {
		// For binary classification, coef_ is [1, 1] and intercept_ is [1]
		// For multiclass, coef_ is [n_classes, n_classes-1] (one-vs-rest)
		// We take the first row for binary (the coefficient for the positive class)
		coef := make([]float64, nOutputGroups)
		if len(cal.Coefficients) > 0 && len(cal.Coefficients[0]) > 0 {
			// Take first coefficient for binary, or map appropriately for multiclass
			if nOutputGroups == 1 && len(cal.Coefficients) >= 1 {
				coef[0] = cal.Coefficients[0][0]
			}
		}
		var intercept float64
		if len(cal.Intercepts) > 0 {
			intercept = cal.Intercepts[0]
		}
		calParams[i] = transformation.CalibratorParams{
			Coefficients: coef,
			Intercept:    intercept,
		}
	}

	return &transformation.TransformCalibratedSigmoid{
		NumOutputGroups: nOutputGroups,
		Calibrators:     calParams,
	}
}
