package transformation

import (
	"fmt"

	"github.com/citizenadam/go-xgboost-lightgrb/util"
)

// CalibratorParams holds the calibration parameters for a single fold's calibrator
type CalibratorParams struct {
	Coefficients []float64 // [a] for binary classification
	Intercept    float64   // [b] for binary classification
}

// TransformCalibratedSigmoid applies ensemble-averaged sigmoid calibration
// This averages predictions from all cv-fold calibrators to minimize Brier Score
type TransformCalibratedSigmoid struct {
	NumOutputGroups int
	Calibrators     []CalibratorParams // One per cv fold
}

// Transform applies the ensemble-averaged sigmoid calibration to raw predictions
// For each calibrator i: p_i = sigmoid(a_i * raw + b_i)
// Final: p_final = mean(p_1, p_2, ..., p_k) where k = number of calibrators
func (t *TransformCalibratedSigmoid) Transform(rawPredictions []float64, outputPredictions []float64, startIndex int) error {
	nCalibrators := len(t.Calibrators)
	if nCalibrators == 0 {
		return fmt.Errorf("no calibrators available")
	}

	if len(rawPredictions) != t.NumOutputGroups {
		return fmt.Errorf("expected len(rawPredictions) = %d (got %d)", t.NumOutputGroups, len(rawPredictions))
	}

	// Initialize averaged predictions
	for k := 0; k < t.NumOutputGroups; k++ {
		outputPredictions[startIndex+k] = 0.0
	}

	// Sum predictions from each calibrator (ensemble averaging)
	for _, cal := range t.Calibrators {
		for k := 0; k < t.NumOutputGroups; k++ {
			// sigmoid(a * raw + b)
			raw := rawPredictions[k]
			if k < len(cal.Coefficients) && cal.Coefficients[k] != 0 {
				raw = cal.Coefficients[k]*rawPredictions[k] + cal.Intercept
			}
			p := util.Sigmoid(raw)
			outputPredictions[startIndex+k] += p
		}
	}

	// Average
	coef := 1.0 / float64(nCalibrators)
	for k := 0; k < t.NumOutputGroups; k++ {
		outputPredictions[startIndex+k] *= coef
	}

	return nil
}

// NOutputGroups returns the number of output groups
func (t *TransformCalibratedSigmoid) NOutputGroups() int {
	return t.NumOutputGroups
}

// Type returns the transformation type
func (t *TransformCalibratedSigmoid) Type() TransformType {
	return CalibratedSigmoid
}

// Name returns the transformation name
func (t *TransformCalibratedSigmoid) Name() string {
	return CalibratedSigmoid.Name()
}
