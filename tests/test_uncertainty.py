import numpy as np
import pytest
from galaxylens.uncertainty import (
    expected_calibration_error, brier_score_multiclass,
    threshold_for_coverage, selective_metrics
)

def test_perfect_confident_predictions_have_zero_ece_and_brier():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, p) == 0.0
    assert brier_score_multiclass(y, p) == 0.0

def test_threshold_is_selected_from_validation_confidence():
    threshold = threshold_for_coverage(np.array([0.2, 0.5, 0.7, 0.9]), 0.5)
    assert threshold == pytest.approx(0.7)

def test_selective_metrics_reports_coverage_and_retained_accuracy():
    result = selective_metrics([0, 1, 1], [0, 0, 1], [0.9, 0.3, 0.8], 0.8)
    assert result["n_retained"] == 2
    assert result["coverage"] == pytest.approx(2/3)
    assert result["accuracy_on_retained"] == pytest.approx(1.0)

def test_invalid_coverage_rejected():
    with pytest.raises(ValueError):
        threshold_for_coverage([0.5, 0.7], 0)
