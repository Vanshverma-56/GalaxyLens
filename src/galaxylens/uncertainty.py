"""Calibration and selective-prediction helpers.

Thresholds must be selected using validation predictions only, then frozen before
the test set is evaluated.
"""
from __future__ import annotations
import numpy as np


def expected_calibration_error(y_true, probabilities, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(y_true) != len(probabilities):
        raise ValueError("Expected probabilities shaped (n_samples, n_classes).")
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    correct = (prediction == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Include confidence=1 in the final bin.
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if i == 0:
            mask |= confidence == 0.0
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_score_multiclass(y_true, probabilities) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def threshold_for_coverage(validation_confidence, target_coverage: float) -> float:
    """Choose a confidence cutoff on validation data to retain approximately target coverage."""
    confidence = np.asarray(validation_confidence, dtype=np.float64)
    if confidence.ndim != 1 or len(confidence) == 0:
        raise ValueError("validation_confidence must be a non-empty 1D array")
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1].")
    # Retain the highest-confidence ceil(target*n) examples; ties may increase coverage.
    n_keep = max(1, int(np.ceil(target_coverage * len(confidence))))
    ordered = np.sort(confidence)
    return float(ordered[-n_keep])


def selective_metrics(y_true, y_pred, confidence, threshold: float) -> dict:
    from sklearn.metrics import accuracy_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence)
    keep = confidence >= threshold
    n = len(y_true)
    retained = int(keep.sum())
    return {
        "threshold": float(threshold),
        "n_total": int(n),
        "n_retained": retained,
        "coverage": float(retained / max(n, 1)),
        "accuracy_on_retained": float(accuracy_score(y_true[keep], y_pred[keep])) if retained else None,
        "n_errors_retained": int(np.sum(y_true[keep] != y_pred[keep])),
        "n_errors_total_before_rejection": int(np.sum(y_true != y_pred)),
    }
