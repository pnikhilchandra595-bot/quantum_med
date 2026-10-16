"""Unit tests for standardized clinical evaluation metrics."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    calculate_metrics,
    compute_confusion_components,
    evaluate_model,
)
from src.models.classical.trees import RandomForestModel


def test_metrics_standard_case():
    """Test standard clinical metrics calculation with known ground truth and predictions."""
    # TP=3, FP=1, TN=4, FN=2
    # Total = 10
    # y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # y_pred = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.6, 0.2, 0.1, 0.1, 0.2])

    tp, fp, tn, fn = compute_confusion_components(y_true, y_pred)
    assert tp == 3
    assert fp == 1
    assert tn == 4
    assert fn == 2

    metrics = calculate_metrics(y_true, y_pred, y_prob, latency_ms=1.23456, round_decimals=4)

    # Required 6 keys
    expected_keys = {"accuracy", "sensitivity", "specificity", "f1_score", "roc_auc", "latency_ms"}
    assert set(metrics.keys()) == expected_keys

    # Accuracy: (3+4)/10 = 0.7
    assert metrics["accuracy"] == 0.7000
    # Sensitivity (Recall): 3 / (3 + 2) = 0.6
    assert metrics["sensitivity"] == 0.6000
    # Specificity: 4 / (4 + 1) = 0.8
    assert metrics["specificity"] == 0.8000
    # Precision: 3 / (3 + 1) = 0.75; F1: 2 * (0.75 * 0.6) / (0.75 + 0.6) = 0.9 / 1.35 = 0.6667
    assert metrics["f1_score"] == 0.6667
    # Latency: rounded to 4 decimals
    assert metrics["latency_ms"] == 1.2346
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_metrics_zero_division_edge_cases():
    """Verify robust edge-case handling for zero division (e.g., all 0s or all 1s)."""
    # All true are 0
    y_true_zeros = np.zeros(10, dtype=int)
    y_pred_zeros = np.zeros(10, dtype=int)
    metrics_zero = calculate_metrics(y_true_zeros, y_pred_zeros, round_decimals=4)
    assert metrics_zero["accuracy"] == 1.0000
    assert metrics_zero["sensitivity"] == 0.0000  # no actual positives
    assert metrics_zero["specificity"] == 1.0000  # all actual negatives identified
    assert metrics_zero["f1_score"] == 0.0000

    # All true are 1, but model predicted all 0 (Worst case clinical false negatives)
    y_true_ones = np.ones(10, dtype=int)
    y_pred_zeros = np.zeros(10, dtype=int)
    metrics_fn = calculate_metrics(y_true_ones, y_pred_zeros, round_decimals=4)
    assert metrics_fn["accuracy"] == 0.0000
    assert metrics_fn["sensitivity"] == 0.0000
    assert metrics_fn["specificity"] == 0.0000
    assert metrics_fn["f1_score"] == 0.0000


def test_metrics_evaluate_model():
    """Test full evaluate_model pipeline with trained RandomForest."""
    X = np.random.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    rf = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
    rf.fit(X, y)

    res = evaluate_model(rf, X, y, round_decimals=4)
    assert len(res) == 6
    assert all(isinstance(v, float) for v in res.values())
    assert res["accuracy"] >= 0.70
