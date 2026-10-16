"""Standardized Clinical Evaluation Metrics Calculator.

Computes the core 5 clinical and statistical performance metrics plus
wall-clock inference latency per sample:
1. Accuracy: (TP + TN) / (TP + TN + FP + FN)
2. Sensitivity (Recall): TP / (TP + FN) - minimizes false negatives in oncology
3. Specificity: TN / (TN + FP)
4. F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
5. ROC-AUC: Area Under the Receiver Operating Characteristic Curve
6. Wall-clock inference latency per sample (ms)

Includes robust edge-case handling for zero division.
"""

from typing import Dict, Optional, Tuple, Union
import time
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_confusion_components(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[int, int, int, int]:
    """Compute True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)."""
    y_true_b = np.asarray(y_true, dtype=int).ravel()
    y_pred_b = np.asarray(y_pred, dtype=int).ravel()

    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    return tp, fp, tn, fn


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    latency_ms: Optional[float] = None,
    round_decimals: Optional[int] = 4,
) -> Dict[str, float]:
    """Calculate clinical metrics dictionary.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted positive class probabilities (or (n, 2) array).
        latency_ms: Inference wall-clock latency per sample in milliseconds.
        round_decimals: Number of decimal places to round (default: 4).

    Returns:
        Standardized dictionary containing all 6 metrics:
        'accuracy', 'sensitivity', 'specificity', 'f1_score', 'roc_auc', 'latency_ms'.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_pred_arr = np.asarray(y_pred, dtype=int).ravel()

    tp, fp, tn, fn = compute_confusion_components(y_true_arr, y_pred_arr)
    total = tp + fp + tn + fn

    # 1. Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # 2. Sensitivity (Recall)
    actual_positives = tp + fn
    sensitivity = tp / actual_positives if actual_positives > 0 else 0.0

    # 3. Specificity
    actual_negatives = tn + fp
    specificity = tn / actual_negatives if actual_negatives > 0 else 0.0

    # 4. Precision & F1-Score
    predicted_positives = tp + fp
    precision = tp / predicted_positives if predicted_positives > 0 else 0.0
    if precision + sensitivity > 0:
        f1_score = 2.0 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0

    # 5. ROC-AUC
    if y_prob is not None:
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim == 2:
            prob_arr = prob_arr[:, 1]
        prob_arr = prob_arr.ravel()
        
        # Check if single class present in y_true
        unique_classes = np.unique(y_true_arr)
        if len(unique_classes) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_arr, prob_arr))
            except Exception:
                roc_auc = 0.5
        else:
            roc_auc = 0.5
    else:
        # Fallback to decision boundary score
        roc_auc = float(accuracy)

    # 6. Latency per sample in ms
    lat = float(latency_ms) if latency_ms is not None else 0.0

    results = {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
        "roc_auc": float(roc_auc),
        "latency_ms": float(lat),
    }

    if round_decimals is not None:
        return {k: round(v, round_decimals) for k, v in results.items()}
    return results


def measure_inference_latency(
    model: any, X: np.ndarray, n_warmup: int = 2, n_runs: int = 5
) -> float:
    """Measure inference latency per sample in milliseconds with warmup runs."""
    n_samples = len(X)
    if n_samples == 0:
        return 0.0

    # Warmup
    for _ in range(n_warmup):
        _ = model.predict(X)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = model.predict(X)
    elapsed_total = time.perf_counter() - start

    avg_elapsed_per_run = elapsed_total / n_runs
    latency_ms_per_sample = (avg_elapsed_per_run * 1000.0) / n_samples
    return latency_ms_per_sample


def evaluate_model(
    model: any, X_test: np.ndarray, y_test: np.ndarray, round_decimals: int = 4
) -> Dict[str, float]:
    """Convenience function to measure latency, predict, and compute all 6 clinical metrics."""
    latency = measure_inference_latency(model, X_test)
    y_pred = model.predict(X_test)
    
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            y_prob = None

    return calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        latency_ms=latency,
        round_decimals=round_decimals,
    )
