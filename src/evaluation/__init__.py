"""Evaluation, Cross-Validation, Metrics, and Pipeline Integrity Package."""

from src.evaluation.fold_audit import FoldIntegrityAuditor, audit_all_datasets
from src.evaluation.cv_runner import CrossValidationRunner
from src.evaluation.metrics import (
    calculate_metrics,
    measure_inference_latency,
    evaluate_model,
    compute_confusion_components,
)

__all__ = [
    "FoldIntegrityAuditor",
    "audit_all_datasets",
    "CrossValidationRunner",
    "calculate_metrics",
    "measure_inference_latency",
    "evaluate_model",
    "compute_confusion_components",
]
