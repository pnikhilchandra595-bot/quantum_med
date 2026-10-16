"""Evaluation module for classical and quantum models."""

from src.evaluation.metrics import (
    calculate_metrics,
    measure_inference_latency,
    evaluate_model,
    compute_confusion_components,
)

__all__ = [
    "calculate_metrics",
    "measure_inference_latency",
    "evaluate_model",
    "compute_confusion_components",
]
