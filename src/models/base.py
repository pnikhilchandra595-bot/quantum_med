"""Base Model Interface for Classical and Hybrid Quantum-Classical Classifiers.

Module: src.models.base
Owner: Member A & Member C
Deliverable: D5 (Unified ML Model Interface)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class BaseModel(ABC):
    """Abstract Base Class specifying unified interface for all classifiers."""

    def __init__(self, name: str = "", random_state: int = 42, **kwargs: Any) -> None:
        self.name = name or self.__class__.__name__
        self.random_state = random_state
        self.params = kwargs
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "BaseModel":
        """Train model on training data with optional validation set for monitoring."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels {0, 1}."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution of shape (N, 2)."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive clinical ML classification metrics.

        Metrics:
            accuracy, precision, recall, f1, roc_auc
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        y_pred = self.predict(X_arr)
        proba = self.predict_proba(X_arr)
        y_score = proba[:, 1] if proba.shape[1] == 2 else proba.ravel()

        metrics = {
            "accuracy": float(accuracy_score(y_arr, y_pred)),
            "precision": float(precision_score(y_arr, y_pred, zero_division=0)),
            "recall": float(recall_score(y_arr, y_pred, zero_division=0)),
            "f1": float(f1_score(y_arr, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_arr, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        return metrics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
