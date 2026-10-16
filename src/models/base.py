"""
Base interface definition for all clinical diagnostic models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BaseModel(ABC):
    """
    Abstract Base Class for classical and quantum diagnostic models.
    Enforces unified fit, predict, predict_proba, and evaluation contracts.
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "BaseModel":
        """
        Fit model to training data.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels [0, 1].
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities of shape (N, 2).
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute standard clinical metrics: Accuracy, Precision, Recall, F1, ROC-AUC.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        preds = self.predict(X)
        probs = self.predict_proba(X)[:, 1]

        acc = float(accuracy_score(y, preds))
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y, probs))
        except Exception:
            auc = 0.5

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def get_params(self) -> Dict[str, Any]:
        """
        Return model hyperparameters.
        """
        return self.params
