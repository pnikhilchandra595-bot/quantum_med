"""Classical Support Vector Machine (Linear & RBF) Baseline.

Module: src.models.classical.svm
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D5 (Classical Baselines)
Task: T1.6
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any, Dict
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC

from src.models.base import BaseModel


class ClassicalSVM(BaseModel):
    """Classical Support Vector Classifier wrapping scikit-learn SVC.
    
    Supports kernel='linear' and kernel='rbf' with probability calibration
    for clinical risk estimation and ROC-AUC computation.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        probability: bool = True,
        random_state: int = 42,
        **kwargs: Any,
    ):
        if kernel not in ["linear", "rbf", "poly", "sigmoid"]:
            raise ValueError(f"Unsupported kernel '{kernel}'. Expected 'linear' or 'rbf'.")

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=self.probability,
            random_state=self.random_state,
            **self.extra_kwargs,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "ClassicalSVM":
        """Fit SVM classifier on feature matrix X and target labels y."""
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic class labels {0, 1}."""
        if not self.is_fitted_:
            raise RuntimeError("ClassicalSVM model must be fitted before predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict(X_arr).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities of shape (N, 2)."""
        if not self.is_fitted_:
            raise RuntimeError("ClassicalSVM model must be fitted before predict_proba.")
        X_arr = np.asarray(X, dtype=np.float64)
        if self.probability:
            return self.model.predict_proba(X_arr)
        else:
            df = self.model.decision_function(X_arr)
            prob_1 = 1.0 / (1.0 + np.exp(-np.clip(df, -50.0, 50.0)))
            prob_0 = 1.0 - prob_1
            return np.column_stack([prob_0, prob_1])

    def save(self, file_path: Union[str, Path]) -> None:
        """Serialize trained model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, file_path: Union[str, Path]) -> "ClassicalSVM":
        """Load serialized model from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self.model = joblib.load(path)
        self.is_fitted_ = True
        return self
