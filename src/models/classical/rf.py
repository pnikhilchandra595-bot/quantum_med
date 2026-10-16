"""Classical Random Forest Classifier Baseline.

Module: src.models.classical.rf
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D5 (Classical Baselines)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel


class ClassicalRandomForest(BaseModel):
    """Classical Random Forest baseline model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 6,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **self.extra_kwargs,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "ClassicalRandomForest":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        self.model.fit(X_arr, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict.")
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict(X_arr).astype(np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_arr = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X_arr)

    def save(self, file_path: Union[str, Path]) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, file_path: Union[str, Path]) -> "ClassicalRandomForest":
        path = Path(file_path)
        self.model = joblib.load(path)
        self.is_fitted_ = True
        return self
