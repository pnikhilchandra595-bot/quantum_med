"""Classical XGBoost Classifier Baseline.

Module: src.models.classical.xgb
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D5 (Classical Baselines)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, Any
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier

from src.models.base import BaseModel


class ClassicalXGBoost(BaseModel):
    """Classical XGBoost baseline model."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs: Any,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.extra_kwargs = kwargs

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
            **self.extra_kwargs,
        )
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "ClassicalXGBoost":
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        if val_data is not None:
            eval_set = [(val_data[0], val_data[1])]
            self.model.fit(X_arr, y_arr, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_arr, y_arr, verbose=False)

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

    def load(self, file_path: Union[str, Path]) -> "ClassicalXGBoost":
        path = Path(file_path)
        self.model = joblib.load(path)
        self.is_fitted_ = True
        return self
