"""Classical Tree-based baseline models: Random Forest and XGBoost.

Follows BaseModel interface with deterministic seeding, fit, predict, and predict_proba.
"""

from typing import Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    """Random Forest Classifier baseline.
    
    Default configuration:
        n_estimators: 100
        max_depth: 5
        random_state: 42
    """

    def __init__(
        self,
        name: str = "Random Forest",
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Fit Random Forest model on training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities of shape (n_samples, 2)."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            # Single class edge case
            p1 = proba[:, 0] if self.model.classes_[0] == 1 else 1.0 - proba[:, 0]
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        return proba

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return Gini feature importances if fitted."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.feature_importances_


class XGBoostModel(BaseModel):
    """XGBoost Classifier baseline.
    
    Default configuration:
        learning_rate: 0.05
        n_estimators: 100
        max_depth: 4
        random_state: 42
    """

    def __init__(
        self,
        name: str = "XGBoost",
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = 4,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = XGBClassifier(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            eval_metric="logloss",
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        """Fit XGBoost model on training data."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities of shape (n_samples, 2)."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet. Call fit() first.")
        proba = self.model.predict_proba(X)
        if proba.shape[1] == 1:
            p1 = proba[:, 0] if self.model.classes_[0] == 1 else 1.0 - proba[:, 0]
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        return proba

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importances if fitted."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.feature_importances_
