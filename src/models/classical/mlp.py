"""Multi-Layer Perceptron (MLP) Neural Network baseline."""

from typing import Any, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.base import BaseModel


class MLPModel(BaseModel):
    """Multi-Layer Perceptron Classifier baseline.
    
    Default configuration:
        hidden_layer_sizes: (32, 16)
        max_iter: 500
        random_state: 42
    """

    def __init__(
        self,
        name: str = "MLP",
        hidden_layer_sizes: Tuple[int, ...] = (32, 16),
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.predict_proba(X)
