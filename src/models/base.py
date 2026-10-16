"""Base model interface for Classical and Quantum ML models."""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseModel(ABC):
    """Abstract base class establishing the standard model interface across the pipeline."""

    def __init__(self, name: str, random_state: int = 42, **kwargs: Any) -> None:
        self.name = name
        self.random_state = random_state
        self.params = kwargs
        self.model: Any = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model to the training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Binary target vector of shape (n_samples,).

        Returns:
            self: The fitted model instance.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels for samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted binary class labels (0 or 1) of shape (n_samples,).
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Class probabilities of shape (n_samples, 2), where column 1
                        is the probability of the positive class (e.g., Malignant).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self.is_fitted})"
