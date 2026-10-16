from typing import Any
import warnings
import numpy as np
from sklearn.svm import SVC

# Suppress sklearn 1.9+ SVC probability deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm")

from src.models.base import BaseModel


class LinearSVMModel(BaseModel):
    """Linear Support Vector Classifier baseline.
    
    Default configuration:
        C: 1.0
        probability: True (for ROC-AUC and probability calibration)
        random_state: 42
    """

    def __init__(
        self,
        name: str = "Linear SVM",
        C: float = 1.0,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.C = C
        self.model = SVC(
            kernel="linear",
            C=self.C,
            probability=True,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMModel":
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


class RBFSVMModel(BaseModel):
    """RBF Kernel Support Vector Classifier baseline.
    
    Default configuration:
        C: 1.0
        gamma: 'scale'
        probability: True
        random_state: 42
    """

    def __init__(
        self,
        name: str = "RBF SVM",
        C: float = 1.0,
        gamma: str = "scale",
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSVMModel":
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
