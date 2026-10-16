"""Feature Reduction Module (PCA & Mutual Information).

Peer-reviewed by Member C (Evaluation & XAI Specialist) to guarantee
STRICT ZERO DATA LEAKAGE across cross-validation splits.

Architectural Rule:
- All scalers, PCA projections, and MI selectors MUST be fit exclusively
  on the training fold (X_train, y_train) and subsequently applied to
  transform the test fold (X_test).
"""

from typing import Tuple, List, Optional, Union, Dict, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler


class PCAReducer(BaseEstimator, TransformerMixin):
    """Zero-Leakage Principal Component Analysis Transformer.
    
    Fits eigenvalues and eigenvectors strictly on training fold.
    """

    def __init__(self, n_components: int = 6, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.effective_k_: int = n_components

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "PCAReducer":
        """Fit scaler and PCA strictly on training fold."""
        X_arr = np.asarray(X, dtype=float)
        n_samples, n_features = X_arr.shape

        # Bound k by min(n_samples, n_features)
        self.effective_k_ = min(self.n_components, n_features, n_samples)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        self.pca = PCA(n_components=self.effective_k_, random_state=self.random_state)
        self.pca.fit(X_scaled)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform unseen test fold using training parameters."""
        if not self.is_fitted or self.pca is None or self.scaler is None:
            raise RuntimeError("PCAReducer must be fitted before transform.")
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_arr)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Return explained variance ratio vector from training fit."""
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("PCAReducer must be fitted.")
        return self.pca.explained_variance_ratio_

    @property
    def cumulative_variance_ratio_(self) -> float:
        """Return total variance retained across selected components."""
        return float(np.sum(self.explained_variance_ratio_))

    def compute_out_of_fold_variance(self, X_test: np.ndarray) -> float:
        """Compute the true reconstruction R^2 variance explained on an unseen test fold.
        
        R^2 = 1 - ||X_test_scaled - X_recon||^2 / ||X_test_scaled||^2
        """
        if not self.is_fitted or self.pca is None or self.scaler is None:
            raise RuntimeError("PCAReducer must be fitted.")
        X_te_scaled = self.scaler.transform(X_test)
        X_te_proj = self.pca.transform(X_te_scaled)
        X_te_recon = self.pca.inverse_transform(X_te_proj)

        res_ss = np.sum((X_te_scaled - X_te_recon) ** 2)
        tot_ss = np.sum(X_te_scaled ** 2)
        if tot_ss == 0:
            return 1.0
        return float(np.clip(1.0 - (res_ss / tot_ss), 0.0, 1.0))


class MutualInfoReducer(BaseEstimator, TransformerMixin):
    """Zero-Leakage Mutual Information Feature Selector.
    
    Computes I(X_j; Y) strictly on the training fold (X_train, y_train).
    """

    def __init__(self, n_features_to_select: int = 6, random_state: int = 42) -> None:
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.selector: Optional[SelectKBest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.scores_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInfoReducer":
        """Compute Mutual Information strictly using training fold data and labels."""
        if y is None:
            raise ValueError("Mutual Information requires target labels y for fitting.")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int).ravel()
        n_samples, n_features = X_arr.shape

        effective_k = min(self.n_features_to_select, n_features)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        def mi_func(X_in, y_in):
            return mutual_info_classif(X_in, y_in, random_state=self.random_state)

        self.selector = SelectKBest(score_func=mi_func, k=effective_k)
        self.selector.fit(X_scaled, y_arr)
        self.scores_ = self.selector.scores_
        self.selected_indices_ = self.selector.get_support(indices=True)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Filter test fold features using training MI indices."""
        if not self.is_fitted or self.selector is None or self.scaler is None:
            raise RuntimeError("MutualInfoReducer must be fitted before transform.")
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self.scaler.transform(X_arr)
        return self.selector.transform(X_scaled)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


class ZeroLeakagePipeline:
    """Unified Zero-Leakage Cross-Validation Preprocessing Pipeline.
    
    Guarantees that test folds remain completely unobserved during
    scaling, PCA projection, and feature ranking.
    """

    @staticmethod
    def split_and_reduce(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        method: str = "pca",
        k: int = 6,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Perform zero-leakage fit on train fold and transform test fold."""
        if method.lower() == "pca":
            reducer = PCAReducer(n_components=k, random_state=random_state)
            X_tr_red = reducer.fit_transform(X_train, y_train)
            X_te_red = reducer.transform(X_test)
            return X_tr_red, X_te_red, reducer
        elif method.lower() in ["mi", "mutual_info"]:
            reducer = MutualInfoReducer(n_features_to_select=k, random_state=random_state)
            X_tr_red = reducer.fit_transform(X_train, y_train)
            X_te_red = reducer.transform(X_test)
            return X_tr_red, X_te_red, reducer
        else:
            raise ValueError(f"Unknown reduction method '{method}'. Supported: 'pca', 'mi'.")
