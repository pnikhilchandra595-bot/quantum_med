"""Principal Component Analysis (PCA) Reducer for Quantum Feature Mapping.

Module: src.features.pca_reducer
Owner: Member A (Lead / Classical ML & Feature Engineering Lead)
Deliverable: D3 (Feature Selection & Dimensionality Reduction)
Task: T2.1
"""

from __future__ import annotations

from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class PCAReducer(BaseEstimator, TransformerMixin):
    """Automated PCA transformer reducing input dimensions to k in [4, 8] features.
    
    Rescales principal components to [-pi, pi] to map directly into quantum
    rotation gates (R_Y, R_Z) and parameterized quantum feature maps (AngleEmbedding,
    ZZFeatureMap) without hard clipping artifacts.
    
    Computes individual and cumulative explained variance to quantify quantum
    latent space information preservation.
    """

    def __init__(
        self,
        n_components: Union[int, str, float] = 4,
        scale_to_pi: bool = True,
        variance_threshold: float = 0.85,
        wrap_periodic: bool = False,
        random_state: int = 42,
    ):
        """Initialize PCAReducer.

        Args:
            n_components: Target number of principal components. Can be:
                - int: explicit number of components (typically k in [4, 8]).
                - 'auto': automatically selects k in [4, 8] satisfying variance_threshold.
                - float in (0, 1): treated as target variance ratio.
            scale_to_pi: If True, rescales principal components to [-pi, pi].
            variance_threshold: Minimum cumulative explained variance when n_components='auto'.
            wrap_periodic: If True, wraps out-of-distribution test points modulo 2*pi
                into [-pi, pi] preserving quantum gate rotational periodicity.
            random_state: Seed for deterministic PCA decomposition.
        """
        self.n_components = n_components
        self.scale_to_pi = scale_to_pi
        self.variance_threshold = variance_threshold
        self.wrap_periodic = wrap_periodic
        self.random_state = random_state

        self.k_effective_: int = 4 if not isinstance(n_components, int) else n_components
        self.pca: Optional[PCA] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.explained_variance_ratio_: np.ndarray = np.array([], dtype=np.float64)
        self.cumulative_explained_variance_: np.ndarray = np.array([], dtype=np.float64)
        self.feature_names_in_: Optional[List[str]] = None
        self.is_fitted_: bool = False

        # Member B & C compatibility attributes
        self.pca_: Optional[PCA] = None
        self.scaler_: Optional[MinMaxScaler] = None
        self.cumulative_variance_: float = 0.0

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> "PCAReducer":
        """Fit PCA and angle rescaler on training samples.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            y: Optional target labels (ignored, PCA is unsupervised).

        Returns:
            self: Fitted PCAReducer instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            self.feature_names_in_ = None

        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D input array, got shape {X_arr.shape}")

        n_samples, n_features = X_arr.shape
        max_possible_k = min(n_samples, n_features)

        # Determine effective k
        if self.n_components == "auto" or isinstance(self.n_components, float):
            threshold = (
                self.variance_threshold
                if self.n_components == "auto"
                else float(self.n_components)
            )
            # Fit full spectrum up to 8 components to find optimal k
            probe_k = min(8, max_possible_k)
            probe_pca = PCA(n_components=probe_k, random_state=self.random_state)
            probe_pca.fit(X_arr)
            cumsum = np.cumsum(probe_pca.explained_variance_ratio_)

            # Find smallest k reaching threshold, clamped between 4 and 8
            k_candidates = np.where(cumsum >= threshold)[0]
            if len(k_candidates) > 0:
                selected_k = int(k_candidates[0] + 1)
            else:
                selected_k = probe_k

            self.k_effective_ = max(4, min(selected_k, min(8, max_possible_k)))
        elif isinstance(self.n_components, int):
            if self.n_components < 1:
                raise ValueError(f"n_components must be >= 1, got {self.n_components}")
            self.k_effective_ = min(self.n_components, max_possible_k)
        else:
            raise ValueError(f"Unsupported n_components specification: {self.n_components}")

        # Fit final PCA on selected k_effective_
        self.pca = PCA(n_components=self.k_effective_, random_state=self.random_state)
        pca_out = self.pca.fit_transform(X_arr)

        self.explained_variance_ratio_ = np.asarray(self.pca.explained_variance_ratio_, dtype=np.float64)
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)

        # Fit MinMaxScaler to map exactly to [-pi, pi]
        if self.scale_to_pi:
            self.scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=not self.wrap_periodic)
            self.scaler.fit(pca_out)

        self.is_fitted_ = True

        # Synchronize compatibility attributes
        self.pca_ = self.pca
        self.scaler_ = self.scaler
        self.cumulative_variance_ = float(self.cumulative_explained_variance_[-1]) if len(self.cumulative_explained_variance_) > 0 else 0.0

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        as_frame: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Project features into principal component space and rescale to [-pi, pi].

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            as_frame: If True, returns a pandas DataFrame with columns ['PC1', 'PC2', ...].

        Returns:
            Projected coordinates of shape (n_samples, k) in [-pi, pi].
        """
        if not self.is_fitted_ or self.pca is None:
            raise RuntimeError("PCAReducer must be fitted before calling transform.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        pca_out = self.pca.transform(X_arr)

        if self.scale_to_pi and self.scaler is not None:
            if self.wrap_periodic:
                # Direct linear mapping without hard clipping, wrapped modulo 2*pi
                data_range = np.where(self.scaler.data_range_ == 0, 1.0, self.scaler.data_range_)
                norm = (pca_out - self.scaler.data_min_) / data_range
                scaled_raw = norm * (2.0 * np.pi) - np.pi
                scaled_out = (scaled_raw + np.pi) % (2.0 * np.pi) - np.pi
            else:
                scaled_out = self.scaler.transform(pca_out)
                scaled_out = np.clip(scaled_out, -np.pi, np.pi)
            final_out = scaled_out
        else:
            final_out = pca_out

        if as_frame:
            cols = [f"PC{i+1}" for i in range(self.k_effective_)]
            return pd.DataFrame(final_out, columns=cols)

        return final_out

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        as_frame: bool = False,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit to samples, then transform."""
        return self.fit(X, y).transform(X, as_frame=as_frame)

    def inverse_transform(self, X_transformed: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Invert the [-pi, pi] transformation and PCA projection back to original space.

        Args:
            X_transformed: Array of shape (n_samples, k) in [-pi, pi].

        Returns:
            Reconstructed feature matrix of shape (n_samples, n_features).
        """
        if not self.is_fitted_ or self.pca is None:
            raise RuntimeError("PCAReducer must be fitted before calling inverse_transform.")

        X_arr = np.asarray(X_transformed, dtype=np.float64)

        if self.scale_to_pi and self.scaler is not None:
            pca_space = self.scaler.inverse_transform(X_arr)
        else:
            pca_space = X_arr

        return self.pca.inverse_transform(pca_space)

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Return the explained variance ratio for each selected principal component."""
        if not self.is_fitted_:
            raise RuntimeError("PCAReducer must be fitted.")
        return np.copy(self.explained_variance_ratio_)

    def get_cumulative_explained_variance(self) -> np.ndarray:
        """Return array of cumulative explained variance across the k components."""
        if not self.is_fitted_:
            raise RuntimeError("PCAReducer must be fitted.")
        return np.copy(self.cumulative_explained_variance_)

    def get_total_explained_variance(self) -> float:
        """Return scalar cumulative variance retained by the k components."""
        if not self.is_fitted_:
            raise RuntimeError("PCAReducer must be fitted.")
        return float(self.cumulative_explained_variance_[-1])

    def get_variance_summary(self) -> Dict[str, Any]:
        """Returns explained variance details."""
        if not self.is_fitted_:
            raise RuntimeError("PCAReducer must be fitted.")
        return {
            "n_components": self.k_effective_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "total_explained_variance": float(self.cumulative_explained_variance_[-1]) if len(self.cumulative_explained_variance_) > 0 else 0.0,
        }
