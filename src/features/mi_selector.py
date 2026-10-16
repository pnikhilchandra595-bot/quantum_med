"""Mutual Information Feature Selector for Biomedical Datasets and Quantum Circuits.

Module: src.features.mi_selector
Owner: Member A (Lead / Classical ML & Feature Engineering Lead)
Deliverable: D3 (Feature Selection)
Task: T2.2
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler


class MISelector(BaseEstimator, TransformerMixin):
    """Ranks and selects top k biomedical features using Mutual Information.
    
    Preserves original clinical feature names (e.g., 'mean radius', 'cholesterol')
    for downstream SHAP explainability audits by Member C, and rescales selected
    features to [-pi, pi] to feed quantum rotation gates (R_Y, R_Z).
    """

    def __init__(
        self,
        k: int = 4,
        scale_to_pi: bool = True,
        feature_range: Tuple[float, float] = (-np.pi, np.pi),
        scale_to_2pi: Optional[bool] = None,
        random_state: int = 42,
        n_neighbors: int = 3,
        return_dataframe: bool = False,
    ):
        """Initialize MISelector.

        Args:
            k: Target number of biomedical features to select (typically k in [4, 8]).
            scale_to_pi: If True and feature_range not explicitly altered, scales to [-pi, pi].
            feature_range: Target numerical bounds for scaling, defaults to [-pi, pi].
            scale_to_2pi: Optional backwards-compatibility flag. If explicitly True,
                sets feature_range to (0.0, 2*pi).
            random_state: Seed for reproducible mutual information estimation.
            n_neighbors: Number of nearest neighbors for continuous KSG entropy estimation.
            return_dataframe: If True, transform returns a pd.DataFrame with clinical
                column names.
        """
        self.k = k
        self.scale_to_pi = scale_to_pi
        self.feature_range = feature_range
        self.scale_to_2pi = scale_to_2pi
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.return_dataframe = return_dataframe

        # Handle backward-compatibility for scale_to_2pi
        if self.scale_to_2pi is True:
            self.effective_feature_range_: Tuple[float, float] = (0.0, 2.0 * np.pi)
        else:
            self.effective_feature_range_ = self.feature_range

        self.selected_indices_: List[int] = []
        self.selected_feature_names_: List[str] = []
        self.feature_names_in_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}

        self.scaler: Optional[MinMaxScaler] = None
        self.is_fitted_: bool = False

        # Compatibility attributes
        self.scores_: Optional[np.ndarray] = None
        self.selected_indices: Optional[List[int]] = None
        self.scaler_: Optional[MinMaxScaler] = None

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
    ) -> "MISelector":
        """Compute Mutual Information scores between features and diagnosis labels.

        Args:
            X: Input biomedical feature matrix of shape (n_samples, n_features).
            y: Binary target diagnosis vector of shape (n_samples,).
            feature_names: Optional explicit list of feature names if X is a NumPy array.

        Returns:
            self: Fitted MISelector instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if feature_names is not None:
                self.feature_names_in_ = list(feature_names)
            else:
                self.feature_names_in_ = [f"Feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.int64).ravel()

        if len(self.feature_names_in_) != X_arr.shape[1]:
            raise ValueError("Length of feature_names must match number of columns in X.")

        n_samples, n_features = X_arr.shape
        actual_k = min(self.k, n_features)

        # Compute mutual information scores using KSG continuous estimator
        scores = mutual_info_classif(
            X_arr,
            y_arr,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        self.scores_ = scores
        self.feature_scores_ = {name: float(scores[i]) for i, name in enumerate(self.feature_names_in_)}

        # Rank features descending by mutual information
        ranked_indices = np.argsort(scores)[::-1]
        self.selected_indices_ = [int(idx) for idx in ranked_indices[:actual_k]]
        self.selected_indices = self.selected_indices_
        self.selected_feature_names_ = [self.feature_names_in_[i] for i in self.selected_indices_]

        # Fit MinMaxScaler if scaling is enabled
        if self.scale_to_pi or self.effective_feature_range_ is not None:
            X_subset = X_arr[:, self.selected_indices_]
            self.scaler = MinMaxScaler(feature_range=self.effective_feature_range_)
            self.scaler.fit(X_subset)
            self.scaler_ = self.scaler

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        as_frame: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Extract top k features and rescale into quantum feature angle range.

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            as_frame: If True, returns a pandas DataFrame with original clinical feature names.

        Returns:
            Transformed feature matrix of shape (n_samples, k).
        """
        if not self.is_fitted_ or not self.selected_indices_:
            raise RuntimeError("MISelector must be fitted before calling transform.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_selected = X_arr[:, self.selected_indices_]

        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_selected)
            low, high = self.effective_feature_range_
            X_scaled = np.clip(X_scaled, low, high)
        else:
            X_scaled = X_selected

        return_as_df = as_frame if as_frame is not None else self.return_dataframe
        if return_as_df:
            return pd.DataFrame(X_scaled, columns=self.selected_feature_names_)

        return X_scaled

    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        as_frame: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Fit on X and y, then transform X."""
        return self.fit(X, y, feature_names=feature_names).transform(X, as_frame=as_frame)

    def inverse_transform(self, X_transformed: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Invert the scaling on selected features back to original measurement units."""
        if not self.is_fitted_ or self.scaler is None:
            raise RuntimeError("MISelector must be fitted before calling inverse_transform.")

        X_arr = np.asarray(X_transformed, dtype=np.float64)
        return self.scaler.inverse_transform(X_arr)

    def get_selected_feature_names(self) -> List[str]:
        """Return clinical names of top k selected features for SHAP explainability."""
        if not self.is_fitted_:
            raise RuntimeError("MISelector not fitted.")
        return list(self.selected_feature_names_)

    def get_selected_indices(self) -> List[int]:
        """Return original column indices of top k selected features."""
        if not self.is_fitted_:
            raise RuntimeError("MISelector not fitted.")
        return list(self.selected_indices_)

    def get_feature_scores(self) -> Dict[str, float]:
        """Return dictionary mapping all feature names to their Mutual Information score."""
        if not self.is_fitted_:
            raise RuntimeError("MISelector not fitted.")
        return dict(self.feature_scores_)

    def get_ranked_features_df(self) -> pd.DataFrame:
        """Return a structured ranking DataFrame of all features with selection flags."""
        if not self.is_fitted_:
            raise RuntimeError("MISelector not fitted.")

        sorted_items = sorted(self.feature_scores_.items(), key=lambda x: x[1], reverse=True)
        rows = []
        for rank, (name, score) in enumerate(sorted_items, start=1):
            rows.append(
                {
                    "rank": rank,
                    "feature_name": name,
                    "mutual_info_nats": score,
                    "selected": name in self.selected_feature_names_,
                }
            )
        return pd.DataFrame(rows)

    def get_ranking_summary(self) -> List[Dict[str, Any]]:
        """Returns sorted list of features with their mutual information scores."""
        summary = []
        for rank, idx in enumerate(self.selected_indices_):
            summary.append({
                "rank": rank + 1,
                "feature_index": int(idx),
                "feature_name": self.selected_feature_names_[rank]
                if rank < len(self.selected_feature_names_) else f"Col_{idx}",
                "mi_score": float(self.scores_[idx]) if self.scores_ is not None else 0.0,
            })
        return summary
