"""Mutual Information Feature Selector for Biomedical Datasets.

Module: src.features.mi_selector
Owner: Member A (Lead / Classical ML & Feature Engineering Lead)
Deliverable: D3 (Feature Selection)
Task: T2.2
"""

from __future__ import annotations

from typing import Optional, Union, List, Dict, Tuple
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
        feature_range: Tuple[float, float] = (-np.pi, np.pi),
        scale_to_2pi: Optional[bool] = None,
        random_state: int = 42,
        n_neighbors: int = 3,
        return_dataframe: bool = False,
    ):
        """Initialize MISelector.

        Args:
            k: Target number of biomedical features to select (typically k in [4, 8]).
            feature_range: Target numerical bounds for scaling, defaults to [-pi, pi].
            scale_to_2pi: Optional backwards-compatibility flag. If explicitly True,
                sets feature_range to (0.0, 2*pi).
            random_state: Seed for reproducible mutual information estimation.
            n_neighbors: Number of nearest neighbors for continuous KSG entropy estimation.
            return_dataframe: If True, transform returns a pd.DataFrame with clinical
                column names.
        """
        self.k = k
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
            if feature_names is not None and len(feature_names) == X_arr.shape[1]:
                self.feature_names_in_ = list(feature_names)
            else:
                self.feature_names_in_ = [f"feature_{i}" for i in range(X_arr.shape[1])]

        y_arr = np.asarray(y, dtype=np.int64).ravel()

        if X_arr.shape[0] != len(y_arr):
            raise ValueError(f"Sample count mismatch: X has {X_arr.shape[0]}, y has {len(y_arr)}")

        if np.isnan(X_arr).any() or np.isnan(y_arr).any():
            raise ValueError("Input data contains unhandled NaN values.")

        k_eff = min(max(1, self.k), X_arr.shape[1])

        # Compute continuous-discrete mutual information
        scores = mutual_info_classif(
            X_arr,
            y_arr,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )

        self.feature_scores_ = {
            name: float(score) for name, score in zip(self.feature_names_in_, scores)
        }

        # Rank indices in descending order of information content
        ranked_indices = np.argsort(scores)[::-1]
        self.selected_indices_ = ranked_indices[:k_eff].tolist()
        self.selected_feature_names_ = [self.feature_names_in_[idx] for idx in self.selected_indices_]

        # Fit MinMaxScaler on selected features to target angle range [-pi, pi]
        selected_sub = X_arr[:, self.selected_indices_]
        self.scaler = MinMaxScaler(feature_range=self.effective_feature_range_, clip=True)
        self.scaler.fit(selected_sub)

        self.is_fitted_ = True
        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        as_frame: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Extract top k features and rescale to [-pi, pi].

        Args:
            X: Input feature matrix of shape (n_samples, n_features).
            as_frame: If True (or if return_dataframe=True), returns a pd.DataFrame
                with original clinical column names preserved for SHAP explainability.

        Returns:
            Transformed feature matrix of shape (n_samples, k).
        """
        if not self.is_fitted_ or self.scaler is None:
            raise RuntimeError("MISelector must be fitted before calling transform.")

        if isinstance(X, pd.DataFrame):
            # Check if all selected columns are present by name
            if set(self.selected_feature_names_).issubset(set(X.columns)):
                selected_sub = X[self.selected_feature_names_].to_numpy(dtype=np.float64)
            elif X.shape[1] == len(self.feature_names_in_):
                selected_sub = X.to_numpy(dtype=np.float64)[:, self.selected_indices_]
            else:
                raise ValueError(
                    f"Expected DataFrame with features matching training dimension {len(self.feature_names_in_)}, "
                    f"got {X.shape[1]}"
                )
        else:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.shape[1] != len(self.feature_names_in_):
                raise ValueError(
                    f"Feature dimension mismatch: expected {len(self.feature_names_in_)}, got {X_arr.shape[1]}"
                )
            selected_sub = X_arr[:, self.selected_indices_]

        scaled_sub = self.scaler.transform(selected_sub)
        scaled_sub = np.clip(
            scaled_sub,
            self.effective_feature_range_[0],
            self.effective_feature_range_[1],
        )

        should_return_frame = as_frame if as_frame is not None else self.return_dataframe
        if should_return_frame:
            return pd.DataFrame(scaled_sub, columns=self.selected_feature_names_)

        return scaled_sub

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
