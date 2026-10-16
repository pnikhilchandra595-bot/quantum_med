"""Wisconsin Diagnostic Breast Cancer (WDBC) Dataset Loader and Preprocessor.

Module: src.data.wdbc_loader
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D2 (Data Ingestion)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union, Optional, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class WDBCPreprocessor(BaseEstimator, TransformerMixin):
    """Leak-free preprocessor pipeline for WDBC dataset.
    
    Fits scaler exclusively on training partition and transforms test/validation
    partitions without data leakage.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "WDBCPreprocessor":
        """Fit StandardScaler on training samples."""
        X_arr = np.asarray(X, dtype=np.float64)
        self.scaler.fit(X_arr)
        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform samples using fitted parameters."""
        if not self.is_fitted_:
            raise RuntimeError("WDBCPreprocessor must be fitted before transforming data.")
        X_arr = np.asarray(X, dtype=np.float64)
        return self.scaler.transform(X_arr)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit on X and transform X."""
        return self.fit(X, y).transform(X)


def load_wdbc(
    as_frame: bool = False,
    return_feature_names: bool = False,
    scale: bool = False,
    cache_processed: bool = True,
    cache_path: str = "data/processed/wdbc_scaled.npy",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series], Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series], List[str]]]:
    """Load and format the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

    Steps:
    1. Fetch WDBC from sklearn.datasets.load_breast_cancer (standard WDBC repository source).
    2. Convert 30 continuous features into float64 NumPy array or Pandas DataFrame.
    3. Binary encode targets: Malignant (M) -> 1, Benign (B) -> 0.
       Note: sklearn assigns 0=malignant, 1=benign, so we map 0 -> 1 and 1 -> 0.
    4. Optionally fit StandardScaler and verify zero mean (~0) and unit variance (~1).
    5. Save processed data cache to data/processed/wdbc_scaled.npy.

    Args:
        as_frame: If True, returns pandas DataFrame and Series. Otherwise returns NumPy arrays.
        return_feature_names: If True, also returns list of 30 feature names.
        scale: If True, returns standard-scaled features. (Note: for CV, use WDBCPreprocessor on folds).
        cache_processed: If True, saves scaled array to cache_path.
        cache_path: Destination path for scaled array cache.

    Returns:
        (X, y) or (X, y, feature_names) where X has shape (569, 30) and y has values {0, 1}.
    """
    raw_data = load_breast_cancer()
    X_raw: np.ndarray = np.asarray(raw_data.data, dtype=np.float64)
    # sklearn target: 0 = 'malignant', 1 = 'benign'
    # Project requirement: Malignant (M) -> 1, Benign (B) -> 0
    y_raw: np.ndarray = (raw_data.target == 0).astype(np.int64)
    feature_names: List[str] = [str(f) for f in raw_data.feature_names]

    # Verify no missing values
    if np.isnan(X_raw).any() or np.isnan(y_raw).any():
        raise ValueError("WDBC raw dataset contains unexpected NaN values.")

    if scale or cache_processed:
        preprocessor = WDBCPreprocessor()
        X_scaled = preprocessor.fit_transform(X_raw)

        # Verify zero mean and unit variance
        means = np.mean(X_scaled, axis=0)
        stds = np.std(X_scaled, axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-7, err_msg="StandardScaler did not achieve zero mean.")
        np.testing.assert_allclose(stds, 1.0, atol=1e-7, err_msg="StandardScaler did not achieve unit variance.")

        if cache_processed:
            out_file = Path(cache_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_file, X_scaled)

    X_final = X_scaled if scale else X_raw

    if as_frame:
        X_out = pd.DataFrame(X_final, columns=feature_names)
        y_out = pd.Series(y_raw, name="diagnosis")
    else:
        X_out = X_final
        y_out = y_raw

    if return_feature_names:
        return X_out, y_out, feature_names
    return X_out, y_out
