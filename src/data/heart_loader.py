"""UCI Heart Disease (Cleveland) Dataset Loader and Preprocessor.

Module: src.data.heart_loader
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D2 (Data Ingestion)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union, Optional, List
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_UCI_HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DEFAULT_RAW_PATH = "data/raw/processed.cleveland.data"
DEFAULT_PROCESSED_PATH = "data/processed/heart_scaled.npy"

RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
NUMERICAL_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "thal"]
PASSTHROUGH_COLS = ["sex", "fbs", "exang", "ca"]

# Known categorical domain values from UCI Cleveland documentation
CATEGORICAL_CATEGORIES = [
    [1.0, 2.0, 3.0, 4.0],  # cp: 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic
    [0.0, 1.0, 2.0],        # restecg: 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy
    [1.0, 2.0, 3.0],        # slope: 1: upsloping, 2: flat, 3: downsloping
    [3.0, 6.0, 7.0]         # thal: 3: normal, 6: fixed defect, 7: reversable defect
]


class HeartDiseasePreprocessor(BaseEstimator, TransformerMixin):
    """Leak-free preprocessor pipeline for UCI Heart Disease Cleveland dataset.
    
    Imputes missing values with column medians, one-hot encodes categorical
    attributes, and standard-scales continuous features.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self.ohe = OneHotEncoder(
            categories=CATEGORICAL_CATEGORIES,
            handle_unknown="ignore",
            sparse_output=False
        )
        self.is_fitted_ = False
        self.feature_names_out_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> "HeartDiseasePreprocessor":
        """Fit imputer, scaler, and encoder on training data."""
        df = self._ensure_dataframe(X)
        
        # 1. Fit median imputer
        self.imputer.fit(df)
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)

        # 2. Fit one-hot encoder on categorical columns
        self.ohe.fit(df_imputed[CATEGORICAL_COLS])
        ohe_cols = self.ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()

        # 3. Fit scaler on numerical columns
        self.scaler.fit(df_imputed[NUMERICAL_COLS])

        # Feature order in output
        self.feature_names_out_ = NUMERICAL_COLS + PASSTHROUGH_COLS + ohe_cols
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Transform data using fitted transformers."""
        if not self.is_fitted_:
            raise RuntimeError("HeartDiseasePreprocessor must be fitted before calling transform.")
        
        df = self._ensure_dataframe(X)
        df_imputed = pd.DataFrame(self.imputer.transform(df), columns=df.columns)

        # Scaled numerical
        scaled_num = self.scaler.transform(df_imputed[NUMERICAL_COLS])
        # Passthrough
        pass_arr = df_imputed[PASSTHROUGH_COLS].to_numpy(dtype=np.float64)
        # One-hot encoded
        cat_arr = self.ohe.transform(df_imputed[CATEGORICAL_COLS])

        # Concatenate horizontally
        X_out = np.hstack([scaled_num, pass_arr, cat_arr])
        return X_out

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return names of output features."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor not fitted.")
        return list(self.feature_names_out_)

    def _ensure_dataframe(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        feature_cols = [c for c in RAW_COLUMNS if c != "target"]
        if isinstance(X, pd.DataFrame):
            # Check if columns match or subset matches
            if set(feature_cols).issubset(set(X.columns)):
                return X[feature_cols].copy()
            elif X.shape[1] == len(feature_cols):
                df_copy = X.copy()
                df_copy.columns = feature_cols
                return df_copy
            else:
                raise ValueError(f"Expected DataFrame with {len(feature_cols)} features, got {X.shape[1]}")
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.shape[1] != len(feature_cols):
                raise ValueError(f"Expected array with {len(feature_cols)} columns, got {arr.shape[1]}")
            return pd.DataFrame(arr, columns=feature_cols)


def ensure_raw_data_downloaded(raw_path: str = DEFAULT_RAW_PATH) -> Path:
    """Download raw UCI Cleveland data if not already present."""
    path = Path(raw_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_UCI_HEART_URL, str(path))
    return path


def load_heart_disease(
    raw_path: Optional[str] = None,
    as_frame: bool = False,
    preprocess: bool = True,
    return_feature_names: bool = False,
    cache_processed: bool = True,
    cache_path: str = DEFAULT_PROCESSED_PATH,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[pd.DataFrame, pd.Series], Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series], List[str]]]:
    """Load, parse, impute, and preprocess UCI Heart Disease dataset.

    Steps:
    1. Parse raw data handling missing values ('?' markers in ca and thal columns).
    2. Impute missing values using MedianImputer.
    3. One-hot encode categorical features (cp, restecg, slope, thal).
    4. Binary encode target (disease presence: >0 -> 1, 0 -> 0).
    5. StandardScale numerical columns (age, trestbps, chol, thalach, oldpeak).

    Returns:
        X, y: (303, n_features) without NaNs, and binary labels {0, 1}.
    """
    raw_file = Path(raw_path) if raw_path else ensure_raw_data_downloaded()
    if not raw_file.exists():
        ensure_raw_data_downloaded(str(raw_file))

    df_raw = pd.read_csv(
        raw_file,
        header=None,
        names=RAW_COLUMNS,
        na_values=["?", "NA", "nan", "NaN"]
    )

    # 4. Binary encode target (>0 -> 1, 0 -> 0)
    y_binary = (df_raw["target"].to_numpy(dtype=np.float64) > 0).astype(np.int64)
    X_features_df = df_raw.drop(columns=["target"])

    if preprocess:
        preprocessor = HeartDiseasePreprocessor()
        X_proc = preprocessor.fit_transform(X_features_df)
        feature_names = preprocessor.get_feature_names_out()

        if cache_processed:
            out_file = Path(cache_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_file, X_proc)
        
        X_out_arr = X_proc
    else:
        # Impute only if raw features requested
        imputer = SimpleImputer(strategy="median")
        X_out_arr = imputer.fit_transform(X_features_df)
        feature_names = [c for c in RAW_COLUMNS if c != "target"]

    # Verify no NaNs
    if np.isnan(X_out_arr).any():
        raise ValueError("Heart disease preprocessed features still contain NaNs.")

    if as_frame:
        X_out = pd.DataFrame(X_out_arr, columns=feature_names)
        y_out = pd.Series(y_binary, name="target")
    else:
        X_out = X_out_arr
        y_out = y_binary

    if return_feature_names:
        return X_out, y_out, feature_names
    return X_out, y_out
