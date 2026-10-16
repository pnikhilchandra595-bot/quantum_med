"""Clinical Dataset Loader and Preprocessing.

Supports:
1. WDBC (Wisconsin Diagnostic Breast Cancer): 569 samples, 30 features.
   - 1 = Malignant (positive condition to detect), 0 = Benign.
2. Heart Disease (Statlog / Cleveland): 270 samples, 13 features.
   - 1 = Heart Disease Present (positive condition), 0 = Absent.

Supports both raw unscaled data (mandatory for zero-leakage cross-validation)
and pre-scaled data for convenient quick evaluation.
"""

from typing import Tuple, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import StandardScaler


def load_wdbc_dataset(
    scaled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

    Args:
        scaled: If True, applies global StandardScaler (use False for zero-leakage CV).

    Returns:
        X: Feature matrix of shape (569, 30).
        y: Binary target array (569,), where 1 = Malignant, 0 = Benign.
        feature_names: List of 30 feature names.
        target_names: ['Benign', 'Malignant'].
    """
    bunch = load_breast_cancer()
    X = bunch.data
    y_raw = bunch.target
    feature_names = list(bunch.feature_names)

    # Standard clinical convention: 1 = Malignant, 0 = Benign
    y_clinical = np.where(y_raw == 0, 1, 0)
    target_names = ["Benign", "Malignant"]

    if scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y_clinical, feature_names, target_names


def load_heart_disease_dataset(
    scaled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load Heart Disease (Statlog / Cleveland) dataset.

    Args:
        scaled: If True, applies global StandardScaler (use False for zero-leakage CV).

    Returns:
        X: Feature matrix of shape (270, 13).
        y: Binary target array (270,), where 1 = Heart Disease Present, 0 = Absent.
        feature_names: List of 13 feature names.
        target_names: ['Absent', 'Present'].
    """
    csv_path = Path("data/heart_disease.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "target"]
        X = df[feature_cols].values.astype(float)
        y = df["target"].values.astype(int)
        feature_names = feature_cols
    else:
        # Fallback to OpenML if local cache is absent
        ds = fetch_openml(name="heart-statlog", as_frame=True)
        df = ds.data
        feature_names = list(df.columns)
        X = df.values.astype(float)
        y = ds.target.map({"absent": 0, "present": 1}).values.astype(int)
        # Cache locally
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_save = df.copy()
        df_save["target"] = y
        df_save.to_csv(csv_path, index=False)

    target_names = ["Absent", "Present"]

    if scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_names, target_names


def load_dataset(
    name: str = "wdbc",
    scaled: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Universal dataset loader supporting 'wdbc' and 'heart' ('heart_disease')."""
    name_clean = name.strip().lower()
    if name_clean in ["wdbc", "breast_cancer", "cancer"]:
        return load_wdbc_dataset(scaled=scaled)
    elif name_clean in ["heart", "heart_disease", "heart-statlog", "cleveland"]:
        return load_heart_disease_dataset(scaled=scaled)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported datasets: 'wdbc', 'heart'."
        )
