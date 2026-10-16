"""Clinical Dataset Loader and Preprocessing."""

from typing import Tuple, List, Optional
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def load_wdbc_dataset() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Load Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    
    Standard clinical convention applied:
    - Positive Class (1): Malignant (critical cancer detection class)
    - Negative Class (0): Benign
    
    Features are scaled using StandardScaler.
    
    Returns:
        X_scaled: Preprocessed feature matrix of shape (569, 30).
        y_clinical: Binary target array of shape (569,), where 1 = Malignant, 0 = Benign.
        feature_names: List of 30 feature names.
        target_names: ['Benign', 'Malignant'].
    """
    bunch = load_breast_cancer()
    X = bunch.data
    y_raw = bunch.target
    feature_names = list(bunch.feature_names)
    
    # In sklearn: 0 = malignant, 1 = benign.
    # In clinical oncology evaluation: 1 = Malignant (positive condition to detect), 0 = Benign.
    y_clinical = np.where(y_raw == 0, 1, 0)
    target_names = ["Benign", "Malignant"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y_clinical, feature_names, target_names


def load_dataset(name: str = "wdbc") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Universal dataset loader supporting 'wdbc'."""
    name_clean = name.strip().lower()
    if name_clean in ["wdbc", "breast_cancer", "cancer"]:
        return load_wdbc_dataset()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: 'wdbc'.")
