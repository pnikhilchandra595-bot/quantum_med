"""
Data Loader and Preprocessing Pipeline for WDBC and Heart Disease Datasets.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

AVAILABLE_DATASETS = ["wdbc", "heart"]


def generate_synthetic_heart_disease(n_samples: int = 303, random_state: int = 42) -> pd.DataFrame:
    """
    Generate realistic clinical Cleveland Heart Disease dataset when offline.
    Follows UCI Heart Disease distribution parameters.
    """
    rng = np.random.RandomState(random_state)

    age = rng.normal(loc=54.4, scale=9.0, size=n_samples).clip(29, 77).astype(int)
    sex = rng.binomial(n=1, p=0.68, size=n_samples)
    cp = rng.choice([0, 1, 2, 3], p=[0.47, 0.17, 0.28, 0.08], size=n_samples)
    trestbps = rng.normal(loc=131.6, scale=17.5, size=n_samples).clip(94, 200).astype(int)
    chol = rng.normal(loc=246.3, scale=51.8, size=n_samples).clip(126, 564).astype(int)
    fbs = rng.binomial(n=1, p=0.15, size=n_samples)
    restecg = rng.choice([0, 1, 2], p=[0.49, 0.48, 0.03], size=n_samples)
    thalach = rng.normal(loc=149.6, scale=22.9, size=n_samples).clip(71, 202).astype(int)
    exang = rng.binomial(n=1, p=0.33, size=n_samples)
    oldpeak = rng.exponential(scale=1.04, size=n_samples).clip(0.0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], p=[0.07, 0.46, 0.47], size=n_samples)
    ca = rng.choice([0, 1, 2, 3], p=[0.58, 0.22, 0.13, 0.07], size=n_samples)
    thal = rng.choice([1, 2, 3], p=[0.06, 0.55, 0.39], size=n_samples)

    # Calculate physiological risk log-odds
    log_odds = (
        -4.2
        + 0.04 * (age - 50)
        + 0.8 * sex
        + 0.7 * cp
        + 0.015 * (trestbps - 120)
        + 0.008 * (chol - 200)
        + 0.6 * fbs
        - 0.03 * (thalach - 150)
        + 1.2 * exang
        + 0.7 * oldpeak
        + 0.6 * ca
        + 0.5 * (thal == 3)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    target = (rng.uniform(size=n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target,
        }
    )
    return df


def get_raw_dataframe(name: str = "wdbc") -> pd.DataFrame:
    """
    Retrieve raw pandas DataFrame for exploration and profiling.
    """
    name_clean = name.lower().strip()
    if name_clean in ["wdbc", "breast_cancer", "cancer"]:
        bunch = load_breast_cancer(as_frame=True)
        df = bunch.frame.copy()
        # Ensure target column is named 'target'
        if "target" not in df.columns and bunch.target_names is not None:
            df["target"] = bunch.target
        return df
    elif name_clean in ["heart", "heart_disease", "cardiac"]:
        return generate_synthetic_heart_disease()
    else:
        raise ValueError(f"Unknown dataset '{name}'. Available: {AVAILABLE_DATASETS}")


def get_dataset_metadata(name: str = "wdbc") -> Dict[str, Any]:
    """
    Return dataset metadata and clinical descriptions.
    """
    name_clean = name.lower().strip()
    if name_clean in ["wdbc", "breast_cancer"]:
        return {
            "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "samples": 569,
            "raw_features": 30,
            "classes": {0: "Malignant", 1: "Benign"},
            "description": "Nuclear features computed from fine-needle aspirate biopsies of breast masses.",
            "target_metric": "ROC-AUC / Sensitivity",
            "clinical_relevance": "High sensitivity crucial to avoid false negatives in malignant detection.",
        }
    elif name_clean in ["heart", "heart_disease"]:
        return {
            "name": "UCI Heart Disease (Cleveland Clinical)",
            "samples": 303,
            "raw_features": 13,
            "classes": {0: "Healthy / No Disease", 1: "Presence of Heart Disease"},
            "description": "Patient biomarkers measuring cardiovascular health and angiographic disease status.",
            "target_metric": "F1-Score / Specificity",
            "clinical_relevance": "Accurate risk stratification for preventative intervention.",
        }
    raise ValueError(f"Unknown dataset '{name}'. Available: {AVAILABLE_DATASETS}")


def load_dataset(
    name: str = "wdbc",
    n_qubits: Optional[int] = 4,
    reduction: str = "pca",
    scaler_type: str = "minmax",
    angle_range: Tuple[float, float] = (-np.pi, np.pi),
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Load, preprocess, reduce, and split a dataset for hybrid quantum-classical ML.

    Returns dictionary with:
        - X_train, X_test, y_train, y_test
        - feature_names
        - reducer (fitted PCA or SelectKBest)
        - scaler (fitted Scaler)
        - metadata
    """
    df = get_raw_dataframe(name)
    target_col = "target"

    feature_cols = [c for c in df.columns if c != target_col]
    X_raw = df[feature_cols].values
    y = df[target_col].values.astype(int)

    # Train-test split before scaling to prevent leakage
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pre-scaling
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=angle_range)
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=(0, 1))

    X_tr_scaled = scaler.fit_transform(X_tr_raw)
    X_te_scaled = scaler.transform(X_te_raw)

    reducer = None
    if n_qubits is not None and n_qubits < X_raw.shape[1]:
        if reduction.lower() == "pca":
            reducer = PCA(n_components=n_qubits, random_state=random_state)
            X_train = reducer.fit_transform(X_tr_scaled)
            X_test = reducer.transform(X_te_scaled)
            feature_names = [f"PC_{i+1}" for i in range(n_qubits)]
        elif reduction.lower() in ["mi", "mutual_info"]:
            reducer = SelectKBest(score_func=mutual_info_classif, k=n_qubits)
            X_train = reducer.fit_transform(X_tr_scaled, y_train)
            X_test = reducer.transform(X_te_scaled)
            mask = reducer.get_support()
            feature_names = [feature_cols[i] for i, val in enumerate(mask) if val]
        else:
            X_train = X_tr_scaled[:, :n_qubits]
            X_test = X_te_scaled[:, :n_qubits]
            feature_names = feature_cols[:n_qubits]
    else:
        X_train = X_tr_scaled
        X_test = X_te_scaled
        feature_names = feature_cols

    # Re-normalize reduced components to angle range for quantum embedding
    if scaler_type == "minmax" and reducer is not None:
        q_scaler = MinMaxScaler(feature_range=angle_range)
        X_train = q_scaler.fit_transform(X_train)
        X_test = q_scaler.transform(X_test)
    else:
        q_scaler = None

    return {
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "y_test": np.asarray(y_test, dtype=np.int64),
        "feature_names": feature_names,
        "raw_features": feature_cols,
        "scaler": scaler,
        "quantum_scaler": q_scaler,
        "reducer": reducer,
        "dataset_name": name,
        "metadata": get_dataset_metadata(name),
    }
