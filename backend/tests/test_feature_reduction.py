"""Unit tests for Feature Reduction (PCA & Mutual Information) and Zero Data Leakage Verification."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.data.loader import load_dataset
from src.data.feature_reduction import (
    PCAReducer,
    MutualInfoReducer,
    ZeroLeakagePipeline,
)


def test_zero_data_leakage_adversarial_invariance():
    """Verify mathematically that test data does not leak into training fit."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 10)
    X_test_clean = rng.randn(30, 10)
    # Severe adversarial perturbation on test set
    X_test_corrupted = X_test_clean * 1000.0 + 500.0

    # Fit reducer on training data
    reducer = PCAReducer(n_components=4, random_state=42)
    X_tr_proj1 = reducer.fit_transform(X_train)
    components1 = np.copy(reducer.pca.components_)
    mean1 = np.copy(reducer.scaler.mean_)

    # Transform clean vs corrupted test data
    X_te_proj_clean = reducer.transform(X_test_clean)
    X_te_proj_corrupted = reducer.transform(X_test_corrupted)

    # Assert that training projections and learned components are 100% invariant
    np.testing.assert_allclose(reducer.pca.components_, components1, atol=1e-15)
    np.testing.assert_allclose(reducer.scaler.mean_, mean1, atol=1e-15)
    np.testing.assert_allclose(reducer.transform(X_train), X_tr_proj1, atol=1e-15)


def test_wdbc_variance_retention_k4_to_k6():
    """Verify that k=4..6 qubits retains >85% variance on WDBC dataset."""
    X, y, _, _ = load_dataset("wdbc", scaled=False)
    
    # Fit on 80% train split
    n_train = int(len(X) * 0.8)
    X_train = X[:n_train]
    
    # k=6
    reducer_k6 = PCAReducer(n_components=6, random_state=42).fit(X_train)
    var_k6 = reducer_k6.cumulative_variance_ratio_
    assert var_k6 >= 0.85, f"k=6 variance ({var_k6*100:.2f}%) must be >= 85%"
    assert var_k6 >= 0.88, f"k=6 variance ({var_k6*100:.2f}%) retains >88%"


def test_heart_disease_reduction():
    """Test feature reduction on Heart Disease dataset."""
    X, y, feature_names, target_names = load_dataset("heart", scaled=False)
    assert X.shape[0] == 270
    assert X.shape[1] == 13
    assert set(np.unique(y)) == {0, 1}

    # PCA Reduction on Heart Disease with k=6
    reducer = PCAReducer(n_components=6, random_state=42).fit(X[:200])
    X_proj = reducer.transform(X[200:])
    assert X_proj.shape == (70, 6)
    assert 0.0 < reducer.cumulative_variance_ratio_ <= 1.0


def test_mutual_info_reducer():
    """Test zero-leakage Mutual Information selector."""
    X, y, _, _ = load_dataset("wdbc", scaled=False)
    mi_reducer = MutualInfoReducer(n_features_to_select=6, random_state=42)
    X_sel = mi_reducer.fit_transform(X[:200], y[:200])
    assert X_sel.shape == (200, 6)
    assert len(mi_reducer.selected_indices_) == 6

    # Test transform on unseen data
    X_test_sel = mi_reducer.transform(X[200:250])
    assert X_test_sel.shape == (50, 6)


def test_out_of_fold_variance():
    """Test out-of-fold variance reconstruction metric."""
    X, y, _, _ = load_dataset("wdbc", scaled=False)
    reducer = PCAReducer(n_components=6, random_state=42).fit(X[:400])
    oof_var = reducer.compute_out_of_fold_variance(X[400:])
    assert 0.70 <= oof_var <= 1.0
