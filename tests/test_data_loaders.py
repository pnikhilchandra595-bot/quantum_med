"""Unit and integration tests for WDBC, Heart Disease data loaders and splitters.

Test File: tests/test_data_loaders.py
"""

import numpy as np
import pytest
from pathlib import Path

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease, HeartDiseasePreprocessor
from src.data.splitter import get_folds, generate_stratified_folds


def test_load_wdbc_shape_and_labels():
    """Verify WDBC dataset dimensions and binary label set."""
    X, y = load_wdbc(as_frame=False, scale=False)
    assert X.shape == (569, 30), f"Expected shape (569, 30), got {X.shape}"
    assert set(np.unique(y)) == {0, 1}, f"Expected binary targets {{0, 1}}, got {set(np.unique(y))}"
    # Malignant count in WDBC is 212, Benign count is 357
    assert np.sum(y == 1) == 212, "Expected 212 malignant samples"
    assert np.sum(y == 0) == 357, "Expected 357 benign samples"


def test_load_wdbc_dataframe_and_feature_names():
    """Verify DataFrame output and feature naming."""
    df, y, feature_names = load_wdbc(as_frame=True, return_feature_names=True)
    assert df.shape == (569, 30)
    assert len(feature_names) == 30
    assert "mean radius" in feature_names
    assert not df.isna().any().any()


def test_wdbc_preprocessor_no_leakage():
    """Verify WDBCPreprocessor fits exclusively on train set."""
    X, y = load_wdbc()
    X_train, X_val = X[:400], X[400:]
    preprocessor = WDBCPreprocessor()
    X_train_s = preprocessor.fit_transform(X_train)
    X_val_s = preprocessor.transform(X_val)

    # Train set should have exact 0 mean and 1 variance
    np.testing.assert_allclose(np.mean(X_train_s, axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.std(X_train_s, axis=0), 1.0, atol=1e-7)
    # Val set scaled by train params should not throw
    assert X_val_s.shape == (169, 30)
    assert not np.isnan(X_val_s).any()


def test_load_heart_disease_shape_and_cleanliness():
    """Verify UCI Heart Disease Cleveland dataset loading and imputation."""
    X, y = load_heart_disease(as_frame=False, preprocess=True)
    assert X.shape[0] == 303, f"Expected 303 samples, got {X.shape[0]}"
    assert not np.isnan(X).any(), "Preprocessed Heart dataset contains NaNs"
    assert set(np.unique(y)) == {0, 1}, f"Expected binary labels {{0, 1}}, got {set(np.unique(y))}"
    # 139 disease positive, 164 negative in Cleveland dataset
    assert np.sum(y == 1) == 139
    assert np.sum(y == 0) == 164


def test_heart_preprocessor_pipeline():
    """Verify HeartDiseasePreprocessor handles raw input dictionary or array."""
    X_raw, y = load_heart_disease(preprocess=False)
    preproc = HeartDiseasePreprocessor()
    X_proc = preproc.fit_transform(X_raw)
    assert X_proc.shape[0] == 303
    assert not np.isnan(X_proc).any()
    names = preproc.get_feature_names_out()
    assert len(names) == X_proc.shape[1]


def test_stratified_folds_wdbc():
    """Verify 5-fold stratification for WDBC dataset."""
    folds = get_folds("wdbc")
    assert len(folds) == 5

    X, y = load_wdbc()
    global_pos_ratio = np.mean(y == 1)

    all_val_indices = []
    for fold in folds:
        train_idx, val_idx = fold
        # Disjoint indices
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

        # Class ratio preservation within 3% tolerance
        val_pos_ratio = np.mean(y[val_idx] == 1)
        assert abs(val_pos_ratio - global_pos_ratio) < 0.03, "Fold class balance deviated significantly"

    # All samples accounted for once
    assert sorted(all_val_indices) == list(range(len(y)))


def test_stratified_folds_heart():
    """Verify 5-fold stratification for Heart dataset."""
    folds = get_folds("heart")
    assert len(folds) == 5

    X, y = load_heart_disease()
    global_pos_ratio = np.mean(y == 1)

    all_val_indices = []
    for fold in folds:
        train_idx, val_idx = fold
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        all_val_indices.extend(val_idx)

        val_pos_ratio = np.mean(y[val_idx] == 1)
        assert abs(val_pos_ratio - global_pos_ratio) < 0.03

    assert sorted(all_val_indices) == list(range(len(y)))
