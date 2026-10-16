"""Unit tests for classical baseline models (Random Forest, XGBoost, SVM, MLP)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.svm import LinearSVMModel, RBFSVMModel
from src.models.classical.mlp import MLPModel


@pytest.fixture
def binary_data():
    """Generate reproducible synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )
    return X, y


def test_trees_random_forest_fit_predict(binary_data):
    """Test RandomForestModel training, prediction, and probability outputs."""
    X, y = binary_data
    rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
    
    # Check unfitted state
    assert not rf.is_fitted
    with pytest.raises(RuntimeError):
        rf.predict(X)
    with pytest.raises(RuntimeError):
        rf.predict_proba(X)

    # Fit model
    fitted = rf.fit(X, y)
    assert fitted is rf
    assert rf.is_fitted

    # Predictions
    preds = rf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    probas = rf.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # Feature importances
    importances = rf.feature_importances_
    assert len(importances) == 10
    np.testing.assert_allclose(np.sum(importances), 1.0, atol=1e-4)


def test_trees_xgboost_fit_predict(binary_data):
    """Test XGBoostModel training, prediction, and probability outputs."""
    X, y = binary_data
    xgb = XGBoostModel(learning_rate=0.05, n_estimators=100, max_depth=4, random_state=42)

    # Check unfitted state
    assert not xgb.is_fitted
    with pytest.raises(RuntimeError):
        xgb.predict(X)

    # Fit model
    fitted = xgb.fit(X, y)
    assert fitted is xgb
    assert xgb.is_fitted

    # Predictions
    preds = xgb.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    probas = xgb.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    # Deterministic check
    xgb2 = XGBoostModel(learning_rate=0.05, n_estimators=100, max_depth=4, random_state=42)
    xgb2.fit(X, y)
    np.testing.assert_allclose(probas, xgb2.predict_proba(X), atol=1e-6)


def test_svm_models_fit_predict(binary_data):
    """Test Linear and RBF SVM models."""
    X, y = binary_data
    for ModelClass in [LinearSVMModel, RBFSVMModel]:
        clf = ModelClass(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        probas = clf.predict_proba(X)
        assert preds.shape == (100,)
        assert probas.shape == (100, 2)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


def test_mlp_model_fit_predict(binary_data):
    """Test MLP model."""
    X, y = binary_data
    mlp = MLPModel(random_state=42)
    mlp.fit(X, y)
    preds = mlp.predict(X)
    probas = mlp.predict_proba(X)
    assert preds.shape == (100,)
    assert probas.shape == (100, 2)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
