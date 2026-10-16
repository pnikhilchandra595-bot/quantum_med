"""Unit tests for Explainability (XAI) adapters and SHAP explainers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.svm import RBFSVMModel
from src.xai.classical_shap import ClassicalSHAPExplainer, SHAPExplanation
from src.xai.quantum_shap import QuantumKernelSHAP
from src.models.quantum.quantum_models import QSVCModel


@pytest.fixture
def sample_data():
    """Create reproducible classification dataset."""
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42,
    )
    feature_names = [f"Gene_{i}" for i in range(6)]
    return X, y, feature_names


def test_classical_shap_tree_explainer(sample_data):
    """Test TreeExplainer adapter for Random Forest and XGBoost."""
    X, y, feature_names = sample_data
    rf = RandomForestModel(n_estimators=20, max_depth=3, random_state=42).fit(X, y)

    explainer = ClassicalSHAPExplainer(
        model=rf,
        background_data=X[:20],
        feature_names=feature_names,
    )
    assert explainer.explainer_type == "tree"

    # Explain single sample
    exp_single = explainer.explain(X[0])
    assert isinstance(exp_single, SHAPExplanation)
    assert len(exp_single.values) == 6
    assert len(exp_single.normalized_importance) == 6
    assert np.all(exp_single.normalized_importance >= 0.0)
    np.testing.assert_allclose(np.sum(exp_single.normalized_importance), 1.0, atol=1e-5)

    # Check top features
    top5 = exp_single.top_features(5)
    assert len(top5) == 5
    assert top5[0][1] >= top5[1][1]  # sorted descending

    # Test XGBoost
    xgb = XGBoostModel(n_estimators=20, max_depth=3, random_state=42).fit(X, y)
    explainer_xgb = ClassicalSHAPExplainer(model=xgb, background_data=X[:20], feature_names=feature_names)
    exp_xgb = explainer_xgb.explain(X[:5])
    assert exp_xgb.values.shape == (5, 6)
    np.testing.assert_allclose(np.sum(exp_xgb.normalized_importance), 1.0, atol=1e-5)


def test_classical_shap_kernel_explainer(sample_data):
    """Test KernelExplainer adapter for SVM."""
    X, y, feature_names = sample_data
    svm = RBFSVMModel(random_state=42).fit(X, y)

    explainer = ClassicalSHAPExplainer(
        model=svm,
        background_data=X[:15],
        feature_names=feature_names,
        max_background_samples=15,
    )
    assert explainer.explainer_type == "kernel"

    exp = explainer.explain(X[0], nsamples=30)
    assert len(exp.values) == 6
    np.testing.assert_allclose(np.sum(exp.normalized_importance), 1.0, atol=1e-5)
    assert isinstance(exp.to_dict(), dict)


def test_classical_shap_vs_quantum_kernel_shap_compatibility(sample_data):
    """Verify QuantumKernelSHAP produces identical output schema to ClassicalSHAPExplainer."""
    X, y, feature_names = sample_data
    qsvc = QSVCModel(random_state=42).fit(X, y)

    q_explainer = QuantumKernelSHAP(
        model=qsvc,
        background_data=X[:15],
        feature_names=feature_names,
    )
    q_exp = q_explainer.explain(X[0], nsamples=30)

    assert isinstance(q_exp, SHAPExplanation)
    assert len(q_exp.normalized_importance) == 6
    np.testing.assert_allclose(np.sum(q_exp.normalized_importance), 1.0, atol=1e-5)
