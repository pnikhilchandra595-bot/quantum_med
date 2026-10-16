"""Unit and integration tests for Classical Baseline ML models (SVM, MLP, RF, XGB).

Test File: tests/test_classical_models.py
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease
from src.data.splitter import get_folds
from src.models.classical import ClassicalSVM, ClassicalMLP, ClassicalRandomForest, ClassicalXGBoost
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


def test_svm_linear_and_rbf_accuracy_and_prob():
    """Verify Linear & RBF SVM achieve >92% accuracy on WDBC validation folds."""
    X, y = load_wdbc()
    folds = get_folds("wdbc")

    for kernel in ["linear", "rbf"]:
        fold_accuracies = []
        for train_idx, val_idx in folds:
            preprocessor = WDBCPreprocessor()
            X_train = preprocessor.fit_transform(X[train_idx])
            X_val = preprocessor.transform(X[val_idx])
            y_train, y_val = y[train_idx], y[val_idx]

            svm = ClassicalSVM(kernel=kernel, C=1.0, probability=True, random_state=42)
            svm.fit(X_train, y_train)

            # Test predict and predict_proba shape & constraints
            y_pred = svm.predict(X_val)
            probs = svm.predict_proba(X_val)

            assert probs.shape == (len(val_idx), 2)
            np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
            assert set(np.unique(y_pred)).issubset({0, 1})

            acc = accuracy_score(y_val, y_pred)
            fold_accuracies.append(acc)

        mean_acc = float(np.mean(fold_accuracies))
        print(f"SVM ({kernel}) Mean CV Accuracy on WDBC: {mean_acc * 100:.2f}%")
        assert mean_acc > 0.92, f"SVM ({kernel}) achieved {mean_acc:.4f}, expected > 0.92"


def test_mlp_training_and_monotonic_loss():
    """Verify PyTorch MLP trains for 30 epochs with loss reduction."""
    X, y = load_wdbc()
    folds = get_folds("wdbc")
    train_idx, val_idx = folds[0]

    preprocessor = WDBCPreprocessor()
    X_train = preprocessor.fit_transform(X[train_idx])
    X_val = preprocessor.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    mlp = ClassicalMLP(
        lr=0.005,
        epochs=30,
        batch_size=32,
        dropout_rate=0.2,
        random_state=42,
    )
    mlp.fit(X_train, y_train, val_data=(X_val, y_val))

    assert len(mlp.train_losses) == 30
    assert mlp.train_losses[-1] < mlp.train_losses[0]

    probs = mlp.predict_proba(X_val)
    assert probs.shape == (len(val_idx), 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    y_pred = mlp.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    assert acc > 0.90, f"PyTorch MLP validation accuracy {acc:.4f} expected > 0.90"


def test_classical_models_save_and_load(tmp_path):
    """Verify serialization and deserialization of classical models."""
    X, y = load_wdbc()
    preprocessor = WDBCPreprocessor()
    X_proc = preprocessor.fit_transform(X[:100])
    y_proc = y[:100]

    # SVM Save & Load
    svm_path = tmp_path / "svm_model.joblib"
    svm = ClassicalSVM(kernel="linear").fit(X_proc, y_proc)
    pred_orig = svm.predict(X_proc)
    svm.save(svm_path)

    svm_loaded = ClassicalSVM(kernel="linear").load(svm_path)
    pred_loaded = svm_loaded.predict(X_proc)
    np.testing.assert_array_equal(pred_orig, pred_loaded)

    # MLP Save & Load
    mlp_path = tmp_path / "mlp_model.pt"
    mlp = ClassicalMLP(epochs=5).fit(X_proc, y_proc)
    mlp_pred_orig = mlp.predict(X_proc)
    mlp.save(mlp_path)

    mlp_loaded = ClassicalMLP(epochs=5).load(mlp_path)
    mlp_pred_loaded = mlp_loaded.predict(X_proc)
    np.testing.assert_array_equal(mlp_pred_orig, mlp_pred_loaded)


def test_rf_and_xgboost():
    """Verify Random Forest and XGBoost baseline implementations."""
    X, y = load_wdbc()
    preprocessor = WDBCPreprocessor()
    X_s = preprocessor.fit_transform(X)

    rf = ClassicalRandomForest(n_estimators=50, random_state=42).fit(X_s[:400], y[:400])
    y_pred_rf = rf.predict(X_s[400:])
    assert accuracy_score(y[400:], y_pred_rf) > 0.90

    xgb = ClassicalXGBoost(n_estimators=50, random_state=42).fit(X_s[:400], y[:400])
    y_pred_xgb = xgb.predict(X_s[400:])
    assert accuracy_score(y[400:], y_pred_xgb) > 0.90


def test_trees_random_forest_fit_predict(binary_data):
    """Test RandomForestModel training, prediction, and probability outputs."""
    X, y = binary_data
    rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=42)
    
    assert not rf.is_fitted
    with pytest.raises(RuntimeError):
        rf.predict(X)
    with pytest.raises(RuntimeError):
        rf.predict_proba(X)

    fitted = rf.fit(X, y)
    assert fitted is rf
    assert rf.is_fitted

    preds = rf.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    probas = rf.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    importances = rf.feature_importances_
    assert len(importances) == 10
    np.testing.assert_allclose(np.sum(importances), 1.0, atol=1e-4)


def test_trees_xgboost_fit_predict(binary_data):
    """Test XGBoostModel training, prediction, and probability outputs."""
    X, y = binary_data
    xgb = XGBoostModel(learning_rate=0.05, n_estimators=100, max_depth=4, random_state=42)

    assert not xgb.is_fitted
    with pytest.raises(RuntimeError):
        xgb.predict(X)

    fitted = xgb.fit(X, y)
    assert fitted is xgb
    assert xgb.is_fitted

    preds = xgb.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})

    probas = xgb.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)


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
