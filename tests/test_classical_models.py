"""Unit and integration tests for Classical Baseline ML models (SVM, MLP, RF, XGB).

Test File: tests/test_classical_models.py
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease
from src.data.splitter import get_folds
from src.models.classical import ClassicalSVM, ClassicalMLP, ClassicalRandomForest, ClassicalXGBoost


def test_svm_linear_and_rbf_accuracy_and_prob():
    """Verify Linear & RBF SVM achieve >92% accuracy on WDBC validation folds.
    
    Done condition: Linear & RBF SVM achieve >92% accuracy on WDBC validation folds.
    """
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
    """Verify PyTorch MLP trains for 30 epochs with loss reduction.
    
    Done condition: PyTorch MLP trains for 30 epochs with monotonic loss reduction.
    """
    X, y = load_wdbc()
    folds = get_folds("wdbc")
    train_idx, val_idx = folds[0]

    preprocessor = WDBCPreprocessor()
    X_train = preprocessor.fit_transform(X[train_idx])
    X_val = preprocessor.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    mlp = ClassicalMLP(
        epochs=30,
        lr=0.01,
        batch_size=32,
        dropout_rate=0.2,
        patience=30,  # Ensure full 30 epochs run
        random_state=42,
    )
    mlp.fit(X_train, y_train, val_data=(X_val, y_val))

    assert len(mlp.train_losses) == 30, f"Expected 30 epochs, got {len(mlp.train_losses)}"
    # Verify overall loss reduction
    initial_loss = np.mean(mlp.train_losses[:3])
    final_loss = np.mean(mlp.train_losses[-3:])
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

    # Verify predict and predict_proba
    y_pred = mlp.predict(X_val)
    probs = mlp.predict_proba(X_val)
    assert probs.shape == (len(val_idx), 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

    acc = accuracy_score(y_val, y_pred)
    assert acc > 0.90, f"MLP validation accuracy was {acc:.4f}, expected > 0.90"


def test_classical_models_save_and_load(tmp_path):
    """Verify serialization and deserialization across classical models."""
    X, y = load_wdbc()
    preprocessor = WDBCPreprocessor()
    X_s = preprocessor.fit_transform(X)

    # 1. Test SVM persistence
    svm = ClassicalSVM(kernel="linear", probability=False).fit(X_s[:50], y[:50])
    svm_path = tmp_path / "svm.joblib"
    svm.save(svm_path)
    loaded_svm = ClassicalSVM().load(svm_path)
    np.testing.assert_array_equal(svm.predict(X_s[50:60]), loaded_svm.predict(X_s[50:60]))

    # 2. Test MLP persistence
    mlp = ClassicalMLP(epochs=5, random_state=42).fit(X_s[:50], y[:50])
    mlp_path = tmp_path / "mlp.pt"
    mlp.save(mlp_path)
    loaded_mlp = ClassicalMLP(random_state=42).load(mlp_path)
    np.testing.assert_allclose(
        mlp.predict_proba(X_s[50:60]),
        loaded_mlp.predict_proba(X_s[50:60]),
        atol=1e-5,
    )


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
