"""
Tests for model checkpointing and serialization engine across classical and quantum models.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from src.data.loader import load_dataset
from src.models.factory import ModelFactory
from src.utils.serialization import load_model, save_model


@pytest.fixture(scope="module")
def dataset_data():
    """
    Small dataset subset fixture for rapid unit testing.
    """
    data = load_dataset(name="wdbc", n_qubits=4, reduction="pca")
    # Take a small subset for fast unit testing
    X_train = data["X_train"][:40]
    y_train = data["y_train"][:40]
    X_test = data["X_test"][:15]
    y_test = data["y_test"][:15]
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("model_name", ["svm", "rf"])
def test_classical_serialization(model_name, dataset_data, temp_dir):
    """
    Assert classical models reload from joblib/pickle and give identical predictions.
    """
    X_tr, y_tr = dataset_data["X_train"], dataset_data["y_train"]
    X_te = dataset_data["X_test"]

    model = ModelFactory.create_model(model_name=model_name)
    model.fit(X_tr, y_tr)

    orig_preds = model.predict(X_te)
    orig_probs = model.predict_proba(X_te)

    chk_path = os.path.join(temp_dir, f"{model_name}.joblib")
    save_model(model, chk_path)

    loaded = load_model(chk_path)
    loaded_preds = loaded.predict(X_te)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)


def test_vqc_serialization(dataset_data, temp_dir):
    """
    Assert VQC reloads weights and bias and produces identical predictions.
    """
    X_tr, y_tr = dataset_data["X_train"], dataset_data["y_train"]
    X_te = dataset_data["X_test"]

    model = ModelFactory.create_model(model_name="vqc", n_qubits=4, n_layers=1, epochs=3, lr=0.05)
    model.fit(X_tr, y_tr)

    orig_preds = model.predict(X_te)
    orig_probs = model.predict_proba(X_te)

    chk_path = os.path.join(temp_dir, "vqc.pt")
    save_model(model, chk_path)

    loaded = load_model(chk_path)
    loaded_preds = loaded.predict(X_te)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-4, atol=1e-4)


def test_qsvc_serialization(dataset_data, temp_dir):
    """
    Assert QSVC reloads kernel state and produces identical predictions.
    """
    # Use smaller slice for QSVC matrix computation speed
    X_tr = dataset_data["X_train"][:12]
    y_tr = dataset_data["y_train"][:12]
    X_te = dataset_data["X_test"][:4]

    model = ModelFactory.create_model(model_name="qsvc", n_qubits=4, C=1.0)
    model.fit(X_tr, y_tr)

    orig_preds = model.predict(X_te)
    orig_probs = model.predict_proba(X_te)

    chk_path = os.path.join(temp_dir, "qsvc.joblib")
    save_model(model, chk_path)

    loaded = load_model(chk_path)
    loaded_preds = loaded.predict(X_te)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-5, atol=1e-5)


def test_hqnn_serialization(dataset_data, temp_dir):
    """
    Assert HQNN reloads state_dict and produces identical predictions.
    """
    pytest.importorskip("torch")
    X_tr = dataset_data["X_train"][:20]
    y_tr = dataset_data["y_train"][:20]
    X_te = dataset_data["X_test"][:6]

    model = ModelFactory.create_model(model_name="hqnn", n_qubits=4, epochs=2, lr=0.02)
    model.fit(X_tr, y_tr)

    orig_preds = model.predict(X_te)
    orig_probs = model.predict_proba(X_te)

    chk_path = os.path.join(temp_dir, "hqnn.pt")
    save_model(model, chk_path)

    loaded = load_model(chk_path)
    loaded_preds = loaded.predict(X_te)
    loaded_probs = loaded.predict_proba(X_te)

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-4, atol=1e-4)
