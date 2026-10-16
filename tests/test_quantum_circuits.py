"""Unit tests for Quantum Core Engine (Member B Deliverables).

Verifies backend initialization, feature maps, ansatz parameterization,
QSVC Gram matrix symmetry, VQC training, and HQNN autograd flow.
"""

import pytest

pytest.importorskip("pennylane")
import numpy as np

# Verify imports
from src.quantum.backend import QuantumBackendManager, DEFAULT_MAX_QUBITS
from src.quantum.feature_maps import get_feature_map
from src.quantum.ansatz import get_ansatz_shape, initialize_weights
from src.models.quantum.qsvc import QuantumSupportVectorClassifier
from src.models.quantum.vqc import VariationalQuantumClassifier
from src.models.quantum.hqnn import HybridQuantumNeuralNetwork


class TestQuantumBackendManager:
    """Tests for QuantumBackendManager device factory and limits."""

    def test_default_device_init(self):
        manager = QuantumBackendManager(backend_name="default.qubit", n_qubits=4)
        dev = manager.get_device()
        assert dev is not None
        assert manager.n_qubits == 4

    def test_max_qubit_guardrail(self):
        # Exceeding safe limit should raise ValueError
        with pytest.raises(ValueError, match="exceeds the maximum safe limit"):
            QuantumBackendManager(backend_name="default.qubit", n_qubits=20)

    def test_invalid_qubits_zero(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            QuantumBackendManager(backend_name="default.qubit", n_qubits=0)


class TestQuantumFeatureMapsAndAnsatz:
    """Tests for feature maps and variational ansatz utilities."""

    def test_feature_map_retrieval(self):
        zz_map = get_feature_map("zz")
        angle_map = get_feature_map("angle_y")
        assert callable(zz_map)
        assert callable(angle_map)

        with pytest.raises(ValueError, match="Unsupported feature map"):
            get_feature_map("invalid_map")

    def test_ansatz_weight_shapes(self):
        shape_strong = get_ansatz_shape("strongly_entangling", n_layers=2, n_wires=4)
        assert shape_strong == (2, 4, 3)

        shape_basic = get_ansatz_shape("basic", n_layers=3, n_wires=4)
        assert shape_basic == (3, 4)

        weights = initialize_weights(shape_strong, seed=42, scale=0.1)
        assert weights.shape == (2, 4, 3)
        assert np.abs(np.mean(weights)) < 0.1  # zero-centered initialization


class TestQSVC:
    """Tests for Quantum Support Vector Classifier kernel computations."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(6, 2))
        y = np.array([0, 0, 0, 1, 1, 1])
        return X, y

    def test_qsvc_gram_matrix_properties(self, synthetic_data):
        X, _ = synthetic_data
        qsvc = QuantumSupportVectorClassifier(n_qubits=2, reps=1)
        gram = qsvc.compute_kernel_matrix(X, X)

        assert gram.shape == (6, 6)
        # Check diagonal unity: K(x_i, x_i) == 1.0
        np.testing.assert_allclose(np.diag(gram), np.ones(6), atol=1e-5)
        # Check symmetry: K(x_i, x_j) == K(x_j, x_i)
        np.testing.assert_allclose(gram, gram.T, atol=1e-5)
        # Check bounded values [0, 1]
        assert np.all(gram >= 0.0) and np.all(gram <= 1.0)

    def test_qsvc_fit_predict(self, synthetic_data):
        X, y = synthetic_data
        qsvc = QuantumSupportVectorClassifier(n_qubits=2, reps=1, C=1.0)
        qsvc.fit(X, y)
        preds = qsvc.predict(X)
        probs = qsvc.predict_proba(X)

        assert len(preds) == len(X)
        assert probs.shape == (6, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(6), atol=1e-5)


class TestVQC:
    """Tests for Variational Quantum Classifier."""

    def test_vqc_training_and_inference(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-0.5, 0.5, size=(8, 2))
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        vqc = VariationalQuantumClassifier(
            n_qubits=2,
            n_layers=1,
            max_epochs=3,
            batch_size=4,
            learning_rate=0.1,
            seed=42,
        )
        vqc.fit(X, y)
        assert len(vqc.loss_history_) == 3

        probs = vqc.predict_proba(X)
        preds = vqc.predict(X)

        assert probs.shape == (8, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(8), atol=1e-5)
        assert set(preds).issubset({0, 1})


class TestHQNN:
    """Tests for Hybrid Quantum Neural Network (PyTorch autograd + TorchLayer)."""

    def test_hqnn_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-1.0, 1.0, size=(8, 3))
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

        hqnn = HybridQuantumNeuralNetwork(
            n_qubits=3,
            n_layers=1,
            max_epochs=3,
            batch_size=4,
            learning_rate=0.02,
            seed=42,
        )
        hqnn.fit(X, y)
        assert len(hqnn.loss_history_) == 3

        probs = hqnn.predict_proba(X)
        preds = hqnn.predict(X)

        assert probs.shape == (8, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(8), atol=1e-5)
        assert set(preds).issubset({0, 1})
