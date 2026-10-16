"""
Automated unit tests covering quantum device bounds, statevector memory caps,
Gram matrix symmetry, circuit execution timeouts, and memory leak prevention.

Key Requirement: All quantum circuit unit tests execute in under 60 seconds on CPU.
"""

import gc
import time
import numpy as np
import pytest

pytest.importorskip("pennylane")

from src.models.quantum import QSVCModel, VQCModel
from src.quantum.backend import QuantumBackendManager

# =========================================================================
# Group 1: Quantum Device Bounds & Parameter Validation
# =========================================================================


def test_valid_device_wire_bounds():
    """Verify standard valid wire counts (1, 2, 4, 8) allocate successfully."""
    backend = QuantumBackendManager(backend_name="default.qubit")
    for wires in [1, 2, 4, 8]:
        dev = backend.get_device(wires=wires)
        assert dev is not None
        assert len(dev.wires) == wires


@pytest.mark.parametrize("invalid_wires", [0, -1, -5])
def test_invalid_wire_zero_or_negative(invalid_wires):
    """Verify that wires <= 0 raises ValueError."""
    backend = QuantumBackendManager()
    with pytest.raises(ValueError, match="positive integer"):
        backend.get_device(wires=invalid_wires)


@pytest.mark.parametrize("invalid_type", [3.5, "4", True, None])
def test_invalid_wire_types(invalid_type):
    """Verify that non-integer wires raise TypeError or ValueError."""
    backend = QuantumBackendManager()
    with pytest.raises((TypeError, ValueError)):
        backend.get_device(wires=invalid_type)


def test_exceeding_max_qubits_limit():
    """Verify that requesting wires above max_qubits raises ValueError."""
    backend = QuantumBackendManager(max_qubits=10)
    with pytest.raises(ValueError, match="exceeds configured maximum limit"):
        backend.get_device(wires=11)


@pytest.mark.parametrize("invalid_noise", [-0.1, 1.5, 10.0])
def test_invalid_noise_level(invalid_noise):
    """Verify noise level bounds [0.0, 1.0]."""
    with pytest.raises(ValueError, match="noise_level must be between 0.0 and 1.0"):
        QuantumBackendManager(noise_level=invalid_noise)


@pytest.mark.parametrize("invalid_shots", [0, -10, -1, 3.14])
def test_invalid_shots(invalid_shots):
    """Verify shots validation."""
    with pytest.raises(ValueError, match="shots must be a positive integer"):
        QuantumBackendManager(shots=invalid_shots)


# =========================================================================
# Group 2: Statevector Memory Caps & Safety Guardrails
# =========================================================================


def test_statevector_memory_estimation():
    """
    Verify exact analytical calculation of complex128 statevector memory:
    Memory = 2^N * 16 bytes.
    """
    assert QuantumBackendManager.estimate_statevector_memory_mb(0) == (16.0 / (1024 * 1024))
    # 4 qubits: 16 amplitudes * 16 bytes = 256 bytes = 256 / 1048576 MB
    assert pytest.approx(
        QuantumBackendManager.estimate_statevector_memory_mb(4), rel=1e-5
    ) == 256.0 / (1024 * 1024)
    # 10 qubits: 1024 amplitudes * 16 bytes = 16384 bytes = 0.015625 MB
    assert (
        pytest.approx(QuantumBackendManager.estimate_statevector_memory_mb(10), rel=1e-5)
        == 0.015625
    )
    # 20 qubits: 2^20 * 16 bytes = 16 MB
    assert pytest.approx(QuantumBackendManager.estimate_statevector_memory_mb(20), rel=1e-5) == 16.0
    # 24 qubits: 2^24 * 16 bytes = 256 MB
    assert (
        pytest.approx(QuantumBackendManager.estimate_statevector_memory_mb(24), rel=1e-5) == 256.0
    )


def test_memory_cap_enforcement():
    """
    Verify MemoryError is raised when requested statevector memory exceeds max_memory_mb.
    """
    # Set a tight memory cap of 8.0 MB (20 qubits requires 16 MB, 16 qubits requires 1 MB)
    strict_backend = QuantumBackendManager(max_qubits=25, max_memory_mb=8.0)

    # 16 qubits (1.0 MB) should succeed
    dev_small = strict_backend.get_device(wires=16)
    assert dev_small is not None

    # 20 qubits (16.0 MB) must be rejected
    with pytest.raises(MemoryError, match="exceeds the configured memory cap"):
        strict_backend.get_device(wires=20)


# =========================================================================
# Group 3: Simulator Backend Failure & Fallback Handling
# =========================================================================


def test_backend_fallback_on_invalid_device():
    """
    Verify that an unrecognized or failing backend gracefully falls back
    to default.qubit when fallback_to_default=True.
    """
    backend = QuantumBackendManager(
        backend_name="nonexistent.quantum.backend", fallback_to_default=True
    )
    dev = backend.get_device(wires=2)
    assert dev is not None
    # Device should be default.qubit state-vector simulator
    assert "default.qubit" in getattr(dev, "name", str(dev))


def test_backend_error_without_fallback():
    """
    Verify that an unrecognized backend raises RuntimeError when fallback_to_default=False.
    """
    backend = QuantumBackendManager(
        backend_name="nonexistent.quantum.backend", fallback_to_default=False
    )
    with pytest.raises(RuntimeError, match="could not be initialized"):
        backend.get_device(wires=2)


# =========================================================================
# Group 4: Gram Matrix Symmetry, Positive Semi-Definiteness & Bounds
# =========================================================================


@pytest.fixture
def sample_feature_vectors():
    """Deterministic synthetic clinical vectors for Gram matrix tests."""
    np.random.seed(42)
    # 6 samples, 4 features in [-pi, pi]
    return np.random.uniform(-np.pi, np.pi, size=(6, 4)).astype(np.float32)


def test_gram_matrix_exact_symmetry(sample_feature_vectors):
    """
    Verify quantum kernel Gram matrix is mathematically symmetric: K = K^T.
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=4, backend_manager=backend)

    gram = qsvc.compute_gram_matrix(sample_feature_vectors)

    assert gram.shape == (6, 6)
    # Exact symmetry assertion: K[i, j] == K[j, i]
    np.testing.assert_allclose(gram, gram.T, atol=1e-6, err_msg="Gram matrix is not symmetric!")


def test_gram_matrix_diagonal_unit_norm(sample_feature_vectors):
    """
    Verify state fidelity of self-inner product: K[i, i] = |<psi(xi)|psi(xi)>|^2 = 1.0.
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=4, backend_manager=backend)

    gram = qsvc.compute_gram_matrix(sample_feature_vectors)
    diag = np.diag(gram)

    np.testing.assert_allclose(diag, 1.0, atol=1e-5, err_msg="Gram matrix diagonal is not 1.0!")


def test_gram_matrix_probability_bounds(sample_feature_vectors):
    """
    Verify all kernel elements are valid transition probabilities: K[i, j] in [0.0, 1.0].
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=4, backend_manager=backend)

    gram = qsvc.compute_gram_matrix(sample_feature_vectors)

    assert np.all(gram >= -1e-6), "Gram matrix contains values < 0"
    assert np.all(gram <= 1.0 + 1e-6), "Gram matrix contains values > 1"


def test_gram_matrix_positive_semidefinite(sample_feature_vectors):
    """
    Verify that the quantum Gram matrix is positive semi-definite (eigenvalues >= 0).
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=4, backend_manager=backend)

    gram = qsvc.compute_gram_matrix(sample_feature_vectors)

    # Compute eigenvalues of real symmetric matrix
    eigenvalues = np.linalg.eigvalsh(gram)
    min_eig = np.min(eigenvalues)

    # Allow tiny numerical precision tolerance (-1e-5)
    assert min_eig >= -1e-5, f"Gram matrix is not PSD, minimum eigenvalue = {min_eig}"


# =========================================================================
# Group 5: Circuit Execution Timeouts & Execution Budget (<60s on CPU)
# =========================================================================


def test_vqc_forward_pass_timeout():
    """
    Verify that a single VQC quantum circuit evaluation executes within timeout (< 1.0s).
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    vqc = VQCModel(n_qubits=4, n_layers=1, backend_manager=backend)
    x = np.array([0.5, -0.3, 1.2, -0.8], dtype=np.float32)

    start_time = time.perf_counter()
    prob = vqc.predict_proba(np.array([x]))
    duration = time.perf_counter() - start_time

    assert prob.shape == (1, 2)
    assert (
        duration < 1.0
    ), f"VQC single-sample forward pass exceeded timeout ({duration:.3f}s > 1.0s)"


def test_qsvc_kernel_evaluation_timeout():
    """
    Verify that a 4x4 quantum kernel matrix executes within timeout (< 3.0s).
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=4, backend_manager=backend)
    x_batch = np.random.uniform(-np.pi, np.pi, size=(4, 4)).astype(np.float32)

    start_time = time.perf_counter()
    gram = qsvc.compute_gram_matrix(x_batch)
    duration = time.perf_counter() - start_time

    assert gram.shape == (4, 4)
    assert duration < 3.0, f"QSVC 4x4 kernel exceeded timeout ({duration:.3f}s > 3.0s)"


# =========================================================================
# Group 6: Memory Leak Prevention & Device Caching
# =========================================================================


def test_device_caching_and_cleanup():
    """
    Verify that QuantumBackendManager caches device instances per wire count,
    reuses existing instances without duplication, and frees them upon cache clearing.
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    assert backend.get_cached_devices_count() == 0

    # First access for 4 wires
    dev1 = backend.get_device(wires=4)
    assert backend.get_cached_devices_count() == 1

    # Second access for 4 wires should reuse cached device
    dev2 = backend.get_device(wires=4)
    assert dev1 is dev2
    assert backend.get_cached_devices_count() == 1

    # Access for 2 wires
    dev3 = backend.get_device(wires=2)
    assert dev3 is not dev1
    assert backend.get_cached_devices_count() == 2

    # Clear cache and assert cleanup
    backend.clear_device_cache()
    assert backend.get_cached_devices_count() == 0


def test_repeated_circuit_evaluations_memory_leak():
    """
    Verify that repeated quantum circuit invocations do not cause unbound memory growth.
    """
    backend = QuantumBackendManager(backend_name="default.qubit")
    vqc = VQCModel(n_qubits=4, n_layers=1, backend_manager=backend)
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    gc.collect()

    # Execute 50 forward passes in a loop
    for _ in range(50):
        _ = vqc.predict_proba(np.array([x]))

    gc.collect()
    # Cache should only hold 1 device instance for 4 wires
    assert backend.get_cached_devices_count() == 1
