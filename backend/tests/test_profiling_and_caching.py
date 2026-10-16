"""
Unit tests for Quantum Execution Profiling (cProfile, tracemalloc) and Statevector Caching.
Validates >20% execution speedup, cache metrics, and unified .pkl/.pt serialization.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

from src.models.factory import ModelFactory
from src.quantum.caching import StatevectorCache
from src.utils.profiler import benchmark_caching_speedup, profile_quantum_execution
from src.utils.serialization import load_model, save_model


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# =========================================================================
# StatevectorCache Unit Tests
# =========================================================================


def test_statevector_cache_put_get():
    """Verify basic storage, retrieval, and hit/miss counters in StatevectorCache."""
    cache = StatevectorCache(max_size=10)
    x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    state = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    # Initially cache should miss
    assert cache.get(x) is None
    assert cache.misses == 1
    assert cache.hits == 0

    # Put in cache
    cache.put(x, state)
    retrieved = cache.get(x)

    assert retrieved is not None
    assert cache.hits == 1
    np.testing.assert_allclose(retrieved, state)
    assert cache.hit_ratio == 0.5


def test_statevector_cache_lru_eviction():
    """Verify LRU capacity enforcement and eviction."""
    cache = StatevectorCache(max_size=3)
    dummy_state = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    x1 = np.array([1.0, 0.0])
    x2 = np.array([2.0, 0.0])
    x3 = np.array([3.0, 0.0])
    x4 = np.array([4.0, 0.0])

    cache.put(x1, dummy_state)
    cache.put(x2, dummy_state)
    cache.put(x3, dummy_state)
    assert len(cache._cache) == 3

    # Adding x4 should evict x1 (the least recently used)
    cache.put(x4, dummy_state)
    assert len(cache._cache) == 3
    assert cache.get(x1) is None
    assert cache.get(x4) is not None


# =========================================================================
# Profiling & Speedup (>20%) Verification Tests
# =========================================================================


def test_cprofile_profiler_metrics():
    """Verify cProfile and tracemalloc capture execution profile and memory metrics."""
    result = profile_quantum_execution(model_name="qsvc", n_samples=10, n_qubits=4, top_n_stats=10)

    assert result["model_name"] == "qsvc"
    assert result["n_samples"] == 10
    assert result["elapsed_seconds"] > 0
    assert result["peak_memory_kb"] > 0
    assert result["total_calls"] > 0
    assert "function calls" in result["profile_summary"]


def test_simulator_speedup_exceeds_20_percent():
    """
    Key Requirement: Validate that Statevector Caching delivers >20% simulator speedup
    compared to uncached pairwise circuit execution.
    """
    bench = benchmark_caching_speedup(n_samples=16, n_qubits=4)

    assert bench["exceeds_20pct_threshold"] is True, (
        f"Statevector caching achieved {bench['speedup_percentage']:.2f}% speedup, "
        f"which is below the required 20% threshold!"
    )
    assert bench["speedup_percentage"] > 20.0
    assert bench["acceleration_factor"] > 1.2
    assert bench["warm_cache_hit_seconds"] < bench["baseline_uncached_seconds"]


# =========================================================================
# Unified Serialization Handling (.pkl and .pt)
# =========================================================================


def test_serialization_explicit_pkl(temp_dir):
    """Verify unified save_model/load_model handling of .pkl for classical & QSVC models."""
    np.random.seed(42)
    X = np.random.uniform(-1, 1, size=(20, 4)).astype(np.float32)
    y = np.random.choice([0, 1], size=20)

    # 1. Classical SVM with .pkl
    model_svm = ModelFactory.create_model("svm")
    model_svm.fit(X, y)
    orig_preds = model_svm.predict(X[:5])

    pkl_path = os.path.join(temp_dir, "svm_model.pkl")
    save_model(model_svm, pkl_path)
    loaded_svm = load_model(pkl_path)
    loaded_preds = loaded_svm.predict(X[:5])
    np.testing.assert_array_equal(orig_preds, loaded_preds)

    # 2. QSVC with .pkl
    model_qsvc = ModelFactory.create_model("qsvc", n_qubits=4)
    model_qsvc.fit(X[:8], y[:8])
    orig_qsvc_preds = model_qsvc.predict(X[:4])

    qsvc_pkl_path = os.path.join(temp_dir, "qsvc_model.pkl")
    save_model(model_qsvc, qsvc_pkl_path)
    loaded_qsvc = load_model(qsvc_pkl_path)
    loaded_qsvc_preds = loaded_qsvc.predict(X[:4])
    np.testing.assert_array_equal(orig_qsvc_preds, loaded_qsvc_preds)


def test_serialization_vqc_pt(temp_dir):
    """Verify unified save_model/load_model handling of .pt for VQC models."""
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, size=(16, 4)).astype(np.float32)
    y = np.random.choice([0, 1], size=16)

    model_vqc = ModelFactory.create_model("vqc", n_qubits=4, epochs=2)
    model_vqc.fit(X, y)
    orig_preds = model_vqc.predict(X[:4])
    orig_probs = model_vqc.predict_proba(X[:4])

    pt_path = os.path.join(temp_dir, "vqc_model.pt")
    save_model(model_vqc, pt_path)
    loaded_vqc = load_model(pt_path)
    loaded_preds = loaded_vqc.predict(X[:4])
    loaded_probs = loaded_vqc.predict_proba(X[:4])

    np.testing.assert_array_equal(orig_preds, loaded_preds)
    np.testing.assert_allclose(orig_probs, loaded_probs, rtol=1e-4, atol=1e-4)
