"""
Quantum Execution Profiler & Performance Benchmarking Engine.
Profiles memory allocations, circuit recompilation overhead via cProfile and tracemalloc,
and verifies >20% simulator execution speedup with statevector caching.
"""

import cProfile
import io
import pstats
import time
import tracemalloc
from typing import Any, Dict

import numpy as np

from src.models.quantum import QSVCModel, VQCModel
from src.quantum.backend import QuantumBackendManager


def profile_quantum_execution(
    model_name: str = "qsvc",
    n_samples: int = 16,
    n_qubits: int = 4,
    top_n_stats: int = 15,
) -> Dict[str, Any]:
    """
    Profile quantum circuit execution using cProfile and tracemalloc.
    Captures call counts, cumulative time, and peak memory usage.
    """
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_qubits)).astype(np.float32)
    y = np.random.choice([0, 1], size=n_samples).astype(np.int64)

    backend = QuantumBackendManager(backend_name="default.qubit")

    # Start memory tracing
    tracemalloc.start()
    tracemalloc.reset_peak()

    # Start CPU profiling
    profiler = cProfile.Profile()
    profiler.enable()

    t0 = time.perf_counter()

    if model_name.lower() == "qsvc":
        model = QSVCModel(n_qubits=n_qubits, backend_manager=backend)
        model.fit(X, y)
        _ = model.predict(X[:4])
    elif model_name.lower() == "vqc":
        model = VQCModel(n_qubits=n_qubits, n_layers=1, epochs=3, backend_manager=backend)
        model.fit(X, y)
        _ = model.predict(X[:4])
    else:
        raise ValueError(f"Profiling not supported for model '{model_name}'.")

    elapsed_time = time.perf_counter() - t0

    profiler.disable()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Format pstats report
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n_stats)
    profile_text = stream.getvalue()

    return {
        "model_name": model_name,
        "n_samples": n_samples,
        "n_qubits": n_qubits,
        "elapsed_seconds": round(elapsed_time, 4),
        "peak_memory_kb": round(peak_mem / 1024, 2),
        "total_calls": ps.total_calls,
        "profile_summary": profile_text,
    }


def benchmark_caching_speedup(
    n_samples: int = 16,
    n_qubits: int = 4,
) -> Dict[str, Any]:
    """
    Empirically benchmark statevector caching acceleration against uncached pairwise execution.
    Verifies that statevector caching delivers >20% execution speedup.
    """
    np.random.seed(42)
    X = np.random.uniform(-np.pi, np.pi, size=(n_samples, n_qubits)).astype(np.float32)

    backend = QuantumBackendManager(backend_name="default.qubit")
    qsvc = QSVCModel(n_qubits=n_qubits, backend_manager=backend)

    # 1. Uncached pairwise execution baseline
    t0 = time.perf_counter()
    gram_uncached = qsvc._compute_kernel_matrix(X, X, force_pairwise=True)
    t_baseline = time.perf_counter() - t0

    # 2. Statevector caching optimized execution
    qsvc.cache.clear()
    t1 = time.perf_counter()
    gram_cached = qsvc._compute_kernel_matrix(X, X, force_pairwise=False)
    t_optimized = time.perf_counter() - t1

    # 3. Repeat to test 100% cache hits
    t2 = time.perf_counter()
    _ = qsvc._compute_kernel_matrix(X, X, force_pairwise=False)
    t_hit = time.perf_counter() - t2

    # Verify mathematical fidelity between cached and uncached Gram matrices
    np.testing.assert_allclose(
        gram_cached,
        gram_uncached,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Statevector caching produced mathematically discrepant Gram matrix!",
    )

    speedup_pct = ((t_baseline - t_optimized) / t_baseline) * 100.0
    factor = (t_baseline / t_optimized) if t_optimized > 0 else float("inf")

    return {
        "n_samples": n_samples,
        "n_qubits": n_qubits,
        "baseline_uncached_seconds": round(t_baseline, 4),
        "optimized_cached_seconds": round(t_optimized, 4),
        "warm_cache_hit_seconds": round(t_hit, 4),
        "speedup_percentage": round(speedup_pct, 2),
        "acceleration_factor": round(factor, 2),
        "exceeds_20pct_threshold": speedup_pct > 20.0,
        "cache_stats": qsvc.cache.stats(),
    }
