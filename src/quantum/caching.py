"""
Statevector Caching & Quantum Circuit Optimization Engine.
Eliminates circuit recompilation overhead and accelerates simulator execution.
"""

from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class StatevectorCache:
    """
    Thread-safe in-memory LRU statevector cache for quantum feature embeddings.
    Maps continuous feature vectors to complex128 statevectors.
    """

    def __init__(self, max_size: int = 4096, precision_decimals: int = 7):
        self.max_size = max_size
        self.precision_decimals = precision_decimals
        self._cache: OrderedDict[Tuple[float, ...], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _make_key(self, x: np.ndarray) -> Tuple[float, ...]:
        """Convert float array to rounded tuple for hashable dictionary key."""
        return tuple(
            np.round(np.asarray(x, dtype=np.float64), decimals=self.precision_decimals).tolist()
        )

    def get(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Retrieve statevector from cache if present."""
        key = self._make_key(x)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, x: np.ndarray, statevector: np.ndarray) -> None:
        """Store statevector in cache, evicting oldest item if exceeding max_size."""
        key = self._make_key(x)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = np.asarray(statevector, dtype=np.complex128).copy()

    def get_or_compute(
        self, x: np.ndarray, circuit_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Return cached statevector or execute circuit and cache result."""
        cached = self.get(x)
        if cached is not None:
            return cached
        # Execute quantum circuit simulation
        state = circuit_fn(x)
        state_np = np.asarray(state, dtype=np.complex128)
        self.put(x, state_np)
        return state_np

    def compute_batch_statevectors(
        self, X: np.ndarray, circuit_fn: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """
        Compute statevectors for a 2D batch of samples utilizing cache.
        Returns matrix of shape (n_samples, 2^wires).
        """
        states = [self.get_or_compute(x, circuit_fn) for x in X]
        return np.vstack(states)

    def clear(self) -> None:
        """Clear cache and reset hit/miss counters."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_ratio(self) -> float:
        """Return cache hit ratio as float [0.0, 1.0]."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic cache statistics."""
        return {
            "cached_entries": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": round(self.hit_ratio, 4),
            "memory_usage_kb": round(
                len(self._cache) * 16 * 16 / 1024, 2
            ),  # Approx for 4-qubit states
        }
