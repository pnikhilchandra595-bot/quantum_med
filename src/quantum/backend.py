"""
QuantumBackendManager: Decoupled hardware and simulator abstraction layer for PennyLane.
Audited for device bounds, memory safety caps, and backend failure error handling.
"""

import logging
from typing import Any, Dict, Optional

import pennylane as qml

logger = logging.getLogger(__name__)


class QuantumBackendManager:
    """
    Manages PennyLane quantum simulator devices, noise profiles, execution backends,
    device bounds, and statevector memory safety caps.
    """

    AVAILABLE_BACKENDS = ["default.qubit", "lightning.qubit", "default.mixed"]
    DEFAULT_MAX_QUBITS = 24  # 2^24 * 16 bytes = 256 MB statevector
    DEFAULT_MAX_MEMORY_MB = 1024.0  # 1 GB safety cap for CI & desktop

    def __init__(
        self,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
        noise_level: float = 0.0,
        seed: Optional[int] = 42,
        max_qubits: int = DEFAULT_MAX_QUBITS,
        max_memory_mb: float = DEFAULT_MAX_MEMORY_MB,
        fallback_to_default: bool = True,
    ):
        if not (0.0 <= noise_level <= 1.0):
            raise ValueError(f"noise_level must be between 0.0 and 1.0, got {noise_level}")
        if shots is not None and (
            not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0
        ):
            raise ValueError(f"shots must be a positive integer > 0 or None, got {shots}")
        if not isinstance(max_qubits, int) or isinstance(max_qubits, bool) or max_qubits <= 0:
            raise ValueError(f"max_qubits must be a positive integer, got {max_qubits}")
        if (not isinstance(max_memory_mb, (int, float))) or max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be a positive number, got {max_memory_mb}")

        self.backend_name = backend_name
        self.shots = shots
        self.noise_level = noise_level
        self.seed = seed
        self.max_qubits = max_qubits
        self.max_memory_mb = float(max_memory_mb)
        self.fallback_to_default = fallback_to_default
        self._cached_devices: Dict[int, Any] = {}

    @staticmethod
    def estimate_statevector_memory_mb(wires: int) -> float:
        """
        Estimate memory required for a complex128 statevector (16 bytes per amplitude) in megabytes.
        """
        if not isinstance(wires, int) or isinstance(wires, bool) or wires < 0:
            raise ValueError(f"wires must be a non-negative integer, got {wires}")
        num_amplitudes = 1 << wires  # 2^wires
        bytes_total = num_amplitudes * 16  # 16 bytes per complex128
        return bytes_total / (1024.0 * 1024.0)

    def validate_wire_bounds(self, wires: int) -> None:
        """
        Validate that wire count is within physical and memory bounds.
        Raises TypeError, ValueError, or MemoryError if bounds are violated.
        """
        if not isinstance(wires, int) or isinstance(wires, bool):
            raise TypeError(f"Wire count must be an integer, got {type(wires).__name__}")
        if wires <= 0:
            raise ValueError(f"Wire count must be a positive integer >= 1, got {wires}")
        if wires > self.max_qubits:
            raise ValueError(
                f"Requested wire count ({wires}) exceeds configured maximum limit of "
                f"{self.max_qubits} qubits to prevent simulator thrashing or OOM."
            )

        est_mem = self.estimate_statevector_memory_mb(wires)
        if est_mem > self.max_memory_mb:
            raise MemoryError(
                f"Statevector memory ({est_mem:.2f} MB) for {wires} qubits exceeds "
                f"the configured memory cap of {self.max_memory_mb:.2f} MB."
            )

    def get_device(self, wires: int) -> Any:
        """
        Instantiate or return cached PennyLane device for the specified wire count,
        enforcing safety bounds and handling simulator fallbacks.
        """
        self.validate_wire_bounds(wires)

        cache_key = wires
        if cache_key in self._cached_devices:
            return self._cached_devices[cache_key]

        target_name = self.backend_name
        dev = None
        error_msgs = []

        try:
            if target_name == "lightning.qubit":
                dev = qml.device("lightning.qubit", wires=wires, shots=self.shots)
            elif target_name == "default.mixed" or self.noise_level > 0.0:
                dev = qml.device("default.mixed", wires=wires, shots=self.shots)
            elif target_name == "default.qubit":
                dev = qml.device("default.qubit", wires=wires, shots=self.shots)
            else:
                # Custom or external simulator backend
                dev = qml.device(target_name, wires=wires, shots=self.shots)
        except Exception as e:
            error_msgs.append(f"Failed to load backend '{target_name}': {e}")
            if self.fallback_to_default and target_name != "default.qubit":
                logger.warning(
                    f"Simulator backend '{target_name}' failed. Falling back to 'default.qubit'."
                )
                try:
                    dev = qml.device("default.qubit", wires=wires, shots=self.shots)
                except Exception as fallback_err:
                    error_msgs.append(f"Fallback to 'default.qubit' also failed: {fallback_err}")
                    raise RuntimeError(" | ".join(error_msgs)) from fallback_err
            else:
                raise RuntimeError(
                    f"Requested simulator backend '{target_name}' could not be initialized: {e}"
                ) from e

        if dev is None:
            raise RuntimeError(f"Could not instantiate quantum device for backend '{target_name}'.")

        self._cached_devices[cache_key] = dev
        return dev

    def clear_device_cache(self) -> None:
        """Clear cached device references to free simulator resources."""
        self._cached_devices.clear()

    def get_cached_devices_count(self) -> int:
        """Return the number of currently cached device instances."""
        return len(self._cached_devices)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return backend configuration metadata.
        """
        return {
            "backend_name": self.backend_name,
            "shots": self.shots if self.shots is not None else "Analytic (Statevector)",
            "noise_level": self.noise_level,
            "seed": self.seed,
            "max_qubits": self.max_qubits,
            "max_memory_mb": self.max_memory_mb,
            "cached_devices": len(self._cached_devices),
        }
