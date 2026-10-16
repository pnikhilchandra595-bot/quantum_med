"""Quantum Backend & Device Manager.

Provides a unified interface for initializing and managing quantum simulation
and hardware backends across PennyLane and Qiskit.
"""

from typing import Optional, Dict, Any
import logging

try:
    import pennylane as qml
except ImportError:
    qml = None

logger = logging.getLogger(__name__)

# Hard limits to prevent exponential statevector explosion (2^N amplitudes)
DEFAULT_MAX_QUBITS: int = 8
ABSOLUTE_MAX_QUBITS: int = 12


class QuantumBackendManager:
    """Manages quantum devices and simulators with memory guardrails."""

    SUPPORTED_BACKENDS = [
        "default.qubit",
        "lightning.qubit",
        "qiskit.aer",
        "qiskit.basicaer",
    ]

    def __init__(
        self,
        backend_name: str = "default.qubit",
        n_qubits: int = 4,
        shots: Optional[int] = None,
        max_qubits: int = DEFAULT_MAX_QUBITS,
    ):
        """Initialize the Quantum Backend Manager.

        Args:
            backend_name: Name of the PennyLane or plugin device.
            n_qubits: Number of addressable qubits (wires).
            shots: Number of measurement shots (None = exact analytic statevector).
            max_qubits: Safety limit for maximum qubits.
        """
        self.backend_name = backend_name
        self.n_qubits = n_qubits
        self.shots = shots
        self.max_qubits = min(max_qubits, ABSOLUTE_MAX_QUBITS)

        self._validate_qubit_count()
        self._device = None

    def _validate_qubit_count(self) -> None:
        """Enforces simulator bounds to prevent OOM."""
        if self.n_qubits < 1:
            raise ValueError(f"Number of qubits must be >= 1, got {self.n_qubits}")
        if self.n_qubits > self.max_qubits:
            raise ValueError(
                f"Requested {self.n_qubits} qubits exceeds the maximum safe limit "
                f"of {self.max_qubits} qubits for classical simulation."
            )

    def get_device(self) -> Any:
        """Instantiate and return the PennyLane quantum device.

        Returns:
            PennyLane Device instance.
        """
        if qml is None:
            raise ImportError(
                "PennyLane is not installed. Please install via: pip install pennylane"
            )

        if self._device is None:
            logger.info(
                f"Initializing quantum device: {self.backend_name} with "
                f"{self.n_qubits} qubits (shots={self.shots})"
            )
            try:
                self._device = qml.device(
                    self.backend_name,
                    wires=self.n_qubits,
                    shots=self.shots,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize requested backend '{self.backend_name}': {e}. "
                    f"Falling back to 'default.qubit'."
                )
                self.backend_name = "default.qubit"
                self._device = qml.device(
                    "default.qubit",
                    wires=self.n_qubits,
                    shots=self.shots,
                )

        return self._device

    def get_metadata(self) -> Dict[str, Any]:
        """Return backend configuration metadata."""
        return {
            "backend_name": self.backend_name,
            "n_qubits": self.n_qubits,
            "shots": self.shots,
            "analytic": self.shots is None,
            "max_safe_qubits": self.max_qubits,
        }


def get_device(
    n_qubits: int = 4,
    backend_name: str = "default.qubit",
    shots: Optional[int] = None,
) -> Any:
    """Convenience helper function to retrieve an active PennyLane device."""
    manager = QuantumBackendManager(
        backend_name=backend_name,
        n_qubits=n_qubits,
        shots=shots,
    )
    return manager.get_device()
