"""Parameterized Quantum Circuits (PQC) and Variational Ansatzes.

Provides parameterized circuit architectures for VQC and Hybrid QNN models.
Includes barren plateau mitigation via shallow depth and controlled parameter initialization.
"""

from typing import Sequence, Tuple, Callable
import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None


def strongly_entangling_ansatz(
    weights: np.ndarray,
    wires: Sequence[int],
) -> None:
    """Strongly Entangling Layers ansatz.

    Consists of single-qubit arbitrary rotations (RZ-RY-RZ) on each wire,
    followed by entangling CNOT gates connecting wires with periodic boundary conditions.

    Args:
        weights: 3D tensor of shape (n_layers, n_wires, 3).
        wires: Sequence of qubit wires.
    """
    if qml is None:
        raise ImportError("PennyLane is required for quantum ansatzes.")

    qml.StronglyEntanglingLayers(weights, wires=wires)


def basic_entangler_ansatz(
    weights: np.ndarray,
    wires: Sequence[int],
) -> None:
    """Basic Entangler Layers ansatz (Hardware-efficient rotation + CNOT ring).

    Applies single-qubit RY rotations followed by a ring of CNOT gates.

    Args:
        weights: 2D tensor of shape (n_layers, n_wires).
        wires: Sequence of qubit wires.
    """
    if qml is None:
        raise ImportError("PennyLane is required for quantum ansatzes.")

    qml.BasicEntanglerLayers(weights, wires=wires)


def hardware_efficient_ansatz(
    weights: np.ndarray,
    wires: Sequence[int],
) -> None:
    """Hardware Efficient Ansatz alternating RY-RZ rotations with linear CNOT entanglement."""
    if qml is None:
        raise ImportError("PennyLane is required for quantum ansatzes.")

    n_layers, n_wires, _ = weights.shape
    for l in range(n_layers):
        for i, wire in enumerate(wires):
            qml.RY(weights[l, i, 0], wires=wire)
            qml.RZ(weights[l, i, 1], wires=wire)

        # Linear nearest-neighbor entanglement
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def get_ansatz_shape(
    ansatz_type: str,
    n_layers: int,
    n_wires: int,
) -> Tuple[int, ...]:
    """Calculate the required shape for the variational weight tensor.

    Args:
        ansatz_type: 'strongly_entangling', 'basic', or 'hardware_efficient'.
        n_layers: Number of repetitions (circuit depth).
        n_wires: Number of qubits.

    Returns:
        Tuple representing tensor dimensions.
    """
    normalized = ansatz_type.lower().strip()
    if "strong" in normalized:
        return (n_layers, n_wires, 3)
    elif "basic" in normalized:
        return (n_layers, n_wires)
    elif "hardware" in normalized:
        return (n_layers, n_wires, 2)
    else:
        raise ValueError(f"Unknown ansatz type: {ansatz_type}")


def initialize_weights(
    shape: Tuple[int, ...],
    seed: int = 42,
    scale: float = 0.1,
) -> np.ndarray:
    """Initializes variational parameters using a zero-centered Gaussian distribution.

    A small initial standard deviation (scale=0.1) helps mitigate the barren plateau
    phenomenon by starting near identity transformations.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=scale, size=shape)
