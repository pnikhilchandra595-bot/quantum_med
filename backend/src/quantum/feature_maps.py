"""Quantum Feature Maps and Data Encoding Circuits.

Implements non-linear quantum embeddings that map classical biomedical
features into high-dimensional Hilbert space states |Phi(x)>.
"""

from typing import Sequence, Callable
import numpy as np

try:
    import pennylane as qml
except ImportError:
    qml = None


def angle_embedding_circuit(
    features: Sequence[float],
    wires: Sequence[int],
    rotation: str = "Y",
) -> None:
    """Encodes classical features into single-qubit rotation angles.

    Args:
        features: 1D array of classical feature values of length <= len(wires).
        wires: Sequence of qubit wire indices.
        rotation: Axis of rotation ('X', 'Y', or 'Z'). Default is 'Y'.
    """
    if qml is None:
        raise ImportError("PennyLane is required for quantum feature maps.")

    qml.AngleEmbedding(features=features, wires=wires, rotation=rotation)


def zz_feature_map_circuit(
    features: Sequence[float],
    wires: Sequence[int],
    reps: int = 1,
) -> None:
    """Second-Order Pauli-Z Feature Map (Hadamard + Rz + CNOT-Rz-CNOT entanglement).

    Mathematical Formulation:
        U_{Phi(x)} = prod_{r=1}^{reps} [ prod_{(j,k)} exp(i * phi_{jk}(x) * Z_j * Z_k)
                                        * prod_j exp(i * phi_j(x) * Z_j) * H^{otimes n} ]
        where:
            phi_j(x) = 2 * x_j
            phi_{jk}(x) = 2 * (pi - x_j) * (pi - x_k)

    Args:
        features: 1D array of classical feature values, scaled to [-pi, pi] or [0, 2*pi].
        wires: Sequence of qubit wire indices.
        reps: Number of feature map repetitions (layers). Default is 1 for shallow circuits.
    """
    if qml is None:
        raise ImportError("PennyLane is required for quantum feature maps.")

    n_wires = len(wires)
    n_features = len(features)

    for rep in range(reps):
        # 1. Layer of Hadamards to create equal superposition |+>^n
        for wire in wires:
            qml.Hadamard(wires=wire)

        # 2. Single-qubit Pauli-Z rotations: exp(i * phi_j(x) * Z_j)
        for i in range(min(n_features, n_wires)):
            phi_j = 2.0 * float(features[i])
            qml.RZ(phi_j, wires=wires[i])

        # 3. Two-qubit ZZ interactions: exp(i * phi_{jk}(x) * Z_j * Z_k)
        for i in range(min(n_features, n_wires)):
            for j in range(i + 1, min(n_features, n_wires)):
                phi_jk = 2.0 * (np.pi - float(features[i])) * (np.pi - float(features[j]))
                # CNOT -> RZ -> CNOT implements the exp(i * phi * Z_i * Z_j) interaction
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(phi_jk, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


def get_feature_map(name: str = "zz") -> Callable:
    """Retrieve feature map circuit callable by name.

    Args:
        name: Name of feature map ('zz', 'angle', 'angle_y', 'angle_x').

    Returns:
        Callable quantum circuit function.
    """
    normalized = name.lower().strip()
    if normalized in ["zz", "zz_feature_map", "zzfeaturemap"]:
        return zz_feature_map_circuit
    elif normalized in ["angle", "angle_y", "angle_embedding"]:
        return lambda feat, wires: angle_embedding_circuit(feat, wires, rotation="Y")
    elif normalized in ["angle_x"]:
        return lambda feat, wires: angle_embedding_circuit(feat, wires, rotation="X")
    elif normalized in ["angle_z"]:
        return lambda feat, wires: angle_embedding_circuit(feat, wires, rotation="Z")
    else:
        raise ValueError(
            f"Unsupported feature map '{name}'. Choose from: 'zz', 'angle_y', 'angle_x', 'angle_z'"
        )
