"""Quantum Circuit Visualization Component (Deliverable D9, Task T4.7).

Generates dynamic ASCII text schematics and publication-grade Matplotlib circuit
diagrams for Streamlit dashboards, notebooks, and reports.
"""

from typing import Optional, Tuple, Any
import numpy as np
import logging

try:
    import pennylane as qml
    import matplotlib.pyplot as plt
    from src.quantum.backend import QuantumBackendManager
    from src.quantum.feature_maps import get_feature_map
    from src.quantum.ansatz import strongly_entangling_ansatz
except ImportError:
    qml = None
    plt = None

logger = logging.getLogger(__name__)


class QuantumCircuitVisualizer:
    """Renders interactive quantum circuit diagrams for hybrid models."""

    def __init__(self, backend_name: str = "default.qubit"):
        self.backend_name = backend_name

    def _get_qnode_for_model(
        self,
        model_type: str = "vqc",
        n_qubits: int = 4,
        n_layers: int = 1,
    ) -> Tuple[Any, list]:
        """Constructs a prototype QNode for visualization."""
        if qml is None:
            raise ImportError("PennyLane is required to draw quantum circuits.")

        dev = qml.device(self.backend_name, wires=n_qubits)
        wires = list(range(n_qubits))
        norm_type = model_type.lower().strip()

        if norm_type in ["vqc", "variational"]:
            f_map = get_feature_map("angle_y")

            @qml.qnode(dev)
            def circuit(features, weights):
                f_map(features, wires=wires)
                strongly_entangling_ansatz(weights, wires=wires)
                return qml.expval(qml.PauliZ(0))

            dummy_features = np.zeros(n_qubits)
            dummy_weights = np.zeros((n_layers, n_qubits, 3))
            return circuit, [dummy_features, dummy_weights]

        elif norm_type in ["qsvc", "kernel", "zz"]:
            f_map = get_feature_map("zz")

            @qml.qnode(dev)
            def circuit(x1, x2):
                f_map(x1, wires=wires)
                qml.adjoint(f_map)(x2, wires=wires)
                return qml.probs(wires=wires)

            dummy_x1 = np.zeros(n_qubits)
            dummy_x2 = np.zeros(n_qubits)
            return circuit, [dummy_x1, dummy_x2]

        elif norm_type in ["hqnn", "torch"]:
            @qml.qnode(dev)
            def circuit(features, weights):
                qml.AngleEmbedding(features=features, wires=wires, rotation="Y")
                strongly_entangling_ansatz(weights, wires=wires)
                return [qml.expval(qml.PauliZ(w)) for w in wires]

            dummy_features = np.zeros(n_qubits)
            dummy_weights = np.zeros((n_layers, n_qubits, 3))
            return circuit, [dummy_features, dummy_weights]

        else:
            raise ValueError(f"Unsupported model type '{model_type}'. Choose 'vqc', 'qsvc', or 'hqnn'.")

    def draw_ascii(
        self,
        model_type: str = "vqc",
        n_qubits: int = 4,
        n_layers: int = 1,
    ) -> str:
        """Returns plain text ASCII circuit schematic.

        Args:
            model_type: 'vqc', 'qsvc', or 'hqnn'.
            n_qubits: Number of addressable qubits.
            n_layers: Number of ansatz layers.

        Returns:
            Formatted string representation of quantum gates on wires.
        """
        qnode, dummy_args = self._get_qnode_for_model(model_type, n_qubits, n_layers)
        return qml.draw(qnode)(*dummy_args)

    def draw_mpl(
        self,
        model_type: str = "vqc",
        n_qubits: int = 4,
        n_layers: int = 1,
        style: str = "pennylane",
    ) -> Any:
        """Generates a Matplotlib Figure suitable for Streamlit (st.pyplot).

        Args:
            model_type: 'vqc', 'qsvc', or 'hqnn'.
            n_qubits: Number of addressable qubits.
            n_layers: Number of ansatz layers.
            style: Visual style ('pennylane', 'sketch', 'black_white').

        Returns:
            matplotlib.figure.Figure instance.
        """
        if plt is None:
            raise ImportError("Matplotlib is required for graphical circuit drawing.")

        qnode, dummy_args = self._get_qnode_for_model(model_type, n_qubits, n_layers)
        fig, ax = qml.draw_mpl(qnode, style=style)(*dummy_args)
        ax.set_title(
            f"Quantum Circuit: {model_type.upper()} ({n_qubits} Qubits, {n_layers} Layers)",
            fontsize=12,
            pad=15,
        )
        return fig
