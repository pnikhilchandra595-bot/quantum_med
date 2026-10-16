"""Hybrid Quantum Neural Network (HQNN).

Integrates classical feedforward PyTorch layers with a PennyLane variational
quantum circuit layer (TorchLayer) trained end-to-end via PyTorch autograd.
"""

from typing import Optional, List, Dict
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    import pennylane as qml
    from src.quantum.backend import QuantumBackendManager
    from src.quantum.ansatz import strongly_entangling_ansatz
except ImportError:
    torch = None
    nn = None
    qml = None

logger = logging.getLogger(__name__)


class _HybridQNNModule(nn.Module if nn is not None else object):
    """PyTorch Module coupling classical Linear layers with a PennyLane TorchLayer."""

    def __init__(
        self,
        in_features: int,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_name: str = "default.qubit",
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev_manager = QuantumBackendManager(backend_name=backend_name, n_qubits=n_qubits)
        self.dev = dev_manager.get_device()
        wires = list(range(n_qubits))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def qnode(inputs, weights):
            # 1. Feature encoding into quantum states
            qml.AngleEmbedding(features=inputs, wires=wires, rotation="Y")
            # 2. Trainable variational ansatz
            strongly_entangling_ansatz(weights, wires=wires)
            # 3. Multi-wire expectation values
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Classical Input Projector: maps arbitrary input dimension down to n_qubits
        self.pre_net = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.Tanh(),  # Constrains inputs to [-1, 1] for stable angle rotations
        )

        # Classical Output Head: maps quantum expectation values to class logits
        self.post_net = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        classical_features = self.pre_net(x)
        # Scaled to [-pi, pi]
        quantum_inputs = classical_features * np.pi
        quantum_features = self.quantum_layer(quantum_inputs)
        logits = self.post_net(quantum_features)
        return logits


class HybridQuantumNeuralNetwork(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible estimator wrapping the PyTorch-PennyLane Hybrid QNN."""

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.01,
        max_epochs: int = 25,
        batch_size: int = 16,
        backend_name: str = "default.qubit",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.backend_name = backend_name
        self.seed = seed

        self.module_ = None
        self.loss_history_: List[float] = []
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridQuantumNeuralNetwork":
        """Fits the hybrid network parameters using PyTorch Adam and CrossEntropyLoss."""
        if torch is None or qml is None:
            raise ImportError("PyTorch and PennyLane are required for HQNN.")

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        in_features = X.shape[1]
        self.module_ = _HybridQNNModule(
            in_features=in_features,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            backend_name=self.backend_name,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.module_.parameters(), lr=self.learning_rate)

        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.loss_history_ = []
        self.module_.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.module_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_loss = epoch_loss / len(X)
            self.loss_history_.append(avg_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"HQNN Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax prediction probabilities."""
        check_is_fitted(self, ["module_"])
        X = check_array(X)
        self.module_.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32)
            logits = self.module_(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
