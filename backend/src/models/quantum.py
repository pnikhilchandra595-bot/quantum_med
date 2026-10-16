"""
Quantum Machine Learning Models: QSVC, VQC, and Hybrid Quantum Neural Network (HQNN).
"""

from typing import Any, Callable, List, Optional

import numpy as np
from sklearn.svm import SVC

from src.models.base import BaseModel
from src.quantum.backend import QuantumBackendManager
from src.quantum.caching import StatevectorCache

try:
    import pennylane as qml
except ImportError:
    qml = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None


class QSVCModel(BaseModel):
    """
    Quantum Support Vector Classifier using a PennyLane Quantum Kernel.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        backend_manager: Optional[QuantumBackendManager] = None,
        C: float = 1.0,
        use_statevector_caching: bool = True,
        **kwargs,
    ):
        super().__init__(name="qsvc", n_qubits=n_qubits, C=C, **kwargs)
        self.n_qubits = n_qubits
        self.C = C
        self.use_statevector_caching = use_statevector_caching
        self.backend_manager = backend_manager or QuantumBackendManager()
        self.dev = self.backend_manager.get_device(wires=self.n_qubits)
        self.svc = SVC(kernel="precomputed", C=self.C, probability=True, random_state=42)
        self.X_train_: Optional[np.ndarray] = None
        self.cache = StatevectorCache()
        self._qnode = self._build_kernel_qnode()
        self._state_qnode = self._build_state_qnode()

    def _build_kernel_qnode(self):
        wires = self.n_qubits

        @qml.qnode(self.dev, interface="auto")
        def _kernel(x1, x2):
            # Encode x1
            for i in range(wires):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x1[i] * x1[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

            # Invert encoding for x2
            for i in reversed(range(wires - 1)):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-x2[i] * x2[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in reversed(range(wires)):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)

            return qml.probs(wires=range(wires))

        return _kernel

    def _build_state_qnode(self):
        wires = self.n_qubits

        @qml.qnode(self.dev, interface="auto")
        def _state_circuit(x):
            for i in range(wires):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            return qml.state()

        return _state_circuit

    def _compute_kernel_matrix(
        self, A: np.ndarray, B: np.ndarray, force_pairwise: bool = False
    ) -> np.ndarray:
        n_a = len(A)
        n_b = len(B)

        is_symmetric = (A is B) or (n_a == n_b and np.array_equal(A, B))

        # Check if statevector simulation and caching can be utilized
        can_use_caching = (
            self.use_statevector_caching
            and not force_pairwise
            and (self.backend_manager.shots is None)
        )

        if can_use_caching:
            V_A = self.cache.compute_batch_statevectors(A, self._state_qnode)
            if is_symmetric:
                S = np.dot(V_A, V_A.conj().T)
                gram = np.abs(S) ** 2
                np.fill_diagonal(gram, 1.0)
                return np.asarray(gram, dtype=np.float32)
            else:
                V_B = self.cache.compute_batch_statevectors(B, self._state_qnode)
                S = np.dot(V_A, V_B.conj().T)
                gram = np.abs(S) ** 2
                return np.asarray(gram, dtype=np.float32)

        # Fallback to pairwise QNode execution
        gram = np.zeros((n_a, n_b), dtype=np.float32)
        for i in range(n_a):
            j_start = i if is_symmetric else 0
            for j in range(j_start, n_b):
                if is_symmetric and i == j:
                    gram[i, j] = 1.0
                else:
                    probs = self._qnode(A[i], B[j])
                    prob_zero = float(probs[0])
                    gram[i, j] = prob_zero
                    if is_symmetric:
                        gram[j, i] = prob_zero

        return gram

    def compute_gram_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the quantum kernel Gram matrix between X1 and X2 (or X1 and itself).
        Enforces symmetry and unit diagonal when X2 is None.
        """
        mat1 = np.asarray(X1, dtype=np.float32)
        if X2 is None:
            return self._compute_kernel_matrix(mat1, mat1)
        mat2 = np.asarray(X2, dtype=np.float32)
        return self._compute_kernel_matrix(mat1, mat2)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "QSVCModel":
        self.X_train_ = np.asarray(X, dtype=np.float32)
        gram_train = self._compute_kernel_matrix(self.X_train_, self.X_train_)
        self.svc.fit(gram_train, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or not self.is_fitted:
            raise RuntimeError("QSVCModel must be fitted before predict.")
        X_mat = np.asarray(X, dtype=np.float32)
        gram_test = self._compute_kernel_matrix(X_mat, self.X_train_)
        return self.svc.predict(gram_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.X_train_ is None or not self.is_fitted:
            raise RuntimeError("QSVCModel must be fitted before predict_proba.")
        X_mat = np.asarray(X, dtype=np.float32)
        gram_test = self._compute_kernel_matrix(X_mat, self.X_train_)
        return self.svc.predict_proba(gram_test)


class VQCModel(BaseModel):
    """
    Variational Quantum Classifier with Angle Embedding and Parameterized Rotations.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        backend_manager: Optional[QuantumBackendManager] = None,
        lr: float = 0.05,
        epochs: int = 25,
        **kwargs,
    ):
        super().__init__(
            name="vqc", n_qubits=n_qubits, n_layers=n_layers, lr=lr, epochs=epochs, **kwargs
        )
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.lr = lr
        self.epochs = epochs
        self.backend_manager = backend_manager or QuantumBackendManager()
        self.dev = self.backend_manager.get_device(wires=self.n_qubits)

        # Variational parameters shape: (n_layers, n_qubits, 3)
        np.random.seed(42)
        self.weights = 0.1 * np.random.randn(self.n_layers, self.n_qubits, 3).astype(np.float32)
        self.bias = 0.0
        self.loss_history: List[float] = []

        self._qnode = self._build_vqc_qnode()

    def _build_vqc_qnode(self):
        wires = self.n_qubits

        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def _circuit(weights, x):
            # Feature map: Angle embedding
            for i in range(wires):
                qml.RY(x[i], wires=i)

            # Variational Ansatz
            for layer_idx in range(weights.shape[0]):
                for i in range(wires):
                    qml.Rot(
                        weights[layer_idx, i, 0],
                        weights[layer_idx, i, 1],
                        weights[layer_idx, i, 2],
                        wires=i,
                    )
                # Entangling layer (circular CNOT)
                for i in range(wires):
                    qml.CNOT(wires=[i, (i + 1) % wires])

            # Measure expectation value of PauliZ on wire 0
            return qml.expval(qml.PauliZ(0))

        return _circuit

    def _forward_sample(self, weights, bias, x):
        val = self._qnode(weights, x)
        return val + bias

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        step_callback: Optional[Callable[[int, float], None]] = None,
        **kwargs,
    ) -> "VQCModel":
        total_epochs = epochs or self.epochs
        learning_rate = lr or self.lr
        self.loss_history = []

        # Convert labels from {0, 1} to {-1, +1} for PauliZ expectation optimization
        y_target = np.where(y == 0, -1.0, 1.0).astype(np.float32)
        X_arr = np.asarray(X, dtype=np.float32)

        opt = qml.AdamOptimizer(stepsize=learning_rate)
        weights = qml.numpy.array(self.weights, requires_grad=True)
        bias = qml.numpy.array(self.bias, requires_grad=True)

        # Mini-batch or sample subset if dataset is large for rapid training
        n_samples = len(X_arr)
        batch_size = min(32, n_samples)

        def cost_fn(w, b, x_batch, y_batch):
            loss = 0.0
            for xi, yi in zip(x_batch, y_batch):
                pred = self._qnode(w, xi) + b
                loss = loss + (pred - yi) ** 2
            return loss / len(x_batch)

        for epoch in range(total_epochs):
            indices = np.random.choice(n_samples, size=batch_size, replace=False)
            x_b = X_arr[indices]
            y_b = y_target[indices]

            (weights, bias), cost = opt.step_and_cost(
                lambda w, b: cost_fn(w, b, x_b, y_b), weights, bias
            )
            cost_val = float(cost)
            self.loss_history.append(cost_val)

            if step_callback:
                step_callback(epoch + 1, cost_val)

        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        raw_outputs = np.array([self._qnode(self.weights, xi) + self.bias for xi in X_arr])
        # Map expectation [-1, 1] to probability [0, 1] using sigmoid
        prob_1 = 1.0 / (1.0 + np.exp(-2.0 * raw_outputs))
        prob_0 = 1.0 - prob_1
        return np.column_stack([prob_0, prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class HQNNModule(nn.Module if nn is not None else object):
    """
    PyTorch Neural Network Module integrating a PennyLane Quantum Layer.
    """

    def __init__(self, in_features: int, n_qubits: int, dev: Any):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits

        diff_method = "backprop" if getattr(dev, "name", "") == "default.qubit" else "parameter-shift"

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def quantum_circuit(inputs, weights):
            for i in range(n_qubits):
                angle = inputs[:, i] if inputs.ndim > 1 else inputs[i]
                qml.RY(angle, wires=i)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (2, n_qubits, 3)}
        self.fc_in = nn.Sequential(nn.Linear(in_features, n_qubits), nn.Tanh())
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.fc_out = nn.Linear(n_qubits, 2)

    def forward(self, x):
        features = self.fc_in(x)
        q_out = self.q_layer(features)
        logits = self.fc_out(q_out)
        return logits


class HQNNModel(BaseModel):
    """
    Hybrid Quantum-Neural Network (HQNN) combining PyTorch Linear layers and PennyLane TorchLayer.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        backend_manager: Optional[QuantumBackendManager] = None,
        lr: float = 0.02,
        epochs: int = 20,
        **kwargs,
    ):
        super().__init__(name="hqnn", n_qubits=n_qubits, lr=lr, epochs=epochs, **kwargs)
        self.n_qubits = n_qubits
        self.lr = lr
        self.epochs = epochs
        self.backend_manager = backend_manager or QuantumBackendManager()
        self.dev = self.backend_manager.get_device(wires=self.n_qubits)
        self.net: Optional[HQNNModule] = None
        self.loss_history: List[float] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        step_callback: Optional[Callable[[int, float], None]] = None,
        **kwargs,
    ) -> "HQNNModel":
        if torch is None or nn is None:
            raise ImportError("PyTorch is required for HQNNModel.")

        total_epochs = epochs or self.epochs
        learning_rate = lr or self.lr
        self.loss_history = []

        in_dim = X.shape[1]
        self.net = HQNNModule(in_features=in_dim, n_qubits=self.n_qubits, dev=self.dev)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(X)), shuffle=True)

        self.net.train()
        for epoch in range(total_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.net(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch_x)

            avg_loss = float(epoch_loss / len(dataset))
            self.loss_history.append(avg_loss)

            if step_callback:
                step_callback(epoch + 1, avg_loss)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.net is None or not self.is_fitted:
            raise RuntimeError("HQNNModel must be fitted before predict_proba.")
        self.net.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.net(X_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
