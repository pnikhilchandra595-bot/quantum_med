"""Variational Quantum Classifier (VQC).

Implements a parameterized quantum variational circuit with analytical parameter-shift
gradient computation, trained using Adam optimizer.
"""

from typing import Optional, Sequence, Tuple, List
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    from src.quantum.backend import QuantumBackendManager
    from src.quantum.feature_maps import get_feature_map
    from src.quantum.ansatz import strongly_entangling_ansatz, get_ansatz_shape, initialize_weights
except ImportError:
    qml = None
    pnp = np

logger = logging.getLogger(__name__)


class VariationalQuantumClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier (VQC).

    Architecture:
        |0> -> [Feature Map U_Phi(x)] -> [Variational Ansatz W(theta)] -> Measure <Z_0>
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        learning_rate: float = 0.05,
        max_epochs: int = 30,
        batch_size: int = 16,
        feature_map_name: str = "angle_y",
        ansatz_type: str = "strongly_entangling",
        backend_name: str = "default.qubit",
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.feature_map_name = feature_map_name
        self.ansatz_type = ansatz_type
        self.backend_name = backend_name
        self.seed = seed

        self.weights_ = None
        self.bias_ = None
        self.loss_history_: List[float] = []
        self.classes_ = None
        self._dev = None
        self._qnode = None

    def _init_circuit(self) -> None:
        """Initializes the PennyLane QNode with parameter-shift differentiation."""
        if qml is None:
            raise ImportError("PennyLane is required for VQC execution.")

        dev_manager = QuantumBackendManager(
            backend_name=self.backend_name,
            n_qubits=self.n_qubits,
        )
        self._dev = dev_manager.get_device()
        wires = list(range(self.n_qubits))
        f_map = get_feature_map(self.feature_map_name)

        @qml.qnode(self._dev, diff_method="parameter-shift")
        def circuit(weights, features):
            f_map(features, wires=wires)
            strongly_entangling_ansatz(weights, wires=wires)
            # Expectation value of PauliZ on wire 0
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))

    def _cost(self, weights, bias, X_batch, y_batch):
        """Binary cross-entropy loss function over a mini-batch."""
        predictions = []
        for x in X_batch:
            expval = self._qnode(weights, x)
            prob = self._sigmoid(expval + bias)
            predictions.append(prob)

        preds = pnp.stack(predictions)
        # Numerical stability clip
        preds = pnp.clip(preds, 1e-7, 1.0 - 1e-7)
        bce = -pnp.mean(y_batch * pnp.log(preds) + (1.0 - y_batch) * pnp.log(1.0 - preds))
        return bce

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VariationalQuantumClassifier":
        """Fits the variational parameters (weights and bias) via Adam gradient descent."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        if X.shape[1] > self.n_qubits:
            raise ValueError(
                f"Features ({X.shape[1]}) exceed qubits ({self.n_qubits}). Pre-reduce dimensionality."
            )

        self._init_circuit()

        # Initialize weights and bias
        shape = get_ansatz_shape(self.ansatz_type, self.n_layers, self.n_qubits)
        init_w = initialize_weights(shape, seed=self.seed, scale=0.1)
        self.weights_ = pnp.array(init_w, requires_grad=True)
        self.bias_ = pnp.array(0.0, requires_grad=True)

        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        n_samples = len(X)
        rng = np.random.default_rng(self.seed)

        self.loss_history_ = []

        for epoch in range(self.max_epochs):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Optimization step
                (self.weights_, self.bias_), cost_val = opt.step_and_cost(
                    lambda w, b: self._cost(w, b, X_batch, y_batch),
                    self.weights_,
                    self.bias_,
                )
                epoch_losses.append(float(cost_val))

            avg_loss = float(np.mean(epoch_losses))
            self.loss_history_.append(avg_loss)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"VQC Epoch {epoch + 1}/{self.max_epochs} - Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for samples in X."""
        check_is_fitted(self, ["weights_", "bias_"])
        X = check_array(X)
        if self._qnode is None:
            self._init_circuit()

        probs_class_1 = []
        for x in X:
            expval = float(self._qnode(self.weights_, x))
            p1 = float(self._sigmoid(expval + float(self.bias_)))
            probs_class_1.append(p1)

        p1_arr = np.array(probs_class_1)
        p0_arr = 1.0 - p1_arr
        return np.column_stack([p0_arr, p1_arr])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels (0 or 1) using 0.5 decision threshold."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
