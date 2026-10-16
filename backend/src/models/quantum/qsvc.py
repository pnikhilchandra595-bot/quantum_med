"""Quantum Support Vector Classifier (QSVC).

Implements a quantum kernel SVM using state fidelity estimation in Hilbert space.
Fully compliant with the scikit-learn BaseEstimator and ClassifierMixin interface.
"""

from typing import Optional, Sequence, Union
import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    import pennylane as qml
    from src.quantum.backend import QuantumBackendManager
    from src.quantum.feature_maps import get_feature_map
except ImportError:
    qml = None

logger = logging.getLogger(__name__)


class QuantumSupportVectorClassifier(BaseEstimator, ClassifierMixin):
    """Quantum Support Vector Classifier with Fidelity-based Quantum Kernel.

    Computes Gram matrix:
        K_{i, j} = |<Phi(x_i) | Phi(x_j)>|^2 = |<0| U_{Phi(x_j)}^dagger U_{Phi(x_i)} |0>|^2
    """

    def __init__(
        self,
        n_qubits: int = 4,
        feature_map_name: str = "zz",
        reps: int = 1,
        C: float = 1.0,
        backend_name: str = "default.qubit",
        shots: Optional[int] = None,
        probability: bool = True,
    ):
        self.n_qubits = n_qubits
        self.feature_map_name = feature_map_name
        self.reps = reps
        self.C = C
        self.backend_name = backend_name
        self.shots = shots
        self.probability = probability

        self._svc = None
        self._dev = None
        self._kernel_qnode = None
        self.X_train_ = None
        self.classes_ = None

    def _init_kernel_circuit(self) -> None:
        """Initializes the PennyLane QNode for pairwise quantum kernel estimation."""
        if qml is None:
            raise ImportError("PennyLane is required for QSVC execution.")

        dev_manager = QuantumBackendManager(
            backend_name=self.backend_name,
            n_qubits=self.n_qubits,
            shots=self.shots,
        )
        self._dev = dev_manager.get_device()
        wires = list(range(self.n_qubits))
        f_map = get_feature_map(self.feature_map_name)

        @qml.qnode(self._dev)
        def kernel_circuit(x1, x2):
            # Encode sample 1: U_Phi(x1) |0>
            f_map(x1, wires=wires)
            # Apply adjoint (inverse) of sample 2 encoding: U_Phi(x2)^dagger
            qml.adjoint(f_map)(x2, wires=wires)
            # Measure transition probability to |00...0>
            return qml.probs(wires=wires)

        self._kernel_qnode = kernel_circuit

    def _compute_kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes fidelity |<Phi(x1)|Phi(x2)>|^2 for a single pair."""
        probs = self._kernel_qnode(x1, x2)
        # Index 0 corresponds to the |00...0> state
        return float(probs[0])

    def compute_kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes the quantum Gram kernel matrix between dataset X1 and X2.

        Exploits Hermitian symmetry: K_{ij} = K_{ji} and K_{ii} = 1.0 when X1 == X2.
        """
        if self._kernel_qnode is None:
            self._init_kernel_circuit()

        is_symmetric = X2 is None or (X1 is X2)
        if X2 is None:
            X2 = X1

        n1 = len(X1)
        n2 = len(X2)
        gram = np.zeros((n1, n2), dtype=np.float64)

        if is_symmetric:
            for i in range(n1):
                gram[i, i] = 1.0
                for j in range(i + 1, n1):
                    val = self._compute_kernel_value(X1[i], X1[j])
                    gram[i, j] = val
                    gram[j, i] = val
        else:
            for i in range(n1):
                for j in range(n2):
                    gram[i, j] = self._compute_kernel_value(X1[i], X2[j])

        # Numerical stabilization: ensure strictly valid values [0, 1]
        np.clip(gram, 0.0, 1.0, out=gram)
        return gram

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumSupportVectorClassifier":
        """Fits the QSVC model on training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Binary target labels (0 or 1).
        """
        X, y = check_X_y(X, y)
        if X.shape[1] > self.n_qubits:
            raise ValueError(
                f"Input dimension {X.shape[1]} exceeds available qubits ({self.n_qubits}). "
                f"Reduce features using PCA or SelectKBest first."
            )

        self.classes_ = np.unique(y)
        self.X_train_ = np.copy(X)

        logger.info(f"Computing QSVC training Gram matrix of shape ({len(X)}, {len(X)})...")
        gram_train = self.compute_kernel_matrix(X, X)

        self._svc = SVC(
            kernel="precomputed",
            C=self.C,
            probability=self.probability,
            random_state=42,
        )
        self._svc.fit(gram_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for query samples in X."""
        check_is_fitted(self, ["X_train_", "_svc"])
        X = check_array(X)
        gram_test = self.compute_kernel_matrix(X, self.X_train_)
        return self._svc.predict(gram_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for query samples in X."""
        check_is_fitted(self, ["X_train_", "_svc"])
        if not self.probability:
            raise ValueError("Probability estimation was disabled at initialization.")
        X = check_array(X)
        gram_test = self.compute_kernel_matrix(X, self.X_train_)
        return self._svc.predict_proba(gram_test)
