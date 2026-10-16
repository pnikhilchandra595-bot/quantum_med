"""Quantum Noise and Environmental Robustness Simulation Module (Deliverable D7, Task T3.10).

Simulates NISQ hardware imperfections (depolarizing quantum channels, gate errors)
and biomedical measurement sensor noise to evaluate model degradation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from sklearn.metrics import accuracy_score, recall_score, f1_score

try:
    import pennylane as qml
except ImportError:
    qml = None

logger = logging.getLogger(__name__)


def apply_depolarizing_channel(wires: List[int], p_error: float = 0.01) -> None:
    """Applies single-qubit depolarizing noise to specified wires.

    With probability p_error, replaces the qubit state with the completely mixed state I/2.
    """
    if qml is None:
        raise ImportError("PennyLane is required to simulate quantum noise channels.")

    if p_error > 0.0:
        for w in wires:
            qml.DepolarizingChannel(p_error, wires=w)


def add_gaussian_sensor_noise(
    X: np.ndarray,
    noise_level: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Simulates biomedical instrument sensor noise and electronic drift.

    Args:
        X: Normalized input matrix.
        noise_level: Standard deviation of additive zero-mean Gaussian noise.
        random_state: Deterministic random seed.

    Returns:
        Perturbed feature matrix with preserved dimensions.
    """
    if noise_level <= 0.0:
        return np.copy(X)

    rng = np.random.default_rng(random_state)
    noise = rng.normal(loc=0.0, scale=noise_level, size=X.shape)
    X_noisy = X + noise

    # Clip to periodic boundaries [-pi, pi] to respect quantum angle embedding domain
    return np.clip(X_noisy, -np.pi, np.pi)


class NoiseRobustnessEvaluator:
    """Benchmarks accuracy and clinical sensitivity degradation under increasing noise."""

    def __init__(
        self,
        noise_levels: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        if noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        else:
            self.noise_levels = sorted(noise_levels)
        self.random_state = random_state

    def evaluate_model_under_noise(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "QuantumModel",
    ) -> List[Dict[str, Any]]:
        """Evaluates model performance across all configured noise levels.

        Returns:
            List of dictionaries containing metrics per noise level.
        """
        results = []

        for noise in self.noise_levels:
            X_noisy = add_gaussian_sensor_noise(
                X_test,
                noise_level=noise,
                random_state=self.random_state,
            )

            preds = model.predict(X_noisy)
            acc = float(accuracy_score(y_test, preds))
            recall = float(recall_score(y_test, preds, zero_division=0))
            f1 = float(f1_score(y_test, preds, zero_division=0))

            results.append({
                "model": model_name,
                "noise_level": noise,
                "accuracy": acc,
                "sensitivity": recall,
                "f1_score": f1,
            })

            logger.info(
                f"[{model_name}] Noise: {noise:4.2f} | Acc: {acc*100:5.2f}% | "
                f"Sensitivity: {recall*100:5.2f}%"
            )

        return results

    def compare_quantum_vs_classical(
        self,
        quantum_model: Any,
        classical_model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Compares degradation curves between quantum and classical models."""
        q_results = self.evaluate_model_under_noise(
            quantum_model, X_test, y_test, model_name="Quantum (QSVC/VQC)"
        )
        c_results = self.evaluate_model_under_noise(
            classical_model, X_test, y_test, model_name="Classical (SVM)"
        )
        return q_results + c_results
