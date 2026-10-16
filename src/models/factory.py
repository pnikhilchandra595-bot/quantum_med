"""
ModelFactory: Unified creation and registry for classical and quantum models.
"""

from typing import List, Optional

from src.models.base import BaseModel
from src.models.classical import ClassicalModel
from src.models.quantum import HQNNModel, QSVCModel, VQCModel
from src.quantum.backend import QuantumBackendManager


class ModelFactory:
    """
    Factory class providing decoupled model construction across classical and quantum tiers.
    """

    CLASSICAL_MODELS = ["svm", "rf", "xgb", "mlp"]
    QUANTUM_MODELS = ["qsvc", "vqc", "hqnn"]
    ALL_MODELS = CLASSICAL_MODELS + QUANTUM_MODELS

    @classmethod
    def is_quantum(cls, model_name: str) -> bool:
        """
        Check if model is part of the quantum suite.
        """
        return model_name.lower().strip() in cls.QUANTUM_MODELS

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all available model identifiers.
        """
        return list(cls.ALL_MODELS)

    @classmethod
    def create_model(
        cls,
        model_name: str,
        n_qubits: int = 4,
        backend_manager: Optional[QuantumBackendManager] = None,
        **kwargs,
    ) -> BaseModel:
        """
        Create and configure a model instance by name.
        """
        name_clean = model_name.lower().strip()

        if name_clean in cls.CLASSICAL_MODELS:
            return ClassicalModel(model_type=name_clean, **kwargs)

        # For quantum models, instantiate backend manager if not supplied
        q_backend = backend_manager or QuantumBackendManager(
            backend_name=kwargs.get("backend_name", "default.qubit"),
            shots=kwargs.get("shots", None),
            noise_level=kwargs.get("noise_level", 0.0),
        )

        kwargs_copy = dict(kwargs)
        if name_clean == "qsvc":
            c_val = kwargs_copy.pop("C", 1.0)
            return QSVCModel(n_qubits=n_qubits, backend_manager=q_backend, C=c_val, **kwargs_copy)

        elif name_clean == "vqc":
            n_layers = kwargs_copy.pop("n_layers", kwargs_copy.pop("reps", 2))
            lr = kwargs_copy.pop("lr", 0.05)
            epochs = kwargs_copy.pop("epochs", 25)
            return VQCModel(
                n_qubits=n_qubits,
                n_layers=n_layers,
                backend_manager=q_backend,
                lr=lr,
                epochs=epochs,
                **kwargs_copy,
            )

        elif name_clean == "hqnn":
            lr = kwargs_copy.pop("lr", 0.02)
            epochs = kwargs_copy.pop("epochs", 20)
            return HQNNModel(
                n_qubits=n_qubits, backend_manager=q_backend, lr=lr, epochs=epochs, **kwargs_copy
            )

        else:
            raise ValueError(
                f"Unknown model identifier '{model_name}'. " f"Supported models: {cls.ALL_MODELS}"
            )
