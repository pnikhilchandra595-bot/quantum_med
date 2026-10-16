"""
Model Checkpointing and Serialization Engine for Classical and Quantum Models.
Handles .pkl / .joblib for Classical and QSVC models, and .pt / .pth state dictionaries for HQNN and VQC.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np

from src.models.base import BaseModel
from src.models.classical import ClassicalModel
from src.models.quantum import HQNNModel, QSVCModel, VQCModel

try:
    import torch
except ImportError:
    torch = None


def save_model(model: BaseModel, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Serialize any supported model to disk.

    Supports:
        - Classical models (SVM, RF, XGB, MLP) via .pkl or .joblib.
        - QSVC via .pkl or .joblib (precomputed SVC, training vectors, quantum kernel params).
        - HQNN via .pt or .pth (PyTorch state_dict + architecture config).
        - VQC via .pt or .pth (tensor state dictionary with weights, bias, circuit config) or .pkl/.joblib.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_path = str(path)
    suffix = path.suffix.lower()

    checkpoint: Dict[str, Any] = {
        "model_name": model.name,
        "is_fitted": model.is_fitted,
        "params": model.get_params(),
        "metadata": metadata or {},
    }

    def _dump_file(data: Any, target_suffix: str) -> None:
        if target_suffix in [".pkl", ".pickle"]:
            with open(clean_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            joblib.dump(data, clean_path)

    if isinstance(model, ClassicalModel):
        checkpoint["estimator"] = model.estimator
        _dump_file(checkpoint, suffix)

    elif isinstance(model, HQNNModel):
        if torch is None:
            raise ImportError("PyTorch required to serialize HQNNModel to .pt format.")
        checkpoint["n_qubits"] = model.n_qubits
        checkpoint["lr"] = model.lr
        checkpoint["epochs"] = model.epochs
        checkpoint["in_features"] = model.net.in_features if model.net else None
        checkpoint["state_dict"] = model.net.state_dict() if model.net else None
        checkpoint["loss_history"] = model.loss_history
        torch.save(checkpoint, clean_path)

    elif isinstance(model, VQCModel):
        checkpoint["n_qubits"] = model.n_qubits
        checkpoint["n_layers"] = model.n_layers
        checkpoint["lr"] = model.lr
        checkpoint["epochs"] = model.epochs
        checkpoint["bias"] = float(model.bias)
        checkpoint["loss_history"] = model.loss_history

        if suffix in [".pt", ".pth"] and torch is not None:
            # Store as PyTorch tensor state dictionary
            checkpoint["weights"] = torch.tensor(model.weights, dtype=torch.float32)
            torch.save(checkpoint, clean_path)
        else:
            checkpoint["weights"] = np.asarray(model.weights, dtype=np.float32)
            _dump_file(checkpoint, suffix)

    elif isinstance(model, QSVCModel):
        checkpoint["n_qubits"] = model.n_qubits
        checkpoint["C"] = model.C
        checkpoint["svc"] = model.svc
        checkpoint["X_train_"] = model.X_train_
        _dump_file(checkpoint, suffix)

    else:
        # Generic fallback
        _dump_file({"model": model, "metadata": metadata or {}}, suffix)

    return clean_path


def load_model(filepath: str) -> BaseModel:
    """
    Deserialize model from disk (.pkl, .joblib, .pt, .pth) and restore full prediction capabilities.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{filepath}'")

    suffix = path.suffix.lower()
    is_torch_file = suffix in [".pt", ".pth"]
    is_pickle_file = suffix in [".pkl", ".pickle"]

    checkpoint = None

    if is_torch_file and torch is not None:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = joblib.load(str(path))
    elif is_pickle_file:
        try:
            with open(str(path), "rb") as f:
                checkpoint = pickle.load(f)
        except Exception:
            checkpoint = joblib.load(str(path))
    else:
        try:
            checkpoint = joblib.load(str(path))
        except Exception:
            if torch is not None:
                checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            else:
                raise

    model_name = checkpoint.get("model_name")

    if model_name in ["svm", "rf", "xgb", "mlp"]:
        model = ClassicalModel(model_type=model_name, **checkpoint.get("params", {}))
        model.estimator = checkpoint["estimator"]
        model.is_fitted = checkpoint.get("is_fitted", True)
        return model

    elif model_name == "hqnn":
        if torch is None:
            raise ImportError("PyTorch required to load HQNNModel.")
        from src.models.quantum import HQNNModule
        from src.quantum.backend import QuantumBackendManager

        n_qubits = checkpoint["n_qubits"]
        in_features = checkpoint["in_features"]
        model = HQNNModel(
            n_qubits=n_qubits, lr=checkpoint.get("lr", 0.02), epochs=checkpoint.get("epochs", 20)
        )
        if in_features is not None:
            backend = QuantumBackendManager()
            dev = backend.get_device(wires=n_qubits)
            net = HQNNModule(in_features=in_features, n_qubits=n_qubits, dev=dev)
            if checkpoint.get("state_dict") is not None:
                net.load_state_dict(checkpoint["state_dict"])
            model.net = net
            model.is_fitted = checkpoint.get("is_fitted", True)
            model.loss_history = checkpoint.get("loss_history", [])
        return model

    elif model_name == "vqc":
        model = VQCModel(
            n_qubits=checkpoint["n_qubits"],
            n_layers=checkpoint["n_layers"],
            lr=checkpoint.get("lr", 0.05),
            epochs=checkpoint.get("epochs", 25),
        )
        raw_weights = checkpoint["weights"]
        if torch is not None and isinstance(raw_weights, torch.Tensor):
            model.weights = raw_weights.detach().cpu().numpy().astype(np.float32)
        else:
            model.weights = np.asarray(raw_weights, dtype=np.float32)
        model.bias = float(checkpoint["bias"])
        model.is_fitted = checkpoint.get("is_fitted", True)
        model.loss_history = checkpoint.get("loss_history", [])
        return model

    elif model_name == "qsvc":
        model = QSVCModel(n_qubits=checkpoint["n_qubits"], C=checkpoint.get("C", 1.0))
        model.svc = checkpoint["svc"]
        model.X_train_ = checkpoint["X_train_"]
        model.is_fitted = checkpoint.get("is_fitted", True)
        return model

    elif "model" in checkpoint:
        return checkpoint["model"]

    else:
        raise ValueError(f"Unrecognized checkpoint format in {filepath}")
