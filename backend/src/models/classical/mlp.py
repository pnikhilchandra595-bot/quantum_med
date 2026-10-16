"""Classical Multi-Layer Perceptron (PyTorch Deep MLP & Scikit-Learn MLP) Baselines.

Module: src.models.classical.mlp
Owner: Member A & Member C
Deliverable: D5 (Classical Baselines)
Task: T1.8
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.neural_network import MLPClassifier

from src.models.base import BaseModel


class PyTorchMLPNet(nn.Module):
    """3-Layer Feedforward Deep Neural Network.
    
    Architecture:
    Linear(in_dim, 32) -> BatchNorm1d(32) -> ReLU -> Dropout(0.2) 
    -> Linear(32, 16) -> ReLU -> Linear(16, 2)
    """

    def __init__(self, in_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.fc1 = nn.Linear(in_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        return out


class ClassicalMLP(BaseModel):
    """Scikit-Learn compatible wrapper around PyTorchMLPNet.
    
    Features:
    - Adam optimizer
    - Cross-entropy loss
    - Early stopping on validation loss
    - Loss history logging
    - Serialization & deserialization
    """

    def __init__(
        self,
        name: str = "ClassicalMLP",
        in_dim: Optional[int] = None,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 30,
        patience: int = 10,
        dropout_rate: float = 0.2,
        random_state: int = 42,
        device: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.in_dim = in_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.dropout_rate = dropout_rate

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net: Optional[PyTorchMLPNet] = None
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.is_fitted = False
        self.is_fitted_ = False

        self._set_seed(self.random_state)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _init_network(self, in_features: int) -> None:
        self.net = PyTorchMLPNet(in_dim=in_features, dropout_rate=self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "ClassicalMLP":
        """Train neural network for specified epochs with early stopping."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if self.net is None or self.in_dim != X_arr.shape[1]:
            self.in_dim = X_arr.shape[1]
            self._init_network(self.in_dim)

        assert self.net is not None
        assert self.optimizer is not None

        # Data loaders
        dataset = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        # Validation tensors if provided
        X_val_t: Optional[torch.Tensor] = None
        y_val_t: Optional[torch.Tensor] = None
        if val_data is not None:
            X_val_t = torch.from_numpy(np.asarray(val_data[0], dtype=np.float32)).to(self.device)
            y_val_t = torch.from_numpy(np.asarray(val_data[1], dtype=np.int64)).to(self.device)

        best_loss = float("inf")
        best_weights = None
        epochs_without_improvement = 0

        self.train_losses.clear()
        self.val_losses.clear()

        for epoch in range(self.epochs):
            self.net.train()
            running_loss = 0.0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * batch_x.size(0)

            epoch_loss = running_loss / len(dataset)
            self.train_losses.append(epoch_loss)

            # Validation step
            if X_val_t is not None and y_val_t is not None:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(X_val_t)
                    val_loss = self.criterion(val_logits, y_val_t).item()
                    self.val_losses.append(val_loss)

                # Early stopping check
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_weights = copy.deepcopy(self.net.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        # Restore best weights and break
                        if best_weights is not None:
                            self.net.load_state_dict(best_weights)
                        break

        self.is_fitted = True
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities for class 0 and class 1."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("ClassicalMLP model must be fitted before calling predict_proba.")
        
        self.net.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X_arr).to(self.device)

        with torch.no_grad():
            logits = self.net(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save network weights and configuration."""
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "in_dim": self.in_dim,
                "state_dict": self.net.state_dict(),
                "hyperparams": {
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "dropout_rate": self.dropout_rate,
                },
            },
            path,
        )

    def load(self, file_path: Union[str, Path]) -> "ClassicalMLP":
        """Load network weights and reconstruct model."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.in_dim = checkpoint["in_dim"]
        self._init_network(self.in_dim)
        assert self.net is not None
        self.net.load_state_dict(checkpoint["state_dict"])
        self.is_fitted = True
        self.is_fitted_ = True
        return self


class MLPModel(BaseModel):
    """Multi-Layer Perceptron Classifier baseline using scikit-learn."""

    def __init__(
        self,
        name: str = "MLP",
        hidden_layer_sizes: Tuple[int, ...] = (32, 16),
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, random_state=random_state, **kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **kwargs,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "MLPModel":
        self.model.fit(X, y)
        self.is_fitted = True
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted yet.")
        return self.model.predict_proba(X)
