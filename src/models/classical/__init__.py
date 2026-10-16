"""Classical Machine Learning Models Suite."""

from src.models.classical.svm import ClassicalSVM, LinearSVMModel, RBFSVMModel
from src.models.classical.mlp import ClassicalMLP, MLPModel
from src.models.classical.rf import ClassicalRandomForest
from src.models.classical.xgb import ClassicalXGBoost

# Backward-compatible model aliases for Member C evaluation pipelines
RandomForestModel = ClassicalRandomForest
XGBoostModel = ClassicalXGBoost

__all__ = [
    "ClassicalSVM",
    "ClassicalMLP",
    "ClassicalRandomForest",
    "ClassicalXGBoost",
    "LinearSVMModel",
    "RBFSVMModel",
    "MLPModel",
    "RandomForestModel",
    "XGBoostModel",
]
