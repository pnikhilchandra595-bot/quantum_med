"""Explainable AI (XAI) module for classical and quantum feature attribution."""

from src.xai.classical_shap import ClassicalSHAPExplainer, SHAPExplanation
from src.xai.quantum_shap import QuantumKernelSHAP

__all__ = [
    "ClassicalSHAPExplainer",
    "SHAPExplanation",
    "QuantumKernelSHAP",
]
