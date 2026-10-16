"""Explainable AI (XAI) Module for Hybrid Quantum and Classical ML Platform."""

from src.xai.quantum_shap import QuantumKernelSHAPExplainer, QuantumPermutationImportance

try:
    from src.xai.classical_shap import ClassicalSHAPExplainer, SHAPExplanation
    from src.xai.quantum_shap import QuantumKernelSHAP
except ImportError:
    pass

__all__ = [
    "QuantumKernelSHAPExplainer",
    "QuantumPermutationImportance",
    "ClassicalSHAPExplainer",
    "SHAPExplanation",
    "QuantumKernelSHAP",
]
