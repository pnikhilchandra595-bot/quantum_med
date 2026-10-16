"""Quantum Core Engine for Hybrid Quantum Machine Learning Platform.

Provides quantum device abstractions, feature map embeddings,
parameterized variational ansatzes, simulator backends, and noise models.
"""

from src.quantum.backend import QuantumBackendManager, get_device
from src.quantum.feature_maps import (
    angle_embedding_circuit,
    zz_feature_map_circuit,
    get_feature_map,
)
from src.quantum.ansatz import (
    strongly_entangling_ansatz,
    basic_entangler_ansatz,
    hardware_efficient_ansatz,
)
from src.quantum.noise import (
    add_gaussian_sensor_noise,
    apply_depolarizing_channel,
    NoiseRobustnessEvaluator,
)

__all__ = [
    "QuantumBackendManager",
    "get_device",
    "angle_embedding_circuit",
    "zz_feature_map_circuit",
    "get_feature_map",
    "strongly_entangling_ansatz",
    "basic_entangler_ansatz",
    "hardware_efficient_ansatz",
    "add_gaussian_sensor_noise",
    "apply_depolarizing_channel",
    "NoiseRobustnessEvaluator",
]
