"""Unit tests for Quantum Noise Simulation and Robustness Evaluation (T3.10)."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.quantum.noise import (
    add_gaussian_sensor_noise,
    NoiseRobustnessEvaluator,
)
from src.xai.quantum_shap import QuantumKernelSHAPExplainer


class TestQuantumNoise:
    """Tests for noise simulation and degradation evaluation."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-np.pi, np.pi, size=(20, 4))
        y = np.array([0] * 10 + [1] * 10)
        return X, y

    def test_gaussian_sensor_noise_properties(self, synthetic_data):
        X, _ = synthetic_data
        # Zero noise should return exact copy
        X_clean = add_gaussian_sensor_noise(X, noise_level=0.0)
        np.testing.assert_array_equal(X, X_clean)

        # Noisy matrix should preserve shape and stay in [-pi, pi]
        X_noisy = add_gaussian_sensor_noise(X, noise_level=0.2, random_state=42)
        assert X_noisy.shape == X.shape
        assert np.all(X_noisy >= -np.pi)
        assert np.all(X_noisy <= np.pi)
        assert not np.array_equal(X, X_noisy)

    def test_noise_robustness_evaluator(self, synthetic_data):
        X, y = synthetic_data
        mock_model = MagicMock()
        mock_model.predict.return_value = y

        evaluator = NoiseRobustnessEvaluator(noise_levels=[0.0, 0.1, 0.2])
        results = evaluator.evaluate_model_under_noise(
            mock_model, X, y, model_name="MockQuantum"
        )

        assert len(results) == 3
        assert results[0]["noise_level"] == 0.0
        assert results[0]["accuracy"] == 1.0
        assert "sensitivity" in results[0]
        assert "f1_score" in results[0]

    def test_patient_attribution_report(self, synthetic_data):
        X, _ = synthetic_data
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        explainer = QuantumKernelSHAPExplainer(
            model=mock_model,
            background_data=X,
            n_background_samples=5,
        )
        report = explainer.generate_patient_attribution_report(
            sample=X[0],
            feature_names=["Marker1", "Marker2", "Marker3", "Marker4"],
            top_k=2,
        )

        assert "prediction_probability" in report
        assert "prediction_label" in report
        assert len(report["top_clinical_factors"]) == 2
        assert report["prediction_probability"] == 0.8
        assert report["prediction_label"] == "MALIGNANT / DISEASE"
