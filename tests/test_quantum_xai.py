"""Unit tests for Quantum Explainability Module (Member B Deliverables)."""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.xai.quantum_shap import QuantumKernelSHAPExplainer, QuantumPermutationImportance


class TestQuantumExplainability:
    """Tests for QuantumKernelSHAP and Permutation Importance."""

    @pytest.fixture
    def mock_quantum_model(self):
        model = MagicMock()
        # Mock predict_proba to return realistic probabilities based on input sum
        def mock_predict_proba(X):
            scores = 1.0 / (1.0 + np.exp(-np.sum(X, axis=1)))
            return np.column_stack([1.0 - scores, scores])

        model.predict_proba = mock_predict_proba
        return model

    def test_background_clustering(self, mock_quantum_model):
        rng = np.random.default_rng(42)
        background = rng.normal(0, 1, size=(50, 4))
        explainer = QuantumKernelSHAPExplainer(
            model=mock_quantum_model,
            background_data=background,
            n_background_samples=10,
        )
        assert explainer.reference_data.shape == (10, 4)

    def test_permutation_importance_calculation(self, mock_quantum_model):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(20, 3))
        y = (np.sum(X, axis=1) > 0).astype(int)

        p_imp = QuantumPermutationImportance(
            model=mock_quantum_model,
            scoring="accuracy",
            n_repeats=3,
        )
        importance = p_imp.compute_importance(X, y, feature_names=["F1", "F2", "F3"])

        assert len(importance) == 3
        assert "F1" in importance
        assert "mean_drop" in importance["F1"]
        assert "std_drop" in importance["F1"]
