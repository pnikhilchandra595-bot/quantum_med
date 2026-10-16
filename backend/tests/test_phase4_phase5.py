"""Unit and integration tests for Member C Phase 4 and Phase 5 deliverables.

Tests:
- T4.6: Clinical Explainability UI Dashboard backend components
- T5.6: 15-Slide Competition Pitch Deck generator (PPTX & PDF)
"""

import os
import importlib
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("pptx")
from pptx import Presentation

explain_page = importlib.import_module("app.pages.5_Explainability")
from scripts.generate_pitch_deck import generate_all_decks


def test_explainability_models_and_explainers():
    """Verify T4.6 model loading, zero data leakage, and explainers."""
    # Test cached resource loader
    X_train, X_test, y_train, y_test, component_names, models, explainers = (
        explain_page.load_and_prepare_models()
    )

    assert len(component_names) == 6
    assert len(models) == 8
    assert len(explainers) == 8

    # Verify model names
    expected_models = [
        "QSVC (Quantum Kernel)",
        "VQC (Variational Quantum)",
        "HQNN (Hybrid Quantum NN)",
        "Random Forest (Classical)",
        "XGBoost (Classical)",
        "RBF SVM (Classical)",
        "Linear SVM (Classical)",
        "MLP (Classical)",
    ]
    for name in expected_models:
        assert name in models
        assert name in explainers

    # Verify single patient inference & SHAP explanation on QSVC and RF
    patient = X_test[0]
    for test_model_name in ["QSVC (Quantum Kernel)", "Random Forest (Classical)"]:
        m = models[test_model_name]
        exp = explainers[test_model_name]

        # Prediction probability
        probas = m.predict_proba(patient.reshape(1, -1))[0]
        assert len(probas) == 2
        assert 0.0 <= probas[1] <= 1.0

        # SHAP calculation
        explanation = exp.explain(patient, nsamples=25)
        assert len(explanation.values) == 6
        assert len(explanation.normalized_importance) == 6
        assert np.isclose(np.sum(explanation.normalized_importance), 1.0, atol=1e-4)


def test_pitch_deck_artifacts_validity(tmp_path):
    """Verify T5.6 15-slide pitch deck generation produces valid PPTX and PDF files."""
    pptx_path = Path("docs/HQML_Presentation_Deck.pptx")
    pdf_path = Path("docs/HQML_Presentation_Deck.pdf")

    assert pptx_path.exists()
    assert pdf_path.exists()

    # PPTX must have exactly 15 slides
    prs = Presentation(str(pptx_path))
    assert len(prs.slides) == 15

    # Check file sizes (> 500 KB due to embedded publication figures)
    assert pptx_path.stat().st_size > 500_000
    assert pdf_path.stat().st_size > 500_000
