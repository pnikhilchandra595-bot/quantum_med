"""Tests for Single-Sample Clinical Inference Service.

Module: tests.test_inference
Owner: Member A (Classical ML & Pipeline Integrity Lead)
"""

import pytest
import numpy as np
from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.models.classical.inference import ClinicalInferenceService, ClinicalInferenceResult
from src.models.classical.xgb import ClassicalXGBoost


def test_wdbc_single_sample_inference_valid():
    """Verify inference service evaluates a valid WDBC patient."""
    service = ClinicalInferenceService(dataset_name="wdbc")
    X, y, feature_names = load_wdbc(as_frame=False, return_feature_names=True)

    sample_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}
    result = service.predict_patient(sample_dict)

    assert isinstance(result, ClinicalInferenceResult)
    assert result.is_valid is True
    assert result.validation_error is None
    assert result.predicted_class in (0, 1)
    assert result.class_label in ("Benign (Non-Malignant)", "Malignant (Carcinoma)")
    assert 0.0 <= result.risk_score <= 1.0
    assert result.risk_category in ("Low Risk", "Moderate Risk", "High Risk")
    assert result.latency_ms > 0.0


def test_wdbc_single_sample_inference_invalid_missing_key():
    """Verify inference service catches missing features and returns clean report."""
    service = ClinicalInferenceService(dataset_name="wdbc")
    incomplete_sample = {"mean radius": 14.5}

    result = service.predict_patient(incomplete_sample)
    assert result.is_valid is False
    assert result.predicted_class is None
    assert "Missing required input column" in result.validation_error


def test_heart_single_sample_inference_valid():
    """Verify inference service evaluates a valid Heart Disease patient."""
    service = ClinicalInferenceService(dataset_name="heart")
    X, y, feature_names = load_heart_disease(as_frame=False, preprocess=False, return_feature_names=True)

    sample_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}
    result = service.predict_patient(sample_dict)

    assert result.is_valid is True
    assert result.validation_error is None
    assert result.predicted_class in (0, 1)
    assert result.class_label in ("No Coronary Heart Disease", "Coronary Artery Disease Present")
    assert 0.0 <= result.risk_score <= 1.0
    assert result.latency_ms > 0.0


def test_heart_single_sample_sensor_out_of_bounds():
    """Verify inference service catches physiological sensor anomalies."""
    service = ClinicalInferenceService(dataset_name="heart")
    X, y, feature_names = load_heart_disease(as_frame=False, preprocess=False, return_feature_names=True)

    sample_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}
    # Set impossible resting blood pressure
    sample_dict["trestbps"] = 999.0

    result = service.predict_patient(sample_dict)
    assert result.is_valid is False
    assert "Sensor value out of valid physiological range" in result.validation_error


def test_inference_with_custom_model():
    """Verify inference service works with custom model (e.g., XGBoost)."""
    xgb_model = ClassicalXGBoost(n_estimators=30, max_depth=3, random_state=42)
    service = ClinicalInferenceService(dataset_name="wdbc", model=xgb_model)

    X, y, feature_names = load_wdbc(as_frame=False, return_feature_names=True)
    sample_dict = {name: float(X[0, i]) for i, name in enumerate(feature_names)}

    result = service.predict_patient(sample_dict)
    assert result.is_valid is True
    assert result.model_name == "ClassicalXGBoost"
