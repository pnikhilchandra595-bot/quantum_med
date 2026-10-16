"""End-to-End Single-Sample Clinical Inference Pipeline.

Module: src.models.classical.inference
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Deliverable: D5 & D2 (Single-Sample Inference Hardening)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd

from src.data.validator import PatientRecordValidator, ValidationError
from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease, HeartDiseasePreprocessor
from src.models.base import BaseModel
from src.models.classical.svm import ClassicalSVM
from src.models.classical.rf import ClassicalRandomForest
from src.models.classical.xgb import ClassicalXGBoost
from src.models.classical.mlp import ClassicalMLP


@dataclass
class ClinicalInferenceResult:
    """Structured report for single-patient clinical inference."""

    dataset: str
    is_valid: bool
    predicted_class: Optional[int]
    class_label: Optional[str]
    risk_score: Optional[float]
    risk_category: Optional[str]
    model_name: str
    latency_ms: float
    validation_error: Optional[str] = None
    sanitized_features: Optional[Dict[str, float]] = None


class ClinicalInferenceService:
    """End-to-end clinical inference service with validation guardrails and fitted pipelines."""

    DIAGNOSIS_LABELS = {
        "wdbc": {0: "Benign (Non-Malignant)", 1: "Malignant (Carcinoma)"},
        "heart": {0: "No Coronary Heart Disease", 1: "Coronary Artery Disease Present"},
    }

    def __init__(
        self,
        dataset_name: str = "wdbc",
        model: Optional[BaseModel] = None,
        preprocessor: Optional[Union[WDBCPreprocessor, HeartDiseasePreprocessor]] = None,
    ):
        """Initialize the clinical inference service.

        Parameters
        ----------
        dataset_name : str
            'wdbc' or 'heart'.
        model : BaseModel, optional
            Fitted classical model. If None, default tuned Linear SVM (WDBC) or
            Random Forest (Heart) will be fitted.
        preprocessor : Preprocessor, optional
            Fitted preprocessor. If None, fits preprocessor on default dataset.
        """
        self.dataset_name = dataset_name.lower().strip()
        if self.dataset_name not in ("wdbc", "heart"):
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Expected 'wdbc' or 'heart'.")

        self.validator = PatientRecordValidator(dataset_name=self.dataset_name)
        self.preprocessor = preprocessor
        self.model = model

        # Auto-initialize and fit if not provided
        if self.preprocessor is None or self.model is None:
            self._fit_default_pipeline()

    def _fit_default_pipeline(self) -> None:
        """Fit default preprocessor and model on training data."""
        if self.dataset_name == "wdbc":
            X, y, _ = load_wdbc(as_frame=False, return_feature_names=True)
            self.preprocessor = WDBCPreprocessor()
            X_proc = self.preprocessor.fit_transform(X)
            if self.model is None:
                self.model = ClassicalSVM(kernel="linear", C=0.1, probability=True, random_state=42)
            self.model.fit(X_proc, y)
        else:
            X, y, _ = load_heart_disease(as_frame=False, preprocess=False, return_feature_names=True)
            self.preprocessor = HeartDiseasePreprocessor()
            X_proc = self.preprocessor.fit_transform(X)
            if self.model is None:
                self.model = ClassicalRandomForest(n_estimators=100, max_depth=4, random_state=42)
            self.model.fit(X_proc, y)

    def predict_patient(self, raw_patient_data: Dict[str, Any]) -> ClinicalInferenceResult:
        """Execute hardened single-patient diagnostic inference.

        Parameters
        ----------
        raw_patient_data : Dict[str, Any]
            Dictionary containing patient raw features.

        Returns
        -------
        ClinicalInferenceResult
            Validated diagnostic outcome, risk score, and execution latency.
        """
        start_time = time.perf_counter()

        # Step 1: Input Validation Guardrails
        try:
            sanitized_dict = self.validator.validate_patient(raw_patient_data)
        except ValidationError as ve:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ClinicalInferenceResult(
                dataset=self.dataset_name,
                is_valid=False,
                predicted_class=None,
                class_label=None,
                risk_score=None,
                risk_category=None,
                model_name=getattr(self.model, "model_name", type(self.model).__name__),
                latency_ms=round(elapsed_ms, 3),
                validation_error=str(ve),
                sanitized_features=None,
            )

        # Step 2: Convert sanitized dict to 2D numpy array matching column order
        cols = self.validator.expected_columns
        patient_row = np.array([[sanitized_dict[c] for c in cols]], dtype=np.float64)

        # Step 3: Leak-Free Preprocessor Transform
        patient_transformed = self.preprocessor.transform(patient_row)

        # Step 4: Model Prediction and Calibrated Probability
        pred_class = int(self.model.predict(patient_transformed)[0])
        prob = self.model.predict_proba(patient_transformed)[0]
        # Positive class probability (index 1)
        risk_score = float(prob[1] if len(prob) > 1 else prob[0])

        # Step 5: Risk Categorization
        if risk_score < 0.30:
            category = "Low Risk"
        elif risk_score < 0.70:
            category = "Moderate Risk"
        else:
            category = "High Risk"

        label = self.DIAGNOSIS_LABELS[self.dataset_name].get(pred_class, f"Class {pred_class}")
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ClinicalInferenceResult(
            dataset=self.dataset_name,
            is_valid=True,
            predicted_class=pred_class,
            class_label=label,
            risk_score=round(risk_score, 4),
            risk_category=category,
            model_name=getattr(self.model, "model_name", type(self.model).__name__),
            latency_ms=round(elapsed_ms, 3),
            validation_error=None,
            sanitized_features=sanitized_dict,
        )
