"""Input Sanitization, Validation Guardrails, and Single-Sample Inference Hardening.

Module: src.data.validator
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D2/D5 (Inference Hardening)
Task: T5.1
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.data.heart_loader import RAW_COLUMNS, NUMERICAL_COLS, CATEGORICAL_COLS, PASSTHROUGH_COLS


class ValidationError(ValueError):
    """Base exception for input sanitization failures."""
    pass


class MissingColumnError(ValidationError):
    """Raised when required biomedical columns are missing from input."""

    def __init__(self, missing_cols: List[str]):
        self.missing_cols = missing_cols
        super().__init__(f"Missing required input column(s): {', '.join(missing_cols)}")


class InvalidDataTypeError(ValidationError):
    """Raised when an input value cannot be coerced to numerical float/int."""

    def __init__(self, col: str, val: Any, reason: str):
        self.col = col
        self.val = val
        self.reason = reason
        super().__init__(f"Invalid data type for column '{col}': received '{val}' ({type(val).__name__}). Reason: {reason}")


class OutOfRangeSensorError(ValidationError):
    """Raised when physiological sensor measurements fall outside valid physical boundaries."""

    def __init__(self, col: str, val: float, min_val: float, max_val: float):
        self.col = col
        self.val = val
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(
            f"Sensor value out of valid physiological range for '{col}': "
            f"received {val}, allowed [{min_val}, {max_val}]."
        )


# Physiological validation boundaries for Heart Disease Cleveland Dataset
HEART_VALIDATION_SCHEMA: Dict[str, Dict[str, Any]] = {
    "age": {"type": float, "min": 18.0, "max": 120.0},
    "sex": {"type": int, "allowed": [0, 1]},
    "cp": {"type": float, "allowed": [1.0, 2.0, 3.0, 4.0]},
    "trestbps": {"type": float, "min": 50.0, "max": 260.0},
    "chol": {"type": float, "min": 50.0, "max": 650.0},
    "fbs": {"type": int, "allowed": [0, 1]},
    "restecg": {"type": float, "allowed": [0.0, 1.0, 2.0]},
    "thalach": {"type": float, "min": 40.0, "max": 260.0},
    "exang": {"type": int, "allowed": [0, 1]},
    "oldpeak": {"type": float, "min": 0.0, "max": 10.0},
    "slope": {"type": float, "allowed": [1.0, 2.0, 3.0]},
    "ca": {"type": float, "allowed": [0.0, 1.0, 2.0, 3.0], "allow_none": True},
    "thal": {"type": float, "allowed": [3.0, 6.0, 7.0], "allow_none": True},
}


def get_wdbc_feature_names() -> List[str]:
    """Retrieve standard 30 feature names for WDBC."""
    raw = load_breast_cancer()
    return [str(f) for f in raw.feature_names]


class DataValidator:
    """Rigorous input sanitizer for clinical inference pipelines."""

    def __init__(self, dataset_name: str = "wdbc"):
        self.dataset_name = dataset_name.lower().strip()
        if self.dataset_name in ["wdbc", "breast_cancer"]:
            self.expected_cols = get_wdbc_feature_names()
            self.schema = {col: {"type": float, "min": 0.0, "max": 10000.0} for col in self.expected_cols}
        elif self.dataset_name in ["heart", "heart_disease", "cleveland"]:
            self.expected_cols = [c for c in RAW_COLUMNS if c != "target"]
            self.schema = HEART_VALIDATION_SCHEMA
        else:
            raise ValueError(f"Unsupported dataset for validation: '{dataset_name}'")

    def validate_single_record(self, record: Dict[str, Any]) -> pd.DataFrame:
        """Validate and sanitize a single patient record dictionary for inference."""
        if not isinstance(record, dict):
            raise ValidationError(f"Expected patient record as dict, got {type(record).__name__}")

        # Check for missing columns
        missing = [col for col in self.expected_cols if col not in record]
        if missing:
            raise MissingColumnError(missing)

        sanitized_row: Dict[str, Any] = {}
        for col in self.expected_cols:
            val = record[col]
            sanitized_val = self._validate_value(col, val)
            sanitized_row[col] = sanitized_val

        return pd.DataFrame([sanitized_row], columns=self.expected_cols)

    def validate_batch(self, data: Union[pd.DataFrame, np.ndarray, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Validate a batch of patient records."""
        if isinstance(data, list):
            rows = [self.validate_single_record(item).iloc[0].to_dict() for item in data]
            return pd.DataFrame(rows, columns=self.expected_cols)
        elif isinstance(data, pd.DataFrame):
            # Check column completeness
            missing = [c for c in self.expected_cols if c not in data.columns]
            if missing:
                raise MissingColumnError(missing)
            df_subset = data[self.expected_cols].copy()
            for col in self.expected_cols:
                for idx, val in df_subset[col].items():
                    df_subset.at[idx, col] = self._validate_value(col, val)
            return df_subset.astype(np.float64)
        elif isinstance(data, np.ndarray):
            arr = np.asarray(data)
            if arr.ndim == 1:
                if len(arr) != len(self.expected_cols):
                    raise ValidationError(f"Expected 1D array with {len(self.expected_cols)} features, got {len(arr)}")
                return self.validate_single_record(dict(zip(self.expected_cols, arr)))
            elif arr.ndim == 2:
                if arr.shape[1] != len(self.expected_cols):
                    raise ValidationError(f"Expected 2D array with {len(self.expected_cols)} columns, got {arr.shape[1]}")
                df = pd.DataFrame(arr, columns=self.expected_cols)
                return self.validate_batch(df)
            else:
                raise ValidationError(f"Unsupported array dimensions: {arr.ndim}")
        else:
            raise ValidationError(f"Unsupported data type for validation: {type(data).__name__}")

    def _validate_value(self, col: str, val: Any) -> float:
        """Validate single attribute against schema rules."""
        rules = self.schema.get(col, {})
        
        # Check None/NaN allowance
        if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
            if rules.get("allow_none", False):
                return float("nan")
            else:
                raise InvalidDataTypeError(col, val, "Field cannot be null/NaN")

        # Type conversion
        try:
            num_val = float(val)
        except (ValueError, TypeError):
            raise InvalidDataTypeError(col, val, "Cannot convert to numerical float")

        # Check discrete allowed values
        if "allowed" in rules:
            if num_val not in rules["allowed"]:
                raise ValidationError(
                    f"Invalid category for '{col}': received {val}, allowed: {rules['allowed']}"
                )

        # Check range constraints
        min_bound = rules.get("min")
        max_bound = rules.get("max")
        if min_bound is not None and num_val < min_bound:
            raise OutOfRangeSensorError(col, num_val, min_bound, max_bound or float("inf"))
        if max_bound is not None and num_val > max_bound:
            raise OutOfRangeSensorError(col, num_val, min_bound or float("-inf"), max_bound)

        return num_val

    @property
    def expected_columns(self) -> List[str]:
        """Alias for expected_cols."""
        return self.expected_cols

    def validate_patient(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Validate and return sanitized dictionary of float values."""
        df = self.validate_single_record(record)
        return df.iloc[0].to_dict()


# Semantic Alias for Clinical Patient Inference
PatientRecordValidator = DataValidator
