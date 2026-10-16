"""Unit and edge-case tests for input sanitization and single-sample inference hardening.

Test File: tests/test_data_validation.py
"""

import numpy as np
import pandas as pd
import pytest

from src.data.validator import (
    DataValidator,
    ValidationError,
    MissingColumnError,
    InvalidDataTypeError,
    OutOfRangeSensorError,
)
from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease


def test_heart_single_sample_valid():
    """Verify clean validation for a single realistic patient record."""
    validator = DataValidator(dataset_name="heart")
    valid_patient = {
        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 240,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
    df = validator.validate_single_record(valid_patient)
    assert df.shape == (1, 13)
    assert df["age"].iloc[0] == 55.0
    assert df["chol"].iloc[0] == 240.0


def test_missing_column_rejection():
    """Verify MissingColumnError is raised with missing features."""
    validator = DataValidator(dataset_name="heart")
    incomplete_patient = {
        "age": 60,
        "sex": 0,
        # Missing 'cp', 'trestbps', etc.
    }
    with pytest.raises(MissingColumnError) as exc_info:
        validator.validate_single_record(incomplete_patient)
    assert "Missing required input column" in str(exc_info.value)
    assert "cp" in exc_info.value.missing_cols


def test_invalid_datatype_rejection():
    """Verify InvalidDataTypeError when string cannot be converted to numerical float."""
    validator = DataValidator(dataset_name="heart")
    corrupt_patient = {
        "age": "invalid_age_str",
        "sex": 1,
        "cp": 1,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 140,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }
    with pytest.raises(InvalidDataTypeError) as exc_info:
        validator.validate_single_record(corrupt_patient)
    assert "Invalid data type for column 'age'" in str(exc_info.value)


def test_out_of_range_sensor_rejection():
    """Verify OutOfRangeSensorError when sensor measurements are outside physiological bounds."""
    validator = DataValidator(dataset_name="heart")
    extreme_patient = {
        "age": 50,
        "sex": 1,
        "cp": 1,
        "trestbps": 450.0,  # Lethal/impossible blood pressure
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 140,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }
    with pytest.raises(OutOfRangeSensorError) as exc_info:
        validator.validate_single_record(extreme_patient)
    assert "Sensor value out of valid physiological range" in str(exc_info.value)
    assert "trestbps" in str(exc_info.value)


def test_invalid_category_rejection():
    """Verify ValidationError when categorical value is outside discrete schema."""
    validator = DataValidator(dataset_name="heart")
    invalid_cat_patient = {
        "age": 50,
        "sex": 1,
        "cp": 9.0,  # Only 1, 2, 3, 4 allowed
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 0,
        "thalach": 140,
        "exang": 0,
        "oldpeak": 0.0,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }
    with pytest.raises(ValidationError) as exc_info:
        validator.validate_single_record(invalid_cat_patient)
    assert "Invalid category for 'cp'" in str(exc_info.value)


def test_wdbc_batch_validation():
    """Verify DataValidator batch validation on WDBC dataset."""
    X, y, feature_names = load_wdbc(as_frame=True, return_feature_names=True)
    validator = DataValidator(dataset_name="wdbc")
    df_valid = validator.validate_batch(X)
    assert df_valid.shape == (569, 30)
    assert list(df_valid.columns) == feature_names
