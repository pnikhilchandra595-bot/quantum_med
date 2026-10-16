"""Unit and integration tests for Fold Integrity Audit, CrossValidationRunner, and Tuning.

Test File: tests/test_audit_and_tuning.py
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Task: T3.3 & T3.6 (Phase 3)
"""

from pathlib import Path
import json
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.evaluation.fold_audit import FoldIntegrityAuditor, audit_all_datasets, AuditFailureError
from src.evaluation.cv_runner import CrossValidationRunner
from src.models.classical import ClassicalSVM
from src.models.classical.tuner import PARAM_GRIDS


def test_fold_integrity_auditor_wdbc():
    """Verify WDBC fold integrity: disjointness, coverage, and zero leakage."""
    auditor = FoldIntegrityAuditor(dataset_name="wdbc")
    cert = auditor.run_full_audit()

    assert cert["status"] == "CERTIFIED_ZERO_LEAKAGE"
    assert cert["disjointness_and_coverage"]["exhaustive_coverage"] is True
    assert cert["disjointness_and_coverage"]["zero_index_overlap"] is True
    assert cert["data_leakage"]["zero_leakage_verified"] is True
    assert len(cert["sha256_hash"]) == 64


def test_fold_integrity_auditor_heart():
    """Verify Heart Disease fold integrity: disjointness, coverage, and zero leakage."""
    auditor = FoldIntegrityAuditor(dataset_name="heart")
    cert = auditor.run_full_audit()

    assert cert["status"] == "CERTIFIED_ZERO_LEAKAGE"
    assert cert["disjointness_and_coverage"]["exhaustive_coverage"] is True
    assert cert["disjointness_and_coverage"]["zero_index_overlap"] is True
    assert cert["data_leakage"]["zero_leakage_verified"] is True
    assert len(cert["sha256_hash"]) == 64


def test_audit_all_datasets_persists_certificate(tmp_path):
    """Verify audit_all_datasets exports valid JSON certificate."""
    cert_path = tmp_path / "test_audit_cert.json"
    full_audit = audit_all_datasets(output_audit_path=str(cert_path))

    assert cert_path.exists()
    with open(cert_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "wdbc_audit" in loaded
    assert "heart_audit" in loaded
    assert loaded["wdbc_audit"]["status"] == "CERTIFIED_ZERO_LEAKAGE"


def test_cv_runner_linear_svm():
    """Verify CrossValidationRunner executes leak-free 5-fold CV and collects OOF predictions."""
    runner = CrossValidationRunner(dataset_name="wdbc")
    results = runner.run(
        model_factory=lambda: ClassicalSVM(kernel="linear", C=1.0, probability=False, random_state=42),
        model_name="FastSVM",
    )

    assert results["n_splits"] == 5
    assert results["n_samples"] == 569
    assert len(results["fold_metrics"]) == 5
    assert len(results["oof_predictions"]) == 569

    summary = results["summary"]
    assert 0.90 <= summary["accuracy_mean"] <= 1.0
    assert summary["accuracy_std"] >= 0.0
    assert summary["train_latency_mean_ms"] > 0.0
    assert summary["infer_latency_mean_ms"] > 0.0


def test_param_grids_specifications():
    """Verify hyperparameter search space definitions are non-empty and well-formed."""
    for model_key in ["svm_linear", "svm_rbf", "random_forest", "xgboost", "mlp"]:
        assert model_key in PARAM_GRIDS
        grid = PARAM_GRIDS[model_key]
        assert len(grid) > 0
        for param_name, values in grid.items():
            assert len(values) >= 2, f"Grid for {model_key}.{param_name} has insufficient values"


def test_baselines_audit_page_renders():
    """Verify app/pages/3_Baselines_Audit.py renders without errors."""
    page_path = Path(__file__).resolve().parent.parent / "app" / "pages" / "3_Baselines_Audit.py"
    at = AppTest.from_file(page_path, default_timeout=25).run()
    assert len(at.exception) == 0, f"Exceptions when rendering page: {at.exception}"
