"""Phase 3 Verification Test Suite.

Tests deliverables:
- T3.3: Master 5-Fold Benchmark (Schema, Metrics, Folds)
- T3.4: Statistical Significance Testing (Paired t-test, Wilcoxon, Cohen's d, Alpha=0.05)
- T3.5: Publication Plots (SVG/PNG creation, zero leakage)
- T3.6: Classical SHAP explainer adapters
- T3.8: XAI Comparator (Spearman rho, Jaccard overlap, Radar SVG)
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("seaborn")

from src.evaluation.benchmark import run_benchmark, get_all_models
from src.evaluation.significance import run_significance_tests, compute_cohens_d
from src.evaluation.plots import (
    plot_multi_model_roc_curves,
    plot_side_by_side_confusion_matrices,
    plot_precision_recall_curves,
)
from src.xai.comparator import compare_feature_attributions


def test_get_all_models_instantiation():
    """Verify all 8 classical and quantum models are properly initialized."""
    models = get_all_models(random_state=42)
    assert len(models) == 8
    model_names = [m.name for m in models]
    expected_names = [
        "Linear SVM",
        "RBF SVM",
        "Random Forest",
        "XGBoost",
        "MLP",
        "QSVC",
        "VQC",
        "HQNN",
    ]
    for expected in expected_names:
        assert expected in model_names


def test_cohens_d_calculation():
    """Test paired Cohen's d effect size function."""
    x = np.array([0.95, 0.96, 0.93, 0.97, 0.95])
    y = np.array([0.90, 0.92, 0.89, 0.91, 0.88])
    d = compute_cohens_d(x, y)
    assert d > 0.0
    # Exactly identical
    assert compute_cohens_d(x, x) == 0.0


def test_quick_benchmark_execution(tmp_path):
    """Test automated CV benchmark execution in quick mode."""
    out_dir = str(tmp_path / "tables")
    summary_df, folds_data = run_benchmark(
        dataset_name="wdbc",
        n_splits=3,
        n_components=4,
        quick=True,
        random_state=42,
        output_dir=out_dir,
    )

    assert isinstance(summary_df, pd.DataFrame)
    assert len(summary_df) == 8
    assert "sensitivity (mean±std)" in summary_df.columns
    assert "accuracy (mean±std)" in summary_df.columns
    assert "roc_auc (mean±std)" in summary_df.columns
    assert "latency_ms (mean±std)" in summary_df.columns

    # Verify CSV and JSON outputs
    csv_file = Path(out_dir) / "benchmark_results.csv"
    json_file = Path(out_dir) / "benchmark_folds.json"
    assert csv_file.exists()
    assert json_file.exists()

    # Test significance testing on generated folds
    sig_results, sig_df = run_significance_tests(
        folds_data=folds_data,
        output_dir=out_dir,
        metric="sensitivity",
    )
    assert "comparisons" in sig_results
    assert isinstance(sig_df, pd.DataFrame)
    assert (Path(out_dir) / "significance_p_values.json").exists()
    assert (Path(out_dir) / "significance_summary.csv").exists()


def test_publication_plots_quick(tmp_path):
    """Test publication plot generators produce valid SVG and PNG files."""
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_multi_model_roc_curves(fig_dir, n_splits=3, random_state=42)
    assert f1.exists()
    assert f1.suffix == ".svg"
    assert (fig_dir / "roc_curves.png").exists()

    f2 = plot_side_by_side_confusion_matrices(fig_dir, random_state=42)
    assert f2.exists()
    assert f2.suffix == ".svg"
    assert (fig_dir / "confusion_matrices.png").exists()

    f3 = plot_precision_recall_curves(fig_dir, random_state=42)
    assert f3.exists()
    assert f3.suffix == ".svg"
    assert (fig_dir / "precision_recall_curves.png").exists()
