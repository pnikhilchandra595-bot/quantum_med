"""T3.3: Master 5-Fold Cross-Validation Benchmark Suite.

Orchestrates automated 5-fold cross-validation across all 7 models:
- Classical: Linear SVM, RBF SVM, Random Forest, XGBoost, MLP
- Quantum: QSVC, VQC, HQNN (ZZ-FeatureMap & Variational Circuits)

Logs mean and standard deviation for all 6 metrics under identical splits.
Saves master summary CSV to reports/tables/benchmark_results.csv.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

from src.data.loader import load_dataset
from src.evaluation.metrics import calculate_metrics, measure_inference_latency
from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.svm import LinearSVMModel, RBFSVMModel
from src.models.classical.mlp import MLPModel
from src.models.quantum.quantum_models import QSVCModel, VQCModel, HQNNModel


def get_all_models(random_state: int = 42) -> List[Any]:
    """Instantiate fresh instances of all 7 classical and quantum models."""
    return [
        LinearSVMModel(random_state=random_state),
        RBFSVMModel(random_state=random_state),
        RandomForestModel(n_estimators=100, max_depth=5, random_state=random_state),
        XGBoostModel(learning_rate=0.05, n_estimators=100, max_depth=4, random_state=random_state),
        MLPModel(hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state),
        QSVCModel(C=1.0, random_state=random_state),
        VQCModel(n_qubits=6, n_layers=3, random_state=random_state),
        HQNNModel(latent_qubits=4, random_state=random_state),
    ]


def run_benchmark(
    dataset_name: str = "wdbc",
    n_splits: int = 5,
    n_components: int = 6,
    quick: bool = False,
    random_state: int = 42,
    output_dir: str = "reports/tables",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, List[float]]]]:
    """Run automated 5-fold cross validation benchmark suite across all models.

    Args:
        dataset_name: Name of dataset (default: 'wdbc').
        n_splits: Number of CV folds (default: 5).
        n_components: PCA dimension matching optimal qubit cutoff (default: 6).
        quick: If True, uses 3 splits and faster training for rapid verification.
        random_state: Deterministic random seed (default: 42).
        output_dir: Directory where benchmark_results.csv is saved.

    Returns:
        summary_df: Formatted benchmark summary table with mean ± std.
        folds_data: Dictionary mapping model_name -> metric_name -> list of fold values.
    """
    actual_splits = 3 if quick else n_splits
    X, y, feature_names, target_names = load_dataset(dataset_name)

    # Dimensionality reduction to optimal qubit dimension k=6 (retaining >88% variance)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)

    # Initialize storage
    model_template_list = get_all_models(random_state)
    metric_keys = ["accuracy", "sensitivity", "specificity", "f1_score", "roc_auc", "latency_ms"]

    folds_data: Dict[str, Dict[str, List[float]]] = {
        m.name: {k: [] for k in metric_keys} for m in model_template_list
    }

    print(f"\n=========================================================================")
    print(f"  HQML Master Benchmark: {actual_splits}-Fold Cross-Validation on {dataset_name.upper()}")
    print(f"  Features: {n_components} Qubits/PCs (Retaining {np.sum(pca.explained_variance_ratio_)*100:.2f}% Variance)")
    print(f"=========================================================================\n")

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_reduced, y), 1):
        X_train, X_test = X_reduced[train_idx], X_reduced[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"[*] Processing Fold {fold_idx}/{actual_splits}...")

        # Fresh model instances per fold
        models_in_fold = get_all_models(random_state + fold_idx)

        for model in models_in_fold:
            # Fit
            model.fit(X_train, y_train)

            # Measure inference latency per sample
            lat = measure_inference_latency(model, X_test, n_warmup=1, n_runs=3)

            # Predictions
            y_pred = model.predict(X_test)
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

            m = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                latency_ms=lat,
                round_decimals=None,
            )

            for k in metric_keys:
                folds_data[model.name][k].append(m[k])

    # Compute mean and standard deviation
    summary_rows = []
    for model_name, metrics_dict in folds_data.items():
        row = {
            "Model": model_name,
            "Type": "Quantum" if model_name in ["QSVC", "VQC", "HQNN"] else "Classical",
        }
        for k in metric_keys:
            vals = metrics_dict[k]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{k}_mean"] = round(mean_v, 4)
            row[f"{k}_std"] = round(std_v, 4)
            row[f"{k} (mean±std)"] = f"{mean_v:.4f} ± {std_v:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save to disk
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "benchmark_results.csv"
    summary_df.to_csv(csv_file, index=False)
    print(f"\n[+] Master benchmark results saved to: {csv_file}")

    # Also save raw fold details for significance testing and plotting
    folds_file = out_path / "benchmark_folds.json"
    with open(folds_file, "w") as f:
        json.dump(folds_data, f, indent=2)
    print(f"[+] Fold-level metrics saved to: {folds_file}\n")

    return summary_df, folds_data
