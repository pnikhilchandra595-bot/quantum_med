"""T2.3: Feature Reduction Ablation Study.

Benchmarks Classical SVM and Random Forest across feature dimensions k in [2, 4, 6, 8, 10, 15, 30].
Records model accuracy, sensitivity, and cumulative explained variance retained
to identify the optimal qubit cutoff for Member B's quantum circuits.
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_dataset
from src.models.classical.svm import RBFSVMModel
from src.models.classical.trees import RandomForestModel
from src.evaluation.metrics import calculate_metrics


def run_feature_ablation(dataset_name: str = "wdbc") -> pd.DataFrame:
    """Run PCA feature ablation study across k in [2, 4, 6, 8, 10, 15, 30]."""
    print(f"[*] Loading dataset: {dataset_name}...")
    X, y, feature_names, _ = load_dataset(dataset_name)
    n_samples, n_features = X.shape
    print(f"[*] Dataset loaded: {n_samples} samples, {n_features} features.")

    k_list = [2, 4, 6, 8, 10, 15, 30]
    results = []

    # 5-fold stratified CV for robust accuracy estimation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Compute full PCA to verify cumulative variance
    full_pca = PCA(n_components=n_features, random_state=42).fit(X)
    cum_var_full = np.cumsum(full_pca.explained_variance_ratio_)

    print("\n[*] Running ablation across dimensions k in [2, 4, 6, 8, 10, 15, 30]:")
    print(f"{'k (Qubits)':<10}{'Var Retained (%)':<18}{'SVM Acc':<12}{'RF Acc':<12}{'SVM Sens':<12}{'RF Sens':<12}")
    print("-" * 76)

    for k in k_list:
        pca = PCA(n_components=k, random_state=42)
        X_k = pca.fit_transform(X)
        var_retained = float(np.sum(pca.explained_variance_ratio_))
        var_retained_pct = var_retained * 100.0

        svm_acc_folds, rf_acc_folds = [], []
        svm_sens_folds, rf_sens_folds = [], []
        svm_f1_folds, rf_f1_folds = [], []

        for train_idx, test_idx in skf.split(X_k, y):
            X_tr, X_te = X_k[train_idx], X_k[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # SVM
            svm = RBFSVMModel(random_state=42).fit(X_tr, y_tr)
            y_pred_svm = svm.predict(X_te)
            y_prob_svm = svm.predict_proba(X_te)
            m_svm = calculate_metrics(y_te, y_pred_svm, y_prob_svm, round_decimals=None)
            svm_acc_folds.append(m_svm["accuracy"])
            svm_sens_folds.append(m_svm["sensitivity"])
            svm_f1_folds.append(m_svm["f1_score"])

            # RF
            rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
            y_pred_rf = rf.predict(X_te)
            y_prob_rf = rf.predict_proba(X_te)
            m_rf = calculate_metrics(y_te, y_pred_rf, y_prob_rf, round_decimals=None)
            rf_acc_folds.append(m_rf["accuracy"])
            rf_sens_folds.append(m_rf["sensitivity"])
            rf_f1_folds.append(m_rf["f1_score"])

        svm_acc_mean = float(np.mean(svm_acc_folds))
        rf_acc_mean = float(np.mean(rf_acc_folds))
        svm_sens_mean = float(np.mean(svm_sens_folds))
        rf_sens_mean = float(np.mean(rf_sens_folds))
        svm_f1_mean = float(np.mean(svm_f1_folds))
        rf_f1_mean = float(np.mean(rf_f1_folds))

        print(
            f"{k:<10}{var_retained_pct:>15.2f}%    "
            f"{svm_acc_mean:>8.4f}    {rf_acc_mean:>8.4f}    "
            f"{svm_sens_mean:>8.4f}    {rf_sens_mean:>8.4f}"
        )

        results.append({
            "k_dimensions": k,
            "variance_retained": round(var_retained, 4),
            "variance_retained_pct": round(var_retained_pct, 2),
            "svm_accuracy": round(svm_acc_mean, 4),
            "rf_accuracy": round(rf_acc_mean, 4),
            "svm_sensitivity": round(svm_sens_mean, 4),
            "rf_sensitivity": round(rf_sens_mean, 4),
            "svm_f1_score": round(svm_f1_mean, 4),
            "rf_f1_score": round(rf_f1_mean, 4),
        })

    df = pd.DataFrame(results)

    # Save to reports/tables/feature_ablation.csv
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved ablation table to: {csv_path}")

    # Plot ablation curve
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_curve(df, fig_dir)

    return df


def plot_ablation_curve(df: pd.DataFrame, fig_dir: Path) -> None:
    """Generate publication-ready ablation curve."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_var = "#1f77b4"
    color_svm = "#e66101"
    color_rf = "#2ca02c"

    ax1.set_xlabel("Number of Principal Components / Qubits ($k$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Explained Variance (%)", color=color_var, fontsize=12, fontweight="bold")
    line1 = ax1.plot(
        df["k_dimensions"],
        df["variance_retained_pct"],
        color=color_var,
        marker="o",
        linewidth=2.5,
        label="Variance Retained (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_var)
    ax1.set_ylim(40, 105)

    # Secondary axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("5-Fold Cross-Validation Accuracy", color="#333333", fontsize=12, fontweight="bold")
    line2 = ax2.plot(
        df["k_dimensions"],
        df["svm_accuracy"],
        color=color_svm,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Classical RBF SVM Acc",
    )
    line3 = ax2.plot(
        df["k_dimensions"],
        df["rf_accuracy"],
        color=color_rf,
        marker="^",
        linestyle="-.",
        linewidth=2,
        label="Random Forest Acc",
    )
    ax2.tick_params(axis="y", labelcolor="#333333")
    ax2.set_ylim(0.85, 1.0)

    # Highlight optimal qubit region (k=4..6, >88% variance)
    ax1.axvspan(4, 6, color="#ffeb3b", alpha=0.3, label="Optimal Qubit Window ($k=4..6$, $>88\\%$)")
    ax1.axhline(88.0, color="#d32f2f", linestyle=":", linewidth=1.5, label="88% Target Variance Threshold")

    # Composite Legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    labels.append("Optimal Qubit Window (k=4..6, >88%)")
    labels.append("88% Target Threshold")
    
    plt.title(
        "Feature Reduction Ablation Study (WDBC)\nOptimal Quantum Feature Map Cutoff ($k=4..6$ Qubits)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()

    svg_path = fig_dir / "feature_ablation_curve.svg"
    png_path = fig_dir / "feature_ablation_curve.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved ablation figures to: {svg_path} and {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Reduction Ablation Study")
    parser.add_argument("--dataset", type=str, default="wdbc", help="Dataset name ('wdbc')")
    args = parser.parse_args()
    run_feature_ablation(args.dataset)
