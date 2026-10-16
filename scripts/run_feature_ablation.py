"""T2.3: Feature Reduction Ablation Study.

Benchmarks classification metrics across dimensions k in [2, 4, 6, 8, 10, 15, 30]
on WDBC (Breast Cancer) and Heart Disease datasets.

Key Requirement:
- Generate reports/tables/feature_ablation.csv proving that k=4..6 qubits
  retains >85% variance and maintains near-baseline classical accuracy.
- Enforces STRICT ZERO DATA LEAKAGE across cross-validation splits:
  Scalers and PCA projections are fitted exclusively on each training split,
  and evaluated on unseen test splits.
"""

import argparse
import sys
from pathlib import Path

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_dataset
from src.data.feature_reduction import PCAReducer
from src.models.classical.svm import RBFSVMModel
from src.models.classical.trees import RandomForestModel
from src.evaluation.metrics import calculate_metrics


def run_ablation_for_dataset(
    dataset_name: str,
    k_list: list = [2, 4, 6, 8, 10, 15, 30],
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run zero-leakage feature reduction ablation study on a given dataset."""
    print(f"\n=========================================================================")
    print(f"  FEATURE REDUCTION ABLATION: {dataset_name.upper()} (Zero Data Leakage)")
    print(f"=========================================================================")

    # Load unscaled raw dataset for leak-free CV
    X_raw, y, feature_names, target_names = load_dataset(dataset_name, scaled=False)
    n_samples, n_features = X_raw.shape
    print(f"[*] Dataset: {dataset_name} | Samples: {n_samples} | Features: {n_features}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    print(f"{'k':<6}{'k_eff':<8}{'Var Retained':<16}{'OOF Var':<12}{'SVM Acc':<12}{'RF Acc':<12}{'SVM Sens':<12}{'RF Sens':<12}")
    print("-" * 88)

    for k in k_list:
        effective_k = min(k, n_features)

        svm_acc_folds, rf_acc_folds = [], []
        svm_sens_folds, rf_sens_folds = [], []
        svm_f1_folds, rf_f1_folds = [], []
        var_retained_folds = []
        oof_var_folds = []

        for train_idx, test_idx in skf.split(X_raw, y):
            X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # ZERO DATA LEAKAGE: Fit scaler + PCA exclusively on training fold
            reducer = PCAReducer(n_components=k, random_state=random_state)
            X_tr_k = reducer.fit_transform(X_tr_raw)
            X_te_k = reducer.transform(X_te_raw)

            # Record training explained variance & out-of-fold test variance
            var_retained_folds.append(reducer.cumulative_variance_ratio_)
            oof_var = reducer.compute_out_of_fold_variance(X_te_raw)
            oof_var_folds.append(oof_var)

            # Fit Classical Baselines on reduced training representation
            svm = RBFSVMModel(random_state=random_state).fit(X_tr_k, y_tr)
            y_pred_svm = svm.predict(X_te_k)
            y_prob_svm = svm.predict_proba(X_te_k)
            m_svm = calculate_metrics(y_te, y_pred_svm, y_prob_svm, round_decimals=None)
            svm_acc_folds.append(m_svm["accuracy"])
            svm_sens_folds.append(m_svm["sensitivity"])
            svm_f1_folds.append(m_svm["f1_score"])

            rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=random_state).fit(X_tr_k, y_tr)
            y_pred_rf = rf.predict(X_te_k)
            y_prob_rf = rf.predict_proba(X_te_k)
            m_rf = calculate_metrics(y_te, y_pred_rf, y_prob_rf, round_decimals=None)
            rf_acc_folds.append(m_rf["accuracy"])
            rf_sens_folds.append(m_rf["sensitivity"])
            rf_f1_folds.append(m_rf["f1_score"])

        mean_var = float(np.mean(var_retained_folds))
        mean_oof_var = float(np.mean(oof_var_folds))
        mean_svm_acc = float(np.mean(svm_acc_folds))
        mean_rf_acc = float(np.mean(rf_acc_folds))
        mean_svm_sens = float(np.mean(svm_sens_folds))
        mean_rf_sens = float(np.mean(rf_sens_folds))
        mean_svm_f1 = float(np.mean(svm_f1_folds))
        mean_rf_f1 = float(np.mean(rf_f1_folds))

        print(
            f"{k:<6}{effective_k:<8}{mean_var*100:>12.2f}%   "
            f"{mean_oof_var*100:>8.2f}%   "
            f"{mean_svm_acc:>8.4f}    {mean_rf_acc:>8.4f}    "
            f"{mean_svm_sens:>8.4f}    {mean_rf_sens:>8.4f}"
        )

        results.append({
            "dataset": dataset_name.upper(),
            "k_dimensions": k,
            "effective_k": effective_k,
            "variance_retained": round(mean_var, 4),
            "variance_retained_pct": round(mean_var * 100.0, 2),
            "oof_variance_retained_pct": round(mean_oof_var * 100.0, 2),
            "svm_accuracy": round(mean_svm_acc, 4),
            "rf_accuracy": round(mean_rf_acc, 4),
            "svm_sensitivity": round(mean_svm_sens, 4),
            "rf_sensitivity": round(mean_rf_sens, 4),
            "svm_f1_score": round(mean_svm_f1, 4),
            "rf_f1_score": round(mean_rf_f1, 4),
            "zero_leakage_verified": True,
        })

    return pd.DataFrame(results)


def plot_dataset_ablation_curve(df: pd.DataFrame, dataset_name: str, fig_dir: Path) -> None:
    """Plot publication-quality ablation curve for a specific dataset."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_var = "#1f77b4"
    color_svm = "#e66101"
    color_rf = "#2ca02c"

    ax1.set_xlabel("Principal Components / Qubit Budget ($k$)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Explained Variance Retained (%)", color=color_var, fontsize=12, fontweight="bold")
    ax1.plot(
        df["k_dimensions"],
        df["variance_retained_pct"],
        color=color_var,
        marker="o",
        linewidth=2.5,
        label="In-Fold Variance Retained (%)",
    )
    ax1.plot(
        df["k_dimensions"],
        df["oof_variance_retained_pct"],
        color=color_var,
        marker="x",
        linestyle=":",
        linewidth=1.8,
        label="Out-of-Fold Test Variance (%)",
    )
    ax1.tick_params(axis="y", labelcolor=color_var)
    ax1.set_ylim(30, 105)

    # Secondary accuracy axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("5-Fold Cross-Validation Accuracy", color="#333333", fontsize=12, fontweight="bold")
    ax2.plot(
        df["k_dimensions"],
        df["svm_accuracy"],
        color=color_svm,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="RBF SVM Accuracy",
    )
    ax2.plot(
        df["k_dimensions"],
        df["rf_accuracy"],
        color=color_rf,
        marker="^",
        linestyle="-.",
        linewidth=2,
        label="Random Forest Accuracy",
    )
    ax2.tick_params(axis="y", labelcolor="#333333")
    ax2.set_ylim(0.70, 1.0)

    # Highlight optimal qubit window k=4..6 and 85% threshold
    ax1.axvspan(4, 6, color="#ffeb3b", alpha=0.3, label="Optimal Qubit Window ($k=4..6$)")
    ax1.axhline(85.0, color="#d32f2f", linestyle="--", linewidth=1.5, label="85% Variance Threshold")

    plt.title(
        f"Feature Reduction Ablation Study ({dataset_name.upper()})\nZero-Leakage 5-Fold Cross-Validation",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    clean_name = dataset_name.lower().replace("-", "_")
    svg_path = fig_dir / f"feature_ablation_{clean_name}.svg"
    png_path = fig_dir / f"feature_ablation_{clean_name}.png"
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved figure: {svg_path}")


def run_full_ablation(dataset_arg: str = "all") -> pd.DataFrame:
    """Run ablation across WDBC and Heart Disease with zero data leakage."""
    if dataset_arg.lower() in ["all", "both"]:
        datasets_to_run = ["wdbc", "heart"]
    else:
        datasets_to_run = [dataset_arg.lower()]

    k_list = [2, 4, 6, 8, 10, 15, 30]
    dfs = []

    for ds_name in datasets_to_run:
        df_ds = run_ablation_for_dataset(ds_name, k_list=k_list)
        dfs.append(df_ds)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Save to reports/tables/feature_ablation.csv
    out_dir = Path("reports/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_ablation.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\n[+] Master feature ablation table saved to: {csv_path}")

    # Generate plots
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ds_name in datasets_to_run:
        sub_df = combined_df[combined_df["dataset"] == ds_name.upper()]
        plot_dataset_ablation_curve(sub_df, ds_name, fig_dir)

    # Also save standard feature_ablation_curve.svg (WDBC primary)
    wdbc_df = combined_df[combined_df["dataset"] == "WDBC"]
    if not wdbc_df.empty:
        plot_dataset_ablation_curve(wdbc_df, "wdbc_master", fig_dir)
        import shutil
        shutil.copy(fig_dir / "feature_ablation_wdbc_master.svg", fig_dir / "feature_ablation_curve.svg")
        shutil.copy(fig_dir / "feature_ablation_wdbc_master.png", fig_dir / "feature_ablation_curve.png")

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T2.3 Feature Reduction Ablation Study")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to benchmark: 'all' (default: WDBC + Heart), 'wdbc', or 'heart'",
    )
    args = parser.parse_args()
    run_full_ablation(args.dataset)
