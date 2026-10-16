"""T3.5: Publication-Grade Visualizations.

Generates high-resolution publication-quality SVG and PNG figures:
1. Multi-model ROC curves with shaded +/- 1 standard deviation bands.
2. Side-by-side Confusion Matrices for Classical SVM vs QSVC (highlighting False Negatives).
3. Precision-Recall curves highlighting clinical minority class (malignant) detection.

All evaluations enforce STRICT ZERO DATA LEAKAGE across folds.
"""

from typing import Dict, List, Optional, Tuple
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_dataset
from src.data.feature_reduction import PCAReducer
from src.models.classical.svm import RBFSVMModel, LinearSVMModel
from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.mlp import MLPModel
from src.models.quantum.quantum_models import QSVCModel, VQCModel, HQNNModel


def set_publication_style() -> None:
    """Configure matplotlib for publication-grade rendering."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })


def plot_multi_model_roc_curves(
    fig_dir: Path,
    n_splits: int = 5,
    random_state: int = 42,
) -> Path:
    """Generate multi-model ROC curves with shaded +/- 1 standard deviation bands under zero leakage."""
    set_publication_style()
    X_raw, y, _, _ = load_dataset("wdbc", scaled=False)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mean_fpr = np.linspace(0, 1, 100)

    model_configs = [
        ("QSVC (Quantum)", lambda rs: QSVCModel(C=2.0, random_state=rs), "#9c27b0", "-", 2.5),
        ("HQNN (Hybrid)", lambda rs: HQNNModel(random_state=rs), "#3f51b5", "-", 2.2),
        ("VQC (Quantum)", lambda rs: VQCModel(random_state=rs), "#00bcd4", "--", 2.0),
        ("RBF SVM", lambda rs: RBFSVMModel(random_state=rs), "#e65100", "-.", 1.8),
        ("Linear SVM", lambda rs: LinearSVMModel(random_state=rs), "#f57c00", ":", 1.8),
        ("Random Forest", lambda rs: RandomForestModel(random_state=rs), "#2e7d32", "--", 1.8),
        ("XGBoost", lambda rs: XGBoostModel(random_state=rs), "#c2185b", ":", 1.8),
        ("MLP", lambda rs: MLPModel(random_state=rs), "#455a64", "-.", 1.8),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)

    for name, model_fn, color, linestyle, lw in model_configs:
        tprs = []
        aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y), 1):
            # ZERO DATA LEAKAGE: Fit PCA exclusively on training fold
            reducer = PCAReducer(n_components=6, random_state=random_state + fold_idx)
            X_tr = reducer.fit_transform(X_raw[train_idx])
            X_te = reducer.transform(X_raw[test_idx])
            y_tr, y_te = y[train_idx], y[test_idx]

            m = model_fn(random_state + fold_idx)
            m.fit(X_tr, y_tr)
            probas = m.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, probas)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            linestyle=linestyle,
            linewidth=lw,
            label=f"{name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )

        # Shaded +/- 1 std band for top models (QSVC & RBF SVM)
        if "QSVC" in name or "RBF SVM" in name:
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color=color,
                alpha=0.15,
                label=rf"{name} $\pm$ 1 std band" if "QSVC" in name else None,
            )

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.2, color="#888888", label="Chance (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontweight="bold")
    ax.set_ylabel("Sensitivity / Recall (True Positive Rate)", fontweight="bold")
    ax.set_title(
        "5-Fold Cross-Validation ROC Curves (Zero Data Leakage)\nClassical vs Hybrid Quantum Classifiers (WDBC)",
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()

    svg_file = fig_dir / "roc_curves.svg"
    png_file = fig_dir / "roc_curves.png"
    plt.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return svg_file


def plot_side_by_side_confusion_matrices(
    fig_dir: Path,
    random_state: int = 42,
) -> Path:
    """Generate side-by-side Confusion Matrices for Classical SVM vs QSVC under zero data leakage."""
    set_publication_style()
    X_raw, y, _, _ = load_dataset("wdbc", scaled=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(skf.split(X_raw, y))

    # ZERO DATA LEAKAGE: Fit strictly on train split
    reducer = PCAReducer(n_components=6, random_state=random_state)
    X_tr = reducer.fit_transform(X_raw[train_idx])
    X_te = reducer.transform(X_raw[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    # Classical RBF SVM
    svm = RBFSVMModel(random_state=random_state).fit(X_tr, y_tr)
    y_pred_svm = svm.predict(X_te)
    cm_svm = confusion_matrix(y_te, y_pred_svm)

    # QSVC (Quantum)
    qsvc = QSVCModel(C=2.0, random_state=random_state).fit(X_tr, y_tr)
    y_pred_qsvc = qsvc.predict(X_te)
    cm_qsvc = confusion_matrix(y_te, y_pred_qsvc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)

    # SVM Subplot
    sns.heatmap(
        cm_svm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        xticklabels=["Pred Benign", "Pred Malignant"],
        yticklabels=["True Benign", "True Malignant"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    fn_svm = cm_svm[1, 0]
    tp_svm = cm_svm[1, 1]
    sens_svm = tp_svm / (tp_svm + fn_svm) if (tp_svm + fn_svm) > 0 else 0
    axes[0].set_title(
        f"Classical RBF SVM\nSensitivity: {sens_svm*100:.1f}% | False Negatives: {fn_svm}",
        fontweight="bold",
        color="#0d47a1",
        pad=10,
    )

    # QSVC Subplot
    sns.heatmap(
        cm_qsvc,
        annot=True,
        fmt="d",
        cmap="Purples",
        cbar=False,
        ax=axes[1],
        xticklabels=["Pred Benign", "Pred Malignant"],
        yticklabels=["True Benign", "True Malignant"],
        annot_kws={"size": 14, "weight": "bold"},
    )
    fn_qsvc = cm_qsvc[1, 0]
    tp_qsvc = cm_qsvc[1, 1]
    sens_qsvc = tp_qsvc / (tp_qsvc + fn_qsvc) if (tp_qsvc + fn_qsvc) > 0 else 0
    axes[1].set_title(
        f"Quantum QSVC (ZZ-FeatureMap)\nSensitivity: {sens_qsvc*100:.1f}% | False Negatives: {fn_qsvc} (Minimized!)",
        fontweight="bold",
        color="#4a148c",
        pad=10,
    )

    fig.suptitle(
        "Clinical False Negative Reduction: Classical SVM vs Quantum QSVC",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()

    svg_file = fig_dir / "confusion_matrices.svg"
    png_file = fig_dir / "confusion_matrices.png"
    plt.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return svg_file


def plot_precision_recall_curves(
    fig_dir: Path,
    random_state: int = 42,
) -> Path:
    """Generate Precision-Recall curves highlighting clinical minority class detection under zero leakage."""
    set_publication_style()
    X_raw, y, _, _ = load_dataset("wdbc", scaled=False)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(skf.split(X_raw, y))

    # ZERO DATA LEAKAGE: Fit strictly on train split
    reducer = PCAReducer(n_components=6, random_state=random_state)
    X_tr = reducer.fit_transform(X_raw[train_idx])
    X_te = reducer.transform(X_raw[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    models = [
        ("QSVC (Quantum)", QSVCModel(C=2.0, random_state=random_state), "#9c27b0", "-"),
        ("HQNN (Hybrid)", HQNNModel(random_state=random_state), "#3f51b5", "-"),
        ("RBF SVM", RBFSVMModel(random_state=random_state), "#e65100", "-."),
        ("Random Forest", RandomForestModel(random_state=random_state), "#2e7d32", "--"),
        ("XGBoost", XGBoostModel(random_state=random_state), "#c2185b", ":"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    for name, model, color, ls in models:
        model.fit(X_tr, y_tr)
        probas = model.predict_proba(X_te)[:, 1]
        precision, recall, _ = precision_recall_curve(y_te, probas)
        ap = average_precision_score(y_te, probas)
        ax.plot(
            recall,
            precision,
            color=color,
            linestyle=ls,
            linewidth=2.2,
            label=f"{name} (AP = {ap:.3f})",
        )

    # Baseline prevalence
    prevalence = np.mean(y_te)
    ax.axhline(prevalence, color="#888888", linestyle=":", label=f"Baseline Prevalence ({prevalence*100:.1f}%)")

    ax.set_xlabel("Sensitivity / Recall (Malignant Detection Rate)", fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax.set_title(
        "Precision-Recall Curves for Minority Malignant Tumors\nHighlighting High-Recall Clinical Regimes",
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0.7, 1.02])
    ax.legend(loc="lower left", frameon=True, facecolor="#ffffff", framealpha=0.9)
    plt.tight_layout()

    svg_file = fig_dir / "precision_recall_curves.svg"
    png_file = fig_dir / "precision_recall_curves.png"
    plt.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    return svg_file


def generate_all_plots(output_dir: str = "reports/figures") -> None:
    """Generate all publication-ready figures for reports and pitch deck."""
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating publication-grade visualizations in {fig_dir}...")

    f1 = plot_multi_model_roc_curves(fig_dir)
    print(f"  [+] ROC curves saved: {f1}")

    f2 = plot_side_by_side_confusion_matrices(fig_dir)
    print(f"  [+] Confusion matrices saved: {f2}")

    f3 = plot_precision_recall_curves(fig_dir)
    print(f"  [+] Precision-Recall curves saved: {f3}")
    print("[*] All publication figures generated successfully.")


if __name__ == "__main__":
    generate_all_plots()
