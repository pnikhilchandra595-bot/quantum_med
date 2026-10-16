"""T3.8: Classical vs Quantum Feature Attribution Comparator.

Compares top-5 clinical risk drivers identified by Classical models vs Quantum (QSVC/VQC).
Calculates interpretability alignment metrics:
1. Spearman rank correlation of feature importances.
2. Top-5 feature overlap (Jaccard index).
3. Radar/spider chart comparing classical vs quantum attribution profiles.

Outputs:
- reports/figures/shap_radar_comparison.svg & .png
- reports/tables/xai_comparison_report.json
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.data.loader import load_dataset
from src.models.classical.trees import RandomForestModel
from src.models.classical.svm import RBFSVMModel
from src.models.quantum.quantum_models import QSVCModel, VQCModel
from src.xai.classical_shap import ClassicalSHAPExplainer
from src.xai.quantum_shap import QuantumKernelSHAP


def compare_feature_attributions(
    dataset_name: str = "wdbc",
    n_components: int = 6,
    random_state: int = 42,
    output_dir: str = "reports",
) -> Dict[str, Any]:
    """Execute comparative XAI analysis between Classical and Quantum models."""
    print("[*] Loading clinical dataset for XAI attribution comparison...")
    X, y, feature_names, _ = load_dataset(dataset_name)

    # Use PCA features matching qubit register
    pca = PCA(n_components=n_components, random_state=random_state)
    X_k = pca.fit_transform(X)
    component_labels = [f"PC{i+1} (Qubit {i})" for i in range(n_components)]

    # Split background and evaluation sets
    bg_size = 40
    eval_size = 60
    X_bg = X_k[:bg_size]
    X_eval = X_k[bg_size : bg_size + eval_size]
    y_eval = y[bg_size : bg_size + eval_size]

    print("[*] Training classical baselines and quantum models...")
    rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=random_state).fit(X_k, y)
    svm = RBFSVMModel(random_state=random_state).fit(X_k, y)
    qsvc = QSVCModel(C=2.0, random_state=random_state).fit(X_k, y)
    vqc = VQCModel(random_state=random_state).fit(X_k, y)

    print("[*] Computing SHAP attributions...")
    rf_explainer = ClassicalSHAPExplainer(rf, X_bg, component_labels)
    svm_explainer = ClassicalSHAPExplainer(svm, X_bg, component_labels)
    qsvc_explainer = QuantumKernelSHAP(qsvc, X_bg, component_labels)
    vqc_explainer = QuantumKernelSHAP(vqc, X_bg, component_labels)

    rf_exp = rf_explainer.explain(X_eval)
    svm_exp = svm_explainer.explain(X_eval, nsamples=50)
    qsvc_exp = qsvc_explainer.explain(X_eval, nsamples=50)
    vqc_exp = vqc_explainer.explain(X_eval, nsamples=50)

    models_dict = {
        "Random Forest": rf_exp,
        "Classical SVM": svm_exp,
        "Quantum QSVC": qsvc_exp,
        "Quantum VQC": vqc_exp,
    }

    print("\n=========================================================================")
    print("  CLINICAL RISK DRIVER ATTRIBUTION SUMMARY (NORMALIZED SHAP IMPORTANCE)")
    print("=========================================================================\n")
    header = f"{'Feature':<20}" + "".join([f"{name:<18}" for name in models_dict.keys()])
    print(header)
    print("-" * len(header))

    importances_matrix = {}
    for name, exp in models_dict.items():
        importances_matrix[name] = exp.normalized_importance

    for idx, feat in enumerate(component_labels):
        row = f"{feat:<20}"
        for name in models_dict.keys():
            val = importances_matrix[name][idx]
            row += f"{val:>14.4f}    "
        print(row)

    # Alignment Metrics: Spearman Rank Correlation & Top-5 Overlap
    print("\n=========================================================================")
    print("  XAI ALIGNMENT METRICS: SPEARMAN RANK CORRELATION (rho) & JACCARD INDEX")
    print("=========================================================================\n")
    print(f"{'Comparison Pair':<35}{'Spearman rho':<15}{'p-value':<15}{'Top-3 Jaccard'}")
    print("-" * 75)

    alignment_report: Dict[str, Any] = {
        "component_labels": component_labels,
        "feature_importances": {
            name: [round(float(v), 4) for v in vals]
            for name, vals in importances_matrix.items()
        },
        "top_features": {
            name: exp.top_features(5) for name, exp in models_dict.items()
        },
        "correlations": {},
    }

    pairs = [
        ("Quantum QSVC", "Random Forest"),
        ("Quantum QSVC", "Classical SVM"),
        ("Quantum VQC", "Random Forest"),
        ("Quantum VQC", "Classical SVM"),
        ("Quantum QSVC", "Quantum VQC"),
    ]

    for m1, m2 in pairs:
        v1 = importances_matrix[m1]
        v2 = importances_matrix[m2]
        corr, pval = spearmanr(v1, v2)

        # Top-3 Jaccard overlap
        top3_m1 = set(name for name, _ in models_dict[m1].top_features(3))
        top3_m2 = set(name for name, _ in models_dict[m2].top_features(3))
        jaccard = len(top3_m1.intersection(top3_m2)) / len(top3_m1.union(top3_m2))

        pair_key = f"{m1.replace(' ', '_')}_vs_{m2.replace(' ', '_')}"
        alignment_report["correlations"][pair_key] = {
            "model_1": m1,
            "model_2": m2,
            "spearman_rho": round(float(corr), 4),
            "spearman_p_value": round(float(pval), 5),
            "top3_jaccard_overlap": round(float(jaccard), 4),
            "clinical_concordance": "High" if corr > 0.6 else ("Moderate" if corr > 0.3 else "Divergent/Novel"),
        }
        print(f"{m1 + ' vs ' + m2:<35}{corr:>12.4f}   {pval:>12.5f}   {jaccard:>12.2f}")

    # Generate Radar Chart
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_shap_radar_chart(importances_matrix, component_labels, fig_dir)

    # Save JSON report
    table_dir = Path(output_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_file = table_dir / "xai_comparison_report.json"
    with open(report_file, "w") as f:
        json.dump(alignment_report, f, indent=2)
    print(f"\n[+] XAI Comparison Report saved to: {report_file}\n")

    return alignment_report


def plot_shap_radar_chart(
    importances_matrix: Dict[str, np.ndarray],
    feature_labels: List[str],
    fig_dir: Path,
) -> Path:
    """Generate multi-model radar (spider) chart comparing feature attributions."""
    num_vars = len(feature_labels)
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)

    color_map = {
        "Random Forest": ("#2e7d32", "--", 1.8),
        "Classical SVM": ("#e65100", "-.", 1.8),
        "Quantum QSVC": ("#9c27b0", "-", 2.5),
        "Quantum VQC": ("#00bcd4", ":", 2.0),
    }

    for name, vals in importances_matrix.items():
        v_list = vals.tolist()
        v_list += v_list[:1]  # Close loop
        color, ls, lw = color_map.get(name, ("#333333", "-", 1.5))
        ax.plot(angles, v_list, color=color, linestyle=ls, linewidth=lw, label=name)
        if "Quantum" in name:
            ax.fill(angles, v_list, color=color, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), feature_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(np.max(list(importances_matrix.values())) * 1.25, 0.35))
    ax.grid(color="#d0d0d0", linestyle="--", alpha=0.7)

    plt.title(
        "Clinical Feature Attribution Alignment Radar\nClassical vs Quantum SHAP Importances (WDBC)",
        size=13,
        fontweight="bold",
        pad=25,
    )
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True, facecolor="#ffffff")
    plt.tight_layout()

    svg_file = fig_dir / "shap_radar_comparison.svg"
    png_file = fig_dir / "shap_radar_comparison.png"
    plt.savefig(svg_file, format="svg", bbox_inches="tight")
    plt.savefig(png_file, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved radar chart figures to: {svg_file} and {png_file}")
    return svg_file


if __name__ == "__main__":
    compare_feature_attributions()
