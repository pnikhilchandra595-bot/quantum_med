"""T3.4: Statistical Significance Testing (Quantum Advantage Check).

Implements Paired Student's t-test and Wilcoxon Signed-Rank test across the 5 CV folds.
Tests formal clinical hypotheses:
    H0: Quantum Sensitivity == Classical Sensitivity (No Advantage)
    H1: Quantum Sensitivity > Classical Sensitivity (Quantum Advantage, alpha=0.05)

Outputs:
- reports/tables/significance_p_values.json
- reports/tables/significance_summary.csv
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.benchmark import run_benchmark


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = x - y
    n = len(diff)
    if n < 2:
        return 0.0
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def run_significance_tests(
    folds_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
    output_dir: str = "reports/tables",
    metric: str = "sensitivity",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Execute hypothesis tests for Quantum Advantage over Classical models.

    Args:
        folds_data: Dictionary of fold-level metric values. If None, reads benchmark_folds.json.
        output_dir: Directory to save results.
        metric: Metric to evaluate (default: 'sensitivity' - oncology priority).

    Returns:
        results: Structured dictionary of p-values, t-statistics, and decisions.
        summary_df: DataFrame table of all pairwise comparisons.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if folds_data is None:
        folds_file = out_path / "benchmark_folds.json"
        if not folds_file.exists():
            print("[*] Benchmark folds file not found. Running 5-fold benchmark...")
            _, folds_data = run_benchmark(dataset_name="wdbc", n_splits=5)
        else:
            with open(folds_file, "r") as f:
                folds_data = json.load(f)

    quantum_models = ["QSVC", "VQC", "HQNN"]
    classical_models = ["Random Forest", "Linear SVM", "RBF SVM", "XGBoost", "MLP"]

    results: Dict[str, Any] = {
        "metric_evaluated": metric,
        "hypotheses": {
            "H0": f"Quantum {metric.capitalize()} == Classical {metric.capitalize()}",
            "H1": f"Quantum {metric.capitalize()} > Classical {metric.capitalize()} (Quantum Advantage, alpha=0.05)",
        },
        "comparisons": {},
    }

    summary_rows = []

    print("\n=========================================================================")
    print(f"  STATISTICAL SIGNIFICANCE TESTING: QUANTUM ADVANTAGE ({metric.upper()})")
    print(f"  Testing H1: Quantum {metric.capitalize()} > Classical {metric.capitalize()} (alpha = 0.05)")
    print("=========================================================================\n")
    print(f"{'Comparison':<30}{'Mean Diff':<12}{'Cohen d':<10}{'t-test p':<12}{'Wilcoxon p':<12}{'Significant?'}")
    print("-" * 88)

    for q_name in quantum_models:
        if q_name not in folds_data:
            continue
        q_scores = np.array(folds_data[q_name][metric], dtype=float)

        for c_name in classical_models:
            if c_name not in folds_data:
                continue
            c_scores = np.array(folds_data[c_name][metric], dtype=float)
            pair_key = f"{q_name}_vs_{c_name.replace(' ', '_')}"

            diff = q_scores - c_scores
            mean_diff = float(np.mean(diff))
            cohen_d = compute_cohens_d(q_scores, c_scores)

            # Paired Student's t-test (one-sided H1: q > c)
            if np.all(diff == diff[0]):
                t_stat = 0.0
                p_val_ttest = 0.5 if diff[0] == 0 else (0.001 if diff[0] > 0 else 0.999)
            else:
                ttest_res = stats.ttest_rel(q_scores, c_scores, alternative="greater")
                t_stat = float(ttest_res.statistic)
                p_val_ttest = float(ttest_res.pvalue)

            # Wilcoxon Signed-Rank test (one-sided)
            if np.all(diff == 0):
                w_stat = 0.0
                p_val_wilcoxon = 1.0
            else:
                try:
                    wilcox_res = stats.wilcoxon(
                        q_scores, c_scores, alternative="greater", zero_method="pratt"
                    )
                    w_stat = float(wilcox_res.statistic)
                    p_val_wilcoxon = float(wilcox_res.pvalue)
                except Exception:
                    w_stat = 0.0
                    p_val_wilcoxon = p_val_ttest

            is_significant = bool(p_val_ttest < 0.05 or p_val_wilcoxon < 0.05)
            decision_str = (
                "Reject H0 -> Quantum Advantage Confirmed"
                if is_significant
                else "Fail to reject H0 -> Parity/Comparable"
            )

            results["comparisons"][pair_key] = {
                "quantum_model": q_name,
                "classical_model": c_name,
                "quantum_scores": [round(float(x), 4) for x in q_scores],
                "classical_scores": [round(float(x), 4) for x in c_scores],
                "mean_difference": round(mean_diff, 4),
                "cohens_d": round(cohen_d, 4),
                "paired_t_statistic": round(t_stat, 4),
                "paired_t_p_value_onesided": round(p_val_ttest, 5),
                "wilcoxon_statistic": round(w_stat, 4),
                "wilcoxon_p_value_onesided": round(p_val_wilcoxon, 5),
                "h1_accepted_p_lt_0_05": is_significant,
                "decision": decision_str,
            }

            sig_str = "YES (p < 0.05)" if is_significant else "No (p >= 0.05)"
            comp_label = f"{q_name} vs {c_name}"
            print(
                f"{comp_label:<30}{mean_diff:>+10.4f}  {cohen_d:>8.2f}  "
                f"{p_val_ttest:>10.4f}  {p_val_wilcoxon:>10.4f}  {sig_str}"
            )

            summary_rows.append({
                "comparison": comp_label,
                "quantum_model": q_name,
                "classical_model": c_name,
                "mean_diff": round(mean_diff, 4),
                "cohens_d": round(cohen_d, 2),
                "t_statistic": round(t_stat, 4),
                "t_test_p_value": round(p_val_ttest, 4),
                "wilcoxon_p_value": round(p_val_wilcoxon, 4),
                "quantum_advantage": is_significant,
                "decision": decision_str,
            })

    # Save JSON and CSV
    json_path = out_path / "significance_p_values.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    csv_path = out_path / "significance_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[+] Significance JSON saved to: {json_path}")
    print(f"[+] Significance Summary CSV saved to: {csv_path}\n")

    return results, summary_df


if __name__ == "__main__":
    run_significance_tests()
