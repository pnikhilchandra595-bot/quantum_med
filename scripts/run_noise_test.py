"""Noise Robustness Benchmark Script (Task T3.10, Member B).

Compares classification and clinical sensitivity degradation between Quantum
and Classical models under increasing levels of simulated biomedical sensor noise.
"""

import os
import sys
import argparse
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.features.pca_reducer import PCAReducer
from src.models.quantum.qsvc import QuantumSupportVectorClassifier
from src.quantum.noise import NoiseRobustnessEvaluator


def run_noise_benchmark(dataset_name: str = "wdbc", quick: bool = True) -> pd.DataFrame:
    """Trains models and benchmarks robustness across noise perturbations."""
    print(f"\n=======================================================")
    print(f" Quantum vs Classical Noise Robustness Benchmark ({dataset_name.upper()})")
    print(f"=======================================================")

    if dataset_name.lower() == "wdbc":
        X, y, feature_names = load_wdbc(scale=True)
    else:
        X, y, feature_names = load_heart_disease(scale=True)

    # Dimensionality reduction to 4 qubits for rapid, deterministic benchmarking
    pca = PCAReducer(n_components=4, scale_to_pi=True, random_state=42)
    X_4q = pca.fit_transform(X)

    if quick:
        # Benchmark on a representative subset of 80 samples for rapid turnaround
        X_4q = X_4q[:80]
        y = y[:80]

    X_train, X_test, y_train, y_test = train_test_split(
        X_4q, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"Training Classical SVM (RBF)...")
    classical_svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
    classical_svm.fit(X_train, y_train)

    print(f"Training Quantum Support Vector Classifier (QSVC, 4 qubits)...")
    qsvc = QuantumSupportVectorClassifier(n_qubits=4, reps=1, C=1.0)
    qsvc.fit(X_train, y_train)

    evaluator = NoiseRobustnessEvaluator(
        noise_levels=[0.0, 0.05, 0.10, 0.20, 0.30, 0.40],
        random_state=42,
    )

    comparison_results = evaluator.compare_quantum_vs_classical(
        quantum_model=qsvc,
        classical_model=classical_svm,
        X_test=X_test,
        y_test=y_test,
    )

    df = pd.DataFrame(comparison_results)
    os.makedirs("reports/tables", exist_ok=True)
    out_path = f"reports/tables/noise_robustness_{dataset_name}.csv"
    df.to_csv(out_path, index=False)

    print("\n--- Summary Robustness Results ---")
    print(df.to_string(index=False))
    print(f"\n[Done] Table exported to: {out_path}\n")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wdbc", choices=["wdbc", "heart"])
    parser.add_argument("--quick", action="store_true", default=True)
    args = parser.parse_args()
    run_noise_benchmark(args.dataset, quick=args.quick)
