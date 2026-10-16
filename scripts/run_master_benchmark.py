"""CLI entrypoint to run the Master 5-Fold Cross-Validation Benchmark Suite."""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.evaluation.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Master 5-Fold Cross-Validation Benchmark Suite")
    parser.add_argument("--dataset", type=str, default="wdbc", help="Dataset name ('wdbc')")
    parser.add_argument("--quick", action="store_true", help="Quick mode (3 splits) for fast testing")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--qubits", type=int, default=6, help="PCA feature reduction dimension (default: 6)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed (default: 42)")
    args = parser.parse_args()

    summary_df, _ = run_benchmark(
        dataset_name=args.dataset,
        n_splits=args.folds,
        n_components=args.qubits,
        quick=args.quick,
        random_state=args.seed,
    )

    print("\n--- MASTER BENCHMARK SUMMARY TABLE ---")
    display_cols = [
        "Model",
        "Type",
        "accuracy (mean±std)",
        "sensitivity (mean±std)",
        "specificity (mean±std)",
        "f1_score (mean±std)",
        "roc_auc (mean±std)",
        "latency_ms (mean±std)",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))


if __name__ == "__main__":
    main()
