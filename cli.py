#!/usr/bin/env python3
"""
CLI Orchestrator: Unified command-line interface for training, inference, and benchmarking.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repository root is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import AVAILABLE_DATASETS, load_dataset  # noqa: E402
from src.models.factory import ModelFactory  # noqa: E402
from src.quantum.backend import QuantumBackendManager  # noqa: E402
from src.utils.serialization import load_model, save_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical ML Clinical Diagnostics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a classical or quantum model")
    train_parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ModelFactory.ALL_MODELS,
        help="Model architecture to train",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="wdbc",
        choices=AVAILABLE_DATASETS,
        help="Dataset name",
    )
    train_parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Number of qubits (and reduced feature dimension)",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Training epochs for iterative models (VQC, HQNN, MLP)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate for gradient optimization",
    )
    train_parser.add_argument(
        "--backend",
        type=str,
        default="default.qubit",
        help="PennyLane simulator backend",
    )
    train_parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save trained model checkpoint",
    )

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Run inference on sample features")
    predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Trained model checkpoint path or model name",
    )
    predict_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSON file containing feature vector or array of vectors",
    )

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark all models on a dataset")
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default="wdbc",
        choices=AVAILABLE_DATASETS,
        help="Dataset name for benchmarking",
    )
    bench_parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Number of qubits for quantum feature encoding",
    )
    bench_parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Epochs for iterative models",
    )
    bench_parser.add_argument(
        "--models",
        nargs="+",
        default=ModelFactory.ALL_MODELS,
        help="Subset of models to evaluate",
    )
    bench_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON benchmark report",
    )

    # Profile subcommand
    prof_parser = subparsers.add_parser(
        "profile", help="Profile quantum execution and statevector caching"
    )
    prof_parser.add_argument(
        "--model",
        type=str,
        default="qsvc",
        choices=["qsvc", "vqc"],
        help="Quantum model to profile",
    )
    prof_parser.add_argument(
        "--samples",
        type=int,
        default=16,
        help="Number of patient samples for profiling run",
    )
    prof_parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Number of qubits in the quantum register",
    )

    return parser


def run_train(args):
    print("\n=======================================================")
    print(f"  Training Model: {args.model.upper()} on Dataset: {args.dataset.upper()}")
    print("=======================================================")

    data = load_dataset(name=args.dataset, n_qubits=args.qubits, reduction="pca")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Loaded {len(X_train)} train samples, {len(X_test)} test samples.")
    print(f"Feature dimension: {X_train.shape[1]} (Qubits: {args.qubits})")

    backend_mgr = QuantumBackendManager(backend_name=args.backend)
    model = ModelFactory.create_model(
        model_name=args.model,
        n_qubits=args.qubits,
        backend_manager=backend_mgr,
        epochs=args.epochs,
        lr=args.lr,
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    wall_time = time.time() - start_time

    metrics = model.evaluate(X_test, y_test)
    print(f"\n--- Training Finished in {wall_time:.2f}s ---")
    print(f"  Test Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"  Precision:      {metrics['precision'] * 100:.2f}%")
    print(f"  Recall:         {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score:       {metrics['f1'] * 100:.2f}%")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")

    save_path = args.save
    if not save_path:
        os.makedirs("checkpoints", exist_ok=True)
        ext = ".pt" if args.model in ["hqnn", "vqc"] else ".joblib"
        save_path = f"checkpoints/{args.model}_{args.dataset}{ext}"

    save_model(model, save_path, metadata={"dataset": args.dataset, "qubits": args.qubits})
    print(f"Saved checkpoint to: {save_path}\n")
    return model, metrics


def run_predict(args):
    print(f"\nLoading model from: {args.model}")
    model_path = Path(args.model)
    if model_path.exists():
        model = load_model(str(model_path))
    else:
        # Check if it's a name in checkpoints
        chk_pt = Path(f"checkpoints/{args.model}.joblib")
        if not chk_pt.exists():
            chk_pt = Path(f"checkpoints/{args.model}.pt")
        if chk_pt.exists():
            model = load_model(str(chk_pt))
        else:
            raise FileNotFoundError(f"Cannot find model file '{args.model}'")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(input_path, "r") as f:
        data_json = json.load(f)

    if isinstance(data_json, dict) and "features" in data_json:
        features = np.array(data_json["features"], dtype=np.float32)
    elif isinstance(data_json, list):
        features = np.array(data_json, dtype=np.float32)
    else:
        raise ValueError("JSON input must be an array or an object with 'features' key.")

    if features.ndim == 1:
        features = features.reshape(1, -1)

    preds = model.predict(features)
    probs = model.predict_proba(features)

    results = []
    for i, (pred, prob) in enumerate(zip(preds, probs)):
        results.append(
            {
                "sample_index": i,
                "prediction": int(pred),
                "class_name": "Disease / Malignant" if pred == 1 else "Healthy / Benign",
                "probability_disease": float(prob[1]),
                "confidence": float(max(prob)),
            }
        )

    print(json.dumps(results, indent=2))
    return results


def run_benchmark(args):
    print("\n=======================================================")
    print(f"  Benchmarking Suite on Dataset: {args.dataset.upper()}")
    print("=======================================================")

    data = load_dataset(name=args.dataset, n_qubits=args.qubits, reduction="pca")
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"Dataset: {data['metadata']['name']}")
    print(
        f"Samples: Train={len(X_train)}, Test={len(X_test)} | Reduced Features/Qubits={args.qubits}\n"
    )

    results = []
    print(
        f"{'Model':<8} | {'Type':<10} | {'Acc (%)':<8} | {'F1 (%)':<8} | {'ROC-AUC':<8} | {'Time (s)':<8}"
    )
    print("-" * 65)

    for m_name in args.models:
        m_clean = m_name.lower().strip()
        is_q = ModelFactory.is_quantum(m_clean)
        m_type = "Quantum" if is_q else "Classical"

        try:
            model = ModelFactory.create_model(
                model_name=m_clean,
                n_qubits=args.qubits,
                epochs=args.epochs,
            )
            t0 = time.time()
            model.fit(X_train, y_train)
            t_train = time.time() - t0

            metrics = model.evaluate(X_test, y_test)
            acc = metrics["accuracy"] * 100
            f1 = metrics["f1"] * 100
            auc = metrics["roc_auc"]

            print(
                f"{m_clean.upper():<8} | {m_type:<10} | {acc:7.2f}% | {f1:7.2f}% | {auc:8.4f} | {t_train:7.2f}s"
            )

            results.append(
                {
                    "model": m_clean,
                    "type": m_type,
                    "accuracy": round(metrics["accuracy"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1": round(metrics["f1"], 4),
                    "roc_auc": round(metrics["roc_auc"], 4),
                    "training_time_sec": round(t_train, 3),
                }
            )
        except Exception as e:
            print(f"{m_clean.upper():<8} | {m_type:<10} | ERROR: {str(e)[:30]}")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(
                {"dataset": args.dataset, "qubits": args.qubits, "results": results}, f, indent=2
            )
        print(f"\nSaved benchmark results to {args.output}")

    return results


def run_profile(args):
    from src.utils.profiler import benchmark_caching_speedup, profile_quantum_execution

    print("\n=======================================================")
    print(f"  Profiling Quantum Execution ({args.model.upper()}) & Statevector Caching")
    print("=======================================================")

    print(
        f"Profiling cProfile call graph and tracemalloc ({args.samples} samples, {args.qubits} qubits)..."
    )
    prof = profile_quantum_execution(
        model_name=args.model,
        n_samples=args.samples,
        n_qubits=args.qubits,
    )
    print("\n--- Execution Metrics ---")
    print(f"  Elapsed Time:     {prof['elapsed_seconds']} s")
    print(f"  Peak Memory:      {prof['peak_memory_kb']} KB")
    print(f"  Total Calls:      {prof['total_calls']:,}")
    print("\n--- Top Cumulative Execution Bottlenecks (cProfile) ---")
    print(prof["profile_summary"])

    print("\n--- Benchmarking Statevector Caching Acceleration (>20% Target) ---")
    bench = benchmark_caching_speedup(n_samples=args.samples, n_qubits=args.qubits)
    print(f"  Uncached Baseline: {bench['baseline_uncached_seconds']} s")
    print(f"  Optimized Cached:  {bench['optimized_cached_seconds']} s")
    print(f"  Speedup:           {bench['speedup_percentage']}%")
    print(f"  Acceleration:      {bench['acceleration_factor']}x")
    print(
        f"  Target (>20%) Met: {'YES [PASSED]' if bench['exceeds_20pct_threshold'] else 'NO [FAILED]'}"
    )
    return bench


def main():
    parser = parse_args()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "profile":
        run_profile(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
