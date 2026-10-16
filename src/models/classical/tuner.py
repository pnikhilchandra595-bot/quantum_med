"""Classical Model Hyperparameter Grid Tuner and Benchmark Suite.

Module: src.models.classical.tuner
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Task: Baseline Support (Hyperparameter Grid Tuning)
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Suppress sklearn SVC probability calibration deprecation warnings during grid search
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease, HeartDiseasePreprocessor
from src.data.splitter import get_folds
from src.evaluation.cv_runner import CrossValidationRunner
from src.models.classical import ClassicalSVM, ClassicalMLP, ClassicalRandomForest, ClassicalXGBoost

# Focused, High-Impact Hyperparameter Grids
PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "svm_linear": {
        "C": [0.1, 1.0, 10.0],
    },
    "svm_rbf": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto", 0.05],
    },
    "random_forest": {
        "n_estimators": [50, 100],
        "max_depth": [4, 6, None],
        "min_samples_split": [2, 5],
    },
    "xgboost": {
        "n_estimators": [50, 100],
        "max_depth": [3, 4],
        "learning_rate": [0.05, 0.1],
    },
    "mlp": {
        "lr": [0.005, 0.01],
        "dropout_rate": [0.1, 0.2],
        "weight_decay": [1e-4, 1e-3],
    },
}


class HyperparameterTuner:
    """Systematic cross-validated hyperparameter optimizer for classical baselines."""

    def __init__(self, dataset_name: str = "wdbc", configs_dir: str = "configs"):
        self.dataset_name = dataset_name.lower().strip()
        self.configs_dir = Path(configs_dir)
        self.runner = CrossValidationRunner(dataset_name=self.dataset_name, configs_dir=str(self.configs_dir))

    def tune_linear_svm(self, metric: str = "accuracy_mean") -> Tuple[Dict[str, Any], float]:
        """Grid search over Linear SVM regularization parameter C."""
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        grid = PARAM_GRIDS["svm_linear"]["C"]

        for c_val in grid:
            # probability=False for fast grid evaluation without Platt scaling overhead
            factory = lambda c=c_val: ClassicalSVM(kernel="linear", C=c, probability=False, random_state=42)
            results = self.runner.run(factory, model_name=f"SVM_Linear_C{c_val}")
            score = results["summary"][metric]
            if score > best_score:
                best_score = score
                best_params = {"kernel": "linear", "C": c_val, "probability": True}

        return best_params, best_score

    def tune_rbf_svm(self, metric: str = "accuracy_mean") -> Tuple[Dict[str, Any], float]:
        """Grid search over RBF SVM C and gamma parameters."""
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        grid_c = PARAM_GRIDS["svm_rbf"]["C"]
        grid_gamma = PARAM_GRIDS["svm_rbf"]["gamma"]

        for c_val, gamma_val in itertools.product(grid_c, grid_gamma):
            factory = lambda c=c_val, g=gamma_val: ClassicalSVM(kernel="rbf", C=c, gamma=g, probability=False, random_state=42)
            results = self.runner.run(factory, model_name=f"SVM_RBF_C{c_val}_g{gamma_val}")
            score = results["summary"][metric]
            if score > best_score:
                best_score = score
                best_params = {"kernel": "rbf", "C": c_val, "gamma": gamma_val, "probability": True}

        return best_params, best_score

    def tune_random_forest(self, metric: str = "accuracy_mean") -> Tuple[Dict[str, Any], float]:
        """Grid search over Random Forest tree estimators and depth."""
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        n_est_list = PARAM_GRIDS["random_forest"]["n_estimators"]
        depth_list = PARAM_GRIDS["random_forest"]["max_depth"]
        split_list = PARAM_GRIDS["random_forest"]["min_samples_split"]

        for n_est, depth, split in itertools.product(n_est_list, depth_list, split_list):
            factory = lambda n=n_est, d=depth, s=split: ClassicalRandomForest(
                n_estimators=n, max_depth=d, min_samples_split=s, random_state=42
            )
            results = self.runner.run(factory, model_name=f"RF_{n_est}_d{depth}")
            score = results["summary"][metric]
            if score > best_score:
                best_score = score
                best_params = {
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "min_samples_split": split,
                }

        return best_params, best_score

    def tune_xgboost(self, metric: str = "accuracy_mean") -> Tuple[Dict[str, Any], float]:
        """Grid search over XGBoost estimators, depth, and learning rates."""
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        n_est_list = PARAM_GRIDS["xgboost"]["n_estimators"]
        depth_list = PARAM_GRIDS["xgboost"]["max_depth"]
        lr_list = PARAM_GRIDS["xgboost"]["learning_rate"]

        for n_est, depth, lr in itertools.product(n_est_list, depth_list, lr_list):
            factory = lambda n=n_est, d=depth, l=lr: ClassicalXGBoost(
                n_estimators=n, max_depth=d, learning_rate=l, random_state=42
            )
            results = self.runner.run(factory, model_name=f"XGB_{n_est}_d{depth}_lr{lr}")
            score = results["summary"][metric]
            if score > best_score:
                best_score = score
                best_params = {
                    "n_estimators": n_est,
                    "max_depth": depth,
                    "learning_rate": lr,
                }

        return best_params, best_score

    def tune_mlp(self, metric: str = "accuracy_mean") -> Tuple[Dict[str, Any], float]:
        """Grid search over PyTorch MLP learning rates and dropout values."""
        best_params: Dict[str, Any] = {}
        best_score = -1.0
        lr_list = PARAM_GRIDS["mlp"]["lr"]
        dropout_list = PARAM_GRIDS["mlp"]["dropout_rate"]
        wd_list = PARAM_GRIDS["mlp"]["weight_decay"]

        for lr, drop, wd in itertools.product(lr_list, dropout_list, wd_list):
            factory = lambda l=lr, d=drop, w=wd: ClassicalMLP(
                lr=l, dropout_rate=d, weight_decay=w, epochs=15, patience=10, random_state=42
            )
            results = self.runner.run(factory, model_name=f"MLP_lr{lr}_drop{drop}")
            score = results["summary"][metric]
            if score > best_score:
                best_score = score
                best_params = {
                    "lr": lr,
                    "dropout_rate": drop,
                    "weight_decay": wd,
                    "epochs": 30,
                }

        return best_params, best_score

    def run_all_tuning(self) -> Dict[str, Any]:
        """Tune all 5 model architectures and compile results."""
        results = {}
        results["svm_linear"], score_sl = self.tune_linear_svm()
        results["svm_rbf"], score_sr = self.tune_rbf_svm()
        results["random_forest"], score_rf = self.tune_random_forest()
        results["xgboost"], score_xgb = self.tune_xgboost()
        results["mlp"], score_mlp = self.tune_mlp()

        return {
            "dataset": self.dataset_name.upper(),
            "optimal_hyperparameters": results,
            "best_cv_accuracies": {
                "svm_linear": score_sl,
                "svm_rbf": score_sr,
                "random_forest": score_rf,
                "xgboost": score_xgb,
                "mlp": score_mlp,
            },
        }


def tune_and_export_benchmarks(
    configs_dir: str = "configs",
    optimal_params_path: str = "configs/optimal_hyperparams.json",
    benchmark_results_path: str = "configs/benchmark_results.json",
) -> Dict[str, Any]:
    """Execute hyperparameter optimization for WDBC and Heart cohorts and persist benchmarks."""
    wdbc_tuner = HyperparameterTuner("wdbc", configs_dir=configs_dir)
    heart_tuner = HyperparameterTuner("heart", configs_dir=configs_dir)

    print("Tuning classical baseline hyperparameters for WDBC...")
    wdbc_tuning = wdbc_tuner.run_all_tuning()
    print("Tuning classical baseline hyperparameters for Heart Disease...")
    heart_tuning = heart_tuner.run_all_tuning()

    optimal_configs = {
        "wdbc": wdbc_tuning["optimal_hyperparameters"],
        "heart": heart_tuning["optimal_hyperparameters"],
    }

    # Save optimal hyperparameters to JSON
    out_params = Path(optimal_params_path)
    out_params.parent.mkdir(parents=True, exist_ok=True)
    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(optimal_configs, f, indent=2)

    # Re-evaluate all tuned models through CrossValidationRunner to produce final benchmark results
    final_benchmarks: Dict[str, Any] = {"wdbc": {}, "heart": {}}

    print("Computing final calibrated 5-fold benchmarks for WDBC...")
    runner_wdbc = CrossValidationRunner("wdbc", configs_dir=configs_dir)
    params_w = optimal_configs["wdbc"]

    final_benchmarks["wdbc"]["svm_linear"] = runner_wdbc.run(
        lambda: ClassicalSVM(**params_w["svm_linear"]), model_name="Classical SVM (Linear)"
    )
    final_benchmarks["wdbc"]["svm_rbf"] = runner_wdbc.run(
        lambda: ClassicalSVM(**params_w["svm_rbf"]), model_name="Classical SVM (RBF)"
    )
    final_benchmarks["wdbc"]["random_forest"] = runner_wdbc.run(
        lambda: ClassicalRandomForest(**params_w["random_forest"]), model_name="Random Forest"
    )
    final_benchmarks["wdbc"]["xgboost"] = runner_wdbc.run(
        lambda: ClassicalXGBoost(**params_w["xgboost"]), model_name="XGBoost"
    )
    final_benchmarks["wdbc"]["mlp"] = runner_wdbc.run(
        lambda: ClassicalMLP(**params_w["mlp"]), model_name="PyTorch Deep MLP"
    )

    print("Computing final calibrated 5-fold benchmarks for Heart Disease...")
    runner_heart = CrossValidationRunner("heart", configs_dir=configs_dir)
    params_h = optimal_configs["heart"]

    final_benchmarks["heart"]["svm_linear"] = runner_heart.run(
        lambda: ClassicalSVM(**params_h["svm_linear"]), model_name="Classical SVM (Linear)"
    )
    final_benchmarks["heart"]["svm_rbf"] = runner_heart.run(
        lambda: ClassicalSVM(**params_h["svm_rbf"]), model_name="Classical SVM (RBF)"
    )
    final_benchmarks["heart"]["random_forest"] = runner_heart.run(
        lambda: ClassicalRandomForest(**params_h["random_forest"]), model_name="Random Forest"
    )
    final_benchmarks["heart"]["xgboost"] = runner_heart.run(
        lambda: ClassicalXGBoost(**params_h["xgboost"]), model_name="XGBoost"
    )
    final_benchmarks["heart"]["mlp"] = runner_heart.run(
        lambda: ClassicalMLP(**params_h["mlp"]), model_name="PyTorch Deep MLP"
    )

    # Save final benchmark results to JSON
    out_bench = Path(benchmark_results_path)
    with open(out_bench, "w", encoding="utf-8") as f:
        json.dump(final_benchmarks, f, indent=2)

    return {
        "optimal_hyperparameters": optimal_configs,
        "benchmark_results": final_benchmarks,
    }


if __name__ == "__main__":
    tune_and_export_benchmarks()
    print("Hyperparameter tuning and benchmark evaluation complete.")
