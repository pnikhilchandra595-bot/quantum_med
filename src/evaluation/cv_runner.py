"""Unified 5-Fold Cross-Validation Runner.

Module: src.evaluation.cv_runner
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Task: T3.3 & T3.6 (Cross-Validation Engine)
"""

from __future__ import annotations

import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease, HeartDiseasePreprocessor
from src.data.splitter import get_folds, SplitFold
from src.models.base import BaseModel


class CrossValidationRunner:
    """Executes leak-free 5-fold cross-validation on persistent fold indices.
    
    Guarantees:
    - Preprocessor fitted strictly on training indices per fold.
    - Captures per-fold and aggregated mean +/- std metrics.
    - Records out-of-fold (OOF) predictions for global ROC and confusion matrix computation.
    - Measures training time and inference latency per sample.
    """

    def __init__(
        self,
        dataset_name: str = "wdbc",
        configs_dir: str = "configs",
        preprocessor_factory: Optional[Callable[[], Any]] = None,
    ):
        self.dataset_name = dataset_name.lower().strip()
        self.configs_dir = configs_dir

        if self.dataset_name in ["wdbc", "breast_cancer"]:
            self.X, self.y, self.feature_names = load_wdbc(as_frame=False, return_feature_names=True)
            self.default_preprocessor_factory = lambda: WDBCPreprocessor()
        elif self.dataset_name in ["heart", "heart_disease", "cleveland"]:
            self.X, self.y, self.feature_names = load_heart_disease(as_frame=False, preprocess=False, return_feature_names=True)
            self.default_preprocessor_factory = lambda: HeartDiseasePreprocessor()
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: 'wdbc', 'heart'")

        self.preprocessor_factory = preprocessor_factory or self.default_preprocessor_factory
        self.folds: List[SplitFold] = get_folds(self.dataset_name, configs_dir=self.configs_dir)

    def run(
        self,
        model_factory: Callable[[], BaseModel],
        model_name: str = "Model",
    ) -> Dict[str, Any]:
        """Run 5-fold CV using the specified model factory.

        Args:
            model_factory: Callable that instantiates a fresh unfitted BaseModel.
            model_name: Descriptive name for reporting.

        Returns:
            Dictionary containing fold-by-fold results, aggregated metrics,
            and out-of-fold predictions.
        """
        n_samples = len(self.y)
        oof_preds = np.zeros(n_samples, dtype=np.int64)
        oof_probs = np.zeros((n_samples, 2), dtype=np.float64)

        fold_metrics_list: List[Dict[str, float]] = []
        train_latencies: List[float] = []
        infer_latencies_per_sample: List[float] = []

        for fold in self.folds:
            train_idx, val_idx = fold["train_idx"], fold["val_idx"]
            X_train_raw, y_train = self.X[train_idx], self.y[train_idx]
            X_val_raw, y_val = self.X[val_idx], self.y[val_idx]

            # 1. Fit preprocessor strictly on train fold
            preprocessor = self.preprocessor_factory()
            X_train = preprocessor.fit_transform(X_train_raw)
            X_val = preprocessor.transform(X_val_raw)

            # 2. Fit model with latency timing
            model = model_factory()
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time_ms = (time.perf_counter() - t0) * 1000.0
            train_latencies.append(train_time_ms)

            # 3. Predict on validation fold with per-sample latency
            t0 = time.perf_counter()
            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)
            infer_time_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(val_idx), 1)
            infer_latencies_per_sample.append(infer_time_ms)

            # 4. Store OOF predictions
            oof_preds[val_idx] = val_preds
            oof_probs[val_idx] = val_probs

            # 5. Compute fold metrics
            acc = float(accuracy_score(y_val, val_preds))
            prec = float(precision_score(y_val, val_preds, zero_division=0))
            rec = float(recall_score(y_val, val_preds, zero_division=0))
            f1 = float(f1_score(y_val, val_preds, zero_division=0))
            try:
                roc = float(roc_auc_score(y_val, val_probs[:, 1]))
            except ValueError:
                roc = float("nan")

            fold_metrics_list.append(
                {
                    "fold": int(fold["fold"]),
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "roc_auc": roc,
                    "train_time_ms": train_time_ms,
                    "infer_latency_ms": infer_time_ms,
                }
            )

        # Compute summary statistics (mean +/- std)
        accs = [m["accuracy"] for m in fold_metrics_list]
        precs = [m["precision"] for m in fold_metrics_list]
        recs = [m["recall"] for m in fold_metrics_list]
        f1s = [m["f1"] for m in fold_metrics_list]
        rocs = [m["roc_auc"] for m in fold_metrics_list if not np.isnan(m["roc_auc"])]

        summary = {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "roc_auc_mean": float(np.mean(rocs)) if rocs else float("nan"),
            "roc_auc_std": float(np.std(rocs)) if rocs else float("nan"),
            "train_latency_mean_ms": float(np.mean(train_latencies)),
            "infer_latency_mean_ms": float(np.mean(infer_latencies_per_sample)),
        }

        # Global out-of-fold metrics
        cm = confusion_matrix(self.y, oof_preds).tolist()
        try:
            global_roc = float(roc_auc_score(self.y, oof_probs[:, 1]))
        except ValueError:
            global_roc = float("nan")

        return {
            "model_name": model_name,
            "dataset": self.dataset_name.upper(),
            "n_samples": n_samples,
            "n_splits": len(self.folds),
            "summary": summary,
            "fold_metrics": fold_metrics_list,
            "global_oof_metrics": {
                "accuracy": float(accuracy_score(self.y, oof_preds)),
                "precision": float(precision_score(self.y, oof_preds, zero_division=0)),
                "recall": float(recall_score(self.y, oof_preds, zero_division=0)),
                "f1": float(f1_score(self.y, oof_preds, zero_division=0)),
                "roc_auc": global_roc,
                "confusion_matrix": cm,
            },
            "oof_predictions": oof_preds.tolist(),
            "oof_probabilities": oof_probs.tolist(),
        }
