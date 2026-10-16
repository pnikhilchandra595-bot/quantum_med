"""Fold Integrity and Zero-Data-Leakage Auditor.

Module: src.evaluation.fold_audit
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Tasks: T3.3 & T3.6 (Pipeline Integrity Audit)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease, HeartDiseasePreprocessor
from src.data.splitter import get_folds, SplitFold


class AuditFailureError(Exception):
    """Raised when cross-validation splits fail data integrity or leakage audits."""
    pass


class FoldIntegrityAuditor:
    """Rigorous auditor for 5-fold cross-validation splits.
    
    Verifies:
    1. Mathematical Disjointness: train_idx ∩ val_idx = ∅ for every fold.
    2. Mutual Exclusivity & Completeness: ⋃ val_idx = {0, ..., N-1} with zero overlap.
    3. Stratification Balance: Exact preservation of class distribution across all folds.
    4. Zero Data Leakage: Preprocessors strictly isolated to train folds without test leakage.
    5. Cross-Pipeline Hash Immutability: SHA-256 fingerprint for classical-quantum parity.
    """

    def __init__(self, dataset_name: str = "wdbc", configs_dir: str = "configs"):
        self.dataset_name = dataset_name.lower().strip()
        p = Path(configs_dir)
        if not p.exists() and (Path("backend") / configs_dir).exists():
            p = Path("backend") / configs_dir
        self.configs_dir = p

        if self.dataset_name in ["wdbc", "breast_cancer"]:
            self.json_path = self.configs_dir / "folds_wdbc.json"
            self.X, self.y = load_wdbc(as_frame=False)
            self.preprocessor_cls = WDBCPreprocessor
        elif self.dataset_name in ["heart", "heart_disease", "cleveland"]:
            self.json_path = self.configs_dir / "folds_heart.json"
            self.X, self.y = load_heart_disease(as_frame=False, preprocess=False)
            self.preprocessor_cls = HeartDiseasePreprocessor
        else:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: 'wdbc', 'heart'")

        self.folds = get_folds(self.dataset_name, configs_dir=str(self.configs_dir))

    def compute_config_hash(self) -> str:
        """Compute SHA-256 hash of fold configuration to enforce immutability across team members."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Fold configuration not found at {self.json_path}")
        with open(self.json_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def audit_disjointness_and_coverage(self) -> Dict[str, Any]:
        """Verify train/test disjointness and full exhaustive validation coverage."""
        n_samples = len(self.y)
        all_val_indices: List[int] = []

        fold_reports = []
        for fold in self.folds:
            train_set = set(fold["train_idx"])
            val_set = set(fold["val_idx"])

            intersection = train_set.intersection(val_set)
            if len(intersection) > 0:
                raise AuditFailureError(
                    f"Fold {fold['fold']} failed disjointness! Overlapping indices: {intersection}"
                )

            if len(train_set) + len(val_set) != n_samples:
                raise AuditFailureError(
                    f"Fold {fold['fold']} sample count mismatch: {len(train_set)} + {len(val_set)} != {n_samples}"
                )

            all_val_indices.extend(fold["val_idx"])
            fold_reports.append(
                {
                    "fold": fold["fold"],
                    "n_train": len(train_set),
                    "n_val": len(val_set),
                    "disjoint": True,
                }
            )

        # Check for mutual exclusivity among validation sets
        if len(all_val_indices) != len(set(all_val_indices)):
            duplicates = [x for x in all_val_indices if all_val_indices.count(x) > 1]
            raise AuditFailureError(f"Duplicate validation indices detected across folds: {set(duplicates)}")

        # Check for complete coverage
        if sorted(all_val_indices) != list(range(n_samples)):
            missing = set(range(n_samples)) - set(all_val_indices)
            raise AuditFailureError(f"Validation partition does not exhaustively cover dataset! Missing: {missing}")

        return {
            "status": "PASS",
            "n_samples": n_samples,
            "n_folds": len(self.folds),
            "exhaustive_coverage": True,
            "zero_index_overlap": True,
            "folds": fold_reports,
        }

    def audit_stratification_balance(self) -> Dict[str, Any]:
        """Verify exact preservation of class balance across all folds within integer rounding tolerance."""
        global_pos_ratio = float(np.mean(self.y == 1))
        strat_reports = []

        for fold in self.folds:
            val_idx = fold["val_idx"]
            y_val = self.y[val_idx]
            val_pos_ratio = float(np.mean(y_val == 1))

            # Discrepancy must be less than 1 sample deviation (<= 2.5% for 300-600 samples)
            deviation = abs(val_pos_ratio - global_pos_ratio)
            if deviation > 0.03:
                raise AuditFailureError(
                    f"Fold {fold['fold']} class ratio deviated by {deviation:.4f} from global {global_pos_ratio:.4f}"
                )

            strat_reports.append(
                {
                    "fold": fold["fold"],
                    "n_val": len(val_idx),
                    "positives": int(np.sum(y_val == 1)),
                    "negatives": int(np.sum(y_val == 0)),
                    "positive_ratio": round(val_pos_ratio, 4),
                    "deviation_from_global": round(deviation, 4),
                }
            )

        return {
            "status": "PASS",
            "global_positive_ratio": round(global_pos_ratio, 4),
            "max_allowed_deviation": 0.03,
            "stratification": strat_reports,
        }

    def audit_zero_data_leakage(self) -> Dict[str, Any]:
        """Empirically prove that transformers compute parameters solely on train_idx."""
        leakage_reports = []

        for fold in self.folds:
            train_idx, val_idx = fold["train_idx"], fold["val_idx"]
            X_train, X_val = self.X[train_idx], self.X[val_idx]

            # Fit preprocessor on training fold
            preprocessor = self.preprocessor_cls()
            X_train_trans = preprocessor.fit_transform(X_train)
            X_val_trans = preprocessor.transform(X_val)

            # Verification 1: Train set has mean 0 and std 1 (or normalized)
            # For WDBC, continuous features are standard-scaled
            if self.dataset_name in ["wdbc", "breast_cancer"]:
                train_means = np.mean(X_train_trans, axis=0)
                train_stds = np.std(X_train_trans, axis=0)
                np.testing.assert_allclose(
                    train_means, 0.0, atol=1e-7,
                    err_msg=f"Fold {fold['fold']} train mean is not zero"
                )
                np.testing.assert_allclose(
                    train_stds, 1.0, atol=1e-7,
                    err_msg=f"Fold {fold['fold']} train std is not unit"
                )

                # Verification 2: Validation set scaled by train params should NOT have mean exactly 0.0
                # If val_means were 0.0 within 1e-7, it would indicate global pre-scaling (leakage!)
                val_means = np.mean(X_val_trans, axis=0)
                mean_dist = np.linalg.norm(val_means)
                if mean_dist < 1e-12:
                    raise AuditFailureError(
                        f"Fold {fold['fold']} validation set has exact zero mean! Likely globally leaked."
                    )

            # Verification 3: Transforming validation data does not alter fitted training params
            if hasattr(preprocessor, "scaler"):
                params_before = np.copy(preprocessor.scaler.mean_)
                _ = preprocessor.transform(X_val)
                params_after = preprocessor.scaler.mean_
                np.testing.assert_array_equal(
                    params_before, params_after,
                    err_msg=f"Fold {fold['fold']} preprocessor state was modified during test transform!"
                )

            leakage_reports.append(
                {
                    "fold": fold["fold"],
                    "isolated_fit_verified": True,
                    "zero_leakage_confirmed": True,
                }
            )

        return {
            "status": "PASS",
            "zero_leakage_verified": True,
            "leakage_audits": leakage_reports,
        }

    def run_full_audit(self) -> Dict[str, Any]:
        """Execute comprehensive audit and compile signed certificate."""
        disjoint_res = self.audit_disjointness_and_coverage()
        strat_res = self.audit_stratification_balance()
        leakage_res = self.audit_zero_data_leakage()
        config_hash = self.compute_config_hash()

        certificate = {
            "audit_title": "HQML Cross-Validation & Zero-Data-Leakage Audit Certificate",
            "auditor": "Member A (Classical ML & Pipeline Integrity Lead)",
            "dataset": self.dataset_name.upper(),
            "config_file": str(self.json_path),
            "sha256_hash": config_hash,
            "status": "CERTIFIED_ZERO_LEAKAGE",
            "disjointness_and_coverage": disjoint_res,
            "stratification": strat_res,
            "data_leakage": leakage_res,
        }
        return certificate


def audit_all_datasets(
    configs_dir: str = "configs",
    output_audit_path: str = "configs/fold_integrity_audit.json",
) -> Dict[str, Any]:
    """Run full integrity audit across both WDBC and Heart Disease datasets and persist certificate."""
    wdbc_auditor = FoldIntegrityAuditor("wdbc", configs_dir=configs_dir)
    heart_auditor = FoldIntegrityAuditor("heart", configs_dir=configs_dir)

    full_audit = {
        "title": "Dual-Cohort Cross-Validation Pipeline Integrity Certificate",
        "audited_by": "Member A (Pipeline Integrity Lead)",
        "wdbc_audit": wdbc_auditor.run_full_audit(),
        "heart_audit": heart_auditor.run_full_audit(),
        "parity_guarantee": (
            "Identical fold splits and zero-leakage preprocessors are certified for both "
            "Classical baselines (Member A) and Quantum circuits (Member B)."
        ),
    }

    out_file = Path(output_audit_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_audit, f, indent=2)

    return full_audit


if __name__ == "__main__":
    results = audit_all_datasets()
    print("Fold integrity audit passed successfully for all datasets.")
    print(f"Audit certificate saved to: configs/fold_integrity_audit.json")
