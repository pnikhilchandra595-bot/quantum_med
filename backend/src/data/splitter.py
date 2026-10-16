"""Stratified 5-Fold Split Generator for WDBC and Heart Disease datasets.

Module: src.data.splitter
Owner: Member A (Lead / Classical ML Engineer)
Deliverable: D2 (Data Ingestion & Splits)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease


class SplitFold(dict):
    """Dictionary representation of a cross-validation fold supporting tuple unpacking.
    
    Can be used as:
        fold["train_idx"], fold["val_idx"]
    Or unpacked directly:
        train_idx, val_idx = fold
    """

    def __init__(self, fold_id: int, train_idx: List[int], val_idx: List[int]):
        super().__init__(
            fold=fold_id,
            train_idx=train_idx,
            val_idx=val_idx,
            n_train=len(train_idx),
            n_val=len(val_idx),
        )
        self.fold_id = fold_id
        self.train_idx = np.array(train_idx, dtype=np.int64)
        self.val_idx = np.array(val_idx, dtype=np.int64)

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield train_idx then val_idx for direct unpacking."""
        yield self.train_idx
        yield self.val_idx


def generate_stratified_folds(
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[SplitFold]:
    """Generate reproducible StratifiedKFold splits preserving exact class balance."""
    y_arr = np.asarray(y, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    dummy_X = np.zeros((len(y_arr), 1))

    folds: List[SplitFold] = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(dummy_X, y_arr)):
        folds.append(
            SplitFold(
                fold_id=fold_id,
                train_idx=train_idx.tolist(),
                val_idx=val_idx.tolist(),
            )
        )
    return folds


def get_stratified_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generates stratified cross-validation train/validation index splits.

    Args:
        X: Feature matrix.
        y: Binary target labels.
        n_splits: Number of folds (default: 5).
        seed: Random seed for reproducibility.

    Returns:
        List of tuples (train_indices, val_indices).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        folds.append((train_idx, val_idx))
    return folds


def save_folds_to_json(folds: List[SplitFold], file_path: Union[str, Path]) -> Path:
    """Save folds to a JSON configuration file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "fold": f["fold"],
            "train_idx": f["train_idx"].tolist() if isinstance(f["train_idx"], np.ndarray) else f["train_idx"],
            "val_idx": f["val_idx"].tolist() if isinstance(f["val_idx"], np.ndarray) else f["val_idx"],
            "n_train": f["n_train"],
            "n_val": f["n_val"],
        }
        for f in folds
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_folds_from_json(file_path: Union[str, Path]) -> List[SplitFold]:
    """Load folds from a JSON configuration file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fold config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        SplitFold(
            fold_id=d["fold"],
            train_idx=d["train_idx"],
            val_idx=d["val_idx"],
        )
        for d in data
    ]


def get_folds(
    dataset_name: str = "wdbc",
    configs_dir: str = "configs",
    force_regenerate: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[SplitFold]:
    """Get deterministic 5-fold CV splits for the requested dataset.

    Args:
        dataset_name: 'wdbc' or 'heart'
        configs_dir: Directory where JSON files reside
        force_regenerate: If True, regenerates splits and overwrites JSON
        n_splits: Number of CV splits (default: 5)
        random_state: Seed for reproducibility (default: 42)

    Returns:
        List of 5 SplitFold objects.
    """
    clean_name = dataset_name.lower().strip()
    configs_path = Path(configs_dir)
    if not configs_path.exists() and (Path("backend") / configs_dir).exists():
        configs_path = Path("backend") / configs_dir

    if clean_name in ["wdbc", "breast_cancer"]:
        json_file = configs_path / "folds_wdbc.json"
        load_fn = lambda: load_wdbc(scale=False, return_feature_names=False)[1]
    elif clean_name in ["heart", "heart_disease", "cleveland"]:
        json_file = configs_path / "folds_heart.json"
        load_fn = lambda: load_heart_disease(scale=False, return_feature_names=False)[1]
    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Supported: 'wdbc', 'heart'")

    if not force_regenerate and json_file.exists():
        return load_folds_from_json(json_file)

    # Generate fresh folds and save to JSON
    y = load_fn()
    folds = generate_stratified_folds(y, n_splits=n_splits, random_state=random_state)
    save_folds_to_json(folds, json_file)
    return folds


if __name__ == "__main__":
    # Ensure both fold files exist upon execution
    wdbc_folds = get_folds("wdbc", force_regenerate=True)
    heart_folds = get_folds("heart", force_regenerate=True)
    print(f"Generated {len(wdbc_folds)} WDBC folds and {len(heart_folds)} Heart folds.")
