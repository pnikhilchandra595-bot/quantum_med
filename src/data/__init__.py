"""Data Ingestion, Preprocessing, and Feature Engineering Module."""

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.data.splitter import get_stratified_folds
from src.data.fundus_loader import load_fundus_dataset, FUNDUS_FEATURE_NAMES

try:
    from src.data.loader import (
        load_wdbc_dataset,
        load_heart_disease_dataset,
        load_dataset,
    )
    from src.data.feature_reduction import (
        PCAReducer as DataPCAReducer,
        MutualInfoReducer,
        ZeroLeakagePipeline,
    )
except ImportError:
    pass

__all__ = [
    "load_wdbc",
    "load_heart_disease",
    "get_stratified_folds",
    "load_fundus_dataset",
    "FUNDUS_FEATURE_NAMES",
    "load_wdbc_dataset",
    "load_heart_disease_dataset",
    "load_dataset",
    "MutualInfoReducer",
    "ZeroLeakagePipeline",
]
