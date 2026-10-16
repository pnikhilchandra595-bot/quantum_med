"""Tree-based classical baseline models (Random Forest, XGBoost).

Module: src.models.classical.trees
Owner: Member A & Member C
"""

from src.models.classical.rf import ClassicalRandomForest
from src.models.classical.xgb import ClassicalXGBoost

RandomForestModel = ClassicalRandomForest
XGBoostModel = ClassicalXGBoost

__all__ = ["RandomForestModel", "XGBoostModel"]
