"""
Classical baseline machine learning models (SVM, Random Forest, XGBoost, MLP).
"""

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.models.base import BaseModel


class ClassicalModel(BaseModel):
    """
    Wrapper around scikit-learn and gradient boosting estimators.
    """

    def __init__(self, model_type: str, **kwargs):
        super().__init__(name=model_type, **kwargs)
        self.model_type = model_type.lower()
        self.estimator = self._build_estimator(self.model_type, kwargs)

    def _build_estimator(self, model_type: str, kwargs: Dict[str, Any]):
        random_state = kwargs.get("random_state", 42)

        if model_type == "svm":
            c_val = kwargs.get("C", 1.0)
            gamma = kwargs.get("gamma", "scale")
            return SVC(C=c_val, gamma=gamma, probability=True, random_state=random_state)

        elif model_type == "rf":
            n_estimators = kwargs.get("n_estimators", 100)
            max_depth = kwargs.get("max_depth", None)
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
            )

        elif model_type == "xgb":
            try:
                import xgboost as xgb

                n_estimators = kwargs.get("n_estimators", 100)
                lr = kwargs.get("lr", 0.1)
                max_depth = kwargs.get("max_depth", 3)
                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    max_depth=max_depth,
                    random_state=random_state,
                    eval_metric="logloss",
                )
            except Exception:
                from sklearn.ensemble import HistGradientBoostingClassifier

                return HistGradientBoostingClassifier(random_state=random_state)

        elif model_type == "mlp":
            hidden_layer_sizes = kwargs.get("hidden_layer_sizes", (64, 32))
            max_iter = kwargs.get("max_iter", kwargs.get("epochs", 200))
            lr = kwargs.get("lr", 0.001)
            return MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                learning_rate_init=lr,
                max_iter=max_iter,
                random_state=random_state,
            )

        else:
            raise ValueError(f"Unknown classical model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "ClassicalModel":
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        decision = self.estimator.decision_function(X)
        prob_1 = 1.0 / (1.0 + np.exp(-decision))
        return np.column_stack([1.0 - prob_1, prob_1])
