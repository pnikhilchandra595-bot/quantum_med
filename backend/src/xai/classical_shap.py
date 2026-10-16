"""T3.6: Classical KernelSHAP Explainer Adapter.

Implements SHAP TreeExplainer (for Tree models: RF, XGBoost) and
KernelExplainer (for Kernel/Neural models: SVM, MLP).
Outputs normalized feature importance vectors and attributions matching
Member B's QuantumKernelSHAP output schema.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
import numpy as np
import shap

from src.models.base import BaseModel
from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.svm import LinearSVMModel, RBFSVMModel
from src.models.classical.mlp import MLPModel


@dataclass
class SHAPExplanation:
    """Standardized SHAP explanation container matching QuantumKernelSHAP format."""

    model_name: str
    feature_names: List[str]
    values: np.ndarray  # (n_samples, n_features) or (n_features,)
    base_value: float
    data: np.ndarray
    mean_abs_importance: np.ndarray  # (n_features,)
    normalized_importance: np.ndarray  # (n_features,), L1 sum == 1.0

    def to_dict(self) -> Dict[str, float]:
        """Return dictionary mapping feature name to normalized importance."""
        return {
            name: float(imp)
            for name, imp in zip(self.feature_names, self.normalized_importance)
        }

    def top_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k features sorted by normalized importance."""
        pairs = list(zip(self.feature_names, self.normalized_importance))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(str(name), float(val)) for name, val in pairs[:k]]


class ClassicalSHAPExplainer:
    """Adaptive SHAP explainer for all classical models."""

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_background_samples: int = 40,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.model_name = getattr(model, "name", model.__class__.__name__)
        n_features = background_data.shape[1]
        self.feature_names = (
            feature_names
            if feature_names is not None
            else [f"Feature_{i}" for i in range(n_features)]
        )
        self.random_state = random_state

        # Subsample background data if too large to ensure fast KernelSHAP evaluation
        if len(background_data) > max_background_samples:
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(background_data), max_background_samples, replace=False)
            self.background = background_data[indices]
        else:
            self.background = np.copy(background_data)

        # Select explainer type based on model architecture
        self.explainer_type = self._determine_explainer_type()
        self.explainer = self._init_explainer()

    def _determine_explainer_type(self) -> str:
        """Identify whether TreeExplainer or KernelExplainer is appropriate."""
        if isinstance(self.model, (RandomForestModel, XGBoostModel)):
            return "tree"
        # Check inner sklearn/xgboost model
        inner = getattr(self.model, "model", self.model)
        inner_type = inner.__class__.__name__.lower()
        if "forest" in inner_type or "xgb" in inner_type or "tree" in inner_type:
            return "tree"
        return "kernel"

    def _init_explainer(self) -> Any:
        """Initialize underlying SHAP explainer."""
        if self.explainer_type == "tree":
            inner_model = getattr(self.model, "model", self.model)
            try:
                return shap.TreeExplainer(inner_model, data=self.background)
            except Exception:
                # Fallback to model without background data
                return shap.TreeExplainer(inner_model)
        else:
            # Predict proba for positive class (column 1)
            def predict_fn(X: np.ndarray) -> np.ndarray:
                if hasattr(self.model, "predict_proba"):
                    probas = self.model.predict_proba(X)
                    if probas.ndim == 2 and probas.shape[1] > 1:
                        return probas[:, 1]
                    return probas.ravel()
                else:
                    return self.model.predict(X).astype(float)

            return shap.KernelExplainer(predict_fn, self.background)

    def explain(
        self, X: np.ndarray, nsamples: Optional[int] = 100
    ) -> SHAPExplanation:
        """Compute SHAP values and return standardized SHAPExplanation container.

        Args:
            X: Input samples of shape (n_samples, n_features) or (n_features,).
            nsamples: Number of background evaluations for KernelExplainer.

        Returns:
            SHAPExplanation instance with raw and normalized attributions.
        """
        X_arr = np.asarray(X)
        single_sample = X_arr.ndim == 1
        if single_sample:
            X_eval = X_arr.reshape(1, -1)
        else:
            X_eval = X_arr

        if self.explainer_type == "tree":
            raw_shap = self.explainer.shap_values(X_eval)
            # For binary classifiers, TreeExplainer might return list [class0, class1] or (n, d, 2)
            if isinstance(raw_shap, list):
                # Pick positive class
                values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
            elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
                values = raw_shap[:, :, 1]
            else:
                values = raw_shap
            base_val = (
                self.explainer.expected_value[1]
                if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1
                else float(np.mean(self.explainer.expected_value))
            )
        else:
            # KernelExplainer
            kwargs = {"nsamples": nsamples} if nsamples is not None else {}
            values = self.explainer.shap_values(X_eval, **kwargs)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            base_val = (
                float(self.explainer.expected_value[1])
                if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1
                else float(self.explainer.expected_value)
            )

        values = np.asarray(values)

        # Global importance (mean absolute attribution)
        if values.ndim == 1:
            mean_abs = np.abs(values)
        else:
            mean_abs = np.mean(np.abs(values), axis=0)

        # L1-normalization matching QuantumKernelSHAP
        total_sum = np.sum(mean_abs)
        if total_sum > 0:
            norm_imp = mean_abs / total_sum
        else:
            norm_imp = np.ones_like(mean_abs) / len(mean_abs)

        out_values = values[0] if single_sample else values

        return SHAPExplanation(
            model_name=self.model_name,
            feature_names=self.feature_names,
            values=out_values,
            base_value=float(base_val),
            data=X_eval,
            mean_abs_importance=mean_abs,
            normalized_importance=norm_imp,
        )

    def get_normalized_feature_importances(self, X: np.ndarray) -> np.ndarray:
        """Convenience method to retrieve normalized importance vector."""
        explanation = self.explain(X)
        return explanation.normalized_importance
