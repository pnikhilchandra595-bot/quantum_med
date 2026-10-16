"""Quantum Kernel SHAP Explainer (Member B Deliverable Adapter).

Implements Shapley value estimation for Quantum Kernel Classifiers (QSVC)
and Variational Quantum Circuits (VQC) in Hilbert space.
"""

from typing import List, Optional, Any
import numpy as np
import shap

from src.xai.classical_shap import SHAPExplanation


class QuantumKernelSHAP:
    """Quantum Kernel SHAP explainer for QSVC and VQC models."""

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_background_samples: int = 30,
        random_state: int = 42,
    ) -> None:
        self.model = model
        self.model_name = getattr(model, "name", "QuantumModel")
        n_features = background_data.shape[1]
        self.feature_names = (
            feature_names
            if feature_names is not None
            else [f"Qubit_{i}" for i in range(n_features)]
        )
        self.random_state = random_state

        if len(background_data) > max_background_samples:
            rng = np.random.RandomState(random_state)
            idx = rng.choice(len(background_data), max_background_samples, replace=False)
            self.background = background_data[idx]
        else:
            self.background = np.copy(background_data)

        def predict_fn(X: np.ndarray) -> np.ndarray:
            probas = self.model.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] > 1:
                return probas[:, 1]
            return probas.ravel()

        self.explainer = shap.KernelExplainer(predict_fn, self.background)

    def explain(self, X: np.ndarray, nsamples: int = 60) -> SHAPExplanation:
        """Compute quantum feature attributions and return standardized SHAPExplanation."""
        X_arr = np.asarray(X)
        single_sample = X_arr.ndim == 1
        X_eval = X_arr.reshape(1, -1) if single_sample else X_arr

        values = self.explainer.shap_values(X_eval, nsamples=nsamples)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        values = np.asarray(values)

        base_val = (
            float(self.explainer.expected_value[1])
            if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1
            else float(self.explainer.expected_value)
        )

        mean_abs = np.abs(values) if values.ndim == 1 else np.mean(np.abs(values), axis=0)
        total_sum = np.sum(mean_abs)
        norm_imp = mean_abs / total_sum if total_sum > 0 else np.ones_like(mean_abs) / len(mean_abs)

        out_values = values[0] if single_sample else values

        return SHAPExplanation(
            model_name=self.model_name,
            feature_names=self.feature_names,
            values=out_values,
            base_value=base_val,
            data=X_eval,
            mean_abs_importance=mean_abs,
            normalized_importance=norm_imp,
        )
