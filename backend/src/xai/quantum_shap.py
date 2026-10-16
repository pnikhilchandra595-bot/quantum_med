"""Quantum Explainability Module (Quantum KernelSHAP, Adapters & Permutation Importance).

Computes local and global feature attributions for quantum models (QSVC, VQC, HQNN)
with computational mitigations for quantum simulator latency and clinical reporting.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import shap
except ImportError:
    shap = None

try:
    from src.xai.classical_shap import SHAPExplanation
except ImportError:
    SHAPExplanation = Any

logger = logging.getLogger(__name__)


class QuantumKernelSHAPExplainer:
    """KernelSHAP explainer specialized for quantum models.

    Mitigates exponential circuit evaluations by k-means clustering the background
    reference dataset down to a small, representative set of medoids (default: 15-25 samples).
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray,
        n_background_samples: int = 20,
        random_state: int = 42,
    ):
        self.model = model
        self.n_background_samples = min(n_background_samples, len(background_data))
        self.random_state = random_state

        self.reference_data = self._summarize_background(background_data)
        self._explainer = None

    def _summarize_background(self, data: np.ndarray) -> np.ndarray:
        if len(data) <= self.n_background_samples:
            return np.copy(data)

        kmeans = KMeans(
            n_clusters=self.n_background_samples,
            random_state=self.random_state,
            n_init=10,
        )
        kmeans.fit(data)
        return kmeans.cluster_centers_

    def _init_explainer(self) -> None:
        if shap is None:
            return

        def predict_fn(x_matrix: np.ndarray) -> np.ndarray:
            probs = self.model.predict_proba(x_matrix)
            if len(probs) != len(x_matrix):
                probs = np.tile(probs[0], (len(x_matrix), 1))
            return probs[:, 1]

        self._explainer = shap.KernelExplainer(predict_fn, self.reference_data)

    def explain_instance(self, sample: np.ndarray, nsamples: int = 50) -> np.ndarray:
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)

        if self._explainer is None:
            self._init_explainer()

        if self._explainer is not None:
            shap_values = self._explainer.shap_values(sample, nsamples=nsamples, l1_reg="auto")
            return shap_values[0] if isinstance(shap_values, list) else shap_values

        base_prob = float(self.model.predict_proba(sample)[0, 1])
        ref_mean = np.mean(self.reference_data, axis=0)
        perturb = 0.05
        attributions = []
        for i in range(sample.shape[1]):
            perturbed_sample = np.copy(sample)
            perturbed_sample[0, i] += perturb
            new_prob = float(self.model.predict_proba(perturbed_sample)[0, 1])
            attributions.append((new_prob - base_prob) / perturb * (sample[0, i] - ref_mean[i]))
        return np.array(attributions)

    def explain_dataset(self, X_test: np.ndarray, nsamples: int = 50) -> np.ndarray:
        if self._explainer is None:
            self._init_explainer()

        if self._explainer is not None:
            shap_values = self._explainer.shap_values(X_test, nsamples=nsamples, l1_reg="auto")
            return np.array(shap_values)

        return np.array([self.explain_instance(x, nsamples=nsamples) for x in X_test])

    def generate_patient_attribution_report(
        self,
        sample: np.ndarray,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if sample.ndim == 1:
            sample = sample.flatten()
        n_features = len(sample)

        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        shap_vals = self.explain_instance(sample).flatten()
        prob_malignant = float(self.model.predict_proba(sample.reshape(1, -1))[0, 1])

        ranked_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]
        top_factors = []
        for idx in ranked_indices:
            top_factors.append({
                "feature": feature_names[idx],
                "feature_value": float(sample[idx]),
                "shap_attribution": float(shap_vals[idx]),
                "impact": "INCREASES_RISK" if shap_vals[idx] > 0 else "PROTECTIVE",
            })

        return {
            "prediction_probability": prob_malignant,
            "prediction_label": "MALIGNANT / DISEASE" if prob_malignant >= 0.5 else "BENIGN / HEALTHY",
            "top_clinical_factors": top_factors,
            "all_attributions": {feature_names[i]: float(shap_vals[i]) for i in range(n_features)},
        }


class QuantumKernelSHAP:
    """Quantum Kernel SHAP adapter compliant with Member C evaluation interface."""

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

        if shap is not None:
            self.explainer = shap.KernelExplainer(predict_fn, self.background)
        else:
            self.explainer = None

    def explain(self, X: np.ndarray, nsamples: int = 60) -> Any:
        X_arr = np.asarray(X)
        single_sample = X_arr.ndim == 1
        X_eval = X_arr.reshape(1, -1) if single_sample else X_arr

        if self.explainer is not None:
            values = self.explainer.shap_values(X_eval, nsamples=nsamples)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            values = np.asarray(values)
            base_val = (
                float(self.explainer.expected_value[1])
                if isinstance(self.explainer.expected_value, (list, np.ndarray)) and len(self.explainer.expected_value) > 1
                else float(self.explainer.expected_value)
            )
        else:
            values = np.zeros(X_eval.shape)
            base_val = 0.5

        mean_abs = np.abs(values) if values.ndim == 1 else np.mean(np.abs(values), axis=0)
        total_sum = np.sum(mean_abs)
        norm_imp = mean_abs / total_sum if total_sum > 0 else np.ones_like(mean_abs) / len(mean_abs)
        out_values = values[0] if single_sample else values

        if SHAPExplanation is not Any:
            return SHAPExplanation(
                model_name=self.model_name,
                feature_names=self.feature_names,
                values=out_values,
                base_value=base_val,
                data=X_eval,
                mean_abs_importance=mean_abs,
                normalized_importance=norm_imp,
            )
        return {"values": out_values, "mean_abs": mean_abs}


class QuantumPermutationImportance:
    """Computes Permutation Feature Importance for quantum models."""

    def __init__(
        self,
        model: Any,
        scoring: str = "roc_auc",
        n_repeats: int = 5,
        random_state: int = 42,
    ):
        self.model = model
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _score(self, y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
        if self.scoring == "roc_auc":
            return roc_auc_score(y_true, y_pred_prob)
        elif self.scoring == "accuracy":
            return accuracy_score(y_true, (y_pred_prob >= 0.5).astype(int))
        else:
            raise ValueError(f"Unsupported scoring: {self.scoring}")

    def compute_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        rng = np.random.default_rng(self.random_state)
        baseline_prob = self.model.predict_proba(X)[:, 1]
        baseline_score = self._score(y, baseline_prob)

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        results = {}
        for j in range(n_features):
            drops = []
            for _ in range(self.n_repeats):
                X_permuted = np.copy(X)
                X_permuted[:, j] = rng.permutation(X_permuted[:, j])
                perm_prob = self.model.predict_proba(X_permuted)[:, 1]
                perm_score = self._score(y, perm_prob)
                drops.append(baseline_score - perm_score)

            results[feature_names[j]] = {
                "mean_drop": float(np.mean(drops)),
                "std_drop": float(np.std(drops)),
            }
        return results
