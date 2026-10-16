"""Unit tests for Fundus Retinopathy Ingestion and Quantum Pipeline Integration."""

import os
import pytest
import numpy as np

from src.data.fundus_loader import load_fundus_dataset, parse_label_from_filename
from src.features.pca_reducer import PCAReducer
from src.models.quantum.qsvc import QuantumSupportVectorClassifier


class TestFundusDataset:
    """Tests for fundus photo extraction and hybrid quantum classifier flow."""

    def test_parse_labels(self):
        assert parse_label_from_filename("REAL_HUMAN_EYE_LEVEL_0_NORMAL.jpg") == 0
        assert parse_label_from_filename("Real_Human_Eye_3_Normal_Healthy.jpg") == 0
        assert parse_label_from_filename("REAL_HUMAN_EYE_LEVEL_2_MODERATE_NPDR.jpg") == 1
        assert parse_label_from_filename("REAL_HUMAN_EYE_LEVEL_4_PROLIFERATIVE_PDR.jpg") == 1

    def test_fundus_dataset_loading(self):
        fundus_dir = os.path.join("data", "Sample_Fundus_Photos")
        if not os.path.exists(fundus_dir):
            pytest.skip(f"Fundus directory not present: {fundus_dir}")

        X, y, paths, feature_names = load_fundus_dataset(fundus_dir=fundus_dir, scale=True)

        assert X.shape[0] == 22
        assert X.shape[1] == 11
        assert len(y) == 22
        assert set(y).issubset({0, 1})
        assert len(feature_names) == 11

    def test_fundus_quantum_e2e_flow(self):
        fundus_dir = os.path.join("data", "Sample_Fundus_Photos")
        if not os.path.exists(fundus_dir):
            pytest.skip("Fundus directory not present.")

        X, y, _, _ = load_fundus_dataset(fundus_dir=fundus_dir, scale=True)

        # 1. Reduce 11 biomedical image features to 4 qubits
        pca = PCAReducer(n_components=4, scale_to_pi=True, random_state=42)
        X_4q = pca.fit_transform(X)
        assert X_4q.shape == (22, 4)

        # 2. Fit Quantum Kernel SVM on small subset (requires pennylane)
        pytest.importorskip("pennylane")
        qsvc = QuantumSupportVectorClassifier(n_qubits=4, reps=1, C=1.0)
        qsvc.fit(X_4q[:12], y[:12])
        preds = qsvc.predict(X_4q[12:])

        assert len(preds) == 10
        assert set(preds).issubset({0, 1})
