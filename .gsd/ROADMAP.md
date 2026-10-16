# ROADMAP.md — Project Roadmap & Phase Decomposition

> **Project:** Hybrid Quantum-Classical Machine Learning Platform for Early Disease Detection (HQML-EDD)  
> **Collaborators:** Member A (Data & Classical ML Lead), Member B (Quantum ML Lead), Member C (Evaluation & XAI Lead)

---

## Unified Milestones Overview

### Milestone 1: Classical Baseline & Scaffolding (Phases 1)
- Data loaders for WDBC and UCI Heart Disease.
- Preprocessing, stratified 5-fold cross-validation splitters with zero data leakage.
- Classical baselines (SVM, Random Forest, XGBoost, MLP) with 100% test pass.

### Milestone 2: Quantum Core & Dimensionality Reduction (Phase 2)
- Dimensionality reduction to fit qubit limits (PCA, Mutual Information) scaled to $[-\pi, \pi]$.
- Quantum backend and device abstraction manager (`default.qubit`, `qasm_simulator`).
- Quantum Feature Maps (ZZFeatureMap, AngleEmbedding).
- Quantum Support Vector Classifier (QSVC) via quantum kernel Gram matrix.
- Variational Quantum Classifier (VQC) with Strongly Entangling Layers.
- Hybrid Quantum Neural Network (HQNN) combining PyTorch Linear layers and PennyLane `TorchLayer`.
- Quantum circuit numerical unit testing suite.

### Milestone 3: Integration, Benchmarking & Explainability (Phase 3)
- Fold integrity audit and SHA-256 signatures for exact split parity.
- Classical hyperparameter grid search (Linear SVM, RBF SVM, Random Forest, XGBoost, MLP).
- Unified evaluation runner (Acc, Sens, Spec, F1, ROC-AUC).
- KernelSHAP and Permutation explainability for hybrid models.

### Milestone 4: Multi-Page Web Platform & Clinical Visual Analytics (Phase 4)
- Multi-page Streamlit application (`app/Home.py`, `app/pages/`).
- Interactive clinical data explorer and single-patient validator (`1_Data_Exploration.py`).
- Feature dimension slider ($k \in [2, 8]$) and 2D/3D Plotly visualizations (`2_Features.py`).
- Baselines and fold integrity audit leaderboard (`3_Baselines_Audit.py`).
- Quantum models studio and real-time interactive inference.

### Milestone 5: Hardening, Inference Service & Final Deliverables (Phase 5)
- Ingestion sanitization guardrails rejecting physiological anomalies.
- End-to-end single-sample clinical inference service.
- IEEE publication-grade technical methodology paper (`docs/paper_source.md`, `docs/HQML_Disease_Detection_Paper.pdf`).
- Pitch presentation deck and complete project documentation.

---

## Member A Detailed Task Breakdown

### Phase 1: Ingestion, Splits & Classical Baseline Models
- [x] **T1.3: WDBC Dataset Loader and Imputer** (`src/data/wdbc_loader.py`, `tests/test_data_loaders.py`)
- [x] **T1.4: UCI Heart Disease Dataset Loader and Preprocessor** (`src/data/heart_loader.py`, `tests/test_data_loaders.py`)
- [x] **T1.5: Stratified 5-Fold Split Generator** (`src/data/splitter.py`, `tests/test_data_loaders.py`)
- [x] **T1.6: Classical SVM Baseline (Linear & RBF)** (`src/models/classical/svm.py`, `tests/test_classical_models.py`)
- [x] **T1.8: Classical PyTorch Multi-Layer Perceptron (MLP)** (`src/models/classical/mlp.py`, `tests/test_classical_models.py`)

### Phase 2: Dimensionality Reduction for Qubit Constraints
- [x] **T2.1: PCA Dimensionality Reduction Module** (`src/features/pca_reducer.py`, `tests/test_features.py`)
- [x] **T2.2: Mutual Information Feature Selector** (`src/features/mi_selector.py`, `tests/test_features.py`)

### Phase 3: Review & Fold Integrity Audit + Hyperparameter Tuning
- [x] **T3.3 & T3.6: Fold Integrity Audit** (`src/evaluation/fold_audit.py`, `configs/fold_integrity_audit.json`)
- [x] **T3.6: Classical Hyperparameter Grid Search & Benchmarks** (`src/models/classical/tuner.py`, `src/evaluation/cv_runner.py`)

### Phase 4: Platform Usability & Visual Analytics
- [x] **T4.3: Feature Engineering UI Controller** (`app/pages/2_Features.py`)
- [x] **T4.4: Baselines & Audit Dashboard** (`app/pages/3_Baselines_Audit.py`)
- [x] **T4.5: Clinical Data Exploration & Ingestion Dashboard** (`app/pages/1_Data_Exploration.py`)

### Phase 5: Single-Sample Hardening, Inference Service & IEEE Report
- [x] **T5.1: Ingestion Sanitization Guardrails** (`src/data/validator.py`, `tests/test_data_validation.py`)
- [x] **T5.2: End-to-End Single-Sample Clinical Inference Service** (`src/models/classical/inference.py`, `tests/test_inference.py`)
- [x] **T5.4: IEEE Technical Methodology Paper Authoring** (`docs/paper_source.md`, `docs/generate_pdf.py`, `docs/HQML_Disease_Detection_Paper.pdf`)
