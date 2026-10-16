# ROADMAP.md — Member A Execution Roadmap

> **Owner:** Member A (Lead / Classical ML Engineer)  
> **Status:** All Phases Active / Completed  
> **Primary Deliverables:** D2 (Data Ingestion), D3 (Feature Selection), D5 (Classical Baselines), D10/D12 (Report)  

---

## Phases Overview

| Phase | Description | Deliverables | Status | Tests Passing |
| :---: | :--- | :--- | :---: | :---: |
| **Phase 1** | Ingestion, Preprocessing, Splits & Classical Baseline Models | D2, D5 | **DONE** | 11/11 |
| **Phase 2** | Qubit Dimensionality Reduction (PCA & Mutual Information) | D3 | **DONE** | 7/7 |
| **Phase 3** | Review & Fold Integrity Audit + Classical Grid Tuning | D5 | **DONE** | 6/6 |
| **Phase 4** | Clinical Usability & Multi-Page UI Platform | D2, D3, D5 | **DONE** | 2/2 |
| **Phase 5** | Single-Sample Hardening, Classical Inference Service & IEEE Report | D2, D10/D12 | **DONE** | 6/6 |

---

## Detailed Task Breakdown

### Phase 1: Ingestion, Splits & Classical Baseline Models
- [x] **T1.3: WDBC Dataset Loader and Imputer** (`src/data/wdbc_loader.py`, `tests/test_data_loaders.py`)
  - Continuous 30 features, binary targets ($M \to 1, B \to 0$), zero missing values.
  - Verified: Shape $(569, 30)$, no data leakage.
- [x] **T1.4: UCI Heart Disease Dataset Loader and Preprocessor** (`src/data/heart_loader.py`, `tests/test_data_loaders.py`)
  - Parses `?` values, median imputes `ca` and `thal`, one-hot encodes categoricals, standard scales continuous.
  - Verified: Shape $(303, 20)$, zero NaNs, binary target normalized.
- [x] **T1.5: Stratified 5-Fold Split Generator** (`src/data/splitter.py`, `tests/test_data_loaders.py`)
  - Deterministic 5-fold cross-validation with exact class preservation.
  - Exported: `configs/folds_wdbc.json` and `configs/folds_heart.json`.
- [x] **T1.6: Classical SVM Baseline (Linear & RBF)** (`src/models/classical/svm.py`, `tests/test_classical_models.py`)
  - Scikit-learn wrapper with probability and sigmoid fallback for fast grid search.
  - Exceeds >97% accuracy on WDBC folds.
- [x] **T1.8: Classical PyTorch Multi-Layer Perceptron (MLP)** (`src/models/classical/mlp.py`, `tests/test_classical_models.py`)
  - 3-layer regularized architecture (`Linear -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear -> ReLU -> Linear`).
  - Monotonic loss reduction, Adam optimizer, early stopping.

---

### Phase 2: Dimensionality Reduction for Qubit Constraints
- [x] **T2.1: PCA Dimensionality Reduction Module** (`src/features/pca_reducer.py`, `tests/test_features.py`)
  - Reduces inputs to $k \in [2, 8]$ principal components for quantum circuits.
  - Rescales to $[-\pi, \pi]$ for $ZZ$-Feature Maps. Computes cumulative explained variance.
- [x] **T2.2: Mutual Information Feature Selector** (`src/features/mi_selector.py`, `tests/test_features.py`)
  - Selects top $k$ features via non-linear mutual information.
  - Preserves clinical column names for SHAP explainability and scales outputs to $[-\pi, \pi]$ (and $[0, 2\pi]$).

---

### Phase 3: Review & Fold Integrity Audit + Hyperparameter Tuning
- [x] **T3.3 & T3.6: Fold Integrity Audit** (`src/evaluation/fold_audit.py`, `configs/fold_integrity_audit.json`)
  - Validates index disjointness ($\text{train} \cap \text{val} = \emptyset$), exhaustive coverage, and zero data leakage.
  - Computes SHA-256 signatures of fold configs for quantum lane split parity.
- [x] **T3.6: Classical Hyperparameter Grid Search & Benchmarks** (`src/models/classical/tuner.py`, `src/evaluation/cv_runner.py`)
  - Exhaustive 5-fold CV tuning for Linear SVM, RBF SVM, Random Forest, XGBoost, and MLP.
  - Exported: `configs/optimal_hyperparams.json` and `configs/benchmark_results.json`.

---

### Phase 4: Platform Usability & Visual Analytics
- [x] **T4.3: Feature Engineering UI Controller** (`app/pages/2_Features.py`)
  - Interactive qubit slider ($k \in [2, 8]$), real-time 2D/3D Plotly scatter plots, variance bar charts.
- [x] **T4.4: Baselines & Audit Dashboard** (`app/pages/3_Baselines_Audit.py`)
  - Audit certificate cards, SHA-256 signatures, leaderboard, interactive OOF ROC curves, confusion matrices.
- [x] **T4.5: Clinical Data Exploration & Ingestion Dashboard** (`app/pages/1_Data_Exploration.py`)
  - Cohort statistics, distribution plots, clinical variable dictionary, and interactive single-patient validator.

---

### Phase 5: Single-Sample Hardening, Inference Service & IEEE Report
- [x] **T5.1: Ingestion Sanitization Guardrails** (`src/data/validator.py`, `tests/test_data_validation.py`)
  - Rejects missing columns, invalid datatypes, and physiological sensor out-of-range anomalies.
- [x] **T5.2: End-to-End Single-Sample Clinical Inference Service** (`src/models/classical/inference.py`, `tests/test_inference.py`)
  - Validates patient dictionary, applies fitted fold preprocessor, runs classical inference, returns structured risk report.
- [x] **T5.4: IEEE Technical Methodology Paper Authoring** (`docs/paper_source.md`, `docs/generate_pdf.py`, `docs/HQML_Disease_Detection_Paper.pdf`)
  - 8-page publication-grade PDF report with mathematical formulations, 5-fold benchmark tables, and audit signatures.
