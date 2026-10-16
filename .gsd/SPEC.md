# SPEC.md — HQML Classical Pipeline & Data Engineering Specification

> **Status: FINALIZED**  
> **Project:** Hybrid Quantum-Classical Machine Learning (HQML) for Biomedical Disease Detection  
> **Lead / Owner:** Member A (Lead / Classical ML Engineer)  
> **Backups:** Member C (Evaluation & XAI) / Member B (Quantum Circuits)  
> **Primary Deliverables:** D2 (Data Ingestion), D3 (Feature Selection), D5 (Classical Baselines), D10/D12 (IEEE Report)  

---

## 1. High-Level System Architecture & Scope

This specification formalizes the deliverables, engineering standards, data contracts, and algorithmic constraints for Member A's lane within the HQML clinical disease detection platform.

```
+-----------------------------------------------------------------------------------------+
|                                  MEMBER A PIPELINE ARCHITECTURE                         |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Clinical Cohorts]             [Data Layer (D2)]                 [Qubit Encodings (D3)] |
|   - WDBC (569 x 30)       -->    - WDBCPreprocessor        -->     - PCAReducer         |
|   - UCI Heart (303 x 13)         - HeartPreprocessor                (k in [2, 8], [-pi,pi])|
|                                  - Splitter (Stratified 5-Fold)    - MISelector         |
|                                  - PatientRecordValidator           (k in [2, 8], [0,2pi]) |
|                                           |                                  |          |
|                                           v                                  v          |
|                                 [Classical Baselines (D5)]         [Handoff to Member B]|
|                                  - Linear & RBF SVM                 - ZZFeatureMap       |
|                                  - PyTorch Deep MLP                 - AngleEmbedding     |
|                                  - Random Forest & XGBoost          - PQC / VQC / QSVM   |
|                                  - 5-Fold CV & Hyperparameter Grid                      |
|                                  - Zero-Leakage Fold Integrity Audit                    |
|                                                                                         |
|                                 [Usability & Dissemination]                             |
|                                  - Streamlit Multi-Page App (Pages 1, 2, 3)             |
|                                  - IEEE Technical Methodology Report (D10/D12)          |
+-----------------------------------------------------------------------------------------+
```

---

## 2. Core Functional Requirements & Specifications

### 2.1 Ingestion & Preprocessing (Deliverable D2)
- **R1.1 (WDBC Dataset):** Ingest 569 patient fine-needle aspirate biopsies across 30 continuous features. Map targets deterministically (`Malignant -> 1`, `Benign -> 0`). Guarantee zero missing values.
- **R1.2 (UCI Heart Disease):** Ingest 303 patient records across 13 clinical attributes. Parse `?` missing markers in fluoroscopy (`ca`) and thallium scintigraphy (`thal`). Impute missing values using train-fitted `MedianImputer`. One-hot encode categorical features (`cp`, `restecg`, `slope`, `thal`). Scale numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`). Binary target encode `disease > 0 -> 1`, `0 -> 0`.
- **R1.3 (Stratified 5-Fold Splitting):** Generate deterministic 5-fold cross-validation partitions (`random_state=42`) preserving class ratios. Persist indices to `configs/folds_wdbc.json` and `configs/folds_heart.json`.
- **R1.4 (Input Sanitization & Guardrails):** Validate patient records against missing attributes, invalid datatypes, and sensor bounds via `PatientRecordValidator`.

### 2.2 Quantum Dimensionality Reduction (Deliverable D3)
- **R2.1 (PCA Reducer):** Dimensionality reduction to $k \in [2, 8]$ principal components matching quantum qubit registers. Rescale outputs to $[-\pi, \pi]$ using `MinMaxScaler` for $ZZ$-Feature Maps.
- **R2.2 (Mutual Information Selector):** Select top $k \in [2, 8]$ non-linear features using `mutual_info_classif`. Preserve original clinical column names for downstream SHAP explainability. Rescale outputs to $[-\pi, \pi]$ (and $[0, 2\pi]$) for parameterized rotation gates.

### 2.3 Classical ML Baselines & Fold Integrity Audit (Deliverable D5)
- **R3.1 (Architectures):** Provide unified `BaseModel` wrappers for Linear SVM, RBF SVM, Random Forest, XGBoost, and a 3-layer regularized PyTorch MLP with dropout and batch normalization.
- **R3.2 (5-Fold CV Runner & Tuner):** Automated cross-validation runner tracking accuracy, ROC-AUC, F1-score, train latency, infer latency per sample, fold confusion matrices, and full cohort out-of-fold predictions.
- **R3.3 (Fold Integrity Audit):** Verify mathematical index disjointness ($\text{train} \cap \text{val} = \emptyset$), exhaustive partition coverage, class balance preservation, and zero data leakage. Sign audit certificate with SHA-256 signatures for exact classical-quantum parity.

### 2.4 User Interface & Clinical Usability
- **R4.1 (PonyTail Aesthetic):** Multi-page Streamlit application (`Home.py`, `1_Data_Exploration.py`, `2_Features.py`, `3_Baselines_Audit.py`) with rich visual cues, glassmorphism cards, metric badges, responsive Plotly charts, and zero placeholders.
- **R4.2 (Interactive Exploration & Testing):** Real-time exploration of patient cohorts, live feature transformation with dynamic qubit slider, and single-patient diagnostic validation playground.

### 2.5 Publication & Dissemination (Deliverable D10/D12)
- **R5.1 (IEEE Technical Methodology Report):** 8-page IEEE conference paper in markdown (`docs/paper_source.md`) and compiled publication-grade PDF (`docs/HQML_Disease_Detection_Paper.pdf`) via ReportLab, incorporating mathematical formulation, empirical benchmarks, and audit certificates.

---

## 3. Non-Functional & Engineering Requirements
- **NFR-1 (Reproducibility):** Zero data leakage across splits. Preprocessors and feature selectors fit exclusively on training folds.
- **NFR-2 (Test Coverage):** 100% pass rate across the full pytest suite.
- **NFR-3 (Execution Speed):** Fast test suite (<45 seconds) and sub-second classical inference per patient.
- **NFR-4 (Security & Validation):** Strict input sanitization rejecting malicious or out-of-bounds sensor inputs.
