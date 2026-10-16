# SPEC.md — Hybrid Quantum Machine Learning Platform for Early Disease Detection (HQML-EDD)

> **Status: FINALIZED**  
> **Project:** Hybrid Quantum-Classical Machine Learning (HQML) for Biomedical Disease Detection  
> **Team Lanes:**  
> - Member A: Data Engineering & Classical ML Lead (D2, D3, D5, D10/D12)  
> - Member B: Quantum Core & Circuit Architecture Lead (D4, D6, D7)  
> - Member C: Evaluation, XAI & Clinical Presentation Lead (D8, D9, D11)  

---

## 1. System Vision & Architecture

HQML-EDD is a clinically interpretable, modular, simulator-first platform that detects early-stage diseases (Breast Cancer and Heart Disease) by exploiting quantum Hilbert space embeddings, quantum kernels, and variational parameterized circuits alongside rigorous classical baseline models.

```
+-----------------------------------------------------------------------------------------+
|                                    HQML PLATFORM ARCHITECTURE                           |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Clinical Cohorts]             [Data Layer (D2)]                 [Qubit Encodings (D3)] |
|   - WDBC (569 x 30)       -->    - WDBCPreprocessor        -->     - PCAReducer         |
|   - UCI Heart (303 x 13)         - HeartPreprocessor                (k in [2, 8], [-pi,pi])|
|                                  - Splitter (Stratified 5-Fold)    - MISelector         |
|                                  - PatientRecordValidator           (k in [2, 8], [0,2pi]) |
|                                           |                                  |          |
|                     +---------------------+-------------------+              |          |
|                     |                                         |              |          |
|                     v                                         v              v          |
|         [Classical Baselines (D5)]                     [Quantum Core Models (D4)]       |
|          - Linear & RBF SVM                             - QSVC (ZZ-Kernel Gram Matrix)  |
|          - PyTorch Deep MLP                             - VQC (Strongly Entangling)     |
|          - Random Forest & XGBoost                      - HQNN (PyTorch + TorchLayer)   |
|          - 5-Fold CV & Hyperparameter Grid                                              |
|          - Zero-Leakage Fold Integrity Audit                                            |
|                     |                                         |                         |
|                     +---------------------+-------------------+                         |
|                                           v                                             |
|                             [Evaluation & Explainability (D8)]                          |
|                              - Out-of-fold metrics (Acc, Sens, Spec, F1, ROC-AUC)       |
|                              - KernelSHAP and Permutation importance                    |
|                              - Fold integrity certificates & SHA-256 signatures         |
|                                           v                                             |
|                             [Interactive Web Platform (D9)]                             |
|                              - Multi-page Streamlit App (Home, Pages 1, 2, 3, 5)        |
|                              - Real-time qubit dimension transformation & inference     |
|                              - IEEE Methodology Paper (D10/D12)                         |
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

### 2.4 Quantum Classifiers & Circuits (Deliverable D4)
- **R4.1 (QSVC):** High-dimensional Hilbert space mapping via PennyLane with $ZZ$-FeatureMap kernels and fidelity-based Gram matrix computation.
- **R4.2 (VQC):** Parameterized quantum circuit using Angle/Amplitude embeddings, Strongly Entangling Layers, and analytical parameter-shift gradient optimization.
- **R4.3 (HQNN):** End-to-end PyTorch hybrid model coupling classical feedforward layers with a PennyLane `TorchLayer` variational quantum bottleneck.

### 2.5 Explainability & Evaluation (Deliverable D8)
- **R5.1 (Metrics):** Unified metrics computation for Accuracy, Sensitivity (Recall), Specificity, F1-Score, and ROC-AUC.
- **R5.2 (XAI):** KernelSHAP and Permutation importance analysis across classical and quantum models.

### 2.6 User Interface & Clinical Usability (Deliverable D9)
- **R6.1 (Multi-page Streamlit App):** Rich UI (`Home.py`, `1_Data_Exploration.py`, `2_Features.py`, `3_Baselines_Audit.py`, `5_Pitch_Presentation.py`) with glassmorphism cards, metric badges, responsive Plotly charts, and single-patient diagnostic validation playground.

### 2.7 Publication & Dissemination (Deliverable D10/D12)
- **R7.1 (IEEE Technical Methodology Report):** 8-page IEEE conference paper in markdown (`docs/paper_source.md`) and compiled publication-grade PDF (`docs/HQML_Disease_Detection_Paper.pdf`).

---

## 3. Technical Constraints & Success Criteria
- **Hardware/Simulators:** PennyLane (`default.qubit`), Qiskit (`aer_simulator`), PyTorch autograd bridge.
- **Compute Ceiling:** $N \le 8$ qubits per circuit to guarantee rapid desktop/laptop execution.
- **Reproducibility:** Seed pinning across NumPy, PyTorch, and PennyLane RNG. Zero data leakage across cross-validation folds.
- **Test Coverage:** All unit and integration tests passing.
