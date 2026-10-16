# 🧬 Hybrid Quantum-Classical Machine Learning (HQML) for Biomedical Disease Detection

[![Tests](https://img.shields.io/badge/pytest-40%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](app/)

> **IEEE CAIBE 2026 Preprint & Research Platform**  
> A modular, simulator-first Hybrid Quantum-Classical Machine Learning (HQML) framework for high-dimensional biomedical disease detection across Wisconsin Diagnostic Breast Cancer (WDBC) and UCI Cleveland Heart Disease cohorts.

---

## 🏛️ System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                     MEMBER A HQML ARCHITECTURE                                    |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Clinical Cohorts]             [Data Layer (D2)]                   [Qubit Encodings (D3)]        |
|   - WDBC (569 x 30)       -->    - WDBCPreprocessor          -->     - PCAReducer                 |
|   - UCI Heart (303 x 13)         - HeartDiseasePreprocessor           (k in [2, 8], [-pi, pi])    |
|                                  - Splitter (Stratified 5-Fold)      - MISelector                 |
|                                  - PatientRecordValidator             (k in [2, 8], [0, 2pi])     |
|                                           |                                    |                  |
|                                           v                                    v                  |
|                                 [Classical Baselines (D5)]           [Handoff to Member B]        |
|                                  - Linear & RBF SVM                   - ZZFeatureMap              |
|                                  - PyTorch Deep MLP                   - AngleEmbedding            |
|                                  - Random Forest & XGBoost            - PQC / VQC / QSVM          |
|                                  - 5-Fold Cross-Validation Runner     - Exact Split Parity        |
|                                  - Zero-Leakage Fold Integrity Audit                              |
|                                  - ClinicalInferenceService (T5.2)                                |
|                                                                                                   |
|                                 [PonyTail Multi-Page UI Platform]                                 |
|                                  - Page 1: Clinical Ingestion & Single-Patient Sandbox            |
|                                  - Page 2: Quantum Feature Space Visualizer (k in [2, 8])         |
|                                  - Page 3: Baselines Leaderboard & Cryptographic Audit            |
|                                                                                                   |
|                                 [Dissemination (D10/D12)]                                         |
|                                  - 8-Page IEEE Methodology Report (PDF & Markdown)                |
+---------------------------------------------------------------------------------------------------+
```

---

## 🔬 Key Features & Deliverables

1. **Deterministic, Leak-Free Ingestion (D2):**
   - WDBC (569 samples, 30 continuous features).
   - UCI Cleveland Heart Disease (303 samples, 13 physiological/angiographic features).
   - Fold-isolated preprocessors (`WDBCPreprocessor`, `HeartDiseasePreprocessor`) guaranteeing zero evaluation set leakage.
2. **Qubit Dimensionality Reduction (D3):**
   - `PCAReducer`: Compresses feature spaces to $k \in [2, 8]$ principal components, scaled to $[-\pi, \pi]$ for $ZZ$-Feature Maps.
   - `MISelector`: Selects top $k$ features via non-linear mutual information (`mutual_info_classif`), preserving clinical column names for SHAP explainability.
3. **Classical Baselines & Tuned Grid Benchmarks (D5):**
   - 5 architectures evaluated across Stratified 5-Fold CV: Linear SVM, RBF SVM, Random Forest, XGBoost, and PyTorch Deep MLP.
   - Full out-of-fold predictions, latency profiling, and confusion matrices.
4. **Fold Integrity & Cryptographic Parity Audit (T3.3/T3.6):**
   - Formally verifies partition disjointness ($\text{train} \cap \text{val} = \emptyset$), exhaustive cohort coverage, and class balance preservation.
   - Signs SHA-256 cryptographic hashes for exact classical-quantum split parity.
5. **Single-Sample Clinical Hardening (T5.1/T5.2):**
   - `PatientRecordValidator`: Catches missing columns, invalid datatypes, and sensor out-of-bounds readings.
   - `ClinicalInferenceService`: Real-time bedside diagnostic pipeline with calibrated risk scoring and categorization.
6. **Multi-Page Streamlit UI (PonyTail Aesthetic):**
   - Page 1: Ingestion & Live Patient Validation Sandbox.
   - Page 2: Quantum Feature Space 2D/3D Manifold Visualizer.
   - Page 3: Baselines Leaderboard & Cryptographic Audit Badges.
7. **IEEE Conference Methodology Report (D10/D12):**
   - 8-page publication-grade PDF report compiled via ReportLab (`docs/HQML_Disease_Detection_Paper.pdf`).

---

## 📊 Tuned 5-Fold Cross-Validation Benchmarks

### Wisconsin Diagnostic Breast Cancer (WDBC)
| Model | Accuracy (Mean $\pm$ Std) | ROC-AUC | F1-Score | Optimal Parameters | Inference Latency / sample |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **RBF SVM** | $\mathbf{97.89\% \pm 1.19\%}$ | $\mathbf{0.9945}$ | $\mathbf{0.9719}$ | $C=10.0, \gamma=\text{'scale'}$ | $0.007\text{ ms}$ |
| **Linear SVM** | $97.54\% \pm 1.70\%$ | $0.9962$ | $0.9673$ | $C=0.1$ | $0.003\text{ ms}$ |
| **PyTorch Deep MLP** | $97.54\% \pm 1.16\%$ | $0.9948$ | $0.9670$ | $\text{lr}=0.005, \text{dropout}=0.1$ | $0.007\text{ ms}$ |
| **XGBoost** | $96.13\% \pm 1.43\%$ | $0.9897$ | $0.9479$ | $n=50, d=4, \eta=0.1$ | $0.008\text{ ms}$ |
| **Random Forest** | $95.43\% \pm 1.28\%$ | $0.9889$ | $0.9385$ | $n=50, d=4, \text{split}=5$ | $0.016\text{ ms}$ |

### UCI Cleveland Heart Disease
| Model | Accuracy (Mean $\pm$ Std) | ROC-AUC | F1-Score | Optimal Parameters | Inference Latency / sample |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **RBF SVM** | $\mathbf{85.48\% \pm 3.81\%}$ | $\mathbf{0.9083}$ | $\mathbf{0.8407}$ | $C=1.0, \gamma=\text{'auto'}$ | $0.008\text{ ms}$ |
| **Linear SVM** | $83.83\% \pm 3.38\%$ | $0.9110$ | $0.8143$ | $C=0.1$ | $0.003\text{ ms}$ |
| **Random Forest** | $83.82\% \pm 3.86\%$ | $0.9058$ | $0.8183$ | $n=100, d=4, \text{split}=2$ | $0.021\text{ ms}$ |
| **XGBoost** | $82.50\% \pm 2.94\%$ | $0.9015$ | $0.8035$ | $n=50, d=3, \eta=0.1$ | $0.008\text{ ms}$ |
| **PyTorch Deep MLP** | $82.18\% \pm 1.19\%$ | $0.8757$ | $0.8052$ | $\text{lr}=0.005, \text{dropout}=0.2$ | $0.007\text{ ms}$ |

---

## 🚀 Quickstart & Usage

### 1. Installation
```powershell
pip install -r requirements.txt # or python -m pip install torch scikit-learn xgboost streamlit plotly reportlab pytest
```

### 2. Run Test Suite (40/40 Passing)
```powershell
pytest -v
```

### 3. Launch Interactive Platform UI
```powershell
streamlit run app/Home.py
```

### 4. Single-Patient Clinical Inference
```python
from src.models.classical.inference import ClinicalInferenceService

service = ClinicalInferenceService(dataset_name="wdbc")
result = service.predict_patient(raw_patient_dict)
print(result.class_label, result.risk_score, result.risk_category)
```

---

## 📜 Repository Structure
```
├── app/
│   ├── Home.py                  # Main Research Portal
│   └── pages/
│       ├── 1_Data_Exploration.py # Ingestion & Bedside Diagnostic Sandbox
│       ├── 2_Features.py         # Quantum Feature Space & Manifold Explorer
│       └── 3_Baselines_Audit.py  # Leaderboard & Cryptographic Audit
├── configs/                     # Deterministic fold splits, audits, optimal params
├── data/                        # Raw & cached processed datasets
├── docs/                        # IEEE conference paper source & PDF compiler
├── src/
│   ├── data/                    # Loaders, preprocessors, validators, splitters
│   ├── features/                # PCA and Mutual Information qubit transformers
│   ├── models/                  # Base interface, classical models, tuner, inference
│   └── evaluation/              # 5-fold CV runner and fold integrity auditor
└── tests/                       # Complete pytest unit and integration test suite
```
