# 🧬 Hybrid Quantum-Classical Machine Learning (HQML) for Biomedical Disease Detection

[![Tests](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](app/)

> **IEEE CAIBE 2026 Preprint & Research Platform**  
> A modular, simulator-first Hybrid Quantum-Classical Machine Learning (HQML) framework for high-dimensional biomedical disease detection across Wisconsin Diagnostic Breast Cancer (WDBC) and UCI Cleveland Heart Disease cohorts.

---

## 👥 Collaborative Multidisciplinary Research Group

- **Member A (Lead & Data Pipelines / Classical ML):** Leak-free clinical data loaders, 5-fold stratification, Qubit dimensionality reduction (PCA & Mutual Information), classical baseline grid tuning (SVM, MLP, RF, XGBoost), cryptographic fold audit, and IEEE paper authoring.
- **Member B (Quantum Circuit Architect):** AngleEmbedding, $ZZ$-Feature Maps, Variational Quantum Classifiers (VQC), Quantum Kernel Support Vector Machines (QSVM), and strongly entangling layers.
- **Member C (Evaluation & Explainable AI):** Master 5-fold benchmarking, hypothesis testing for quantum advantage ($p < 0.05$), SHAP/LIME clinical biomarker concordance, and interactive explainability dashboard.

---

## 🏛️ System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                     HQML SYSTEM ARCHITECTURE                                      |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Clinical Cohorts]             [Data Layer (D2)]                   [Qubit Encodings (D3)]        |
|   - WDBC (569 x 30)       -->    - WDBCPreprocessor          -->     - PCAReducer                 |
|   - UCI Heart (303 x 13)         - HeartDiseasePreprocessor           (k in [2, 8], [-pi, pi])    |
|                                  - Splitter (Stratified 5-Fold)      - MISelector                 |
|                                  - PatientRecordValidator             (k in [2, 8], [0, 2pi])     |
|                                           |                                    |                  |
|                                           v                                    v                  |
|                                 [Classical Baselines (D5)]           [Quantum Circuits (D4)]      |
|                                  - Linear & RBF SVM                   - ZZFeatureMap              |
|                                  - PyTorch Deep MLP                   - AngleEmbedding            |
|                                  - Random Forest & XGBoost            - PQC / VQC / QSVM          |
|                                  - 5-Fold Cross-Validation Runner     - Exact Split Parity        |
|                                  - Zero-Leakage Fold Integrity Audit                              |
|                                  - ClinicalInferenceService (T5.2)                                |
|                                           |                                    |                  |
|                                           +------------------+-----------------+                  |
|                                                              v                                    |
|                                                [Evaluation & XAI (D7, D8)]                        |
|                                                 - Paired t-test (p = 0.035)                       |
|                                                 - KernelSHAP Biomarker Alignment                  |
|                                                 - Full Out-of-Fold Diagnostics                    |
|                                                                                                   |
|                                 [PonyTail Multi-Page UI Platform]                                 |
|                                  - Page 1: Clinical Ingestion & Single-Patient Sandbox            |
|                                  - Page 2: Quantum Feature Space Visualizer (k in [2, 8])         |
|                                  - Page 3: Baselines Leaderboard & Cryptographic Audit            |
|                                  - Page 5: Interactive SHAP Patient Explainability (XAI)          |
|                                                                                                   |
|                                 [Dissemination (D10/D12)]                                         |
|                                  - 8-Page IEEE Methodology Report (docs/HQML_Disease_Detection_Paper.pdf)|
|                                  - 15-Slide Presentation Deck (docs/HQML_Presentation_Deck.pdf)   |
+---------------------------------------------------------------------------------------------------+
```

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
pip install -r requirements.txt # or install dependencies
```

### 2. Run Test Suite
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
│       ├── 3_Baselines_Audit.py  # Leaderboard & Cryptographic Audit
│       └── 5_Explainability.py   # SHAP & Feature Attributions
├── configs/                     # Deterministic fold splits, audits, optimal params
├── data/                        # Raw & cached processed datasets
├── docs/                        # IEEE conference paper source & pitch decks
├── reports/                     # Figures, tables, and XAI comparison reports
├── src/
│   ├── data/                    # Loaders, preprocessors, validators, splitters
│   ├── features/                # PCA and Mutual Information qubit transformers
│   ├── models/                  # Base interface, classical & quantum models, tuner, inference
│   ├── evaluation/              # 5-fold CV runner, metrics, significance, audit
│   └── xai/                     # Classical and quantum SHAP explainability
└── tests/                       # Complete pytest unit and integration test suite
```
