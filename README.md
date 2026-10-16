# 🧬 Hybrid Quantum Machine Learning Platform for Early Disease Detection (HQML-EDD)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38+-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](app/)

> **IEEE CAIBE 2026 Preprint & Research Platform**  
> A modular, simulator-first Hybrid Quantum-Classical Machine Learning (HQML) framework for high-dimensional biomedical disease detection across Wisconsin Diagnostic Breast Cancer (WDBC), UCI Cleveland Heart Disease, and Diabetic Retinopathy Fundus Photo cohorts.

---

## 👥 Collaborative Multidisciplinary Research Group

- **Member A (Lead & Data Pipelines / Classical ML):** Leak-free clinical data loaders, 5-fold stratification, Qubit dimensionality reduction (PCA & Mutual Information), classical baseline grid tuning (SVM, MLP, RF, XGBoost), cryptographic fold audit, and IEEE paper authoring.
- **Member B (Quantum Circuit Architect):** AngleEmbedding, $ZZ$-Feature Maps, Variational Quantum Classifiers (VQC), Quantum Kernel Support Vector Machines (QSVC), and strongly entangling layers.
- **Member C (Evaluation & Explainable AI):** Master 5-fold benchmarking, hypothesis testing for quantum advantage ($p < 0.05$), SHAP/LIME clinical biomarker concordance, and interactive explainability dashboard.

---

## 🎯 Clinical Benchmarks & Breakthroughs

In clinical diagnostics, false negatives carry severe mortality risks. HQML-EDD compares **5 Classical Models** against **3 Quantum Machine Learning Architectures** across **5 Clinical Performance Metrics + Inference Latency**.

1. **Quantum Sensitivity Advantage**: Quantum Classifiers (QSVC & VQC) reach **95.77% Sensitivity**, significantly reducing false negatives compared to classical models.
2. **Statistical Significance**: Paired Student's t-test across 5 folds confirms **Quantum Advantage** ($p = 0.0353 < 0.05$).
3. **Qubit Budget Pareto Frontier**: Ablation study shows $k=6$ principal components retains **88.76% of total variance** with negligible accuracy drop.
4. **Transparent Explainability**: Classical & Quantum Kernel SHAP achieve **Spearman $\rho = 0.943 - 1.000$ concordance**, dispelling the "quantum black box" misconception.

---

## 🏗️ Architecture Overview

```
Biomedical Data (WDBC / Heart / Fundus)
               │
               ▼
   [ Classical Feature Preprocessing ]
   (StandardScaler + PCA / Mutual Information → 4–8 Qubits in [-π, π])
               │
               ▼
   [ Quantum-Classical Engine Tier ]
   ┌────────────────────────────────────────────────────────┐
   │ • QSVC (Fidelity Transition Kernel: ZZ-FeatureMap)     │
   │ • VQC (Variational Ansatz + Parameter-Shift Adam)      │
   │ • HQNN (PyTorch nn.Linear + PennyLane TorchLayer)      │
   │ • Classical Baselines (SVM, Random Forest, XGB, MLP)   │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   [ Evaluation & Clinical Explainability (XAI) ]
   (5-Fold CV Benchmarks + Quantum KernelSHAP Patient Risk Attribution)
                               │
                               ▼
   [ Multi-Page Streamlit GUI Console & Visualizer ]
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

## ⚡ Quickstart & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Launch Interactive Platform UI
```bash
streamlit run app/Home.py
```
Open `http://localhost:8501` in your browser.

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
│   ├── Home.py                             # Streamlit diagnostic platform home
│   ├── main.py                             # Multi-modal interactive web dashboard
│   ├── components/circuit_viz.py           # Quantum circuit diagram generator
│   └── pages/
│       ├── 1_Data_Exploration.py           # Ingestion & Bedside Diagnostic Sandbox
│       ├── 2_Features.py                   # Quantum Feature Space & Manifold Explorer
│       ├── 3_Baselines_Audit.py            # Leaderboard & Cryptographic Audit
│       └── 5_Explainability.py             # Interactive patient risk gauge & SHAP UI
├── configs/                                # Deterministic fold splits, audits, optimal params
├── data/
│   ├── Sample_Fundus_Photos/               # 22 Clinical human eye fundus images
│   └── heart_disease.csv                   # UCI Heart disease benchmark data
├── docs/
│   ├── HQML_Disease_Detection_Paper.pdf    # 8-Page IEEE methodology publication report
│   ├── HQML_Presentation_Deck.pdf          # Publication-ready PDF deck for jury review
│   └── quantum_limitations.md              # NISQ constraints & hardware whitepaper
├── reports/
│   ├── figures/                            # Publication-grade ROC, Confusion Matrix, & PR SVGs
│   └── tables/                             # 5-Fold CV summary tables & p-values
├── src/
│   ├── data/                               # WDBC, Heart, and Fundus image loaders
│   ├── features/                           # PCA & Mutual Information quantum reducers
│   ├── quantum/                            # PennyLane backend, feature maps, ansatzes, noise
│   ├── models/quantum/                     # QSVC, VQC, and Hybrid QNN models
│   ├── models/classical/                   # Linear/RBF SVM, Random Forest, XGBoost, MLP
│   ├── evaluation/                         # 5-Fold CV runner, metrics, significance, audit
│   └── xai/                                # Classical and quantum SHAP explainability
└── tests/                                  # Comprehensive Pytest suites
```
