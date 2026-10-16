# QuantumHealth: Hybrid Quantum-Classical Clinical Diagnostics Platform (HQML-EDD)

[![CI Pipeline](https://img.shields.io/badge/CI-Passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.34+-brightgreen.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

> **QuantumHealth (HQML-EDD)** is an enterprise-grade hybrid quantum-classical machine learning platform engineered to detect subtle non-linear biomarker correlations in oncology (**Wisconsin Diagnostic Breast Cancer - WDBC**), cardiology (**UCI Heart Disease**), and ophthalmology (**Diabetic Retinopathy Fundus Photos**) by exploiting high-dimensional quantum Hilbert spaces alongside rigorous classical baselines.

---

## 👥 Collaborative Multidisciplinary Research Group

| Team Member | Role & Responsibilities | Deliverables Handled |
|---|---|---|
| **Member A** | **Data Engineering & Classical ML Lead** | D2, D3, D5, D10/D12 (Ingestion, Preprocessing, PCA/MI Reducers, Classical Baselines, Fold Audit, IEEE Paper) |
| **Member B** | **Quantum Circuits & Backend Lead** | D2, D4, D6 (PennyLane Circuits, ZZFeatureMap, VQC, QSVC, HQNN, Device Abstraction) |
| **Member C** | **Evaluation & Explainability Lead** | D5, D7, D8, D11 (Hypothesis Testing, Statistical Tests, KernelSHAP, Pitch Presentation Deck) |
| **Member D** | **Platform, UI, DevOps & Docs Lead** | D1, D6, D9, D10, D11, D12 (Streamlit UI, ModelFactory, Headless CLI, CI/CD, Docker, Video, Docs) |

---

## 🔬 System Architecture Overview

The system features a **3-Tier Decoupled Architecture** bridging raw clinical record ingestion, quantum state preparation and variational ansatz optimization, and interactive clinical delivery with explainable AI (XAI).

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

![QuantumHealth 3-Tier Architecture](docs/assets/system_diagram.png)

### Key Architectural Highlights:
1. **Tier 1 (Data & Preprocessing)**: Robust clinical loaders for Wisconsin Diagnostic Breast Cancer (569 samples, 30 features), UCI Heart Disease (303 records, 13 features), and Retinal Fundus Photos. Scaled via PCA and Mutual Information for NISQ qubit registers ($N \in [2, 8]$).
2. **Tier 2 (Quantum-Classical Compute Engine)**: Unified `ModelFactory` and `QuantumBackendManager` supporting 7 architectures:
   - **Quantum Suite**: Quantum Support Vector Classifier (QSVC), Variational Quantum Classifier (VQC), Hybrid Quantum Neural Network (HQNN).
   - **Classical Baselines**: Linear & RBF SVM, Random Forest, XGBoost, Multi-Layer Perceptron (MLP).
3. **Tier 3 (Clinical Delivery & Explainability)**: Multi-page interactive Streamlit dashboard, model-agnostic KernelSHAP feature attributions, and headless CLI orchestrator.

---

## 📊 Comparative Benchmark Summary

### Wisconsin Diagnostic Breast Cancer (WDBC) — 5-Fold Cross-Validation
| Model | Accuracy (Mean $\pm$ Std) | ROC-AUC | F1-Score | Optimal Parameters | Inference Latency / sample |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **RBF SVM** | $\mathbf{97.89\% \pm 1.19\%}$ | $\mathbf{0.9945}$ | $\mathbf{0.9719}$ | $C=10.0, \gamma=\text{'scale'}$ | $0.007\text{ ms}$ |
| **Linear SVM** | $97.54\% \pm 1.70\%$ | $0.9962$ | $0.9673$ | $C=0.1$ | $0.003\text{ ms}$ |
| **PyTorch Deep MLP** | $97.54\% \pm 1.16\%$ | $0.9948$ | $0.9670$ | $\text{lr}=0.005, \text{dropout}=0.1$ | $0.007\text{ ms}$ |
| **VQC (Variational Quantum)** | $96.49\% \pm 1.30\%$ | $0.9854$ | $0.9668$ | $N=4\text{ qubits}, \text{layers}=2$ | $4.120\text{ s}$ |
| **XGBoost** | $96.13\% \pm 1.43\%$ | $0.9897$ | $0.9479$ | $n=50, d=4, \eta=0.1$ | $0.008\text{ ms}$ |
| **QSVC (Quantum Kernel)** | $95.61\% \pm 1.45\%$ | $0.9782$ | $0.9583$ | $N=4\text{ qubits}, \text{reps}=2$ | $3.850\text{ s}$ |
| **Random Forest** | $95.43\% \pm 1.28\%$ | $0.9889$ | $0.9385$ | $n=50, d=4, \text{split}=5$ | $0.016\text{ ms}$ |

### UCI Cleveland Heart Disease — 5-Fold Cross-Validation
| Model | Accuracy (Mean $\pm$ Std) | ROC-AUC | F1-Score | Optimal Parameters | Inference Latency / sample |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **RBF SVM** | $\mathbf{85.48\% \pm 3.81\%}$ | $\mathbf{0.9083}$ | $\mathbf{0.8407}$ | $C=1.0, \gamma=\text{'auto'}$ | $0.008\text{ ms}$ |
| **Linear SVM** | $83.83\% \pm 3.38\%$ | $0.9110$ | $0.8143$ | $C=0.1$ | $0.003\text{ ms}$ |
| **Random Forest** | $83.82\% \pm 3.86\%$ | $0.9058$ | $0.8183$ | $n=100, d=4, \text{split}=2$ | $0.021\text{ ms}$ |
| **XGBoost** | $82.50\% \pm 2.94\%$ | $0.9015$ | $0.8035$ | $n=50, d=3, \eta=0.1$ | $0.008\text{ ms}$ |
| **PyTorch Deep MLP** | $82.18\% \pm 1.19\%$ | $0.8757$ | $0.8052$ | $\text{lr}=0.005, \text{dropout}=0.2$ | $0.007\text{ ms}$ |

---

## ⚡ 3-Step Quickstart

### 1. Installation
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v
```

### 3. Launch Web Dashboard
```bash
streamlit run app/main.py
# Or launch portal:
streamlit run app/Home.py
```
Open **http://localhost:8501** in your browser to inspect datasets, explore quantum feature spaces, train models, and evaluate clinical risk scores.

---

## 💻 Unified CLI Orchestrator

The platform includes a CLI (`cli.py`) for automated terminal pipelines and batch inference:

```bash
# 1. Train a 4-qubit VQC on WDBC
python cli.py train --model vqc --dataset wdbc --qubits 4 --epochs 25 --lr 0.05

# 2. Train a Classical Baseline
python cli.py train --model svm --dataset heart --qubits 4

# 3. Single-Patient Diagnostic Inference
python cli.py predict --model checkpoints/vqc_wdbc.pt --input data/sample_patient.json

# 4. Run Benchmark Suite across all models
python cli.py benchmark --dataset wdbc --qubits 4 --output outputs/benchmark_wdbc.json
```

---

## 🐳 Docker Deployment

Deploy the pre-configured container using Docker Compose:

```bash
docker compose up --build -d
```
Access the application at `http://localhost:8501`. Check service health:
```bash
curl -I http://localhost:8501/_stcore/health
```

---

## 📁 Repository Directory Structure

```text
hybrid-qml/
├── .github/workflows/ci.yml       # GitHub Actions CI/CD Pipeline
├── app/
│   ├── Home.py                    # Streamlit diagnostic platform portal
│   ├── main.py                    # Streamlit Entry Point & Overview
│   ├── styles.css                 # Health Informatics Custom CSS
│   └── pages/
│       ├── 1_Data_Explorer.py     # Dataset Profiling & Correlation Heatmap
│       ├── 1_Data_Exploration.py  # Ingestion & Bedside Diagnostic Sandbox
│       ├── 2_Feature_Studio.py    # PCA, Scree Plots & Quantum Encoding
│       ├── 2_Features.py          # Quantum Feature Space & Manifold Explorer
│       ├── 3_Model_Studio.py      # Real-Time QML Trainer & Live Curves
│       ├── 3_Baselines_Audit.py   # Leaderboard & Cryptographic Audit
│       ├── 4_Benchmark.py         # Comparative Leaderboard & ROC Curves
│       └── 5_Explainability.py    # KernelSHAP Clinical Waterfall Plots
├── checkpoints/                   # Serialized Model Weights (.pt / .joblib)
├── cli.py                         # Unified CLI Orchestrator
├── configs/                       # Deterministic fold splits, audits, optimal params
├── data/
│   ├── Sample_Fundus_Photos/      # 22 Clinical human eye fundus images
│   ├── sample_patient.json        # Sample Biopsy Feature Vector for Inference
│   └── heart_disease.csv          # UCI Heart disease benchmark data
├── docs/
│   ├── HQML_Disease_Detection_Paper.pdf  # 8-Page IEEE methodology publication report
│   ├── HQML_Presentation_Deck.pdf        # Publication-ready PDF deck for jury review
│   ├── architecture.md            # Detailed 3-Tier Architectural Specification
│   ├── quickstart.md              # Installation & Deployment Guide
│   ├── demo_video_script.md       # 5-Minute HD Video Narration Script
│   └── video_link.md              # Video Metadata & Hosting Links
├── src/
│   ├── data/                      # Dataset Loaders (WDBC, UCI Heart, Fundus Photos)
│   ├── features/                  # PCA & Mutual Information quantum reducers
│   ├── models/                    # ModelFactory, Classical & Quantum models, inference
│   ├── quantum/                   # QuantumBackendManager & PennyLane Circuits
│   ├── evaluation/                # 5-Fold CV runner, metrics, significance, audit
│   └── xai/                       # Classical and quantum SHAP explainability
└── tests/                         # Comprehensive Pytest suites
```

---

## 📜 Citation & License

This project is licensed under the [MIT License](LICENSE).

```bibtex
@software{quantumhealth2026,
  author = {Team QuantumHealth},
  title = {QuantumHealth: A Production-Grade Hybrid Quantum-Classical Machine Learning Platform for Clinical Diagnostics},
  url = {https://github.com/pnikhilchandra595-bot/quantum_med},
  version = {1.0.0},
  year = {2026}
}
```
