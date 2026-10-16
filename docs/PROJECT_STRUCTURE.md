# Project Architecture & Directory Segregation

This document outlines the organized, modular segregation of the **QuantumHealth (HQML-EDD)** repository into dedicated frontend, backend, documentation, and auxiliary subsystems.

```
quantum_med/
├── backend/                       # Dedicated Python Backend Platform
│   ├── src/                       # Core ML, Quantum & API algorithms
│   │   ├── api/                   # FastAPI REST API (server.py: predict, benchmark, telemetry)
│   │   ├── data/                  # Data ingestion, validation, preprocessors, loaders
│   │   ├── evaluation/            # 5-fold CV runners, fold integrity audit, metrics, statistical tests
│   │   ├── features/              # PCA and Mutual Information dimensionality reduction
│   │   ├── models/                # Classical (SVM, RF, XGB, MLP) and Quantum (QSVC, VQC, HQNN)
│   │   ├── quantum/               # PennyLane circuit ansatzes, noise simulation, backend manager
│   │   ├── utils/                 # Serialization, profiling, performance caching
│   │   └── xai/                   # Classical TreeSHAP and Quantum KernelSHAP explainers
│   ├── app/                       # Streamlit Clinical Research Intelligence Portal
│   │   ├── components/            # UI widgets (FundusViewer, WhatIfSimulator, TelemetryHUD, etc.)
│   │   ├── pages/                 # Multi-page clinical research consoles
│   │   ├── Home.py                # Main Streamlit entrance
│   │   ├── main.py                # Alternate streamlined portal
│   │   └── styles.css             # Glassmorphism styling rules
│   ├── checkpoints/               # Trained model weights and benchmark cache files
│   │   ├── qsvc_wdbc.joblib
│   │   ├── svm_wdbc.joblib
│   │   ├── vqc_wdbc.pt
│   │   └── benchmark_results.json
│   ├── configs/                   # Stratified fold splits and hyperparameter JSONs
│   │   ├── folds_heart.json
│   │   ├── folds_wdbc.json
│   │   └── optimal_hyperparams.json
│   ├── data/                      # Medical datasets (Fundus images, Heart CSV, Kaggle/UCI)
│   │   ├── Sample_Fundus_Photos/  # Clinical ophthalmic fundus photos
│   │   ├── kaggle_uci_datasets/   # Multi-organ clinical cohorts (Liver, Kidney, Lung, etc.)
│   │   └── heart_disease.csv
│   ├── tests/                     # 115 passing Pytest automated test suite
│   ├── cli.py                     # Backend command-line orchestrator
│   ├── Dockerfile                 # Backend container definition
│   ├── pyproject.toml             # Backend package metadata & dependencies
│   ├── pytest.ini                 # Backend pytest runner configuration
│   ├── requirements.txt           # Production Python dependencies
│   └── requirements-dev.txt       # Development & testing Python dependencies
│
├── frontend/                      # Modern Web Application (React + TypeScript + Vite)
│   ├── src/
│   │   ├── components/            # Dynamic clinical UI (RadarArena, TelemetryHUD, WhatIfSimulator, etc.)
│   │   ├── lib/api.ts             # Typed REST API client communicating with backend:8000
│   │   ├── App.tsx                # Main single-page application orchestrator
│   │   └── main.tsx               # React DOM entry point
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.ts             # Vite build & proxy configuration
│   └── tsconfig.json              # TypeScript configuration
│
├── docs/                          # Comprehensive Technical & Clinical Documentation
│   ├── DEMO_SCRIPT.md             # 5-minute Master live demo script & judge defense Q&A
│   ├── demo_video_script.md       # Exact 300-second calibrated video walkthrough script
│   ├── DELIVERABLES.md            # Deliverable matrix and completion audit
│   ├── PROJECT_STRUCTURE.md       # This directory map
│   ├── PROJECT_RULES.md           # Engineering, coding, and architectural standards
│   ├── GSD-STYLE.md               # Styling & presentation guide
│   ├── MEMBER_D_TASKS.md          # Multi-member task breakdown & responsibilities
│   ├── architecture.md            # In-depth system architecture specification
│   ├── api_reference.md           # FastAPI REST API documentation & schema guide
│   ├── benchmarks.md              # 5-fold cross-validation benchmarks & statistical significance
│   ├── deployment.md              # Docker, local, and production deployment runbook
│   ├── pitch_deck.md              # Hackathon / conference pitch presentation slides
│   ├── paper_source.md            # Full IEEE-formatted research paper manuscript
│   ├── quantum_limitations.md     # NISQ constraints & practical quantum advantage criteria
│   ├── quickstart.md              # Developer getting-started guide
│   └── video_link.md              # Video submission links & asset indices
│
├── notebooks/                     # Interactive Jupyter Research Notebooks
│   └── demo_walkthrough.ipynb     # Step-by-step interactive quantum diagnostic workflow
│
├── scripts/                       # Maintenance, benchmarks, and automation scripts
│   ├── run_master_benchmark.py
│   ├── audit_api_endpoints.py
│   └── audit_assets.py
│
├── reports/                       # Generated audit figures, reviews, and tables
│   ├── figures/
│   ├── reviews/
│   └── tables/
│
├── cli.py                         # Root CLI launcher forwarding directly to backend/cli.py
├── pytest.ini                     # Root pytest config enabling testing across backend/tests
├── Dockerfile                     # Root multi-stage Docker build
├── docker-compose.yml             # Docker Compose orchestration
├── README.md                      # Primary repository overview
└── LICENSE                        # MIT License
```

---

## Quick Command Reference

### Backend Execution
```bash
# Navigate to backend
cd backend

# Run FastAPI Server (port 8000)
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit Portal (port 8501)
streamlit run app/main.py --server.port 8501

# Run Tests
pytest
```

### Frontend Execution
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server (port 5173)
npm run dev

# Build for production
npm run build
```

### Root Convenience Commands
```bash
# Run tests from repository root
pytest

# Run CLI from repository root
python cli.py train --model qsvc --dataset wdbc --qubits 4
```
