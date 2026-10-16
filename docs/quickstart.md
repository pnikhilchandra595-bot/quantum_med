# QuantumHealth: Quickstart & Setup Guide

This guide details the step-by-step procedure to set up, train, and deploy the Hybrid Quantum-Classical Clinical Diagnostics Platform on local workstations or cloud environments.

---

## 1. System Requirements

- **Operating System:** Linux (Ubuntu 20.04+), macOS (Monterey+), or Windows 10/11 (PowerShell or WSL2).
- **Python:** Version `3.10` or `3.11` (64-bit recommended).
- **Memory:** Minimum 8 GB RAM (16 GB recommended for multi-qubit tensor simulations).
- **Disk Space:** ~2 GB for virtual environment, PyTorch, and PennyLane wheels.

---

## 2. Local Installation (3-Step Setup)

### Step 1: Clone the Repository
```bash
git clone https://github.com/quantumhealth/hybrid-qml.git
cd hybrid-qml
```

### Step 2: Create and Activate Virtual Environment
On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
Install PyTorch CPU first (ensures lightweight footprint without requiring 3GB+ CUDA binaries):
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

*(Optional: Install developer test & linting packages)*:
```bash
pip install -r requirements-dev.txt
pre-commit install
```

---

## 3. Running the Streamlit Web Application

Launch the 6-page interactive clinical dashboard with a single command:
```bash
streamlit run app/main.py
```
Open your browser and navigate to:
```text
http://localhost:8501
```

### Navigating the Interface:
1. **Overview**: Architectural topology and real-time system metrics.
2. **1 Data Explorer**: Profile WDBC or Heart Disease data, compute statistics, and render interactive Plotly correlation heatmaps.
3. **2 Feature Studio**: Dimensionality reduction (PCA / Mutual Information) and quantum state angle-embedding circuits.
4. **3 Model Studio**: Real-time training of 4-qubit VQC, QSVC, or HQNN models with live loss convergence charts.
5. **4 Benchmark**: Comparator leaderboard with interactive ROC curves and wall-clock latency curves.
6. **5 Explainability**: KernelSHAP feature attribution waterfall plots explaining patient-level diagnostic decisions.

---

## 4. Headless CLI Orchestrator Usage

The platform includes a CLI (`cli.py`) for automated terminal execution, batch processing, and CI pipelines:

### 1. Train a Quantum Model (VQC)
```bash
python cli.py train --model vqc --dataset wdbc --qubits 4 --epochs 25 --lr 0.05
```

### 2. Train a Classical Baseline (SVM or Random Forest)
```bash
python cli.py train --model svm --dataset heart --qubits 4
python cli.py train --model rf --dataset wdbc
```

### 3. Run Single-Patient Inference
```bash
python cli.py predict --model checkpoints/vqc_wdbc.pt --input data/sample_patient.json
```

### 4. Run Comprehensive Benchmark Suite
```bash
python cli.py benchmark --dataset wdbc --qubits 4 --output outputs/benchmark_wdbc.json
```

---

## 5. Docker Containerized Deployment

Deploy the entire stack in an isolated, multi-stage container without local dependency conflicts:

### Using Docker Compose (Recommended)
```bash
docker compose up --build -d
```
Verify the container status:
```bash
docker compose ps
curl -I http://localhost:8501/_stcore/health
```
Stop the service:
```bash
docker compose down
```

---

## 6. Running the Automated Test Suite

Execute the unit test and serialization verification suite:
```bash
pytest tests/ -v
```

Run code formatting and style checks:
```bash
black --check src app tests cli.py
flake8 src app tests cli.py
```

---

## 7. Troubleshooting & FAQ

- **Issue:** `ImportError: DLL load failed while importing _multiarray_umath` (Windows).  
  **Resolution:** Install the latest Microsoft Visual C++ Redistributable 2015–2022.
- **Issue:** PennyLane execution is slow on CPU.  
  **Resolution:** For $N \le 8$ qubits, `default.qubit` provides state-vector execution in < 5 seconds. To accelerate higher qubit registers, install `pennylane-lightning`: `pip install pennylane-lightning`.
- **Issue:** Port 8501 is already in use.  
  **Resolution:** Specify an alternate port: `streamlit run app/main.py --server.port 8502`.
