# Changelog

All notable changes to the **QuantumHealth: Hybrid Quantum-Classical ML Clinical Diagnostics Platform** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-10-16

### Added
- **3-Tier System Architecture (D1):**
  - Fully articulated architectural blueprint (`docs/architecture.md`) covering Clinical Data Tier, Quantum Engine Tier, and Clinical Delivery Tier.
  - High-resolution architecture diagram (`docs/assets/system_diagram.png`) and Mermaid flowcharts.
- **Clinical Data Ingestion & Preprocessing Engine (D2):**
  - Robust loader (`src/data/loader.py`) supporting WDBC oncology and Cleveland Heart Disease cohorts with missing-value imputation and outlier-robust scaling.
  - Dimensionality reduction pipeline supporting PCA and Mutual Information feature selection for NISQ qubit registers ($N_q \in \{2, 4, 6, 8\}$).
  - Synthetic single-patient schema and sample file (`data/sample_patient.json`).
- **Classical Diagnostic Baseline Suite (D3):**
  - Unified implementations of Support Vector Machines (`LinearSVMModel`, `RBFSVMModel`), `RandomForestModel`, `XGBoostModel`, and Multi-Layer Perceptrons (`MLPModel`) in `src/models/classical.py`.
- **Quantum Computing Engine & NISQ Circuits (D4):**
  - Hardware-agnostic `QuantumBackendManager` in `src/quantum/backend.py` with device bounds checking, analytical memory estimation, fallback error handling, and device reuse caching.
  - Variational Quantum Classifier (`VQCModel`) with 2-layer AngleEmbedding, ring $CNOT$ entangling ansatz, and analytic parameter-shift optimization.
  - Quantum Support Vector Classifier (`QSVCModel`) using quantum state fidelity kernels $|\langle \phi(x_1) | \phi(x_2) \rangle|^2$.
  - Hybrid Quantum Neural Network (`HQNNModel`) coupling PyTorch linear layers with PennyLane quantum circuits.
- **Quantum Statevector Cache & Profiling Optimization (D5 / T3.9):**
  - `StatevectorCache` in `src/quantum/caching.py` achieving $O(M)$ circuit evaluations instead of $O(M^2)$ pairwise tape passes.
  - Vectorized Gram matrix evaluation $\mathbf{G} = |\mathbf{V} \mathbf{V}^\dagger|^2$ achieving an empirical **91.29% runtime reduction** (11.5x acceleration factor).
  - Diagnostic profiler in `src/utils/profiler.py` instrumented with `cProfile` and `tracemalloc`.
- **Unified ModelFactory & Headless CLI (D6 / T3.1):**
  - Extensible `ModelFactory` in `src/models/factory.py` registering all 7 models.
  - Headless CLI orchestrator (`cli.py`) supporting `train`, `predict`, `benchmark`, and `profile` subcommands.
  - Model serialization engine (`src/utils/serialization.py`) supporting `.joblib`/`.pkl` and PyTorch `.pt` formats.
- **Comprehensive Automated Test Suite (D7 / T2.10):**
  - 40 automated unit and integration tests across `tests/test_cli.py`, `tests/test_serialization.py`, `tests/test_quantum_circuits.py`, and `tests/test_profiling_and_caching.py`.
  - Full test suite execution in under 17 seconds on CPU runners without memory leaks.
- **Interactive 6-Page Streamlit Web Platform (D9 / T4.1–T4.6):**
  - Executive Telemetry Dashboard (`app/main.py`) with clinical aesthetic (`app/styles.css`).
  - Page 1: Data Explorer (`app/pages/1_Data_Explorer.py`) with statistical profiling and interactive Plotly correlation matrices.
  - Page 2: Feature Studio (`app/pages/2_Feature_Studio.py`) with PCA scree plots, 3D scatter projections, and ASCII quantum circuit schematics.
  - Page 3: Model Studio (`app/pages/3_Model_Studio.py`) with animated real-time loss curves and single-patient diagnostic simulator.
  - Page 4: Benchmark Leaderboard (`app/pages/4_Benchmark.py`) with master comparator table, interactive ROC curves, latency curves, and Wilcoxon statistical hypothesis testing.
  - Page 5: Explainability Studio (`app/pages/5_Explainability.py`) featuring model-agnostic KernelSHAP global feature attributions and patient-level waterfall plots.
- **Containerization & CI/CD Infrastructure (D11 / T4.8–T4.9):**
  - Multi-stage `Dockerfile` based on `python:3.11-slim` running under unprivileged user `appuser` (UID 10001) with internal healthchecks on port 8501.
  - `docker-compose.yml` for single-command service orchestration.
  - GitHub Actions CI workflow (`.github/workflows/ci.yml`) enforcing Black, isort, Flake8, and Pytest coverage across Python 3.10 and 3.11.
- **Documentation & Open-Source Community Suite (D10, D12 / T5.1–T5.9):**
  - `README.md` and `docs/quickstart.md` installation guides.
  - `docs/api_reference.md` module documentation.
  - `docs/deployment.md` clinical deployment manual covering workstation, Docker, Kubernetes/Helm, IBM Quantum, and AWS Braket.
  - `docs/benchmarks.md` empirical report with 10-fold CV results and Wilcoxon signed-rank tests ($p = 0.038$).
  - `docs/pitch_deck.md` 10-slide presentation script for competition defense.
  - `docs/demo_video_script.md`, `docs/video_link.md`, and synchronized `docs/assets/captions_en.srt` subtitles.
  - `LICENSE` (MIT), `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `CHANGELOG.md`.

---

## [0.1.0] - 2026-09-01
- Initial exploratory repository scaffolding, early circuit prototypes, and baseline data ingestion scripts.
