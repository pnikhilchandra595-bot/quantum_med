# MEMBER_D_TASKS.md — Platform, UI, DevOps & Documentation Lead

> **Owner:** Member D (Full-Stack & DevOps Engineer)  
> **Backup:** Member A (Architecture & CLI) / Member C (Testing & UI)  
> **Primary Deliverables:** D1 (Architecture), D6 (CLI & Workflows), D9 (Streamlit UI), D10 (Docs), D11 (CI/Docker), D12 (Video & Repo)  
> **Total Direct Task Hours:** ~84 Hours across 5 Phases  

---

## Lane Objective
Build the end-to-end software platform infrastructure: author system architecture documentation, build the unified ModelFactory and CLI, develop the 6-page interactive Streamlit web dashboard, containerize the stack in Docker, configure GitHub Actions CI, produce the 5-minute video demo, and prepare the final open-source GitHub repository.

---

## Phase 1 & 2: Architecture, Repo Setup & CLI Engine

<task type="auto" effort="medium">
  <name>T1.1: Architecture Design Document Authoring</name>
  <files>
    docs/architecture.md
    docs/assets/system_diagram.png
  </files>
  <action>
    Document 3-tier hybrid quantum-classical architecture in docs/architecture.md:
    1. Data & Pre-processing Tier (Loaders, Scalers, PCA/MI reducers).
    2. Quantum-Classical Engine Tier (PennyLane simulators, QSVC, VQC, HQNN, Scikit-learn baselines).
    3. Evaluation, XAI & User Interface Tier (Streamlit GUI, KernelSHAP, Benchmarks).
    Include complete Mermaid component and sequence diagrams.
  </action>
  <verify>
    All team members review and sign off on module interfaces.
  </verify>
  <done>
    Architecture document committed and ratified by all 4 members.
  </done>
</task>

<task type="auto" effort="low">
  <name>T1.2: Repository Scaffolding & Pre-commit Hooks</name>
  <files>
    .pre-commit-config.yaml
    pyproject.toml
    .gitignore
  </files>
  <action>
    Configure repository standards:
    1. Gitignore for data caches, checkpoints, and bytecode.
    2. Setup pre-commit hooks for black (code formatting), flake8 (linting), and isort.
    3. Setup Poetry or pip requirements structure.
  </action>
  <verify>
    pre-commit run --all-files
  </verify>
  <done>
    Pre-commit runs and enforces uniform code formatting on git commit.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T3.1: ModelFactory and Unified CLI Orchestrator</name>
  <files>
    src/models/factory.py
    cli.py
  </files>
  <action>
    Build unified CLI using argparse or click:
    Options:
    - `python cli.py train --model [svm|rf|xgb|mlp|qsvc|vqc|hqnn] --dataset [wdbc|heart] --qubits [4] --epochs [30]`
    - `python cli.py predict --model [name] --input sample.json`
    - `python cli.py benchmark --dataset [wdbc|heart]`
    Decouple backend instantiation using Member B's QuantumBackendManager.
  </action>
  <verify>
    python cli.py --help
  </verify>
  <done>
    All 7 models trainable and callable through standard terminal flags.
  </done>
</task>

<task type="auto" effort="low">
  <name>T3.2: Model Checkpointing and Serialization Engine</name>
  <files>
    src/utils/serialization.py
    tests/test_serialization.py
  </files>
  <action>
    Implement save_model(model, filepath) and load_model(filepath):
    - For classical models: joblib/pickle.
    - For PyTorch/HQNN: torch.save state_dict.
    - For VQC/QSVC: store weights, bias, scalers, and hyperparameter config dict.
  </action>
  <verify>
    pytest tests/test_serialization.py
  </verify>
  <done>
    Saved model reload test asserts identical predictions on test set.
  </done>
</task>

---

## Phase 4: Interactive Web Platform (Streamlit) & Containerization

<task type="auto" effort="medium">
  <name>T4.1: Streamlit Dashboard Skeleton & Navigation</name>
  <files>
    app/main.py
    app/styles.css
  </files>
  <action>
    Build multi-page Streamlit dashboard layout:
    - Sidebar navigation: Overview, Data Explorer, Feature Studio, Model Trainer, Leaderboard, Explainability.
    - Modern medical dark/light theme CSS styling with health informatics aesthetics.
  </action>
  <verify>
    streamlit run app/main.py
  </verify>
  <done>
    Clean, responsive web app launches in browser at http://localhost:8501.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T4.2: Data Explorer UI Page</name>
  <files>
    app/pages/1_Data_Explorer.py
  </files>
  <action>
    Build dataset inspection page:
    1. Select pre-loaded WDBC or Heart Disease dataset, or upload custom CSV.
    2. Render statistical summary table (mean, std, missing values).
    3. Render interactive correlation heatmap (Plotly).
  </action>
  <verify>
    Upload custom test CSV in browser; verify interactive table renders.
  </verify>
  <done>
    Data profiling and correlation graphs render smoothly.
  </done>
</task>

<task type="auto" effort="high">
  <name>T4.4: Model Studio & Real-Time Trainer UI</name>
  <files>
    app/pages/3_Model_Studio.py
  </files>
  <action>
    Build interactive model training interface:
    1. Select model type (Classical vs Quantum).
    2. Configure hyperparameters (qubits: 4, reps: 1, epochs: 25, lr: 0.05).
    3. Click "Start Training" with animated progress bar and live loss curve chart.
    4. Save trained weights to user session state.
  </action>
  <verify>
    Train 4-qubit VQC in browser; verify loss curve updates dynamically.
  </verify>
  <done>
    Non-technical user can train and test quantum models with 1 click.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T4.5: Benchmarking Leaderboard UI Page</name>
  <files>
    app/pages/4_Benchmark.py
  </files>
  <action>
    Render interactive comparator leaderboard:
    - Master metrics table with color-coded high scores.
    - Interactive Plotly ROC curves allowing users to toggle individual models on/off.
    - Bar charts comparing training wall-clock time vs accuracy.
  </action>
  <verify>
    Toggle ROC curve checkboxes in browser; verify chart updates.
  </verify>
  <done>
    Interactive benchmarking dashboard operational.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T4.8: Multi-Stage Docker Containerization</name>
  <files>
    Dockerfile
    docker-compose.yml
    .dockerignore
  </files>
  <action>
    Package entire application into lightweight Docker image:
    1. Python 3.11-slim base image.
    2. Install PennyLane, PyTorch CPU, scikit-learn, Streamlit.
    3. Expose port 8501.
    4. Provide one-command startup via docker-compose up.
  </action>
  <verify>
    docker compose up --build -d && curl -I http://localhost:8501
  </verify>
  <done>
    Container builds and serves application cleanly without host dependencies.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T4.9: GitHub Actions CI/CD Pipeline</name>
  <files>
    .github/workflows/ci.yml
  </files>
  <action>
    Automate quality control in GitHub Actions:
    1. Triggers on PR to main.
    2. Runs flake8 and black --check.
    3. Executes pytest test suite across classical and quantum modules.
  </action>
  <verify>
    Push branch to GitHub; verify green CI checkmark.
  </verify>
  <done>
    Automated CI passes under 4 minutes.
  </done>
</task>

---

## Phase 5: Documentation, Demo Video & Final Release

<task type="auto" effort="medium">
  <name>T5.3: Production README.md & Installation Guide</name>
  <files>
    README.md
    docs/quickstart.md
  </files>
  <action>
    Write polished GitHub README:
    - Badges (CI, Python 3.11, License, Streamlit).
    - Architecture diagram and GIF walkthrough.
    - Quickstart: 3-step setup via git clone, pip install, and streamlit run.
    - Benchmark results summary table.
  </action>
  <verify>
    Follow README on a clean virtual environment and verify setup completes in < 15 mins.
  </verify>
  <done>
    World-class open-source documentation.
  </done>
</task>

<task type="auto" effort="high">
  <name>T5.7: 5-Minute HD Demo Video Production</name>
  <files>
    docs/demo_video_script.md
    docs/video_link.md
  </files>
  <action>
    Produce high-quality 5-minute video presentation:
    - 0:00-1:00: Problem Statement & Clinical Motivation.
    - 1:00-2:00: Hybrid Quantum-Classical Architecture (Hilbert space advantage).
    - 2:00-3:30: Live Streamlit Platform Walkthrough (Train VQC, view SHAP).
    - 3:30-4:30: Benchmarking Results & Statistical Significance ($p < 0.05$).
    - 4:30-5:00: Near-Term Hardware Scalability & Conclusion.
  </action>
  <verify>
    Review video with all team members for clarity, pacing, and audio quality.
  </verify>
  <done>
    Rendered 1080p MP4 demo video uploaded with captions.
  </done>
</task>

<task type="auto" effort="low">
  <name>T5.9: Final Repository Audit & Release v1.0.0 Tagging</name>
  <files>
    LICENSE
    CITATION.cff
  </files>
  <action>
    Perform final repository polish:
    1. Add MIT/Apache-2.0 License.
    2. Add CITATION.cff for academic citations.
    3. Tag git release v1.0.0 and verify all deliverables D1-D12 are accounted for.
  </action>
  <verify>
    git status is clean; git tag -l confirms v1.0.0.
  </verify>
  <done>
    Project ready for final SIH / competition submission.
  </done>
</task>
