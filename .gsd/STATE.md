# STATE.md — Project Execution State Memory

> **Status:** All Milestones Integrated & Verified  
> **Collaborators:** Member A (Data & Classical ML Lead), Member B (Quantum ML Lead), Member C (Evaluation & XAI Lead)  
> **Updated:** 2026-10-16  

---

## Deliverables & Artifact Checklist

### Deliverable D2: Data Ingestion & Sanitization
- [x] `src/data/wdbc_loader.py` (WDBC loader, `WDBCPreprocessor` leak-free pipeline)
- [x] `src/data/heart_loader.py` (UCI Heart Disease loader, `HeartDiseasePreprocessor`)
- [x] `src/data/splitter.py` (Stratified 5-Fold split generator)
- [x] `src/data/validator.py` (`DataValidator` and `PatientRecordValidator` detecting missing columns, invalid datatypes, sensor bounds)
- [x] `src/data/fundus_loader.py` (Clinical eye fundus photos feature extractor with 11 biomarkers)
- [x] `data/Sample_Fundus_Photos/` (22 real human eye fundus clinical photographs)
- [x] `configs/folds_wdbc.json` & `configs/folds_heart.json` (Persisted split configurations)

### Deliverable D3: Feature Selection & Quantum Encoding
- [x] `src/features/pca_reducer.py` (`PCAReducer` compressing to $k \in [2, 8]$ features, scaled to $[-\pi, \pi]$)
- [x] `src/features/mi_selector.py` (`MISelector` ranking features via mutual info, column name preservation, $[-\pi, \pi]$ scaling)

### Deliverable D4: Quantum Core Models & Circuits
- [x] `src/quantum/backend.py` (Quantum device abstraction for PennyLane `default.qubit` and Qiskit)
- [x] `src/quantum/feature_maps.py` (ZZFeatureMap and AngleEmbedding)
- [x] `src/quantum/ansatz.py` (Strongly entangling layers ansatz)
- [x] `src/quantum/noise.py` (Depolarizing and thermal noise simulation)
- [x] `src/models/quantum/qsvc.py` (Quantum Support Vector Classifier)
- [x] `src/models/quantum/vqc.py` (Variational Quantum Classifier)
- [x] `src/models/quantum/hqnn.py` (Hybrid Quantum Neural Network with PyTorch TorchLayer)

### Deliverable D5: Classical ML Baselines & Audit
- [x] `src/models/classical/svm.py` (`ClassicalSVM`, `LinearSVMModel`, `RBFSVMModel`)
- [x] `src/models/classical/mlp.py` (`ClassicalMLP`, `MLPModel`, `PyTorchMLPNet`)
- [x] `src/models/classical/rf.py` (`ClassicalRandomForest`, `RandomForestModel`)
- [x] `src/models/classical/xgb.py` (`ClassicalXGBoost`, `XGBoostModel`)
- [x] `src/models/classical/inference.py` (`ClinicalInferenceService` single-patient bedside inference with validation guardrails)
- [x] `src/evaluation/fold_audit.py` (`FoldIntegrityAuditor` with SHA-256 cryptographic signatures)
- [x] `src/evaluation/cv_runner.py` (`CrossValidationRunner` with latency benchmarks and OOF collections)
- [x] `src/models/classical/tuner.py` (`HyperparameterTuner` grid optimizer)
- [x] `configs/fold_integrity_audit.json` (Signed Fold Integrity Certificate)
- [x] `configs/optimal_hyperparams.json` (Tuned hyperparameter grids)
- [x] `configs/benchmark_results.json` (Full 5-fold evaluation metrics & OOF predictions)

### Deliverable D8 & D9: Explainability & Multi-Page Platform
- [x] `src/xai/quantum_shap.py` (KernelSHAP and Permutation importance for quantum circuits)
- [x] `app/Home.py` (Main portal with PonyTail styling and lane cards)
- [x] `app/pages/1_Data_Exploration.py` (Interactive clinical dataset explorer and live patient diagnostic sandbox)
- [x] `app/pages/2_Features.py` (Interactive Qubit Projection UI with slider $k \in [2, 8]$)
- [x] `app/pages/3_Baselines_Audit.py` (Baselines leaderboard & Fold Integrity Audit Dashboard)
- [x] `app/pages/5_Pitch_Presentation.py` (Executive slide deck)

### Deliverable D10/D12: Publication & Documentation
- [x] `docs/paper_source.md` (IEEE conference paper markdown synchronized with Phase 3 tuned results and audit signatures)
- [x] `docs/generate_pdf.py` (ReportLab script building publication-grade PDF)
- [x] `docs/HQML_Disease_Detection_Paper.pdf` (Compiled 8-page IEEE conference report)
- [x] `docs/quantum_limitations.md` (Analytical report on barren plateaus and circuit depth)
