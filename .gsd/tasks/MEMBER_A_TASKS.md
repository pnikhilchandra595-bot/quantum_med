# MEMBER_A_TASKS.md — Data Pipelines & Classical ML Lead

> **Owner:** Member A (Lead / Classical ML Engineer)  
> **Backup:** Member C (Evaluation & XAI) / Member B (Quantum)  
> **Primary Deliverables:** D2 (Data Ingestion), D3 (Feature Selection), D5 (Classical Baselines), D10/D12 (Report)  
> **Total Direct Task Hours:** ~55 Hours across 5 Phases  

---

## Lane Objective
Deliver bulletproof, reproducible data loaders for Wisconsin Diagnostic Breast Cancer (WDBC) and UCI Heart Disease datasets, implement dimensionality reduction (PCA and Mutual Information) to feed Member B's 4–8 qubit quantum circuits, train 4 classical baseline models (SVM, Random Forest, XGBoost, MLP), and co-author the final IEEE methodology report.

---

## Phase 1: Ingestion, Splits & Classical Baseline Models

<task type="auto" effort="medium">
  <name>T1.3: WDBC Dataset Loader and Imputer</name>
  <files>
    src/data/wdbc_loader.py
    tests/test_data_loaders.py
  </files>
  <action>
    Implement download/load logic for Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
    Steps:
    1. Fetch WDBC from UCI ML repository or sklearn.datasets.load_breast_cancer.
    2. Convert 30 continuous features into float64 NumPy array/Pandas DataFrame.
    3. Binary encode targets: Malignant (M) -> 1, Benign (B) -> 0.
    4. Fit StandardScaler and verify zero mean and unit variance.
    5. Save processed data cache to data/processed/wdbc_scaled.npy.

    AVOID: Fitting scalers on full dataset before CV split (causes data leakage).
    USE: A fit/transform pipeline interface that accepts train/test splits cleanly.
  </action>
  <verify>
    python -c "from src.data.wdbc_loader import load_wdbc; X, y = load_wdbc(); assert X.shape == (569, 30); assert set(y) == {0, 1}; print('WDBC Loader verified successfully.')"
  </verify>
  <done>
    Returns clean (569, 30) feature matrix with 0 missing values and binary labels.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T1.4: UCI Heart Disease Dataset Loader and Preprocessor</name>
  <files>
    src/data/heart_loader.py
    tests/test_data_loaders.py
  </files>
  <action>
    Implement loader for UCI Heart Disease (Cleveland dataset, 303 samples, 13 features).
    Steps:
    1. Parse raw data handling missing values ('?' markers in ca and thal columns).
    2. Impute missing values using MedianImputer.
    3. One-hot encode categorical features (cp, restecg, slope, thal).
    4. Binary encode target (disease presence: >0 -> 1, 0 -> 0).
    5. StandardScale numerical columns (age, trestbps, chol, thalach, oldpeak).
  </action>
  <verify>
    python -c "from src.data.heart_loader import load_heart_disease; X, y = load_heart_disease(); assert X.shape[0] == 303; assert not np.isnan(X).any(); print('Heart loader verified successfully.')"
  </verify>
  <done>
    Dataset loaded without NaNs, scaled, and binary target normalized.
  </done>
</task>

<task type="auto" effort="low">
  <name>T1.5: Stratified 5-Fold Split Generator</name>
  <files>
    src/data/splitter.py
    tests/test_data_loaders.py
  </files>
  <action>
    Build reproducible StratifiedKFold cross-validation splitter.
    Steps:
    1. Use sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42).
    2. Save fold indices to configs/folds_wdbc.json and configs/folds_heart.json.
    3. Ensure training and validation fold generators preserve exact class balance across folds.
  </action>
  <verify>
    python -c "from src.data.splitter import get_folds; folds = get_folds('wdbc'); assert len(folds) == 5; print('Folds verified.')"
  </verify>
  <done>
    Deterministic 5-fold indices generated and saved to JSON.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T1.6: Classical SVM Baseline (Linear & RBF)</name>
  <files>
    src/models/classical/svm.py
    tests/test_classical_models.py
  </files>
  <action>
    Implement classical Support Vector Machine baseline.
    Steps:
    1. Create ClassicalSVM wrapper conforming to BaseModel interface.
    2. Support kernel='linear' and kernel='rbf' with probability=True.
    3. Implement fit, predict, and predict_proba methods.
  </action>
  <verify>
    pytest tests/test_classical_models.py -k "test_svm"
  </verify>
  <done>
    Linear & RBF SVM achieve >92% accuracy on WDBC validation folds.
  </done>
</task>

<task type="auto" effort="high">
  <name>T1.8: Classical PyTorch Multi-Layer Perceptron (MLP)</name>
  <files>
    src/models/classical/mlp.py
    tests/test_classical_models.py
  </files>
  <action>
    Build 3-layer feedforward PyTorch neural network matching input dimensions.
    Architecture:
    Linear(in_dim, 32) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(32, 16) -> ReLU -> Linear(16, 2).
    Include Adam optimizer, early stopping on validation loss, and scikit-learn API wrapper.
  </action>
  <verify>
    pytest tests/test_classical_models.py -k "test_mlp"
  </verify>
  <done>
    PyTorch MLP trains for 30 epochs with monotonic loss reduction.
  </done>
</task>

---

## Phase 2: Dimensionality Reduction for Qubit Constraints

<task type="auto" effort="medium">
  <name>T2.1: PCA Dimensionality Reduction Module</name>
  <files>
    src/features/pca_reducer.py
    tests/test_features.py
  </files>
  <action>
    Implement PCA reducer that compresses feature spaces to $k \in [4, 8]$ dimensions.
    Steps:
    1. Wrap sklearn.decomposition.PCA.
    2. Provide method get_explained_variance_ratio().
    3. Scale output principal components to $[-\pi, \pi]$ using MinMaxScaler to match Member B's AngleEmbedding and ZZFeatureMap circuits.
  </action>
  <verify>
    python -c "from src.features.pca_reducer import PCAReducer; import numpy as np; r = PCAReducer(n_components=4); out = r.fit_transform(np.random.randn(20, 30)); assert out.shape == (20, 4); assert np.all(out >= -np.pi) and np.all(out <= np.pi); print('PCA reducer verified.')"
  </verify>
  <done>
    Inputs successfully compressed to target qubit count and scaled to $[-\pi, \pi]$.
  </done>
</task>

<task type="auto" effort="low">
  <name>T2.2: Mutual Information Feature Selector</name>
  <files>
    src/features/mi_selector.py
    tests/test_features.py
  </files>
  <action>
    Implement Mutual Information / SelectKBest feature selector.
    Steps:
    1. Wrap sklearn.feature_selection.mutual_info_classif.
    2. Rank top $k$ biomedical features and preserve original column names for clinical interpretability.
    3. Scale selected features to $[0, 2\pi]$ for quantum rotation gates.
  </action>
  <verify>
    python -c "from src.features.mi_selector import MISelector; import numpy as np; s = MISelector(k=4); print('MI selector ready.')"
  </verify>
  <done>
    Selects top $k$ most informative features with clinical column names preserved.
  </done>
</task>

---

## Phase 4 & 5: Platform Usability, Edge-Cases & IEEE Report

<task type="auto" effort="medium">
  <name>T4.3: Feature Engineering UI Controller</name>
  <files>
    app/pages/2_Features.py
  </files>
  <action>
    Connect PCAReducer and MISelector to Streamlit UI page.
    Provide an interactive slider for number of qubits $k \in [2, 8]$ and render a Plotly 2D/3D scatter plot of the projected patient clusters.
  </action>
  <verify>
    Verify Streamlit page 2 renders without errors when slider is moved.
  </verify>
  <done>
    Real-time interactive projection rendered inside browser.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T5.1: Ingestion Sanitization & Single-Sample Inference Hardening</name>
  <files>
    src/data/validator.py
    tests/test_data_validation.py
  </files>
  <action>
    Build input validation guardrails: detect missing columns, invalid data types, and out-of-range sensor inputs. Support single patient record inference dictionary.
  </action>
  <verify>
    pytest tests/test_data_validation.py
  </verify>
  <done>
    Gracefully catches and rejects malformed inputs with clean user errors.
  </done>
</task>

<task type="auto" effort="high">
  <name>T5.4: IEEE Technical Methodology Paper Authoring</name>
  <files>
    docs/HQML_Disease_Detection_Paper.pdf
    docs/paper_source.md
  </files>
  <action>
    Author 8-page IEEE conference-style report:
    1. Abstract & Introduction (Biomedical high-dimensional challenges).
    2. Hybrid Quantum-Classical Mathematical Formulation (Hilbert space mapping, ZZ-kernel, StronglyEntangling ansatz).
    3. Experimental Setup & Benchmarking (5-fold CV on WDBC and Heart Disease).
    4. Results, Confusion Matrices, and XAI Clinical Interpretability.
  </action>
  <verify>
    All team members review and sign off on technical rigor and formatting.
  </verify>
  <done>
    Completed, publication-grade technical report ready for submission.
  </done>
</task>
