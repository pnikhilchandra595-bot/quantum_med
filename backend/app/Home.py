"""HQML Clinical Diagnostic Intelligence Platform - Main Research Portal.

Streamlit App: app/Home.py
Collaborative Platform: Member A (Lead/Pipelines), Member B (Quantum), Member C (Evaluation/XAI)
Aesthetic: PonyTail Clinical Dark/Light Glassmorphism
"""

import sys
from pathlib import Path

# Ensure backend directory and repository root are on sys.path
root_dir = Path(__file__).resolve().parent.parent
for p in [root_dir, root_dir.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import streamlit as st

st.set_page_config(
    page_title="HQML Clinical Evaluation Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .portal-hero {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .nav-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .nav-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-right: 6px;
    }
    .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
    .badge-orange { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero Header
st.markdown(
    """
    <div class="portal-hero">
        <h1 style="color: #f8fafc; margin-bottom: 6px;">🧬 Hybrid Quantum-Classical Machine Learning (HQML)</h1>
        <h3 style="color: #38bdf8; font-weight: 500; margin-top: 0;">Biomedical High-Dimensional Disease Detection & Clinical AI Platform</h3>
        <p style="color: #94a3b8; font-size: 1.0rem; margin-bottom: 0;">
            A collaborative research platform integrating leak-free data pipelines, 
            quantum register encoding, classical ML baselines, and cryptographic cross-validation integrity audits.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Clinical KPI Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        """
        <div class="metric-card">
            <span class="badge-pill badge-blue">WDBC Baseline</span>
            <h3 style="margin: 8px 0 4px 0; color: #38bdf8;">97.89%</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Tuned Classical SVM (RBF)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="metric-card">
            <span class="badge-pill badge-green">Sensitivity</span>
            <h3 style="margin: 8px 0 4px 0; color: #4ade80;">96.70%</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Minimized False Negatives</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """
        <div class="metric-card">
            <span class="badge-pill badge-purple">Fold Audit</span>
            <h3 style="margin: 8px 0 4px 0; color: #c084fc;">0.00%</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">SHA-256 Zero Data Leakage</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        """
        <div class="metric-card">
            <span class="badge-pill badge-orange">Heart Disease</span>
            <h3 style="margin: 8px 0 4px 0; color: #fb923c;">85.48%</h3>
            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">Cleveland Cardiac Baseline</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# Team Lanes
st.markdown("### 👥 Multidisciplinary Research Team Lanes")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="nav-card">
            <span class="badge-pill badge-blue">Member A — Lead</span>
            <h4 style="color: #f8fafc; margin-top: 10px;">Data Pipelines & Classical ML</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                <li>Leak-free WDBC & Heart loaders (D2)</li>
                <li>PCA & Mutual Info Qubit Encodings (D3)</li>
                <li>Tuned SVM, MLP, RF, XGBoost baselines (D5)</li>
                <li>Cryptographic Fold Integrity Audit (T3.3)</li>
                <li>IEEE Methodology Report Authoring (D10/D12)</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="nav-card">
            <span class="badge-pill badge-purple">Member B</span>
            <h4 style="color: #f8fafc; margin-top: 10px;">Quantum Circuit Architect</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                <li>AngleEmbedding & ZZFeatureMap circuits</li>
                <li>Variational Quantum Classifier (VQC)</li>
                <li>Quantum Kernel Support Vector Machine (QSVM)</li>
                <li>StronglyEntangling Layers on NISQ registers</li>
                <li>Identical 5-Fold split consumption</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="nav-card">
            <span class="badge-pill badge-green">Member C</span>
            <h4 style="color: #f8fafc; margin-top: 10px;">Evaluation & Explainable AI</h4>
            <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px;">
                <li>Out-of-fold confusion matrix evaluation</li>
                <li>Clinical sensitivity & specificity audits</li>
                <li>SHAP & LIME biomarker importance</li>
                <li>Comparative classical vs quantum statistical tests</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>", unsafe_allow_html=True)

# Navigation Cards
st.markdown("### 🧭 Interactive Application Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown(
        """
        <div class="nav-card">
            <h4 style="color: #38bdf8;">🔬 1. Data Exploration</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Browse WDBC and Heart cohorts, class distributions, correlation heatmaps, 
                and test single-patient diagnostic validation with real-time sensor guardrails.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav2:
    st.markdown(
        """
        <div class="nav-card">
            <h4 style="color: #38bdf8;">🌌 2. Quantum Feature Space</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Interactive qubit slider (k ∈ [2, 8]), 2D/3D Plotly manifold projections, 
                explained variance ratios, and [-π, π] rotation gate boundary checks.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav3:
    st.markdown(
        """
        <div class="nav-card">
            <h4 style="color: #38bdf8;">🛡️ 3. Baselines & Audit</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Examine SHA-256 signed fold integrity certificates, 5-fold cross-validation 
                leaderboards, latency benchmarks, and interactive out-of-fold ROC curves.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav4:
    st.markdown(
        """
        <div class="nav-card">
            <h4 style="color: #38bdf8;">🔍 5. Explainability (XAI)</h4>
            <p style="color: #94a3b8; font-size: 0.85rem;">
                Interactive patient-level SHAP waterfall attributions, feature radar plots, 
                and clinical biomarker alignment metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

# Cohort Details
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.subheader("📁 Wisconsin Diagnostic Breast Cancer (WDBC)")
    st.markdown(
        """
        - **Total Cohort Size:** 569 patient biopsies (357 Benign, 212 Malignant)
        - **Dimensionality:** 30 continuous nuclear characteristics (radius, texture, concavity)
        - **Partitioning:** Stratified 5-Fold Cross-Validation (`configs/folds_wdbc.json`)
        - **Target Baseline:** Classical RBF SVM achieving **97.89% Accuracy** and **0.9945 ROC-AUC**
        """
    )
with col_d2:
    st.subheader("📁 UCI Heart Disease (Cleveland)")
    st.markdown(
        """
        - **Total Cohort Size:** 303 cardiac examinations (164 Absence, 139 Disease Present)
        - **Dimensionality:** 13 clinical, physiological & angiographic variables
        - **Partitioning:** Stratified 5-Fold Cross-Validation (`configs/folds_heart.json`)
        - **Target Baseline:** Classical RBF SVM achieving **85.48% Accuracy** and **0.9083 ROC-AUC**
        """
    )
