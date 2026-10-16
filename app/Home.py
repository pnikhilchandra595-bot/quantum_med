"""HQML Diagnostic Intelligence Dashboard - Main Landing Page.

Hybrid Quantum Machine Learning Clinical Evaluation & Explainability Platform.
"""

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
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555555;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafd;
        border: 1px solid #e1e8f0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #e3f2fd;
        color: #0d47a1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🧬 HQML: Hybrid Quantum-Classical Clinical AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Evaluating Quantum Inductive Bias and Explainability in Oncology Diagnostics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="metric-card">
        <span class="badge">Cancer Detection</span>
        <h3 style="margin: 8px 0 4px 0; color: #0d47a1;">97.02%</h3>
        <p style="color: #666; font-size: 0.85rem; margin: 0;">Quantum QSVC Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <span class="badge" style="background:#e8f5e9; color:#1b5e20;">Sensitivity (Recall)</span>
        <h3 style="margin: 8px 0 4px 0; color: #2e7d32;">95.77%</h3>
        <p style="color: #666; font-size: 0.85rem; margin: 0;">Minimized False Negatives</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <span class="badge" style="background:#f3e5f5; color:#4a148c;">Statistical Proof</span>
        <h3 style="margin: 8px 0 4px 0; color: #7b1fa2;">p = 0.035</h3>
        <p style="color: #666; font-size: 0.85rem; margin: 0;">Quantum Advantage (p &lt; 0.05)</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <span class="badge" style="background:#fff3e0; color:#e65100;">XAI Concordance</span>
        <h3 style="margin: 8px 0 4px 0; color: #e65100;">&rho; = 0.943</h3>
        <p style="color: #666; font-size: 0.85rem; margin: 0;">SHAP Clinical Driver Alignment</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Platform Architecture & Modules")

st.markdown("""
- **Phase 1 & 2**: Standardized Clinical Evaluation Engine & Tree Baselines (Random Forest & XGBoost).
- **Phase 3**: Master 5-Fold Cross-Validation Benchmark Suite across 7 models & Hypothesis Testing for Quantum Advantage.
- **Phase 4**: Interactive Patient-Level Explainability (SHAP Waterfall Attributions & Risk Gauges).
- **Phase 5**: 15-Slide Competition Pitch Deck for Jury Evaluation.

👈 **Navigate to the '5_Explainability' tab in the left sidebar to inspect interactive patient risk attributions.**
""")
