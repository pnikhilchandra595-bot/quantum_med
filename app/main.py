"""
QuantumHealth: Production-Grade Hybrid Quantum-Classical ML Clinical Platform.
Main Entry Point & Executive Dashboard.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

st.set_page_config(
    page_title="QuantumHealth | Hybrid QML Platform",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS Styles
css_path = Path(__file__).resolve().parent / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/doc/_static/pennylane_logo.png",
        width=180,
    )
    st.markdown("### **QuantumHealth v1.0**")
    st.caption("Hybrid Quantum-Classical Diagnostic System")
    st.divider()
    st.markdown("**Platform Navigation**")
    st.markdown("""
    - **Overview**: Executive Platform Telemetry
    - **1 Data Explorer**: Profiling & Correlation
    - **2 Feature Studio**: PCA & Quantum Encoding
    - **3 Model Studio**: Real-Time QML Training
    - **4 Benchmark**: Comparative Leaderboard
    - **5 Explainability**: KernelSHAP Attribution
    """)
    st.divider()
    st.caption("Engine: PennyLane 0.34+ | PyTorch 2.0+ | Scikit-learn")

# Hero Section
st.markdown(
    """
<div class="hero-container">
    <div class="hero-title">QuantumHealth Diagnostics</div>
    <div class="hero-subtitle">
        A clinical-grade hybrid quantum-classical machine learning framework bridging Hilbert-space
        expressivity with classical deep learning for breast cancer oncology and cardiovascular risk diagnosis.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# Top Telemetry Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    <div class="quantum-card quantum-card-cyan">
        <div class="metric-label">Benchmark Models</div>
        <div class="metric-value">7 Models</div>
        <p style="margin-top: 8px; font-size: 0.85rem; color: #9ca3af;">
            <span class="badge badge-quantum">3 Quantum</span>
            <span class="badge badge-classical">4 Classical</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
    <div class="quantum-card quantum-card-teal">
        <div class="metric-label">Clinical Cohorts</div>
        <div class="metric-value">872 Patients</div>
        <p style="margin-top: 8px; font-size: 0.85rem; color: #9ca3af;">
            <span class="badge badge-clinical">WDBC (569)</span>
            <span class="badge badge-clinical">Heart (303)</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
    <div class="quantum-card quantum-card-purple">
        <div class="metric-label">NISQ Hilbert Dimension</div>
        <div class="metric-value">2⁴ = 16 Dim</div>
        <p style="margin-top: 8px; font-size: 0.85rem; color: #9ca3af;">
            <span class="badge badge-quantum">4-Qubit Circuits</span>
            <span class="badge badge-quantum">Entangled</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
    <div class="quantum-card quantum-card-cyan">
        <div class="metric-label">System Readiness</div>
        <div class="metric-value">Ready (v1.0)</div>
        <p style="margin-top: 8px; font-size: 0.85rem; color: #9ca3af;">
            <span class="badge badge-clinical">Docker Ready</span>
            <span class="badge badge-clinical">CI Passing</span>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.write("")

# 3-Tier Architecture Tabs
tab1, tab2, tab3 = st.tabs(
    [
        "📐 3-Tier System Architecture",
        "🔬 Quantum Diagnostic Models",
        "🏥 Clinical Workflow & Ethics",
    ]
)

with tab1:
    st.subheader("Architectural Topology")
    arch_img_path = root_dir / "docs" / "assets" / "system_diagram.png"
    if arch_img_path.exists():
        st.image(
            str(arch_img_path),
            caption="End-to-End 3-Tier Hybrid Quantum-Classical Platform Topology",
            use_container_width=True,
        )

    st.markdown("""
    1. **Tier 1 — Data & Preprocessing Tier**:
       - Standardized ingestion of high-dimensional clinical datasets (WDBC 30 features, UCI Heart 13 features).
       - Outlier treatment, robust scaling, and dimension adaptation (PCA / Mutual Information) tailored for NISQ hardware limits.
    2. **Tier 2 — Quantum-Classical Engine Tier**:
       - Unified `ModelFactory` orchestrating 7 architectures:
         - **Classical Baselines**: Support Vector Classifier (RBF), Random Forest, XGBoost, Multi-Layer Perceptron.
         - **Quantum Classifiers**: QSVC (Quantum Kernel Estimation), VQC (Strongly Entangled Variational Circuit), HQNN (PyTorch `TorchLayer`).
       - Quantum device management via `QuantumBackendManager` supporting `default.qubit` state-vector simulator and physical noise injection.
    3. **Tier 3 — Clinical Delivery & Explainability Tier**:
       - 6-Page interactive Streamlit diagnostic workstation.
       - Model-agnostic KernelSHAP feature attribution providing transparent explanations to physicians.
       - Headless CLI orchestrator (`cli.py`) for automated training and batch inference.
    """)

with tab2:
    st.subheader("Model Catalogue")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚡ Quantum Classifiers")
        st.markdown(r"""
        - **QSVC (Quantum Support Vector Classifier)**: Maps input data into high-dimensional quantum Hilbert state-space via ZZ-entangling feature maps. Computes state fidelity kernels $|\langle \Phi(x) | \Phi(x') \rangle|^2$ for optimal margin separation.
        - **VQC (Variational Quantum Classifier)**: Parameterized quantum circuit consisting of $R_y$ angle embedding followed by layers of universal rotation gates and circular CNOT entanglement. Trained end-to-end using the parameter-shift gradient rule.
        - **HQNN (Hybrid Quantum Neural Network)**: Deep hybrid network coupling PyTorch linear compression layers with a PennyLane parameterized quantum layer, measuring multi-wire Pauli-Z expectation values.
        """)
    with c2:
        st.markdown("#### 🌲 Classical Baselines")
        st.markdown("""
        - **SVM (RBF Kernel)**: Non-linear soft-margin benchmark evaluating dual-formulation boundaries in reproducing kernel Hilbert spaces.
        - **Random Forest (RF)**: Ensemble of 100 de-correlated decision trees measuring Gini-impurity feature importances.
        - **XGBoost (XGB)**: Scalable extreme gradient boosted decision trees optimizing pseudo-residuals.
        - **MLP (Multi-Layer Perceptron)**: Fully connected classical artificial neural network with ReLU activations.
        """)

with tab3:
    st.subheader("Clinical Diagnostic Pipeline")
    st.markdown("""
    | Stage | Operation | Clinical Target |
    |---|---|---|
    | **1. Biopsy / Vitals** | Patient FNA image / lab blood work | Raw biomarker acquisition |
    | **2. Quantum Reduction** | PCA to 4-8 orthogonal eigen-features | Maintain >85% diagnostic variance |
    | **3. Hilbert Mapping** | Rotational angle embedding ($R_y$) | Non-linear feature separation in $2^N$ space |
    | **4. Inference** | Variational expectation measurement | Malignant / Cardiac Risk Probability |
    | **5. XAI Audit** | KernelSHAP attribution | Clinical biomarker transparency for oncologists |
    """)

st.divider()
st.info(
    "💡 **Getting Started:** Select **1 Data Explorer** in the sidebar to inspect patient datasets, or go directly to **3 Model Studio** to train a 4-qubit quantum classifier in real time!"
)
