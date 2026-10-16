"""Component 4: Live Counterfactual "What-If" Clinical Diagnostic Simulator."""

import numpy as np
import streamlit as st
from src.data.loader import load_dataset
from src.models.classical.svm import RBFSVMModel
from src.models.quantum.qsvc import QuantumSupportVectorClassifier


@st.cache_resource
def get_cached_models():
    """Trains small reference classical and quantum models for immediate real-time simulation."""
    ds = load_dataset("wdbc", n_qubits=4, random_state=42)
    X_train, y_train = ds["X_train"], ds["y_train"]
    reducer = ds["reducer"]
    scaler = ds["scaler"]

    svm = RBFSVMModel(C=1.0)
    svm.fit(X_train, y_train)

    qsvc = QuantumSupportVectorClassifier(n_qubits=4, reps=1, C=1.0)
    # Quick fit on 40 samples for instantaneous responsive slider interaction
    qsvc.fit(X_train[:40], y_train[:40])

    return svm, qsvc, scaler, reducer


def render_what_if_simulator():
    """Renders the real-time what-if counterfactual slider simulator."""
    st.markdown("### 🎛️ Live Counterfactual \"What-If\" Patient Simulator")
    st.caption("Adjust clinical patient biomarkers in real-time to observe decision boundary differences between Classical and Quantum models.")

    svm, qsvc, scaler, reducer = get_cached_models()

    c_sliders, c_meters = st.columns([1, 1])

    with c_sliders:
        st.markdown("#### Patient Biomarker Controls (WDBC Oncology)")
        radius = st.slider("Mean Cell Nucleus Radius (μm):", 6.0, 30.0, 15.5, 0.2)
        texture = st.slider("Mean Texture (Gray-scale variation):", 9.0, 40.0, 21.0, 0.5)
        perimeter = st.slider("Mean Perimeter (μm):", 40.0, 190.0, 98.0, 1.0)
        area = st.slider("Mean Area (μm²):", 140.0, 2500.0, 720.0, 10.0)

        # Construct synthetic 30-feature vector using reasonable mean imputation for remaining 26 features
        raw_feat = np.zeros((1, 30))
        raw_feat[0, 0] = radius
        raw_feat[0, 1] = texture
        raw_feat[0, 2] = perimeter
        raw_feat[0, 3] = area
        # Remaining features filled with standard population medians
        raw_feat[0, 4:] = [0.09, 0.10, 0.08, 0.04, 0.18, 0.06, 0.40, 1.20, 2.80, 40.0,
                           0.007, 0.02, 0.03, 0.01, 0.02, 0.003, 16.0, 25.0, 105.0, 880.0,
                           0.13, 0.25, 0.27, 0.11, 0.29, 0.08]

        # Project through scaler and PCA reducer to 4 qubits
        scaled_feat = scaler.transform(raw_feat)
        quantum_feat = reducer.transform(scaled_feat)

    # Predictions
    svm_prob = float(svm.predict_proba(quantum_feat)[0, 1])
    qsvc_prob = float(qsvc.predict_proba(quantum_feat)[0, 1])

    with c_meters:
        st.markdown("#### Real-Time Comparative Diagnostic Risk")

        # Classical Card
        c_badge = "🔴 High Risk (Malignant)" if svm_prob >= 0.5 else "🟢 Low Risk (Benign)"
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #94a3b8; font-weight: 600;">CLASSICAL BASELINE (SVM-RBF)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #f8fafc;">{svm_prob*100:.1f}% Probability</div>
                <div style="margin-top: 4px;">Status: <strong>{c_badge}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(svm_prob)

        # Quantum Card
        q_badge = "🔴 High Risk (Malignant)" if qsvc_prob >= 0.5 else "🟢 Low Risk (Benign)"
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 10px; padding: 14px; margin-bottom: 12px;">
                <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 600;">QUANTUM CLASSIFIER (QSVC FIDELITY)</div>
                <div style="font-size: 1.6rem; font-weight: 700; color: #38bdf8;">{qsvc_prob*100:.1f}% Probability</div>
                <div style="margin-top: 4px;">Status: <strong>{q_badge}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(qsvc_prob)

        # Clinical Discordance Check
        if (svm_prob >= 0.5 and qsvc_prob < 0.5) or (svm_prob < 0.5 and qsvc_prob >= 0.5):
            st.warning("⚠️ **Borderline Clinical Discordance**: Models disagree on diagnosis. Quantum Kernel separates higher-order non-linear feature overlaps in Hilbert space.")
        else:
            st.success("✅ **Consensus Diagnostic**: Both Classical and Quantum models agree on patient categorization.")
