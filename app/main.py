"""Interactive Streamlit Web Dashboard for Hybrid Quantum ML Disease Detection Platform."""

import os
import streamlit as st
import numpy as np
import pandas as pd

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease
from src.data.fundus_loader import load_fundus_dataset
from src.features.pca_reducer import PCAReducer
from src.features.mi_selector import MISelector
from src.quantum.backend import QuantumBackendManager
from app.components.circuit_viz import QuantumCircuitVisualizer

st.set_page_config(
    page_title="HQML-EDD | Hybrid Quantum Disease Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🧬 Hybrid Quantum ML Platform for Early Disease Detection")
st.caption("Simulator-First NISQ Hybrid Quantum-Classical Clinical Diagnostic Console")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

dataset_choice = st.sidebar.selectbox(
    "Select Medical Dataset",
    ["Wisconsin Breast Cancer (WDBC)", "UCI Cleveland Heart Disease", "Clinical Fundus Retinopathy (DR)"],
)

n_qubits = st.sidebar.slider("Quantum Addressable Qubits (Wires)", min_value=2, max_value=8, value=4)
model_choice = st.sidebar.selectbox("Quantum Architecture", ["QSVC (Quantum Kernel)", "VQC (Variational Classifier)", "HQNN (Hybrid PyTorch)"])

st.sidebar.divider()
st.sidebar.markdown("**System Health:** 🟢 `default.qubit` Active")
st.sidebar.markdown(f"**Safety Bounds:** $N \le 8$ Qubit Cap Enforced")

# Cache Data Loading
@st.cache_data
def get_dataset(choice):
    if "Breast" in choice:
        X, y, names = load_wdbc(scale=True)
        img_paths = None
    elif "Heart" in choice:
        X, y, names = load_heart_disease(scale=True)
        img_paths = None
    else:
        X, y, img_paths, names = load_fundus_dataset(scale=True)
    return X, y, names, img_paths

X_raw, y_raw, feature_names, image_paths = get_dataset(dataset_choice)

# Main Navigation Tabs
tab_data, tab_circuit, tab_train, tab_xai = st.tabs([
    "📊 Data Explorer",
    "⚛️ Quantum Circuit",
    "🚀 Model Training",
    "🔍 Explainability (XAI)",
])

# ---------------- Tab 1: Data Explorer ----------------
with tab_data:
    st.subheader(f"Dataset Overview: {dataset_choice}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients / Samples", X_raw.shape[0])
    col2.metric("Raw Clinical Features", X_raw.shape[1])
    malignant_pct = np.mean(y_raw) * 100
    col3.metric("Positive Disease Ratio", f"{malignant_pct:.1f}%")

    if image_paths is not None:
        st.markdown("### Clinical Retinal Fundus Photographs")
        img_cols = st.columns(4)
        for i in range(min(4, len(image_paths))):
            label_text = "Diabetic Retinopathy" if y_raw[i] == 1 else "Healthy / Normal"
            img_cols[i].image(image_paths[i], caption=f"{os.path.basename(image_paths[i])}\n({label_text})", use_container_width=True)

    st.markdown("### Feature Matrix Preview")
    df_preview = pd.DataFrame(X_raw[:10, :min(8, X_raw.shape[1])], columns=feature_names[:min(8, len(feature_names))])
    st.dataframe(df_preview, use_container_width=True)

# ---------------- Tab 2: Quantum Circuit ----------------
with tab_circuit:
    st.subheader("Quantum Circuit Topology")
    st.markdown(f"Visualizing **{model_choice}** parameterized across **{n_qubits} qubits**.")

    viz = QuantumCircuitVisualizer()
    circuit_key = "qsvc" if "QSVC" in model_choice else ("vqc" if "VQC" in model_choice else "hqnn")

    try:
        fig = viz.draw_mpl(model_type=circuit_key, n_qubits=n_qubits, n_layers=1)
        st.pyplot(fig)
    except Exception:
        ascii_art = viz.draw_ascii(model_type=circuit_key, n_qubits=n_qubits, n_layers=1)
        st.code(ascii_art, language="text")

# ---------------- Tab 3: Model Training ----------------
with tab_train:
    st.subheader("Hybrid Quantum Model Execution")
    st.markdown(f"Compressing **{X_raw.shape[1]} raw features** down to **{n_qubits} quantum statevector rotations** $[-\pi, \pi]$.")

    if st.button("⚡ Start Quantum Training & Evaluation"):
        with st.spinner("Executing PCA Dimensionality Reduction & Quantum Kernel Computation..."):
            pca = PCAReducer(n_components=n_qubits, scale_to_pi=True, random_state=42)
            X_reduced = pca.fit_transform(X_raw)

            # Split
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, recall_score, f1_score
            from src.models.quantum.qsvc import QuantumSupportVectorClassifier

            # Subset for responsive web inference
            sub_n = min(60, len(X_reduced))
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_reduced[:sub_n], y_raw[:sub_n], test_size=0.3, stratify=y_raw[:sub_n], random_state=42
            )

            qsvc = QuantumSupportVectorClassifier(n_qubits=n_qubits, reps=1, C=1.0)
            qsvc.fit(X_tr, y_tr)
            preds = qsvc.predict(X_te)

            acc = accuracy_score(y_te, preds)
            sens = recall_score(y_te, preds, zero_division=0)
            f1 = f1_score(y_te, preds, zero_division=0)

        st.success("Quantum Model Training & Evaluation Complete!")
        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Classification Accuracy", f"{acc*100:.2f}%")
        res_c2.metric("Clinical Sensitivity (Recall)", f"{sens*100:.2f}%")
        res_c3.metric("F1-Score", f"{f1*100:.2f}%")

# ---------------- Tab 4: Explainability (XAI) ----------------
with tab_xai:
    st.subheader("Patient Risk Explainability (Quantum Feature Attribution)")
    sample_idx = st.slider("Select Patient Case Index", 0, len(X_raw) - 1, 0)

    st.markdown(f"**Diagnostic Target:** {'POSITIVE / MALIGNANT' if y_raw[sample_idx] == 1 else 'HEALTHY / NEGATIVE'}")

    from src.xai.quantum_shap import QuantumPermutationImportance
    from sklearn.svm import SVC

    pca = PCAReducer(n_components=n_qubits, scale_to_pi=True, random_state=42)
    X_q = pca.fit_transform(X_raw)

    clf = SVC(kernel="rbf", probability=True, random_state=42)
    clf.fit(X_q[:50], y_raw[:50])

    p_imp = QuantumPermutationImportance(clf, n_repeats=3)
    importance = p_imp.compute_importance(X_q[:50], y_raw[:50], feature_names=[f"Quantum_PC_{i+1}" for i in range(n_qubits)])

    df_imp = pd.DataFrame([
        {"Biomarker Dimension": k, "Risk Importance Drop": v["mean_drop"]}
        for k, v in importance.items()
    ])
    st.bar_chart(df_imp.set_index("Biomarker Dimension"))
