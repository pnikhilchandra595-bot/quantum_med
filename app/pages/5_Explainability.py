"""T4.6: Clinical Explainability UI Dashboard.

Interactive Patient Risk Explanation in Streamlit:
1. Select sample patient from clinical test set.
2. Display prediction confidence gauge (e.g. 87% Malignant).
3. Render interactive Plotly SHAP waterfall plot showing positive/negative risk contributors.
"""

import sys
from pathlib import Path

# Ensure workspace root in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

from src.data.loader import load_dataset
from src.data.feature_reduction import PCAReducer
from src.models.classical.trees import RandomForestModel, XGBoostModel
from src.models.classical.svm import RBFSVMModel, LinearSVMModel
from src.models.classical.mlp import MLPModel
from src.models.quantum.quantum_models import QSVCModel, VQCModel, HQNNModel
from src.xai.classical_shap import ClassicalSHAPExplainer
from src.xai.quantum_shap import QuantumKernelSHAP

st.set_page_config(
    page_title="HQML Clinical Explainability",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .title-banner {
        padding: 12px 20px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 25px;
    }
    .stMetric {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px;
    }
    .risk-high {
        color: #d32f2f;
        font-weight: 700;
    }
    .risk-low {
        color: #2e7d32;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-banner">
        <h2 style="margin:0; color:white;">🔬 Clinical Explainability (XAI) Dashboard</h2>
        <p style="margin:4px 0 0 0; opacity:0.9;">
            Patient-level feature attribution via Classical & Quantum Kernel SHAP (Shapley Values)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_and_prepare_models():
    """Load dataset with zero data leakage, train all models on 6 PCs, and prepare explainers."""
    X_raw, y, feature_names, _ = load_dataset("wdbc", scaled=False)

    component_names = [f"PC{i+1} (Qubit {i})" for i in range(6)]

    # Out-of-fold stratified split
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    # ZERO DATA LEAKAGE: Fit PCA strictly on training partition
    reducer = PCAReducer(n_components=6, random_state=42)
    X_train = reducer.fit_transform(X_tr_raw)
    X_test = reducer.transform(X_te_raw)

    # Train all models
    rf = RandomForestModel(n_estimators=100, max_depth=5, random_state=42).fit(X_train, y_train)
    xgb = XGBoostModel(n_estimators=100, max_depth=4, random_state=42).fit(X_train, y_train)
    svm_rbf = RBFSVMModel(random_state=42).fit(X_train, y_train)
    svm_lin = LinearSVMModel(random_state=42).fit(X_train, y_train)
    mlp = MLPModel(hidden_layer_sizes=(32, 16), max_iter=600, random_state=42).fit(X_train, y_train)
    qsvc = QSVCModel(C=2.0, random_state=42).fit(X_train, y_train)
    vqc = VQCModel(random_state=42).fit(X_train, y_train)
    hqnn = HQNNModel(random_state=42).fit(X_train, y_train)

    # Prepare explainers with background dataset
    bg = X_train[:30]
    explainers = {
        "QSVC (Quantum Kernel)": QuantumKernelSHAP(qsvc, bg, component_names),
        "VQC (Variational Quantum)": QuantumKernelSHAP(vqc, bg, component_names),
        "HQNN (Hybrid Quantum NN)": QuantumKernelSHAP(hqnn, bg, component_names),
        "Random Forest (Classical)": ClassicalSHAPExplainer(rf, bg, component_names),
        "XGBoost (Classical)": ClassicalSHAPExplainer(xgb, bg, component_names),
        "RBF SVM (Classical)": ClassicalSHAPExplainer(svm_rbf, bg, component_names),
        "Linear SVM (Classical)": ClassicalSHAPExplainer(svm_lin, bg, component_names),
        "MLP (Classical)": ClassicalSHAPExplainer(mlp, bg, component_names),
    }

    models = {
        "QSVC (Quantum Kernel)": qsvc,
        "VQC (Variational Quantum)": vqc,
        "HQNN (Hybrid Quantum NN)": hqnn,
        "Random Forest (Classical)": rf,
        "XGBoost (Classical)": xgb,
        "RBF SVM (Classical)": svm_rbf,
        "Linear SVM (Classical)": svm_lin,
        "MLP (Classical)": mlp,
    }

    return X_train, X_test, y_train, y_test, component_names, models, explainers


X_train, X_test, y_train, y_test, component_names, models, explainers = load_and_prepare_models()

# Sidebar: Controls
st.sidebar.header("🎯 Patient & Model Controls")

# Patient selection
patient_indices = list(range(len(X_test)))
patient_labels = [
    f"Patient #{i+1:03d} (Ground Truth: {'Malignant' if y_test[i]==1 else 'Benign'})"
    for i in patient_indices
]
selected_idx = st.sidebar.selectbox("Select Test Patient Sample:", patient_indices, format_func=lambda i: patient_labels[i])

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model:",
    list(models.keys()),
    index=0,
)

# Patient Data
patient_features = X_test[selected_idx]
patient_true_label = y_test[selected_idx]
model = models[selected_model_name]
explainer = explainers[selected_model_name]

# Inference
prob_dist = model.predict_proba(patient_features.reshape(1, -1))[0]
prob_malignant = float(prob_dist[1])
pred_label = 1 if prob_malignant >= 0.5 else 0

# SHAP Attribution
with st.spinner("Calculating Shapley risk attributions..."):
    explanation = explainer.explain(patient_features, nsamples=60)

shap_values = explanation.values
base_value = explanation.base_value

# Top row: Patient Info & Confidence Gauge
col_info, col_gauge = st.columns([1, 1.2])

with col_info:
    st.subheader("Patient Clinical Profile")
    status_color = "#d32f2f" if pred_label == 1 else "#2e7d32"
    status_text = "MALIGNANT (High Risk)" if pred_label == 1 else "BENIGN (Low Risk)"
    ground_truth_text = "Malignant" if patient_true_label == 1 else "Benign"

    st.markdown(
        f"""
        <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:18px; margin-bottom:15px;">
            <p style="margin:0 0 6px 0; color:#666; font-size:0.9rem;">Model Classification Verdict</p>
            <h2 style="margin:0; color:{status_color};">{status_text}</h2>
            <hr style="margin:12px 0;">
            <p style="margin:4px 0;"><b>Patient ID:</b> #{selected_idx+1:03d}</p>
            <p style="margin:4px 0;"><b>Ground Truth Biopsy:</b> <span style="font-weight:600;">{ground_truth_text}</span></p>
            <p style="margin:4px 0;"><b>Model Evaluated:</b> {selected_model_name}</p>
            <p style="margin:4px 0;"><b>Prior Base Risk E[f(x)]:</b> {base_value*100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gauge:
    st.subheader("Diagnostic Confidence Gauge")
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_malignant * 100.0,
            delta={"reference": 50.0, "increasing": {"color": "#d32f2f"}, "decreasing": {"color": "#2e7d32"}},
            number={"suffix": "%", "font": {"size": 36, "weight": "bold"}},
            title={"text": "Malignant Probability Score", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#333333"},
                "bar": {"color": "#9c27b0" if "Quantum" in selected_model_name else "#1976d2", "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#cccccc",
                "steps": [
                    {"range": [0, 40], "color": "rgba(46, 125, 50, 0.2)"},
                    {"range": [40, 60], "color": "rgba(255, 179, 0, 0.25)"},
                    {"range": [60, 100], "color": "rgba(211, 47, 47, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#d32f2f", "width": 3},
                    "thickness": 0.85,
                    "value": 50.0,
                },
            },
        )
    )
    fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
    st.plotly_chart(fig_gauge, width="stretch")

st.markdown("---")
st.subheader("Interactive Plotly SHAP Waterfall Plot")
st.caption(
    "Visualizing individual feature contributions shifting patient prediction from prior baseline E[f(x)] to final malignant risk score."
)

# Plotly Waterfall Plot
# Sort features by absolute attribution magnitude
order = np.argsort(np.abs(shap_values))[::-1]
sorted_names = [component_names[i] for i in order]
sorted_values = [shap_values[i] for i in order]
sorted_patient_vals = [patient_features[i] for i in order]

waterfall_x = ["Baseline E[f(x)]"] + [f"{name} = {val:.2f}" for name, val in zip(sorted_names, sorted_patient_vals)] + ["Patient Output"]
waterfall_measure = ["absolute"] + ["relative"] * len(sorted_names) + ["total"]
waterfall_y = [base_value] + sorted_values + [0]  # total computes sum automatically
waterfall_text = [f"{base_value*100:.1f}%"] + [f"{'+' if v>0 else ''}{v*100:.1f}%" for v in sorted_values] + [f"{prob_malignant*100:.1f}%"]

fig_waterfall = go.Figure(
    go.Waterfall(
        name="SHAP Attribution",
        orientation="v",
        measure=waterfall_measure,
        x=waterfall_x,
        textposition="outside",
        text=waterfall_text,
        y=waterfall_y,
        connector={"line": {"color": "#9e9e9e", "dash": "dot"}},
        decreasing={"marker": {"color": "#2e7d32"}},   # Protective (decreases malignant probability)
        increasing={"marker": {"color": "#d32f2f"}},   # Risk driver (increases malignant probability)
        totals={"marker": {"color": "#9c27b0" if "Quantum" in selected_model_name else "#1976d2"}},
    )
)

fig_waterfall.update_layout(
    title=f"SHAP Waterfall Attribution: {selected_model_name} (Patient #{selected_idx+1:03d})",
    showlegend=False,
    height=480,
    yaxis=dict(title="Probability / Model Output Impact", tickformat=".2f"),
    xaxis=dict(title="Clinical Feature Principal Components / Qubits"),
    plot_bgcolor="#fafafa",
)
st.plotly_chart(fig_waterfall, width="stretch")

# Feature table
st.subheader("Clinical Risk Contribution Breakdown")
breakdown_df = pd.DataFrame({
    "Feature / Qubit Component": sorted_names,
    "Patient Value (Normalized)": [f"{v:.4f}" for v in sorted_patient_vals],
    "SHAP Risk Contribution": [f"{'+' if v>0 else ''}{v*100:.2f}%" for v in sorted_values],
    "Clinical Impact": ["Increases Malignancy Risk (FN Prevention)" if v > 0 else "Protective / Benign Driver" for v in sorted_values],
})
st.dataframe(breakdown_df, width="stretch")
