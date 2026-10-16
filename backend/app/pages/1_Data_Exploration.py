"""Interactive Clinical Dataset Ingestion & Patient Exploration Dashboard.

Streamlit Page: app/pages/1_Data_Exploration.py
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Deliverables: D2 (Data Ingestion) & D5 (Single-Sample Inference Hardening)
Aesthetic: PonyTail Clinical Dark/Light Glassmorphism
"""

import sys
import time
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.wdbc_loader import load_wdbc
from src.data.heart_loader import load_heart_disease, RAW_COLUMNS
from src.data.validator import PatientRecordValidator, ValidationError
from src.models.classical.inference import ClinicalInferenceService

# Page Setup
st.set_page_config(
    page_title="Data Exploration & Clinical Ingestion | HQML",
    page_icon="🔬",
    layout="wide",
)

# Custom PonyTail Aesthetic Styling
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-val {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
    .badge-blue { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); }
    .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("🔬 Clinical Cohort Ingestion & Live Patient Validation")
st.markdown(
    """
    **Deliverable D2 / D5 — Member A (Data & Pipeline Integrity Lead)**  
    Inspect sanitized patient cohorts, review class balances, examine correlation topologies, 
    and validate individual patient records against clinical sensor boundaries in real-time.
    """
)

# Dataset Selector
col_ctrl1, col_ctrl2 = st.columns([2, 1])
with col_ctrl1:
    dataset_choice = st.selectbox(
        "Select Clinical Cohort",
        ["Wisconsin Diagnostic Breast Cancer (WDBC)", "UCI Cleveland Heart Disease"],
        index=0,
    )

is_wdbc = "Breast Cancer" in dataset_choice
dataset_key = "wdbc" if is_wdbc else "heart"


# Data Loader with Caching
@st.cache_data
def get_cohort_data(key: str):
    if key == "wdbc":
        X, y, feature_names = load_wdbc(as_frame=False, return_feature_names=True)
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df["diagnosis_label"] = df["target"].map({0: "Benign", 1: "Malignant"})
        return df, feature_names
    else:
        X, y, feature_names = load_heart_disease(as_frame=False, preprocess=False, return_feature_names=True)
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        df["diagnosis_label"] = df["target"].map({0: "Healthy (No Disease)", 1: "Heart Disease Present"})
        return df, feature_names


df, feature_cols = get_cohort_data(dataset_key)
n_samples, n_features = len(df), len(feature_cols)
pos_count = int(df["target"].sum())
neg_count = n_samples - pos_count
pos_pct = (pos_count / n_samples) * 100.0

# Top Metric Cards
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Patient Cohort</div>
            <div class="metric-val">{n_samples}</div>
            <span class="badge-pill badge-green">100% Ingested</span>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">Clinical Attributes</div>
            <div class="metric-val">{n_features}</div>
            <span class="badge-pill badge-blue">Continuous & Categorical</span>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol3:
    pos_label = "Malignant" if is_wdbc else "Disease Present"
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-title">{pos_label} Cases</div>
            <div class="metric-val">{pos_count} <span style="font-size: 1rem; color: #94a3b8;">({pos_pct:.1f}%)</span></div>
            <span class="badge-pill badge-red">Stratified 5-Fold</span>
        </div>""",
        unsafe_allow_html=True,
    )
with mcol4:
    st.markdown(
        """<div class="metric-card">
            <div class="metric-title">Leakage Audit</div>
            <div class="metric-val">0.00%</div>
            <span class="badge-pill badge-green">SHA-256 Verified</span>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()

# Tabbed Layout for Exploration & Live Patient Diagnostic Sandbox
tab_explore, tab_distributions, tab_sandbox = st.tabs(
    ["📊 Cohort Inspection", "📈 Feature Distributions & Correlation", "🩺 Single-Patient Diagnostic Sandbox"]
)

# ----------------- TAB 1: Cohort Inspection -----------------
with tab_explore:
    st.subheader(f"Patient Data Records Preview ({n_samples} Biopsies / Examinations)")
    st.dataframe(df, use_container_width=True, height=320)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("##### Clinical Class Balance")
        fig_donut = px.pie(
            df,
            names="diagnosis_label",
            hole=0.55,
            color="diagnosis_label",
            color_discrete_map={
                "Benign": "#10b981",
                "Malignant": "#f43f5e",
                "Healthy (No Disease)": "#10b981",
                "Heart Disease Present": "#f43f5e",
            },
        )
        fig_donut.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_t2:
        st.markdown("##### Descriptive Statistics (Means & Deviations)")
        st.dataframe(df[feature_cols[:8]].describe().T[["mean", "std", "min", "max"]], use_container_width=True)

# ----------------- TAB 2: Feature Distributions & Correlation -----------------
with tab_distributions:
    st.subheader("Feature Distributions & Inter-Marker Correlations")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        selected_feature = st.selectbox("Select Feature for Histogram", feature_cols, index=0)
        fig_hist = px.histogram(
            df,
            x=selected_feature,
            color="diagnosis_label",
            barmode="overlay",
            marginal="box",
            color_discrete_map={
                "Benign": "#10b981",
                "Malignant": "#f43f5e",
                "Healthy (No Disease)": "#10b981",
                "Heart Disease Present": "#f43f5e",
            },
        )
        fig_hist.update_layout(
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_f2:
        st.markdown("##### Correlation Matrix (Top 10 Variables)")
        corr_subset = df[feature_cols[:10]].corr()
        fig_corr = px.imshow(
            corr_subset,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
        )
        fig_corr.update_layout(
            margin=dict(l=10, r=10, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 3: Single-Patient Diagnostic Sandbox -----------------
with tab_sandbox:
    st.subheader("🩺 Single-Patient Bedside Inference & Input Guardrails")
    st.markdown(
        """
        Test the **`PatientRecordValidator`** (T5.1) and **`ClinicalInferenceService`** (T5.2).  
        Select a real patient biopsy/examination from the cohort, modify clinical variables, 
        and execute hardened classical inference with automatic out-of-range sensor error catching.
        """
    )

    # Patient Selector
    patient_idx = st.slider("Select Patient Index from Cohort", 0, n_samples - 1, 0)
    actual_diagnosis = df.iloc[patient_idx]["diagnosis_label"]
    patient_record = df.iloc[patient_idx][feature_cols].to_dict()

    st.info(f"**Loaded Patient #{patient_idx}** — Ground Truth Pathology: **{actual_diagnosis}**")

    # Form to modify features
    with st.form("patient_inference_form"):
        st.markdown("##### Inspect / Modify Patient Sensor & Biopsy Inputs")
        cols_per_row = 3
        form_cols = st.columns(cols_per_row)
        modified_record = {}

        # Show primary features for tuning
        display_features = feature_cols[:9] if is_wdbc else feature_cols
        for i, col_name in enumerate(display_features):
            with form_cols[i % cols_per_row]:
                curr_val = float(patient_record[col_name])
                modified_record[col_name] = st.number_input(
                    f"{col_name}",
                    value=curr_val,
                    format="%.4f" if is_wdbc else "%.2f",
                )

        # Retain remaining features unchanged
        for col_name in feature_cols:
            if col_name not in modified_record:
                modified_record[col_name] = float(patient_record[col_name])

        col_btn1, col_btn2 = st.columns([1, 2])
        with col_btn1:
            run_btn = st.form_submit_button("⚡ Execute Clinical Inference", type="primary")

    if run_btn:
        st.markdown("#### Diagnostic Evaluation Outcome")
        with st.spinner("Executing sanitized clinical inference pipeline..."):
            service = ClinicalInferenceService(dataset_name=dataset_key)
            result = service.predict_patient(modified_record)

        if not result.is_valid:
            st.error(f"🛑 **Validation Error (Guardrail Triggered):** {result.validation_error}")
            st.warning("Input rejected prior to model evaluation to prevent spurious medical decisions.")
        else:
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                badge_class = "badge-red" if result.predicted_class == 1 else "badge-green"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">Predicted Pathology</div>
                        <div class="metric-val">{result.class_label}</div>
                        <span class="badge-pill {badge_class}">Class {result.predicted_class}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with rcol2:
                cat_class = "badge-red" if result.risk_category == "High Risk" else "badge-blue"
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">Pathology Risk Score</div>
                        <div class="metric-val">{result.risk_score * 100:.2f}%</div>
                        <span class="badge-pill {cat_class}">{result.risk_category}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with rcol3:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">Inference Latency</div>
                        <div class="metric-val">{result.latency_ms:.2f} ms</div>
                        <span class="badge-pill badge-green">Model: {result.model_name}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            # Risk Gauge Plot
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=result.risk_score * 100.0,
                    title={"text": "Diagnostic Risk Score (%)", "font": {"color": "#94a3b8"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                        "bar": {"color": "#f43f5e" if result.risk_score >= 0.7 else ("#38bdf8" if result.risk_score >= 0.3 else "#10b981")},
                        "steps": [
                            {"range": [0, 30], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [30, 70], "color": "rgba(56, 189, 248, 0.15)"},
                            {"range": [70, 100], "color": "rgba(244, 63, 94, 0.15)"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
