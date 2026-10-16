"""
Data Explorer UI Page: Dataset inspection, statistical summary, and interactive correlation heatmap.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import get_dataset_metadata, get_raw_dataframe  # noqa: E402

st.set_page_config(page_title="Data Explorer | QuantumHealth", page_icon="📊", layout="wide")

# Load CSS Styles
css_path = root_dir / "app" / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("📊 Clinical Data Explorer & Profiler")
st.markdown(
    "Inspect patient cohorts, compute statistical moments, and examine correlation structures."
)

# Data Source Selector
col_src, col_filt = st.columns([2, 1])

with col_src:
    source_choice = st.radio(
        "Select Data Source",
        [
            "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "UCI Heart Disease (Cleveland)",
            "Upload Custom CSV",
        ],
        horizontal=True,
    )

df: pd.DataFrame = None
dataset_name = ""

if source_choice == "Wisconsin Diagnostic Breast Cancer (WDBC)":
    df = get_raw_dataframe("wdbc")
    meta = get_dataset_metadata("wdbc")
    dataset_name = meta["name"]
    st.info(f"ℹ️ **{meta['name']}**: {meta['description']}")

elif source_choice == "UCI Heart Disease (Cleveland)":
    df = get_raw_dataframe("heart")
    meta = get_dataset_metadata("heart")
    dataset_name = meta["name"]
    st.info(f"ℹ️ **{meta['name']}**: {meta['description']}")

else:
    uploaded_file = st.file_uploader("Upload patient dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            dataset_name = uploaded_file.name
            st.success(
                f"Successfully loaded '{uploaded_file.name}' with {len(df)} rows and {len(df.columns)} columns."
            )
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
    else:
        st.warning("Please upload a CSV file or select a pre-loaded dataset above.")
        st.stop()

if df is not None:
    # Key Metrics Bar
    m1, m2, m3, m4 = st.columns(4)
    missing_count = int(df.isna().sum().sum())
    n_features = len([c for c in df.columns if c != "target"])

    with m1:
        st.metric("Total Patients", f"{len(df):,}")
    with m2:
        st.metric("Clinical Features", f"{n_features}")
    with m3:
        st.metric("Missing Values", f"{missing_count}")
    with m4:
        if "target" in df.columns:
            pos_ratio = (df["target"] == 1).mean() * 100
            st.metric("Target Positive", f"{pos_ratio:.1f}%")
        else:
            st.metric("Target Column", "Not Found")

    st.write("")

    # Raw Data Preview & Statistical Summary
    tab_summary, tab_corr, tab_dist = st.tabs(
        ["📋 Statistical Profiling", "🔥 Correlation Heatmap", "📈 Feature Distributions"]
    )

    with tab_summary:
        st.subheader("Descriptive Statistics")
        desc = df.describe().T
        desc["missing"] = df.isnull().sum()
        desc["missing_pct"] = (df.isnull().sum() / len(df) * 100).round(2)
        desc["skewness"] = df.skew(numeric_only=True).round(3)

        st.dataframe(
            desc.style.background_gradient(cmap="Blues", subset=["mean", "std"]).format(
                "{:.2f}", subset=["mean", "std", "min", "25%", "50%", "75%", "max", "skewness"]
            ),
            use_container_width=True,
        )

        with st.expander("🔍 View Raw Patient Data (First 50 Rows)"):
            st.dataframe(df.head(50), use_container_width=True)

    with tab_corr:
        st.subheader("Interactive Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Option to limit features for display if large
        selected_cols = numeric_cols
        if len(numeric_cols) > 15:
            st.caption(
                "Note: Dataset has >15 numeric features. You can filter specific features below or view all."
            )
            default_sel = list(numeric_cols[:12])
            if "target" in numeric_cols and "target" not in default_sel:
                default_sel.append("target")
            selected_cols = st.multiselect(
                "Select features for correlation analysis", numeric_cols, default=default_sel
            )

        if len(selected_cols) >= 2:
            corr_mat = df[selected_cols].corr()
            fig = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                labels=dict(color="Correlation"),
                title=f"Feature Correlation Heatmap ({dataset_name})",
            )
            fig.update_layout(
                template="plotly_dark", height=650, margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least 2 features to render the correlation matrix.")

    with tab_dist:
        st.subheader("Single Feature Distribution")
        target_feat = st.selectbox(
            "Select feature to plot", [c for c in df.columns if c != "target"]
        )
        has_target = "target" in df.columns

        if has_target:
            fig_hist = px.histogram(
                df,
                x=target_feat,
                color="target",
                barmode="overlay",
                marginal="box",
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8", "#f43f5e"],
                title=f"Distribution of '{target_feat}' stratified by Clinical Target",
            )
        else:
            fig_hist = px.histogram(
                df,
                x=target_feat,
                marginal="box",
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8"],
                title=f"Distribution of '{target_feat}'",
            )
        fig_hist.update_layout(height=450)
        st.plotly_chart(fig_hist, use_container_width=True)
