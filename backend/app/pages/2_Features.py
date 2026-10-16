"""Feature Engineering UI Controller - Quantum Dimensionality Reduction.

Streamlit Page: app/pages/2_Features.py
Owner: Member A (Lead / Classical ML Engineer)
Task: T4.3
"""

import sys
from pathlib import Path

# Add project root to sys.path so modules import reliably
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.data.heart_loader import load_heart_disease
from src.features.pca_reducer import PCAReducer
from src.features.mi_selector import MISelector

st.set_page_config(
    page_title="Quantum Feature Engineering | HQML Disease Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Ponytail UI Aesthetic: modern dark glassmorphism)
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .stSlider > div {
        color: #38bdf8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚛️ Dimensionality Reduction for Qubit-Constrained HQML")
st.caption(
    "Compress high-dimensional biomedical feature spaces into 4–8 qubit quantum states "
    "for Member B's AngleEmbedding, ZZFeatureMap, and StronglyEntangling circuits."
)

# Sidebar Controls
with st.sidebar:
    st.header("🎛️ Control Panel")
    dataset_choice = st.selectbox(
        "Biomedical Dataset",
        ["Wisconsin Diagnostic Breast Cancer (WDBC)", "UCI Heart Disease (Cleveland)"],
        index=0,
    )

    reduction_method = st.radio(
        "Dimensionality Reduction Method",
        ["PCA (Principal Component Analysis)", "Mutual Information (SelectKBest)"],
        index=0,
    )

    k_qubits = st.slider(
        "Target Qubit Count (k)",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Target number of qubits (Member B's NISQ circuit constraint: 4–8 qubits).",
    )

    viz_dimension = st.radio("Visualization Mode", ["3D Projection", "2D Projection"], index=0)

# Data Loading with Caching
@st.cache_data
def get_dataset(choice: str):
    if "Breast Cancer" in choice:
        df, y, names = load_wdbc(as_frame=True, return_feature_names=True)
        label_map = {0: "Benign (B)", 1: "Malignant (M)"}
        target_name = "Diagnosis"
        y_labels = [label_map[val] for val in y]
        return df, y, names, y_labels, target_name
    else:
        df, y, names = load_heart_disease(as_frame=True, return_feature_names=True)
        label_map = {0: "Healthy / No Disease", 1: "Heart Disease Present"}
        target_name = "Condition"
        y_labels = [label_map[val] for val in y]
        return df, y, names, y_labels, target_name

df_raw, y_raw, feature_names, y_labels, target_name = get_dataset(dataset_choice)

# Dimensionality Reduction Execution
if "PCA" in reduction_method:
    reducer = PCAReducer(n_components=k_qubits, scale_to_pi=True, random_state=42)
    # Scale first using leak-free preprocessor
    scaler = WDBCPreprocessor()
    X_scaled = scaler.fit_transform(df_raw.to_numpy())
    projected_data = reducer.fit_transform(X_scaled)

    evr = reducer.get_explained_variance_ratio()
    total_var = reducer.get_total_explained_variance()
    component_names = [f"PC{i+1}" for i in range(k_qubits)]
    df_projected = pd.DataFrame(projected_data, columns=component_names)
    df_projected[target_name] = y_labels

    gate_encoding = "AngleEmbedding / ZZFeatureMap: scaled to [−π, +π]"

else:
    selector = MISelector(k=k_qubits, random_state=42)
    projected_data = selector.fit_transform(df_raw, y_raw, feature_names=feature_names)
    selected_features = selector.get_selected_feature_names()
    mi_scores = selector.get_feature_scores()

    component_names = [f"Q{i+1}: {name[:16]}" for i, name in enumerate(selected_features)]
    df_projected = pd.DataFrame(projected_data, columns=component_names)
    df_projected[target_name] = y_labels

    gate_encoding = "Quantum Rotation Gates (RY, RZ) & SHAP Features: scaled to [−π, +π]"

# Top Metric Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Input Dimensions</div>'
        f'<div class="metric-value">{df_raw.shape[1]} Features</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Compressed Qubits</div>'
        f'<div class="metric-value">{k_qubits} Qubits</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    if "PCA" in reduction_method:
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Cumulative Variance</div>'
            f'<div class="metric-value">{total_var * 100:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    else:
        top_mi = max(mi_scores.values()) if mi_scores else 0.0
        st.markdown(
            f'<div class="metric-card"><div class="metric-title">Max MI Score</div>'
            f'<div class="metric-value">{top_mi:.3f} nats</div></div>',
            unsafe_allow_html=True,
        )
with col4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-title">Total Patients</div>'
        f'<div class="metric-value">{len(df_raw)} Records</div></div>',
        unsafe_allow_html=True,
    )

st.info(f"📐 **Quantum State Encoding**: {gate_encoding}")

# Patient Clusters Plotly Visualization
st.subheader("🧬 Projected Patient State Space")
color_discrete_map = {
    "Benign (B)": "#06b6d4",           # Cyan
    "Malignant (M)": "#ec4899",        # Hot Pink / Magenta
    "Healthy / No Disease": "#10b981", # Emerald Green
    "Heart Disease Present": "#f43f5e" # Rose Red
}

if viz_dimension == "3D Projection" and k_qubits >= 3:
    fig_scatter = px.scatter_3d(
        df_projected,
        x=component_names[0],
        y=component_names[1],
        z=component_names[2],
        color=target_name,
        color_discrete_map=color_discrete_map,
        title=f"3D Patient State Space Projection ({k_qubits} Qubits)",
        template="plotly_dark",
        opacity=0.85,
    )
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(
        height=620,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

else:
    fig_scatter = px.scatter(
        df_projected,
        x=component_names[0],
        y=component_names[1],
        color=target_name,
        color_discrete_map=color_discrete_map,
        title=f"2D Patient State Space Projection (Primary Latent Dimensions)",
        template="plotly_dark",
        opacity=0.85,
    )
    fig_scatter.update_traces(marker=dict(size=8, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(
        height=520,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Secondary Analysis Row
col_left, col_right = st.columns(2)

with col_left:
    if "PCA" in reduction_method:
        st.subheader("📊 Explained Variance by Principal Component")
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=[f"PC{i+1}" for i in range(k_qubits)],
                    y=evr * 100,
                    text=[f"{v*100:.1f}%" for v in evr],
                    textposition="auto",
                    marker_color="#38bdf8",
                )
            ]
        )
        fig_bar.update_layout(
            template="plotly_dark",
            yaxis_title="Explained Variance (%)",
            xaxis_title="Principal Component",
            height=360,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.subheader("📊 Mutual Information Feature Ranking")
        sorted_mi = sorted(mi_scores.items(), key=lambda item: item[1], reverse=True)[:k_qubits]
        fig_bar = go.Figure(
            data=[
                go.Bar(
                    x=[item[0] for item in sorted_mi],
                    y=[item[1] for item in sorted_mi],
                    marker_color="#a855f7",
                )
            ]
        )
        fig_bar.update_layout(
            template="plotly_dark",
            yaxis_title="Mutual Information Score (nats)",
            xaxis_title="Biomedical Feature",
            height=360,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("📋 Top Clinical Features & Quantum Mapping")
    if "PCA" in reduction_method:
        summary_rows = []
        for i in range(k_qubits):
            summary_rows.append(
                {
                    "Qubit": f"|q_{i}⟩",
                    "Component": f"PC{i+1}",
                    "Variance (%)": f"{evr[i]*100:.2f}%",
                    "Angle Range": "[−π, +π]",
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    else:
        summary_rows = []
        for i, name in enumerate(selected_features):
            summary_rows.append(
                {
                    "Qubit": f"|q_{i}⟩",
                    "Clinical Feature": name,
                    "MI Score": f"{mi_scores[name]:.4f}",
                    "Angle Range": "[−π, +π]",
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.success(
        f"✅ Transformed matrix shape: `({projected_data.shape[0]}, {projected_data.shape[1]})`. "
        f"Ready for quantum circuit simulation."
    )
