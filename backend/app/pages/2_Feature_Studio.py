"""
Feature Studio UI Page: Dimensionality reduction, PCA variance scree, and quantum feature mapping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import get_raw_dataframe  # noqa: E402

st.set_page_config(page_title="Feature Studio | QuantumHealth", page_icon="🎛️", layout="wide")

# Load CSS Styles
css_path = root_dir / "app" / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("🎛️ Feature Studio & Quantum Encoding")
st.markdown(
    "Bridge high-dimensional medical records to NISQ qubit registers via PCA and Mutual Information."
)

c1, c2 = st.columns([1, 1])

with c1:
    dataset_choice = st.selectbox(
        "Select Clinical Dataset",
        ["wdbc", "heart"],
        format_func=lambda x: "WDBC (Breast Cancer)" if x == "wdbc" else "UCI Heart Disease",
    )

with c2:
    n_qubits = st.slider(
        "Target Quantum Register Size (Qubits / Dimensions)", min_value=2, max_value=8, value=4
    )

df = get_raw_dataframe(dataset_choice)
feature_cols = [c for c in df.columns if c != "target"]
X_raw = df[feature_cols].values
y = df["target"].values

# Pre-scaling
scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = scaler.fit_transform(X_raw)

tab_pca, tab_mi, tab_quantum = st.tabs(
    [
        "📉 PCA Scree & Projections",
        "🎯 Mutual Information Ranking",
        "⚛️ Quantum Feature Map Preview",
    ]
)

with tab_pca:
    st.subheader("Principal Component Analysis (PCA)")
    pca_full = PCA().fit(X_scaled)
    var_exp = pca_full.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_exp)

    col_scree, col_proj = st.columns([1, 1])

    with col_scree:
        fig_scree = go.Figure()
        fig_scree.add_trace(
            go.Bar(
                x=[f"PC{i+1}" for i in range(len(var_exp[:10]))],
                y=var_exp[:10],
                name="Individual Variance",
                marker_color="#06b6d4",
            )
        )
        fig_scree.add_trace(
            go.Scatter(
                x=[f"PC{i+1}" for i in range(len(var_exp[:10]))],
                y=cum_var[:10],
                name="Cumulative Variance",
                marker_color="#a855f7",
                mode="lines+markers",
            )
        )
        fig_scree.add_hline(
            y=80, line_dash="dash", line_color="#ef4444", annotation_text="80% Threshold"
        )
        fig_scree.update_layout(
            title="Explained Variance Scree Plot (Top 10 PCs)",
            yaxis_title="Variance Explained (%)",
            template="plotly_dark",
            height=400,
        )
        st.plotly_chart(fig_scree, use_container_width=True)

        retained_var = cum_var[n_qubits - 1]
        st.success(
            f"**{n_qubits}-Qubit Register Retains:** `{retained_var:.2f}%` of total clinical variance."
        )

    with col_proj:
        pca_proj = PCA(n_components=min(3, n_qubits)).fit_transform(X_scaled)
        if n_qubits >= 3:
            fig_3d = px.scatter_3d(
                x=pca_proj[:, 0],
                y=pca_proj[:, 1],
                z=pca_proj[:, 2],
                color=[f"Class {val}" for val in y],
                title="3D PCA Feature Projection",
                labels={"x": "PC1", "y": "PC2", "z": "PC3"},
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8", "#f43f5e"],
            )
            fig_3d.update_layout(height=400, margin=dict(l=0, r=0, b=0, t=30))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            fig_2d = px.scatter(
                x=pca_proj[:, 0],
                y=pca_proj[:, 1],
                color=[f"Class {val}" for val in y],
                title="2D PCA Feature Projection",
                labels={"x": "PC1", "y": "PC2"},
                template="plotly_dark",
                color_discrete_sequence=["#38bdf8", "#f43f5e"],
            )
            fig_2d.update_layout(height=400)
            st.plotly_chart(fig_2d, use_container_width=True)

with tab_mi:
    st.subheader("Mutual Information (Non-Linear Dependency Ranking)")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": feature_cols, "MI_Score": mi_scores}).sort_values(
        by="MI_Score", ascending=True
    )

    fig_mi = px.bar(
        mi_df.tail(15),
        x="MI_Score",
        y="Feature",
        orientation="h",
        color="MI_Score",
        color_continuous_scale="Teal",
        title="Top Clinical Biomarkers Ranked by Mutual Information",
        template="plotly_dark",
    )
    fig_mi.update_layout(height=450)
    st.plotly_chart(fig_mi, use_container_width=True)

with tab_quantum:
    st.subheader(f"Quantum State Preparation Circuit ({n_qubits} Qubits)")
    st.markdown(r"""
    When data is encoded via **Angle Embedding**, each reduced feature $x_i \in [-\pi, \pi]$ rotates
    a single qubit around the Y-axis:

    $$|\psi(\mathbf{x})\rangle = \bigotimes_{i=1}^{N} R_y(x_i) |0\rangle = \bigotimes_{i=1}^{N} \left( \cos\left(\frac{x_i}{2}\right)|0\rangle + \sin\left(\frac{x_i}{2}\right)|1\rangle \right)$$
    """)

    # Render interactive ASCII/Markdown Circuit Blueprint
    st.markdown("```text")
    circuit_str = ""
    for q in range(n_qubits):
        circuit_str += f"|0⟩ Wire {q} ───[ Ry(x_{q}) ]───●───\n"
        circuit_str += "                           │   \n" if q < n_qubits - 1 else ""
    circuit_str += "                           └───[ CNOT Entanglement ]───> ⟨Z⟩"
    st.text(circuit_str)
    st.markdown("```")

    st.info(
        r"✨ **Hilbert Space Expansion:** A 4-qubit circuit operates in a 16-dimensional complex state-space $\mathbb{C}^{16}$, and an 8-qubit register operates in $\mathbb{C}^{256}$!"
    )
