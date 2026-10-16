"""Component 5: Interactive Sensor Noise Robustness Stress-Tester."""

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_noise_stress_tester():
    """Renders the interactive sensor noise degradation stress-test studio."""
    st.markdown("### ⚡ Biomedical Sensor Noise Robustness Stress-Tester")
    st.caption("Investigates classification & sensitivity degradation under increasing levels of Gaussian biomedical telemetry noise.")

    csv_path = Path("reports/tables/noise_robustness_wdbc.csv")
    if not csv_path.exists():
        st.info("Run `python scripts/run_noise_test.py` to generate the noise benchmark table.")
        return

    df = pd.read_csv(csv_path)

    # Interactive Noise Slider
    noise_levels = sorted(df["noise_level"].unique())
    selected_noise = st.select_slider(
        "Inject Simulated Sensor Noise Level (σ):",
        options=noise_levels,
        value=noise_levels[2] if len(noise_levels) > 2 else noise_levels[0],
        format_func=lambda x: f"{int(x*100)}% Perturbation Noise",
    )

    # Filter for current slider level
    sub_df = df[df["noise_level"] == selected_noise]

    # Metrics Row
    m_col1, m_col2 = st.columns(2)
    q_row = sub_df[sub_df["model"].str.contains("Quantum")].iloc[0] if not sub_df[sub_df["model"].str.contains("Quantum")].empty else None
    c_row = sub_df[sub_df["model"].str.contains("Classical")].iloc[0] if not sub_df[sub_df["model"].str.contains("Classical")].empty else None

    if q_row is not None and c_row is not None:
        with m_col1:
            st.metric(
                label=f"Quantum Sensitivity (Recall at {int(selected_noise*100)}% Noise)",
                value=f"{q_row['sensitivity']*100:.1f}%",
                delta=f"{(q_row['sensitivity'] - c_row['sensitivity'])*100:+.1f}% vs Classical",
            )
        with m_col2:
            st.metric(
                label=f"Classical SVM Sensitivity at {int(selected_noise*100)}% Noise",
                value=f"{c_row['sensitivity']*100:.1f}%",
            )

    # Plotly Degradation Curves
    fig = go.Figure()

    q_data = df[df["model"].str.contains("Quantum")]
    c_data = df[df["model"].str.contains("Classical")]

    fig.add_trace(
        go.Scatter(
            x=[f"{int(x*100)}%" for x in q_data["noise_level"]],
            y=q_data["sensitivity"] * 100,
            mode="lines+markers",
            name="Quantum QSVC (Sensitivity)",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"{int(x*100)}%" for x in c_data["noise_level"]],
            y=c_data["sensitivity"] * 100,
            mode="lines+markers",
            name="Classical SVM (Sensitivity)",
            line=dict(color="#f43f5e", width=3, dash="dash"),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Diagnostic Sensitivity Retention across Sensor Noise Spectrum",
        xaxis_title="Biomedical Noise Level (Additive Gaussian Perturbation)",
        yaxis_title="Clinical Sensitivity / True Positive Rate (%)",
        yaxis=dict(range=[50, 105]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(color="#94a3b8"),
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
