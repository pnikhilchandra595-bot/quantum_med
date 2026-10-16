"""Component 3: Interactive 3D Bloch Sphere Visualizer with WebGL."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def create_bloch_sphere_figure(theta: float, phi: float, qubit_index: int = 0) -> go.Figure:
    """Constructs an interactive 3D Bloch Sphere with the statevector arrow."""
    # 1. Sphere Surface Mesh
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    fig = go.Figure()

    # Wireframe Sphere Surface
    fig.add_trace(
        go.Surface(
            x=x_sphere,
            y=y_sphere,
            z=z_sphere,
            opacity=0.15,
            colorscale=[[0, "#38bdf8"], [1, "#818cf8"]],
            showscale=False,
            hoverinfo="none",
        )
    )

    # Equator ring
    theta_eq = np.linspace(0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter3d(
            x=np.cos(theta_eq),
            y=np.sin(theta_eq),
            z=np.zeros_like(theta_eq),
            mode="lines",
            line=dict(color="rgba(148, 163, 184, 0.4)", width=3),
            hoverinfo="none",
            name="Equator (XY)",
        )
    )

    # Coordinate Axes Lines
    axes_lines = [
        ([-1.2, 1.2], [0, 0], [0, 0], "X-axis"),
        ([0, 0], [-1.2, 1.2], [0, 0], "Y-axis"),
        ([0, 0], [0, 0], [-1.2, 1.2], "Z-axis"),
    ]
    for lx, ly, lz, name in axes_lines:
        fig.add_trace(
            go.Scatter3d(
                x=lx, y=ly, z=lz,
                mode="lines",
                line=dict(color="#475569", width=2),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Basis State Poles
    poles = [
        ([0], [0], [1.18], "|0⟩", "#10b981"),
        ([0], [0], [-1.22], "|1⟩", "#f43f5e"),
        ([1.22], [0], [0], "|+⟩", "#38bdf8"),
        ([-1.22], [0], [0], "|-⟩", "#94a3b8"),
        ([0], [1.22], [0], "|i⟩", "#a855f7"),
        ([0], [-1.22], [0], "|-i⟩", "#a855f7"),
    ]
    for px, py, pz, label, col in poles:
        fig.add_trace(
            go.Scatter3d(
                x=px, y=py, z=pz,
                mode="text",
                text=[label],
                textposition="middle center",
                textfont=dict(size=14, color=col),
                showlegend=False,
            )
        )

    # State Vector Calculation
    # |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
    vx = np.sin(theta) * np.cos(phi)
    vy = np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)

    # Draw State Vector Line
    fig.add_trace(
        go.Scatter3d(
            x=[0, vx],
            y=[0, vy],
            z=[0, vz],
            mode="lines",
            line=dict(color="#fbbf24", width=7),
            name="State Vector |ψ⟩",
        )
    )

    # Tip point
    fig.add_trace(
        go.Scatter3d(
            x=[vx],
            y=[vy],
            z=[vz],
            mode="markers",
            marker=dict(size=8, color="#fbbf24"),
            hoverinfo="text",
            hovertext=f"|ψ⟩ on Qubit q[{qubit_index}]<br>θ = {theta:.2f} rad ({np.degrees(theta):.1f}°)<br>φ = {phi:.2f} rad ({np.degrees(phi):.1f}°)",
            name="Tip",
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        showlegend=False,
    )
    return fig


def render_bloch_sphere():
    """Renders the 3D Bloch sphere visualizer."""
    st.markdown("### 🌐 3D Interactive Bloch Sphere Visualizer")
    st.caption("Visualizes the single-qubit quantum statevector |ψ⟩ in the complex projective Hilbert space (SU(2)).")

    c_ctrl, c_viz = st.columns([1, 2])

    with c_ctrl:
        st.markdown("**Qubit & Rotation Controls**")
        qubit_idx = st.selectbox("Select Target Qubit:", [f"Qubit q[{i}]" for i in range(4)])
        idx = int(qubit_idx.split("[")[1].split("]")[0])

        theta = st.slider("Polar Angle θ (Feature Encoding Ry):", min_value=0.0, max_value=float(np.pi), value=1.05, step=0.05)
        phi = st.slider("Azimuthal Phase φ (Entangling Phase Rz):", min_value=0.0, max_value=float(2 * np.pi), value=0.78, step=0.05)

        p0 = float(np.cos(theta / 2.0) ** 2)
        p1 = float(np.sin(theta / 2.0) ** 2)

        st.markdown("#### Measurement Probabilities")
        st.progress(p0, text=f"P(|0⟩ Ground State): {p0*100:.1f}%")
        st.progress(p1, text=f"P(|1⟩ Excited State): {p1*100:.1f}%")

        st.info(f"**State Formula:**<br>`|ψ⟩ = {np.cos(theta/2):.3f}|0⟩ + ({np.sin(theta/2)*np.cos(phi):.3f} + {np.sin(theta/2)*np.sin(phi):.3f}i)|1⟩`")

    with c_viz:
        fig = create_bloch_sphere_figure(theta, phi, qubit_index=idx)
        st.plotly_chart(fig, use_container_width=True)
