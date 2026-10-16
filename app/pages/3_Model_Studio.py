"""
Model Studio & Real-Time Trainer UI: Live interactive training for classical and quantum models.
"""

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Ensure repository root is on sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data.loader import AVAILABLE_DATASETS, load_dataset  # noqa: E402
from src.models.factory import ModelFactory  # noqa: E402
from src.quantum.backend import QuantumBackendManager  # noqa: E402
from src.utils.serialization import save_model  # noqa: E402

st.set_page_config(page_title="Model Studio | QuantumHealth", page_icon="⚡", layout="wide")

# Load CSS Styles
css_path = root_dir / "app" / "styles.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚡ Model Studio & Real-Time QML Trainer")
st.markdown(
    "Train, optimize, and inspect classical and quantum neural architectures with live gradient telemetry."
)

# Configuration Columns
c_left, c_right = st.columns([1, 1])

with c_left:
    dataset_name = st.selectbox(
        "Clinical Dataset",
        AVAILABLE_DATASETS,
        format_func=lambda x: "WDBC (Breast Cancer)" if x == "wdbc" else "UCI Heart Disease",
    )

    model_category = st.radio(
        "Model Category", ["Quantum Models", "Classical Baselines"], horizontal=True
    )
    if model_category == "Quantum Models":
        model_name = st.selectbox(
            "Quantum Model Architecture",
            ["vqc", "qsvc", "hqnn"],
            format_func=lambda x: {
                "vqc": "VQC (Variational Quantum Classifier)",
                "qsvc": "QSVC (Quantum Support Vector Classifier)",
                "hqnn": "HQNN (Hybrid Quantum Neural Network)",
            }[x],
        )
    else:
        model_name = st.selectbox(
            "Classical Model Architecture",
            ["svm", "rf", "xgb", "mlp"],
            format_func=lambda x: {
                "svm": "Support Vector Machine (RBF)",
                "rf": "Random Forest Classifier",
                "xgb": "XGBoost Classifier",
                "mlp": "Multi-Layer Perceptron (MLP)",
            }[x],
        )

with c_right:
    n_qubits = st.slider("Qubits / Reduced Features", min_value=2, max_value=8, value=4)
    reps = st.slider("Circuit Layers / Repetitions", min_value=1, max_value=4, value=1)
    epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=25)
    lr = st.select_slider("Learning Rate", options=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], value=0.05)

st.divider()

# Start Training Action
train_btn = st.button("🚀 Start Model Training", type="primary", use_container_width=True)

if train_btn:
    st.markdown("### 🔄 Training Progress & Optimization Telemetry")

    # Load dataset
    with st.spinner("Preparing clinical datasets and tensor pipelines..."):
        data = load_dataset(name=dataset_name, n_qubits=n_qubits, reduction="pca")
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]

    # Backend
    backend = QuantumBackendManager(backend_name="default.qubit")
    model = ModelFactory.create_model(
        model_name=model_name,
        n_qubits=n_qubits,
        n_layers=reps,
        backend_manager=backend,
        lr=lr,
        epochs=epochs,
    )

    prog_bar = st.progress(0, text="Initializing training pipeline...")
    loss_chart_placeholder = st.empty()
    status_placeholder = st.empty()

    live_epochs = []
    live_losses = []

    def on_step(epoch_idx, loss_val):
        pct = int((epoch_idx / epochs) * 100)
        prog_bar.progress(pct, text=f"Epoch {epoch_idx}/{epochs} | Current Loss: {loss_val:.4f}")
        live_epochs.append(epoch_idx)
        live_losses.append(loss_val)

        # Update dynamic loss curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=live_epochs,
                y=live_losses,
                mode="lines+markers",
                line=dict(color="#06b6d4", width=3),
                marker=dict(size=6, color="#38bdf8"),
                name="Training Loss",
            )
        )
        fig.update_layout(
            title=f"Real-Time Convergence Curve — {model_name.upper()}",
            xaxis_title="Epoch",
            yaxis_title="Loss (Cost)",
            template="plotly_dark",
            height=340,
            margin=dict(l=40, r=40, t=50, b=40),
        )
        loss_chart_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.01)

    t0 = time.time()
    try:
        if model_name in ["vqc", "hqnn"]:
            model.fit(X_train, y_train, epochs=epochs, lr=lr, step_callback=on_step)
        else:
            prog_bar.progress(50, text="Fitting non-iterative kernel / tree estimator...")
            model.fit(X_train, y_train)
            prog_bar.progress(100, text="Fit completed!")
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    train_time = time.time() - t0
    prog_bar.progress(100, text=f"✅ Optimization complete in {train_time:.2f} seconds.")

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    # Save to session state
    st.session_state["trained_model"] = model
    st.session_state["model_name"] = model_name
    st.session_state["metrics"] = metrics
    st.session_state["n_qubits"] = n_qubits
    st.session_state["dataset_name"] = dataset_name
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

# Display Evaluation & Inference if model exists in session
if "trained_model" in st.session_state:
    st.divider()
    st.subheader(f"📊 Evaluation Metrics ({st.session_state['model_name'].upper()})")

    metrics = st.session_state["metrics"]
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    with m2:
        st.metric("Sensitivity / Recall", f"{metrics['recall'] * 100:.2f}%")
    with m3:
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    with m4:
        st.metric("F1-Score", f"{metrics['f1'] * 100:.2f}%")
    with m5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    # Checkpoint export
    st.write("")
    chk_filename = f"{st.session_state['model_name']}_{st.session_state['dataset_name']}.pt"
    tmp_path = f"checkpoints/{chk_filename}"
    save_model(st.session_state["trained_model"], tmp_path)

    with open(tmp_path, "rb") as f:
        st.download_button(
            label=f"💾 Download Trained Weights ({chk_filename})",
            data=f,
            file_name=chk_filename,
            mime="application/octet-stream",
        )

    # Interactive Patient Inference Simulator
    st.write("")
    st.subheader("🩺 Single-Patient Diagnostic Inference Simulator")
    st.markdown(
        "Adjust quantum rotation angles to test diagnostic classification on a synthetic patient:"
    )

    q_cols = st.columns(st.session_state["n_qubits"])
    patient_angles = []
    for i, col in enumerate(q_cols):
        with col:
            val = st.slider(f"Qubit {i} Angle ($x_{i}$)", -3.14, 3.14, 0.0, step=0.1)
            patient_angles.append(val)

    patient_arr = np.array([patient_angles], dtype=np.float32)
    t_model = st.session_state["trained_model"]
    pred = t_model.predict(patient_arr)[0]
    prob = t_model.predict_proba(patient_arr)[0]

    inf_c1, inf_c2 = st.columns([1, 1])
    with inf_c1:
        if pred == 1:
            st.error(f"🚨 **Prediction:** Malignant / High Risk (Confidence: {prob[1]*100:.1f}%)")
        else:
            st.success(f"🟢 **Prediction:** Benign / Low Risk (Confidence: {prob[0]*100:.1f}%)")
    with inf_c2:
        st.progress(float(prob[1]), text=f"Disease Probability: {prob[1]*100:.1f}%")
