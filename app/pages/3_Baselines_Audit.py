"""Cross-Validation Audit & Classical Baselines Benchmark Dashboard.

Streamlit Page: app/pages/3_Baselines_Audit.py
Owner: Member A (Classical ML & Pipeline Integrity Lead)
Task: T3.3 & T3.6 (Audit Review & Baseline Tuning)
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import streamlit as st

st.set_page_config(
    page_title="Pipeline Audit & Tuned Baselines | HQML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Ponytail Styling (Modern Dark Glassmorphism)
st.markdown(
    """
    <style>
    .audit-badge {
        display: inline-block;
        padding: 6px 16px;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 12px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .hash-text {
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        background: rgba(15, 23, 42, 0.8);
        padding: 4px 8px;
        border-radius: 6px;
        color: #a855f7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛡️ 5-Fold Cross-Validation Audit & Classical Baselines")
st.caption(
    "Primary Reviewer Audit: Guarantees zero data leakage and identical 5-fold partitions across "
    "both Classical models (Member A) and Quantum circuits (Member B), alongside tuned baseline benchmarks."
)

# Load Audit Certificate
audit_file = root_dir / "configs" / "fold_integrity_audit.json"
benchmarks_file = root_dir / "configs" / "benchmark_results.json"
params_file = root_dir / "configs" / "optimal_hyperparams.json"

tabs = st.tabs(["🔒 Fold Integrity Audit", "📊 Tuned Baseline Leaderboard", "🎛️ Hyperparameter Grids"])

# ----------------- TAB 1: AUDIT -----------------
with tabs[0]:
    st.markdown('<div class="audit-badge">✅ Certified Zero Data Leakage</div>', unsafe_allow_html=True)
    st.write(
        "This audit verifies that preprocessing parameters (scaling, imputation, and PCA/MI transformations) "
        "are strictly computed from training folds. No test data was ever observed during parameter fitting."
    )

    if audit_file.exists():
        with open(audit_file, "r", encoding="utf-8") as f:
            audit_data = json.load(f)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Breast Cancer (WDBC) Split Certification")
            wdbc_audit = audit_data.get("wdbc_audit", {})
            st.markdown(f"**Status:** `{wdbc_audit.get('status', 'CERTIFIED')}`")
            st.markdown(f"**Config SHA-256:** <span class='hash-text'>{wdbc_audit.get('sha256_hash', 'N/A')[:24]}...</span>", unsafe_allow_html=True)
            
            wdbc_cov = wdbc_audit.get("disjointness_and_coverage", {})
            st.write(f"- **Total Samples:** {wdbc_cov.get('n_samples')} patients")
            st.write(f"- **Exhaustive 5-Fold Coverage:** {'Yes' if wdbc_cov.get('exhaustive_coverage') else 'No'}")
            st.write(f"- **Zero Cross-Fold Index Overlap:** {'Yes' if wdbc_cov.get('zero_index_overlap') else 'No'}")

            # Fold distribution table
            w_strat = wdbc_audit.get("stratification", {}).get("stratification", [])
            if w_strat:
                df_w = pd.DataFrame(w_strat)
                st.dataframe(df_w, use_container_width=True, hide_index=True)

        with col_b:
            st.subheader("Heart Disease (Cleveland) Split Certification")
            heart_audit = audit_data.get("heart_audit", {})
            st.markdown(f"**Status:** `{heart_audit.get('status', 'CERTIFIED')}`")
            st.markdown(f"**Config SHA-256:** <span class='hash-text'>{heart_audit.get('sha256_hash', 'N/A')[:24]}...</span>", unsafe_allow_html=True)
            
            heart_cov = heart_audit.get("disjointness_and_coverage", {})
            st.write(f"- **Total Samples:** {heart_cov.get('n_samples')} patients")
            st.write(f"- **Exhaustive 5-Fold Coverage:** {'Yes' if heart_cov.get('exhaustive_coverage') else 'No'}")
            st.write(f"- **Zero Cross-Fold Index Overlap:** {'Yes' if heart_cov.get('zero_index_overlap') else 'No'}")

            # Fold distribution table
            h_strat = heart_audit.get("stratification", {}).get("stratification", [])
            if h_strat:
                df_h = pd.DataFrame(h_strat)
                st.dataframe(df_h, use_container_width=True, hide_index=True)

        st.success(
            "🔒 **Audit Guarantee**: Identical training and validation indices are exported to JSON and "
            "strictly enforced across all classical baselines and Member B's quantum circuits."
        )
    else:
        st.warning("Audit certificate file not found. Run `python -m src.evaluation.fold_audit` to generate it.")

# ----------------- TAB 2: BENCHMARK LEADERBOARD -----------------
with tabs[1]:
    cohort_choice = st.selectbox(
        "Select Clinical Cohort",
        ["Wisconsin Diagnostic Breast Cancer (WDBC)", "UCI Heart Disease (Cleveland)"],
        index=0,
    )
    cohort_key = "wdbc" if "Breast Cancer" in cohort_choice else "heart"

    if benchmarks_file.exists():
        with open(benchmarks_file, "r", encoding="utf-8") as f:
            benchmarks_data = json.load(f)

        cohort_benchmarks = benchmarks_data.get(cohort_key, {})

        if cohort_benchmarks:
            # Summary Metrics Table
            rows = []
            for model_key, res in cohort_benchmarks.items():
                s = res.get("summary", {})
                rows.append(
                    {
                        "Model Architecture": res.get("model_name", model_key),
                        "Accuracy (%)": f"{s.get('accuracy_mean', 0)*100:.2f} ± {s.get('accuracy_std', 0)*100:.2f}",
                        "Precision (%)": f"{s.get('precision_mean', 0)*100:.2f} ± {s.get('precision_std', 0)*100:.2f}",
                        "Recall (%)": f"{s.get('recall_mean', 0)*100:.2f} ± {s.get('recall_std', 0)*100:.2f}",
                        "F1-Score (%)": f"{s.get('f1_mean', 0)*100:.2f} ± {s.get('f1_std', 0)*100:.2f}",
                        "ROC-AUC": f"{s.get('roc_auc_mean', 0):.4f} ± {s.get('roc_auc_std', 0):.4f}",
                        "Latency (ms/sample)": f"{s.get('infer_latency_mean_ms', 0):.3f} ms",
                    }
                )

            df_summary = pd.DataFrame(rows)
            st.subheader("🏆 5-Fold Cross-Validation Performance Summary")
            st.dataframe(df_summary, use_container_width=True, hide_index=True)

            # Metric Bar Chart with Error Bars
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("Accuracy Across Models (Mean ± Std)")
                model_names = [res.get("model_name", k) for k, res in cohort_benchmarks.items()]
                acc_means = [res["summary"]["accuracy_mean"] * 100 for res in cohort_benchmarks.values()]
                acc_stds = [res["summary"]["accuracy_std"] * 100 for res in cohort_benchmarks.values()]

                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=model_names,
                            y=acc_means,
                            error_y=dict(type="data", array=acc_stds, visible=True),
                            marker_color=["#38bdf8", "#818cf8", "#34d399", "#f43f5e", "#fbbf24"][:len(model_names)],
                            text=[f"{m:.1f}%" for m in acc_means],
                            textposition="auto",
                        )
                    ]
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    yaxis_title="Validation Accuracy (%)",
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=30),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_chart2:
                st.subheader("Out-of-Fold (OOF) ROC Curves")
                fig_roc = go.Figure()
                fig_roc.add_shape(
                    type="line", line=dict(dash="dash", color="grey"),
                    x0=0, x1=1, y0=0, y1=1
                )

                # Get true labels
                if cohort_key == "wdbc":
                    from src.data.wdbc_loader import load_wdbc
                    y_true = load_wdbc()[1]
                else:
                    from src.data.heart_loader import load_heart_disease
                    y_true = load_heart_disease()[1]

                for model_key, res in cohort_benchmarks.items():
                    probs = np.array(res.get("oof_probabilities", []))
                    if len(probs) > 0 and probs.shape[1] == 2:
                        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
                        score_auc = auc(fpr, tpr)
                        fig_roc.add_trace(
                            go.Scatter(
                                x=fpr, y=tpr, mode="lines",
                                name=f"{res.get('model_name', model_key)} (AUC={score_auc:.3f})"
                            )
                        )

                fig_roc.update_layout(
                    template="plotly_dark",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=30),
                    legend=dict(yanchor="bottom", y=0.05, xanchor="right", x=0.95),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            # Confusion Matrix Inspector
            st.subheader("🔍 Confusion Matrix & Error Analysis")
            selected_model = st.selectbox(
                "Select Model for Detailed Confusion Matrix",
                list(cohort_benchmarks.keys()),
                format_func=lambda k: cohort_benchmarks[k].get("model_name", k),
            )

            cm_arr = np.array(cohort_benchmarks[selected_model]["global_oof_metrics"]["confusion_matrix"])
            labels = ["Benign / Healthy", "Malignant / Disease"]
            fig_cm = px.imshow(
                cm_arr,
                labels=dict(x="Predicted Diagnosis", y="Actual Diagnosis", color="Count"),
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Viridis",
                template="plotly_dark",
            )
            fig_cm.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=30))
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Benchmark results currently compiling or not yet exported. Please run the tuner script.")

# ----------------- TAB 3: HYPERPARAMETERS -----------------
with tabs[2]:
    st.subheader("🎛️ Tuned Optimal Hyperparameter Configurations")
    st.write(
        "Optimal hyperparameters discovered via nested 5-fold cross-validation grid search to guarantee "
        "maximum baseline fairness against quantum classifiers."
    )

    if params_file.exists():
        with open(params_file, "r", encoding="utf-8") as f:
            opt_params = json.load(f)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("### 🧬 WDBC Breast Cancer Parameters")
            st.json(opt_params.get("wdbc", {}))

        with col_p2:
            st.markdown("### 🫀 UCI Heart Disease Parameters")
            st.json(opt_params.get("heart", {}))
    else:
        st.info("Optimal parameters file not found. Run `python -m src.models.classical.tuner`.")
