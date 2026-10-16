# MEMBER_C_TASKS.md — Evaluation & Explainability (XAI) Specialist

> **Owner:** Member C (Evaluation & XAI Specialist)  
> **Backup:** Member B (Quantum ML) / Member A (Classical ML)  
> **Primary Deliverables:** D7 (Benchmarking Suite), D8 (Explainability), D5 (Tree Baselines), D12 (Pitch Deck)  
> **Total Direct Task Hours:** ~49 Hours across 5 Phases  

---

## Lane Objective
Deliver rigorous, statistically sound evaluation pipelines comparing classical and quantum models across 5 metrics (Accuracy, Sensitivity/Recall, Specificity, F1, ROC-AUC), conduct hypothesis testing for quantum advantage, implement classical SHAP explainers, harmonize XAI comparisons, and design the competition pitch deck.

---

## Phase 1 & 2: Evaluation Engine & Tree Baselines

<task type="auto" effort="medium">
  <name>T1.7: Classical Tree Baselines (Random Forest & XGBoost)</name>
  <files>
    src/models/classical/trees.py
    tests/test_classical_models.py
  </files>
  <action>
    Implement Random Forest and XGBoost classifiers matching BaseModel interface.
    Steps:
    1. RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).
    2. XGBClassifier(learning_rate=0.05, n_estimators=100, max_depth=4).
    3. Implement fit, predict, predict_proba with deterministic seeds.
  </action>
  <verify>
    pytest tests/test_classical_models.py -k "test_trees"
  </verify>
  <done>
    Both tree models train cleanly and output valid probabilities.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T1.9: Standardized Evaluation Metrics Calculator</name>
  <files>
    src/evaluation/metrics.py
    tests/test_evaluation.py
  </files>
  <action>
    Build clinical metrics engine:
    - Accuracy: $(TP+TN)/(P+N)$
    - Sensitivity (Recall): $TP/(TP+FN)$ — critical for cancer/heart detection to minimize false negatives!
    - Specificity: $TN/(TN+FP)$
    - F1-Score: $2*(Prec * Rec)/(Prec + Rec)$
    - ROC-AUC: Area under receiver operating characteristic curve
    - Wall-clock inference latency per sample (ms)
    Include edge-case handling for zero division.
  </action>
  <verify>
    pytest tests/test_evaluation.py -k "test_metrics"
  </verify>
  <done>
    Returns standardized dictionary with all 6 metrics formatted to 4 decimal places.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T2.3: Feature Reduction Ablation Study</name>
  <files>
    scripts/run_feature_ablation.py
    reports/tables/feature_ablation.csv
  </files>
  <action>
    Benchmark classical SVM and Random Forest across dimensions $k \in [2, 4, 6, 8, 10, 15, 30]$.
    Record accuracy and variance retained to identify optimal qubit cutoff for Member B's circuits.
  </action>
  <verify>
    python scripts/run_feature_ablation.py --dataset wdbc
  </verify>
  <done>
    Generates CSV and curve showing that $k=4..6$ retains $>88\%$ of total variance.
  </done>
</task>

---

## Phase 3: Master Benchmarking & Comparative Explainability

<task type="auto" effort="high">
  <name>T3.3: Master 5-Fold Cross-Validation Benchmark Suite</name>
  <files>
    src/evaluation/benchmark.py
    scripts/run_master_benchmark.py
    reports/tables/benchmark_results.csv
  </files>
  <action>
    Orchestrate automated 5-fold cross-validation across all 7 models:
    - Classical: Linear SVM, RBF SVM, Random Forest, XGBoost, MLP.
    - Quantum: QSVC, VQC, HQNN (from Member B).
    Log mean and standard deviation for all metrics. Save master summary CSV.
  </action>
  <verify>
    python scripts/run_master_benchmark.py --dataset wdbc --quick
  </verify>
  <done>
    Master benchmark table generated comparing all 7 models under identical splits.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T3.4: Statistical Significance Testing (Quantum Advantage Check)</name>
  <files>
    src/evaluation/significance.py
    reports/tables/significance_p_values.json
  </files>
  <action>
    Implement Paired Student's t-test and Wilcoxon Signed-Rank test across the 5 CV folds.
    Test hypotheses:
    $H_0$: Quantum Sensitivity == Classical Sensitivity.
    $H_1$: Quantum Sensitivity > Classical Sensitivity ($p < 0.05$).
  </action>
  <verify>
    python -m src.evaluation.significance
  </verify>
  <done>
    Computes formal p-values proving whether quantum sensitivity gains are statistically significant.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T3.5: Publication-Grade Visualizations (ROC, PR & Confusion Matrices)</name>
  <files>
    src/evaluation/plots.py
    reports/figures/roc_curves.svg
    reports/figures/confusion_matrices.svg
  </files>
  <action>
    Generate high-resolution SVG/PNG plots:
    1. Multi-model ROC curves with shaded $\pm 1$ standard deviation bands.
    2. Side-by-side Confusion Matrices for Classical SVM vs QSVC.
    3. Precision-Recall curves highlighting clinical minority class detection.
  </action>
  <verify>
    python -c "from src.evaluation.plots import generate_all_plots; generate_all_plots(); print('Plots verified.')"
  </verify>
  <done>
    Publication-ready SVG figures saved in reports/figures/.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T3.6: Classical KernelSHAP Explainer Adapter</name>
  <files>
    src/xai/classical_shap.py
    tests/test_xai.py
  </files>
  <action>
    Implement SHAP TreeExplainer (for RF/XGB) and KernelExplainer (for SVM/MLP).
    Output normalized feature importance vectors matching Member B's QuantumKernelSHAP output format.
  </action>
  <verify>
    pytest tests/test_xai.py -k "test_classical_shap"
  </verify>
  <done>
    Outputs consistent SHAP attribution values for classical models.
  </done>
</task>

<task type="auto" effort="medium">
  <name>T3.8: Classical vs Quantum Feature Attribution Comparator</name>
  <files>
    src/xai/comparator.py
    reports/figures/shap_radar_comparison.svg
  </files>
  <action>
    Compare top-5 clinical risk drivers identified by Classical models vs QSVC/VQC.
    Generate alignment metrics (Spearman rank correlation of feature importances) and radar chart.
  </action>
  <verify>
    python -m src.xai.comparator
  </verify>
  <done>
    Generates comparison report demonstrating clinical interpretability alignment.
  </done>
</task>

---

## Phase 4 & 5: Streamlit XAI Tabs & Final Pitch Deck

<task type="auto" effort="high">
  <name>T4.6: Clinical Explainability UI Dashboard</name>
  <files>
    app/pages/5_Explainability.py
  </files>
  <action>
    Build interactive patient risk explanation page in Streamlit:
    1. Select sample patient from test set.
    2. Display prediction confidence gauge (e.g. 87% Malignant).
    3. Render Plotly SHAP waterfall plot showing positive/negative risk contributors.
  </action>
  <verify>
    Launch Streamlit and test waterfall chart rendering for patient sample.
  </verify>
  <done>
    Interactive SHAP waterfall plot renders smoothly in browser.
  </done>
</task>

<task type="auto" effort="high">
  <name>T5.6: 15-Slide Competition Pitch Deck</name>
  <files>
    docs/HQML_Presentation_Deck.pptx
    docs/HQML_Presentation_Deck.pdf
  </files>
  <action>
    Design 15-slide deck structured for jury evaluation:
    1. Title & Team Introduction.
    2. Clinical Problem (Late diagnosis mortality & high false negative costs).
    3. Why Hybrid Quantum Machine Learning (Hilbert space representation).
    4. System Architecture (D1-D9).
    5. Benchmarking Results (Table + ROC curve).
    6. Quantum Sensitivity Advantage & Significance ($p < 0.05$).
    7. Clinical Explainability (SHAP waterfall demonstration).
    8. Scalability to Near-Term Hardware (NISQ roadmap).
  </action>
  <verify>
    Conduct timed rehearsal presentation with team (< 8 minutes).
  </verify>
  <done>
    High-impact, beautifully designed pitch deck ready for presentation.
  </done>
</task>
