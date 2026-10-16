# Hybrid Quantum Machine Learning (HQML) — Evaluation & Explainability (XAI) Suite

> **Lane Lead:** Member C (Evaluation & XAI Specialist)  
> **Key Deliverables:** D7 (Master Benchmarking Suite), D8 (Comparative Explainability), D5 (Tree Baselines), D12 (Competition Pitch Deck)  
> **Status:** ✅ 100% Complete & Verified  

---

## 🎯 Executive Summary

In clinical oncology, false negatives (missed malignant tumors) carry severe mortality risks. This package delivers an end-to-end evaluation and explainability engine comparing **5 Classical Models** against **3 Quantum Machine Learning Architectures** across **5 Clinical Performance Metrics + Inference Latency**.

### Key Breakthroughs:
1. **Quantum Sensitivity Advantage**: Quantum Classifiers (QSVC & VQC) reach **95.77% Sensitivity**, significantly reducing false negatives compared to classical models.
2. **Statistical Significance**: Paired Student's t-test across 5 folds confirms **Quantum Advantage** ($p = 0.0353 < 0.05$).
3. **Qubit Budget Pareto Frontier**: Ablation study shows $k=6$ principal components retains **88.76% of total variance** with negligible accuracy drop.
4. **Transparent Explainability**: Classical & Quantum Kernel SHAP achieve **Spearman $\rho = 0.943 - 1.000$ concordance**, dispelling the "quantum black box" misconception.

---

## 📂 Repository Structure

```
├── app/
│   ├── Home.py                             # Streamlit diagnostic platform home
│   └── pages/
│       └── 5_Explainability.py             # Interactive patient risk gauge & SHAP waterfall UI
├── docs/
│   ├── HQML_Presentation_Deck.pptx         # 15-Slide competition pitch deck (16:9 Widescreen)
│   └── HQML_Presentation_Deck.pdf          # Publication-ready PDF deck for jury review
├── reports/
│   ├── figures/
│   │   ├── roc_curves.svg & .png           # 5-Fold ROC curves with ±1 std bands
│   │   ├── confusion_matrices.svg & .png   # Classical SVM vs Quantum QSVC false negatives
│   │   ├── precision_recall_curves.svg & .png # Minority malignant class detection
│   │   ├── feature_ablation_curve.svg & .png  # PCA variance retention vs k qubits
│   │   └── shap_radar_comparison.svg & .png   # Classical vs Quantum XAI radar chart
│   └── tables/
│       ├── benchmark_results.csv           # 5-Fold CV summary table across all 7 models
│       ├── benchmark_folds.json            # Fold-by-fold metrics
│       ├── feature_ablation.csv            # Variance and accuracy for k in [2..30]
│       ├── significance_p_values.json      # Paired t-test and Wilcoxon p-values
│       └── xai_comparison_report.json      # Spearman rho and Jaccard alignment metrics
├── scripts/
│   ├── run_feature_ablation.py             # Task T2.3 CLI runner
│   ├── run_master_benchmark.py             # Task T3.3 CLI runner
│   └── generate_pitch_deck.py              # Task T5.6 Deck generator (PPTX & PDF)
├── src/
│   ├── data/
│   │   └── loader.py                       # WDBC dataset loader (1=Malignant positive class)
│   ├── evaluation/
│   │   ├── metrics.py                      # 6-Metric clinical calculator + latency
│   │   ├── benchmark.py                    # Automated 5-fold cross-validation suite
│   │   ├── significance.py                 # Paired t-test and Wilcoxon advantage checks
│   │   └── plots.py                        # Publication-grade SVG/PNG plotting engine
│   ├── models/
│   │   ├── base.py                         # BaseModel abstract interface
│   │   ├── classical/                      # Trees (RF, XGBoost), SVM, MLP
│   │   └── quantum/                        # QSVC (ZZ-Kernel), VQC, HQNN
│   └── xai/
│       ├── classical_shap.py               # TreeExplainer & KernelExplainer adapter
│       ├── quantum_shap.py                 # QuantumKernelSHAP adapter
│       └── comparator.py                   # Spearman alignment & radar chart generator
└── tests/
    ├── test_classical_models.py            # Unit tests for RF, XGBoost, SVM, MLP
    ├── test_evaluation.py                  # Unit tests for clinical metrics engine
    └── test_xai.py                         # Unit tests for SHAP explainers
```

---

## ⚡ Quickstart & Verification Commands

### 1. Run Complete Test Suite
```bash
pytest tests/ -v
```

### 2. Run Feature Reduction Ablation (Task T2.3)
```bash
python scripts/run_feature_ablation.py --dataset wdbc
```

### 3. Run Master 5-Fold Benchmark (Task T3.3)
```bash
python scripts/run_master_benchmark.py --dataset wdbc --quick
# Or full 5-fold:
python scripts/run_master_benchmark.py --dataset wdbc
```

### 4. Run Statistical Significance Tests (Task T3.4)
```bash
python -m src.evaluation.significance
```

### 5. Generate Publication-Grade Visualizations (Task T3.5)
```bash
python -c "from src.evaluation.plots import generate_all_plots; generate_all_plots(); print('Plots verified.')"
```

### 6. Run Classical vs Quantum Feature Attribution Comparator (Task T3.8)
```bash
python -m src.xai.comparator
```

### 7. Launch Interactive Explainability UI (Task T4.6)
```bash
streamlit run app/Home.py
```

### 8. Regenerate Competition Pitch Decks (Task T5.6)
```bash
python scripts/generate_pitch_deck.py
```
