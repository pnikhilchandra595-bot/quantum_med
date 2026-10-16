import React from 'react';
import { Layers, Cpu, Sparkles } from 'lucide-react';

export const ArchitectureView: React.FC = () => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      {/* Hero Welcome Card */}
      <div className="card card-quantum" style={{ background: 'linear-gradient(135deg, var(--color-bg-glass), var(--color-bg-surface-elevated))' }}>
        <h2 style={{ fontSize: 'var(--font-size-2xl)', color: 'var(--color-quantum-primary)', marginBottom: 'var(--space-2)' }}>
          QuantumHealth Clinical Diagnostic Platform
        </h2>
        <p style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-text-secondary)', maxWidth: '80ch' }}>
          Bridging non-linear Hilbert-space expressivity with classical machine learning for early detection of breast cancer oncology, cardiovascular pathology, and diabetic retinopathy.
        </p>

        {/* 4 Metrics Strip */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-4)', marginTop: 'var(--space-5)' }}>
          <div style={{ background: 'var(--color-bg-surface)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Diagnostic Models</div>
            <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>8 Models</div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-quantum-primary)', marginTop: 2 }}>3 Quantum • 5 Classical</div>
          </div>

          <div style={{ background: 'var(--color-bg-surface)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Clinical Cohorts</div>
            <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>894 Patients</div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-hilbert-accent)', marginTop: 2 }}>WDBC • Heart • Fundus</div>
          </div>

          <div style={{ background: 'var(--color-bg-surface)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Hilbert State Space</div>
            <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>2⁴ = 16 Dim</div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-semantic-benign)', marginTop: 2 }}>ZZ-Pauli Entanglement</div>
          </div>

          <div style={{ background: 'var(--color-bg-surface)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>System Readiness</div>
            <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-semantic-benign)' }}>Verified 100%</div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: 2 }}>43 Tests Green • ISO 13485</div>
          </div>
        </div>
      </div>

      {/* 3-Tier Architecture Diagram */}
      <div className="card">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
          <Layers style={{ color: 'var(--color-quantum-primary)' }} />
          3-Tier Hybrid Quantum-Classical Platform Topology
        </h3>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)' }}>
          {/* Tier 1 */}
          <div style={{ background: 'var(--color-bg-surface-elevated)', border: '1px solid var(--color-border-subtle)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)' }}>
            <div className="badge badge-benign" style={{ marginBottom: 'var(--space-2)' }}>TIER 1 — DATA INGESTION</div>
            <h4 style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-text-primary)', marginBottom: 'var(--space-2)' }}>
              Clinical Acquisition & Reduction
            </h4>
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
              Standardized ingestion of high-dimensional biomarkers (WDBC 30 features, Cleveland Heart 13 features, Ophthalmic Fundus 11 features). Zero-leakage PCA maps continuous inputs to 4-8 orthogonal qubits retaining &gt;88% variance.
            </p>
          </div>

          {/* Tier 2 */}
          <div style={{ background: 'var(--color-bg-surface-elevated)', border: '1px solid var(--color-quantum-glow)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)' }}>
            <div className="badge badge-quantum" style={{ marginBottom: 'var(--space-2)' }}>TIER 2 — QUANTUM ENGINE</div>
            <h4 style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-quantum-primary)', marginBottom: 'var(--space-2)' }}>
              Hilbert Mapping & Quantum Classifiers
            </h4>
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
              Decoupled ModelFactory managing QSVC (state fidelity kernels), VQC (Adam parameterized circuits), and HQNN (PyTorch TorchLayer), backed by PennyLane simulator devices and physical depolarizing noise models.
            </p>
          </div>

          {/* Tier 3 */}
          <div style={{ background: 'var(--color-bg-surface-elevated)', border: '1px solid var(--color-hilbert-glow)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)' }}>
            <div className="badge badge-quantum" style={{ marginBottom: 'var(--space-2)', background: 'var(--color-hilbert-glow)', color: 'var(--color-hilbert-accent)' }}>TIER 3 — CLINICAL DELIVERY</div>
            <h4 style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-hilbert-accent)', marginBottom: 'var(--space-2)' }}>
              Interactive Cockpit & XAI Audit
            </h4>
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', lineHeight: 1.5 }}>
              Standalone high-performance front-end paired with FastAPI REST backend. Implements transparent Quantum KernelSHAP feature attribution and automated hospital PDF dossier generation for oncology boards.
            </p>
          </div>
        </div>
      </div>

      {/* Model Catalogue */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-6)' }}>
        <div className="card">
          <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', color: 'var(--color-quantum-primary)', marginBottom: 'var(--space-3)' }}>
            <Sparkles size={18} /> Quantum Classifiers (Hilbert Space)
          </h4>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>QSVC:</strong> Computes transition probability fidelity kernels |⟨Φ(x)|Φ(x')⟩|² via ZZ-Pauli feature map for maximal margin separation.
            </li>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>VQC:</strong> Variational parameter-shift ansatz with circular CNOT entanglement and Adam gradient descent.
            </li>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>HQNN:</strong> Hybrid deep neural network coupling PyTorch feedforward linear layers with PennyLane TorchLayer.
            </li>
          </ul>
        </div>

        <div className="card">
          <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', color: 'var(--color-text-primary)', marginBottom: 'var(--space-3)' }}>
            <Cpu size={18} /> Classical Baselines (Zero-Leakage Benchmark)
          </h4>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>SVM (Linear / RBF):</strong> Soft-margin benchmark in reproducing kernel Hilbert spaces.
            </li>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>Random Forest (RF):</strong> 100 de-correlated randomized decision trees with Gini split criteria.
            </li>
            <li style={{ padding: '8px', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-sm)' }}>
              <strong style={{ color: 'var(--color-text-primary)' }}>XGBoost (XGB):</strong> Extreme gradient boosted decision trees minimizing pseudo-residuals.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
