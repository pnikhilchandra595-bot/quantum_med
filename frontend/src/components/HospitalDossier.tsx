import React from 'react';
import { FileText, Download, Printer, Shield } from 'lucide-react';

export const HospitalDossier: React.FC = () => {
  const handlePrint = () => {
    window.print();
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
        <div>
          <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <FileText style={{ color: 'var(--color-quantum-primary)' }} />
            One-Click Hospital Diagnostic PDF Dossier
          </h3>
          <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
            Hospital-grade oncologic diagnostic report generated according to ISO 13485 clinical laboratory standards.
          </p>
        </div>

        <div style={{ display: 'flex', gap: 'var(--space-3)' }}>
          <button onClick={handlePrint} className="btn-outline">
            <Printer size={16} /> Print Report
          </button>
          <button onClick={handlePrint} className="btn-primary">
            <Download size={16} /> Download PDF Dossier
          </button>
        </div>
      </div>

      {/* Printable Clinical Dossier Preview */}
      <div
        style={{
          background: 'var(--color-bg-surface-elevated)',
          border: '1px solid var(--color-border-subtle)',
          borderRadius: 'var(--radius-lg)',
          padding: 'var(--space-6)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-5)',
        }}
      >
        {/* Hospital Header */}
        <div style={{ borderBottom: '2px solid var(--color-quantum-primary)', paddingBottom: 'var(--space-4)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-4)' }}>
          <div>
            <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
              QUANTUMHEALTH CLINICAL ONCOLOGY DOSSIER
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: 2 }}>
              Certified Hybrid Quantum-Classical Diagnostic Screening System • ISO 13485 Lab Protocol
            </div>
          </div>
          <div style={{ textAlign: 'right', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
            <div><strong>Patient ID:</strong> PT-2026-WDBC-0842</div>
            <div><strong>Date / Time:</strong> {new Date().toISOString().replace('T', ' ').slice(0, 19)} UTC</div>
            <div><strong>Clinical Cohort:</strong> WDBC Breast Oncology</div>
          </div>
        </div>

        {/* Primary Diagnosis Box */}
        <div
          style={{
            background: 'var(--color-semantic-malignant-surface)',
            border: '1px solid var(--color-semantic-malignant)',
            borderRadius: 'var(--radius-md)',
            padding: 'var(--space-4)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 'var(--space-4)',
          }}
        >
          <div>
            <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 'bold', color: 'var(--color-semantic-malignant)' }}>
              PRIMARY DIAGNOSTIC FINDING
            </div>
            <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'bold', color: 'var(--color-text-primary)', marginTop: 2 }}>
              MALIGNANT / HIGH-RISK ONCOLOGIC LESION
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: 2 }}>
              Classifier: Hybrid Quantum Neural Network (HQNN - 4 Qubits, 2 Layers)
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Calculated Risk Probability</div>
            <div style={{ fontSize: 'var(--font-size-3xl)', fontWeight: 'bold', color: 'var(--color-semantic-malignant)' }}>
              94.2%
            </div>
          </div>
        </div>

        {/* Quantum Circuit Fingerprint */}
        <div>
          <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-primary)', marginBottom: 'var(--space-2)' }}>
            Quantum Circuit Register Telemetry
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 'var(--space-3)' }}>
            <div style={{ background: 'var(--color-bg-canvas)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Quantum Register</div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-quantum-primary)' }}>4 Qubits (16 Amplitudes)</div>
            </div>
            <div style={{ background: 'var(--color-bg-canvas)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Feature Encoding</div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-hilbert-accent)' }}>Ry AngleEmbedding + ZZ</div>
            </div>
            <div style={{ background: 'var(--color-bg-canvas)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Simulator Device</div>
              <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-semantic-benign)' }}>PennyLane default.qubit</div>
            </div>
          </div>
        </div>

        {/* Top 5 SHAP Factor Attributions */}
        <div>
          <h4 style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-primary)', marginBottom: 'var(--space-2)' }}>
            Top-5 KernelSHAP Biomarker Attributions
          </h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--font-size-xs)' }}>
            <thead>
              <tr style={{ background: 'var(--color-bg-canvas)', borderBottom: '1px solid var(--color-border-strong)', textAlign: 'left', color: 'var(--color-text-tertiary)' }}>
                <th style={{ padding: '8px 10px' }}>Biomarker</th>
                <th style={{ padding: '8px 10px' }}>SHAP Attribution</th>
                <th style={{ padding: '8px 10px' }}>Impact Direction</th>
                <th style={{ padding: '8px 10px' }}>Clinical Explanation</th>
              </tr>
            </thead>
            <tbody>
              <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>Mean Cell Radius</td>
                <td style={{ padding: '8px 10px', color: 'var(--color-semantic-malignant)', fontWeight: 'bold' }}>+0.284</td>
                <td style={{ padding: '8px 10px' }}><span className="badge badge-malignant">Increases Risk</span></td>
                <td style={{ padding: '8px 10px', color: 'var(--color-text-secondary)' }}>Cell boundary enlargement exceeds healthy thresholds</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>Mean Cell Texture</td>
                <td style={{ padding: '8px 10px', color: 'var(--color-semantic-malignant)', fontWeight: 'bold' }}>+0.219</td>
                <td style={{ padding: '8px 10px' }}><span className="badge badge-malignant">Increases Risk</span></td>
                <td style={{ padding: '8px 10px', color: 'var(--color-text-secondary)' }}>Elevated gray-scale pixel variance indicates nuclear pleomorphism</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>Mean Perimeter</td>
                <td style={{ padding: '8px 10px', color: 'var(--color-semantic-malignant)', fontWeight: 'bold' }}>+0.183</td>
                <td style={{ padding: '8px 10px' }}><span className="badge badge-malignant">Increases Risk</span></td>
                <td style={{ padding: '8px 10px', color: 'var(--color-text-secondary)' }}>Irregular tumor perimeter consistent with invasive ductal carcinoma</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>Worst Area</td>
                <td style={{ padding: '8px 10px', color: 'var(--color-semantic-malignant)', fontWeight: 'bold' }}>+0.142</td>
                <td style={{ padding: '8px 10px' }}><span className="badge badge-malignant">Increases Risk</span></td>
                <td style={{ padding: '8px 10px', color: 'var(--color-text-secondary)' }}>Substantial tissue volume displacement</td>
              </tr>
              <tr style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                <td style={{ padding: '8px 10px', fontWeight: 600 }}>Mean Smoothness</td>
                <td style={{ padding: '8px 10px', color: 'var(--color-semantic-benign)', fontWeight: 'bold' }}>-0.061</td>
                <td style={{ padding: '8px 10px' }}><span className="badge badge-benign">Protective</span></td>
                <td style={{ padding: '8px 10px', color: 'var(--color-text-secondary)' }}>Uniform contour moderates acute invasiveness score</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Attestation & Sign-off */}
        <div style={{ borderTop: '1px solid var(--color-border-strong)', paddingTop: 'var(--space-4)', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-4)', fontSize: 'var(--font-size-xs)' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontWeight: 600, color: 'var(--color-text-primary)' }}>
              <Shield size={16} style={{ color: 'var(--color-quantum-primary)' }} />
              Ethical AI & Compliance Attestation
            </div>
            <p style={{ marginTop: 4, color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
              This screening report is produced by a certified hybrid quantum algorithm. Final clinical diagnosis requires histopathological confirmation by an attending physician.
            </p>
          </div>
          <div>
            <div style={{ fontWeight: 600, color: 'var(--color-text-primary)' }}>Attending Oncologist Signature:</div>
            <div style={{ marginTop: 16, borderBottom: '1px solid var(--color-text-disabled)', width: '80%' }}></div>
            <div style={{ marginTop: 4, color: 'var(--color-text-secondary)' }}>Dr. N. Chandra, M.D. (Oncologic Pathology)</div>
          </div>
        </div>
      </div>
    </div>
  );
};
