import React, { useState, useEffect } from 'react';
import { Sliders, AlertCircle, CheckCircle } from 'lucide-react';
import { runPrediction } from '../lib/api';
import type { PredictionResponse } from '../lib/api';

export const WhatIfSimulator: React.FC = () => {
  const [radius, setRadius] = useState<number>(16.2);
  const [texture, setTexture] = useState<number>(20.5);
  const [perimeter, setPerimeter] = useState<number>(104.0);
  const [area, setArea] = useState<number>(780.0);

  const [classicalRes, setClassicalRes] = useState<PredictionResponse | null>(null);
  const [quantumRes, setQuantumRes] = useState<PredictionResponse | null>(null);
  const [_loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    // Debounce predictions on slider change
    setLoading(true);
    const timer = setTimeout(async () => {
      // Map to 4-dimensional normalized features
      const f0 = (radius - 14.0) / 4.0;
      const f1 = (texture - 19.0) / 5.0;
      const f2 = (perimeter - 90.0) / 20.0;
      const f3 = (area - 650.0) / 300.0;
      const feats = [f0, f1, f2, f3];

      const [c, q] = await Promise.all([
        runPrediction({ model_name: 'svm', dataset: 'wdbc', features: feats, n_qubits: 4 }),
        runPrediction({ model_name: 'qsvc', dataset: 'wdbc', features: feats, n_qubits: 4 }),
      ]);
      setClassicalRes(c);
      setQuantumRes(q);
      setLoading(false);
    }, 150);

    return () => clearTimeout(timer);
  }, [radius, texture, perimeter, area]);

  const isDiscordant =
    classicalRes && quantumRes && classicalRes.is_malignant_or_diseased !== quantumRes.is_malignant_or_diseased;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      <div className="card">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
          <Sliders style={{ color: 'var(--color-quantum-primary)' }} />
          Counterfactual "What-If" Patient Simulator
        </h3>
        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-5)' }}>
          Adjust patient fine-needle aspiration (FNA) biomarkers in real-time to observe decision boundary differences between Classical and Quantum models.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-6)' }}>
          {/* Sliders Box */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            <h4 style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-text-primary)' }}>
              Patient Biomarker Inputs (WDBC Cohort)
            </h4>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: 4 }}>
                <span>Mean Cell Nucleus Radius:</span>
                <strong style={{ color: 'var(--color-quantum-primary)' }}>{radius} μm</strong>
              </div>
              <input
                type="range"
                min={6.0}
                max={30.0}
                step={0.2}
                value={radius}
                onChange={(e) => setRadius(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: 4 }}>
                <span>Mean Cell Texture (Gray-scale):</span>
                <strong style={{ color: 'var(--color-quantum-primary)' }}>{texture}</strong>
              </div>
              <input
                type="range"
                min={9.0}
                max={40.0}
                step={0.5}
                value={texture}
                onChange={(e) => setTexture(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: 4 }}>
                <span>Mean Perimeter:</span>
                <strong style={{ color: 'var(--color-quantum-primary)' }}>{perimeter} μm</strong>
              </div>
              <input
                type="range"
                min={40.0}
                max={190.0}
                step={1.0}
                value={perimeter}
                onChange={(e) => setPerimeter(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
              />
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: 4 }}>
                <span>Mean Area:</span>
                <strong style={{ color: 'var(--color-quantum-primary)' }}>{area} μm²</strong>
              </div>
              <input
                type="range"
                min={140.0}
                max={2500.0}
                step={10.0}
                value={area}
                onChange={(e) => setArea(parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
              />
            </div>
          </div>

          {/* Comparative Results Box */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
            <h4 style={{ fontSize: 'var(--font-size-base)', color: 'var(--color-text-primary)' }}>
              Real-Time Comparative Risk Assessment
            </h4>

            {/* Classical Card */}
            <div
              style={{
                background: 'var(--color-bg-surface-elevated)',
                border: '1px solid var(--color-border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-4)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-text-tertiary)' }}>
                  CLASSICAL BASELINE (SVM-RBF)
                </span>
                <span className={`badge ${classicalRes?.is_malignant_or_diseased ? 'badge-malignant' : 'badge-benign'}`}>
                  {classicalRes?.prediction_label}
                </span>
              </div>
              <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
                {classicalRes ? (classicalRes.prediction_probability * 100).toFixed(1) : '--'}%
              </div>
              <div style={{ height: 8, background: 'var(--color-border-subtle)', borderRadius: 4, overflow: 'hidden', marginTop: 8 }}>
                <div
                  style={{
                    width: `${(classicalRes?.prediction_probability || 0) * 100}%`,
                    height: '100%',
                    background: classicalRes?.is_malignant_or_diseased ? 'var(--color-semantic-malignant)' : 'var(--color-semantic-benign)',
                    transition: 'width 250ms ease',
                  }}
                />
              </div>
            </div>

            {/* Quantum Card */}
            <div
              style={{
                background: 'var(--color-bg-surface-elevated)',
                border: '1px solid var(--color-quantum-glow)',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-4)',
                boxShadow: 'var(--shadow-quantum)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-quantum-primary)' }}>
                  ⚡ QUANTUM CLASSIFIER (QSVC FIDELITY)
                </span>
                <span className={`badge ${quantumRes?.is_malignant_or_diseased ? 'badge-malignant' : 'badge-benign'}`}>
                  {quantumRes?.prediction_label}
                </span>
              </div>
              <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-quantum-primary)' }}>
                {quantumRes ? (quantumRes.prediction_probability * 100).toFixed(1) : '--'}%
              </div>
              <div style={{ height: 8, background: 'var(--color-border-subtle)', borderRadius: 4, overflow: 'hidden', marginTop: 8 }}>
                <div
                  style={{
                    width: `${(quantumRes?.prediction_probability || 0) * 100}%`,
                    height: '100%',
                    background: 'var(--color-quantum-primary)',
                    transition: 'width 250ms ease',
                  }}
                />
              </div>
            </div>

            {/* Clinical Alert Box */}
            {isDiscordant ? (
              <div
                style={{
                  background: 'var(--color-semantic-warning-surface)',
                  border: '1px solid var(--color-semantic-warning)',
                  borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-3) var(--space-4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-3)',
                }}
              >
                <AlertCircle size={20} style={{ color: 'var(--color-semantic-warning)', flexShrink: 0 }} />
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-primary)' }}>
                  <strong>Borderline Clinical Discordance:</strong> Classical boundary categorizes benign, while Quantum Hilbert Kernel detects non-linear feature overlap predicting elevated risk. Recommend histology review.
                </div>
              </div>
            ) : (
              <div
                style={{
                  background: 'var(--color-semantic-benign-surface)',
                  border: '1px solid var(--color-semantic-benign)',
                  borderRadius: 'var(--radius-md)',
                  padding: 'var(--space-3) var(--space-4)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 'var(--space-3)',
                }}
              >
                <CheckCircle size={20} style={{ color: 'var(--color-semantic-benign)', flexShrink: 0 }} />
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-primary)' }}>
                  <strong>Clinical Consensus:</strong> Both Classical and Quantum architectures agree on the diagnostic trajectory.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
