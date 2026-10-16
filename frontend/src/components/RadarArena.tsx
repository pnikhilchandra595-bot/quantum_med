import React, { useState, useEffect } from 'react';
import { Target, CheckSquare, Square } from 'lucide-react';
import { getBenchmarkSummary } from '../lib/api';
import type { BenchmarkRecord } from '../lib/api';

const AXES = [
  'Sensitivity (Recall)',
  'Specificity',
  'ROC-AUC',
  'F1-Score',
  'Speed (1/Latency)',
];

const COLORS: Record<string, string> = {
  HQNN: '#38bdf8',
  QSVC: '#818cf8',
  VQC: '#a855f7',
  'RBF SVM': '#10b981',
  'Random Forest': '#fbbf24',
  XGBoost: '#f43f5e',
  'Linear SVM': '#06b6d4',
  MLP: '#ec4899',
};

export const RadarArena: React.FC = () => {
  const [data, setData] = useState<BenchmarkRecord[]>([]);
  const [selected, setSelected] = useState<string[]>(['HQNN', 'QSVC', 'RBF SVM', 'Random Forest']);

  useEffect(() => {
    getBenchmarkSummary().then(setData);
  }, []);

  const toggleModel = (name: string) => {
    setSelected((prev) =>
      prev.includes(name) ? prev.filter((m) => m !== name) : [...prev, name]
    );
  };

  // SVG Radar Settings
  const size = 360;
  const center = size / 2;
  const radius = size * 0.38;
  const numAxes = AXES.length;

  // Calculate polygon points for each model
  const getPoints = (record: BenchmarkRecord) => {
    const sens = record.sensitivity_mean * 100;
    const spec = record.specificity_mean * 100;
    const auc = record.roc_auc_mean * 100;
    const f1 = record.f1_score_mean * 100;
    const speed = Math.max(70, Math.min(100, 100 - record.latency_ms_mean * 150));

    const vals = [sens, spec, auc, f1, speed];

    return vals.map((v, i) => {
      const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
      // Map value [70, 100] to distance [0, radius]
      const dist = Math.max(0, (v - 70) / 30) * radius;
      const x = center + dist * Math.cos(angle);
      const y = center + dist * Math.sin(angle);
      return `${x},${y}`;
    }).join(' ');
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      <div>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
          <Target style={{ color: 'var(--color-quantum-primary)' }} />
          5-Axis Clinical Radar Comparison Arena
        </h3>
        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-5)' }}>
          Head-to-head architectural evaluation across clinical diagnostic sensitivity, specificity, discrimination, and computational efficiency.
        </p>

        {/* Model Filter Pills */}
        <div style={{ display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap', marginBottom: 'var(--space-6)' }}>
          {data.map((m) => {
            const isChecked = selected.includes(m.Model);
            const col = COLORS[m.Model] || '#94a3b8';
            return (
              <button
                key={m.Model}
                onClick={() => toggleModel(m.Model)}
                className="btn-outline"
                style={{
                  minHeight: 36,
                  height: 36,
                  borderColor: isChecked ? col : 'var(--color-border-subtle)',
                  background: isChecked ? `${col}15` : 'transparent',
                  color: isChecked ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: 6,
                  fontSize: 'var(--font-size-xs)',
                }}
              >
                {isChecked ? <CheckSquare size={14} style={{ color: col }} /> : <Square size={14} />}
                <span>{m.Model} ({m.Type})</span>
              </button>
            );
          })}
        </div>

        {/* Radar SVG and Table Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-6)', alignItems: 'center' }}>
          {/* Radar Chart */}
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: 360, height: 'auto', overflow: 'visible' }}>
              {/* Concentric Web Rings */}
              {[0.33, 0.66, 1.0].map((ring, idx) => {
                const ringPoints = Array.from({ length: numAxes }).map((_, i) => {
                  const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
                  const x = center + radius * ring * Math.cos(angle);
                  const y = center + radius * ring * Math.sin(angle);
                  return `${x},${y}`;
                }).join(' ');
                return (
                  <polygon
                    key={idx}
                    points={ringPoints}
                    fill="none"
                    stroke="var(--color-border-subtle)"
                    strokeWidth="1"
                    strokeDasharray="3 3"
                  />
                );
              })}

              {/* Axis Spokes */}
              {AXES.map((label, i) => {
                const angle = (i * 2 * Math.PI) / numAxes - Math.PI / 2;
                const x = center + radius * Math.cos(angle);
                const y = center + radius * Math.sin(angle);
                const textX = center + (radius + 22) * Math.cos(angle);
                const textY = center + (radius + 18) * Math.sin(angle);
                return (
                  <g key={i}>
                    <line x1={center} y1={center} x2={x} y2={y} stroke="var(--color-border-subtle)" strokeWidth="1" />
                    <text
                      x={textX}
                      y={textY}
                      fill="var(--color-text-secondary)"
                      fontSize="9"
                      fontWeight="600"
                      textAnchor="middle"
                      dominantBaseline="central"
                    >
                      {label}
                    </text>
                  </g>
                );
              })}

              {/* Model Polygons */}
              {data
                .filter((m) => selected.includes(m.Model))
                .map((m) => {
                  const pts = getPoints(m);
                  const col = COLORS[m.Model] || '#38bdf8';
                  return (
                    <polygon
                      key={m.Model}
                      points={pts}
                      fill={`${col}20`}
                      stroke={col}
                      strokeWidth="2"
                    />
                  );
                })}
            </svg>
          </div>

          {/* Metrics Table */}
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--font-size-xs)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--color-border-strong)', textAlign: 'left', color: 'var(--color-text-tertiary)' }}>
                  <th style={{ padding: '8px 10px' }}>Model</th>
                  <th style={{ padding: '8px 10px' }}>Accuracy</th>
                  <th style={{ padding: '8px 10px' }}>Sensitivity</th>
                  <th style={{ padding: '8px 10px' }}>Specificity</th>
                  <th style={{ padding: '8px 10px' }}>ROC-AUC</th>
                </tr>
              </thead>
              <tbody>
                {data
                  .filter((m) => selected.includes(m.Model))
                  .map((m) => {
                    const col = COLORS[m.Model] || '#38bdf8';
                    return (
                      <tr key={m.Model} style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                        <td style={{ padding: '8px 10px', fontWeight: 600, color: col }}>
                          {m.Model}
                        </td>
                        <td style={{ padding: '8px 10px' }}>{(m.accuracy_mean * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 10px', color: 'var(--color-semantic-benign)', fontWeight: 600 }}>
                          {(m.sensitivity_mean * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '8px 10px' }}>{(m.specificity_mean * 100).toFixed(1)}%</td>
                        <td style={{ padding: '8px 10px' }}>{(m.roc_auc_mean * 100).toFixed(1)}%</td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
