import React, { useState } from 'react';
import { Zap } from 'lucide-react';

interface NoiseDataPoint {
  noise: number; // e.g. 0.0, 0.05, 0.1, 0.2, 0.3, 0.4
  quantumSens: number;
  classicalSens: number;
}

const NOISE_BENCHMARK: NoiseDataPoint[] = [
  { noise: 0.0, quantumSens: 94.1, classicalSens: 100.0 },
  { noise: 0.05, quantumSens: 100.0, classicalSens: 100.0 },
  { noise: 0.10, quantumSens: 100.0, classicalSens: 100.0 },
  { noise: 0.20, quantumSens: 94.1, classicalSens: 100.0 },
  { noise: 0.30, quantumSens: 100.0, classicalSens: 94.1 },
  { noise: 0.40, quantumSens: 94.1, classicalSens: 91.7 },
];

export const NoiseStressTester: React.FC = () => {
  const [selectedNoise, setSelectedNoise] = useState<number>(0.20);

  const currentPoint = NOISE_BENCHMARK.find((p) => Math.abs(p.noise - selectedNoise) < 0.01) || NOISE_BENCHMARK[3];

  // SVG dimensions for degradation curve
  const width = 600;
  const height = 260;
  const padL = 45;
  const padR = 20;
  const padT = 30;
  const padB = 40;

  // Map noise [0, 0.4] to X [padL, width - padR]
  const mapX = (noise: number) => padL + (noise / 0.4) * (width - padL - padR);
  // Map sensitivity [85, 102] to Y [height - padB, padT]
  const mapY = (sens: number) => height - padB - ((sens - 85) / 17) * (height - padT - padB);

  // Generate SVG path points
  const qPath = NOISE_BENCHMARK.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.noise)} ${mapY(p.quantumSens)}`).join(' ');
  const cPath = NOISE_BENCHMARK.map((p, i) => `${i === 0 ? 'M' : 'L'} ${mapX(p.noise)} ${mapY(p.classicalSens)}`).join(' ');

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      <div>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
          <Zap style={{ color: 'var(--color-quantum-primary)' }} />
          Biomedical Sensor Noise Robustness Stress-Tester
        </h3>
        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-5)' }}>
          Assesses diagnostic sensitivity retention across increasing levels of simulated Gaussian patient vital sensor telemetry noise.
        </p>

        {/* Noise Slider */}
        <div style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)', marginBottom: 'var(--space-6)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Injected Sensor Noise Level (σ):
            </span>
            <span className="badge badge-quantum" style={{ fontSize: 'var(--font-size-sm)' }}>
              {(selectedNoise * 100).toFixed(0)}% Gaussian Perturbation
            </span>
          </div>
          <input
            type="range"
            min={0.0}
            max={0.4}
            step={0.05}
            value={selectedNoise}
            onChange={(e) => setSelectedNoise(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
          />
        </div>

        {/* Live Delta Metrics Row */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 'var(--space-4)', marginBottom: 'var(--space-6)' }}>
          <div style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-quantum-glow)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Quantum QSVC Sensitivity</div>
            <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-quantum-primary)', marginTop: 2 }}>
              {currentPoint.quantumSens.toFixed(1)}%
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-semantic-benign)', marginTop: 4 }}>
              Maintains high recall under NISQ noise
            </div>
          </div>

          <div style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Classical SVM Sensitivity</div>
            <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-text-primary)', marginTop: 2 }}>
              {currentPoint.classicalSens.toFixed(1)}%
            </div>
            <div style={{ fontSize: 'var(--font-size-xs)', color: currentPoint.classicalSens < 95 ? 'var(--color-semantic-warning)' : 'var(--color-text-secondary)', marginTop: 4 }}>
              {currentPoint.classicalSens < 95 ? 'Degrading under heavy noise' : 'Stable at low noise'}
            </div>
          </div>
        </div>

        {/* Degradation Chart */}
        <div style={{ background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)', border: '1px solid var(--color-border-subtle)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
            <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
              Sensitivity Retention Curve across Noise Spectrum
            </span>
            <div style={{ display: 'flex', gap: 'var(--space-4)', fontSize: 'var(--font-size-xs)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--color-quantum-primary)' }}>
                <span style={{ width: 12, height: 3, background: 'var(--color-quantum-primary)', display: 'inline-block' }} /> Quantum QSVC
              </span>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: '#f43f5e' }}>
                <span style={{ width: 12, height: 3, background: '#f43f5e', display: 'inline-block' }} /> Classical SVM
              </span>
            </div>
          </div>

          <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', overflow: 'visible' }}>
            {/* Grid Lines */}
            {[90, 95, 100].map((v) => (
              <g key={v}>
                <line x1={padL} y1={mapY(v)} x2={width - padR} y2={mapY(v)} stroke="var(--color-border-subtle)" strokeWidth="1" strokeDasharray="3 3" />
                <text x={padL - 8} y={mapY(v) + 4} fill="var(--color-text-tertiary)" fontSize="10" textAnchor="end">{v}%</text>
              </g>
            ))}

            {/* X-axis ticks */}
            {NOISE_BENCHMARK.map((p) => (
              <g key={p.noise}>
                <line x1={mapX(p.noise)} y1={height - padB} x2={mapX(p.noise)} y2={height - padB + 5} stroke="var(--color-border-subtle)" strokeWidth="1" />
                <text x={mapX(p.noise)} y={height - padB + 18} fill="var(--color-text-tertiary)" fontSize="10" textAnchor="middle">
                  {(p.noise * 100).toFixed(0)}%
                </text>
              </g>
            ))}

            {/* Curves */}
            <path d={cPath} fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeDasharray="5 3" />
            <path d={qPath} fill="none" stroke="var(--color-quantum-primary)" strokeWidth="3" />

            {/* Points */}
            {NOISE_BENCHMARK.map((p) => (
              <g key={p.noise}>
                <circle cx={mapX(p.noise)} cy={mapY(p.classicalSens)} r="4" fill="#f43f5e" />
                <circle cx={mapX(p.noise)} cy={mapY(p.quantumSens)} r="4.5" fill="var(--color-quantum-primary)" />
              </g>
            ))}

            {/* Current Active Noise Indicator */}
            <line
              x1={mapX(selectedNoise)}
              y1={padT}
              x2={mapX(selectedNoise)}
              y2={height - padB}
              stroke="#fbbf24"
              strokeWidth="2"
              strokeDasharray="2 2"
            />
          </svg>
        </div>
      </div>
    </div>
  );
};
