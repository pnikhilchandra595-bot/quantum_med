import React, { useState } from 'react';
import { Compass, Sparkles } from 'lucide-react';

export const BlochSphere3D: React.FC = () => {
  const [theta, setTheta] = useState<number>(1.05); // radians (~60 deg)
  const [phi, setPhi] = useState<number>(0.785);  // radians (~45 deg)
  const [activeQubit, setActiveQubit] = useState<number>(0);

  // Spherical to 2D isometric projection
  // Radius R = 110 px
  const R = 110;
  const cx = 160;
  const cy = 160;

  // 3D coordinates on unit sphere
  const x = Math.sin(theta) * Math.cos(phi);
  const y = Math.sin(theta) * Math.sin(phi);
  const z = Math.cos(theta);

  // Simple orthographic isometric projection:
  // proj_x = cx + R * (x * cos(30 deg) - y * cos(30 deg))
  // proj_y = cy - R * (z - (x + y) * 0.3)
  const px = cx + R * (x * 0.866 - y * 0.5);
  const py = cy - R * (z * 0.85 + (x * 0.3 + y * 0.3));

  const p0 = Math.cos(theta / 2) ** 2;
  const p1 = Math.sin(theta / 2) ** 2;

  return (
    <div className="card" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-6)' }}>
      {/* Controls Column */}
      <div>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-2)' }}>
          <Compass style={{ color: 'var(--color-quantum-primary)' }} />
          3D Bloch Sphere Studio
        </h3>
        <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginBottom: 'var(--space-5)' }}>
          Rotational statevector geometric projection in the complex projective Hilbert space (SU(2)).
        </p>

        <div style={{ marginBottom: 'var(--space-4)' }}>
          <label style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-secondary)', display: 'block', marginBottom: 'var(--space-2)' }}>
            Target Qubit Register:
          </label>
          <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
            {[0, 1, 2, 3].map((q) => (
              <button
                key={q}
                onClick={() => setActiveQubit(q)}
                className={activeQubit === q ? 'btn-primary' : 'btn-outline'}
                style={{ flex: 1, minHeight: 38 }}
              >
                q[{q}]
              </button>
            ))}
          </div>
        </div>

        {/* Theta Slider */}
        <div style={{ marginBottom: 'var(--space-4)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: '4px' }}>
            <span style={{ color: 'var(--color-text-secondary)' }}>Polar Angle θ (Feature Map Ry):</span>
            <strong style={{ color: 'var(--color-quantum-primary)' }}>{theta.toFixed(2)} rad ({(theta * 180 / Math.PI).toFixed(1)}°)</strong>
          </div>
          <input
            type="range"
            min={0}
            max={Math.PI}
            step={0.02}
            value={theta}
            onChange={(e) => setTheta(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-quantum-primary)' }}
          />
        </div>

        {/* Phi Slider */}
        <div style={{ marginBottom: 'var(--space-5)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-xs)', marginBottom: '4px' }}>
            <span style={{ color: 'var(--color-text-secondary)' }}>Azimuthal Phase φ (Entangling Rz):</span>
            <strong style={{ color: 'var(--color-hilbert-accent)' }}>{phi.toFixed(2)} rad ({(phi * 180 / Math.PI).toFixed(1)}°)</strong>
          </div>
          <input
            type="range"
            min={0}
            max={2 * Math.PI}
            step={0.02}
            value={phi}
            onChange={(e) => setPhi(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--color-hilbert-accent)' }}
          />
        </div>

        {/* Measurement Probabilities */}
        <div style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)' }}>
          <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-text-tertiary)', marginBottom: 'var(--space-2)' }}>
            PROJECTIVE MEASUREMENT PROBABILITIES
          </div>
          <div style={{ marginBottom: 'var(--space-3)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-sm)', marginBottom: '4px' }}>
              <span>P(|0⟩ Ground State):</span>
              <strong style={{ color: 'var(--color-semantic-benign)' }}>{(p0 * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ height: 6, background: 'var(--color-border-subtle)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${p0 * 100}%`, height: '100%', background: 'var(--color-semantic-benign)' }} />
            </div>
          </div>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-size-sm)', marginBottom: '4px' }}>
              <span>P(|1⟩ Excited State):</span>
              <strong style={{ color: 'var(--color-semantic-malignant)' }}>{(p1 * 100).toFixed(1)}%</strong>
            </div>
            <div style={{ height: 6, background: 'var(--color-border-subtle)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ width: `${p1 * 100}%`, height: '100%', background: 'var(--color-semantic-malignant)' }} />
            </div>
          </div>
        </div>
      </div>

      {/* 3D Sphere SVG Canvas */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <svg viewBox="0 0 320 320" style={{ width: '100%', maxWidth: 360, height: 'auto', overflow: 'visible' }}>
          {/* Outer Sphere Silhouette */}
          <circle cx={cx} cy={cy} r={R} fill="var(--color-quantum-surface)" stroke="var(--color-border-strong)" strokeWidth="1.5" />
          
          {/* Equator Ellipse */}
          <ellipse cx={cx} cy={cy} rx={R} ry={R * 0.3} fill="none" stroke="var(--color-border-subtle)" strokeWidth="1.5" strokeDasharray="4 4" />
          
          {/* Vertical Meridian */}
          <ellipse cx={cx} cy={cy} rx={R * 0.3} ry={R} fill="none" stroke="var(--color-border-subtle)" strokeWidth="1.5" strokeDasharray="4 4" />

          {/* Coordinate Axes */}
          {/* Z Axis (North-South) */}
          <line x1={cx} y1={cy - R - 20} x2={cx} y2={cy + R + 20} stroke="var(--color-text-disabled)" strokeWidth="1.5" />
          {/* X Axis */}
          <line x1={cx - R - 15} y1={cy + 25} x2={cx + R + 15} y2={cy - 25} stroke="var(--color-text-disabled)" strokeWidth="1.5" />

          {/* Basis State Labels */}
          <text x={cx} y={cy - R - 26} fill="var(--color-semantic-benign)" fontSize="13" fontWeight="bold" textAnchor="middle">|0⟩</text>
          <text x={cx} y={cy + R + 34} fill="var(--color-semantic-malignant)" fontSize="13" fontWeight="bold" textAnchor="middle">|1⟩</text>
          <text x={cx + R + 24} y={cy - 22} fill="var(--color-quantum-primary)" fontSize="12" fontWeight="bold" textAnchor="middle">|+⟩</text>
          <text x={cx - R - 24} y={cy + 30} fill="var(--color-text-secondary)" fontSize="12" fontWeight="bold" textAnchor="middle">|-⟩</text>

          {/* State Vector Line |ψ⟩ */}
          <line x1={cx} y1={cy} x2={px} y2={py} stroke="#fbbf24" strokeWidth="4" strokeLinecap="round" />
          
          {/* Tip Arrow Marker */}
          <circle cx={px} cy={py} r="6" fill="#fbbf24" stroke="#fff" strokeWidth="2" />
        </svg>

        {/* State Equation Monospace Box */}
        <div style={{ marginTop: 'var(--space-2)', padding: 'var(--space-2) var(--space-4)', background: 'var(--color-bg-surface-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--color-border-subtle)', fontFamily: 'var(--font-mono)', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
          <Sparkles size={14} style={{ display: 'inline', color: '#fbbf24', marginRight: 6 }} />
          |ψ⟩ = {Math.cos(theta/2).toFixed(3)}|0⟩ + {(Math.sin(theta/2)*Math.cos(phi)).toFixed(3)}e^(i·{phi.toFixed(2)})|1⟩
        </div>
      </div>
    </div>
  );
};
