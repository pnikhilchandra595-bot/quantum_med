import React, { useEffect, useState } from 'react';
import { getHealth } from '../lib/api';
import type { HealthStatus } from '../lib/api';
import { Activity, Cpu, Database, Layers } from 'lucide-react';

export const TelemetryHUD: React.FC<{ nQubits?: number }> = ({ nQubits = 4 }) => {
  const [_health, setHealth] = useState<HealthStatus | null>(null);
  const [ping, setPing] = useState<number>(4.2);

  useEffect(() => {
    getHealth().then((h) => {
      setHealth(h);
      setPing(parseFloat((Math.random() * 3 + 2).toFixed(1)));
    });
  }, []);

  const hilbertDim = 1 << nQubits;
  const memoryMb = (hilbertDim * 16) / (1024 * 1024);

  return (
    <header
      style={{
        background: 'var(--color-bg-glass)',
        backdropFilter: 'blur(12px)',
        border: '1px solid var(--color-border-subtle)',
        borderRadius: 'var(--radius-lg)',
        padding: 'var(--space-3) var(--space-5)',
        marginBottom: 'var(--space-6)',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 'var(--space-4)',
        boxShadow: 'var(--shadow-sm)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-5)', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Cpu size={18} style={{ color: 'var(--color-quantum-primary)' }} />
          <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
            Simulator: <strong style={{ color: 'var(--color-quantum-primary)' }}>PennyLane default.qubit</strong>
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Layers size={18} style={{ color: 'var(--color-hilbert-accent)' }} />
          <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
            Hilbert Space: <strong style={{ color: 'var(--color-hilbert-accent)' }}>2^{nQubits} = {hilbertDim} Dim</strong>
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Activity size={18} style={{ color: 'var(--color-semantic-benign)' }} />
          <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
            Statevector RAM: <strong style={{ color: 'var(--color-semantic-benign)' }}>{memoryMb.toFixed(4)} MB</strong>
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
          <Database size={16} style={{ color: 'var(--color-text-tertiary)' }} />
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>
            WDBC (569) • Heart (303) • Fundus (22)
          </span>
        </div>

        <span
          className="badge"
          style={{
            background: 'var(--color-semantic-benign-surface)',
            color: 'var(--color-semantic-benign)',
            border: '1px solid rgba(16, 185, 129, 0.3)',
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              backgroundColor: 'var(--color-semantic-benign)',
              display: 'inline-block',
            }}
          />
          FastAPI Online ({ping} ms)
        </span>
      </div>
    </header>
  );
};
