import React from 'react';
import { Atom, Moon, Sun } from 'lucide-react';

export type TabId = 'arch' | 'fundus' | 'body' | 'bloch' | 'whatif' | 'noise' | 'radar' | 'dossier';

interface NavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  isDark: boolean;
  onToggleTheme: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  activeTab,
  onTabChange,
  isDark,
  onToggleTheme,
}) => {
  const tabs: { id: TabId; label: string; icon: string }[] = [
    { id: 'arch', label: 'Architecture & Overview', icon: '🏛️' },
    { id: 'fundus', label: 'Retinal Fundus (22 Photos)', icon: '👁️' },
    { id: 'body', label: 'Whole-Body Multi-Organ', icon: '🩺' },
    { id: 'bloch', label: '3D Bloch Sphere Studio', icon: '🌐' },
    { id: 'whatif', label: 'What-If Patient Simulator', icon: '🎛️' },
    { id: 'noise', label: 'Noise Stress-Tester', icon: '⚡' },
    { id: 'radar', label: '5-Axis Clinical Radar', icon: '🕸️' },
    { id: 'dossier', label: 'Hospital PDF Dossier', icon: '📄' },
  ];

  return (
    <nav
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 'var(--space-4)',
        paddingBottom: 'var(--space-4)',
        borderBottom: '1px solid var(--color-border-subtle)',
        marginBottom: 'var(--space-6)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)' }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 'var(--radius-md)',
            background: 'var(--color-quantum-surface)',
            border: '1px solid var(--color-quantum-glow)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'var(--color-quantum-primary)',
          }}
        >
          <Atom size={24} />
        </div>
        <div>
          <div style={{ fontSize: 'var(--font-size-lg)', fontWeight: 'var(--font-weight-bold)', color: 'var(--color-text-primary)' }}>
            QuantumHealth
          </div>
          <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
            Hybrid QML Clinical Intelligence Cockpit
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div
        role="tablist"
        style={{
          display: 'flex',
          gap: 'var(--space-2)',
          flexWrap: 'wrap',
          background: 'var(--color-bg-surface-elevated)',
          padding: '4px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--color-border-subtle)',
        }}
      >
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              role="tab"
              aria-selected={isActive}
              onClick={() => onTabChange(tab.id)}
              style={{
                minHeight: '36px',
                height: '36px',
                padding: '0 var(--space-3)',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: isActive ? 'var(--color-quantum-primary)' : 'transparent',
                color: isActive ? 'var(--color-text-inverse)' : 'var(--color-text-secondary)',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: 'var(--space-2)',
                fontSize: 'var(--font-size-xs)',
                fontWeight: isActive ? 600 : 500,
                transition: 'all var(--duration-fast) var(--ease-standard)',
              }}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Theme Toggle Button */}
      <button
        onClick={onToggleTheme}
        aria-label="Toggle color theme"
        className="btn-outline"
        style={{
          minWidth: 44,
          minHeight: 44,
          padding: 0,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 'var(--radius-md)',
        }}
      >
        {isDark ? <Sun size={18} style={{ color: '#fbbf24' }} /> : <Moon size={18} style={{ color: '#64748b' }} />}
      </button>
    </nav>
  );
};
