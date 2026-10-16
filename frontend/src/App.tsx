import { useState, useEffect } from 'react';
import { Navigation } from './components/Navigation';
import type { TabId } from './components/Navigation';
import { TelemetryHUD } from './components/TelemetryHUD';
import { ArchitectureView } from './components/ArchitectureView';
import { RetinaViewer } from './components/RetinaViewer';
import { WholeBodyExplorer } from './components/WholeBodyExplorer';
import { MultiModalStudio } from './components/MultiModalStudio';
import { WhatIfSimulator } from './components/WhatIfSimulator';
import { NoiseStressTester } from './components/NoiseStressTester';
import { RadarArena } from './components/RadarArena';
import { HospitalDossier } from './components/HospitalDossier';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('arch');
  const [isDark, setIsDark] = useState<boolean>(true);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.setAttribute('data-theme', 'light');
    }
  }, [isDark]);

  return (
    <div className="main-content">
      {/* Top Clinical Navigation Bar with Theme Switcher */}
      <Navigation
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isDark={isDark}
        onToggleTheme={() => setIsDark(!isDark)}
      />

      {/* Persistent Quantum Simulator Telemetry HUD */}
      <TelemetryHUD nQubits={4} />

      {/* Main Tab View */}
      <main role="main">
        {activeTab === 'arch' && <ArchitectureView />}
        {activeTab === 'fundus' && <RetinaViewer />}
        {activeTab === 'body' && <WholeBodyExplorer />}
        {activeTab === 'multimodal' && <MultiModalStudio />}
        {activeTab === 'whatif' && <WhatIfSimulator />}
        {activeTab === 'noise' && <NoiseStressTester />}
        {activeTab === 'radar' && <RadarArena />}
        {activeTab === 'dossier' && <HospitalDossier />}
      </main>

      {/* Clinical Footer */}
      <footer
        style={{
          marginTop: 'var(--space-12)',
          paddingTop: 'var(--space-6)',
          borderTop: '1px solid var(--color-border-subtle)',
          textAlign: 'center',
          fontSize: 'var(--font-size-xs)',
          color: 'var(--color-text-tertiary)',
          display: 'flex',
          flexDirection: 'column',
          gap: 'var(--space-2)',
        }}
      >
        <div>
          QuantumHealth Clinical Platform • ISO 13485 Research Laboratory Protocol • DeepMind Antigravity
        </div>
        <div>
          PennyLane Quantum Engine (Statevector Simulator) • PyTorch 2.0 • Scikit-Learn • FastAPI REST Architecture
        </div>
      </footer>
    </div>
  );
};

export default App;
