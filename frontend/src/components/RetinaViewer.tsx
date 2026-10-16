import React, { useState } from 'react';
import { Eye, Sliders, CheckCircle2, AlertTriangle } from 'lucide-react';

interface FundusSample {
  id: string;
  name: string;
  diagnosis: 'Normal / Healthy' | 'Diabetic Retinopathy';
  isPositive: boolean;
  biomarkers: { name: string; value: number; unit: string; normal: string }[];
}

const SAMPLES: FundusSample[] = [
  {
    id: 'f1',
    name: 'Retina_Patient_01_DR.jpg',
    diagnosis: 'Diabetic Retinopathy',
    isPositive: true,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.412, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.884, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.128, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.082, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.48, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.76, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.18, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.34, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f2',
    name: 'Retina_Patient_04_Healthy.jpg',
    diagnosis: 'Normal / Healthy',
    isPositive: false,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.542, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.612, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.284, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.048, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.14, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.49, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.19, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.05, unit: 'index', normal: '< 1.15' },
    ],
  },
];

export const RetinaViewer: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState<FundusSample>(SAMPLES[0]);
  const [opacity, setOpacity] = useState<number>(50);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <div>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Eye style={{ color: 'var(--color-quantum-primary)' }} />
              Clinical Ophthalmic Fundus & Lesion Saliency Inspector
            </h3>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              High-resolution ocular fundus exploration with localized vascular & microaneurysm attention overlays.
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
            <select
              value={selectedSample.id}
              onChange={(e) => {
                const s = SAMPLES.find((item) => item.id === e.target.value);
                if (s) setSelectedSample(s);
              }}
              style={{
                background: 'var(--color-bg-surface)',
                color: 'var(--color-text-primary)',
                border: '1px solid var(--color-border-strong)',
                padding: '0 var(--space-3)',
                borderRadius: 'var(--radius-md)',
              }}
            >
              {SAMPLES.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.diagnosis})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Heatmap Opacity Slider */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-3) var(--space-4)', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-5)' }}>
          <Sliders size={18} style={{ color: 'var(--color-quantum-primary)' }} />
          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-secondary)', minWidth: 200 }}>
            Lesion Attention Opacity: {opacity}%
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={opacity}
            onChange={(e) => setOpacity(parseInt(e.target.value, 10))}
            style={{ flex: 1, accentColor: 'var(--color-quantum-primary)' }}
          />
        </div>

        {/* Side by Side Image Container */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-5)' }}>
          {/* Original View */}
          <div style={{ background: '#000', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--color-border-subtle)', position: 'relative' }}>
            <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(0,0,0,0.7)', color: '#fff', padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 'var(--font-size-xs)', zIndex: 2 }}>
              Original Clinical Fundus
            </div>
            <svg viewBox="0 0 400 300" style={{ width: '100%', height: 'auto', display: 'block' }}>
              <circle cx="200" cy="150" r="130" fill="#993311" />
              <circle cx="200" cy="150" r="125" fill="#bb4422" />
              {/* Optic Disc */}
              <circle cx="140" cy="145" r="30" fill="#ffddaa" opacity="0.85" />
              {/* Blood Vessels */}
              <path d="M 140 145 Q 180 110, 260 80" stroke="#771100" strokeWidth="5" fill="none" />
              <path d="M 140 145 Q 190 180, 270 210" stroke="#771100" strokeWidth="6" fill="none" />
              <path d="M 140 145 Q 120 70, 160 40" stroke="#771100" strokeWidth="4" fill="none" />
              <path d="M 140 145 Q 110 210, 150 250" stroke="#771100" strokeWidth="4.5" fill="none" />
              {selectedSample.isPositive && (
                <>
                  {/* Microaneurysms and exudates */}
                  <circle cx="220" cy="130" r="8" fill="#ffeebb" opacity="0.9" />
                  <circle cx="240" cy="140" r="6" fill="#ffeebb" opacity="0.85" />
                  <circle cx="190" cy="95" r="4" fill="#440000" />
                  <circle cx="210" cy="175" r="5" fill="#440000" />
                </>
              )}
            </svg>
          </div>

          {/* Saliency Attention Overlay View */}
          <div style={{ background: '#000', borderRadius: 'var(--radius-lg)', overflow: 'hidden', border: '1px solid var(--color-border-subtle)', position: 'relative' }}>
            <div style={{ position: 'absolute', top: 12, left: 12, background: 'rgba(0,0,0,0.7)', color: '#38bdf8', padding: '4px 10px', borderRadius: 'var(--radius-sm)', fontSize: 'var(--font-size-xs)', zIndex: 2 }}>
              Attention Heatmap ({opacity}% Blend)
            </div>
            <svg viewBox="0 0 400 300" style={{ width: '100%', height: 'auto', display: 'block' }}>
              <circle cx="200" cy="150" r="130" fill="#993311" />
              <circle cx="200" cy="150" r="125" fill="#bb4422" />
              <circle cx="140" cy="145" r="30" fill="#ffddaa" opacity="0.85" />
              <path d="M 140 145 Q 180 110, 260 80" stroke="#771100" strokeWidth="5" fill="none" />
              <path d="M 140 145 Q 190 180, 270 210" stroke="#771100" strokeWidth="6" fill="none" />
              {/* Heatmap Radial Hotspots */}
              {selectedSample.isPositive ? (
                <>
                  <circle cx="230" cy="135" r="45" fill="red" opacity={opacity / 150} filter="blur(8px)" />
                  <circle cx="230" cy="135" r="25" fill="yellow" opacity={opacity / 120} filter="blur(4px)" />
                  <circle cx="140" cy="145" r="35" fill="cyan" opacity={opacity / 250} filter="blur(6px)" />
                </>
              ) : (
                <circle cx="140" cy="145" r="35" fill="blue" opacity={opacity / 300} filter="blur(6px)" />
              )}
            </svg>
          </div>
        </div>

        {/* Diagnosis Pill */}
        <div style={{ marginTop: 'var(--space-4)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            {selectedSample.isPositive ? (
              <span className="badge badge-malignant" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <AlertTriangle size={16} /> Diagnosis: {selectedSample.diagnosis} (High Risk)
              </span>
            ) : (
              <span className="badge badge-benign" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <CheckCircle2 size={16} /> Diagnosis: {selectedSample.diagnosis} (Normal)
              </span>
            )}
          </div>
          <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
            Quantum HQNN Confidence: <strong>{selectedSample.isPositive ? '94.2%' : '98.7%'}</strong>
          </span>
        </div>
      </div>

      {/* 11 Biomarkers Grid */}
      <div className="card">
        <h4 style={{ marginBottom: 'var(--space-4)', color: 'var(--color-text-primary)' }}>
          Extracted Retinal Biomarkers (11 Ophthalmic Markers)
        </h4>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 'var(--space-4)' }}>
          {selectedSample.biomarkers.map((b, i) => (
            <div
              key={i}
              style={{
                background: 'var(--color-bg-surface-elevated)',
                border: '1px solid var(--color-border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: 'var(--space-3) var(--space-4)',
              }}
            >
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>{b.name}</div>
              <div style={{ fontSize: 'var(--font-size-xl)', fontWeight: 'bold', color: 'var(--color-quantum-primary)', marginTop: '2px' }}>
                {b.value} <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', fontWeight: 'normal' }}>{b.unit}</span>
              </div>
              <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
                Ref: {b.normal}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
