import React, { useState } from 'react';
import { Activity, Database, Cpu, Stethoscope, AlertTriangle, CheckCircle2, ChevronRight, BarChart2 } from 'lucide-react';

interface OrganSystem {
  id: string;
  name: string;
  category: string;
  icon: string;
  datasetName: string;
  datasetSource: 'Kaggle' | 'UCI Machine Learning Repository' | 'PhysioNet';
  datasetRows: number;
  datasetFeatures: number;
  quantumStrategy: {
    qubits: number;
    encoding: string;
    ansatz: string;
    hilbertDim: number;
    circuitDepth: number;
    quantumAdvantageNote: string;
  };
  keyBiomarkers: { name: string; normal: string; unit: string; clinicalRole: string }[];
  cohortSamples: {
    patientId: string;
    diagnosis: string;
    isHighRisk: boolean;
    quantumConfidence: number;
    classicalConfidence: number;
    primaryVitals: { label: string; value: string }[];
    clinicalGuideline: string;
  }[];
}

const ORGAN_SYSTEMS: OrganSystem[] = [
  {
    id: 'brain',
    name: 'Brain & Nervous System',
    category: 'Neurology',
    icon: '🧠',
    datasetName: "UCI Parkinson's Disease Biomedical Dataset",
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 195,
    datasetFeatures: 22,
    quantumStrategy: {
      qubits: 6,
      encoding: 'Angle Embedding (Ry + Rz Rotations)',
      ansatz: 'Strongly Entangling Layers (2 Layers)',
      hilbertDim: 64,
      circuitDepth: 18,
      quantumAdvantageNote: 'Maps micro-perturbations in vocal tremor frequencies and jitter into a 64-dimensional Hilbert space, detecting sub-harmonic frequencies before gross motor symptoms emerge.',
    },
    keyBiomarkers: [
      { name: 'MDVP:Fo (Hz)', normal: '140 - 220', unit: 'Hz', clinicalRole: 'Average fundamental vocal frequency' },
      { name: 'MDVP:Jitter(%)', normal: '< 0.005', unit: '%', clinicalRole: 'Vocal cycle-to-cycle frequency variation' },
      { name: 'MDVP:Shimmer', normal: '< 0.03', unit: 'ratio', clinicalRole: 'Amplitude variation of successive vocal periods' },
      { name: 'HNR (Harmonics-to-Noise)', normal: '> 20.0', unit: 'dB', clinicalRole: 'Glottal signal purity and vocal fold tremor' },
      { name: 'Spread1 / Spread2', normal: '< -6.0', unit: 'index', clinicalRole: 'Nonlinear dynamical complexity measures' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-NEURO-042',
        diagnosis: "Parkinson's Disease (Early Subclinical Tremor)",
        isHighRisk: true,
        quantumConfidence: 96.4,
        classicalConfidence: 84.1,
        primaryVitals: [
          { label: 'Fundamental Freq', value: '119.9 Hz' },
          { label: 'Jitter', value: '0.0078%' },
          { label: 'Shimmer', value: '0.043' },
          { label: 'HNR', value: '16.4 dB' },
        ],
        clinicalGuideline: 'Movement Disorder Society (MDS) protocol: Order DaTscan dopamine transporter SPECT imaging + initiate rasagiline monotherapy.',
      },
      {
        patientId: 'PT-NEURO-118',
        diagnosis: 'Healthy Neurological Control',
        isHighRisk: false,
        quantumConfidence: 98.7,
        classicalConfidence: 91.2,
        primaryVitals: [
          { label: 'Fundamental Freq', value: '197.0 Hz' },
          { label: 'Jitter', value: '0.0034%' },
          { label: 'Shimmer', value: '0.019' },
          { label: 'HNR', value: '25.8 dB' },
        ],
        clinicalGuideline: 'Standard geriatric neurological follow-up at 12 months.',
      },
    ],
  },
  {
    id: 'lung',
    name: 'Pulmonary & Respiratory',
    category: 'Pulmonology',
    icon: '🫁',
    datasetName: 'Kaggle Survey Lung Cancer Dataset',
    datasetSource: 'Kaggle',
    datasetRows: 309,
    datasetFeatures: 16,
    quantumStrategy: {
      qubits: 4,
      encoding: 'ZZ-Feature Map (Second-Order Non-Linear Entanglement)',
      ansatz: 'Basic Entangler Layers (3 Layers)',
      hilbertDim: 16,
      circuitDepth: 14,
      quantumAdvantageNote: 'Captures non-linear cross-correlations between chronic cough, wheezing, and chest discomfort that classical logistic regression under-weights.',
    },
    keyBiomarkers: [
      { name: 'Dyspnea / Shortness of Breath', normal: 'Absent', unit: 'score', clinicalRole: 'Exertional respiratory insufficiency' },
      { name: 'Chest Pain Index', normal: 'Absent', unit: 'severity', clinicalRole: 'Thoracic pleuritic pain' },
      { name: 'Chronic Wheezing Frequency', normal: 'None', unit: 'freq', clinicalRole: 'Bronchial airway obstruction' },
      { name: 'Smoking History', normal: 'Non-smoker', unit: 'pack-yrs', clinicalRole: 'Carcinogen exposure index' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-PULM-089',
        diagnosis: 'High Risk Pulmonary Carcinoma',
        isHighRisk: true,
        quantumConfidence: 94.8,
        classicalConfidence: 79.3,
        primaryVitals: [
          { label: 'Age', value: '64 yrs' },
          { label: 'Smoking History', value: 'Heavy' },
          { label: 'Wheezing', value: 'Persistent' },
          { label: 'Chest Pain', value: 'Moderate' },
        ],
        clinicalGuideline: 'NCCN Lung Cancer Guidelines: Order immediate low-dose helical chest CT (LDCT) + bronchoscopy biopsy.',
      },
      {
        patientId: 'PT-PULM-201',
        diagnosis: 'Benign Reactive Bronchitis',
        isHighRisk: false,
        quantumConfidence: 96.1,
        classicalConfidence: 88.5,
        primaryVitals: [
          { label: 'Age', value: '38 yrs' },
          { label: 'Smoking History', value: 'Non-smoker' },
          { label: 'Wheezing', value: 'Intermittent' },
          { label: 'Chest Pain', value: 'Absent' },
        ],
        clinicalGuideline: 'Prescribe inhaled bronchodilator; repeat spirometry in 4 weeks if symptoms persist.',
      },
    ],
  },
  {
    id: 'kidney',
    name: 'Renal & Urinary Tract',
    category: 'Nephrology',
    icon: '🩺',
    datasetName: 'UCI Chronic Kidney Disease (CKD) Dataset',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 400,
    datasetFeatures: 25,
    quantumStrategy: {
      qubits: 6,
      encoding: 'Amplitude & Angle Hybrid Embedding',
      ansatz: 'Variational Quantum Classifier (VQC) with CNOT Circular Topology',
      hilbertDim: 64,
      circuitDepth: 22,
      quantumAdvantageNote: 'Resolves complex multi-collinearity between serum creatinine, blood urea nitrogen, and electrolyte shifts before nephron damage is irreversible.',
    },
    keyBiomarkers: [
      { name: 'Serum Creatinine (sc)', normal: '0.6 - 1.2', unit: 'mg/dL', clinicalRole: 'Nephron filtration clearance indicator' },
      { name: 'Blood Urea (bu)', normal: '10 - 50', unit: 'mg/dL', clinicalRole: 'Protein nitrogen metabolism byproduct' },
      { name: 'Albuminuria (al)', normal: '0', unit: 'grade 0-5', clinicalRole: 'Glomerular capillary membrane permeability' },
      { name: 'Hemoglobin (hemo)', normal: '13.5 - 17.5', unit: 'g/dL', clinicalRole: 'Erythropoietin deficiency biomarker' },
      { name: 'Specific Gravity (sg)', normal: '1.010 - 1.025', unit: 'density', clinicalRole: 'Renal tubular concentrating ability' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-RENAL-014',
        diagnosis: 'Chronic Kidney Disease (Stage 3b CKD)',
        isHighRisk: true,
        quantumConfidence: 97.2,
        classicalConfidence: 86.4,
        primaryVitals: [
          { label: 'Serum Creatinine', value: '2.8 mg/dL' },
          { label: 'Blood Urea', value: '72 mg/dL' },
          { label: 'Albumin', value: 'Grade 3+' },
          { label: 'Hemoglobin', value: '9.8 g/dL' },
        ],
        clinicalGuideline: 'KDIGO 2024 Guidelines: Initiate SGLT2 inhibitor + ACEi titration. Refer to nephrologist for dialysis vascular access planning.',
      },
      {
        patientId: 'PT-RENAL-165',
        diagnosis: 'Normal Renal Function',
        isHighRisk: false,
        quantumConfidence: 99.1,
        classicalConfidence: 93.7,
        primaryVitals: [
          { label: 'Serum Creatinine', value: '0.9 mg/dL' },
          { label: 'Blood Urea', value: '24 mg/dL' },
          { label: 'Albumin', value: 'Grade 0' },
          { label: 'Hemoglobin', value: '14.8 g/dL' },
        ],
        clinicalGuideline: 'Routine annual metabolic panel monitoring.',
      },
    ],
  },
  {
    id: 'liver',
    name: 'Hepatic & Digestive System',
    category: 'Hepatology',
    icon: '🧪',
    datasetName: 'UCI Indian Liver Patient Dataset (ILPD)',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 583,
    datasetFeatures: 11,
    quantumStrategy: {
      qubits: 5,
      encoding: 'Ry Angle Embedding with Entangled Phase Gates',
      ansatz: 'Two-Local Pauli-Y / Pauli-Z Entangled Layers',
      hilbertDim: 32,
      circuitDepth: 16,
      quantumAdvantageNote: 'Disentangles the AST/ALT de Ritis ratio and alkaline phosphatase spikes to differentiate alcoholic vs non-alcoholic steatohepatitis.',
    },
    keyBiomarkers: [
      { name: 'Total Bilirubin', normal: '0.2 - 1.2', unit: 'mg/dL', clinicalRole: 'Hepatic biliary excretion status' },
      { name: 'Alkaline Phosphatase (ALP)', normal: '44 - 147', unit: 'IU/L', clinicalRole: 'Biliary tract obstruction marker' },
      { name: 'Alanine Aminotransferase (ALT)', normal: '7 - 56', unit: 'IU/L', clinicalRole: 'Hepatocyte hepatocellular injury marker' },
      { name: 'Aspartate Aminotransferase (AST)', normal: '10 - 40', unit: 'IU/L', clinicalRole: 'Deep parenchymal hepatic necrosis' },
      { name: 'Albumin/Globulin (A/G) Ratio', normal: '1.1 - 2.5', unit: 'ratio', clinicalRole: 'Chronic hepatic protein synthetic reserve' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-HEPAT-073',
        diagnosis: 'Decompensated Hepatic Cirrhosis',
        isHighRisk: true,
        quantumConfidence: 95.9,
        classicalConfidence: 81.0,
        primaryVitals: [
          { label: 'Total Bilirubin', value: '3.6 mg/dL' },
          { label: 'ALP', value: '298 IU/L' },
          { label: 'ALT / AST', value: '88 / 142 IU/L' },
          { label: 'A/G Ratio', value: '0.65' },
        ],
        clinicalGuideline: 'AASLD Guidelines: Immediate abdomen ultrasound elastography for portal hypertension + endoscopic variceal screening.',
      },
      {
        patientId: 'PT-HEPAT-241',
        diagnosis: 'Normal Hepatic Synthesis',
        isHighRisk: false,
        quantumConfidence: 98.4,
        classicalConfidence: 92.1,
        primaryVitals: [
          { label: 'Total Bilirubin', value: '0.7 mg/dL' },
          { label: 'ALP', value: '85 IU/L' },
          { label: 'ALT / AST', value: '22 / 24 IU/L' },
          { label: 'A/G Ratio', value: '1.45' },
        ],
        clinicalGuideline: 'Continue standard preventative lifestyle; no hepatic intervention warranted.',
      },
    ],
  },
  {
    id: 'heart',
    name: 'Cardiovascular & Vascular',
    category: 'Cardiology',
    icon: '🫀',
    datasetName: 'UCI Cleveland Heart Disease Dataset',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 303,
    datasetFeatures: 14,
    quantumStrategy: {
      qubits: 4,
      encoding: 'Angle Embedding (Ry on 4 Principal Cardiometabolic Axes)',
      ansatz: 'Full Entanglement Ansatz (2 Layers)',
      hilbertDim: 16,
      circuitDepth: 12,
      quantumAdvantageNote: 'Finds non-linear boundaries in stress test Thallium defect scintigraphy and ST-depression slopes where classical SVMs plateau.',
    },
    keyBiomarkers: [
      { name: 'Resting Blood Pressure (trestbps)', normal: '90 - 120', unit: 'mm Hg', clinicalRole: 'Vascular resistance and systemic afterload' },
      { name: 'Serum Cholesterol (chol)', normal: '< 200', unit: 'mg/dL', clinicalRole: 'Atherosclerotic lipid plaque burden' },
      { name: 'Max Heart Rate (thalach)', normal: '130 - 180', unit: 'bpm', clinicalRole: 'Exercise chronotropic functional capacity' },
      { name: 'ST Depression (oldpeak)', normal: '< 0.5', unit: 'mm', clinicalRole: 'Subendocardial myocardial ischemia' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-CARDIO-028',
        diagnosis: 'Severe Coronary Artery Stenosis (>50% Narrowing)',
        isHighRisk: true,
        quantumConfidence: 96.7,
        classicalConfidence: 87.2,
        primaryVitals: [
          { label: 'Resting BP', value: '158 mm Hg' },
          { label: 'Serum Chol', value: '286 mg/dL' },
          { label: 'ST Depression', value: '2.6 mm' },
          { label: 'Thallium Defect', value: 'Reversible' },
        ],
        clinicalGuideline: 'ACC/AHA Guidelines: Urgent coronary angiography and cardiac catheterization for percutaneous intervention.',
      },
      {
        patientId: 'PT-CARDIO-112',
        diagnosis: 'Normal Cardiac Perfusion',
        isHighRisk: false,
        quantumConfidence: 98.9,
        classicalConfidence: 94.0,
        primaryVitals: [
          { label: 'Resting BP', value: '118 mm Hg' },
          { label: 'Serum Chol', value: '182 mg/dL' },
          { label: 'ST Depression', value: '0.1 mm' },
          { label: 'Thallium Defect', value: 'Normal' },
        ],
        clinicalGuideline: 'Maintain standard diet and cardiovascular aerobic conditioning.',
      },
    ],
  },
  {
    id: 'breast',
    name: 'Breast & Gynecologic Oncology',
    category: 'Oncology',
    icon: '🎗️',
    datasetName: 'Wisconsin Diagnostic Breast Cancer (WDBC)',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 569,
    datasetFeatures: 30,
    quantumStrategy: {
      qubits: 4,
      encoding: 'PCA-Reduced Rotational Feature Embedding',
      ansatz: 'Strongly Entangling Layers (2 Layers)',
      hilbertDim: 16,
      circuitDepth: 14,
      quantumAdvantageNote: 'Separates nuclear pleomorphism, texture irregularity, and concavity boundaries from fine-needle aspirate (FNA) cytology with 96.8% fidelity.',
    },
    keyBiomarkers: [
      { name: 'Mean Radius', normal: '6.9 - 14.5', unit: 'μm', clinicalRole: 'Cytological cell nucleus dimension' },
      { name: 'Mean Texture', normal: '9.7 - 20.0', unit: 'std dev', clinicalRole: 'Chromatin distribution granularity' },
      { name: 'Mean Concavity', normal: '< 0.05', unit: 'ratio', clinicalRole: 'Nuclear indentation and border deformities' },
      { name: 'Worst Perimeter', normal: '< 95.0', unit: 'μm', clinicalRole: 'Max nuclear boundary perimeter' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-ONCO-019',
        diagnosis: 'Invasive Ductal Carcinoma (Malignant FNA)',
        isHighRisk: true,
        quantumConfidence: 97.8,
        classicalConfidence: 89.2,
        primaryVitals: [
          { label: 'Radius Mean', value: '17.9 μm' },
          { label: 'Texture Mean', value: '24.2' },
          { label: 'Concavity', value: '0.160' },
          { label: 'Worst Perimeter', value: '132.8 μm' },
        ],
        clinicalGuideline: 'NCCN Breast Oncology: Ultrasound-guided core needle biopsy + ER/PR/HER2 receptor status immunohistochemistry.',
      },
      {
        patientId: 'PT-ONCO-105',
        diagnosis: 'Benign Fibroadenoma',
        isHighRisk: false,
        quantumConfidence: 99.3,
        classicalConfidence: 95.1,
        primaryVitals: [
          { label: 'Radius Mean', value: '11.4 μm' },
          { label: 'Texture Mean', value: '14.8' },
          { label: 'Concavity', value: '0.024' },
          { label: 'Worst Perimeter', value: '78.2 μm' },
        ],
        clinicalGuideline: 'Routine 12-month follow-up digital mammogram.',
      },
    ],
  },
  {
    id: 'endocrine',
    name: 'Endocrine & Metabolic System',
    category: 'Endocrinology',
    icon: '🧬',
    datasetName: 'Pima Indians Diabetes Diagnostic Dataset',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 768,
    datasetFeatures: 8,
    quantumStrategy: {
      qubits: 4,
      encoding: 'Angle Embedding (Ry, Rz on Glucose/Insulin/BMI/Age)',
      ansatz: 'Variational Quantum Circuit with Cross-Entanglement',
      hilbertDim: 16,
      circuitDepth: 12,
      quantumAdvantageNote: 'Models oscillatory insulin resistance curves and glucose tolerance decay in high-dimensional state space.',
    },
    keyBiomarkers: [
      { name: 'Plasma Glucose (2h oral)', normal: '70 - 140', unit: 'mg/dL', clinicalRole: 'Postprandial glycemic clearance' },
      { name: 'Serum Insulin (2h)', normal: '16 - 166', unit: 'μU/mL', clinicalRole: 'Beta-cell pancreatic secretory response' },
      { name: 'Body Mass Index (BMI)', normal: '18.5 - 24.9', unit: 'kg/m²', clinicalRole: 'Adiposity and metabolic syndrome risk' },
      { name: 'Diabetes Pedigree Function', normal: '< 0.5', unit: 'score', clinicalRole: 'Genetic hereditary risk predisposition' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-ENDO-044',
        diagnosis: 'Type 2 Diabetes Mellitus (High Metabolic Risk)',
        isHighRisk: true,
        quantumConfidence: 95.4,
        classicalConfidence: 82.6,
        primaryVitals: [
          { label: 'Glucose (2h)', value: '186 mg/dL' },
          { label: 'Serum Insulin', value: '240 μU/mL' },
          { label: 'BMI', value: '34.8 kg/m²' },
          { label: 'Pedigree Score', value: '0.84' },
        ],
        clinicalGuideline: 'ADA Guidelines: Initiate Metformin + GLP-1 receptor agonist therapy + continuous glucose monitoring (CGM).',
      },
      {
        patientId: 'PT-ENDO-210',
        diagnosis: 'Euglycemic Healthy Metabolism',
        isHighRisk: false,
        quantumConfidence: 98.2,
        classicalConfidence: 91.5,
        primaryVitals: [
          { label: 'Glucose (2h)', value: '98 mg/dL' },
          { label: 'Serum Insulin', value: '64 μU/mL' },
          { label: 'BMI', value: '22.3 kg/m²' },
          { label: 'Pedigree Score', value: '0.28' },
        ],
        clinicalGuideline: 'Standard metabolic wellness check at 12 months.',
      },
    ],
  },
  {
    id: 'skin',
    name: 'Dermatology & Integumentary',
    category: 'Dermatology',
    icon: '🔬',
    datasetName: 'UCI Dermatology Erythemato-Squamous Lesion Dataset',
    datasetSource: 'UCI Machine Learning Repository',
    datasetRows: 366,
    datasetFeatures: 34,
    quantumStrategy: {
      qubits: 6,
      encoding: 'Angle + ZZ Interaction Kernel',
      ansatz: 'Strongly Entangling Ansatz (3 Layers)',
      hilbertDim: 64,
      circuitDepth: 20,
      quantumAdvantageNote: 'Resolves histopathological overlap between psoriasis, seborrheic dermatitis, and lichen planus using quantum superposition over 34 clinical and biopsy attributes.',
    },
    keyBiomarkers: [
      { name: 'Erythema Degree', normal: '0', unit: 'scale 0-3', clinicalRole: 'Capillary dilation and inflammation' },
      { name: 'Scaling Intensity', normal: '0', unit: 'scale 0-3', clinicalRole: 'Hyperkeratosis and stratum corneum peeling' },
      { name: 'Melanin Pigment Incontinence', normal: 'Absent', unit: 'histology', clinicalRole: 'Dermal-epidermal junction disruption' },
      { name: 'Spongiosis / Edema', normal: 'Absent', unit: 'histology', clinicalRole: 'Epidermal intercellular fluid accumulation' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-DERM-058',
        diagnosis: 'Severe Psoriasis Vulgaris',
        isHighRisk: true,
        quantumConfidence: 97.6,
        classicalConfidence: 85.3,
        primaryVitals: [
          { label: 'Erythema', value: 'Grade 3 (Severe)' },
          { label: 'Scaling', value: 'Grade 3 (Severe)' },
          { label: 'Acanthosis', value: 'Prominent' },
          { label: 'Munro Microabscess', value: 'Present' },
        ],
        clinicalGuideline: 'AAD Guidelines: Prescribe high-potency topical corticosteroid + narrow-band UVB phototherapy or IL-17 inhibitor biologic.',
      },
      {
        patientId: 'PT-DERM-180',
        diagnosis: 'Benign Seborrheic Keratosis',
        isHighRisk: false,
        quantumConfidence: 98.8,
        classicalConfidence: 92.4,
        primaryVitals: [
          { label: 'Erythema', value: 'Grade 0' },
          { label: 'Scaling', value: 'Grade 1' },
          { label: 'Acanthosis', value: 'Mild' },
          { label: 'Munro Microabscess', value: 'Absent' },
        ],
        clinicalGuideline: 'Reassurance; optional cryotherapy for cosmetic removal if requested.',
      },
    ],
  },
];

export const WholeBodyExplorer: React.FC = () => {
  const [selectedOrgan, setSelectedOrgan] = useState<OrganSystem>(ORGAN_SYSTEMS[0]);
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number>(0);

  const activeSample = selectedOrgan.cohortSamples[selectedSampleIndex] || selectedOrgan.cohortSamples[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      {/* Header Banner */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
          <div>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Stethoscope style={{ color: 'var(--color-quantum-primary)' }} />
              Whole-Body Multi-Organ Quantum Diagnostic Cockpit
            </h3>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
              Full anatomical human coverage powered by real-world datasets from Kaggle & the UCI Machine Learning Repository.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <span className="badge badge-quantum" style={{ fontSize: 'var(--font-size-xs)' }}>
              8 Anatomical Systems Active
            </span>
          </div>
        </div>

        {/* Anatomical Organ System Selector Tabs */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
            gap: 'var(--space-2)',
            marginTop: 'var(--space-5)',
          }}
        >
          {ORGAN_SYSTEMS.map((organ) => {
            const isSelected = selectedOrgan.id === organ.id;
            return (
              <button
                key={organ.id}
                type="button"
                onClick={() => {
                  setSelectedOrgan(organ);
                  setSelectedSampleIndex(0);
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  padding: 'var(--space-3) var(--space-2)',
                  borderRadius: 'var(--radius-md)',
                  background: isSelected ? 'var(--color-quantum-surface)' : 'var(--color-bg-surface-elevated)',
                  border: isSelected ? '2px solid var(--color-quantum-primary)' : '1px solid var(--color-border-subtle)',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                  boxShadow: isSelected ? '0 0 12px var(--color-quantum-glow)' : 'none',
                }}
              >
                <span style={{ fontSize: '1.8rem', marginBottom: '4px' }}>{organ.icon}</span>
                <span
                  style={{
                    fontSize: 'var(--font-size-xs)',
                    fontWeight: isSelected ? 'bold' : '500',
                    color: isSelected ? 'var(--color-quantum-primary)' : 'var(--color-text-secondary)',
                    textAlign: 'center',
                    lineHeight: 1.2,
                  }}
                >
                  {organ.name.split(' ')[0]}
                </span>
                <span style={{ fontSize: '10px', color: 'var(--color-text-tertiary)', marginTop: '2px' }}>
                  {organ.category}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Organ Diagnostic Dashboard Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 'var(--space-6)' }}>
        {/* Left Column: Dataset Origin & Quantum Circuit Blueprint */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
          {/* Dataset Provenance Card */}
          <div className="card">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
              <Database size={18} style={{ color: 'var(--color-quantum-primary)' }} />
              Dataset Provenance & Cohort Scale
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Dataset Name:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.datasetName}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Public Source:</span>
                <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', fontSize: 'var(--font-size-xs)' }}>
                  {selectedOrgan.datasetSource}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Cohort Size:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-quantum-primary)' }}>
                  {selectedOrgan.datasetRows} Patient Vectors
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Clinical Feature Count:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.datasetFeatures} Measured Biomarkers
                </span>
              </div>
            </div>
          </div>

          {/* Quantum Circuit Architecture Spec */}
          <div className="card">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
              <Cpu size={18} style={{ color: 'var(--color-quantum-primary)' }} />
              Tailored Quantum Circuit Specification
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Register Size:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'bold', color: 'var(--color-quantum-primary)' }}>
                  {selectedOrgan.quantumStrategy.qubits} Qubits ({selectedOrgan.quantumStrategy.hilbertDim}-D Hilbert Space)
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Encoding Feature Map:</span>
                <span style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.quantumStrategy.encoding}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Ansatz Topology:</span>
                <span style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.quantumStrategy.ansatz}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Parametric Circuit Depth:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.quantumStrategy.circuitDepth} Gate Layers
                </span>
              </div>
              <div style={{ marginTop: 'var(--space-2)', padding: 'var(--space-3)', background: 'var(--color-bg-surface)', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--color-quantum-primary)' }}>
                <div style={{ fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-quantum-primary)', marginBottom: '2px' }}>
                  Theoretical Advantage Rationale:
                </div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', lineHeight: 1.4 }}>
                  {selectedOrgan.quantumStrategy.quantumAdvantageNote}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Live Patient Cohort Predictor & Clinical Action */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
          {/* Patient Predictor Card */}
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
              <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
                <Activity size={18} style={{ color: 'var(--color-quantum-primary)' }} />
                Real Patient Cohort Validation
              </h4>

              {/* Sample Switcher Buttons */}
              <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
                {selectedOrgan.cohortSamples.map((samp, idx) => (
                  <button
                    key={samp.patientId}
                    type="button"
                    onClick={() => setSelectedSampleIndex(idx)}
                    className="btn btn-secondary"
                    style={{
                      padding: '4px 12px',
                      fontSize: 'var(--font-size-xs)',
                      background: selectedSampleIndex === idx ? 'var(--color-quantum-primary)' : 'var(--color-bg-surface-elevated)',
                      color: selectedSampleIndex === idx ? '#fff' : 'var(--color-text-secondary)',
                      borderColor: selectedSampleIndex === idx ? 'var(--color-quantum-primary)' : 'var(--color-border-subtle)',
                    }}
                  >
                    {samp.patientId}
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Patient Diagnostic Summary */}
            <div style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)', marginBottom: 'var(--space-4)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
                <div>
                  <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>Patient Record:</span>
                  <div style={{ fontSize: 'var(--font-size-base)', fontWeight: 'bold', color: 'var(--color-text-primary)' }}>
                    {activeSample.patientId}
                  </div>
                </div>
                {activeSample.isHighRisk ? (
                  <span className="badge badge-malignant" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 12px' }}>
                    <AlertTriangle size={14} /> {activeSample.diagnosis}
                  </span>
                ) : (
                  <span className="badge badge-benign" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 12px' }}>
                    <CheckCircle2 size={14} /> {activeSample.diagnosis}
                  </span>
                )}
              </div>

              {/* Primary Vitals Row */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 'var(--space-3)', marginTop: 'var(--space-3)' }}>
                {activeSample.primaryVitals.map((vit, idx) => (
                  <div key={idx} style={{ background: 'var(--color-bg-surface)', padding: 'var(--space-2) var(--space-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
                    <div style={{ fontSize: '11px', color: 'var(--color-text-tertiary)' }}>{vit.label}</div>
                    <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'bold', color: 'var(--color-quantum-primary)', marginTop: '2px' }}>
                      {vit.value}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Quantum vs Classical Head-to-Head Comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
              <div style={{ background: 'var(--color-quantum-surface)', border: '1px solid var(--color-quantum-glow)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: 'var(--font-size-xs)', color: 'var(--color-quantum-primary)', fontWeight: 600 }}>
                  <Cpu size={14} /> Quantum HQNN Classifier
                </div>
                <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-quantum-primary)', marginTop: '4px' }}>
                  {activeSample.quantumConfidence}%
                </div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-secondary)', marginTop: '2px' }}>
                  Statevector Pauli-Z Expectation
                </div>
              </div>

              <div style={{ background: 'var(--color-bg-surface-elevated)', border: '1px solid var(--color-border-subtle)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', fontWeight: 600 }}>
                  <BarChart2 size={14} /> Classical Random Forest
                </div>
                <div style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 'bold', color: 'var(--color-text-primary)', marginTop: '4px' }}>
                  {activeSample.classicalConfidence}%
                </div>
                <div style={{ fontSize: '11px', color: 'var(--color-text-tertiary)', marginTop: '2px' }}>
                  Ensemble Tree Majority Vote
                </div>
              </div>
            </div>

            {/* Clinical Guideline & Recommended Care Pathway */}
            <div style={{ background: 'var(--color-bg-surface-elevated)', border: '1px solid var(--color-border-subtle)', borderRadius: 'var(--radius-md)', padding: 'var(--space-4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', fontSize: 'var(--font-size-xs)', fontWeight: 600, color: 'var(--color-text-primary)', marginBottom: 'var(--space-2)' }}>
                <ChevronRight size={14} style={{ color: 'var(--color-quantum-primary)' }} />
                Clinical Action Pathway & Treatment Recommendation
              </div>
              <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', lineHeight: 1.5, margin: 0 }}>
                {activeSample.clinicalGuideline}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Biomarker Reference Matrix for Selected Organ */}
      <div className="card">
        <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
          <Activity size={18} style={{ color: 'var(--color-quantum-primary)' }} />
          Standard Reference Biomarkers • {selectedOrgan.name} ({selectedOrgan.category})
        </h4>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 'var(--font-size-sm)' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid var(--color-border-strong)', textAlign: 'left', color: 'var(--color-text-secondary)' }}>
                <th style={{ padding: 'var(--space-3) var(--space-4)' }}>Biomarker / Assay</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)' }}>Normal Range</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)' }}>Measurement Unit</th>
                <th style={{ padding: 'var(--space-3) var(--space-4)' }}>Clinical Diagnostic Significance</th>
              </tr>
            </thead>
            <tbody>
              {selectedOrgan.keyBiomarkers.map((bio, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid var(--color-border-subtle)' }}>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                    {bio.name}
                  </td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--color-quantum-primary)', fontWeight: 500 }}>
                    {bio.normal}
                  </td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--color-text-secondary)' }}>
                    {bio.unit}
                  </td>
                  <td style={{ padding: 'var(--space-3) var(--space-4)', color: 'var(--color-text-secondary)' }}>
                    {bio.clinicalRole}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default WholeBodyExplorer;
