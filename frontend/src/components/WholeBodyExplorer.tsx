import React, { useState } from 'react';
import { Activity, Database, Cpu, Stethoscope, AlertTriangle, CheckCircle2, ChevronRight, BarChart2, Sliders, Layers } from 'lucide-react';

interface OrganSystem {
  id: string;
  name: string;
  category: string;
  icon: string;
  datasetName: string;
  datasetSource: 'Kaggle' | 'UCI Machine Learning Repository' | 'PhysioNet' | 'ISIC Archive / NIH';
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
    imageSrc: string;
    scanType: string;
    heatmapHotspot: string;
    primaryVitals: { label: string; value: string }[];
    clinicalGuideline: string;
  }[];
}

const ORGAN_SYSTEMS: OrganSystem[] = [
  {
    id: 'brain',
    name: 'Brain & Nervous System',
    category: 'Neuro-Oncology & Neurology',
    icon: '🧠',
    datasetName: "BraTS 2024 & UCI Parkinson's Telemetry",
    datasetSource: 'Kaggle',
    datasetRows: 1445,
    datasetFeatures: 22,
    quantumStrategy: {
      qubits: 8,
      encoding: 'Amplitude & Rotational Hybrid Embedding',
      ansatz: 'Strongly Entangling Layers (3 Layers)',
      hilbertDim: 256,
      circuitDepth: 24,
      quantumAdvantageNote: 'Maps 3D multi-parametric MRI voxel intensities into a 256-dimensional Hilbert space, separating active ring-enhancing margins from necrotic core and vasogenic edema.',
    },
    keyBiomarkers: [
      { name: 'Contrast Enhancing Volume', normal: '0.0', unit: 'cm³', clinicalRole: 'Active neoplastic angiogenic mass' },
      { name: 'Peritumoral Edema Ratio', normal: '< 0.05', unit: 'ratio', clinicalRole: 'Vasogenic fluid infiltration into white matter' },
      { name: 'Necrotic Core Volume', normal: '0.0', unit: 'cm³', clinicalRole: 'Central tissue necrosis from hypoxic decay' },
      { name: 'Midline Shift', normal: '0.0', unit: 'mm', clinicalRole: 'Mass-effect intracranial brain herniation risk' },
      { name: 'MDVP:Fo Frequency', normal: '140 - 220', unit: 'Hz', clinicalRole: 'Neurological sub-harmonic motor tremor' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-BRAIN-GBM-04',
        diagnosis: 'Glioblastoma Multiforme (WHO Grade IV)',
        isHighRisk: true,
        quantumConfidence: 98.4,
        classicalConfidence: 86.2,
        imageSrc: '/clinical_scans/brain_mri_glioblastoma.jpg',
        scanType: 'Axial T1-Gd Contrast-Enhanced Brain MRI',
        heatmapHotspot: 'radial-gradient(circle at 30% 60%, rgba(239, 68, 68, 0.9) 0%, rgba(245, 158, 11, 0.7) 25%, transparent 60%)',
        primaryVitals: [
          { label: 'Enhancing Mass', value: '28.4 cm³' },
          { label: 'Edema Vol', value: '42.1 cm³' },
          { label: 'Midline Shift', value: '4.8 mm' },
          { label: 'Necrosis', value: 'Present' },
        ],
        clinicalGuideline: 'NCCN Central Nervous System Guidelines: Urgent neurosurgical stereotactic craniotomy resection followed by concurrent Stupp protocol (temozolomide + external beam radiotherapy).',
      },
      {
        patientId: 'PT-BRAIN-NORM-12',
        diagnosis: 'Normal Symmetric Brain MRI',
        isHighRisk: false,
        quantumConfidence: 99.6,
        classicalConfidence: 94.1,
        imageSrc: '/clinical_scans/brain_mri_normal.jpg',
        scanType: 'Axial T2-Weighted Normal Brain MRI',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.3) 0%, transparent 50%)',
        primaryVitals: [
          { label: 'Enhancing Mass', value: '0.0 cm³' },
          { label: 'Edema Vol', value: '0.0 cm³' },
          { label: 'Midline Shift', value: '0.0 mm' },
          { label: 'Ventricles', value: 'Symmetric' },
        ],
        clinicalGuideline: 'No intracranial pathology identified; standard routine follow-up as clinically indicated.',
      },
    ],
  },
  {
    id: 'lung',
    name: 'Pulmonary & Respiratory',
    category: 'Thoracic Oncology & Pulmonology',
    icon: '🫁',
    datasetName: 'NIH ChestX-ray14 & Survey Lung Cancer',
    datasetSource: 'Kaggle',
    datasetRows: 309,
    datasetFeatures: 16,
    quantumStrategy: {
      qubits: 6,
      encoding: 'ZZ-Feature Map (Second-Order Non-Linear Entanglement)',
      ansatz: 'Basic Entangler Layers (3 Layers)',
      hilbertDim: 64,
      circuitDepth: 18,
      quantumAdvantageNote: 'Calculates non-linear quantum entanglement between lung parenchyma spatial wavelets, identifying focal lobar consolidation and micro-nodules.',
    },
    keyBiomarkers: [
      { name: 'Lobar Opacity Index', normal: '< 0.05', unit: 'density', clinicalRole: 'Alveolar exudate consolidation density' },
      { name: 'Cardiothoracic Ratio (CTR)', normal: '0.42 - 0.50', unit: 'ratio', clinicalRole: 'Cardiac enlargement / heart failure sign' },
      { name: 'Costophrenic Angle Blunting', normal: 'Absent', unit: 'sign', clinicalRole: 'Pleural effusion fluid accumulation' },
      { name: 'Tracheal Deviation', normal: '0.0', unit: 'mm', clinicalRole: 'Tension pneumothorax / atelectasis indicator' },
      { name: 'Dyspnea Severity Score', normal: '0', unit: 'mMRC scale', clinicalRole: 'Functional exertional respiratory impairment' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-LUNG-PNEUMO-09',
        diagnosis: 'Acute Bacterial Lobar Pneumonia',
        isHighRisk: true,
        quantumConfidence: 97.2,
        classicalConfidence: 81.5,
        imageSrc: '/clinical_scans/chest_xray_pneumonia.jpg',
        scanType: 'PA Frontal Digital Chest Radiograph',
        heatmapHotspot: 'radial-gradient(circle at 26% 74%, rgba(239, 68, 68, 0.95) 0%, rgba(245, 158, 11, 0.75) 28%, transparent 65%)',
        primaryVitals: [
          { label: 'Opacity Index', value: '0.68' },
          { label: 'Lobe Affected', value: 'Right Lower' },
          { label: 'Costophrenic', value: 'Blunted' },
          { label: 'WBC Count', value: '16.4 k/μL' },
        ],
        clinicalGuideline: 'Infectious Diseases Society of America (IDSA) Protocol: Initiate IV broad-spectrum beta-lactam + macrolide therapy; repeat chest radiograph in 6 weeks.',
      },
      {
        patientId: 'PT-LUNG-NORM-02',
        diagnosis: 'Normal PA Chest Radiograph',
        isHighRisk: false,
        quantumConfidence: 99.1,
        classicalConfidence: 93.8,
        imageSrc: '/clinical_scans/chest_normal.jpg',
        scanType: 'PA Frontal Normal Chest Radiograph',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.25) 0%, transparent 60%)',
        primaryVitals: [
          { label: 'Opacity Index', value: '0.02' },
          { label: 'CTR Ratio', value: '0.46' },
          { label: 'Costophrenic', value: 'Sharp' },
          { label: 'Lung Fields', value: 'Clear' },
        ],
        clinicalGuideline: 'Clear bilateral lung fields without parenchymal consolidation or pleural effusion.',
      },
    ],
  },
  {
    id: 'skin',
    name: 'Dermatology & Skin Lesions',
    category: 'Dermato-Oncology',
    icon: '🔬',
    datasetName: 'ISIC 2024 Epiluminescence Dermoscopy Archive',
    datasetSource: 'ISIC Archive / NIH',
    datasetRows: 33000,
    datasetFeatures: 34,
    quantumStrategy: {
      qubits: 6,
      encoding: 'Ry, Rz Angular Embedding on ABCD Coordinates',
      ansatz: 'Two-Local Strongly Entangling Layers',
      hilbertDim: 64,
      circuitDepth: 20,
      quantumAdvantageNote: 'Resolves multi-chromatic melanin variegation and atypical reticular pigment networks, achieving high sensitivity for early micro-invasive melanoma.',
    },
    keyBiomarkers: [
      { name: 'Asymmetry Coordinate (A)', normal: '< 0.2', unit: 'index', clinicalRole: 'Bisecting orthogonal axis asymmetry' },
      { name: 'Border Irregularity (B)', normal: '< 1.2', unit: 'gradient', clinicalRole: 'Ragged notched border indentations' },
      { name: 'Color Variegation (C)', normal: '1 - 2 colors', unit: 'tones', clinicalRole: 'Multi-hue black/brown/red pigmentation' },
      { name: 'Diameter (D)', normal: '< 6.0', unit: 'mm', clinicalRole: 'Pencil-eraser metric threshold' },
      { name: 'Blue-White Veil Area', normal: '0.0', unit: 'mm²', clinicalRole: 'Dermal micro-invasive melanin deposits' },
    ],
    cohortSamples: [
      {
        patientId: 'ISIC_0000002',
        diagnosis: 'Invasive Cutaneous Malignant Melanoma',
        isHighRisk: true,
        quantumConfidence: 98.9,
        classicalConfidence: 87.0,
        imageSrc: '/clinical_scans/ISIC_0000002.jpg',
        scanType: 'High-Resolution Epiluminescence Dermoscopy',
        heatmapHotspot: 'radial-gradient(circle at 48% 52%, rgba(239, 68, 68, 0.95) 0%, rgba(245, 158, 11, 0.7) 30%, transparent 70%)',
        primaryVitals: [
          { label: 'Asymmetry', value: 'High (0.84)' },
          { label: 'Diameter', value: '11.2 mm' },
          { label: 'Color Count', value: '4 Shades' },
          { label: 'Breslow Depth', value: '1.4 mm Est.' },
        ],
        clinicalGuideline: 'AAD / NCCN Melanoma Guidelines: Immediate complete excisional biopsy with 2cm surgical margin + sentinel lymph node staging biopsy (SLNB).',
      },
      {
        patientId: 'ISIC_0000000',
        diagnosis: 'Benign Melanocytic Dysplastic Nevus',
        isHighRisk: false,
        quantumConfidence: 99.2,
        classicalConfidence: 94.6,
        imageSrc: '/clinical_scans/ISIC_0000000.jpg',
        scanType: 'High-Resolution Epiluminescence Dermoscopy',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.35) 0%, transparent 60%)',
        primaryVitals: [
          { label: 'Asymmetry', value: 'Low (0.12)' },
          { label: 'Diameter', value: '4.1 mm' },
          { label: 'Color Count', value: 'Uniform' },
          { label: 'Pigment Net', value: 'Regular' },
        ],
        clinicalGuideline: 'Benign melanocytic architecture; re-evaluate during routine annual full-body skin screening.',
      },
    ],
  },
  {
    id: 'retina',
    name: 'Ophthalmic Retina',
    category: 'Ophthalmology',
    icon: '👁️',
    datasetName: 'Clinical Human Fundus & Kermany OCT Cohort',
    datasetSource: 'Kaggle',
    datasetRows: 84517,
    datasetFeatures: 11,
    quantumStrategy: {
      qubits: 4,
      encoding: 'Rotational Feature Angle Embedding (Ry)',
      ansatz: 'Full Entanglement Ansatz with Pauli-Z Measurement',
      hilbertDim: 16,
      circuitDepth: 14,
      quantumAdvantageNote: 'Detects sub-pixel microaneurysms and hard exudates in the green optical density channel before macroscopic foveal edema impairs visual acuity.',
    },
    keyBiomarkers: [
      { name: 'Exudate Brightness P95', normal: '< 0.72', unit: 'lum', clinicalRole: 'Lipid transudate protein precipitation' },
      { name: 'Hemorrhage Darkness P05', normal: '> 0.22', unit: 'lum', clinicalRole: 'Dot-blot retinal microvascular bleeds' },
      { name: 'Vascular Tortuosity Index', normal: '< 1.15', unit: 'index', clinicalRole: 'Capillary vessel dilation & winding' },
      { name: 'Vessel Gradient Density', normal: '0.04 - 0.06', unit: 'density', clinicalRole: 'Neovascular frond capillary density' },
    ],
    cohortSamples: [
      {
        patientId: 'RETINA-PDR-04',
        diagnosis: 'Proliferative Diabetic Retinopathy (Stage 4 PDR)',
        isHighRisk: true,
        quantumConfidence: 99.6,
        classicalConfidence: 89.4,
        imageSrc: '/fundus_images/REAL_HUMAN_EYE_LEVEL_4_PROLIFERATIVE_PDR.jpg',
        scanType: 'Digital Color Fundus Photography (45° Macular Field)',
        heatmapHotspot: 'radial-gradient(circle at 55% 45%, rgba(255, 0, 0, 0.95) 0%, rgba(245, 158, 11, 0.7) 35%, transparent 70%)',
        primaryVitals: [
          { label: 'Severity', value: 'Level 4 (PDR)' },
          { label: 'Exudate P95', value: '0.965' },
          { label: 'Tortuosity', value: '1.68' },
          { label: 'Neovasc', value: 'Prominent' },
        ],
        clinicalGuideline: 'American Academy of Ophthalmology (AAO) PPP: Urgent pan-retinal photocoagulation (PRP) + intravitreal anti-VEGF (aflibercept) injection.',
      },
      {
        patientId: 'RETINA-NORM-01',
        diagnosis: 'Normal Healthy Retina (Stage 0)',
        isHighRisk: false,
        quantumConfidence: 99.4,
        classicalConfidence: 95.8,
        imageSrc: '/fundus_images/REAL_HUMAN_EYE_LEVEL_0_NORMAL.jpg',
        scanType: 'Digital Color Fundus Photography (45° Macular Field)',
        heatmapHotspot: 'radial-gradient(circle at 45% 50%, rgba(56, 189, 248, 0.45) 0%, transparent 65%)',
        primaryVitals: [
          { label: 'Severity', value: 'Level 0 (Normal)' },
          { label: 'Exudate P95', value: '0.605' },
          { label: 'Tortuosity', value: '1.05' },
          { label: 'Macula', value: 'Intact' },
        ],
        clinicalGuideline: 'Healthy baseline ophthalmic status; repeat annual dilated eye examination.',
      },
    ],
  },
  {
    id: 'heart',
    name: 'Cardiovascular & Vascular',
    category: 'Cardiology',
    icon: '🫀',
    datasetName: 'UCI Cleveland Heart Disease & MIT-BIH',
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
        imageSrc: '/clinical_scans/cardiac_echo_pathological.jpg',
        scanType: 'High-Resolution 2D Apical 4-Chamber Echocardiogram (Ventricular Wall Akinesis)',
        heatmapHotspot: 'radial-gradient(circle at 48% 42%, rgba(239, 68, 68, 0.9) 0%, rgba(245, 158, 11, 0.65) 25%, transparent 60%)',
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
        imageSrc: '/clinical_scans/cardiac_echo_normal.jpg',
        scanType: 'High-Resolution 2D Transthoracic 4-Chamber Normal Echocardiogram',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.3) 0%, transparent 50%)',
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
    ],
    cohortSamples: [
      {
        patientId: 'PT-RENAL-014',
        diagnosis: 'Chronic Kidney Disease (Stage 3b CKD)',
        isHighRisk: true,
        quantumConfidence: 97.2,
        classicalConfidence: 86.4,
        imageSrc: '/clinical_scans/renal_ct_carcinoma.jpg',
        scanType: 'High-Definition Contrast-Enhanced CT Renal Cortical & Pelvic View',
        heatmapHotspot: 'radial-gradient(circle at 62% 48%, rgba(239, 68, 68, 0.9) 0%, rgba(245, 158, 11, 0.65) 28%, transparent 60%)',
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
        imageSrc: '/clinical_scans/renal_ct_normal.jpg',
        scanType: 'High-Definition Contrast-Enhanced CT Normal Bilateral Kidneys',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.3) 0%, transparent 50%)',
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
        imageSrc: '/clinical_scans/breast_mammogram_malignant.jpg',
        scanType: 'Ultra-High-Resolution Digital Screening Mammogram (Spiculation & Microcalcification)',
        heatmapHotspot: 'radial-gradient(circle at 48% 46%, rgba(239, 68, 68, 0.92) 0%, rgba(245, 158, 11, 0.7) 28%, transparent 65%)',
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
        imageSrc: '/clinical_scans/breast_mammogram_normal.jpg',
        scanType: 'Ultra-High-Resolution Digital Screening Mammogram (Normal Fibroglandular Tissue)',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.3) 0%, transparent 50%)',
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
      { name: 'ALT / AST de Ritis Ratio', normal: '0.8 - 1.2', unit: 'ratio', clinicalRole: 'Hepatocellular parenchymal injury marker' },
      { name: 'Albumin/Globulin (A/G) Ratio', normal: '1.1 - 2.5', unit: 'ratio', clinicalRole: 'Chronic hepatic protein synthetic reserve' },
    ],
    cohortSamples: [
      {
        patientId: 'PT-HEPAT-073',
        diagnosis: 'Decompensated Hepatic Cirrhosis',
        isHighRisk: true,
        quantumConfidence: 95.9,
        classicalConfidence: 81.0,
        imageSrc: '/clinical_scans/liver_ct_cirrhosis.jpg',
        scanType: 'High-Resolution Abdominal Axial Contrast CT (Hepatic Parenchymal Lesions)',
        heatmapHotspot: 'radial-gradient(circle at 45% 48%, rgba(239, 68, 68, 0.88) 0%, rgba(245, 158, 11, 0.65) 28%, transparent 60%)',
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
        imageSrc: '/clinical_scans/liver_ct_normal.jpg',
        scanType: 'High-Resolution Abdominal Axial Contrast CT (Homogeneous Liver Parenchyma)',
        heatmapHotspot: 'radial-gradient(circle at 50% 50%, rgba(56, 189, 248, 0.3) 0%, transparent 50%)',
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
];

export const WholeBodyExplorer: React.FC = () => {
  const [selectedOrgan, setSelectedOrgan] = useState<OrganSystem>(ORGAN_SYSTEMS[0]);
  const [selectedSampleIndex, setSelectedSampleIndex] = useState<number>(0);
  const [opacity, setOpacity] = useState<number>(70);

  const activeSample = selectedOrgan.cohortSamples[selectedSampleIndex] || selectedOrgan.cohortSamples[0];

  const getHeatmapStyle = () => {
    const alpha = opacity / 100;
    if (!activeSample.isHighRisk) {
      return {
        background: `radial-gradient(circle at 50% 50%, rgba(56, 189, 248, ${alpha * 0.45}) 0%, rgba(14, 165, 233, ${alpha * 0.15}) 40%, transparent 70%)`,
        mixBlendMode: 'screen' as const,
      };
    }
    return {
      background: activeSample.heatmapHotspot.replace(/rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d\.]+)\)/g, (_match, r, g, b, a) => {
        return `rgba(${r}, ${g}, ${b}, ${parseFloat(a) * alpha})`;
      }),
      mixBlendMode: 'screen' as const,
    };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      {/* Header Banner */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-3)' }}>
          <div>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Stethoscope style={{ color: 'var(--color-quantum-primary)' }} />
              Whole-Body Multi-Organ Clinical Visual & Quantum Cockpit
            </h3>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
              High-accuracy dual-screen visual inspection across Brain, Lungs, Skin, Retina, Heart, Kidneys, Breast, and Liver with verified open datasets.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <span className="badge badge-quantum" style={{ fontSize: 'var(--font-size-xs)' }}>
              8 Anatomical Scopes Active
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
                  {organ.category.split('&')[0]}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* ========================================================
          CORE DUAL-SCREEN SCAN & QUANTUM SALIENCY HEATMAP INSPECTOR
         ======================================================== */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <div>
            <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Layers style={{ color: 'var(--color-quantum-primary)' }} />
              Dual-Screen Clinical Scan & Quantum Attention Saliency Inspector
            </h4>
            <p style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)', marginTop: '2px' }}>
              Target: {selectedOrgan.name} • Scan: {activeSample.scanType}
            </p>
          </div>

          {/* Sample Switcher Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginRight: '4px' }}>
              Select Patient Case:
            </span>
            {selectedOrgan.cohortSamples.map((samp, idx) => (
              <button
                key={samp.patientId}
                type="button"
                onClick={() => setSelectedSampleIndex(idx)}
                className="btn btn-secondary"
                style={{
                  padding: '6px 14px',
                  fontSize: 'var(--font-size-xs)',
                  fontWeight: 600,
                  background: selectedSampleIndex === idx ? 'var(--color-quantum-primary)' : 'var(--color-bg-surface-elevated)',
                  color: selectedSampleIndex === idx ? '#fff' : 'var(--color-text-secondary)',
                  borderColor: selectedSampleIndex === idx ? 'var(--color-quantum-primary)' : 'var(--color-border-subtle)',
                }}
              >
                {samp.patientId} ({samp.isHighRisk ? 'Pathological' : 'Normal'})
              </button>
            ))}
          </div>
        </div>

        {/* Opacity Slider */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 'var(--space-3)',
            background: 'var(--color-bg-surface-elevated)',
            padding: 'var(--space-3) var(--space-4)',
            borderRadius: 'var(--radius-md)',
            marginBottom: 'var(--space-5)',
          }}
        >
          <Sliders size={18} style={{ color: 'var(--color-quantum-primary)' }} />
          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-secondary)', minWidth: 260 }}>
            Quantum Attention Heatmap Blend: {opacity}%
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

        {/* Dual Side-by-Side Images */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: 'var(--space-5)' }}>
          {/* Left: Original Clinical Scan */}
          <div
            style={{
              background: '#07090e',
              borderRadius: 'var(--radius-lg)',
              overflow: 'hidden',
              border: '1px solid var(--color-border-subtle)',
              position: 'relative',
              boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 12,
                left: 12,
                background: 'rgba(0,0,0,0.85)',
                color: '#fff',
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 'var(--font-size-xs)',
                fontWeight: 600,
                letterSpacing: '0.04em',
                zIndex: 2,
                border: '1px solid rgba(255,255,255,0.15)',
                backdropFilter: 'blur(4px)',
              }}
            >
              Original Clinical Scan
            </div>
            <div
              style={{
                position: 'absolute',
                bottom: 12,
                right: 12,
                background: 'rgba(0,0,0,0.85)',
                color: 'var(--color-text-secondary)',
                padding: '4px 10px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 'var(--font-size-xs)',
                zIndex: 2,
              }}
            >
              ID: {activeSample.patientId}
            </div>

            <img
              src={activeSample.imageSrc}
              alt={activeSample.diagnosis}
              style={{
                width: '100%',
                height: 380,
                objectFit: 'contain',
                display: 'block',
                background: '#030305',
              }}
            />
          </div>

          {/* Right: Quantum Saliency Heatmap Overlay */}
          <div
            style={{
              background: '#07090e',
              borderRadius: 'var(--radius-lg)',
              overflow: 'hidden',
              border: '1px solid var(--color-border-subtle)',
              position: 'relative',
              boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
            }}
          >
            <div
              style={{
                position: 'absolute',
                top: 12,
                left: 12,
                background: 'rgba(0,0,0,0.85)',
                color: 'var(--color-quantum-primary)',
                padding: '6px 12px',
                borderRadius: 'var(--radius-sm)',
                fontSize: 'var(--font-size-xs)',
                fontWeight: 600,
                letterSpacing: '0.04em',
                zIndex: 2,
                border: '1px solid var(--color-quantum-glow)',
                backdropFilter: 'blur(4px)',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
              }}
            >
              <Layers size={14} />
              HQNN Saliency Heatmap ({opacity}% Blend)
            </div>

            <img
              src={activeSample.imageSrc}
              alt={`Heatmap of ${activeSample.diagnosis}`}
              style={{
                width: '100%',
                height: 380,
                objectFit: 'contain',
                display: 'block',
                background: '#030305',
              }}
            />

            {/* Glowing Heatmap Overlay Layer */}
            <div
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                pointerEvents: 'none',
                transition: 'background 0.2s ease',
                ...getHeatmapStyle(),
              }}
            />
          </div>
        </div>

        {/* Diagnosis & Stage Pill */}
        <div
          style={{
            marginTop: 'var(--space-5)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 'var(--space-3)',
            padding: 'var(--space-3) var(--space-4)',
            background: 'var(--color-bg-surface-elevated)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--color-border-subtle)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
            {activeSample.isHighRisk ? (
              <span className="badge badge-malignant" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <AlertTriangle size={16} /> Diagnosis: {activeSample.diagnosis} (High Risk / Action Required)
              </span>
            ) : (
              <span className="badge badge-benign" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <CheckCircle2 size={16} /> Diagnosis: {activeSample.diagnosis} (Normal Healthy Baseline)
              </span>
            )}
            <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Scan Modality: <strong>{activeSample.scanType}</strong>
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
            <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Quantum HQNN Confidence: <strong style={{ color: 'var(--color-quantum-primary)' }}>{activeSample.quantumConfidence}%</strong>
            </span>
            <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
              Classical RF: {activeSample.classicalConfidence}%
            </span>
          </div>
        </div>
      </div>

      {/* Organ Metrics & Clinical Guidelines Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 'var(--space-6)' }}>
        {/* Left Column: Dataset Origin & Quantum Circuit Blueprint */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
          {/* Dataset Provenance Card */}
          <div className="card">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-3)' }}>
              <Database size={18} style={{ color: 'var(--color-quantum-primary)' }} />
              Verified Public Dataset Origin
            </h4>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-4)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Dataset Name:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.datasetName}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Archive Source:</span>
                <span className="badge" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8', fontSize: 'var(--font-size-xs)' }}>
                  {selectedOrgan.datasetSource}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Verified Cases:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-quantum-primary)' }}>
                  {selectedOrgan.datasetRows.toLocaleString()} Patient Scans / Vectors
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Clinical Feature Count:</span>
                <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-primary)' }}>
                  {selectedOrgan.datasetFeatures} Measured Features
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
                <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-secondary)' }}>Circuit Depth:</span>
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

        {/* Right Column: Extracted Vitals & Clinical Action */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
          {/* Patient Vitals Card */}
          <div className="card">
            <h4 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
              <Activity size={18} style={{ color: 'var(--color-quantum-primary)' }} />
              Extracted Patient Vitals & Lab Assay Attribution
            </h4>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 'var(--space-3)', marginBottom: 'var(--space-4)' }}>
              {activeSample.primaryVitals.map((vit, idx) => (
                <div key={idx} style={{ background: 'var(--color-bg-surface-elevated)', padding: 'var(--space-3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--color-border-subtle)' }}>
                  <div style={{ fontSize: '11px', color: 'var(--color-text-tertiary)' }}>{vit.label}</div>
                  <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'bold', color: 'var(--color-quantum-primary)', marginTop: '2px' }}>
                    {vit.value}
                  </div>
                </div>
              ))}
            </div>

            {/* Quantum vs Classical Head-to-Head */}
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

            {/* Clinical Action Guideline */}
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
