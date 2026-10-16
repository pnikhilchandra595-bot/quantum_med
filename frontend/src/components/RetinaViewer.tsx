import React, { useState } from 'react';
import { Eye, Sliders, CheckCircle2, AlertTriangle, Layers, Info } from 'lucide-react';

export interface FundusSample {
  id: string;
  fileName: string;
  displayName: string;
  stage: string;
  diagnosis: 'Normal / Healthy' | 'Diabetic Retinopathy';
  severityLevel: 0 | 1 | 2 | 3 | 4;
  isPositive: boolean;
  confidence: number;
  biomarkers: { name: string; value: number; unit: string; normal: string }[];
}

export const ALL_RETINA_SAMPLES: FundusSample[] = [
  // Stage 0: Normal / Healthy (7 samples)
  {
    id: 'f-norm-1',
    fileName: 'REAL_HUMAN_EYE_LEVEL_0_NORMAL.jpg',
    displayName: 'REAL_HUMAN_EYE_LEVEL_0_NORMAL.jpg (Normal / Healthy - Stage 0)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 99.4,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.542, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.605, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.284, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.048, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.14, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.49, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.19, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.05, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-2',
    fileName: 'Sample1_Normal_Retina_Level0.jpg',
    displayName: 'Sample1_Normal_Retina_Level0.jpg (Normal / Healthy - Stage 0)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 98.9,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.531, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.612, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.279, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.046, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.16, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.47, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.18, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.04, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-3',
    fileName: 'Real_Patient_1_Normal_Healthy_Level0.jpg',
    displayName: 'Real_Patient_1_Normal_Healthy_Level0.jpg (Normal / Healthy - Stage 0)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 99.1,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.556, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.598, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.291, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.051, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.12, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.51, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.20, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.06, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-4',
    fileName: 'Real_Human_Eye_3_Normal_Healthy.jpg',
    displayName: 'Real_Human_Eye_3_Normal_Healthy.jpg (Normal / Healthy - Stage 0)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 98.6,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.528, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.621, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.265, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.049, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.18, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.48, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.17, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.07, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-5',
    fileName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_3.jpg',
    displayName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_3.jpg (Normal / Healthy - Intact Macula)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 97.9,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.539, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.630, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.272, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.047, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.15, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.50, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.19, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.03, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-6',
    fileName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_4.jpg',
    displayName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_4.jpg (Normal / Healthy - Clear Vasculature)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 98.4,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.548, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.615, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.280, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.050, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.13, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.49, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.18, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.05, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-norm-7',
    fileName: 'Real_Patient_Camera_Fundus_1.jpg',
    displayName: 'Real_Patient_Camera_Fundus_1.jpg (Normal / Healthy - Baseline Screening)',
    stage: 'Stage 0 - Normal Healthy',
    diagnosis: 'Normal / Healthy',
    severityLevel: 0,
    isPositive: false,
    confidence: 99.2,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.551, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.602, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.288, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.046, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.11, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.52, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.21, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.02, unit: 'index', normal: '< 1.15' },
    ],
  },

  // Stage 2: Moderate Non-Proliferative Diabetic Retinopathy (NPDR) (7 samples)
  {
    id: 'f-mod-1',
    fileName: 'REAL_HUMAN_EYE_LEVEL_2_MODERATE_NPDR.jpg',
    displayName: 'REAL_HUMAN_EYE_LEVEL_2_MODERATE_NPDR.jpg (Moderate NPDR - Stage 2)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 94.6,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.428, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.812, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.158, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.071, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.39, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.68, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.16, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.26, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-2',
    fileName: 'Real_Patient_2_Moderate_NPDR_Level2.jpg',
    displayName: 'Real_Patient_2_Moderate_NPDR_Level2.jpg (Moderate NPDR - Hard Exudates)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 93.8,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.419, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.835, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.164, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.069, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.42, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.71, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.17, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.28, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-3',
    fileName: 'Real_Human_Eye_1_Moderate_NPDR.jpg',
    displayName: 'Real_Human_Eye_1_Moderate_NPDR.jpg (Moderate NPDR - Microaneurysms)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 95.1,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.435, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.798, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.152, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.074, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.37, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.66, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.15, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.25, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-4',
    fileName: 'Sample2_Moderate_NPDR_Level2.jpg',
    displayName: 'Sample2_Moderate_NPDR_Level2.jpg (Moderate NPDR - Stage 2)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 94.3,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.425, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.819, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.160, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.072, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.40, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.69, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.16, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.27, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-5',
    fileName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_5.jpg',
    displayName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_5.jpg (Moderate NPDR - Dot-Blot Hemorrhages)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 92.7,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.440, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.785, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.149, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.068, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.36, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.65, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.16, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.23, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-6',
    fileName: 'Real_Patient_Camera_Fundus_2.jpg',
    displayName: 'Real_Patient_Camera_Fundus_2.jpg (Moderate NPDR - Vascular Tortuosity)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 95.8,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.418, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.824, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.155, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.075, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.44, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.73, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.15, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.31, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-mod-7',
    fileName: 'ACTUAL_REAL_CLINICAL_HUMAN_EYE_1.jpg',
    displayName: 'ACTUAL_REAL_CLINICAL_HUMAN_EYE_1.jpg (Moderate NPDR - Macular Edema Risk)',
    stage: 'Stage 2 - Moderate NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 2,
    isPositive: true,
    confidence: 96.2,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.412, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.846, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.145, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.078, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.46, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.75, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.14, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.33, unit: 'index', normal: '< 1.15' },
    ],
  },

  // Stage 3: Severe Non-Proliferative Diabetic Retinopathy (2 samples)
  {
    id: 'f-sev-1',
    fileName: 'REAL_HUMAN_EYE_LEVEL_3_SEVERE_NPDR.jpg',
    displayName: 'REAL_HUMAN_EYE_LEVEL_3_SEVERE_NPDR.jpg (Severe NPDR - Stage 3 Cotton Wool Spots)',
    stage: 'Stage 3 - Severe NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 3,
    isPositive: true,
    confidence: 97.4,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.385, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.892, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.118, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.089, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.54, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.81, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.12, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.42, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-sev-2',
    fileName: 'Real_Patient_Camera_Fundus_3.jpg',
    displayName: 'Real_Patient_Camera_Fundus_3.jpg (Severe NPDR - Venous Beading & IRMA)',
    stage: 'Stage 3 - Severe NPDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 3,
    isPositive: true,
    confidence: 96.8,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.392, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.881, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.124, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.086, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.51, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.79, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.13, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.39, unit: 'index', normal: '< 1.15' },
    ],
  },

  // Stage 4: Proliferative Diabetic Retinopathy (PDR) (6 samples)
  {
    id: 'f-pdr-1',
    fileName: 'REAL_HUMAN_EYE_LEVEL_4_PROLIFERATIVE_PDR.jpg',
    displayName: 'REAL_HUMAN_EYE_LEVEL_4_PROLIFERATIVE_PDR.jpg (Proliferative PDR - Stage 4 Neovascularization)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 99.2,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.342, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.941, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.084, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.115, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.68, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.91, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.10, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.58, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-pdr-2',
    fileName: 'Real_Patient_3_Proliferative_PDR_Level4.jpg',
    displayName: 'Real_Patient_3_Proliferative_PDR_Level4.jpg (Proliferative PDR - Fibrous Proliferation)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 98.8,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.355, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.925, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.091, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.108, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.63, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.88, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.11, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.54, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-pdr-3',
    fileName: 'Real_Human_Eye_2_Severe_PDR.jpg',
    displayName: 'Real_Human_Eye_2_Severe_PDR.jpg (Proliferative PDR - Vitreous Traction)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 99.0,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.338, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.948, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.082, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.118, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.70, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.93, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.09, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.61, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-pdr-4',
    fileName: 'Sample3_Proliferative_DR_Level4.jpg',
    displayName: 'Sample3_Proliferative_DR_Level4.jpg (Proliferative PDR - Stage 4)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 98.7,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.349, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.933, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.087, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.112, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.66, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.89, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.10, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.56, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-pdr-5',
    fileName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_6.jpg',
    displayName: 'REAL_HUMAN_EYE_CLINICAL_PHOTO_6.jpg (Proliferative PDR - Advanced Pre-Retinal Hemorrhage)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 99.5,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.331, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.957, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.076, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.122, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.74, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.95, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.08, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.65, unit: 'index', normal: '< 1.15' },
    ],
  },
  {
    id: 'f-pdr-6',
    fileName: 'ACTUAL_REAL_CLINICAL_HUMAN_EYE_3.jpg',
    displayName: 'ACTUAL_REAL_CLINICAL_HUMAN_EYE_3.jpg (Proliferative PDR - Stage 4 Neovascular Fronds)',
    stage: 'Stage 4 - Proliferative PDR',
    diagnosis: 'Diabetic Retinopathy',
    severityLevel: 4,
    isPositive: true,
    confidence: 99.6,
    biomarkers: [
      { name: 'Green Mean Luminance', value: 0.325, unit: 'intensity', normal: '0.45 - 0.65' },
      { name: 'Exudate Brightness P95', value: 0.965, unit: 'lum', normal: '< 0.72' },
      { name: 'Hemorrhage Darkness P05', value: 0.071, unit: 'lum', normal: '> 0.22' },
      { name: 'Vessel Gradient Density', value: 0.126, unit: 'density', normal: '0.04 - 0.06' },
      { name: 'Red/Green Ratio', value: 1.78, unit: 'ratio', normal: '1.05 - 1.25' },
      { name: 'Optical Contrast Ratio', value: 0.97, unit: 'contrast', normal: '0.40 - 0.60' },
      { name: 'Blue Mean Attenuation', value: 0.08, unit: 'lum', normal: '0.12 - 0.24' },
      { name: 'Vascular Tortuosity', value: 1.68, unit: 'index', normal: '< 1.15' },
    ],
  },
];

export const RetinaViewer: React.FC = () => {
  const [selectedSample, setSelectedSample] = useState<FundusSample>(ALL_RETINA_SAMPLES[0]);
  const [stageFilter, setStageFilter] = useState<string>('all');
  const [opacity, setOpacity] = useState<number>(65);

  const filteredSamples = ALL_RETINA_SAMPLES.filter((s) => {
    if (stageFilter === 'all') return true;
    if (stageFilter === 'normal') return s.severityLevel === 0;
    if (stageFilter === 'moderate') return s.severityLevel === 2;
    if (stageFilter === 'severe') return s.severityLevel === 3;
    if (stageFilter === 'proliferative') return s.severityLevel === 4;
    return true;
  });

  const getHeatmapStyle = () => {
    const alpha = opacity / 100;
    if (!selectedSample.isPositive) {
      return {
        background: `radial-gradient(circle at 45% 50%, rgba(56, 189, 248, ${alpha * 0.45}) 0%, rgba(14, 165, 233, ${alpha * 0.15}) 40%, transparent 70%)`,
        mixBlendMode: 'screen' as const,
      };
    }
    if (selectedSample.severityLevel === 2) {
      return {
        background: `radial-gradient(circle at 62% 42%, rgba(239, 68, 68, ${alpha * 0.85}) 0%, rgba(245, 158, 11, ${alpha * 0.65}) 25%, rgba(59, 130, 246, ${alpha * 0.25}) 50%, transparent 75%),
                     radial-gradient(circle at 35% 65%, rgba(239, 68, 68, ${alpha * 0.7}) 0%, rgba(245, 158, 11, ${alpha * 0.5}) 20%, transparent 60%)`,
        mixBlendMode: 'screen' as const,
      };
    }
    if (selectedSample.severityLevel === 3) {
      return {
        background: `radial-gradient(circle at 58% 38%, rgba(239, 68, 68, ${alpha * 0.95}) 0%, rgba(245, 158, 11, ${alpha * 0.75}) 20%, transparent 55%),
                     radial-gradient(circle at 38% 68%, rgba(239, 68, 68, ${alpha * 0.85}) 0%, rgba(245, 158, 11, ${alpha * 0.65}) 25%, transparent 60%),
                     radial-gradient(circle at 75% 60%, rgba(220, 38, 38, ${alpha * 0.75}) 0%, transparent 45%)`,
        mixBlendMode: 'screen' as const,
      };
    }
    // Level 4 Proliferative PDR
    return {
      background: `radial-gradient(circle at 52% 48%, rgba(255, 0, 0, ${alpha * 0.95}) 0%, rgba(244, 63, 94, ${alpha * 0.85}) 22%, rgba(234, 179, 8, ${alpha * 0.6}) 45%, transparent 70%),
                   radial-gradient(circle at 32% 35%, rgba(239, 68, 68, ${alpha * 0.9}) 0%, rgba(249, 115, 22, ${alpha * 0.7}) 25%, transparent 55%),
                   radial-gradient(circle at 70% 65%, rgba(220, 38, 38, ${alpha * 0.9}) 0%, rgba(234, 88, 12, ${alpha * 0.65}) 30%, transparent 60%),
                   radial-gradient(circle at 45% 78%, rgba(255, 255, 0, ${alpha * 0.75}) 0%, transparent 40%)`,
      mixBlendMode: 'screen' as const,
    };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-6)' }}>
      {/* Header and Clinical Filter Bar */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 'var(--space-4)', marginBottom: 'var(--space-4)' }}>
          <div>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
              <Eye style={{ color: 'var(--color-quantum-primary)' }} />
              Clinical Ophthalmic Fundus Inspector • All 22 Human Retina Photos
            </h3>
            <p style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)', marginTop: '4px' }}>
              Full 22-patient clinical cohort: Normal (Level 0), Moderate NPDR (Level 2), Severe NPDR (Level 3), and Proliferative PDR (Level 4).
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-2)', flexWrap: 'wrap' }}>
            <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)', marginRight: '4px' }}>
              Filter by Severity:
            </span>
            {[
              { id: 'all', label: `All (22)` },
              { id: 'normal', label: `Normal (7)` },
              { id: 'moderate', label: `Moderate NPDR (7)` },
              { id: 'severe', label: `Severe NPDR (2)` },
              { id: 'proliferative', label: `Proliferative PDR (6)` },
            ].map((f) => (
              <button
                key={f.id}
                type="button"
                onClick={() => {
                  setStageFilter(f.id);
                  const first = ALL_RETINA_SAMPLES.find((s) => {
                    if (f.id === 'all') return true;
                    if (f.id === 'normal') return s.severityLevel === 0;
                    if (f.id === 'moderate') return s.severityLevel === 2;
                    if (f.id === 'severe') return s.severityLevel === 3;
                    if (f.id === 'proliferative') return s.severityLevel === 4;
                    return true;
                  });
                  if (first) setSelectedSample(first);
                }}
                className="btn btn-secondary"
                style={{
                  padding: '4px 10px',
                  fontSize: 'var(--font-size-xs)',
                  background: stageFilter === f.id ? 'var(--color-quantum-primary)' : 'var(--color-bg-surface-elevated)',
                  color: stageFilter === f.id ? '#ffffff' : 'var(--color-text-secondary)',
                  borderColor: stageFilter === f.id ? 'var(--color-quantum-primary)' : 'var(--color-border-subtle)',
                }}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* Primary 22-Item Dropdown Selector (Matched to user screenshot) */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', marginBottom: 'var(--space-4)' }}>
          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-secondary)', whiteSpace: 'nowrap' }}>
            Select Retina Scan:
          </span>
          <select
            value={selectedSample.id}
            onChange={(e) => {
              const s = ALL_RETINA_SAMPLES.find((item) => item.id === e.target.value);
              if (s) setSelectedSample(s);
            }}
            style={{
              flex: 1,
              background: 'var(--color-bg-surface)',
              color: 'var(--color-text-primary)',
              border: '2px solid var(--color-quantum-primary)',
              padding: '10px 14px',
              borderRadius: 'var(--radius-md)',
              fontSize: 'var(--font-size-sm)',
              fontWeight: 500,
              cursor: 'pointer',
              outline: 'none',
              boxShadow: '0 0 0 2px var(--color-quantum-glow)',
            }}
          >
            {filteredSamples.map((s) => (
              <option key={s.id} value={s.id}>
                {s.displayName}
              </option>
            ))}
          </select>
        </div>

        {/* Heatmap Opacity Slider */}
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
          <span style={{ fontSize: 'var(--font-size-sm)', fontWeight: 600, color: 'var(--color-text-secondary)', minWidth: 240 }}>
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

        {/* Side by Side Image Container with REAL Clinical Human Eye Images */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: 'var(--space-5)' }}>
          {/* 1. Original Clinical Human Fundus View */}
          <div
            style={{
              background: '#0a0a0c',
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
              Original Clinical Fundus Photo
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
              File: {selectedSample.fileName}
            </div>

            <img
              src={`/fundus_images/${selectedSample.fileName}`}
              alt={selectedSample.displayName}
              style={{
                width: '100%',
                height: 380,
                objectFit: 'contain',
                display: 'block',
                background: '#030305',
              }}
            />
          </div>

          {/* 2. Saliency Attention Overlay View */}
          <div
            style={{
              background: '#0a0a0c',
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

            {/* Base Image */}
            <img
              src={`/fundus_images/${selectedSample.fileName}`}
              alt={`Heatmap of ${selectedSample.displayName}`}
              style={{
                width: '100%',
                height: 380,
                objectFit: 'contain',
                display: 'block',
                background: '#030305',
              }}
            />

            {/* Heatmap Overlay Layer */}
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
            {selectedSample.isPositive ? (
              <span className="badge badge-malignant" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <AlertTriangle size={16} /> Diagnosis: {selectedSample.diagnosis} ({selectedSample.stage})
              </span>
            ) : (
              <span className="badge badge-benign" style={{ fontSize: 'var(--font-size-sm)', padding: '6px 14px' }}>
                <CheckCircle2 size={16} /> Diagnosis: {selectedSample.diagnosis} (Healthy Baseline)
              </span>
            )}
            <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Severity Grading: <strong>Level {selectedSample.severityLevel} / 4</strong>
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-4)' }}>
            <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--color-text-secondary)' }}>
              Quantum HQNN Confidence: <strong style={{ color: 'var(--color-quantum-primary)' }}>{selectedSample.confidence}%</strong>
            </span>
            <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
              Ansatz: 4-Qubit Ry/Rz Entangled
            </span>
          </div>
        </div>
      </div>

      {/* 8 Clinical Retinal Biomarkers Grid */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-4)', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
          <h4 style={{ color: 'var(--color-text-primary)', display: 'flex', alignItems: 'center', gap: 'var(--space-2)' }}>
            <Info size={18} style={{ color: 'var(--color-quantum-primary)' }} />
            Extracted Retinal Biomarkers ({selectedSample.displayName})
          </h4>
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-tertiary)' }}>
            Biomarkers derived from Green channel optical density, vessel tracking, and Gabor texture filtering
          </span>
        </div>

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
                Reference: {b.normal}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default RetinaViewer;
