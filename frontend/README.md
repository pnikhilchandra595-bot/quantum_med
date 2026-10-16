# QuantumHealth Clinical Platform — Front-End Application

A production-grade, standalone React + Vite + TypeScript diagnostic intelligence interface for the **QuantumHealth (HQML-EDD)** platform. Designed for oncologists, ophthalmologists, and quantum researchers without generic "AI-template" tropes.

---

## 1. Architecture & Tech Stack

- **Framework:** React 19 + Vite 8 (TypeScript)
- **Styling:** Vanilla CSS Design System with CSS Custom Properties (`src/styles/tokens.css`)
- **Design Paradigm:** Dark Technical Clinical Glassmorphism & Light Mode (WCAG AA compliant)
- **Icons:** `lucide-react`
- **Backend API:** FastAPI REST Service running on `http://localhost:8000/api/v1/`
- **Port:** `http://localhost:5173/`

---

## 2. Quickstart & Local Execution

### Prerequisites
- Node.js >= 18 (Tested on Node v22.16.0)
- Python 3.13 running the FastAPI backend (`py -3.13 -m uvicorn src.api.server:app --port 8000`)

### Installation & Run
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the Vite local development server
npm run dev

# Or directly:
npx vite --host --port 5173
```
Access the application locally at: **[http://localhost:5173/](http://localhost:5173/)**

---

## 3. Directory Structure

```text
frontend/
├── dist/                  # Production optimized bundle (606ms build)
├── public/                # Static clinical assets
├── src/
│   ├── assets/            # SVG illustrations and logos
│   ├── components/        # Reusable diagnostic modules
│   │   ├── ArchitectureView.tsx   # 3-Tier topology & model catalogue
│   │   ├── BlochSphere3D.tsx      # 3D Bloch sphere & Hilbert space projection
│   │   ├── HospitalDossier.tsx    # ISO 13485 clinical PDF dossier preview & print
│   │   ├── Navigation.tsx         # Tab bar navigation & dark/light theme switch
│   │   ├── NoiseStressTester.tsx  # Sensor noise degradation retention curves
│   │   ├── RadarArena.tsx         # 5-Axis clinical spider chart & model filter
│   │   ├── RetinaViewer.tsx       # Fundus photo inspection & attention heatmap
│   │   ├── TelemetryHUD.tsx       # Live simulator memory & FastAPI latency HUD
│   │   └── WhatIfSimulator.tsx    # Live counterfactual patient slider engine
│   ├── lib/
│   │   └── api.ts                 # FastAPI client, data schemas, and fallback mocks
│   ├── styles/
│   │   └── tokens.css             # Unified design system tokens (colors, radii, spacing)
│   ├── App.tsx                    # Main shell wiring HUD, navigation, and views
│   ├── index.css                  # Global resets, glassmorphic card classes, touch targets
│   └── main.tsx                   # React root entrypoint
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## 4. Key Design Decisions

1. **Tokens-First Design System (`tokens.css`)**:
   - Strict 4/8px spacing scale (`--space-1` = 4px to `--space-16` = 64px) preventing arbitrary pixel drift.
   - Fluid typography with a non-negotiable 16px minimum body size for clinical readability.
   - High-contrast, accessibility-checked colors:
     - Quantum Core Cyan (`#38bdf8`)
     - Hilbert Phase Indigo (`#818cf8`)
     - Benign/Success Emerald (`#10b981`)
     - Malignant/High-Risk Crimson (`#f43f5e`)

2. **Zero-Lag Interactive Projections**:
   - The 3D Bloch Sphere and 5-Axis Radar are implemented using lightweight SVG and Canvas geometry, avoiding heavy external bundle bloat and rendering at 60 FPS.

3. **Clinician Ergonomics & Touch Targets**:
   - Minimum 44x44px touch targets on all interactive selectors and tab pills.
   - Dual-gauge comparative risk display alerting clinicians to borderline discordance between classical and quantum boundaries.

---

## 5. Next UX Improvements

1. **DICOM File Drag-and-Drop:** Native client-side parsing of medical `.dcm` files directly into the fundus/biopsy pipeline.
2. **Side-by-Side Multi-Patient Cohort Comparison:** Ability to pin two patient dossiers simultaneously for comparative differential diagnosis.
3. **Web Worker Offloading:** Offload complex synthetic parameter sweeps and Monte Carlo noise simulations to background Web Workers to guarantee zero main-thread hitching.
