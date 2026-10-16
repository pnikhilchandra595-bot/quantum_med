"""QuantumHealth REST API Backend Server.

FastAPI backend exposing classical and quantum disease prediction,
clinical biomarker explanation (XAI), and benchmark telemetry.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel, Field

from src.models.factory import ModelFactory
from src.data.loader import AVAILABLE_DATASETS, load_dataset
from src.data.fundus_loader import load_fundus_dataset
from src.quantum.backend import QuantumBackendManager
from src.xai.quantum_shap import QuantumKernelSHAPExplainer

app = FastAPI(
    title="QuantumHealth Hybrid QML Backend API",
    description="REST API for Hybrid Quantum-Classical Early Disease Detection Platform (HQML-EDD)",
    version="1.0.0",
)

# Enable CORS for Streamlit and external frontend web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Schemas
class PredictionRequest(PydanticBaseModel):
    model_name: str = Field(default="qsvc", description="Model architecture: 'svm', 'rf', 'xgb', 'mlp', 'qsvc', 'vqc', 'hqnn'")
    dataset: str = Field(default="wdbc", description="Dataset identifier: 'wdbc', 'heart', 'fundus'")
    features: List[float] = Field(..., description="Continuous clinical biomarker feature vector")
    n_qubits: int = Field(default=4, ge=2, le=8, description="Number of qubits for quantum dimensionality")


class PredictionResponse(PydanticBaseModel):
    model: str
    prediction_label: str
    prediction_probability: float
    is_malignant_or_diseased: bool
    latency_ms: float
    feature_dimension: int


class ExplainRequest(PydanticBaseModel):
    model_name: str = Field(default="qsvc", description="Model name to explain")
    dataset: str = Field(default="wdbc", description="Dataset identifier")
    sample: List[float] = Field(..., description="Patient biomarker values")
    n_qubits: int = Field(default=4, ge=2, le=8)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get("/")
def root():
    return {
        "platform": "QuantumHealth Hybrid Quantum-Classical Platform",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "datasets": "/api/v1/datasets",
            "models": "/api/v1/models",
            "benchmark": "/api/v1/benchmark",
            "predict": "/api/v1/predict",
            "explain": "/api/v1/explain",
        },
    }


@app.get("/health")
def health_check():
    backend = QuantumBackendManager(backend_name="default.qubit", max_qubits=8)
    dev = backend.get_device(wires=4)
    return {
        "status": "healthy",
        "quantum_backend": "default.qubit",
        "quantum_wires": 4,
        "device_initialized": dev is not None,
        "available_datasets": list(AVAILABLE_DATASETS) + ["fundus"],
        "available_models": ModelFactory.list_models(),
    }


@app.get("/api/v1/datasets")
def list_datasets():
    datasets_info = {
        "wdbc": {
            "name": "Wisconsin Diagnostic Breast Cancer (WDBC)",
            "samples": 569,
            "raw_features": 30,
            "target": "Malignant vs Benign",
        },
        "heart": {
            "name": "Heart Disease (Cleveland Clinical)",
            "samples": 303,
            "raw_features": 13,
            "target": "Presence vs Absence of Heart Disease",
        },
        "fundus": {
            "name": "Clinical Diabetic Retinopathy Fundus Photos",
            "samples": 22,
            "biomarkers_extracted": 11,
            "target": "Diabetic Retinopathy vs Healthy",
        },
    }
    return {"datasets": datasets_info}


@app.get("/api/v1/models")
def list_models():
    return {
        "classical_models": [
            {"id": "svm", "name": "Support Vector Machine (RBF/Linear)", "type": "classical"},
            {"id": "rf", "name": "Random Forest Classifier", "type": "classical"},
            {"id": "xgb", "name": "eXtreme Gradient Boosting (XGBoost)", "type": "classical"},
            {"id": "mlp", "name": "Multi-Layer Perceptron", "type": "classical"},
        ],
        "quantum_models": [
            {"id": "qsvc", "name": "Quantum Support Vector Classifier (QSVC)", "type": "quantum", "kernel": "Fidelity Kernel"},
            {"id": "vqc", "name": "Variational Quantum Classifier (VQC)", "type": "quantum", "ansatz": "Strongly Entangling Layers"},
            {"id": "hqnn", "name": "Hybrid Quantum Neural Network (HQNN)", "type": "quantum", "layer": "PennyLane TorchLayer"},
        ],
    }


@app.get("/api/v1/benchmark")
def get_benchmark_results():
    csv_path = WORKSPACE_ROOT / "reports" / "tables" / "benchmark_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return {"benchmark_summary": df.to_dict(orient="records")}
    return {"message": "Run scripts/run_master_benchmark.py to generate benchmark results"}


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    model_name = req.model_name.lower().strip()
    if model_name not in ModelFactory.ALL_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model_name}'. Choose from: {ModelFactory.ALL_MODELS}",
        )

    t0 = time.perf_counter()

    # Load small reference training data to fit the model quickly
    if req.dataset.lower() == "fundus":
        X_train, y_train, _, _ = load_fundus_dataset()
        # Pad or slice features to match n_qubits
        X_train = X_train[:, :req.n_qubits]
    else:
        ds = load_dataset(req.dataset, n_qubits=req.n_qubits)
        X_train, y_train = ds["X_train"], ds["y_train"]

    model = ModelFactory.create_model(model_name, n_qubits=req.n_qubits, epochs=5)
    model.fit(X_train, y_train)

    feat_array = np.array(req.features, dtype=float).reshape(1, -1)
    if feat_array.shape[1] > req.n_qubits:
        feat_array = feat_array[:, :req.n_qubits]
    elif feat_array.shape[1] < req.n_qubits:
        # Zero-pad if fewer features provided
        padded = np.zeros((1, req.n_qubits))
        padded[0, :feat_array.shape[1]] = feat_array[0]
        feat_array = padded

    probs = model.predict_proba(feat_array)[0]
    prob_disease = float(probs[1]) if len(probs) > 1 else float(probs[0])
    is_positive = prob_disease >= 0.5
    label = "DISEASED / POSITIVE" if is_positive else "HEALTHY / NEGATIVE"

    latency = (time.perf_counter() - t0) * 1000.0

    return PredictionResponse(
        model=model_name.upper(),
        prediction_label=label,
        prediction_probability=round(prob_disease, 4),
        is_malignant_or_diseased=is_positive,
        latency_ms=round(latency, 2),
        feature_dimension=req.n_qubits,
    )


@app.post("/api/v1/explain")
def explain(req: ExplainRequest):
    model_name = req.model_name.lower().strip()
    if model_name not in ModelFactory.ALL_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model: {model_name}")

    ds = load_dataset(req.dataset, n_qubits=req.n_qubits)
    X_train, y_train = ds["X_train"], ds["y_train"]

    model = ModelFactory.create_model(model_name, n_qubits=req.n_qubits, epochs=3)
    model.fit(X_train, y_train)

    sample = np.array(req.sample, dtype=float).flatten()
    if len(sample) > req.n_qubits:
        sample = sample[:req.n_qubits]
    elif len(sample) < req.n_qubits:
        padded = np.zeros(req.n_qubits)
        padded[:len(sample)] = sample
        sample = padded

    feature_names = [f"Biomarker_{i+1}" for i in range(req.n_qubits)]

    explainer = QuantumKernelSHAPExplainer(
        model=model,
        background_data=X_train,
        n_background_samples=min(5, len(X_train)),
    )
    report = explainer.generate_patient_attribution_report(
        sample=sample,
        feature_names=feature_names,
        top_k=req.top_k,
    )
    return {"explanation_report": report}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
