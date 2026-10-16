"""Component 1: Quantum & System Telemetry HUD (Heads-Up Display)."""

import time
import urllib.request
import json
import streamlit as st
from src.quantum.backend import QuantumBackendManager


def check_api_status() -> dict:
    """Checks latency and health of the FastAPI backend server."""
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/health", headers={"User-Agent": "Streamlit-HUD"})
        with urllib.request.urlopen(req, timeout=0.3) as response:
            latency = (time.perf_counter() - t0) * 1000.0
            data = json.loads(response.read().decode())
            return {
                "online": True,
                "latency_ms": round(latency, 1),
                "backend": data.get("quantum_backend", "default.qubit"),
            }
    except Exception:
        return {"online": False, "latency_ms": None, "backend": "default.qubit (Internal)"}


def render_telemetry_hud(n_qubits: int = 4):
    """Renders the top persistent clinical telemetry bar."""
    status = check_api_status()
    mem_mb = (2 ** n_qubits * 16) / (1024.0 * 1024.0)

    api_badge = (
        f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.4);">'
        f'🟢 API Online ({status["latency_ms"]} ms)</span>'
        if status["online"]
        else '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; padding: 4px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.4);">'
        '🟡 Engine Embedded</span>'
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.85));
                    border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px;
                    padding: 10px 18px; margin-bottom: 20px; display: flex;
                    flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);">
            <div style="display: flex; align-items: center; gap: 14px;">
                <span style="font-size: 1.1rem;">⚛️</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">
                    Device: <strong style="color: #38bdf8;">PennyLane default.qubit</strong>
                </span>
                <span style="color: rgba(255, 255, 255, 0.15);">|</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">
                    Hilbert Space: <strong style="color: #a78bfa;">2^{n_qubits} = {2**n_qubits} States</strong>
                </span>
                <span style="color: rgba(255, 255, 255, 0.15);">|</span>
                <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 500;">
                    RAM Footprint: <strong style="color: #34d399;">{mem_mb:.4f} MB</strong>
                </span>
            </div>
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 0.82rem; color: #cbd5e1;">Datasets: <strong>WDBC (569)</strong> • <strong>Heart (303)</strong> • <strong>Fundus (22)</strong></span>
                {api_badge}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
