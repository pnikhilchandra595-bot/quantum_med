# ==========================================
# Stage 1: Build & Dependencies
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compilation dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install CPU PyTorch first to prevent massive CUDA download in lightweight container
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Minimal Runtime Environment
# ==========================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Team QuantumHealth <research@quantumhealth.org>"
LABEL version="1.0.0"
LABEL description="Hybrid Quantum-Classical ML Clinical Platform"

WORKDIR /app

# Install OpenMP runtime required by scikit-learn, PyTorch, and PennyLane
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged application user
RUN useradd -m -u 1001 appuser && \
    mkdir -p /app/checkpoints /app/outputs /app/data && \
    chown -R appuser:appuser /app

# Copy application codebase
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser cli.py /app/cli.py
COPY --chown=appuser:appuser pyproject.toml /app/pyproject.toml

USER appuser

# Healthcheck to verify Streamlit responsiveness
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
