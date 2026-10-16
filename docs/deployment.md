# Clinical Deployment & Production Guide

This document specifies the enterprise and clinical deployment architecture for the **QuantumHealth: Hybrid Quantum-Classical Clinical Diagnostics Platform**. It details workstation setup, Docker containerization, Kubernetes orchestration, physical Quantum Processing Unit (QPU) cloud integrations (IBM Quantum and AWS Braket), and regulatory compliance frameworks (HIPAA / HITRUST).

---

## 1. System Requirements & Hardware Specifications

### 1.1 Minimum & Recommended Hardware

| Component | Minimum Specification (Local / Dev) | Recommended Production (Clinic / Hospital Node) |
|---|---|---|
| **CPU Architecture** | x86_64 or ARM64 (Apple Silicon), 4 Cores | x86_64 (AVX-512 support recommended), 16+ Cores |
| **System Memory (RAM)** | 8 GB DDR4 | 32 GB DDR5 ECC (enables up to 22-qubit statevector simulation) |
| **Storage** | 10 GB available SSD space | 50 GB NVMe PCIe Gen4 SSD (Fast statevector caching) |
| **GPU Acceleration** | Optional | NVIDIA GPU with CUDA 12+ (cuQuantum / PennyLane-Lightning-GPU) |
| **Host Operating System** | Windows 10/11, Ubuntu 20.04+, macOS 12+ | Ubuntu Server 22.04 LTS / Red Hat Enterprise Linux 9 |
| **Network** | Broadband (10 Mbps) | Dedicated Gigabit LAN with TLS 1.3 egress to QPU endpoints |

### 1.2 Software Prerequisites
- **Python:** Version `3.10`, `3.11`, or `3.12` (Python `3.11-slim` is standard in containers).
- **Container Runtime:** Docker Engine `24.0+` and Docker Compose `v2.20+` or Podman `4.5+`.
- **Orchestrator:** Kubernetes `1.26+` with `kubectl` and `helm` v3.

---

## 2. Deployment Architecture Overview

```mermaid
flowchart TD
    subgraph Client ["Clinician Access Layer"]
        A1[Hospital Workstation Browser] -->|HTTPS / WSS| ING[Ingress Controller / TLS 1.3]
        A2[EMR / PACS REST Client] -->|mTLS / Bearer Token| ING
    end

    subgraph K8s ["Kubernetes Clinical Cluster (HIPAA Compliant Boundary)"]
        ING --> SVC[ClusterIP Service: port 8501]
        SVC --> POD1[QuantumHealth Pod 1]
        SVC --> POD2[QuantumHealth Pod 2]
        SVC --> PODN[QuantumHealth Pod N (HPA)]
        
        subgraph PodArch ["Container Architecture"]
            POD1 --> ST[Streamlit App :8501]
            POD1 --> MF[Unified ModelFactory]
            POD1 --> CACHE[LRU Statevector Cache]
            POD1 --> PL[PennyLane Quantum Engine]
        end
        
        PVC[(Persistent Volume: Checkpoints & Models)] -.-> POD1
        PVC -.-> POD2
    end

    subgraph Backends ["Execution Backends"]
        PL -->|Default| SIM[Local Statevector Simulator: lightning.qubit]
        PL -->|CUDA Enabled| GPU[NVIDIA cuQuantum / lightning.gpu]
        PL -->|Cloud API| IBMQ[IBM Quantum Platform: ibm_brisbane]
        PL -->|Cloud API| AWS[AWS Braket QPU: Rigetti / IonQ]
    end
```

---

## 3. Workstation Deployment (Bare Metal)

### 3.1 Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/quantumhealth/quantum-diagnostics.git
   cd quantum-diagnostics
   ```

2. **Create and activate an isolated virtual environment:**
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS (Bash)
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install production dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Verify the installation:**
   ```bash
   python cli.py --help
   pytest tests/ -v
   ```

5. **Launch the interactive clinical dashboard:**
   ```bash
   streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
   ```
   Open your browser to `http://localhost:8501`.

---

## 4. Multi-Stage Docker Deployment

The project provides a multi-stage `Dockerfile` based on `python:3.11-slim` that compiles wheels in a temporary builder stage and copies only production artifacts to an unprivileged non-root runtime environment (`appuser:appgroup`).

### 4.1 Build Docker Image

```bash
docker build -t quantumhealth:1.0.0 -t quantumhealth:latest .
```

### 4.2 Run Docker Container

```bash
docker run -d \
  --name quantumhealth-service \
  -p 8501:8501 \
  --restart unless-stopped \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50 \
  quantumhealth:latest
```

### 4.3 Container Health Check Verification

The Docker container embeds an automated healthcheck querying Streamlit's internal health endpoint:
```bash
docker inspect --format='{{json .State.Health}}' quantumhealth-service
```
Expected output: `"Status": "healthy"`.

---

## 5. Docker Compose Orchestration

For unified single-node clinical server deployments with persistent volumes:

```bash
# Launch background services
docker-compose up -d

# Inspect live container logs
docker-compose logs -f app

# Inspect container health
docker-compose ps

# Graceful shutdown
docker-compose down
```

`docker-compose.yml` mounts:
- `./checkpoints` $\to$ `/app/checkpoints` (persistent model weights).
- `./data` $\to$ `/app/data` (cohort datasets and test schemas).

---

## 6. Enterprise Kubernetes & Helm Deployment

### 6.1 Kubernetes Manifests

#### 6.1.1 Deployment (`k8s/deployment.yaml`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantumhealth-deployment
  namespace: clinical-diagnostics
  labels:
    app: quantumhealth
    tier: presentation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: quantumhealth
  template:
    metadata:
      labels:
        app: quantumhealth
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        runAsGroup: 10001
        fsGroup: 10001
      containers:
      - name: quantumhealth-app
        image: ghcr.io/quantumhealth/quantum-diagnostics:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8501
          name: http
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 15
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
        volumeMounts:
        - name: checkpoint-storage
          mountPath: /app/checkpoints
          readOnly: true
      volumes:
      - name: checkpoint-storage
        persistentVolumeClaim:
          claimName: quantumhealth-checkpoints-pvc
```

#### 6.1.2 Service & Ingress (`k8s/ingress.yaml`)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: quantumhealth-service
  namespace: clinical-diagnostics
spec:
  type: ClusterIP
  selector:
    app: quantumhealth
  ports:
  - port: 8501
    targetPort: 8501
    name: http
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: quantumhealth-ingress
  namespace: clinical-diagnostics
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/websocket-services: quantumhealth-service
spec:
  tls:
  - hosts:
    - diagnostics.hospital.internal
    secretName: quantumhealth-tls-cert
  rules:
  - host: diagnostics.hospital.internal
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: quantumhealth-service
            port:
              number: 8501
```

#### 6.1.3 Horizontal Pod Autoscaler (`k8s/hpa.yaml`)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: quantumhealth-hpa
  namespace: clinical-diagnostics
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: quantumhealth-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 7. Cloud Quantum Hardware Integration (Physical QPUs)

The platform's decoupled architecture via `src/quantum/backend.py` allows seamless switching between high-performance local simulators and production quantum cloud services without changing model logic.

### 7.1 IBM Quantum Platform Integration

To execute circuits on physical superconducting transmon QPUs (e.g., `ibm_brisbane`, `ibm_sherbrooke`):

1. **Install the PennyLane Qiskit plugin:**
   ```bash
   pip install pennylane-qiskit qiskit-ibm-runtime
   ```

2. **Configure API Token:**
   Set the environment variable or configure local credentials:
   ```bash
   export QISKIT_IBM_TOKEN="your_ibm_quantum_api_token"
   export QISKIT_IBM_CHANNEL="ibm_quantum" # or ibm_cloud
   ```

3. **Backend Routing in `src/quantum/backend.py`:**
   ```python
   import pennylane as qml
   from qiskit_ibm_runtime import QiskitRuntimeService

   service = QiskitRuntimeService(channel="ibm_quantum", token=os.environ["QISKIT_IBM_TOKEN"])
   backend = service.least_busy(operational=True, simulator=False, min_num_qubits=4)

   dev = qml.device(
       "qiskit.remote",
       wires=4,
       backend=backend,
       shots=4000,
   )
   ```

4. **Error Mitigation Pipeline:**
   - **Measurement Error Mitigation:** M3 (Matrix-free Measurement Mitigation) via `qiskit-experiments`.
   - **Gate Error Mitigation:** Zero-Noise Extrapolation (ZNE) via PennyLane transforms (`qml.transforms.mitigate_with_zne`).

### 7.2 Amazon Braket Integration

To access trapped-ion (IonQ) or neutral-atom (QuEra) processors:

1. **Install Braket Plugin:**
   ```bash
   pip install pennylane-braket amazon-braket-sdk
   ```

2. **AWS IAM Configuration:**
   Ensure your node or pod has an IAM role with `AmazonBraketFullAccess` and an Amazon S3 staging bucket for quantum task outputs:
   ```bash
   export AWS_DEFAULT_REGION="us-east-1"
   export BRAKET_S3_BUCKET="amazon-braket-clinical-diagnostics-staging"
   ```

3. **Device Instantiation:**
   ```python
   import pennylane as qml

   s3_folder = (os.environ["BRAKET_S3_BUCKET"], "wdbc-runs")
   dev = qml.device(
       "braket.aws.qubit",
       device_arn="arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
       wires=4,
       s3_destination_folder=s3_folder,
       shots=2500,
   )
   ```

---

## 8. Clinical Security, HIPAA & Regulatory Compliance

Clinical decision support systems handling Protected Health Information (PHI) must adhere to strict regulatory boundaries under HIPAA (Health Insurance Portability and Accountability Act, 45 CFR Part 160 and 164) and GDPR.

### 8.1 De-Identification & Data Minimization
- **Safe Harbor Method:** All incoming patient records must be stripped of all 18 HIPAA identifiers prior to ingestion. The platform enforces that patient datasets only contain numerical morphological biomarkers (e.g., cell radius, concavity) and synthetic patient IDs (`Patient_42`).
- **In-Memory Volatility:** Raw feature vectors are maintained strictly in ephemeral volatile memory (RAM) and are zeroed upon session termination. No raw biomarker vectors are persisted to disk or cloud logging services.

### 8.2 Cryptographic Standards
- **In-Transit:** Mandatory TLS 1.3 encryption across all clinical workstation connections and cloud QPU endpoints using modern cipher suites (`TLS_AES_256_GCM_SHA384`).
- **At-Rest:** Checkpoints, serialized models, and caching artifacts are encrypted using AES-256 via Kubernetes volume encryption.

### 8.3 Non-Root Execution & Least Privilege
The container drops all Linux capabilities, runs as unprivileged UID `10001`, and enforces a read-only root filesystem when configured with K8s `securityContext.readOnlyRootFilesystem: true`.

---

## 9. Observability & Operational Verification

### 9.1 Prometheus Metrics Endpoint
Streamlit and Python performance metrics are scraped via standard sidecars. Key Service Level Indicators (SLIs) monitored:
- **`qml_inference_latency_seconds`:** P95 latency must remain $< 150\text{ ms}$ for simulated inference.
- **`quantum_gram_matrix_duration_seconds`:** Monitored to detect cache misses.
- **`statevector_cache_hit_ratio`:** Target $> 85\%$ during batch inference.

### 9.2 Automated Verification Smoke Test

Execute this verification script on newly deployed clinical nodes:

```bash
#!/bin/bash
set -e

echo "[1/4] Checking HTTP endpoint health..."
curl -s -f http://localhost:8501/_stcore/health > /dev/null
echo "  -> Streamlit healthcheck OK."

echo "[2/4] Testing ModelFactory CLI prediction..."
python cli.py predict --model checkpoints/qsvc_wdbc.joblib --input data/sample_patient.json > /dev/null
echo "  -> QSVC model inference OK."

echo "[3/4] Running quantum unit test suite..."
pytest tests/test_quantum_circuits.py -q
echo "  -> Quantum circuit bounds OK."

echo "[4/4] Verifying statevector acceleration cache..."
pytest tests/test_profiling_and_caching.py -q
echo "  -> Caching and profiler verified."

echo "All clinical deployment smoke tests PASSED."
```
