# Security Policy & Vulnerability Disclosure

The maintainers of **QuantumHealth** take software security and clinical patient confidentiality with the utmost seriousness. Given that this platform is designed for biomedical research and clinical decision support, strict safeguards are maintained against unauthorized data access, memory leakage, and code execution vulnerabilities.

---

## 1. Supported Versions

Security updates and vulnerability patches are actively provided for the following releases:

| Version | Supported | Security Maintenance Status |
|:---:|:---:|---|
| **1.0.x** | :white_check_mark: | Current Active Release (LTS) |
| **< 1.0.0** | :x: | Deprecated / Unsupported |

---

## 2. Reporting a Vulnerability

If you discover a potential security flaw, vulnerability, or data leak within QuantumHealth:

1. **Do NOT open a public GitHub issue.**
2. Send an encrypted email with detailed reproduction steps to our Security Response Team at:  
   `security@quantumhealth-project.org`
3. If necessary, encrypt your advisory using our team PGP Public Key (`0x9E4C2F81A5B3D7C0`).

### What to Include in Your Report
- A clear description of the vulnerability.
- Step-by-step instructions or proof-of-concept (PoC) code to reproduce the issue.
- Impact assessment (e.g. arbitrary code execution, denial of service, memory leakage, HIPAA non-compliance).
- Any recommended mitigations or patches.

---

## 3. Vulnerability Response Timeline

We commit to the following response timeline for all reported security incidents:
- **Initial Acknowledgment:** Within **24 hours** of report receipt.
- **Preliminary Assessment & Severity Triage:** Within **72 hours**.
- **Remediation & Patch Release:** Within **14 business days** for critical severity vulnerabilities.
- **Public Disclosure:** Coordinated disclosure following patch publication.

---

## 4. Biomedical Security & HIPAA Safeguards

### 4.1 Ephemeral Data Processing
The platform operates under a strict ephemeral processing model:
- Raw patient biometric vectors uploaded via the Streamlit web dashboard or CLI are processed strictly in volatile memory (RAM).
- No patient biomarkers, names, or unhashed identifiers are ever written to disk or transmitted to external telemetric sinks.
- Model checkpoints store only weight matrices, variational angles $\theta$, and support vector indices—never raw patient training vectors.

### 4.2 Non-Root Execution
The official production container drops all root privileges, enforcing execution under unprivileged non-root user `appuser` (UID 10001) with a read-only root filesystem.

### 4.3 Quantum Cloud Communication Security
When interfacing with physical QPUs on IBM Quantum or AWS Braket, communications are restricted to authenticated, TLS 1.3-encrypted REST and WebSocket channels. Quantum task tokens must be supplied via environment variables (`QISKIT_IBM_TOKEN`, `AWS_ACCESS_KEY_ID`) and are never hardcoded in source trees.
