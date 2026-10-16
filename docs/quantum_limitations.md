# Quantum Methodology & NISQ Hardware Limitations Whitepaper

> **Author:** Member B (Quantum ML Engineer)  
> **Platform:** Hybrid Quantum Machine Learning for Early Disease Detection (HQML-EDD)  
> **Deliverable:** D10 / Task T5.5  

---

## 1. Executive Summary
This whitepaper documents the theoretical foundations, algorithmic design choices, and physical hardware constraints governing our hybrid quantum-classical disease detection platform. While Quantum Machine Learning (QML) offers mathematical advantages through high-dimensional Hilbert space mappings and quantum entanglement, deploying these models on near-term Noisy Intermediate-Scale Quantum (NISQ) devices requires explicit architectural mitigations. We analyze why our platform enforces a simulator-first paradigm, how we counter barren plateaus, and what is required to transition to physical quantum processing units (QPUs).

---

## 2. Quantum Algorithmic Formulations

### 2.1 Hilbert Space Feature Mapping
Classical data $x \in \mathbb{R}^d$ cannot be processed directly by unitary quantum operations. We apply a non-linear unitary mapping $U_{\Phi(x)}$ to prepare an $N$-qubit quantum state:

$$|\Phi(x)\rangle = U_{\Phi(x)} |0\rangle^{\otimes N}$$

In our platform, we utilize two complementary feature maps:
1. **Angle Embedding:** Single-qubit rotations $R_Y(x_i)$ where $x_i \in [-\pi, \pi]$.
2. **Second-Order Pauli-Z Expansion ($ZZ$-FeatureMap):**

$$U_{\Phi(x)} = \prod_{l=1}^L \left[ \prod_{(j,k)} \exp\left(i \, \phi_{jk}(x) Z_j Z_k\right) \prod_j \exp\left(i \, \phi_j(x) Z_j\right) H^{\otimes N} \right]$$

where $\phi_j(x) = 2x_j$ and $\phi_{jk}(x) = 2(\pi - x_j)(\pi - x_k)$. The entangling two-qubit $Z_j Z_k$ interactions create non-linear state correlations that classical linear kernels cannot reproduce.

### 2.2 Quantum Kernel Estimation (QSVC)
The transition fidelity between two patient feature states defines the quantum Gram kernel:

$$K(x_i, x_j) = \left| \langle \Phi(x_i) | \Phi(x_j) \rangle \right|^2 = \left| \langle 0^{\otimes N} | U_{\Phi(x_j)}^\dagger U_{\Phi(x_i)} | 0^{\otimes N} \rangle \right|^2$$

By measuring the probability of observing the vacuum state $|0\rangle^{\otimes N}$, the Gram matrix is constructed and fed directly to a maximal-margin hyperplane optimizer ($C$-SVM).

---

## 3. Physical NISQ Constraints & Limitations

### 3.1 Qubit Coherence Times ($T_1$ and $T_2$)
Physical transmon qubits suffer from two primary decoherence channels:
* **$T_1$ (Energy Relaxation):** Qubits decay from excited state $|1\rangle$ to ground state $|0\rangle$ (typically $100–300\,\mu\text{s}$ on contemporary superconducting processors).
* **$T_2$ (Dephasing):** Relative phase randomization destroying superpositions (typically $80–200\,\mu\text{s}$).

**Platform Design Decision:** Circuit depth is strictly constrained to $L \le 2$ layers. On typical gate execution times ($20\,\text{ns}$ for single-qubit, $200–500\,\text{ns}$ for two-qubit CNOTs), our total circuit execution time remains under $3\,\mu\text{s}$, well within the coherent coherence window.

### 3.2 Two-Qubit Gate Error Rates
Two-qubit entangling gates (CNOT / CZ) exhibit error rates between $0.5\%$ and $2.0\%$ per operation on NISQ hardware. A circuit with 20 CNOT gates accumulates a compound fidelity degradation:

$$F_{\text{circuit}} \approx \prod_{g} (1 - \epsilon_g) \approx (1 - 0.015)^{20} \approx 73.9\%$$

For this reason, our platform limits quantum inputs to $N \le 8$ addressable qubits via classical PCA / Mutual Information pre-reduction, keeping entangler gate counts within acceptable error bounds.

---

## 4. The Barren Plateau Phenomenon & Mitigations

### 4.1 Theoretical Problem
McClean et al. (2018) proved that for deep parameterized quantum circuits initialized with random Haar-distributed parameters, the gradient of the loss function vanishes exponentially in the number of qubits:

$$\text{Var}_\theta \left[ \frac{\partial \mathcal{L}(\theta)}{\partial \theta_k} \right] \in \mathcal{O}\left( 2^{-N} \right)$$

On an 8-qubit system, gradients become smaller than standard floating-point precision and measurement shot noise, arresting gradient descent during VQC training.

### 4.2 Mitigations Implemented in HQML-EDD
1. **Shallow Depth Ansatz:** We employ $L=1$ to $L=2$ layers of `StronglyEntanglingLayers`, preventing the parameterized circuit from forming a 2-design over the unitary group $\mathcal{U}(2^N)$.
2. **Local Observables:** Rather than measuring a global parity observable $Z_1 \otimes Z_2 \otimes \dots \otimes Z_N$ (which exacerbates barren plateaus), our VQC measures local single-qubit Pauli expectations $\langle Z_0 \rangle$, where gradients decay polynomially rather than exponentially.
3. **Small-Scale Gaussian Parameter Initialization:** Variational parameters are initialized with $\theta \sim \mathcal{N}(0, 0.1^2)$ (implemented in `src/quantum/ansatz.py`), starting the circuit near identity transformations rather than random scrambled states.

---

## 5. Transition Path to Physical Cloud Hardware

Our architecture separates algorithmic logic from execution backends via `src/quantum/backend.py`. To transition from local statevector simulation (`default.qubit`) to physical hardware:

```
[ Algorithmic Model: QSVC / VQC ]
                │
                ▼
     [ QuantumBackendManager ]
                │
    ┌───────────┴───────────┐
    ▼                       ▼
[ Simulator ]          [ Physical QPU ]
 default.qubit          qiskit_ibm_runtime
 aer_simulator          IBM Quantum Heron (156 qubits)
```

To switch to cloud QPUs, a developer simply supplies their API token:
```python
manager = QuantumBackendManager(
    backend_name="qiskit.ibmq",
    n_qubits=4,
    shots=1024
)
```
No circuit code or estimator interfaces require alteration, fulfilling Deliverable D6.
