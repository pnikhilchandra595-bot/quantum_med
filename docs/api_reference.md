# QuantumHealth API Reference

Complete developer documentation for modules in `src/` and the unified CLI orchestrator.

---

## 1. Data Ingestion & Preprocessing (`src.data.loader`)

### `get_raw_dataframe(name: str = "wdbc") -> pd.DataFrame`
Retrieves clinical dataset as a raw pandas DataFrame.
- **Parameters:**
  - `name`: Dataset identifier (`"wdbc"`, `"heart"`).
- **Returns:** `pd.DataFrame` containing patient biomarker records and target binary label.

### `load_dataset(name, n_qubits=4, reduction="pca", scaler_type="minmax", test_size=0.2, random_state=42) -> Dict[str, Any]`
Preprocesses, scales, dimensionality-reduces, and stratifies clinical datasets for quantum execution.
- **Returns Dictionary:**
  - `X_train`, `X_test`: Reduced continuous features of shape `(N, n_qubits)`.
  - `y_train`, `y_test`: Binary labels in `{0, 1}`.
  - `scaler`, `quantum_scaler`: Fitted feature scalers.
  - `reducer`: Fitted PCA or SelectKBest transformer.
  - `feature_names`: Labels for reduced dimensions.

---

## 2. Quantum Backend Management (`src.quantum.backend`)

### `class QuantumBackendManager`
Abstracts PennyLane execution backends, memory safety caps, and simulator fallbacks.

#### `__init__(backend_name="default.qubit", shots=None, noise_level=0.0, max_qubits=24, max_memory_mb=1024.0, fallback_to_default=True)`
- **Parameters:**
  - `backend_name`: Target device (`"default.qubit"`, `"lightning.qubit"`, `"default.mixed"`).
  - `shots`: Positive integer for finite sampling, or `None` for analytic statevectors.
  - `noise_level`: Depolarizing noise probability in `[0.0, 1.0]`.
  - `max_qubits`: Maximum allowed wires (default `24`).
  - `max_memory_mb`: Maximum allowed statevector RAM allocation in MB (default `1024.0`).
  - `fallback_to_default`: If `True`, falls back to `"default.qubit"` when target backend fails.

#### `get_device(wires: int) -> qml.Device`
Returns a cached or newly allocated PennyLane device with bounds checking.
- **Raises:**
  - `ValueError`: If `wires <= 0` or `wires > max_qubits`.
  - `MemoryError`: If statevector memory ($2^N \times 16\text{ bytes}$) exceeds `max_memory_mb`.

#### `estimate_statevector_memory_mb(wires: int) -> float`
Calculates analytical memory requirement: $\frac{2^N \times 16}{1024^2}\text{ MB}$.

---

## 3. Quantum Caching Engine (`src.quantum.caching`)

### `class StatevectorCache`
Thread-safe LRU statevector cache eliminating redundant quantum tape compilation and simulations.

#### `get_or_compute(x: np.ndarray, circuit_fn: Callable) -> np.ndarray`
Returns cached statevector $|\Phi(x)\rangle \in \mathbb{C}^{2^N}$, or executes `circuit_fn(x)` and caches the state.

#### `compute_batch_statevectors(X: np.ndarray, circuit_fn: Callable) -> np.ndarray`
Computes statevectors for a batch of samples, leveraging previous cache hits.

---

## 4. Model Factory & Registry (`src.models.factory`)

### `class ModelFactory`
Factory class for unified classical and quantum model instantiation.

#### `create_model(model_name: str, n_qubits: int = 4, backend_manager: Optional[QuantumBackendManager] = None, **kwargs) -> BaseModel`
- **Supported Models:**
  - Classical: `"svm"`, `"rf"`, `"xgb"`, `"mlp"`
  - Quantum: `"qsvc"`, `"vqc"`, `"hqnn"`

---

## 5. Quantum Classifiers (`src.models.quantum`)

### `class QSVCModel(BaseModel)`
Quantum Support Vector Classifier utilizing quantum kernel state fidelity inner products.
- **Attributes:**
  - `use_statevector_caching`: Enables $O(M)$ statevector computation with matrix product Gram calculation ($11.5\times$ speedup).
- **Methods:**
  - `fit(X, y)`: Fits precomputed dual SVM on quantum Gram matrix.
  - `compute_gram_matrix(X1, X2=None)`: Evaluates symmetric positive semi-definite quantum kernel.

### `class VQCModel(BaseModel)`
Variational Quantum Classifier with $R_y$ angle embedding, universal parameterized rotation layers, and circular CNOT entanglement.
- **Methods:**
  - `fit(X, y, epochs=25, lr=0.05, step_callback=None)`: Optimizes variational parameters using analytic parameter-shift gradients or adjoint simulation.
  - `loss_history`: List of per-epoch loss values.

### `class HQNNModel(BaseModel)`
Hybrid Quantum Neural Network coupling PyTorch Linear compression layers with PennyLane `TorchLayer`.

---

## 6. Serialization Engine (`src.utils.serialization`)

### `save_model(model: BaseModel, filepath: str, metadata: Optional[Dict] = None) -> str`
Serializes model to disk:
- `.pkl` / `.joblib` for Classical models and QSVC.
- `.pt` / `.pth` state dictionaries for PyTorch HQNN and VQC.

### `load_model(filepath: str) -> BaseModel`
Deserializes model from disk and restores weights, scalers, and prediction heads.

---

## 7. Execution Profiler (`src.utils.profiler`)

### `profile_quantum_execution(model_name="qsvc", n_samples=16, n_qubits=4) -> Dict[str, Any]`
Captures CPU call graph via `cProfile` and memory peaks via `tracemalloc`.

### `benchmark_caching_speedup(n_samples=16, n_qubits=4) -> Dict[str, Any]`
Measures speedup of statevector caching over uncached pairwise circuit execution.
