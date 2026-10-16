# Contributing to QuantumHealth

Thank you for your interest in contributing to **QuantumHealth: Hybrid Quantum-Classical ML Clinical Diagnostics Platform**! We welcome contributions from quantum computing researchers, biomedical data scientists, clinical practitioners, and software engineers.

---

## 1. Code of Conduct

All contributors and maintainers are expected to abide by the [Code of Conduct](CODE_OF_CONDUCT.md). Please report any unacceptable behavior to `compliance@quantumhealth-project.org`.

---

## 2. Getting Started

### 2.1 Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.11 recommended).
- Git.
- (Optional) Docker Engine for containerized testing.

### 2.2 Setting Up Your Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/quantum-diagnostics.git
   cd quantum-diagnostics
   ```

2. **Create an isolated virtual environment:**
   ```bash
   python -m venv .venv

   # Linux / macOS:
   source .venv/bin/activate

   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

---

## 3. Development Workflow

### 3.1 Branching Strategy
- `main`: Production-ready, deployable code. Tagged releases (e.g. `v1.0.0`) are cut from `main`.
- `feature/<name>`: New features, architectures, or UI pages (e.g. `feature/m3-error-mitigation`).
- `bugfix/<name>`: Bug fixes and performance patches (e.g. `bugfix/caching-leak`).
- `docs/<name>`: Documentation, guides, or benchmark updates.

### 3.2 Code Quality & Formatting Standards
All code submitted to QuantumHealth must adhere to our automated style guide:
- **Code Formatter:** `black` with line length 100.
  ```bash
  black --line-length 100 src/ tests/ app/ cli.py
  ```
- **Import Sorter:** `isort` with Black profile.
  ```bash
  isort --profile black --line-length 100 src/ tests/ app/ cli.py
  ```
- **Linter:** `flake8` with max line length 100.
  ```bash
  flake8 src/ tests/ app/ cli.py --max-line-length=100 --extend-ignore=E203,W503,E501
  ```

### 3.3 Conventional Commits
Commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat: add Zero-Noise Extrapolation mitigation transform to QuantumBackendManager`
- `fix: resolve statevector cache eviction memory leak in multi-threaded runs`
- `docs: update benchmarks.md with 10-fold cross validation figures`
- `test: add unit tests for Gram matrix symmetry on 8-qubit registers`
- `refactor: optimize batch statevector tensor contractions in QSVC`

---

## 4. Testing Guidelines

Automated testing is mandatory for all code additions.
- Place unit tests in the `tests/` directory following the naming pattern `test_<module>.py`.
- Ensure tests execute in **under 60 seconds** on standard CPU runners.
- Run the full test suite before submitting a Pull Request:
  ```bash
  pytest tests/ -v
  ```

---

## 5. Pull Request (PR) Checklist

Before submitting a Pull Request, please verify:
- [ ] The branch is rebased onto the latest `main`.
- [ ] All unit and integration tests pass cleanly (`pytest tests/ -v`).
- [ ] Linters and formatters report zero errors (`black --check`, `isort --check`, `flake8`).
- [ ] Any new public methods, classes, or CLI arguments include complete type annotations and docstrings.
- [ ] No Protected Health Information (PHI) or unhashed credentials are committed.
- [ ] The PR description details the motivation, changes, and verification evidence.

Thank you for helping push the frontier of hybrid quantum healthcare!
