"""
Tests for unified CLI orchestrator.
"""

import subprocess
import sys


def test_cli_help():
    """Verify cli.py responds to --help cleanly."""
    res = subprocess.run([sys.executable, "cli.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Hybrid Quantum-Classical ML Clinical Diagnostics CLI" in res.stdout
    assert "train" in res.stdout
    assert "predict" in res.stdout
    assert "benchmark" in res.stdout


def test_cli_train_help():
    """Verify cli.py train --help displays flags."""
    res = subprocess.run(
        [sys.executable, "cli.py", "train", "--help"], capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "--model" in res.stdout
    assert "--dataset" in res.stdout
    assert "--qubits" in res.stdout
