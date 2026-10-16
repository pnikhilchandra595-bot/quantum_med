"""Unit tests for Quantum Circuit Visualizer (Task T4.7)."""

import pytest

pytest.importorskip("pennylane")
from unittest.mock import MagicMock

from app.components.circuit_viz import QuantumCircuitVisualizer


class TestCircuitVisualizer:
    """Tests for ASCII and Matplotlib circuit diagram generators."""

    @pytest.fixture
    def visualizer(self):
        return QuantumCircuitVisualizer(backend_name="default.qubit")

    def test_ascii_drawing_vqc(self, visualizer):
        ascii_art = visualizer.draw_ascii(model_type="vqc", n_qubits=2, n_layers=1)
        assert isinstance(ascii_art, str)
        assert len(ascii_art) > 0
        assert "0" in ascii_art  # wire 0

    def test_ascii_drawing_qsvc(self, visualizer):
        ascii_art = visualizer.draw_ascii(model_type="qsvc", n_qubits=2)
        assert isinstance(ascii_art, str)
        assert len(ascii_art) > 0

    def test_invalid_model_type(self, visualizer):
        with pytest.raises(ValueError, match="Unsupported model type"):
            visualizer.draw_ascii(model_type="invalid_model", n_qubits=2)
