"""Streamlit UI Test Suite.

Module: tests.test_ui
Owner: Member A (Classical ML & Pipeline Integrity Lead)
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parent.parent / "app"
HOME_PATH = APP_DIR / "Home.py"
PAGE1_PATH = APP_DIR / "pages" / "1_Data_Exploration.py"
PAGE2_PATH = APP_DIR / "pages" / "2_Features.py"
PAGE3_PATH = APP_DIR / "pages" / "3_Baselines_Audit.py"


def test_home_page_renders():
    """Verify Streamlit Home.py renders without exceptions."""
    at = AppTest.from_file(HOME_PATH, default_timeout=25).run()
    assert len(at.exception) == 0, f"Exceptions on Home page: {at.exception}"


def test_data_exploration_page_renders_and_switches_dataset():
    """Verify Streamlit page 1 (Data Exploration) renders and allows dataset switching."""
    at = AppTest.from_file(PAGE1_PATH, default_timeout=25).run()
    assert len(at.exception) == 0, f"Exceptions on page 1 load: {at.exception}"

    # Switch to UCI Heart Disease
    if at.selectbox:
        at.selectbox[0].select("UCI Cleveland Heart Disease").run()
        assert len(at.exception) == 0, f"Exceptions after switching dataset on page 1: {at.exception}"


def test_features_page_renders_and_slider_interacts():
    """Verify Streamlit page 2 renders without errors when slider is moved."""
    at = AppTest.from_file(PAGE2_PATH, default_timeout=25).run()
    assert len(at.exception) == 0, f"Exceptions on initial load: {at.exception}"

    # Move slider to 6 qubits
    if at.slider:
        slider = at.slider[0]
        slider.set_value(6).run()
        assert len(at.exception) == 0, f"Exceptions after moving slider to 6: {at.exception}"

        # Move slider to 8 qubits
        slider.set_value(8).run()
        assert len(at.exception) == 0, f"Exceptions after moving slider to 8: {at.exception}"

        # Move slider to 2 qubits
        slider.set_value(2).run()
        assert len(at.exception) == 0, f"Exceptions after moving slider to 2: {at.exception}"


def test_features_page_switch_dataset_and_method():
    """Verify switching to Heart Disease and Mutual Information method on page 2."""
    at = AppTest.from_file(PAGE2_PATH, default_timeout=25).run()

    # Switch dataset to Heart Disease
    if at.selectbox:
        at.selectbox[0].select("UCI Heart Disease (Cleveland)").run()
        assert len(at.exception) == 0

    # Switch method to Mutual Information
    if at.radio:
        at.radio[0].set_value("Mutual Information (SelectKBest)").run()
        assert len(at.exception) == 0


def test_baselines_audit_page_renders_direct():
    """Verify Streamlit page 3 renders without exceptions."""
    at = AppTest.from_file(PAGE3_PATH, default_timeout=25).run()
    assert len(at.exception) == 0, f"Exceptions on page 3 load: {at.exception}"
