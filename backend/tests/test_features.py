"""Unit and integration tests for PCA reducer and MI feature selector.

Test File: tests/test_features.py
Owner: Member A (Lead / Classical ML & Feature Engineering Lead) & Member B (Quantum Circuits)
Tasks: T2.1, T2.2 (Phase 2)
"""

import numpy as np
import pandas as pd
import pytest

from src.data.wdbc_loader import load_wdbc, WDBCPreprocessor
from src.features.pca_reducer import PCAReducer
from src.features.mi_selector import MISelector


def test_pca_reducer_shapes_and_bounds():
    """Verify PCAReducer projects to k in [4, 8] dimensions bounded within [-pi, pi]."""
    X, y = load_wdbc(scale=False, return_feature_names=False)
    preprocessor = WDBCPreprocessor()
    X_s = preprocessor.fit_transform(X)

    for k in [4, 6, 8]:
        pca = PCAReducer(n_components=k, scale_to_pi=True, random_state=42)
        X_pca = pca.fit_transform(X_s)

        assert X_pca.shape == (len(X), k), f"Expected shape ({len(X)}, {k}), got {X_pca.shape}"
        assert np.all(X_pca >= -np.pi - 1e-6), f"Values found below -pi in k={k}"
        assert np.all(X_pca <= np.pi + 1e-6), f"Values found above pi in k={k}"

        evr = pca.get_explained_variance_ratio()
        assert len(evr) == k
        assert np.all(evr > 0.0)

        # Cumulative explained variance checks
        cum_var = pca.get_cumulative_explained_variance()
        assert len(cum_var) == k
        np.testing.assert_allclose(cum_var, np.cumsum(evr))
        assert np.all(np.diff(cum_var) >= 0.0), "Cumulative variance must be non-decreasing"
        assert 0.0 < pca.get_total_explained_variance() <= 1.0


def test_pca_reducer_auto_k_and_dataframe():
    """Verify automated k selection within [4, 8] and DataFrame output."""
    X, y = load_wdbc(as_frame=True, return_feature_names=False)
    pca_auto = PCAReducer(n_components="auto", variance_threshold=0.85, random_state=42)
    df_pca = pca_auto.fit_transform(X, as_frame=True)

    assert isinstance(df_pca, pd.DataFrame)
    assert 4 <= pca_auto.k_effective_ <= 8, f"k_effective_ {pca_auto.k_effective_} not in [4, 8]"
    assert list(df_pca.columns) == [f"PC{i+1}" for i in range(pca_auto.k_effective_)]
    assert np.all(df_pca.values >= -np.pi - 1e-6) and np.all(df_pca.values <= np.pi + 1e-6)

    # Test inverse_transform reconstruction shape
    reconstructed = pca_auto.inverse_transform(df_pca.values)
    assert reconstructed.shape == X.shape


def test_pca_reducer_test_transform():
    """Verify PCAReducer transforms unseen test partition into [-pi, pi]."""
    X, y = load_wdbc(scale=False, return_feature_names=False)
    X_train, X_test = X[:400], X[400:]

    pca = PCAReducer(n_components=4, scale_to_pi=True, random_state=42)
    pca.fit(X_train)
    X_test_pca = pca.transform(X_test)

    assert X_test_pca.shape == (len(X_test), 4)
    assert np.all(X_test_pca >= -np.pi - 1e-6) and np.all(X_test_pca <= np.pi + 1e-6)


def test_mi_selector_feature_names_and_pi_bounds():
    """Verify MISelector retains clinical feature names and scales to [-pi, pi] by default."""
    df, y, feature_names = load_wdbc(as_frame=True, return_feature_names=True)

    for k in [4, 6, 8]:
        selector = MISelector(k=k, random_state=42)
        X_selected = selector.fit_transform(df, y)

        assert X_selected.shape == (len(df), k)
        # Verify strict [-pi, pi] bounds
        assert np.all(X_selected >= -np.pi - 1e-6), f"Values found below -pi for k={k}"
        assert np.all(X_selected <= np.pi + 1e-6), f"Values found above pi for k={k}"

        selected_names = selector.get_selected_feature_names()
        assert len(selected_names) == k
        for name in selected_names:
            assert name in feature_names

        scores = selector.get_feature_scores()
        assert len(scores) == 30
        for name in selected_names:
            assert scores[name] > 0.0


def test_mi_selector_dataframe_preservation_for_shap():
    """Verify MISelector returns DataFrame with original clinical column names for SHAP."""
    df, y, _ = load_wdbc(as_frame=True, return_feature_names=True)
    selector = MISelector(k=5, return_dataframe=True, random_state=42)
    df_selected = selector.fit_transform(df, y)

    assert isinstance(df_selected, pd.DataFrame)
    assert df_selected.shape == (len(df), 5)
    # Exact clinical column names retained
    for col in df_selected.columns:
        assert col in df.columns

    # Verify structured ranking DataFrame
    ranked_df = selector.get_ranked_features_df()
    assert len(ranked_df) == 30
    assert ranked_df["selected"].sum() == 5
    assert list(ranked_df.columns) == ["rank", "feature_name", "mutual_info_nats", "selected"]

    # Verify inverse_transform restores original scale
    inversed = selector.inverse_transform(df_selected.values)
    assert inversed.shape == (len(df), 5)


def test_mi_selector_backward_compat_2pi():
    """Verify backwards compatibility when scale_to_2pi=True is requested."""
    df, y, _ = load_wdbc(as_frame=True, return_feature_names=True)
    selector = MISelector(k=4, scale_to_2pi=True, random_state=42)
    X_selected = selector.fit_transform(df, y)

    assert X_selected.shape == (len(df), 4)
    assert np.all(X_selected >= -1e-6)
    assert np.all(X_selected <= 2.0 * np.pi + 1e-6)


def test_unfitted_guards():
    """Verify runtime error raised if transform or inverse_transform called prior to fit."""
    pca = PCAReducer(n_components=4)
    with pytest.raises(RuntimeError):
        pca.transform(np.random.randn(10, 30))
    with pytest.raises(RuntimeError):
        pca.inverse_transform(np.random.randn(10, 4))

    mi = MISelector(k=4)
    with pytest.raises(RuntimeError):
        mi.transform(np.random.randn(10, 30))
    with pytest.raises(RuntimeError):
        mi.inverse_transform(np.random.randn(10, 4))


class TestPCAReducer:
    """Tests for PCA reduction module from Member B suite."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(40, 10))
        return X

    def test_pca_shape_and_range(self, synthetic_data):
        X = synthetic_data
        reducer = PCAReducer(n_components=4, scale_to_pi=True, random_state=42)
        X_reduced = reducer.fit_transform(X)

        assert X_reduced.shape == (40, 4)
        # Check angle scale [-pi, pi]
        assert np.all(X_reduced >= -np.pi - 1e-6)
        assert np.all(X_reduced <= np.pi + 1e-6)

        summary = reducer.get_variance_summary()
        assert summary["n_components"] == 4
        assert 0.0 < summary["total_explained_variance"] <= 1.0


class TestMISelector:
    """Tests for Mutual Information feature selector from Member B suite."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.0, scale=1.0, size=(50, 8))
        # Label correlated primarily with feature 0 and 1
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        names = [f"BioMarker_{i}" for i in range(8)]
        return X, y, names

    def test_mi_selection_and_ranking(self, synthetic_data):
        X, y, names = synthetic_data
        selector = MISelector(k=3, scale_to_pi=True, random_state=42)
        X_selected = selector.fit_transform(X, y, feature_names=names)

        assert X_selected.shape == (50, 3)
        # Check angle scale [-pi, pi]
        assert np.all(X_selected >= -np.pi - 1e-6)
        assert np.all(X_selected <= np.pi + 1e-6)

        ranking = selector.get_ranking_summary()
        assert len(ranking) == 3
        assert "feature_name" in ranking[0]
        assert "mi_score" in ranking[0]
