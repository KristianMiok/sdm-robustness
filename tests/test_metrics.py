from __future__ import annotations

import numpy as np

from sdm_robustness.metrics import (
    compute_performance_metrics,
    compute_delta_performance,
    spearman_importance_stability,
    topk_jaccard,
    integrated_absolute_difference,
    centroid_displacement,
    niche_breadth_change,
    schoeners_d,
    warrens_i,
    range_area_change,
    spatial_mismatch_summary,
    compute_ist,
)


def test_compute_performance_metrics_basic():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.3, 0.8, 0.9])
    out = compute_performance_metrics(y_true, y_score)
    assert out["auc"] > 0.9
    assert out["brier"] >= 0.0
    assert out["tss"] <= 1.0


def test_compute_delta_performance():
    b = {"auc": 0.9, "tss": 0.6}
    c = {"auc": 0.8, "tss": 0.5}
    out = compute_delta_performance(b, c)
    assert np.isclose(out["delta_auc"], -0.1)
    assert np.isclose(out["delta_tss"], -0.1)


def test_spearman_importance_stability():
    a = np.array([1, 2, 3, 4])
    b = np.array([1, 2, 3, 4])
    assert np.isclose(spearman_importance_stability(a, b), 1.0)


def test_topk_jaccard():
    a = np.array([0.1, 0.2, 0.9, 0.8])
    b = np.array([0.05, 0.1, 0.95, 0.7])
    assert np.isclose(topk_jaccard(a, b, k=2), 1.0)


def test_integrated_absolute_difference():
    x = np.array([0, 1, 2])
    y1 = np.array([0, 1, 2])
    y2 = np.array([0, 1, 1])
    assert integrated_absolute_difference(x, y1, y2) >= 0.0


def test_centroid_displacement():
    a = np.array([[0, 0], [1, 1]])
    b = np.array([[1, 1], [2, 2]])
    assert centroid_displacement(a, b) > 0.0


def test_niche_breadth_change():
    a = np.array([[0, 0], [1, 1], [2, 2]])
    b = np.array([[0, 0], [0.5, 0.5], [1, 1]])
    assert np.isfinite(niche_breadth_change(a, b))


def test_schoeners_d_and_warrens_i():
    a = np.array([[0.2, 0.8], [0.1, 0.9]])
    b = np.array([[0.2, 0.8], [0.1, 0.9]])
    assert np.isclose(schoeners_d(a, b), 1.0)
    assert np.isclose(warrens_i(a, b), 1.0)


def test_range_area_change():
    a = np.array([[0.6, 0.6], [0.1, 0.1]])
    b = np.array([[0.6, 0.6], [0.6, 0.1]])
    assert range_area_change(a, b, threshold=0.5) > 0.0


def test_spatial_mismatch_summary():
    a = np.array([[0.6, 0.6], [0.1, 0.1]])
    b = np.array([[0.6, 0.1], [0.6, 0.1]])
    out = spatial_mismatch_summary(a, b, threshold=0.5)
    assert out["total_changed_cells"] >= 0.0


def test_compute_ist():
    levels = [0, 1, 2, 5]
    vals = [0.95, 0.85, 0.69, 0.5]
    assert compute_ist(levels, vals, threshold=0.7) == 2.0
