from __future__ import annotations

import pandas as pd

from sdm_robustness.execution import (
    get_panel_entity,
    assign_basin_folds,
    sample_rf_xgb_pseudoabsences,
    sample_maxent_background,
)


def test_get_panel_entity():
    row = get_panel_entity("Austropotamobius fulcisianus (pooled)")
    assert row["entity"] == "Austropotamobius fulcisianus (pooled)"
    assert row["type"] == "DUAL"


def test_assign_basin_folds_grouped():
    basins = pd.Series(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
    out = assign_basin_folds(basins, n_splits=5, looo_threshold=15)
    assert set(out.values()) == {0, 1, 2, 3, 4}


def test_assign_basin_folds_leave_one_out():
    basins = pd.Series(["1", "2", "3"])
    out = assign_basin_folds(basins, n_splits=5, looo_threshold=15)
    assert len(set(out.values())) == 3


def test_sample_rf_xgb_pseudoabsences():
    df = pd.DataFrame({"x": range(100)})
    out = sample_rf_xgb_pseudoabsences(df, benchmark_presence_n=20, ratio=1.0, seed=42)
    assert len(out) == 20


def test_sample_maxent_background():
    df = pd.DataFrame({"x": range(500)})
    out = sample_maxent_background(df, n_background=10_000, seed=42)
    assert len(out) == 500
