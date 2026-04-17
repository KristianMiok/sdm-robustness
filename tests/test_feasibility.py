"""Tests for revised audit.feasibility."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.audit.feasibility import compute_feasibility


def test_feasibility_pool_sums():
    inv = pd.DataFrame(
        [
            {
                "species": "sp1",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 20,
                "n_snap_500_1000": 40,
                "n_low_acc_dedup": 250,
            }
        ]
    )
    fea = compute_feasibility(inv)
    row = fea.iloc[0]
    assert row["n_snap_pool"] == 60
    assert row["n_lowacc_pool"] == 250


def test_revised_feasibility_flags():
    inv = pd.DataFrame(
        [
            {
                "species": "sp1",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 20,
                "n_snap_500_1000": 40,
                "n_low_acc_dedup": 250,
            }
        ]
    )
    fea = compute_feasibility(inv)
    row = fea.iloc[0]

    assert row["feas_snap_1"] == 1
    assert row["feas_snap_2"] == 1
    assert row["feas_snap_5"] == 1
    assert row["feas_lowacc_3"] == 1
    assert row["feas_lowacc_10"] == 1
    assert row["feas_lowacc_20"] == 1

    assert row["max_snap_contamination_pct"] == 5
    assert row["max_lowacc_contamination_pct"] == 20


def test_revised_feasibility_partial_support():
    inv = pd.DataFrame(
        [
            {
                "species": "sp2",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 0,
                "n_snap_500_1000": 15,
                "n_low_acc_dedup": 80,
            }
        ]
    )
    fea = compute_feasibility(inv)
    row = fea.iloc[0]

    assert row["feas_snap_1"] == 1
    assert row["feas_snap_2"] == 0
    assert row["feas_snap_5"] == 0
    assert row["feas_lowacc_3"] == 1
    assert row["feas_lowacc_10"] == 0
    assert row["feas_lowacc_20"] == 0

    assert row["max_snap_contamination_pct"] == 1
    assert row["max_lowacc_contamination_pct"] == 3
