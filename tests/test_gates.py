"""Tests for revised audit.gates."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.audit.feasibility import compute_feasibility
from sdm_robustness.audit.gates import classify_candidates


def _gates():
    return {
        "gate_1_minimum_benchmark": {
            "widespread": 500,
            "regional": 200,
            "endemic": 80,
        },
        "gate_2_snapping_pool": {"enabled": True},
        "gate_3_lowacc_pool": {"enabled": True},
        "gate_4_basin_spread": {"min_basins": 3},
        "gate_5_strahler_spread": {"min_distinct_orders": 3},
    }


def test_dual_axis_candidate():
    inv = pd.DataFrame(
        [
            {
                "species": "sp_dual",
                "category_petko2026": "regional",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 20,
                "n_snap_500_1000": 40,
                "n_low_acc_dedup": 250,
                "n_basins": 10,
                "strahler_min": 1,
                "strahler_max": 4,
            }
        ]
    )
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, _gates()).set_index("species")
    assert cls.loc["sp_dual", "classification"] == "DUAL-AXIS"


def test_snapping_only_candidate():
    inv = pd.DataFrame(
        [
            {
                "species": "sp_snap",
                "category_petko2026": "regional",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 10,
                "n_snap_500_1000": 45,
                "n_low_acc_dedup": 50,
                "n_basins": 10,
                "strahler_min": 1,
                "strahler_max": 4,
            }
        ]
    )
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, _gates()).set_index("species")
    assert cls.loc["sp_snap", "classification"] == "SNAPPING-ONLY"


def test_lowacc_only_candidate():
    inv = pd.DataFrame(
        [
            {
                "species": "sp_low",
                "category_petko2026": "regional",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 0,
                "n_snap_500_1000": 15,
                "n_low_acc_dedup": 250,
                "n_basins": 10,
                "strahler_min": 1,
                "strahler_max": 4,
            }
        ]
    )
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, _gates()).set_index("species")
    assert cls.loc["sp_low", "classification"] == "LOW-ACC-ONLY"


def test_ineligible_when_core_gate_fails():
    inv = pd.DataFrame(
        [
            {
                "species": "sp_bad",
                "category_petko2026": "regional",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 20,
                "n_snap_500_1000": 40,
                "n_low_acc_dedup": 250,
                "n_basins": 1,
                "strahler_min": 1,
                "strahler_max": 1,
            }
        ]
    )
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, _gates()).set_index("species")
    assert cls.loc["sp_bad", "classification"] == "INELIGIBLE"
