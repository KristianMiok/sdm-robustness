"""Tests for audit.inventory."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.audit.inventory import build_inventory


def test_inventory_row_per_species(synthetic_master_table):
    inv = build_inventory(synthetic_master_table)
    assert set(inv["species"]) == {
        "Austropotamobius_mega",
        "Astacus_mini",
        "Procambarus_partial",
    }
    assert len(inv) == 3


def test_inventory_counts_mega(synthetic_master_table):
    inv = build_inventory(synthetic_master_table).set_index("species")
    row = inv.loc["Austropotamobius_mega"]

    # By construction: 600 clean + 600 snap + 300 snap + 700 lowacc = 2200 total
    assert row["n_total_raw"] == 2200
    # High-acc = 600 + 600 + 300 = 1500
    assert row["n_high_acc"] == 1500
    assert row["n_low_acc"] == 700

    # Clean ≤200m = 600
    assert row["n_clean_200m"] == 600
    # 200 < dist ≤ 500
    assert row["n_snap_200_500"] == 600
    # 500 < dist ≤ 1000
    assert row["n_snap_500_1000"] == 300
    # > 1000m
    assert row["n_snap_above_1000"] == 0


def test_inventory_dedup_counts(synthetic_master_table):
    inv = build_inventory(synthetic_master_table).set_index("species")
    # Every synthetic record has a unique subc_id, so dedup is a no-op
    assert inv.loc["Austropotamobius_mega", "n_clean_dedup_200m"] == 600
    assert inv.loc["Astacus_mini", "n_clean_dedup_200m"] == 120
    assert inv.loc["Procambarus_partial", "n_clean_dedup_200m"] == 250


def test_inventory_basin_spread(synthetic_master_table):
    inv = build_inventory(synthetic_master_table).set_index("species")
    assert inv.loc["Austropotamobius_mega", "n_basins"] == 6
    assert inv.loc["Astacus_mini", "n_basins"] == 4
    assert inv.loc["Procambarus_partial", "n_basins"] == 3


def test_inventory_strahler_range(synthetic_master_table):
    inv = build_inventory(synthetic_master_table).set_index("species")
    # Mega: 2..7
    assert inv.loc["Austropotamobius_mega", "strahler_min"] == 2
    assert inv.loc["Austropotamobius_mega", "strahler_max"] == 7
    # Mini: 2..4
    assert inv.loc["Astacus_mini", "strahler_min"] == 2
    assert inv.loc["Astacus_mini", "strahler_max"] == 4


def test_inventory_status(synthetic_master_table):
    inv = build_inventory(synthetic_master_table).set_index("species")
    assert inv.loc["Austropotamobius_mega", "status"] == "Native"
    assert inv.loc["Procambarus_partial", "status"] == "Alien"


def test_dedup_collapses_duplicates():
    """Two records on the same subc_id of the same species should collapse to 1."""
    rows = []
    for i in range(10):
        rows.append({
            "WoCID": f"X{i}", "lat_or": 45.0, "long_or": 9.0,
            "lat_snap": 45.0, "long_snap": 9.0,
            "Accuracy": "High", "Crayfish_scientific_name": "TestSp",
            "Status": "Native", "basin_id": 1, "subc_id": 42,  # all same subc!
            "strahler": 3, "distance_m": 100,
            "ab_200m": True, "ab_500m": True, "ab_1000m": True, "is_coastal": False,
        })
    df = pd.DataFrame(rows)
    df["Accuracy"] = df["Accuracy"].astype("category")
    df["Status"] = df["Status"].astype("category")
    inv = build_inventory(df).set_index("species")
    assert inv.loc["TestSp", "n_clean_200m"] == 10
    assert inv.loc["TestSp", "n_clean_dedup_200m"] == 1


def test_species_with_zero_clean_records_not_dropped():
    """A species with only Low-accuracy records should still appear in the
    inventory with n_clean_dedup_200m == 0 (not NaN, not missing).

    Regression test for the IntCastingNaNError observed on Olga's real data
    where species had 0 clean records and feasibility.astype(int) crashed.
    """
    from sdm_robustness.audit.feasibility import compute_feasibility

    rows = []
    # Species A: has clean records — normal case
    for i in range(5):
        rows.append({
            "WoCID": f"A{i}", "lat_or": 45.0, "long_or": 9.0,
            "lat_snap": 45.0, "long_snap": 9.0,
            "Accuracy": "High", "Crayfish_scientific_name": "SpeciesA",
            "Status": "Native", "basin_id": 1, "subc_id": 10 + i,
            "strahler": 3, "distance_m": 100,
            "ab_200m": True, "ab_500m": True, "ab_1000m": True, "is_coastal": False,
        })
    # Species B: only Low-accuracy records — zero clean pool
    for i in range(3):
        rows.append({
            "WoCID": f"B{i}", "lat_or": 45.0, "long_or": 9.0,
            "lat_snap": 45.0, "long_snap": 9.0,
            "Accuracy": "Low", "Crayfish_scientific_name": "SpeciesB",
            "Status": "Native", "basin_id": 2, "subc_id": 20 + i,
            "strahler": 3, "distance_m": 100,
            "ab_200m": False, "ab_500m": False, "ab_1000m": False, "is_coastal": False,
        })
    df = pd.DataFrame(rows)
    df["Accuracy"] = df["Accuracy"].astype("category")
    df["Status"] = df["Status"].astype("category")

    inv = build_inventory(df).set_index("species")
    # Both species appear
    assert "SpeciesA" in inv.index
    assert "SpeciesB" in inv.index
    # SpeciesB has zero clean records — must be 0, not NaN
    assert inv.loc["SpeciesB", "n_clean_dedup_200m"] == 0
    assert inv.loc["SpeciesB", "n_low_acc_dedup"] == 3

    # Feasibility must not crash on the zero-pool case
    fea = compute_feasibility(inv.reset_index()).set_index("species")
    assert fea.loc["SpeciesB", "n_experiment_assumed"] == 0
    assert fea.loc["SpeciesB", "n_lowacc_pool"] == 3
