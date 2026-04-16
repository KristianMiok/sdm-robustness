"""Tests for audit.gates."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.audit.feasibility import compute_feasibility
from sdm_robustness.audit.gates import classify_candidates
from sdm_robustness.audit.inventory import build_inventory


def test_classification_end_to_end(synthetic_master_table, default_gates_config):
    petko_cats = pd.DataFrame(
        {
            "species": [
                "Austropotamobius_mega",
                "Astacus_mini",
                "Procambarus_partial",
            ],
            "category": ["widespread", "endemic", "regional"],
        }
    )
    inv = build_inventory(synthetic_master_table, petko_categories=petko_cats)
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")

    assert cls.loc["Austropotamobius_mega", "classification"] == "PRIMARY"
    assert cls.loc["Astacus_mini", "classification"] == "PRIMARY"
    assert cls.loc["Procambarus_partial", "classification"] == "PARTIAL"


def test_category_fallback_when_petko_missing(synthetic_master_table, default_gates_config):
    inv = build_inventory(synthetic_master_table)
    assert inv["category_petko2026"].isna().all()
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")
    assert cls.loc["Astacus_mini", "classification"] == "INELIGIBLE"


def test_category_fallback_endemic_passes_mini(synthetic_master_table, default_gates_config):
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv)
    cls = classify_candidates(
        inv, fea, default_gates_config, default_category="endemic"
    ).set_index("species")
    assert cls.loc["Astacus_mini", "classification"] == "PRIMARY"


def test_gate_failure_reasons(synthetic_master_table, default_gates_config):
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")
    partial = cls.loc["Procambarus_partial"]

    assert partial["gate_1_min_benchmark"] == 1
    assert partial["gate_2_snap_pool"] == 1
    assert partial["gate_3_lowacc_pool"] == 1
    assert partial["strict_50pct_snap"] == 1
    assert partial["strict_50pct_lowacc"] == 0
    assert partial["gate_4_basin_spread"] == 1
    assert partial["gate_5_strahler_spread"] == 1
    assert partial["classification"] == "PARTIAL"


def test_species_with_minimal_snap_pool_is_partial():
    inv = pd.DataFrame(
        [
            {
                "species": "BorderlineSp",
                "category_petko2026": "regional",
                "n_clean_dedup_200m": 1000,
                "n_snap_200_500": 60,
                "n_snap_500_1000": 10,
                "n_low_acc_dedup": 55,
                "n_basins": 5,
                "strahler_min": 2,
                "strahler_max": 6,
            }
        ]
    )
    fea = compute_feasibility(inv)
    gates = {
        "gate_1_minimum_benchmark": {
            "widespread": 500,
            "regional": 200,
            "endemic": 80,
        },
        "gate_2_snapping_pool": {"enabled": True},
        "gate_3_lowacc_pool": {"enabled": True},
        "gate_4_basin_spread": {"min_basins": 3},
        "gate_5_strahler_spread": {"min_distinct_orders": 3},
        "borderline_margin_pct": 10.0,
    }
    cls = classify_candidates(inv, fea, gates).set_index("species")

    assert cls.loc["BorderlineSp", "classification"] == "PARTIAL"
    assert cls.loc["BorderlineSp", "strict_50pct_snap"] == 0
    assert cls.loc["BorderlineSp", "strict_50pct_lowacc"] == 0
