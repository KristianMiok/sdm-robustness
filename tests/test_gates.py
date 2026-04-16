"""Tests for audit.gates."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.audit.feasibility import compute_feasibility
from sdm_robustness.audit.gates import classify_candidates
from sdm_robustness.audit.inventory import build_inventory


def test_classification_end_to_end(synthetic_master_table, default_gates_config):
    """End-to-end with explicit Petko categories supplied."""
    petko_cats = pd.DataFrame({
        "species": [
            "Austropotamobius_mega",
            "Astacus_mini",
            "Procambarus_partial",
        ],
        "category": ["widespread", "endemic", "regional"],
    })
    inv = build_inventory(synthetic_master_table, petko_categories=petko_cats)
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")

    # Mega: widespread, 600 clean ≥ 500 widespread gate → PRIMARY
    assert cls.loc["Austropotamobius_mega", "classification"] == "PRIMARY"
    # Mini: endemic, 120 clean ≥ 80 endemic gate, 4 basins, 3 strahler orders,
    # snap pool 140 ≥ 120 (gate 2), lowacc 150 ≥ 120 (gate 3) → PRIMARY
    assert cls.loc["Astacus_mini", "classification"] == "PRIMARY"
    # Partial: regional, 250 ≥ 200 (gate 1), snap 300 ≥ 250 (gate 2),
    # but lowacc 50 < 250 (gate 3 fails) → PARTIAL
    assert cls.loc["Procambarus_partial", "classification"] == "PARTIAL"


def test_category_fallback_when_petko_missing(synthetic_master_table, default_gates_config):
    """Without petko_categories, default fallback is 'regional'. Mini (120 clean)
    would fail the regional gate (200) but passes endemic (80). Verify the
    default behaviour classifies Mini as INELIGIBLE without Petko input.
    """
    inv = build_inventory(synthetic_master_table)
    # Ensure the category column is NA
    assert inv["category_petko2026"].isna().all()
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")

    # With default 'regional' fallback, Mini (120 clean) < 200 regional → INELIGIBLE
    # This documents the current behaviour and flags to Kristian why supplying
    # petko_categories matters.
    assert cls.loc["Astacus_mini", "classification"] == "INELIGIBLE"


def test_category_fallback_endemic_passes_mini(synthetic_master_table, default_gates_config):
    """With explicit endemic override, Mini passes gate 1."""
    from sdm_robustness.audit.gates import classify_candidates

    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv)
    cls = classify_candidates(
        inv, fea, default_gates_config, default_category="endemic"
    ).set_index("species")
    assert cls.loc["Astacus_mini", "classification"] == "PRIMARY"


def test_gate_failure_reasons(synthetic_master_table, default_gates_config):
    """PARTIAL species should have gate_3 == 0 but gate_2 == 1."""
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv)
    cls = classify_candidates(inv, fea, default_gates_config).set_index("species")

    partial = cls.loc["Procambarus_partial"]
    assert partial["gate_1_min_benchmark"] == 1   # 250 ≥ 200 regional
    assert partial["gate_2_snap_pool"] == 1       # 300 ≥ 250
    assert partial["gate_3_lowacc_pool"] == 0     # 50 < 250
    assert partial["gate_4_basin_spread"] == 1
    assert partial["gate_5_strahler_spread"] == 1
