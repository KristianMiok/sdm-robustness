"""Tests for audit.feasibility."""

from __future__ import annotations

from sdm_robustness.audit.feasibility import compute_feasibility
from sdm_robustness.audit.inventory import build_inventory


def test_feasibility_pool_sums(synthetic_master_table):
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv).set_index("species")

    # Mega: snap pool = 600 + 300 = 900
    assert fea.loc["Austropotamobius_mega", "n_snap_pool"] == 900
    # Mega: lowacc pool = 700
    assert fea.loc["Austropotamobius_mega", "n_lowacc_pool"] == 700


def test_feasibility_benchmark_policy(synthetic_master_table):
    """Under 'benchmark' policy, n_experiment = n_clean_dedup_200m."""
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv, n_experiment_policy="benchmark").set_index("species")
    assert fea.loc["Austropotamobius_mega", "n_experiment_assumed"] == 600


def test_feasibility_snap_flags_mega(synthetic_master_table):
    """Mega: 900 snap pool vs 600 benchmark → supports up to 50% (need 300)."""
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv).set_index("species")
    for level in (5, 10, 20, 35, 50):
        assert fea.loc["Austropotamobius_mega", f"feas_snap_{level}"] == 1
    assert fea.loc["Austropotamobius_mega", "max_snap_contamination_pct"] == 50


def test_feasibility_lowacc_flags_mega(synthetic_master_table):
    """Mega: 700 lowacc pool vs 600 benchmark → supports up to 50% (need 300)."""
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv).set_index("species")
    for level in (5, 10, 20, 35, 50):
        assert fea.loc["Austropotamobius_mega", f"feas_lowacc_{level}"] == 1


def test_feasibility_partial_species_fails_lowacc(synthetic_master_table):
    """Procambarus_partial: only 50 lowacc records, benchmark = 250.
    50% contamination needs 125 → infeasible. 20% needs 50 → feasible.
    """
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv).set_index("species")

    assert fea.loc["Procambarus_partial", "n_lowacc_pool"] == 50
    assert fea.loc["Procambarus_partial", "n_experiment_assumed"] == 250

    # 50% needs 125 > 50 → fails
    assert fea.loc["Procambarus_partial", "feas_lowacc_50"] == 0
    # 20% needs 50 == 50 → passes
    assert fea.loc["Procambarus_partial", "feas_lowacc_20"] == 1
    # 35% needs 87.5 > 50 → fails
    assert fea.loc["Procambarus_partial", "feas_lowacc_35"] == 0
    # Max feasible in the level set
    assert fea.loc["Procambarus_partial", "max_lowacc_contamination_pct"] == 20


def test_feasibility_2d_flag(synthetic_master_table):
    inv = build_inventory(synthetic_master_table)
    fea = compute_feasibility(inv).set_index("species")
    # Mega: both axes support 50% → 2D feasible
    assert fea.loc["Austropotamobius_mega", "feas_2d"] == 1
    # Partial: lowacc fails → 2D infeasible
    assert fea.loc["Procambarus_partial", "feas_2d"] == 0
