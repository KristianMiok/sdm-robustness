"""Task 6 — analysis and synthesis.

Per-species degradation curves, breakpoint detection, cross-species meta-
synthesis, hypothesis tests H1–H3, practical guideline table.

Status: SCAFFOLD.
"""

from __future__ import annotations

import pandas as pd


def degradation_curves(
    results_raw: pd.DataFrame, *, metric: str, ci_pct: int = 95
) -> pd.DataFrame:
    """Aggregate replicate-level results into per-(species, axis, level) curves
    with bootstrap CIs.

    TODO: implement.
    """
    raise NotImplementedError("Task 6 — implement")


def detect_breakpoint(curve: pd.Series) -> dict:
    """Segmented regression / changepoint detection on a degradation curve.

    TODO: implement (segmented via pwlf or ruptures).
    """
    raise NotImplementedError("Task 6 — implement")


def cross_species_mixed_effects(
    results_raw: pd.DataFrame,
    *,
    fixed_effects: tuple[str, ...] = ("level_pct", "axis", "domain"),
    random_effect: str = "species",
) -> dict:
    """Mixed-effects synthesis across species.

    TODO: implement via statsmodels.formula.api.mixedlm.
    """
    raise NotImplementedError("Task 6 — implement")


def test_h1_masked_uncertainty(tier1_ist: pd.DataFrame, tier2_ist: pd.DataFrame) -> dict:
    """H1: Tier 2 IST systematically lower than Tier 1 IST across species.

    TODO: paired tests across species.
    """
    raise NotImplementedError("Task 6 — implement")


def test_h2_regime_difference(
    snap_curves: pd.DataFrame, lowacc_curves: pd.DataFrame
) -> dict:
    """H2: different error regimes between axes (shape, not just magnitude).

    TODO: implement.
    """
    raise NotImplementedError("Task 6 — implement")


def test_h3_local_upstream_divergence(results_by_scale: pd.DataFrame) -> dict:
    """H3: Local predictors degrade faster than Upstream under snapping.

    TODO: paired contrasts in the mixed-effects framework.
    """
    raise NotImplementedError("Task 6 — implement")


def build_practical_guideline_table(ist_results: pd.DataFrame) -> pd.DataFrame:
    """Produce the Safe/Caution/Unreliable table by (species-type × axis).

    Most citable output for applied users. TODO: implement.
    """
    raise NotImplementedError("Task 6 — implement")
