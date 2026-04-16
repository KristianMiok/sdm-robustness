"""Task 5 — stress-test execution orchestrator.

Coordinates the full factorial:
    ~10 species × 2 axes × 6 levels × 30 replicates × 2 algorithms × 3 scale tracks
    = ~21,600 RF + ~21,600 XGBoost runs.

Plus:
    Task 5c — benchmark sanity check (30 benchmark-only replicates per species
              for the intrinsic noise floor)
    Task 5d — transferability test (3 top species at 10/30/50% — Kristian's
              amended recommendation vs. the single-30%-level in the briefing)

Outputs: `results/task5_execution/results_raw.parquet` indexed by
    (species, axis, level, replicate, algorithm, scale_track, random_seed)

Status: SCAFFOLD — uses contamination.draw_substitution_sample() and
pipeline.fit_model() once those are finalised.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def run_core_factorial(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    algorithms: tuple[str, ...] = ("random_forest", "xgboost"),
    axes: tuple[str, ...] = ("snapping", "lowacc"),
    levels_pct: tuple[int, ...] = (0, 5, 10, 20, 35, 50),
    scale_tracks: tuple[str, ...] = ("local_only", "upstream_only", "combined"),
    n_replicates_default: int = 30,
    n_replicates_low_levels: int = 50,
    low_level_threshold_pct: int = 10,
    master_seed: int = 20260416,
) -> Path:
    """Run the core factorial stress-test.

    TODO: implement the outer loop over all factors, dispatch to
    execution.contamination.draw_substitution_sample() for sampling, then
    pipeline.fit_model() + metrics.* to produce per-run result rows.

    Writes results_raw.parquet and returns its path.
    """
    raise NotImplementedError("Task 5 — implement after Tasks 3 and 4 signed off")


def run_benchmark_sanity_check(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    n_replicates: int = 30,
    master_seed: int = 20260416,
) -> Path:
    """Task 5c — benchmark-to-benchmark noise floor.

    TODO: implement per briefing §7.2.
    """
    raise NotImplementedError("Task 5c — implement")


def run_transferability_test(
    top_species: list[str],
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    levels_pct: tuple[int, ...] = (10, 30, 50),
    master_seed: int = 20260416,
) -> Path:
    """Task 5d — train-clean/test-contaminated and train-contaminated/test-clean.

    Three-point version per Kristian's amendment (briefing specified a single
    30% point; three points give bias direction).

    TODO: implement per briefing §7.3 with the three-level amendment.
    """
    raise NotImplementedError("Task 5d — implement")


def run_null_model(
    species_panel: pd.DataFrame,
    master_table: pd.DataFrame,
    *,
    output_dir: Path | str,
    levels_pct: tuple[int, ...] = (0, 5, 10, 20, 35, 50),
    n_replicates: int = 30,
    master_seed: int = 20260416,
) -> Path:
    """Kristian's amendment — null-model baseline.

    'Contamination' here is resampling clean pool records with no positional
    modification. Gives the structureless-contamination reference line.

    TODO: implement.
    """
    raise NotImplementedError("Null-model — implement")
