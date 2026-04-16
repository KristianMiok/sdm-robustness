"""Task 3 — Locked modelling pipeline.

Implements the locked protocol: per-species predictor cleaning (once on clean
benchmark, frozen thereafter), algorithm training (Random Forest + XGBoost,
optionally Maxent), spatial block CV with basin_id grouping, and output
artefact generation.

Status: SCAFFOLD — filled in after Task 1 delivery and Task 3 protocol sign-off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class PipelineConfig:
    """Frozen pipeline config for one species.

    Built ONCE on the clean benchmark per species, then used identically for
    every contamination run of that species.
    """

    species: str
    scale_track: str                      # 'local_only' | 'upstream_only' | 'combined'
    kept_predictors: list[str]            # after missingness + correlation filter
    rf_hyperparams: dict[str, Any]
    xgb_hyperparams: dict[str, Any]
    maxent_hyperparams: dict[str, Any] | None
    presence_absence_ratio: float
    cv_type: str                          # 'spatial_block' | 'spatial_blocks_within_basin'
    cv_groups_col: str                    # 'basin_id' or custom fallback
    n_clean_benchmark: int


def clean_predictors(
    df: pd.DataFrame,
    predictor_cols: list[str],
    *,
    missing_threshold_pct: float = 30.0,
    correlation_threshold: float = 0.98,
) -> list[str]:
    """Drop predictors exceeding the missingness threshold and prune correlated
    pairs. Returns the kept column list.

    TODO: implement per briefing §5.1 — applied once on clean benchmark, then
    frozen for all contamination runs of that species.
    """
    raise NotImplementedError("Task 3 — implement after protocol sign-off")


def tune_hyperparameters(
    df: pd.DataFrame,
    predictor_cols: list[str],
    *,
    algorithm: str,
    cv_groups: pd.Series,
    seed: int,
) -> dict[str, Any]:
    """Nested-CV hyperparameter tuning on the clean benchmark.

    TODO: implement per briefing §5.1 — once per species, then frozen.
    """
    raise NotImplementedError("Task 3 — implement after protocol sign-off")


def generate_pseudo_absences(
    presence_df: pd.DataFrame,
    *,
    ratio: float = 1.0,
    stratify_by_strahler: bool = True,
    accessible_area_buffer_km: float = 50.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Network-constrained pseudo-absence sampling.

    Kristian's recommendation (pending Lucian sign-off): stratify density by
    Strahler order so snapping-axis shifts between segments of different
    orders don't induce an absence-density confound.

    TODO: implement per Task 3 protocol.
    """
    raise NotImplementedError("Task 3 — implement after pseudo-absence strategy finalised")


def fit_model(
    training_df: pd.DataFrame,
    config: PipelineConfig,
    *,
    algorithm: str,
    seed: int,
) -> dict[str, Any]:
    """Fit one model (one species, one algorithm, one training set).

    TODO: dispatch to RF / XGBoost / Maxent implementations.
    """
    raise NotImplementedError("Task 3 — implement after protocol sign-off")


def predict_suitability_raster(
    model: Any,
    network_gdf: Any,
    config: PipelineConfig,
) -> Any:
    """Project the fitted model onto the species' accessible network.

    TODO: returns a geopandas GeoDataFrame keyed on subc_id with a
    suitability column.
    """
    raise NotImplementedError("Task 3 — implement after protocol sign-off")
