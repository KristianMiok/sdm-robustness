from __future__ import annotations

import numpy as np
import pandas as pd


def _sample_rows(
    df: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"Requested sample size {n} exceeds pool size {len(df)}")
    return df.sample(n=n, replace=False, random_state=seed).copy()


def sample_rf_xgb_pseudoabsences(
    accessible_area: pd.DataFrame,
    benchmark_presence_n: int,
    *,
    ratio: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample pseudo-absences for RF/XGBoost from the accessible area.
    Default ratio = 1:1 relative to benchmark presence count.
    """
    n = int(round(benchmark_presence_n * ratio))
    return _sample_rows(accessible_area, n=n, seed=seed)


def sample_maxent_background(
    accessible_area: pd.DataFrame,
    *,
    n_background: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sample background points for Maxent from the accessible area.
    """
    n = min(n_background, len(accessible_area))
    return _sample_rows(accessible_area, n=n, seed=seed)
