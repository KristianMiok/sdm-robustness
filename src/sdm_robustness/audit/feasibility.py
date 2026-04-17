"""Task 1 — Step 1.2: feasibility metrics for revised asymmetric design."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.utils import logger


SNAP_LEVELS = (1, 2, 5)
LOWACC_LEVELS = (3, 10, 20)


def _max_supported_level(pool: pd.Series, n_exp: pd.Series, levels: tuple[int, ...]) -> pd.Series:
    out = pd.Series(0, index=pool.index, dtype="int64")
    for level in levels:
        feasible = pool >= (level / 100.0) * n_exp
        out = out.where(~feasible, level)
    return out


def compute_feasibility(
    inventory: pd.DataFrame,
    *,
    snap_levels: tuple[int, ...] = SNAP_LEVELS,
    lowacc_levels: tuple[int, ...] = LOWACC_LEVELS,
    policy: str = "benchmark",
) -> pd.DataFrame:
    """Compute revised Task 1 feasibility metrics.

    Uses benchmark-size substitution design:
    n_experiment_assumed = n_clean_dedup_200m
    """

    df = inventory.copy()

    n_exp = df["n_clean_dedup_200m"].fillna(0).astype(float)

    n_snap_200_500 = df.get("n_snap_200_500", pd.Series(0, index=df.index)).fillna(0).astype(float)
    n_snap_500_1000 = df.get("n_snap_500_1000", pd.Series(0, index=df.index)).fillna(0).astype(float)
    n_snap_pool = n_snap_200_500 + n_snap_500_1000

    if "n_low_acc_dedup" in df.columns:
        n_lowacc_pool = df["n_low_acc_dedup"].fillna(0).astype(float)
    elif "n_lowacc_pool" in df.columns:
        n_lowacc_pool = df["n_lowacc_pool"].fillna(0).astype(float)
    else:
        n_lowacc_pool = pd.Series(0, index=df.index, dtype=float)

    out = pd.DataFrame(
        {
            "species": df["species"].values,
            "n_experiment_assumed": n_exp.astype(int).values,
            "n_snap_pool": n_snap_pool.astype(int).values,
            "n_lowacc_pool": n_lowacc_pool.astype(int).values,
        }
    )

    for level in snap_levels:
        out[f"feas_snap_{level}"] = (n_snap_pool >= (level / 100.0) * n_exp).astype(int).values

    for level in lowacc_levels:
        out[f"feas_lowacc_{level}"] = (n_lowacc_pool >= (level / 100.0) * n_exp).astype(int).values

    out["max_snap_contamination_pct"] = _max_supported_level(n_snap_pool, n_exp, snap_levels).values
    out["max_lowacc_contamination_pct"] = _max_supported_level(n_lowacc_pool, n_exp, lowacc_levels).values

    logger.info(
        f"Feasibility computed for {len(out)} species "
        f"(policy={policy}, snap_levels={snap_levels}, lowacc_levels={lowacc_levels})."
    )
    return out
