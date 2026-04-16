"""Task 1 — Step 1.2: per-species feasibility metrics.

Because contamination is substitution (not addition) with N_experiment held
constant, feasibility at level p% requires:
    (p/100) × N_experiment records available in the contamination pool, AND
    (1 - p/100) × N_experiment records available in the clean pool.

Default convention (briefing §3.4): N_experiment is set to the deduplicated
clean benchmark size per species. Under this choice, the clean-pool side is
trivially satisfied (clean pool IS the benchmark), so feasibility reduces to
a contamination-pool check.

We expose the N_experiment choice as an argument so Task 2 (panel selection)
can explore alternatives.
"""

from __future__ import annotations

import pandas as pd

from sdm_robustness.utils import logger

CONTAMINATION_LEVELS_PCT = (5, 10, 20, 35, 50)


def compute_feasibility(
    inventory: pd.DataFrame,
    *,
    levels_pct: tuple[int, ...] = CONTAMINATION_LEVELS_PCT,
    n_experiment_policy: str = "benchmark",
) -> pd.DataFrame:
    """Compute feasibility metrics per species.

    Parameters
    ----------
    inventory : DataFrame
        Output of audit.inventory.build_inventory().
    levels_pct : tuple of int
        Contamination levels to check (in percent).
    n_experiment_policy : {'benchmark', 'max_feasible'}
        'benchmark' — N_experiment = n_clean_dedup_200m (briefing default).
        'max_feasible' — N_experiment = 2 × min(n_clean_dedup_200m, pool)
                         (exploration only; not the production default).

    Returns
    -------
    DataFrame with feasibility metrics, indexed by species.
    """
    df = inventory.set_index("species").copy()

    # Core pool sizes — fillna(0) guards against species with zero records
    # in a given pool (these rows might otherwise carry NaN through reindex).
    n_clean = df["n_clean_dedup_200m"].fillna(0).astype(int)
    n_snap_pool = (
        df["n_snap_200_500"].fillna(0) + df["n_snap_500_1000"].fillna(0)
    ).astype(int)
    n_lowacc_pool = df["n_low_acc_dedup"].fillna(0).astype(int)

    # Max feasible experiment sizes (briefing §3.4 formula)
    n_exp_max_snap = 2 * n_clean.combine(n_snap_pool, min)
    n_exp_max_lowacc = 2 * n_clean.combine(n_lowacc_pool, min)

    # Choice of N_experiment for the feasibility flags
    if n_experiment_policy == "benchmark":
        n_experiment = n_clean
    elif n_experiment_policy == "max_feasible":
        # Use whichever axis is more limiting — symmetric.
        n_experiment = pd.concat([n_exp_max_snap, n_exp_max_lowacc], axis=1).min(axis=1)
    else:
        raise ValueError(f"Unknown n_experiment_policy: {n_experiment_policy}")

    out = pd.DataFrame(index=df.index)
    out["n_clean_dedup_200m"] = n_clean
    out["n_snap_pool"] = n_snap_pool
    out["n_lowacc_pool"] = n_lowacc_pool
    out["n_experiment_assumed"] = n_experiment
    out["n_exp_max_snap"] = n_exp_max_snap
    out["n_exp_max_lowacc"] = n_exp_max_lowacc

    # Largest contamination % achievable per axis (value in levels_pct or the
    # largest feasible lower value, -1 if no level is feasible).
    out["max_snap_contamination_pct"] = _max_feasible_level(
        n_experiment, n_snap_pool, levels_pct
    )
    out["max_lowacc_contamination_pct"] = _max_feasible_level(
        n_experiment, n_lowacc_pool, levels_pct
    )

    # Binary feasibility flags per level per axis
    for level in levels_pct:
        required = (level / 100.0) * n_experiment
        out[f"feas_snap_{level}"] = (n_snap_pool >= required).astype(int)
        out[f"feas_lowacc_{level}"] = (n_lowacc_pool >= required).astype(int)

    # 2D joint feasibility — only meaningful if both pools can jointly supply
    # contaminated records on the same benchmark. Conservative: require each
    # axis to independently satisfy 50% contamination.
    out["feas_2d"] = (
        (n_snap_pool >= 0.5 * n_experiment) & (n_lowacc_pool >= 0.5 * n_experiment)
    ).astype(int)

    logger.info(
        f"Feasibility computed for {len(out)} species "
        f"(policy={n_experiment_policy}, levels={levels_pct})."
    )
    return out.reset_index()


def _max_feasible_level(
    n_experiment: pd.Series, pool: pd.Series, levels_pct: tuple[int, ...]
) -> pd.Series:
    """Return the largest contamination level in levels_pct that fits.

    -1 if no level from the list is feasible.
    """
    result = pd.Series(-1, index=n_experiment.index, dtype=int)
    # Iterate levels from highest to lowest — first feasible wins.
    for level in sorted(levels_pct, reverse=True):
        required = (level / 100.0) * n_experiment
        feasible = pool >= required
        # Only update cells that haven't been assigned yet (-1).
        mask = feasible & (result == -1)
        result.loc[mask] = level
    return result
