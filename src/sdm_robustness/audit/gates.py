"""Task 1 — Step 1.3: candidacy gates and classification.

Data-driven-ceiling variant (2026-04-16): Petko-filtered data can support
low contamination for many species, but 50% contamination is too strict for
almost all species on the snapping axis. So gates 2/3 now require support
for the minimum tested contamination level (default 5%), while PRIMARY keeps
the strict 50% requirement on both axes.

Gates:
    1       Minimum benchmark size (category-dependent)
    2       Snapping pool supports >= min_level_pct contamination
    3       Low-accuracy pool supports >= min_level_pct contamination
    4       Basin spread: >= 3 unique basins
    5       Strahler spread: >= 3 distinct orders

Classification:
    PRIMARY     = passes gates {1,4,5} and both axes support strict_ceiling_pct
    PARTIAL     = passes gates {1,4,5} and at least one axis supports min_level_pct
    INELIGIBLE  = otherwise
"""

from __future__ import annotations

import pandas as pd

from sdm_robustness.utils import logger


def classify_candidates(
    inventory: pd.DataFrame,
    feasibility: pd.DataFrame,
    gates_config: dict,
    *,
    default_category: str = "regional",
    min_level_pct: int = 5,
    strict_ceiling_pct: int = 50,
) -> pd.DataFrame:
    """Apply Task 1 gates and classify species."""
    inv = inventory.set_index("species")
    fea = feasibility.set_index("species")
    both = inv.join(fea, how="inner", rsuffix="_fea")

    cats = both["category_petko2026"].fillna(default_category)
    cats = (
        cats.astype(str)
        .str.lower()
        .map(
            {
                "endemic": "endemic",
                "regional": "regional",
                "widespread": "widespread",
                "cosmopolitan": "widespread",
                "narrow": "endemic",
            }
        )
        .fillna(default_category)
    )

    min_map = gates_config["gate_1_minimum_benchmark"]
    min_required = cats.map(min_map).astype(float)
    gate_1 = (both["n_clean_dedup_200m"] >= min_required).astype(int)

    n_exp = both["n_experiment_assumed"].astype(float)
    required_min = (min_level_pct / 100.0) * n_exp
    required_strict = (strict_ceiling_pct / 100.0) * n_exp

    gate_2 = (both["n_snap_pool"].astype(float) >= required_min).astype(int)
    gate_3 = (both["n_lowacc_pool"].astype(float) >= required_min).astype(int)

    min_basins = gates_config["gate_4_basin_spread"]["min_basins"]
    gate_4 = (both["n_basins"].fillna(0).astype(float) >= float(min_basins)).astype(int)

    min_orders = gates_config["gate_5_strahler_spread"]["min_distinct_orders"]
    n_strahler = (
        both["strahler_max"].fillna(0).astype(float)
        - both["strahler_min"].fillna(0).astype(float)
        + 1
    )
    gate_5 = (n_strahler >= float(min_orders)).astype(int)

    strict_snap = both["n_snap_pool"].astype(float) >= required_strict
    strict_lowacc = both["n_lowacc_pool"].astype(float) >= required_strict

    core_pass = (gate_1 == 1) & (gate_4 == 1) & (gate_5 == 1)
    any_pool_min = (gate_2 == 1) | (gate_3 == 1)
    both_pools_strict = strict_snap & strict_lowacc

    status = pd.Series("INELIGIBLE", index=both.index, dtype="object")
    status.loc[core_pass & any_pool_min] = "PARTIAL"
    status.loc[core_pass & both_pools_strict] = "PRIMARY"

    out = pd.DataFrame(
        {
            "species": both.index,
            "category_used": cats.values,
            "n_clean_dedup_200m": both["n_clean_dedup_200m"].values,
            "n_snap_pool": both["n_snap_pool"].values,
            "n_lowacc_pool": both["n_lowacc_pool"].values,
            "max_snap_contamination_pct": both["max_snap_contamination_pct"].values,
            "max_lowacc_contamination_pct": both["max_lowacc_contamination_pct"].values,
            "n_basins": both["n_basins"].values,
            "strahler_min": both["strahler_min"].values,
            "strahler_max": both["strahler_max"].values,
            "gate_1_min_benchmark": gate_1.values,
            "gate_2_snap_pool": gate_2.values,
            "gate_3_lowacc_pool": gate_3.values,
            "gate_4_basin_spread": gate_4.values,
            "gate_5_strahler_spread": gate_5.values,
            "strict_50pct_snap": strict_snap.astype(int).values,
            "strict_50pct_lowacc": strict_lowacc.astype(int).values,
            "classification": status.values,
        }
    ).reset_index(drop=True)

    counts = out["classification"].value_counts()
    logger.info(f"Classification counts: {counts.to_dict()}")
    logger.info(
        f"Gate config: min_level_pct={min_level_pct}, strict_ceiling_pct={strict_ceiling_pct}"
    )
    return out
