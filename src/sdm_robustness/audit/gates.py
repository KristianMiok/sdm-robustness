"""Task 1 — Step 1.3: revised candidacy gates after Lucian's design revision."""

from __future__ import annotations

import pandas as pd

from sdm_robustness.utils import logger


def classify_candidates(
    inventory: pd.DataFrame,
    feasibility: pd.DataFrame,
    gates_config: dict,
    *,
    default_category: str = "regional",
) -> pd.DataFrame:
    """Apply revised Task 1 gates and classify each species.

    Revised classes:
      - DUAL-AXIS
      - SNAPPING-ONLY
      - LOW-ACC-ONLY
      - INELIGIBLE

    Revised feasibility requirements:
      - Gate 2: supports snapping 5%
      - Gate 3: supports low-accuracy 20%
    """
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
                "narrow": "endemic",
                "endemic/narrow-range": "endemic",
                "narrow-range": "endemic",
                "regional": "regional",
                "widespread": "widespread",
                "cosmopolitan": "widespread",
                "widespread/cosmopolitan": "widespread",
            }
        )
        .fillna(default_category)
    )

    min_map = gates_config["gate_1_minimum_benchmark"]
    min_required = cats.map(min_map).astype(float)
    gate_1 = (both["n_clean_dedup_200m"].fillna(0).astype(float) >= min_required).astype(int)

    gate_2 = both["feas_snap_5"].fillna(0).astype(int)
    gate_3 = both["feas_lowacc_20"].fillna(0).astype(int)

    min_basins = gates_config["gate_4_basin_spread"]["min_basins"]
    gate_4 = (both["n_basins"].fillna(0).astype(float) >= float(min_basins)).astype(int)

    min_orders = gates_config["gate_5_strahler_spread"]["min_distinct_orders"]
    n_strahler = (
        both["strahler_max"].fillna(0).astype(float)
        - both["strahler_min"].fillna(0).astype(float)
        + 1
    )
    gate_5 = (n_strahler >= float(min_orders)).astype(int)

    core_pass = (gate_1 == 1) & (gate_4 == 1) & (gate_5 == 1)

    status = pd.Series("INELIGIBLE", index=both.index, dtype="object")
    status.loc[core_pass & (gate_2 == 1) & (gate_3 == 1)] = "DUAL-AXIS"
    status.loc[core_pass & (gate_2 == 1) & (gate_3 == 0)] = "SNAPPING-ONLY"
    status.loc[core_pass & (gate_2 == 0) & (gate_3 == 1)] = "LOW-ACC-ONLY"

    out = pd.DataFrame(
        {
            "species": both.index,
            "category_used": cats.values,
            "n_clean_dedup_200m": both["n_clean_dedup_200m"].values,
            "n_snap_pool": both["n_snap_pool"].values,
            "n_lowacc_pool": both["n_lowacc_pool"].values,
            "feas_snap_1": both.get("feas_snap_1", 0),
            "feas_snap_2": both.get("feas_snap_2", 0),
            "feas_snap_5": both.get("feas_snap_5", 0),
            "feas_lowacc_3": both.get("feas_lowacc_3", 0),
            "feas_lowacc_10": both.get("feas_lowacc_10", 0),
            "feas_lowacc_20": both.get("feas_lowacc_20", 0),
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
            "classification": status.values,
        }
    ).reset_index(drop=True)

    counts = out["classification"].value_counts()
    logger.info(f"Classification counts: {counts.to_dict()}")
    return out
