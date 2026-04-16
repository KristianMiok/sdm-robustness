"""Task 1 — Step 1.3: candidacy gates and classification.

Apply category-dependent quantitative gates to produce PRIMARY / PARTIAL /
INELIGIBLE classifications. Gates are:

    1a/b/c  Minimum benchmark size (category-dependent)
    2       Snapping pool supports 50% contamination
    3       Low-accuracy pool supports 50% contamination
    4       Basin spread: ≥ 3 unique basins
    5       Strahler spread: ≥ 3 distinct orders

Classification rules (briefing §3.5):
    PRIMARY     = passes gates {1, 2, 3, 4, 5}. Usable for both axes, full 50%.
    PARTIAL     = passes gates {1, 4, 5} and at least one of {2, 3}.
    INELIGIBLE  = fails gates {1, 4, or 5}.
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
) -> pd.DataFrame:
    """Apply gates and classify each species.

    Parameters
    ----------
    inventory : DataFrame
        From audit.inventory.build_inventory().
    feasibility : DataFrame
        From audit.feasibility.compute_feasibility().
    gates_config : dict
        Loaded task1_gates.yaml.
    default_category : str
        Category to use if category_petko2026 is NA. 'regional' is the safer
        middle ground — 'endemic' would pass more species, 'widespread' fewer.

    Returns
    -------
    DataFrame with per-gate pass/fail flags and the classification label.
    """
    inv = inventory.set_index("species")
    fea = feasibility.set_index("species")
    both = inv.join(fea, how="inner", rsuffix="_fea")

    # Resolve category per species (Petko value or fallback)
    cats = both["category_petko2026"].fillna(default_category)
    # Normalise just in case of case differences
    cats = cats.str.lower().map(
        {"endemic": "endemic", "regional": "regional", "widespread": "widespread",
         "cosmopolitan": "widespread", "narrow": "endemic"}
    ).fillna(default_category)

    # --- Gate 1: minimum benchmark (category-dependent) ---
    min_map = gates_config["gate_1_minimum_benchmark"]
    min_required = cats.map(min_map).astype(float)
    gate_1 = (both["n_clean_dedup_200m"] >= min_required).astype(int)

    # --- Gate 2: snapping pool supports 50% contamination ---
    # Required: n_snap_pool ≥ n_clean_dedup_200m (equivalent to briefing statement)
    gate_2 = (both["n_snap_pool"] >= both["n_clean_dedup_200m"]).astype(int)

    # --- Gate 3: low-acc pool supports 50% contamination ---
    gate_3 = (both["n_lowacc_pool"] >= both["n_clean_dedup_200m"]).astype(int)

    # --- Gate 4: basin spread ---
    min_basins = gates_config["gate_4_basin_spread"]["min_basins"]
    gate_4 = (both["n_basins"] >= min_basins).astype(int)

    # --- Gate 5: Strahler spread ---
    min_orders = gates_config["gate_5_strahler_spread"]["min_distinct_orders"]
    n_strahler = (both["strahler_max"] - both["strahler_min"] + 1).fillna(0)
    gate_5 = (n_strahler >= min_orders).astype(int)

    # --- Classification ---
    all_core = (gate_1 == 1) & (gate_4 == 1) & (gate_5 == 1)
    both_pools = (gate_2 == 1) & (gate_3 == 1)
    any_pool = (gate_2 == 1) | (gate_3 == 1)

    status = pd.Series("INELIGIBLE", index=both.index, dtype="object")
    status.loc[all_core & both_pools] = "PRIMARY"
    status.loc[all_core & ~both_pools & any_pool] = "PARTIAL"

    out = pd.DataFrame(
        {
            "species": both.index,
            "category_used": cats.values,
            "n_clean_dedup_200m": both["n_clean_dedup_200m"].values,
            "n_snap_pool": both["n_snap_pool"].values,
            "n_lowacc_pool": both["n_lowacc_pool"].values,
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
