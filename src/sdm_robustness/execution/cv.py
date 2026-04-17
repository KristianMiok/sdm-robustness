from __future__ import annotations

import pandas as pd


def assign_basin_folds(
    basin_ids: pd.Series,
    *,
    n_splits: int = 5,
    looo_threshold: int = 15,
) -> dict[str, int]:
    """
    Deterministic basin-to-fold assignment.

    Default:
    - grouped 5-fold CV
    - basins sorted by descending benchmark presence count, then basin_id
    - greedy assignment to the fold with current smallest total count

    Fallback:
    - if unique basins < looo_threshold, use leave-one-out by basin
    """
    s = pd.Series(basin_ids).dropna().astype(str)
    basins = sorted(s.unique().tolist())
    n_basins = len(basins)

    if n_basins == 0:
        raise ValueError("No basin IDs available for fold assignment.")

    if n_basins < looo_threshold:
        return {b: i for i, b in enumerate(basins)}

    counts = s.value_counts()
    basin_order = sorted(
        counts.index.tolist(),
        key=lambda b: (-int(counts[b]), str(b)),
    )

    fold_loads = {i: 0 for i in range(n_splits)}
    assignment: dict[str, int] = {}

    for basin in basin_order:
        target_fold = min(fold_loads, key=lambda f: (fold_loads[f], f))
        assignment[str(basin)] = target_fold
        fold_loads[target_fold] += int(counts[basin])

    return assignment
