"""Task 4 — Tier 3: spatial prediction instability metrics.

Metrics:
    - Schoener's D between suitability rasters
    - Warren's I
    - Predicted range area percent change (binarised at 0.5 and MaxSSS)
    - Spatial mismatch maps: gained / lost / stable pixels

Status: SCAFFOLD.
"""

from __future__ import annotations

import numpy as np


def schoeners_d(r1: np.ndarray, r2: np.ndarray) -> float:
    """Schoener's D between two suitability rasters (values in [0, 1]).

    D = 1 - 0.5 * sum(|p1 - p2|), after normalising each raster to sum to 1.
    TODO: implement with NaN handling.
    """
    raise NotImplementedError("Task 4 — implement")


def warrens_i(r1: np.ndarray, r2: np.ndarray) -> float:
    """Warren's I between two suitability rasters.

    I = 1 - 0.5 * sum((sqrt(p1) - sqrt(p2))^2).
    TODO: implement.
    """
    raise NotImplementedError("Task 4 — implement")


def range_area_percent_change(
    bench_binary: np.ndarray, contam_binary: np.ndarray, cell_area_km2: float
) -> float:
    """Percent change in predicted range area.

    TODO: implement — sum True pixels * cell area, return (contam - bench)/bench.
    """
    raise NotImplementedError("Task 4 — implement")


def spatial_mismatch_map(
    bench_binary: np.ndarray, contam_binary: np.ndarray
) -> np.ndarray:
    """Per-pixel category: 0 stable-absent, 1 stable-present, 2 gained, 3 lost.

    TODO: implement.
    """
    raise NotImplementedError("Task 4 — implement")
