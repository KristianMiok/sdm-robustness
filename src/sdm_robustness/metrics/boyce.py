"""Continuous Boyce index (ecospat-style).

The continuous Boyce index measures how well a habitat suitability model
predicts presence-only data. It compares the predicted/expected (P/E) ratio
across moving suitability bins, then computes Spearman correlation between
the bin midpoints and the P/E ratios.

Reference: Hirzel et al. 2006, Ecological Modelling 199:142-152.
Implementation follows the ecospat R package convention (continuous,
moving-window, Spearman).

A model that predicts presences well will produce monotonically increasing
P/E ratios with suitability — yielding a Boyce index near +1. A random
model produces ratios near 1 across all bins → correlation near 0.
A model that predicts the opposite of the truth → correlation near -1.

Range: [-1, 1]. NaN if too few non-empty bins.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def boyce_index(
    suitability_at_presences: np.ndarray,
    suitability_background: np.ndarray,
    n_bins: int = 10,
    window_width: float | None = None,
) -> float:
    """Continuous Boyce index using moving-window approach.

    Parameters
    ----------
    suitability_at_presences : array-like of float
        Predicted suitability values at observed presence points.
    suitability_background : array-like of float
        Predicted suitability values across the full study area
        (typically all background/accessible-area points).
    n_bins : int, default 10
        Number of moving windows to use across the suitability range.
    window_width : float or None, default None
        Width of each moving window in suitability units. If None,
        defaults to (range / n_bins) * 2 — overlapping windows that
        each cover ~20% of the range. Standard ecospat behaviour.

    Returns
    -------
    float
        Spearman rank correlation between bin midpoints and P/E ratios.
        Range [-1, 1]. Returns NaN if fewer than 3 non-empty bins remain.
    """
    pres = np.asarray(suitability_at_presences, dtype=float)
    bkg = np.asarray(suitability_background, dtype=float)

    pres = pres[~np.isnan(pres)]
    bkg = bkg[~np.isnan(bkg)]

    if len(pres) == 0 or len(bkg) == 0:
        return float("nan")

    s_min = float(min(pres.min(), bkg.min()))
    s_max = float(max(pres.max(), bkg.max()))
    if s_max <= s_min:
        return float("nan")

    if window_width is None:
        window_width = (s_max - s_min) / n_bins * 2.0

    # Bin centres span the full range
    centres = np.linspace(s_min + window_width / 2,
                          s_max - window_width / 2, n_bins)

    n_pres_total = len(pres)
    n_bkg_total = len(bkg)

    midpoints = []
    pe_ratios = []
    for c in centres:
        lo = c - window_width / 2
        hi = c + window_width / 2

        f_pres = ((pres >= lo) & (pres < hi)).sum() / n_pres_total
        f_bkg = ((bkg >= lo) & (bkg < hi)).sum() / n_bkg_total

        if f_bkg == 0:
            # Avoid division by zero — skip bin (ecospat convention)
            continue

        pe = f_pres / f_bkg
        midpoints.append(c)
        pe_ratios.append(pe)

    if len(midpoints) < 3:
        return float("nan")

    rho, _ = spearmanr(midpoints, pe_ratios)
    return float(rho)
