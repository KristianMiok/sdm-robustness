"""Task 4 — Inference Stability Threshold (IST).

The central citable output of the paper (briefing §6.4).

For each (species, metric, axis), the IST is the lowest contamination level
at which a chosen diagnostic drops below an operational bound.

Defaults:
    diagnostic      = Spearman rank correlation of variable importance
    operational_bound = 0.7 (below this, rank agreement is considered weak)
    replicate aggr. = mean over 30 replicates

Robustness: IST curves are also computed at 0.6 and 0.8 bounds.

Status: SCAFFOLD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class IST:
    """Inference Stability Threshold result for one (species, metric, axis)."""

    species: str
    axis: str
    metric: str
    bound: float
    ist_pct: float | None          # None if no tested level violates the bound
    diagnostic_curve: pd.Series    # level_pct → mean diagnostic value


def compute_ist(
    degradation_curve: pd.Series,
    *,
    bound: float = 0.7,
    monotone_only: bool = False,
) -> float | None:
    """Find the lowest contamination level where the diagnostic falls below
    `bound`.

    Parameters
    ----------
    degradation_curve : Series
        Indexed by contamination level (percent), values are the diagnostic
        (e.g., mean Spearman across replicates).
    bound : float
        Operational threshold. Default 0.7.
    monotone_only : bool
        If True, require that the curve stays below `bound` for all higher
        levels — guards against noisy single-level dips.

    Returns
    -------
    float or None
        The smallest level_pct at which the curve drops below `bound`, or
        None if it never does.

    TODO: implement with ascending-level iteration and optional monotone check.
    """
    raise NotImplementedError("Task 4 — implement")
