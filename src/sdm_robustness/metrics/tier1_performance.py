"""Task 4 — Tier 1: performance degradation metrics.

ΔAUC, ΔTSS, ΔBoyce, ΔBrier, sensitivity, specificity.

Every comparison (benchmark vs. one contaminated run) produces a row of
standardised Tier 1 metrics.

Status: SCAFFOLD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tier1Metrics:
    auc: float
    tss: float
    boyce: float
    brier: float
    sensitivity: float
    specificity: float


def compute_tier1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: str | float = "max_sss",
) -> Tier1Metrics:
    """Compute all Tier 1 metrics for one model run.

    TODO: implement.
    - AUC via sklearn.metrics.roc_auc_score
    - TSS = sensitivity + specificity - 1 at the chosen threshold
    - Boyce via continuous Boyce index (bin-based correlation)
    - Brier via sklearn.metrics.brier_score_loss
    - threshold: 'max_sss' (max sensitivity+specificity), or a fixed float
    """
    raise NotImplementedError("Task 4 — implement")


def delta_tier1(benchmark: Tier1Metrics, contaminated: Tier1Metrics) -> Tier1Metrics:
    """Element-wise contaminated − benchmark deltas."""
    return Tier1Metrics(
        auc=contaminated.auc - benchmark.auc,
        tss=contaminated.tss - benchmark.tss,
        boyce=contaminated.boyce - benchmark.boyce,
        brier=contaminated.brier - benchmark.brier,
        sensitivity=contaminated.sensitivity - benchmark.sensitivity,
        specificity=contaminated.specificity - benchmark.specificity,
    )
