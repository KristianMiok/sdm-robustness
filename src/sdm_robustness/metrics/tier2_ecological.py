"""Task 4 — Tier 2: ecological signal instability metrics.

Captures whether the variables driving the model and the shapes of their
responses remain stable under contamination. This is the tier most likely
to destabilise first (hypothesis H1).

Metrics:
    - Spearman rank correlation of variable importance (benchmark vs. contam.)
    - Jaccard overlap on top-5 and top-10 variables
    - Response curve distance: integrated absolute difference per top-10 variable
    - Niche centroid displacement in environmental PCA space
    - Niche breadth change (Petko et al. 2026 standardised range)

Status: SCAFFOLD.

Kristian's note: plain Spearman on variable importance is sensitive to the
long tail of near-zero-importance predictors. Default diagnostic here will
be weighted top-K overlap; plain Spearman reported alongside for robustness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def importance_rank_correlation(
    bench_importance: pd.Series, contam_importance: pd.Series, *, method: str = "spearman"
) -> float:
    """Rank correlation between two variable-importance vectors.

    TODO: align indices, handle NAs, return Spearman or Kendall tau.
    """
    raise NotImplementedError("Task 4 — implement")


def top_k_jaccard(
    bench_importance: pd.Series,
    contam_importance: pd.Series,
    *,
    k: int = 10,
    weighted: bool = False,
) -> float:
    """Jaccard overlap on the top-K most important variables.

    If weighted=True, each variable contributes its normalised importance
    (Kristian's recommended robust variant).

    TODO: implement.
    """
    raise NotImplementedError("Task 4 — implement")


def response_curve_distance(
    bench_curves: dict[str, np.ndarray],
    contam_curves: dict[str, np.ndarray],
    *,
    top_k_vars: list[str],
) -> float:
    """Mean integrated absolute difference across top-K response curves.

    TODO: implement — assumes curves evaluated on a common grid per variable.
    """
    raise NotImplementedError("Task 4 — implement")


def niche_centroid_displacement(
    bench_env: pd.DataFrame, contam_env: pd.DataFrame, *, n_components: int = 2
) -> float:
    """Euclidean distance between benchmark and contaminated centroids in
    environmental PCA space.

    TODO: implement — fit PCA on combined data, project both, compare centroids.
    """
    raise NotImplementedError("Task 4 — implement")


def niche_breadth_change(bench_scores: np.ndarray, contam_scores: np.ndarray) -> float:
    """Relative change in standardised niche range.

    Defined analogously to Petko et al. 2026 (median standardised range).
    TODO: implement.
    """
    raise NotImplementedError("Task 4 — implement")
