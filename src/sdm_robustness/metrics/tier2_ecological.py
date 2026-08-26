"""Tier 2 — stability of the environmental signal under contamination.

T8.2: these are NOT niche metrics. The species' niche does not change during
the experiment; what changes is the environmental space the occurrence sample
covers. Names reflect that. The old names are kept as aliases.

T8.3: the PCA is documented rather than implicit. It is fitted on the two
samples combined, so both are projected into one space; inputs are standardised
because the raw predictors span very different units; the number of retained
components is explicit; and the same fitted PCA is reused for every treatment
within an entity when `fitted_pca` is supplied.

T8.5: full-vector rank correlation is the primary importance metric. Top-K
Jaccard is retained for continuity but is arbitrary among 398 correlated
predictors - a swap between ranks 5 and 6 moves it with no ecological meaning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def importance_rank_correlation(
    bench_importance: pd.Series,
    contam_importance: pd.Series,
    *,
    method: str = "spearman",
) -> float:
    """Rank correlation over the FULL importance vector (T8.5 primary metric)."""
    b = pd.Series(bench_importance).astype(float)
    c = pd.Series(contam_importance).astype(float)
    idx = b.index.union(c.index)
    b, c = b.reindex(idx).fillna(0.0), c.reindex(idx).fillna(0.0)
    if len(b) < 3 or b.nunique() < 2 or c.nunique() < 2:
        return float("nan")
    f = spearmanr if method == "spearman" else kendalltau
    return float(f(b.values, c.values).statistic)


def top_k_jaccard(
    bench_importance: pd.Series,
    contam_importance: pd.Series,
    *,
    k: int = 10,
    weighted: bool = False,
) -> float:
    """Jaccard overlap on the top-K variables. Reported for continuity only."""
    b = pd.Series(bench_importance).astype(float).nlargest(k)
    c = pd.Series(contam_importance).astype(float).nlargest(k)
    sb, sc = set(b.index), set(c.index)
    if not sb and not sc:
        return float("nan")
    if not weighted:
        return len(sb & sc) / len(sb | sc)
    w = pd.concat([b, c]).groupby(level=0).max()
    w = w / w.sum() if w.sum() else w
    inter = float(w.reindex(list(sb & sc)).sum())
    union = float(w.reindex(list(sb | sc)).sum())
    return inter / union if union else float("nan")


def fit_environmental_pca(
    *samples: pd.DataFrame,
    n_components: int = 2,
    random_state: int = 0,
) -> tuple[StandardScaler, PCA, list[str]]:
    """Fit one standardised PCA on the samples combined (T8.3).

    Returns (scaler, pca, columns) so the identical transform can be reused
    across every treatment of an entity rather than refitted per comparison.
    """
    cols = list(samples[0].columns)
    X = pd.concat([s[cols] for s in samples], axis=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.loc[:, X.notna().any(axis=0)]
    cols = list(X.columns)
    X = X.fillna(X.median(numeric_only=True))
    sc = StandardScaler().fit(X)
    n = min(n_components, X.shape[1], max(1, X.shape[0] - 1))
    pca = PCA(n_components=n, random_state=random_state).fit(sc.transform(X))
    return sc, pca, cols


def _project(df, fitted):
    sc, pca, cols = fitted
    X = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return pca.transform(sc.transform(X))


def sample_centroid_displacement(
    bench_env: pd.DataFrame,
    contam_env: pd.DataFrame,
    *,
    n_components: int = 2,
    fitted_pca=None,
) -> float:
    """Distance between the two samples' centroids in standardised PCA space.

    This is displacement of the OCCURRENCE SAMPLE in environmental space, not
    of the species' niche (T8.2). Because it is computed from records rather
    than from fitted models, it belongs in Tier 0 (T8.1).
    """
    fitted = fitted_pca or fit_environmental_pca(
        bench_env, contam_env, n_components=n_components)
    b, c = _project(bench_env, fitted), _project(contam_env, fitted)
    if not len(b) or not len(c):
        return float("nan")
    return float(np.linalg.norm(b.mean(axis=0) - c.mean(axis=0)))


def sample_dispersion_change(
    bench_env: pd.DataFrame,
    contam_env: pd.DataFrame,
    *,
    n_components: int = 2,
    fitted_pca=None,
    relative: bool = True,
) -> float:
    """Change in the dispersion of the occurrence sample in PCA space.

    Dispersion is the mean distance to the sample's own centroid. Returns the
    relative change (contaminated / benchmark - 1) by default, the absolute
    difference otherwise.
    """
    fitted = fitted_pca or fit_environmental_pca(
        bench_env, contam_env, n_components=n_components)
    b, c = _project(bench_env, fitted), _project(contam_env, fitted)
    if not len(b) or not len(c):
        return float("nan")
    db = float(np.mean(np.linalg.norm(b - b.mean(axis=0), axis=1)))
    dc = float(np.mean(np.linalg.norm(c - c.mean(axis=0), axis=1)))
    if not relative:
        return dc - db
    return (dc / db - 1.0) if db else float("nan")


def response_curve_distance(
    bench_curves: dict[str, np.ndarray],
    contam_curves: dict[str, np.ndarray],
    *,
    top_k_vars: list[str],
) -> float:
    """Mean integrated absolute difference across the given response curves (T8.6).

    Curves are assumed evaluated on a common grid per variable and are
    normalised to [0, 1] on the benchmark range before comparison, so the
    result is comparable across variables of different scale.
    """
    vals = []
    for v in top_k_vars:
        b, c = bench_curves.get(v), contam_curves.get(v)
        if b is None or c is None or len(b) != len(c) or len(b) < 2:
            continue
        b, c = np.asarray(b, float), np.asarray(c, float)
        lo, hi = np.nanmin(b), np.nanmax(b)
        if not np.isfinite(lo) or hi <= lo:
            continue
        vals.append(float(np.trapezoid(np.abs((c - lo)/(hi - lo) - (b - lo)/(hi - lo)),
                                       np.linspace(0, 1, len(b)))))
    return float(np.mean(vals)) if vals else float("nan")


# Deprecated names (T8.2). Kept so existing imports do not break.
niche_centroid_displacement = sample_centroid_displacement
niche_breadth_change = sample_dispersion_change
