"""Core metric functions for robustness analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from sdm_robustness.metrics.boyce import boyce_index


def compute_performance_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    tss = sensitivity + specificity - 1 if np.isfinite(sensitivity) and np.isfinite(specificity) else np.nan

    # Continuous Boyce index (Hirzel et al. 2006, ecospat-style)
    presence_scores = y_score[y_true == 1]
    background_scores = y_score
    boyce = boyce_index(presence_scores, background_scores)

    return {
        "auc": float(auc),
        "brier": float(brier),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tss": float(tss),
        "boyce": float(boyce),
    }


def compute_delta_performance(
    benchmark: dict[str, float],
    current: dict[str, float],
) -> dict[str, float]:
    out = {}
    for k, v in current.items():
        bk = benchmark.get(k, np.nan)
        out[f"delta_{k}"] = float(v - bk) if np.isfinite(v) and np.isfinite(bk) else np.nan
    return out


def spearman_importance_stability(
    benchmark_importance: np.ndarray,
    current_importance: np.ndarray,
) -> float:
    benchmark_importance = np.asarray(benchmark_importance)
    current_importance = np.asarray(current_importance)
    corr, _ = spearmanr(benchmark_importance, current_importance)
    return float(corr)


def topk_jaccard(
    benchmark_importance: np.ndarray,
    current_importance: np.ndarray,
    k: int,
) -> float:
    b = np.asarray(benchmark_importance)
    c = np.asarray(current_importance)

    b_idx = set(np.argsort(b)[-k:])
    c_idx = set(np.argsort(c)[-k:])
    union = b_idx | c_idx
    if not union:
        return np.nan
    return len(b_idx & c_idx) / len(union)


def integrated_absolute_difference(
    x: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
) -> float:
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    return float(np.trapezoid(np.abs(y1 - y2), x))


def centroid_displacement(
    benchmark_points: np.ndarray,
    current_points: np.ndarray,
) -> float:
    b = np.asarray(benchmark_points)
    c = np.asarray(current_points)
    b_cent = b.mean(axis=0)
    c_cent = c.mean(axis=0)
    return float(np.linalg.norm(b_cent - c_cent))


def niche_breadth_change(
    benchmark_points: np.ndarray,
    current_points: np.ndarray,
) -> float:
    b = np.asarray(benchmark_points)
    c = np.asarray(current_points)
    b_breadth = np.mean(np.std(b, axis=0))
    c_breadth = np.mean(np.std(c, axis=0))
    return float(c_breadth - b_breadth)


def _normalise_surface(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    s = arr.sum()
    if s == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / s


def schoeners_d(
    benchmark_surface: np.ndarray,
    current_surface: np.ndarray,
) -> float:
    p = _normalise_surface(benchmark_surface.ravel())
    q = _normalise_surface(current_surface.ravel())
    return float(1.0 - 0.5 * np.sum(np.abs(p - q)))


def warrens_i(
    benchmark_surface: np.ndarray,
    current_surface: np.ndarray,
) -> float:
    p = _normalise_surface(benchmark_surface.ravel())
    q = _normalise_surface(current_surface.ravel())
    return float(1.0 - 0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def range_area_change(
    benchmark_surface: np.ndarray,
    current_surface: np.ndarray,
    threshold: float = 0.5,
) -> float:
    b = (np.asarray(benchmark_surface) >= threshold).astype(int)
    c = (np.asarray(current_surface) >= threshold).astype(int)
    b_area = b.sum()
    c_area = c.sum()
    if b_area == 0:
        return np.nan
    return float((c_area - b_area) / b_area * 100.0)


def spatial_mismatch_summary(
    benchmark_surface: np.ndarray,
    current_surface: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    b = (np.asarray(benchmark_surface) >= threshold).astype(int)
    c = (np.asarray(current_surface) >= threshold).astype(int)

    stable = np.sum((b == 1) & (c == 1))
    loss = np.sum((b == 1) & (c == 0))
    gain = np.sum((b == 0) & (c == 1))
    total_change = gain + loss

    return {
        "stable_presence_cells": float(stable),
        "loss_cells": float(loss),
        "gain_cells": float(gain),
        "total_changed_cells": float(total_change),
    }


def compute_ist(
    levels: Iterable[float],
    mean_spearman_values: Iterable[float],
    threshold: float = 0.7,
) -> float | None:
    pairs = sorted(zip(levels, mean_spearman_values), key=lambda x: x[0])
    for level, value in pairs:
        if np.isfinite(value) and value < threshold:
            return float(level)
    return None


def compute_domain_importance_shift(
    benchmark_importance: dict[str, float],
    current_importance: dict[str, float],
) -> dict[str, float]:
    domains = sorted(set(benchmark_importance) | set(current_importance))
    b_total = sum(float(benchmark_importance.get(d, 0.0)) for d in domains)
    c_total = sum(float(current_importance.get(d, 0.0)) for d in domains)

    out: dict[str, float] = {}
    for d in domains:
        b_share = (float(benchmark_importance.get(d, 0.0)) / b_total) if b_total > 0 else np.nan
        c_share = (float(current_importance.get(d, 0.0)) / c_total) if c_total > 0 else np.nan
        out[f"{d}_benchmark_share"] = b_share
        out[f"{d}_current_share"] = c_share
        out[f"{d}_share_shift"] = c_share - b_share if np.isfinite(b_share) and np.isfinite(c_share) else np.nan
    return out


def domain_rank_stability(
    benchmark_importance: dict[str, float],
    current_importance: dict[str, float],
) -> float:
    domains = sorted(set(benchmark_importance) | set(current_importance))
    b = np.array([float(benchmark_importance.get(d, 0.0)) for d in domains])
    c = np.array([float(current_importance.get(d, 0.0)) for d in domains])
    corr, _ = spearmanr(b, c)
    return float(corr)


def classify_against_benchmark_envelope(
    contaminated_mean: float,
    contaminated_ci_low: float,
    contaminated_ci_high: float,
    benchmark_mean: float,
    benchmark_sd: float,
) -> str:
    env_low = benchmark_mean - 2.0 * benchmark_sd
    env_high = benchmark_mean + 2.0 * benchmark_sd

    if env_low <= contaminated_mean <= env_high:
        return "within_noise"

    overlaps = not (contaminated_ci_high < env_low or contaminated_ci_low > env_high)
    if overlaps:
        return "marginal"
    return "significant"
