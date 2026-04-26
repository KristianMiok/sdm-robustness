from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    from elapid import MaxentModel
except ImportError:  # pragma: no cover
    MaxentModel = None

from sdm_robustness.execution import assign_basin_folds
from sdm_robustness.metrics import (
    compute_performance_metrics,
    spearman_importance_stability,
    topk_jaccard,
    centroid_displacement,
    niche_breadth_change,
    schoeners_d,
    warrens_i,
    range_area_change,
    spatial_mismatch_summary,
    compute_domain_importance_shift,
    domain_rank_stability,
)
from sdm_robustness.metrics.domain_map import aggregate_to_domain_share


@dataclass
class PipelineConfig:
    species: str
    scale_track: str
    kept_predictors: list[str]
    rf_hyperparams: dict[str, Any]
    xgb_hyperparams: dict[str, Any]
    maxent_hyperparams: dict[str, Any] | None
    presence_absence_ratio: float
    cv_type: str
    cv_groups_col: str
    n_clean_benchmark: int


def get_track_columns(df: pd.DataFrame, track: str) -> list[str]:
    if track == "local_only":
        return [c for c in df.columns if c.startswith("l_")]
    if track == "upstream_only":
        return [c for c in df.columns if c.startswith("u_")]
    if track == "combined":
        return [c for c in df.columns if c.startswith("l_") or c.startswith("u_")]
    raise ValueError(f"Unknown track: {track}")


def clean_predictors(
    df: pd.DataFrame,
    predictor_cols: list[str],
    *,
    missing_threshold_pct: float = 30.0,
    correlation_threshold: float = 0.98,
) -> list[str]:
    x = df[predictor_cols].copy()

    keep = x.columns[x.isna().mean() <= (missing_threshold_pct / 100.0)].tolist()
    if not keep:
        raise ValueError("No predictors remain after missingness filtering.")

    x = x[keep].copy()
    x = x.fillna(x.median(numeric_only=True))

    if x.shape[1] <= 1:
        return keep

    corr = x.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for col in upper.columns:
        if any(upper[col] > correlation_threshold):
            to_drop.add(col)

    kept = [c for c in keep if c not in to_drop]
    if not kept:
        raise ValueError("No predictors remain after correlation filtering.")
    return kept


def tune_hyperparameters(
    df: pd.DataFrame,
    predictor_cols: list[str],
    *,
    algorithm: str,
    cv_groups: pd.Series,
    seed: int,
) -> dict[str, Any]:
    if algorithm == "random_forest":
        return {
            "n_estimators": 500,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": -1,
        }
    if algorithm == "xgboost":
        return {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": seed,
            "n_jobs": -1,
        }
    if algorithm == "maxent":
        return {
            "feature_types": ["linear", "hinge", "product"],
            "tau": 0.5,
            "transform": "cloglog",
            "beta_multiplier": 1.5,
            "n_cpus": 1,
            "use_sklearn": True,
            "random_state": seed,
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def generate_pseudo_absences(
    presence_df: pd.DataFrame,
    *,
    ratio: float = 1.0,
    stratify_by_strahler: bool = True,
    accessible_area_buffer_km: float = 50.0,
    seed: int = 0,
) -> pd.DataFrame:
    raise NotImplementedError(
        "Use prepare_accessible_area(...) plus explicit sampling in execution code."
    )


def build_model(
    algorithm: str,
    *,
    seed: int,
    n_jobs: int = -1,
    maxent_n_cpus: int = 1,
):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if algorithm == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=n_jobs,
        )

    if algorithm == "maxent":
        if MaxentModel is None:
            raise ImportError("elapid is not installed, so MaxentModel is unavailable.")
        return MaxentModel(
            feature_types=["linear", "hinge", "product"],
            tau=0.5,
            transform="cloglog",
            beta_multiplier=1.5,
            n_cpus=maxent_n_cpus,
            use_sklearn=True,
            random_state=seed,
        )

    raise ValueError(f"Unknown algorithm: {algorithm}")


def prepare_accessible_area(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    basins = set(benchmark["basin_id"].dropna().astype(str))
    occupied = set(benchmark["subc_id"].dropna().astype(str))

    acc = df[df["basin_id"].astype(str).isin(basins)].copy()
    acc = acc[~acc["subc_id"].astype(str).isin(occupied)].copy()
    acc = acc.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    pred_cols = [c for c in acc.columns if c.startswith("l_") or c.startswith("u_")]
    if pred_cols:
        acc = acc[acc[pred_cols].notna().any(axis=1)].copy()

    return acc


def contaminate_presence_set(
    benchmark: pd.DataFrame,
    contamination_pool: pd.DataFrame,
    level_pct: float,
    seed: int,
) -> pd.DataFrame:
    if level_pct == 0:
        return benchmark.copy()

    n_total = len(benchmark)
    n_replace = int(round(n_total * level_pct / 100.0))
    n_keep = n_total - n_replace

    if n_replace > len(contamination_pool):
        raise ValueError(
            f"Contamination pool too small: need {n_replace}, have {len(contamination_pool)}"
        )

    kept = benchmark.sample(n=n_keep, replace=False, random_state=seed).copy()
    repl = contamination_pool.sample(n=n_replace, replace=False, random_state=seed).copy()
    return pd.concat([kept, repl], ignore_index=True)


def fit_model(
    training_df: pd.DataFrame,
    config: PipelineConfig,
    *,
    algorithm: str,
    seed: int,
) -> dict[str, Any]:
    model = build_model(algorithm, seed=seed)
    x = training_df[config.kept_predictors]
    y = training_df["label"].to_numpy()
    model.fit(x, y)
    return {"model": model, "config": config}


def predict_suitability_raster(
    model: Any,
    network_gdf: Any,
    config: PipelineConfig,
) -> Any:
    raise NotImplementedError("Raster/network projection is not implemented in Phase 1.")


def extract_importance(model, feature_names: list[str], x_eval=None, y_eval=None) -> dict[str, float]:
    """Return {feature_name: importance} for any of the 3 supported algorithms.

    For RF/XGBoost: uses model.feature_importances_ (sklearn API, fast).
    For Maxent (elapid): uses permutation_importance_scores(x_eval, y_eval),
    which requires evaluation data and is significantly slower.
    """
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        return dict(zip(feature_names, importances))

    if hasattr(model, "permutation_importance_scores"):
        if x_eval is None or y_eval is None:
            return {name: float("nan") for name in feature_names}
        try:
            scores_2d = model.permutation_importance_scores(
                x_eval, y_eval, n_repeats=5, n_jobs=-1
            )
            mean_scores = np.asarray(scores_2d, dtype=float).mean(axis=1)
            return dict(zip(feature_names, mean_scores))
        except Exception:
            return {name: float("nan") for name in feature_names}

    return {name: float("nan") for name in feature_names}


def predict_suitability_surface(model, accessible_area_features) -> np.ndarray:
    """Predict probability over rows in accessible_area_features. Returns 1D array."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(accessible_area_features)
        scores = np.asarray(scores)
        if scores.ndim == 2:
            scores = scores[:, -1]
        return scores
    return np.asarray(model.predict(accessible_area_features), dtype=float)


def fit_cv_cell(
    *,
    benchmark: pd.DataFrame,
    contamination_pool: pd.DataFrame,
    accessible_area: pd.DataFrame,
    entity: str,
    algorithm: str,
    track: str,
    axis: str,
    level: int,
    replicate: int,
    seed: int,
    n_splits: int = 5,
    looo_threshold: int = 15,
    maxent_background_n: int = 10_000,
    rf_xgb_pa_ratio: float = 1.0,
    maxent_n_cpus: int = 1,
    n_experiment: int | None = None,
    benchmark_importance: dict[str, float] | None = None,
    benchmark_surface: np.ndarray | None = None,
    domain_map: dict[str, str] | None = None,
    return_artifacts: bool = False,
) -> dict[str, Any]:
    feat_cols = get_track_columns(benchmark, track)
    if not feat_cols:
        raise ValueError(f"No feature columns found for track={track}")

    kept = clean_predictors(benchmark, feat_cols)
    benchmark_x = benchmark[kept].copy()
    medians = benchmark_x.median(numeric_only=True)

    # Subsample benchmark to n_experiment if specified, so the training-set
    # size is held constant across contamination levels and axes.
    if n_experiment is not None and n_experiment < len(benchmark):
        bench_for_contam = benchmark.sample(
            n=n_experiment, replace=False, random_state=seed
        ).copy()
    else:
        bench_for_contam = benchmark

    contaminated_pres = contaminate_presence_set(
        benchmark=bench_for_contam,
        contamination_pool=contamination_pool,
        level_pct=level,
        seed=seed,
    ).copy()

    contaminated_pres = contaminated_pres[["subc_id", "basin_id"] + kept].copy()
    contaminated_pres[kept] = contaminated_pres[kept].fillna(medians)

    acc = accessible_area[["subc_id", "basin_id"] + kept].copy()
    acc[kept] = acc[kept].fillna(medians)

    pres_fold_map = assign_basin_folds(
        contaminated_pres["basin_id"],
        n_splits=n_splits,
        looo_threshold=looo_threshold,
    )
    contaminated_pres["fold"] = contaminated_pres["basin_id"].astype(str).map(pres_fold_map)

    fold_metrics: list[dict[str, float]] = []
    unique_folds = sorted(pd.Series(contaminated_pres["fold"]).dropna().unique().tolist())

    for fold in unique_folds:
        pres_train = contaminated_pres[contaminated_pres["fold"] != fold].copy()
        pres_test = contaminated_pres[contaminated_pres["fold"] == fold].copy()

        if pres_train.empty or pres_test.empty:
            continue

        if algorithm in {"random_forest", "xgboost"}:
            n_neg = int(round(len(contaminated_pres) * rf_xgb_pa_ratio))
            if n_neg > len(acc):
                raise ValueError(
                    f"Accessible area too small for pseudo-absence draw: need {n_neg}, have {len(acc)}"
                )
            neg = acc.sample(n=n_neg, replace=False, random_state=seed + int(fold)).copy()
        elif algorithm == "maxent":
            bg_n = min(maxent_background_n, len(acc))
            neg = acc.sample(n=bg_n, replace=False, random_state=seed + int(fold)).copy()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        neg["fold"] = neg["basin_id"].astype(str).map(pres_fold_map).fillna(0).astype(int)

        neg_train = neg[neg["fold"] != fold].copy()
        neg_test = neg[neg["fold"] == fold].copy()

        x_train = pd.concat([pres_train[kept], neg_train[kept]], axis=0)
        y_train = np.array([1] * len(pres_train) + [0] * len(neg_train))

        x_test = pd.concat([pres_test[kept], neg_test[kept]], axis=0)
        y_test = np.array([1] * len(pres_test) + [0] * len(neg_test))

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = build_model(
            algorithm,
            seed=seed,
            n_jobs=-1,
            maxent_n_cpus=maxent_n_cpus,
        )
        model.fit(x_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)
            y_score = np.asarray(y_score)
            if y_score.ndim == 2:
                y_score = y_score[:, -1]
        else:
            pred = model.predict(x_test)
            y_score = np.asarray(pred, dtype=float)

        perf = compute_performance_metrics(y_test, y_score, threshold=0.5)
        fold_metrics.append(perf)

    if not fold_metrics:
        return {
            "entity": entity,
            "algorithm": algorithm,
            "track": track,
            "axis": axis,
            "level": level,
            "replicate": replicate,
            "seed": seed,
            "status": "no_valid_folds",
            "n_folds_completed": 0,
            "n_features": int(len(kept)),
            "benchmark_presence_n": int(len(contaminated_pres)),
            "contrast_pool_n": int(len(acc)),
        }

    agg: dict[str, float] = {}
    for key in fold_metrics[0].keys():
        vals = [m[key] for m in fold_metrics]
        agg[key] = float(np.nanmean(vals))

    # === Tier 2/3 metrics (only computed if benchmark artifacts provided) ===
    tier23: dict[str, Any] = {}

    # Fast path: legacy callers (no benchmark artifacts, no return_artifacts)
    # skip the extra fit + prediction entirely.
    needs_tier23 = (
        benchmark_importance is not None
        or benchmark_surface is not None
        or return_artifacts
    )

    if not needs_tier23:
        return {
            "entity": entity,
            "algorithm": algorithm,
            "track": track,
            "axis": axis,
            "level": level,
            "replicate": replicate,
            "seed": seed,
            "status": "ok",
            "n_folds_completed": len(fold_metrics),
            "n_features": int(len(kept)),
            "benchmark_presence_n": int(len(contaminated_pres)),
            "contrast_pool_n": int(len(acc)),
            "accessible_area_segment_count": int(len(acc)),
            **agg,
        }

    # Refit on FULL contaminated set to get this run's importance + surface
    if algorithm == "maxent":
        bg_n_final = min(maxent_background_n, len(acc))
    else:
        bg_n_final = min(int(round(len(contaminated_pres) * rf_xgb_pa_ratio)), len(acc))

    final_neg = acc.sample(n=bg_n_final, replace=False, random_state=seed)
    final_x = pd.concat([contaminated_pres[kept], final_neg[kept]], axis=0)
    n_pres = len(contaminated_pres)
    n_neg = len(final_x) - n_pres
    final_y = np.array([1] * n_pres + [0] * n_neg)
    final_model = build_model(algorithm, seed=seed, n_jobs=-1, maxent_n_cpus=maxent_n_cpus)
    try:
        final_model.fit(final_x, final_y)
        run_importance = extract_importance(final_model, kept, x_eval=final_x, y_eval=final_y)
        run_surface = predict_suitability_surface(final_model, acc[kept])
    except Exception:
        run_importance = {name: float("nan") for name in kept}
        run_surface = np.array([])

    if benchmark_importance is not None:
        b_imp = np.array([benchmark_importance.get(k, np.nan) for k in kept])
        c_imp = np.array([run_importance.get(k, np.nan) for k in kept])
        try:
            tier23["importance_spearman"] = spearman_importance_stability(b_imp, c_imp)
        except Exception:
            tier23["importance_spearman"] = float("nan")
        try:
            tier23["importance_jaccard_top5"] = topk_jaccard(b_imp, c_imp, k=5)
            tier23["importance_jaccard_top10"] = topk_jaccard(b_imp, c_imp, k=10)
        except Exception:
            tier23["importance_jaccard_top5"] = float("nan")
            tier23["importance_jaccard_top10"] = float("nan")

        try:
            bench_pres_x = benchmark[kept].fillna(medians).values
            cont_pres_x = contaminated_pres[kept].values
            tier23["niche_centroid_disp"] = centroid_displacement(bench_pres_x, cont_pres_x)
            tier23["niche_breadth_change"] = niche_breadth_change(bench_pres_x, cont_pres_x)
        except Exception:
            tier23["niche_centroid_disp"] = float("nan")
            tier23["niche_breadth_change"] = float("nan")

        if domain_map is not None:
            try:
                bench_share = aggregate_to_domain_share(benchmark_importance, domain_map)
                cont_share = aggregate_to_domain_share(run_importance, domain_map)
                b_dom = {k.replace("_share", ""): v for k, v in bench_share.items()}
                c_dom = {k.replace("_share", ""): v for k, v in cont_share.items()}
                shift = compute_domain_importance_shift(b_dom, c_dom)
                for k, v in shift.items():
                    if k.endswith("_share_shift"):
                        tier23[k.replace("_share_shift", "_shift")] = v
                tier23["domain_rank_stability"] = domain_rank_stability(b_dom, c_dom)
            except Exception:
                for d in ("CLI", "TOP", "SOL", "LAC"):
                    tier23[f"{d}_shift"] = float("nan")
                tier23["domain_rank_stability"] = float("nan")

    if benchmark_surface is not None and len(run_surface) > 0:
        try:
            tier23["schoener_d"] = schoeners_d(benchmark_surface, run_surface)
            tier23["warren_i"] = warrens_i(benchmark_surface, run_surface)
        except Exception:
            tier23["schoener_d"] = float("nan")
            tier23["warren_i"] = float("nan")
        try:
            tier23["range_area_pct_change_05"] = range_area_change(
                benchmark_surface, run_surface, threshold=0.5
            )
            mismatch = spatial_mismatch_summary(benchmark_surface, run_surface, threshold=0.5)
            tier23["gain_segments"] = int(mismatch.get("gain_cells", 0))
            tier23["loss_segments"] = int(mismatch.get("loss_cells", 0))
            tier23["stable_segments"] = int(mismatch.get("stable_presence_cells", 0))
        except Exception:
            tier23["range_area_pct_change_05"] = float("nan")
            tier23["gain_segments"] = 0
            tier23["loss_segments"] = 0
            tier23["stable_segments"] = 0

    result = {
        "entity": entity,
        "algorithm": algorithm,
        "track": track,
        "axis": axis,
        "level": level,
        "replicate": replicate,
        "seed": seed,
        "status": "ok",
        "n_folds_completed": len(fold_metrics),
        "n_features": int(len(kept)),
        "benchmark_presence_n": int(len(contaminated_pres)),
        "contrast_pool_n": int(len(acc)),
        "accessible_area_segment_count": int(len(acc)),
        **agg,
        **tier23,
    }

    if return_artifacts:
        result["_run_importance"] = run_importance
        result["_run_surface"] = run_surface

    return result
