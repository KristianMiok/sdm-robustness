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
from sdm_robustness.metrics import compute_performance_metrics


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
) -> dict[str, Any]:
    feat_cols = get_track_columns(benchmark, track)
    if not feat_cols:
        raise ValueError(f"No feature columns found for track={track}")

    kept = clean_predictors(benchmark, feat_cols)
    benchmark_x = benchmark[kept].copy()
    medians = benchmark_x.median(numeric_only=True)

    contaminated_pres = contaminate_presence_set(
        benchmark=benchmark,
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
        **agg,
    }
