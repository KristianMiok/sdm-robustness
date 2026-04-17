#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from elapid import MaxentModel

from sdm_robustness.execution import assign_basin_folds
from sdm_robustness.io import load_master_table
from sdm_robustness.metrics import compute_performance_metrics
from sdm_robustness.utils import setup_logging, logger, get_git_commit, get_git_dirty, project_root

DATA_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
OUT_ROOT = Path("/Users/kristianmiok/Desktop/sdm-robustness/results/pilot_cv")


def get_track_columns(df: pd.DataFrame, track: str) -> list[str]:
    if track == "local":
        return [c for c in df.columns if c.startswith("l_")]
    if track == "upstream":
        return [c for c in df.columns if c.startswith("u_")]
    if track == "combined":
        return [c for c in df.columns if c.startswith("l_") or c.startswith("u_")]
    raise ValueError(f"Unknown track: {track}")


def build_model(algorithm: str):
    if algorithm == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
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
            random_state=42,
            n_jobs=-1,
        )
    if algorithm == "maxent":
        return MaxentModel(
            feature_types=["linear", "hinge", "product"],
            tau=0.5,
            transform="cloglog",
            beta_multiplier=1.5,
            n_cpus=14,
            use_sklearn=True,
            random_state=42,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


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


def prepare_accessible_area(df: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    basins = set(benchmark["basin_id"].dropna().astype(str))
    occupied = set(benchmark["subc_id"].dropna().astype(str))

    acc = df[df["basin_id"].astype(str).isin(basins)].copy()
    acc = acc[~acc["subc_id"].astype(str).isin(occupied)].copy()
    acc = acc.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    pred_cols = [c for c in acc.columns if c.startswith("l_") or c.startswith("u_")]
    acc = acc[acc[pred_cols].notna().any(axis=1)].copy()
    return acc


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, level="INFO")

    logger.info(f"Pilot CV starting. Output → {out_dir}")
    commit = get_git_commit(project_root())
    dirty = get_git_dirty(project_root())
    logger.info(f"Git commit: {commit}{' (dirty)' if dirty else ''}")

    with open("config/pilot_a_fulcisianus.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df, _ = load_master_table(Path(DATA_PATH))
    species = "Austropotamobius fulcisianus"
    base = df[df["Crayfish_scientific_name"] == species].copy()

    benchmark = base[
        (base["Accuracy"] == "High")
        & (base["distance_m"] <= 200)
    ].copy()
    benchmark = benchmark.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    snap_pool = base[
        (base["Accuracy"] == "High")
        & (base["distance_m"] > 200)
        & (base["distance_m"] <= 1000)
    ].copy()
    snap_pool = snap_pool.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    lowacc_pool = base[
        (base["Accuracy"] != "High")
    ].copy()
    lowacc_pool = lowacc_pool.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    accessible = prepare_accessible_area(df, benchmark)

    # smaller but CV-aware pilot grid
    cells = [
        ("rf", "local", "snap", 0),
        ("rf", "local", "snap", 5),
        ("rf", "local", "lowacc", 20),
        ("xgboost", "local", "snap", 0),
        ("xgboost", "local", "snap", 5),
        ("maxent", "local", "snap", 0),
        ("maxent", "local", "snap", 5),
    ]

    rows = []

    for algorithm, track, axis, level in cells:
        t0 = time.time()

        feat_cols = get_track_columns(benchmark, track)
        Xb = benchmark[feat_cols].copy()
        keep = Xb.columns[Xb.isna().mean() <= 0.30]
        Xb = Xb[keep].copy()
        medians = Xb.median(numeric_only=True)

        pool = snap_pool if axis == "snap" else lowacc_pool
        contaminated_pres = contaminate_presence_set(
            benchmark=benchmark,
            contamination_pool=pool,
            level_pct=level,
            seed=42,
        ).copy()

        contaminated_pres = contaminated_pres[["subc_id", "basin_id"] + list(keep)].copy()
        contaminated_pres[list(keep)] = contaminated_pres[list(keep)].fillna(medians)

        pres_fold_map = assign_basin_folds(
            contaminated_pres["basin_id"],
            n_splits=cfg["cv"]["n_splits"],
            looo_threshold=cfg["cv"]["looo_threshold"],
        )
        contaminated_pres["fold"] = contaminated_pres["basin_id"].astype(str).map(pres_fold_map)

        acc = accessible[["subc_id", "basin_id"] + list(keep)].copy()
        acc[list(keep)] = acc[list(keep)].fillna(medians)

        fold_metrics = []
        fold_runtimes = []

        unique_folds = sorted(pd.Series(contaminated_pres["fold"]).dropna().unique().tolist())

        for fold in unique_folds:
            f0 = time.time()

            pres_train = contaminated_pres[contaminated_pres["fold"] != fold].copy()
            pres_test = contaminated_pres[contaminated_pres["fold"] == fold].copy()

            if pres_test.empty or pres_train.empty:
                continue

            if algorithm in {"rf", "xgboost"}:
                neg = acc.sample(n=len(contaminated_pres), replace=False, random_state=42 + int(fold)).copy()
            else:
                bg_n = min(cfg["maxent_background_n"], len(acc))
                neg = acc.sample(n=bg_n, replace=False, random_state=42 + int(fold)).copy()

            neg["fold"] = neg["basin_id"].astype(str).map(pres_fold_map).fillna(0).astype(int)

            neg_train = neg[neg["fold"] != fold].copy()
            neg_test = neg[neg["fold"] == fold].copy()

            X_train = pd.concat([pres_train[list(keep)], neg_train[list(keep)]], axis=0)
            y_train = np.array([1] * len(pres_train) + [0] * len(neg_train))

            X_test = pd.concat([pres_test[list(keep)], neg_test[list(keep)]], axis=0)
            y_test = np.array([1] * len(pres_test) + [0] * len(neg_test))

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            model = build_model(algorithm)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)
                    y_score = np.asarray(y_score)
                    if y_score.ndim == 2:
                        y_score = y_score[:, -1]
                else:
                    y_score = model.predict(X_test)

            perf = compute_performance_metrics(y_test, y_score, threshold=0.5)
            fold_metrics.append(perf)
            fold_runtimes.append(time.time() - f0)

        if not fold_metrics:
            rows.append({
                "algorithm": algorithm,
                "track": track,
                "axis": axis,
                "level": level,
                "status": "no_valid_folds",
                "runtime_seconds": round(time.time() - t0, 3),
            })
            continue

        agg = {}
        for k in fold_metrics[0].keys():
            vals = [m[k] for m in fold_metrics]
            agg[k] = float(np.nanmean(vals))

        rows.append({
            "algorithm": algorithm,
            "track": track,
            "axis": axis,
            "level": level,
            "status": "ok",
            "n_folds_completed": len(fold_metrics),
            "n_features": int(len(keep)),
            "benchmark_presence_n": int(len(contaminated_pres)),
            "contrast_pool_n": int(len(acc)),
            "runtime_seconds": round(time.time() - t0, 3),
            "mean_fold_runtime_seconds": round(float(np.mean(fold_runtimes)), 3),
            **agg,
        })

        logger.info(
            f"Done: alg={algorithm}, track={track}, axis={axis}, level={level}, "
            f"folds={len(fold_metrics)}, runtime={rows[-1]['runtime_seconds']}s"
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pilot_cv_results.csv", index=False)

    summary = {
        "n_cells": int(len(out)),
        "n_ok": int((out["status"] == "ok").sum()),
        "total_runtime_seconds": float(out["runtime_seconds"].sum()),
        "mean_runtime_seconds_ok": float(out.loc[out["status"] == "ok", "runtime_seconds"].mean()),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note = []
    note.append("# Pilot CV summary — A. fulcisianus")
    note.append("")
    note.append(f"- cells run: **{summary['n_cells']}**")
    note.append(f"- successful cells: **{summary['n_ok']}**")
    note.append(f"- mean runtime per successful CV cell: **{summary['mean_runtime_seconds_ok']:.3f} s**")
    note.append(f"- total runtime: **{summary['total_runtime_seconds']:.3f} s**")
    note.append("")
    note.append("## Mean runtime by algorithm")
    rt = out.loc[out["status"] == "ok"].groupby("algorithm")["runtime_seconds"].mean()
    for alg, val in rt.items():
        note.append(f"- {alg}: **{val:.3f} s**")
    (out_dir / "note.md").write_text("\n".join(note), encoding="utf-8")

    logger.info(f"  → {out_dir / 'pilot_cv_results.csv'}")
    logger.info(f"  → {out_dir / 'summary.json'}")
    logger.info(f"  → {out_dir / 'note.md'}")
    logger.info("Pilot CV complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
