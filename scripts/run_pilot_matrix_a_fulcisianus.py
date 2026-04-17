#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from elapid import MaxentModel

from sdm_robustness.execution import assign_basin_folds
from sdm_robustness.io import load_master_table
from sdm_robustness.metrics import compute_performance_metrics
from sdm_robustness.utils import setup_logging, logger, get_git_commit, get_git_dirty, project_root

DATA_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
OUT_ROOT = Path("/Users/kristianmiok/Desktop/sdm-robustness/results/pilot_matrix")


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
    out = pd.concat([kept, repl], ignore_index=True)
    return out


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

    logger.info(f"Pilot matrix starting. Output → {out_dir}")
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

    algorithms = cfg["algorithms"]
    tracks = cfg["spatial_tracks"]

    smoke_cells = [
        ("snap", 0),
        ("snap", 5),
        ("lowacc", 0),
        ("lowacc", 20),
    ]

    rows = []

    for algorithm in algorithms:
        for track in tracks:
            feat_cols = get_track_columns(benchmark, track)

            Xb = benchmark[feat_cols].copy()
            keep = Xb.columns[Xb.isna().mean() <= 0.30]
            Xb = Xb[keep].copy()
            medians = Xb.median(numeric_only=True)
            Xb = Xb.fillna(medians)

            Xa = accessible[keep].copy().fillna(medians)

            snap_aligned = snap_pool.copy()
            if not snap_aligned.empty:
                snap_aligned = snap_aligned.copy()
                snap_aligned = snap_aligned.reindex(columns=benchmark.columns, fill_value=np.nan)

            lowacc_aligned = lowacc_pool.copy()
            if not lowacc_aligned.empty:
                lowacc_aligned = lowacc_aligned.copy()
                lowacc_aligned = lowacc_aligned.reindex(columns=benchmark.columns, fill_value=np.nan)

            for axis, level in smoke_cells:
                if axis == "lowacc" and algorithm == "maxent":
                    # still valid; background is different, presence contamination remains same logic
                    pass

                t0 = time.time()

                pool = snap_aligned if axis == "snap" else lowacc_aligned
                contaminated_pres = contaminate_presence_set(
                    benchmark=benchmark,
                    contamination_pool=pool,
                    level_pct=level,
                    seed=42,
                )

                Xp = contaminated_pres[keep].copy().fillna(medians)

                if algorithm in {"rf", "xgboost"}:
                    neg = accessible.sample(
                        n=len(contaminated_pres),
                        replace=False,
                        random_state=42,
                    ).copy()
                else:
                    bg_n = min(cfg["maxent_background_n"], len(accessible))
                    neg = accessible.sample(
                        n=bg_n,
                        replace=False,
                        random_state=42,
                    ).copy()

                Xn = neg[keep].copy().fillna(medians)

                X = pd.concat([Xp, Xn], axis=0)
                y = np.array([1] * len(Xp) + [0] * len(Xn))

                pres_fold_map = assign_basin_folds(
                    contaminated_pres["basin_id"],
                    n_splits=cfg["cv"]["n_splits"],
                    looo_threshold=cfg["cv"]["looo_threshold"],
                )
                pres_folds = contaminated_pres["basin_id"].astype(str).map(pres_fold_map)
                neg_folds = neg["basin_id"].astype(str).map(pres_fold_map).fillna(0).astype(int)
                folds = np.concatenate([pres_folds.to_numpy(), neg_folds.to_numpy()])

                test_mask = folds == 0
                train_mask = ~test_mask

                X_train = X.iloc[train_mask]
                y_train = y[train_mask]
                X_test = X.iloc[test_mask]
                y_test = y[test_mask]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    row = {
                        "algorithm": algorithm,
                        "track": track,
                        "axis": axis,
                        "level": level,
                        "status": "degenerate_split",
                        "runtime_seconds": round(time.time() - t0, 3),
                    }
                    rows.append(row)
                    continue

                model = build_model(algorithm)
                model.fit(X_train, y_train)

                if hasattr(model, "predict_proba"):
                    y_score = model.predict_proba(X_test)[:, 1]
                else:
                    y_score = model.decision_function(X_test)

                perf = compute_performance_metrics(y_test, y_score, threshold=0.5)

                row = {
                    "algorithm": algorithm,
                    "track": track,
                    "axis": axis,
                    "level": level,
                    "status": "ok",
                    "benchmark_presence_n": int(len(contaminated_pres)),
                    "contrast_n": int(len(neg)),
                    "train_n": int(len(X_train)),
                    "test_n": int(len(X_test)),
                    "n_features": int(len(keep)),
                    "runtime_seconds": round(time.time() - t0, 3),
                    **perf,
                }
                rows.append(row)
                logger.info(
                    f"Done: alg={algorithm}, track={track}, axis={axis}, level={level}, "
                    f"runtime={row['runtime_seconds']}s"
                )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "pilot_matrix_results.csv", index=False)

    summary = {
        "n_cells": int(len(out)),
        "n_ok": int((out["status"] == "ok").sum()),
        "mean_runtime_seconds_ok": float(out.loc[out["status"] == "ok", "runtime_seconds"].mean()),
        "total_runtime_seconds": float(out["runtime_seconds"].sum()),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note = []
    note.append("# Pilot matrix summary — A. fulcisianus")
    note.append("")
    note.append(f"- cells run: **{summary['n_cells']}**")
    note.append(f"- successful cells: **{summary['n_ok']}**")
    note.append(f"- mean runtime per successful cell: **{summary['mean_runtime_seconds_ok']:.3f} s**")
    note.append(f"- total runtime: **{summary['total_runtime_seconds']:.3f} s**")
    note.append("")
    note.append("## Mean runtime by algorithm")
    rt = out.loc[out["status"] == "ok"].groupby("algorithm")["runtime_seconds"].mean()
    for alg, val in rt.items():
        note.append(f"- {alg}: **{val:.3f} s**")
    (out_dir / "note.md").write_text("\n".join(note), encoding="utf-8")

    logger.info(f"  → {out_dir / 'pilot_matrix_results.csv'}")
    logger.info(f"  → {out_dir / 'summary.json'}")
    logger.info(f"  → {out_dir / 'note.md'}")
    logger.info("Pilot matrix complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
