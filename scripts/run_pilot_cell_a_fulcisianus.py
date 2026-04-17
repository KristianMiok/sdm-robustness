#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sdm_robustness.execution import assign_basin_folds
from sdm_robustness.io import load_master_table
from sdm_robustness.metrics import compute_performance_metrics, MetricRecord
from sdm_robustness.utils import setup_logging, logger, get_git_commit, get_git_dirty, project_root

DATA_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
OUT_ROOT = Path("/Users/kristianmiok/Desktop/sdm-robustness/results/pilot_cell")


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, level="INFO")

    logger.info(f"Pilot cell starting. Output → {out_dir}")
    commit = get_git_commit(project_root())
    dirty = get_git_dirty(project_root())
    logger.info(f"Git commit: {commit}{' (dirty)' if dirty else ''}")

    t0 = time.time()

    df, _ = load_master_table(Path(DATA_PATH))
    species = "Austropotamobius fulcisianus"

    # benchmark presences
    pres = df[
        (df["Crayfish_scientific_name"] == species)
        & (df["Accuracy"] == "High")
        & (df["distance_m"] <= 200)
    ].copy()
    pres = pres.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    if pres.empty:
        raise RuntimeError("No benchmark presences found.")

    # local track only
    feature_cols = [c for c in pres.columns if c.startswith("l_")]
    Xp = pres[feature_cols].copy()

    # simple cleaning for pilot cell only:
    # keep columns with <=30% missing, median-impute remaining
    keep = Xp.columns[Xp.isna().mean() <= 0.30]
    Xp = Xp[keep].copy()
    Xp = Xp.fillna(Xp.median(numeric_only=True))

    pres = pres.loc[Xp.index].copy()

    # accessible area from same basins, excluding occupied segments
    basins = set(pres["basin_id"].dropna().astype(str))
    occupied = set(pres["subc_id"].dropna().astype(str))

    acc = df[df["basin_id"].astype(str).isin(basins)].copy()
    acc = acc[~acc["subc_id"].astype(str).isin(occupied)].copy()
    acc = acc.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    Xa = acc[keep].copy()
    Xa = Xa.fillna(Xa.median(numeric_only=True))

    # 1:1 pseudo-absence sample
    neg = acc.sample(n=len(pres), replace=False, random_state=42).copy()
    Xn = Xa.loc[neg.index].copy()

    # combine
    X = pd.concat([Xp, Xn], axis=0)
    y = np.array([1] * len(Xp) + [0] * len(Xn))

    # deterministic fold assignment using presences only
    fold_map = assign_basin_folds(pres["basin_id"], n_splits=5, looo_threshold=15)
    pres_folds = pres["basin_id"].astype(str).map(fold_map)

    # assign negatives to folds by their basin when possible, else 0
    neg_folds = neg["basin_id"].astype(str).map(fold_map).fillna(0).astype(int)

    folds = np.concatenate([pres_folds.to_numpy(), neg_folds.to_numpy()])

    # take fold 0 as test, rest train
    test_mask = folds == 0
    train_mask = ~test_mask

    X_train = X.iloc[train_mask]
    y_train = y[train_mask]
    X_test = X.iloc[test_mask]
    y_test = y[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise RuntimeError("Pilot cell produced degenerate train/test split.")

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]

    perf = compute_performance_metrics(y_test, y_score, threshold=0.5)

    records = []
    for metric_name, metric_value in perf.items():
        tier = "tier1"
        rec = MetricRecord(
            entity="Austropotamobius fulcisianus (pooled)",
            axis="snap",
            contamination_level=0.0,
            replicate=1,
            algorithm="rf",
            spatial_scale="local",
            seed=42,
            benchmark=True,
            metric_name=metric_name,
            metric_value=float(metric_value),
            metric_tier=tier,
        )
        records.append(rec.to_dict())

    pd.DataFrame(records).to_csv(out_dir / "metric_records.csv", index=False)

    summary = {
        "entity": "Austropotamobius fulcisianus (pooled)",
        "algorithm": "rf",
        "spatial_scale": "local",
        "axis": "snap",
        "contamination_level": 0.0,
        "replicate": 1,
        "benchmark_presence_n": int(len(pres)),
        "pseudoabsence_n": int(len(neg)),
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "metrics": perf,
        "runtime_seconds": round(time.time() - t0, 3),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    note = [
        "# Pilot cell summary",
        "",
        f"- entity: **{summary['entity']}**",
        f"- algorithm: **{summary['algorithm']}**",
        f"- spatial scale: **{summary['spatial_scale']}**",
        f"- benchmark presence n: **{summary['benchmark_presence_n']}**",
        f"- pseudo-absence n: **{summary['pseudoabsence_n']}**",
        f"- train n: **{summary['train_n']}**",
        f"- test n: **{summary['test_n']}**",
        f"- features: **{summary['n_features']}**",
        f"- runtime seconds: **{summary['runtime_seconds']}**",
        "",
        "## Metrics",
    ]
    for k, v in perf.items():
        note.append(f"- {k}: **{v:.4f}**")
    (out_dir / "note.md").write_text("\n".join(note), encoding="utf-8")

    logger.info(f"  → {out_dir / 'metric_records.csv'}")
    logger.info(f"  → {out_dir / 'summary.json'}")
    logger.info(f"  → {out_dir / 'note.md'}")
    logger.info("Pilot cell complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
