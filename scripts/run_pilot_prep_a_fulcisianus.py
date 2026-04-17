#!/usr/bin/env python
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from sdm_robustness.execution import (
    get_panel_entity,
    assign_basin_folds,
    sample_rf_xgb_pseudoabsences,
    sample_maxent_background,
)
from sdm_robustness.io import load_master_table
from sdm_robustness.utils import (
    get_git_commit,
    get_git_dirty,
    project_root,
    setup_logging,
    logger,
)

DATA_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
OUT_ROOT = Path("/Users/kristianmiok/Desktop/sdm-robustness/results/pilot_prep")


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, level="INFO")

    commit = get_git_commit(project_root())
    dirty = get_git_dirty(project_root())
    logger.info(f"Pilot prep starting. Output → {out_dir}")
    logger.info(f"Git commit: {commit}{' (dirty)' if dirty else ''}")

    with open("config/pilot_a_fulcisianus.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    entity_name = cfg["entity"]
    panel_row = get_panel_entity(entity_name)
    logger.info(f"Loaded panel entity: {entity_name}")

    df, info = load_master_table(Path(DATA_PATH))

    species = "Austropotamobius fulcisianus"

    # pooled nat+alien treatment
    df_sp = df[df["Crayfish_scientific_name"] == species].copy()

    # benchmark presence set
    benchmark = df_sp[
        (df_sp["Accuracy"] == "High")
        & (df_sp["distance_m"] <= 200)
    ].copy()

    benchmark = benchmark.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    if benchmark.empty:
        raise RuntimeError("Benchmark presence set is empty.")

    # deterministic basin folds
    fold_map = assign_basin_folds(
        benchmark["basin_id"],
        n_splits=cfg["cv"]["n_splits"],
        looo_threshold=cfg["cv"]["looo_threshold"],
    )
    benchmark["fold"] = benchmark["basin_id"].astype(str).map(fold_map)

    # accessible area: all segments in benchmark basins, excluding occupied benchmark segments
    benchmark_basins = set(benchmark["basin_id"].dropna().astype(str))
    occupied_subc = set(benchmark["subc_id"].dropna().astype(str))

    accessible = df[
        df["basin_id"].astype(str).isin(benchmark_basins)
    ].copy()

    accessible = accessible[~accessible["subc_id"].astype(str).isin(occupied_subc)].copy()

    # keep only rows with usable predictors for now: at least one local/upstream predictor present
    pred_cols = [c for c in accessible.columns if c.startswith("l_") or c.startswith("u_")]
    accessible = accessible[accessible[pred_cols].notna().any(axis=1)].copy()

    # deduplicate accessible area by segment
    accessible = accessible.sort_values("subc_id").drop_duplicates(subset=["subc_id"]).copy()

    rf_xgb_pa = sample_rf_xgb_pseudoabsences(
        accessible,
        benchmark_presence_n=len(benchmark),
        ratio=cfg["rf_xgb_pa_ratio"],
        seed=42,
    )

    maxent_bg = sample_maxent_background(
        accessible,
        n_background=cfg["maxent_background_n"],
        seed=42,
    )

    benchmark.to_csv(out_dir / "benchmark_presence.csv", index=False)
    accessible.to_csv(out_dir / "accessible_area.csv", index=False)
    rf_xgb_pa.to_csv(out_dir / "rf_xgb_pseudoabsences.csv", index=False)
    maxent_bg.to_csv(out_dir / "maxent_background.csv", index=False)

    fold_counts = benchmark.groupby("fold").size().to_dict()

    summary = {
        "entity": entity_name,
        "panel_type": panel_row["type"],
        "category": panel_row["category"],
        "benchmark_n": int(len(benchmark)),
        "n_unique_basins": int(benchmark["basin_id"].astype(str).nunique()),
        "fold_counts": fold_counts,
        "accessible_area_segments": int(len(accessible)),
        "rf_xgb_pseudoabsence_n": int(len(rf_xgb_pa)),
        "maxent_background_n": int(len(maxent_bg)),
        "algorithms": cfg["algorithms"],
        "spatial_tracks": cfg["spatial_tracks"],
        "snapping_levels": cfg["snapping_levels"],
        "lowacc_levels": cfg["lowacc_levels"],
        "n_replicates": int(cfg["n_replicates"]),
    }

    with open(out_dir / "pilot_prep_summary.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    note = []
    note.append("# Pilot prep summary — A. fulcisianus")
    note.append("")
    note.append(f"- entity: **{entity_name}**")
    note.append(f"- benchmark_n: **{len(benchmark)}**")
    note.append(f"- unique basins: **{benchmark['basin_id'].astype(str).nunique()}**")
    note.append(f"- accessible area segments: **{len(accessible)}**")
    note.append(f"- RF/XGBoost pseudo-absence sample: **{len(rf_xgb_pa)}**")
    note.append(f"- Maxent background sample: **{len(maxent_bg)}**")
    note.append(f"- fold counts: **{fold_counts}**")
    (out_dir / "pilot_prep_note.md").write_text("\n".join(note), encoding="utf-8")

    logger.info(f"  → {out_dir / 'pilot_prep_summary.yaml'}")
    logger.info(f"  → {out_dir / 'pilot_prep_note.md'}")
    logger.info("Pilot prep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
