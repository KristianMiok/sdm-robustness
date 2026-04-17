#!/usr/bin/env python
"""Task 1b — native/alien split audit for invasive species.

Outputs:
- invasive_split_audit.csv
- invasive_split_feasibility.csv
- invasive_split_note.md
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sdm_robustness.audit import build_inventory, classify_candidates, compute_feasibility
from sdm_robustness.io import load_master_table
from sdm_robustness.utils import (
    config_hash,
    get_git_commit,
    get_git_dirty,
    load_frozen_design,
    load_task1_gates,
    logger,
    project_root,
    resolve_path,
    setup_logging,
)

DATA_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/combined_data_true_master.csv"
CATEGORY_PATH = "/Users/kristianmiok/Desktop/Lucian/Global/Descriptive Paper/Data/kristian_outputs/distribution_categories_by_species_corrected.csv"
OUT_ROOT = "/Users/kristianmiok/Desktop/sdm-robustness/results/task1b_invasive_split"

INVASIVE_SPECIES = [
    "Procambarus clarkii",
    "Pacifastacus leniusculus",
    "Faxonius limosus",
    "Cherax quadricarinatus",
]

NATIVE_STATUSES = {"Native"}
ALIEN_STATUSES = {"Alien", "Introduced", "Non-indigenous", "Non indigenous", "Non_indigenous"}


def _load_categories(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "category_corrected" in df.columns:
        df["category"] = df["category_corrected"]
    df = df[["species", "category"]].copy()
    return df


def _subset_status(df: pd.DataFrame, which: str) -> pd.DataFrame:
    if which == "native":
        return df[df["Status"].isin(NATIVE_STATUSES)].copy()
    if which == "alien":
        return df[df["Status"].isin(ALIEN_STATUSES)].copy()
    raise ValueError(which)


def _entity_label(species: str, which: str) -> str:
    return f"{species} ({which})"


def _make_inventory_for_entity(
    df_all: pd.DataFrame,
    species: str,
    which: str,
    strict_m: int,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    df_sp = df_all[df_all["Crayfish_scientific_name"] == species].copy()
    df_sp = _subset_status(df_sp, which)

    inv = build_inventory(
        df_sp,
        strict_m=strict_m,
        intermediate_m=500,
        maximum_m=1000,
        dedup_key="subc_id",
        petko_categories=categories,
    ).copy()

    if inv.empty:
        return pd.DataFrame()

    inv["base_species"] = species
    inv["range_status"] = which
    inv["entity"] = _entity_label(species, which)

    # make entity the unique analysis unit
    inv["species"] = inv["entity"]

    # native = use corrected ecological category from file
    # alien = force widespread threshold as Lucian requested
    if which == "alien":
        inv["category_petko2026"] = "widespread"

    return inv


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUT_ROOT) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(out_dir, level="INFO")
    logger.info(f"Task 1b invasive split audit starting. Output → {out_dir}")

    commit = get_git_commit(project_root())
    dirty = get_git_dirty(project_root())
    if commit:
        logger.info(f"Git commit: {commit}{' (dirty)' if dirty else ''}")

    frozen = load_frozen_design()
    gates = load_task1_gates()
    fhash = config_hash({"frozen": frozen, "gates": gates})
    logger.info(f"Config hash: {fhash}")

    logger.info(f"Loading master table from {DATA_PATH}")
    df, info = load_master_table(Path(DATA_PATH))

    categories = _load_categories(CATEGORY_PATH)
    strict_m = frozen["benchmark"]["snapping_threshold_m"]

    inventories = []
    for species in INVASIVE_SPECIES:
        for which in ("native", "alien"):
            inv = _make_inventory_for_entity(
                df_all=df,
                species=species,
                which=which,
                strict_m=strict_m,
                categories=categories,
            )
            if inv.empty:
                logger.warning(f"No records for {species} ({which}) after status filtering.")
                continue
            inventories.append(inv)

    if not inventories:
        raise RuntimeError("No Task 1b entities were created.")

    inv_all = pd.concat(inventories, ignore_index=True)

    fea = compute_feasibility(inv_all)
    cls = classify_candidates(inv_all, fea, gates)

    # merge for one-row audit table
    audit = inv_all.merge(
        fea.drop(columns=["n_experiment_assumed"], errors="ignore"),
        on="species",
        how="left",
        suffixes=("", "_fea"),
    ).merge(
        cls.drop(columns=["n_clean_dedup_200m", "n_snap_pool", "n_lowacc_pool", "n_basins", "strahler_min", "strahler_max"], errors="ignore"),
        on="species",
        how="left",
        suffixes=("", "_cls"),
    )

    # tidy leading columns
    lead = [
        "species",
        "entity",
        "base_species",
        "range_status",
        "status",
        "category_petko2026",
        "classification",
    ]
    other = [c for c in audit.columns if c not in lead]
    audit = audit[lead + other]

    audit_path = out_dir / "invasive_split_audit.csv"
    fea_path = out_dir / "invasive_split_feasibility.csv"
    note_path = out_dir / "invasive_split_note.md"

    audit.to_csv(audit_path, index=False)
    fea.to_csv(fea_path, index=False)

    vc = cls["classification"].value_counts()
    dual = int(vc.get("DUAL-AXIS", 0))
    snap = int(vc.get("SNAPPING-ONLY", 0))
    low = int(vc.get("LOW-ACC-ONLY", 0))
    ine = int(vc.get("INELIGIBLE", 0))

    lines = []
    lines.append("# Task 1b — invasive native/alien split note")
    lines.append("")
    lines.append(f"Run: {timestamp}")
    lines.append(f"Git commit: {commit}{' (dirty)' if dirty else ''}")
    lines.append(f"Config hash: {fhash}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- DUAL-AXIS: **{dual}**")
    lines.append(f"- SNAPPING-ONLY: **{snap}**")
    lines.append(f"- LOW-ACC-ONLY: **{low}**")
    lines.append(f"- INELIGIBLE: **{ine}**")
    lines.append("")
    lines.append("## Entity-level results")
    lines.append("")
    view_cols = [
        "species",
        "base_species",
        "range_status",
        "category_petko2026",
        "n_clean_dedup_200m",
        "n_snap_pool",
        "n_lowacc_pool",
        "n_basins",
        "max_snap_contamination_pct",
        "max_lowacc_contamination_pct",
        "classification",
    ]
    note_df = audit[view_cols].copy().sort_values(["base_species", "range_status"])
    lines.append(note_df.to_markdown(index=False))

    note_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(f"  → {audit_path}")
    logger.info(f"  → {fea_path}")
    logger.info(f"  → {note_path}")
    logger.info(f"Task 1b complete. All outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
