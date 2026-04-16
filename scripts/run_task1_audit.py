#!/usr/bin/env python
"""Run Task 1 — dataset audit and candidate species selection.

Usage
-----
    python scripts/run_task1_audit.py
    python scripts/run_task1_audit.py --data /path/to/combined_data_true_master.csv
    python scripts/run_task1_audit.py --output-dir results/task1_audit_2026-04-16
    python scripts/run_task1_audit.py --petko-categories data/petko_categories.csv

Outputs (in --output-dir):
    species_audit_full.csv
    feasibility_metrics.csv
    candidate_shortlist.csv
    scenario_feasibility.csv
    scenario_feasibility.md
    stratification_diagnostic.pdf
    technical_memo.md
    run.log
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sdm_robustness.audit import (
    build_inventory,
    build_scenario_matrix,
    classify_candidates,
    compute_feasibility,
    plot_stratification_diagnostic,
    render_scenario_matrix_markdown,
    write_technical_memo,
)
from sdm_robustness.io import load_master_table
from sdm_robustness.utils import (
    config_hash,
    get_git_commit,
    get_git_dirty,
    load_frozen_design,
    load_paths,
    load_task1_gates,
    logger,
    project_root,
    resolve_path,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 — dataset audit.")
    p.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to master CSV. Overrides configs/paths.yaml.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/task1_audit/<timestamp>/",
    )
    p.add_argument(
        "--petko-categories",
        type=str,
        default=None,
        help="Optional CSV with columns (species, category) from Petko 2026 Supp. Table 3.",
    )
    p.add_argument(
        "--dedup-key",
        type=str,
        default="subc_id",
        help="Column defining Hydrography90m segment for deduplication.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ----- Resolve output directory -----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        paths = load_paths()
        output_dir = resolve_path(paths["task1_results_dir"]) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Logging -----
    setup_logging(output_dir, level="DEBUG" if args.verbose else "INFO")
    logger.info(f"Task 1 audit starting. Output → {output_dir}")

    # ----- Provenance -----
    commit = get_git_commit(project_root())
    dirty = get_git_dirty(project_root())
    if commit:
        logger.info(f"Git commit: {commit}{' (dirty)' if dirty else ''}")
    frozen = load_frozen_design()
    gates = load_task1_gates()
    fhash = config_hash({"frozen": frozen, "gates": gates})
    logger.info(f"Config hash: {fhash}")

    # ----- Load data -----
    data_path = Path(args.data) if args.data else None
    df, info = load_master_table(data_path)

    # ----- Optional Petko categories -----
    petko_cats = None
    if args.petko_categories:
        petko_path = Path(args.petko_categories)
        if petko_path.exists():
            petko_cats = pd.read_csv(petko_path)
            logger.info(f"Loaded Petko categories for {len(petko_cats)} species.")
        else:
            logger.warning(
                f"Petko categories file not found at {petko_path} — continuing without."
            )

    # ----- Step 1.1: inventory -----
    logger.info("Step 1.1 — building inventory …")
    strict_m = frozen["benchmark"]["snapping_threshold_m"]
    inv = build_inventory(
        df,
        strict_m=strict_m,
        intermediate_m=500,
        maximum_m=1000,
        dedup_key=args.dedup_key,
        petko_categories=petko_cats,
    )
    inv_path = output_dir / "species_audit_full.csv"
    inv.to_csv(inv_path, index=False)
    logger.info(f"  → {inv_path}  ({len(inv)} species)")

    # ----- Step 1.2: feasibility -----
    logger.info("Step 1.2 — computing feasibility metrics …")
    fea = compute_feasibility(inv)
    fea_path = output_dir / "feasibility_metrics.csv"
    fea.to_csv(fea_path, index=False)
    logger.info(f"  → {fea_path}")

    # ----- Step 1.3: gates -----
    logger.info("Step 1.3 — applying candidacy gates …")
    cls = classify_candidates(inv, fea, gates)
    cls_path = output_dir / "candidate_shortlist.csv"
    cls.to_csv(cls_path, index=False)
    logger.info(f"  → {cls_path}")

    # ----- Step 1.4: stratification diagnostic -----
    logger.info("Step 1.4 — stratification diagnostic figure …")
    fig_path = output_dir / "stratification_diagnostic.pdf"
    try:
        plot_stratification_diagnostic(cls, inv, fig_path)
    except Exception as e:  # noqa: BLE001 — we want to continue on figure error
        logger.exception(f"Diagnostic figure failed: {e}")

    # ----- Step 1.5: scenario matrix -----
    logger.info("Step 1.5 — scenario feasibility matrix …")
    margin = gates.get("borderline_margin_pct", 10.0)
    matrix = build_scenario_matrix(cls, fea, borderline_margin_pct=margin)
    matrix_csv = output_dir / "scenario_feasibility.csv"
    matrix.to_csv(matrix_csv, index=False)
    matrix_md = output_dir / "scenario_feasibility.md"
    render_scenario_matrix_markdown(matrix, matrix_md)
    logger.info(f"  → {matrix_csv}")
    logger.info(f"  → {matrix_md}")

    # ----- Technical memo -----
    logger.info("Writing technical memo …")
    memo_path = output_dir / "technical_memo.md"
    write_technical_memo(
        inventory=inv,
        classification=cls,
        feasibility=fea,
        output_path=memo_path,
        config_hash=fhash,
        git_commit=commit,
        run_id=timestamp,
    )
    logger.info(f"  → {memo_path}")

    logger.info(f"Task 1 complete. All outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
