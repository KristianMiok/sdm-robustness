#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from sdm_robustness.execution.runner import (
    load_panel_and_master,
    run_benchmark_sanity_check,
    run_core_factorial,
    run_grid_b_factorial,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 5 for one entity or the whole panel.")
    parser.add_argument("--entity", type=str, default=None, help="Exact entity name from config/final_panel.csv")
    parser.add_argument("--data-path", type=str, default=None, help="Path to combined_data_true_master.csv")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/task5_execution",
        help="Output directory for parquet results",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run only benchmark stability repeats",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Run core factorial only",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["random_forest", "xgboost", "maxent"],
        help="Algorithms to run",
    )
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=["local_only", "upstream_only", "combined"],
        help="Spatial tracks to run",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 5, 10, 20, 35, 50],
        help="Contamination levels to run",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=["snapping", "lowacc"],
        help="Axes to allow (runner will still respect entity type)",
    )
    parser.add_argument(
        "--n-replicates-default",
        type=int,
        default=30,
        help="Default replicate count for levels above threshold",
    )
    parser.add_argument(
        "--n-replicates-low-levels",
        type=int,
        default=50,
        help="Replicate count for levels <= low-level-threshold",
    )
    parser.add_argument(
        "--low-level-threshold-pct",
        type=int,
        default=10,
        help="Threshold for using low-level replicate count",
    )
    parser.add_argument(
        "--n-experiment",
        type=int,
        default=None,
        help="Override n_experiment (force constant sample size across sub-jobs)",
    )
    parser.add_argument(
        "--grid",
        choices=["A", "B"],
        default="A",
        help="A = legacy 6-level grid (Grid A). B = Lucian's asymmetric grid with Tier 2/3 metrics.",
    )
    parser.add_argument(
        "--snap-levels",
        nargs="+",
        type=int,
        default=None,
        help="Override snapping levels for Grid B (default: 0 1 2 5)",
    )
    parser.add_argument(
        "--lowacc-levels",
        nargs="+",
        type=int,
        default=None,
        help="Override lowacc levels for Grid B (default: 0 3 10 20)",
    )
    parser.add_argument(
        "--no-surfaces",
        action="store_true",
        help="Grid B only: skip saving suitability surface parquets (faster, smaller output)",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=20260426,
        help="Master seed for reproducible benchmark-stability chunks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel, master_df = load_panel_and_master(
        data_path=args.data_path,
        entity=args.entity,
    )

    if args.benchmark_only:
        benchmark_path = run_benchmark_sanity_check(
            panel,
            master_df,
            output_dir=output_dir,
            algorithms=tuple(args.algorithms),
            scale_tracks=tuple(args.tracks),
            n_replicates=args.n_replicates_default,
            master_seed=args.master_seed,
        )
        print(f"Wrote: {benchmark_path}")
        return 0

    if args.grid == "B":
        # Grid B defaults follow Lucian's asymmetric design:
        #   snapping: 0/1/2/5
        #   lowacc:   0/3/10/20
        #
        # For smoke tests and VEGA sub-jobs, respect explicit --axes and --levels.
        # Example:
        #   --grid B --axes snapping --levels 1
        # should run only snapping level 1, not the full Grid B.
        requested_axes = tuple(args.axes) if args.axes else ("snapping", "lowacc")

        if args.levels:
            # Manual override for smoke tests / sub-jobs.
            snap_levels = tuple(args.levels) if "snapping" in requested_axes else tuple()
            lowacc_levels = tuple(args.levels) if "lowacc" in requested_axes else tuple()
        else:
            # Full Grid B defaults, with optional axis-specific overrides.
            snap_levels = (
                tuple(args.snap_levels)
                if args.snap_levels
                else ((0, 1, 2, 5) if "snapping" in requested_axes else tuple())
            )
            lowacc_levels = (
                tuple(args.lowacc_levels)
                if args.lowacc_levels
                else ((0, 3, 10, 20) if "lowacc" in requested_axes else tuple())
            )

        results_path = run_grid_b_factorial(
            panel,
            master_df,
            output_dir=output_dir,
            algorithms=tuple(args.algorithms),
            scale_tracks=tuple(args.tracks),
            snap_levels_pct=snap_levels,
            lowacc_levels_pct=lowacc_levels,
            n_replicates=args.n_replicates_default,
            save_surfaces=not args.no_surfaces,
        )
        print(f"Wrote: {results_path}")
        return 0

    results_path = run_core_factorial(
        panel,
        master_df,
        output_dir=output_dir,
        algorithms=tuple(args.algorithms),
        axes=tuple(args.axes),
        levels_pct=tuple(args.levels),
        scale_tracks=tuple(args.tracks),
        n_replicates_default=args.n_replicates_default,
        n_replicates_low_levels=args.n_replicates_low_levels,
        low_level_threshold_pct=args.low_level_threshold_pct,
        n_experiment_override=args.n_experiment,
    )
    print(f"Wrote: {results_path}")

    if not args.skip_benchmark:
        benchmark_path = run_benchmark_sanity_check(
            panel,
            master_df,
            output_dir=output_dir,
            algorithms=tuple(args.algorithms),
            scale_tracks=tuple(args.tracks),
            n_replicates=args.n_replicates_default,
        )
        print(f"Wrote: {benchmark_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
