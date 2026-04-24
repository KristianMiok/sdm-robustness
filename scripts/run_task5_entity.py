#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from sdm_robustness.execution.runner import (
    load_panel_and_master,
    run_benchmark_sanity_check,
    run_core_factorial,
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
        )
        print(f"Wrote: {benchmark_path}")
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
