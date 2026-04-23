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
        )
        print(f"Wrote: {benchmark_path}")
        return 0

    results_path = run_core_factorial(
        panel,
        master_df,
        output_dir=output_dir,
    )
    print(f"Wrote: {results_path}")

    if not args.skip_benchmark:
        benchmark_path = run_benchmark_sanity_check(
            panel,
            master_df,
            output_dir=output_dir,
        )
        print(f"Wrote: {benchmark_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
