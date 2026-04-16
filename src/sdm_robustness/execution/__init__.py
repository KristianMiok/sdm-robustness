"""Task 5 — stress-test execution."""

from sdm_robustness.execution.contamination import (
    ContaminationDraw,
    draw_substitution_sample,
)
from sdm_robustness.execution.runner import (
    run_benchmark_sanity_check,
    run_core_factorial,
    run_null_model,
    run_transferability_test,
)

__all__ = [
    "ContaminationDraw",
    "draw_substitution_sample",
    "run_benchmark_sanity_check",
    "run_core_factorial",
    "run_null_model",
    "run_transferability_test",
]
