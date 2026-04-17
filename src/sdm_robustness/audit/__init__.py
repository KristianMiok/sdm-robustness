"""Audit subpackage exports."""

from sdm_robustness.audit.inventory import build_inventory
from sdm_robustness.audit.feasibility import (
    SNAP_LEVELS,
    LOWACC_LEVELS,
    compute_feasibility,
)
from sdm_robustness.audit.gates import classify_candidates
from sdm_robustness.audit.stratification import plot_stratification_diagnostic
from sdm_robustness.audit.scenario_matrix import (
    build_scenario_matrix,
    write_scenario_markdown,
)
from sdm_robustness.audit.memo import write_technical_memo

# Backward-compatible alias for older runner code
render_scenario_matrix_markdown = write_scenario_markdown

__all__ = [
    "SNAP_LEVELS",
    "LOWACC_LEVELS",
    "build_inventory",
    "compute_feasibility",
    "classify_candidates",
    "plot_stratification_diagnostic",
    "build_scenario_matrix",
    "write_scenario_markdown",
    "render_scenario_matrix_markdown",
    "write_technical_memo",
]
