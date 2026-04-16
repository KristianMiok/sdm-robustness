"""Task 1 — dataset audit and candidate species selection.

Five steps per briefing §3:
    1.1  inventory         per-species pool sizes
    1.2  feasibility       feasibility metrics under substitution design
    1.3  gates             candidacy gates → PRIMARY / PARTIAL / INELIGIBLE
    1.4  stratification    diagnostic figure on PRIMARY distribution
    1.5  scenario_matrix   per-species feasibility matrix
    (memo)                 one-page technical memo for Lucian
"""

from sdm_robustness.audit.feasibility import (
    CONTAMINATION_LEVELS_PCT,
    compute_feasibility,
)
from sdm_robustness.audit.gates import classify_candidates
from sdm_robustness.audit.inventory import build_inventory
from sdm_robustness.audit.memo import write_technical_memo
from sdm_robustness.audit.scenario_matrix import (
    build_scenario_matrix,
    render_scenario_matrix_markdown,
)
from sdm_robustness.audit.stratification import plot_stratification_diagnostic

__all__ = [
    "CONTAMINATION_LEVELS_PCT",
    "build_inventory",
    "build_scenario_matrix",
    "classify_candidates",
    "compute_feasibility",
    "plot_stratification_diagnostic",
    "render_scenario_matrix_markdown",
    "write_technical_memo",
]
