"""Task 1 — Step 1.5: scenario feasibility matrix.

For each PRIMARY and PARTIAL candidate, produce a per-species matrix across
the full level set {5, 10, 20, 35, 50} on both axes:

    ✓  feasible
    ✗  infeasible
    ⚠  borderline (< 10% margin on required pool size)

Output is a CSV with integer-coded cells (1 = ✓, 0 = ✗, -1 = ⚠) plus a
companion human-readable .md rendering with the symbols.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdm_robustness.utils import logger

# Integer codes used in the CSV
CODE_FEASIBLE = 1
CODE_BORDERLINE = -1
CODE_INFEASIBLE = 0

# Symbol mapping for the markdown rendering
_SYMBOLS = {CODE_FEASIBLE: "✓", CODE_BORDERLINE: "⚠", CODE_INFEASIBLE: "✗"}


def build_scenario_matrix(
    classification: pd.DataFrame,
    feasibility: pd.DataFrame,
    *,
    levels_pct: tuple[int, ...] = (5, 10, 20, 35, 50),
    borderline_margin_pct: float = 10.0,
) -> pd.DataFrame:
    """Build the per-species feasibility matrix.

    Parameters
    ----------
    classification : DataFrame
        Output of audit.gates.classify_candidates().
    feasibility : DataFrame
        Output of audit.feasibility.compute_feasibility().
    levels_pct : tuple
        Contamination levels to evaluate.
    borderline_margin_pct : float
        Pool margin below which a scenario is flagged as borderline.

    Returns
    -------
    DataFrame with one row per candidate species and columns for each
    (axis, level) combination plus a 2D feasibility flag.
    """
    # Keep only PRIMARY and PARTIAL
    candidates = classification[
        classification["classification"].isin(["PRIMARY", "PARTIAL"])
    ][["species", "classification"]].copy()

    fea = feasibility.set_index("species")
    mat = candidates.set_index("species").join(fea, how="left")

    margin = borderline_margin_pct / 100.0
    records = []
    for species, row in mat.iterrows():
        n_exp = row["n_experiment_assumed"]
        rec = {"species": species, "classification": row["classification"]}
        for axis, pool_col in (("snap", "n_snap_pool"), ("lowacc", "n_lowacc_pool")):
            pool = row[pool_col]
            for level in levels_pct:
                required = (level / 100.0) * n_exp
                if pool >= required:
                    if pool < required * (1 + margin):
                        code = CODE_BORDERLINE
                    else:
                        code = CODE_FEASIBLE
                else:
                    code = CODE_INFEASIBLE
                rec[f"{axis}_{level}pct"] = code
        rec["feas_2d"] = int(row.get("feas_2d", 0))
        records.append(rec)

    out = pd.DataFrame(records)
    logger.info(f"Scenario matrix built: {len(out)} candidate species.")
    return out


def render_scenario_matrix_markdown(
    matrix: pd.DataFrame, output_path: Path | str
) -> Path:
    """Render the integer-coded matrix as human-readable markdown."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = matrix.copy()
    code_cols = [c for c in display.columns if c.endswith("pct") or c == "feas_2d"]
    for col in code_cols:
        display[col] = display[col].map(lambda v: _SYMBOLS.get(int(v), "?"))

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Scenario feasibility matrix\n\n")
        fh.write("Legend: ✓ feasible  ⚠ borderline (<10% margin)  ✗ infeasible\n\n")
        fh.write(display.to_markdown(index=False))
        fh.write("\n")

    logger.info(f"Scenario matrix markdown written to {output_path}")
    return output_path
