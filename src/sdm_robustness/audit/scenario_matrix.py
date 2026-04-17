"""Task 1 — Step 1.5: revised scenario feasibility matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sdm_robustness.utils import logger


def _mark(x: int) -> str:
    return "✓" if int(x) == 1 else "–"


def build_scenario_matrix(
    classified: pd.DataFrame,
    feasibility: pd.DataFrame | None = None,
    borderline_margin_pct: float | None = None,
) -> pd.DataFrame:
    """Build revised scenario feasibility matrix.

    Backward-compatible with older runner signatures that still pass:
    - feasibility
    - borderline_margin_pct

    These are ignored in the revised Task 1 design because the matrix is
    derived directly from the classified table and exact feasibility flags.
    """
    keep = classified[classified["classification"] != "INELIGIBLE"].copy()

    if keep.empty:
        return pd.DataFrame(
            columns=[
                "species",
                "classification",
                "category_used",
                "snap_1",
                "snap_2",
                "snap_5",
                "lowacc_3",
                "lowacc_10",
                "lowacc_20",
                "max_snap_contamination_pct",
                "max_lowacc_contamination_pct",
            ]
        )

    out = pd.DataFrame(
        {
            "species": keep["species"].values,
            "classification": keep["classification"].values,
            "category_used": keep["category_used"].values,
            "snap_1": keep["feas_snap_1"].map(_mark).values,
            "snap_2": keep["feas_snap_2"].map(_mark).values,
            "snap_5": keep["feas_snap_5"].map(_mark).values,
            "lowacc_3": keep["feas_lowacc_3"].map(_mark).values,
            "lowacc_10": keep["feas_lowacc_10"].map(_mark).values,
            "lowacc_20": keep["feas_lowacc_20"].map(_mark).values,
            "max_snap_contamination_pct": keep["max_snap_contamination_pct"].values,
            "max_lowacc_contamination_pct": keep["max_lowacc_contamination_pct"].values,
        }
    )

    if "n_clean_dedup_200m" in keep.columns:
        out["n_clean_dedup_200m"] = keep["n_clean_dedup_200m"].values
        out = out.sort_values(
            ["classification", "n_clean_dedup_200m", "species"],
            ascending=[True, False, True],
        )
        out = out.drop(columns=["n_clean_dedup_200m"])
    else:
        out = out.sort_values(["classification", "species"], ascending=[True, True])

    logger.info("Scenario matrix built: %d candidate species.", len(out))
    return out.reset_index(drop=True)


def write_scenario_markdown(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    lines: list[str] = []
    lines.append("# Task 1 — Scenario feasibility matrix")
    lines.append("")
    lines.append("Revised contamination grid:")
    lines.append("- Snapping: 0 / 1 / 2 / 5 %")
    lines.append("- Low-accuracy: 0 / 3 / 10 / 20 %")
    lines.append("")
    if len(df):
        lines.append(df.to_markdown(index=False))
    else:
        lines.append("No usable candidates.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Scenario matrix markdown written to %s", out_path)
