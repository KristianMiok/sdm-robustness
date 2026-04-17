"""Task 1 — technical memo writer for revised asymmetric design."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_technical_memo(
    out_path: str | Path | None = None,
    inventory: pd.DataFrame | None = None,
    feasibility: pd.DataFrame | None = None,
    classified: pd.DataFrame | None = None,
    *,
    output_path: str | Path | None = None,
    classification: pd.DataFrame | None = None,
    run_id: str,
    git_commit: str,
    config_hash: str,
) -> None:
    if out_path is None:
        out_path = output_path
    if out_path is None:
        raise ValueError("write_technical_memo requires `out_path` or `output_path`.")
    if classified is None:
        classified = classification
    if classified is None:
        raise ValueError("write_technical_memo requires `classified` or `classification`.")
    if inventory is None:
        raise ValueError("write_technical_memo requires `inventory`.")
    if feasibility is None:
        raise ValueError("write_technical_memo requires `feasibility`.")

    out_path = Path(out_path)

    n_total = len(inventory)
    vc = classified["classification"].value_counts()
    n_dual = int(vc.get("DUAL-AXIS", 0))
    n_snap_only = int(vc.get("SNAPPING-ONLY", 0))
    n_low_only = int(vc.get("LOW-ACC-ONLY", 0))
    n_ineligible = int(vc.get("INELIGIBLE", 0))

    usable = classified[classified["classification"] != "INELIGIBLE"].copy()

    cat_tab = (
        usable.groupby(["category_used", "classification"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        if len(usable)
        else pd.DataFrame()
    )

    gate_cols = [
        "gate_1_min_benchmark",
        "gate_2_snap_pool",
        "gate_3_lowacc_pool",
        "gate_4_basin_spread",
        "gate_5_strahler_spread",
    ]
    fail_all = pd.DataFrame(
        {
            "gate": gate_cols,
            "n_failing": [int((classified[g] == 0).sum()) for g in gate_cols],
        }
    )

    top_usable = usable.sort_values(
        ["n_clean_dedup_200m", "max_snap_contamination_pct", "max_lowacc_contamination_pct"],
        ascending=[False, False, False],
    ).head(15)[
        [
            "species",
            "classification",
            "category_used",
            "n_clean_dedup_200m",
            "max_snap_contamination_pct",
            "max_lowacc_contamination_pct",
            "n_basins",
        ]
    ]

    lines: list[str] = []
    lines.append("# Task 1 — Technical memo")
    lines.append("")
    lines.append(f"**Run:** {run_id}  ")
    lines.append(f"**Git commit:** `{git_commit}`  ")
    lines.append(f"**Config hash:** `{config_hash}`  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total taxa in raw dataset: **{n_total}**")
    lines.append(f"- DUAL-AXIS candidates: **{n_dual}**")
    lines.append(f"- SNAPPING-ONLY candidates: **{n_snap_only}**")
    lines.append(f"- LOW-ACC-ONLY candidates: **{n_low_only}**")
    lines.append(f"- INELIGIBLE: **{n_ineligible}**")
    lines.append("")
    lines.append(
        "- Interpretation: the revised asymmetric design is focused on realistic contamination regimes. "
        "Snapping support at 5% remains the main limiting factor, while the low-accuracy axis is broader but still selective."
    )
    lines.append("")
    if n_dual < 6:
        lines.append("- **Flag:** the DUAL-AXIS pool is below ~6 species and should be discussed before Task 2.")
        lines.append("")

    lines.append("## Candidate distribution by category")
    lines.append("")
    if len(cat_tab):
        lines.append(cat_tab.to_markdown(index=False))
    else:
        lines.append("No usable candidates.")
    lines.append("")

    lines.append("## Gate-failure analysis across all taxa")
    lines.append("")
    lines.append(fail_all.to_markdown(index=False))
    lines.append("")

    lines.append("## Top usable candidates")
    lines.append("")
    if len(top_usable):
        lines.append(top_usable.to_markdown(index=False))
    else:
        lines.append("No usable candidates.")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "- Prioritise DUAL-AXIS species for the main panel, then add SNAPPING-ONLY or LOW-ACC-ONLY taxa where they broaden ecological coverage."
    )
    lines.append(
        "- Check category assignment carefully before final panel selection; the revised rerun should avoid a blanket regional fallback."
    )
    lines.append(
        "- If the DUAL-AXIS set stays small, use it as the core panel and treat single-axis species as targeted extensions rather than trying to force symmetry."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
