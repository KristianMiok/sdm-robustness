"""Task 1 — technical memo generator.

Produces the one-page technical memo required as the final Task 1 deliverable.
Summarises the audit, highlights the most limiting gates, flags anomalies,
and gives a recommendation on whether to tighten or relax gates.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from sdm_robustness.utils import logger


def write_technical_memo(
    inventory: pd.DataFrame,
    classification: pd.DataFrame,
    feasibility: pd.DataFrame,
    output_path: Path | str,
    *,
    config_hash: str = "",
    git_commit: str | None = None,
    run_id: str = "",
) -> Path:
    """Write the technical memo (markdown)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = len(inventory)
    class_counts = classification["classification"].value_counts().to_dict()
    n_primary = class_counts.get("PRIMARY", 0)
    n_partial = class_counts.get("PARTIAL", 0)
    n_ineligible = class_counts.get("INELIGIBLE", 0)

    # Per-category PRIMARY counts
    primary = classification[classification["classification"] == "PRIMARY"]
    by_cat = primary["category_used"].value_counts().to_dict()

    # Gate-failure analysis — which gate is most limiting among non-PRIMARY?
    gate_cols = [c for c in classification.columns if c.startswith("gate_")]
    failing = classification[classification["classification"] != "PRIMARY"]
    gate_fail_rates = {
        col: int((failing[col] == 0).sum()) for col in gate_cols
    }

    # Top species by benchmark size (a sanity check — widespread species first)
    top_by_benchmark = (
        inventory.sort_values("n_clean_dedup_200m", ascending=False)
        .head(15)[["species", "status", "n_clean_dedup_200m", "n_basins"]]
    )

    # Pool-size asymmetries among PRIMARY (snapping-rich vs. lowacc-rich)
    fea_idx = feasibility.set_index("species")
    primary_fea = fea_idx.loc[fea_idx.index.intersection(primary["species"])]
    if not primary_fea.empty:
        snap_vs_lowacc_diff = (
            primary_fea["n_snap_pool"] - primary_fea["n_lowacc_pool"]
        ).describe()
    else:
        snap_vs_lowacc_diff = pd.Series(dtype=float)

    # Anomalies: species with n_total_raw > 100 but n_clean_dedup_200m == 0
    anomalies = inventory[
        (inventory["n_total_raw"] > 100) & (inventory["n_clean_dedup_200m"] == 0)
    ]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("# Task 1 — Technical memo\n\n")
        fh.write(f"**Run:** {run_id or now}  \n")
        if git_commit:
            fh.write(f"**Git commit:** `{git_commit}`  \n")
        if config_hash:
            fh.write(f"**Config hash:** `{config_hash}`  \n")
        fh.write("\n---\n\n")

        fh.write("## Summary\n\n")
        fh.write(f"- Total taxa in raw dataset: **{n_total}**\n")
        fh.write(f"- PRIMARY candidates: **{n_primary}**\n")
        fh.write(f"- PARTIAL candidates: **{n_partial}** (single-axis or capped)\n")
        fh.write(f"- INELIGIBLE: **{n_ineligible}**\n\n")

        fh.write("## PRIMARY distribution by category\n\n")
        for cat in ("endemic", "regional", "widespread"):
            fh.write(f"- {cat}: **{by_cat.get(cat, 0)}**\n")
        fh.write("\n")

        fh.write("## Gate-failure analysis (among non-PRIMARY species)\n\n")
        fh.write("| Gate | N failing |\n|------|-----------|\n")
        for gate, n in gate_fail_rates.items():
            fh.write(f"| {gate} | {n} |\n")
        fh.write(
            "\n*Reading this table:* the most limiting gate is the one with the "
            "highest count. If Gate 2 (snapping pool) or Gate 3 (low-acc pool) "
            "dominates, consider whether the substitution-design ceiling of "
            "50% should be reduced for a subset of species, or whether the "
            "200 m benchmark boundary should be relaxed to 500 m for feasibility.\n\n"
        )

        fh.write("## Top 15 species by benchmark size\n\n")
        fh.write(top_by_benchmark.to_markdown(index=False))
        fh.write("\n\n")

        if not snap_vs_lowacc_diff.empty:
            fh.write("## Contamination-pool asymmetry (PRIMARY candidates)\n\n")
            fh.write(
                "Positive values mean snapping pool exceeds low-accuracy pool.\n\n"
            )
            fh.write("```\n")
            fh.write(snap_vs_lowacc_diff.to_string())
            fh.write("\n```\n\n")

        if not anomalies.empty:
            fh.write("## Anomalies worth inspecting\n\n")
            fh.write(
                f"{len(anomalies)} taxa have > 100 raw records but "
                "zero records in the clean deduplicated benchmark. "
                "Likely causes: all records are Low accuracy, all have "
                "snapping > 200 m, or all collapse to a single segment.\n\n"
            )
            fh.write(anomalies[["species", "n_total_raw", "n_high_acc", "n_low_acc"]].head(15).to_markdown(index=False))
            fh.write("\n\n")

        fh.write("## Recommendation\n\n")
        if n_primary >= 10:
            fh.write(
                "- Candidate pool is sufficient for the target panel of ~10 species. "
                "Proceed to Task 2 (ecological selection) with the current gates.\n"
            )
        elif n_primary + n_partial >= 10:
            fh.write(
                "- PRIMARY pool is thin but PARTIAL candidates can fill the panel. "
                "Task 2 should consider single-axis PARTIAL species as valid "
                "entries — their limitation is itself informative.\n"
            )
        else:
            fh.write(
                "- Candidate pool is too small at the current gate thresholds. "
                "Recommend relaxing the snapping benchmark boundary from 200 m "
                "to 500 m and re-running Task 1 before Task 2.\n"
            )
        fh.write(
            "- Kristian's methodological recommendations pending Lucian's sign-off "
            "(documented in `docs/protocols/kristian_recommendations.md`): "
            "50 replicates for low-contamination levels; spatial-CV fallback for "
            "endemic species with n_basins < 5; weighted top-K for IST; "
            "transferability at 10/30/50 instead of a single 30%; null-model run; "
            "Strahler-stratified pseudo-absence density.\n"
        )

    logger.info(f"Technical memo written to {output_path}")
    return output_path
