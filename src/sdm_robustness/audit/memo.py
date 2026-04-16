"""Task 1 — technical memo writer."""

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
    """
    Write Task 1 technical memo.

    Backward-compatible with older call sites:
    - out_path / output_path
    - classified / classification
    """

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
    n_primary = int(vc.get("PRIMARY", 0))
    n_partial = int(vc.get("PARTIAL", 0))
    n_ineligible = int(vc.get("INELIGIBLE", 0))

    core_pass = classified[
        (classified["gate_1_min_benchmark"] == 1)
        & (classified["gate_4_basin_spread"] == 1)
        & (classified["gate_5_strahler_spread"] == 1)
    ].copy()

    gate_cols = [
        "gate_1_min_benchmark",
        "gate_2_snap_pool",
        "gate_3_lowacc_pool",
        "gate_4_basin_spread",
        "gate_5_strahler_spread",
    ]
    gate_fail_all = {g: int((classified[g] == 0).sum()) for g in gate_cols}
    gate_fail_core = {
        "gate_2_snap_pool": int((core_pass["gate_2_snap_pool"] == 0).sum()) if len(core_pass) else 0,
        "gate_3_lowacc_pool": int((core_pass["gate_3_lowacc_pool"] == 0).sum()) if len(core_pass) else 0,
    }

    top_species = inventory.sort_values(
        ["n_clean_dedup_200m", "n_basins"], ascending=[False, False]
    ).head(15)[["species", "status", "n_clean_dedup_200m", "n_basins"]]

    usable = classified[classified["classification"].isin(["PRIMARY", "PARTIAL"])].copy()
    usable = usable.sort_values(
        ["n_clean_dedup_200m", "max_snap_contamination_pct", "max_lowacc_contamination_pct"],
        ascending=[False, False, False],
    )
    usable_small = usable[
        [
            "species",
            "classification",
            "category_used",
            "n_clean_dedup_200m",
            "max_snap_contamination_pct",
            "max_lowacc_contamination_pct",
            "n_basins",
        ]
    ].head(15)

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
    lines.append(f"- PRIMARY candidates: **{n_primary}**")
    lines.append(f"- PARTIAL candidates: **{n_partial}**")
    lines.append(f"- INELIGIBLE: **{n_ineligible}**")
    lines.append("")
    lines.append(
        "- Interpretation: this dataset supports robustness experiments for a limited but usable panel of taxa, "
        "mostly at low-to-moderate contamination ceilings. The snapping axis is the main limiting factor."
    )
    lines.append("")
    lines.append("## Classification distribution by category")
    lines.append("")
    if len(classified):
        tab = (
            classified.groupby(["category_used", "classification"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        lines.append(tab.to_markdown(index=False))
    else:
        lines.append("No rows.")
    lines.append("")
    lines.append("## Gate-failure analysis")
    lines.append("")
    lines.append("### Across all taxa")
    lines.append("")
    fail_all_df = pd.DataFrame(
        {"gate": list(gate_fail_all.keys()), "n_failing": list(gate_fail_all.values())}
    )
    lines.append(fail_all_df.to_markdown(index=False))
    lines.append("")
    lines.append("### Among taxa that already pass the core ecological gates (1, 4, 5)")
    lines.append("")
    if len(core_pass):
        fail_core_df = pd.DataFrame(
            {
                "gate": list(gate_fail_core.keys()),
                "n_failing": list(gate_fail_core.values()),
            }
        )
        lines.append(fail_core_df.to_markdown(index=False))
    else:
        lines.append("No taxa passed the core ecological gates.")
    lines.append("")
    lines.append(
        "*Reading this section:* if gate 2 dominates among core-pass taxa, the snapping pool is the main "
        "constraint. If gate 3 dominates, low-accuracy records are the main constraint."
    )
    lines.append("")
    lines.append("## Top 15 species by benchmark size")
    lines.append("")
    lines.append(top_species.to_markdown(index=False))
    lines.append("")
    lines.append("## Top usable candidates")
    lines.append("")
    if len(usable_small):
        lines.append(usable_small.to_markdown(index=False))
    else:
        lines.append("No usable candidates.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "- Use PARTIAL species as the main candidate pool for Task 2. Their per-species contamination ceilings "
        "are part of the result, not a reason to discard them."
    )
    lines.append(
        "- Build the initial analysis panel around the strongest broad taxa "
        "(e.g. Procambarus clarkii, Pacifastacus leniusculus, Faxonius limosus, Astacus astacus, "
        "Pontastacus leptodactylus)."
    )
    lines.append(
        "- Add at least one balanced species with support on both axes at >=10% "
        "(especially Austropotamobius fulcisianus; optionally Cherax quadricarinatus)."
    )
    lines.append(
        "- Revisit candidate counts after adding Petko categories, because the current run treats missing categories "
        "conservatively via the default regional threshold."
    )

    out_path.write_text("\n".join(lines), encoding="utf-8")
