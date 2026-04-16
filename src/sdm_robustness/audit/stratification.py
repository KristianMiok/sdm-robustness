"""Task 1 — Step 1.4: stratification diagnostic.

Produce a 2-panel figure showing how PRIMARY candidates distribute across
(a) Petko et al. 2026 distributional categories and (b) Status
(Native / Alien / Mixed).

This tells Lucian immediately whether final stratified selection (Task 2)
is feasible, or whether gates need relaxing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sdm_robustness.utils import logger


def plot_stratification_diagnostic(
    classification: pd.DataFrame,
    inventory: pd.DataFrame,
    output_path: Path | str,
    *,
    title_suffix: str = "",
) -> Path:
    """Plot 2-panel stratification diagnostic for PRIMARY candidates.

    Gracefully handles the zero-PRIMARY case by producing an informative
    placeholder figure instead of crashing.

    Parameters
    ----------
    classification : DataFrame
        Output of audit.gates.classify_candidates().
    inventory : DataFrame
        Output of audit.inventory.build_inventory() (for status lookup).
    output_path : path
        Where to write the PDF.
    title_suffix : str
        Appended to the figure title (e.g., run date).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary = classification[classification["classification"] == "PRIMARY"].copy()

    # Handle the empty-PRIMARY case explicitly — don't hand NaN to matplotlib.
    if primary.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "No PRIMARY candidates under current gates.\n\n"
            "See candidate_shortlist.csv and technical_memo.md\n"
            "for gate-failure analysis and remediation options.",
            ha="center", va="center", fontsize=12,
            transform=ax.transAxes,
        )
        # Also show a small side-panel with PARTIAL / INELIGIBLE counts for context
        counts = classification["classification"].value_counts()
        subtitle = "  |  ".join(f"{k}: {v}" for k, v in counts.items())
        fig.suptitle(
            f"Task 1 — stratification diagnostic{title_suffix}\n{subtitle}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        logger.warning(
            f"No PRIMARY candidates — stratification diagnostic written as "
            f"placeholder to {output_path}"
        )
        return output_path

    # Merge status from inventory
    status_map = inventory.set_index("species")["status"]
    primary["status"] = primary["species"].map(status_map).fillna("Unknown")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: category distribution
    cat_counts = primary["category_used"].value_counts().reindex(
        ["endemic", "regional", "widespread"], fill_value=0
    )
    axes[0].bar(
        cat_counts.index,
        cat_counts.values,
        color=["#d97757", "#5b8ec2", "#6aa668"],
        edgecolor="black",
        linewidth=0.5,
    )
    for i, v in enumerate(cat_counts.values):
        axes[0].text(i, v + 0.5, str(int(v)), ha="center", fontsize=10)
    axes[0].set_title(f"PRIMARY candidates by category{title_suffix}", fontsize=11)
    axes[0].set_ylabel("N species")
    axes[0].set_ylim(0, max(cat_counts.max() * 1.15, 1))
    axes[0].spines[["top", "right"]].set_visible(False)

    # Panel B: status distribution
    status_counts = primary["status"].value_counts()
    axes[1].bar(
        status_counts.index,
        status_counts.values,
        color=["#5b8ec2", "#d97757", "#999999"][: len(status_counts)],
        edgecolor="black",
        linewidth=0.5,
    )
    for i, v in enumerate(status_counts.values):
        axes[1].text(i, v + 0.5, str(int(v)), ha="center", fontsize=10)
    axes[1].set_title(f"PRIMARY candidates by status{title_suffix}", fontsize=11)
    axes[1].set_ylabel("N species")
    axes[1].set_ylim(0, max(status_counts.max() * 1.15, 1))
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Task 1 — stratification diagnostic (n = {len(primary)} PRIMARY)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Stratification diagnostic written to {output_path}")
    return output_path
