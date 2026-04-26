"""
Appendix figures for Grid A degradation curves across all 13 entities.
Figure A1: 8 DUAL-AXIS entities, both axes (combined track).
Figure A2: 5 SNAPPING-ONLY entities, snapping axis (combined track).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("manuscript/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DUAL_ENTITIES = [
    ("astacus_astacus",                      "Astacus\nastacus"),
    ("Pontastacus_leptodactylus_pooled",     "Pontastacus\nleptodactylus"),
    ("Austropotamobius_torrentium_pooled",   "Austropotamobius\ntorrentium"),
    ("Austropotamobius_fulcisianus_pooled",  "Austropotamobius\nfulcisianus"),
    ("Procambarus_clarkii_native",           "P. clarkii\n(native)"),
    ("Procambarus_clarkii_alien",            "P. clarkii\n(alien)"),
    ("Pacifastacus_leniusculus_alien",       "Pacifastacus\nleniusculus (alien)"),
    ("Faxonius_limosus_alien",               "Faxonius\nlimosus (alien)"),
]

SNAP_ENTITIES = [
    ("Lacunicambarus_diogenes",              "Lacunicambarus\ndiogenes"),
    ("Cambarus_latimanus",                   "Cambarus\nlatimanus"),
    ("Cambarus_striatus",                    "Cambarus\nstriatus"),
    ("Creaserinus_fodiens",                  "Creaserinus\nfodiens"),
    ("Faxonius_limosus_native",              "Faxonius\nlimosus (native)"),
]

ALG_ORDER  = ["random_forest", "xgboost", "maxent"]
ALG_LABEL  = {"random_forest": "RF", "xgboost": "XGBoost", "maxent": "Maxent"}
ALG_COLOR  = {"random_forest": "#1f77b4", "xgboost": "#ff7f0e", "maxent": "#2ca02c"}
AXIS_STYLE = {"snapping": "-", "lowacc": "--"}
LEVELS     = [0, 5, 10, 20, 35, 50]

RNG = np.random.default_rng(20260426)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    means = RNG.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(v.mean()), float(lo), float(hi))


def load_entity_summary(entities, axes):
    """Load and summarize results for a list of entities, restricted to combined track."""
    frames, n_exp = [], {}
    for ent_id, _ in entities:
        df = pd.read_parquet(f"results/task5_execution/{ent_id}/results_raw.parquet")
        df_ok = df[(df["status"] == "ok") & (df["track"] == "combined")].copy()
        df_ok["entity_id"] = ent_id
        frames.append(df_ok)
        n_exp[ent_id] = int(df["n_experiment"].iloc[0])
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["axis"].isin(axes)]

    rows = []
    for (ent, alg, ax, lvl), sub in all_df.groupby(["entity_id","algorithm","axis","level"]):
        mean, lo, hi = bootstrap_ci(sub["auc"].values)
        rows.append({
            "entity_id": ent, "algorithm": alg, "axis": ax, "level": lvl,
            "n_reps_ok": len(sub),
            "auc_mean": mean, "auc_lo": lo, "auc_hi": hi,
        })
    return pd.DataFrame(rows), n_exp


def make_panel_figure(entities, axes, title_suffix, out_basename, summary, n_exp):
    n_entities = len(entities)
    n_algs = len(ALG_ORDER)

    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    })

    fig, axes_grid = plt.subplots(n_entities, n_algs,
                                  figsize=(11.5, 2.0 * n_entities),
                                  sharex=True, sharey=False)
    if n_entities == 1:
        axes_grid = axes_grid.reshape(1, -1)

    for i, (ent_id, ent_label) in enumerate(entities):
        nexp = n_exp[ent_id]
        for j, alg in enumerate(ALG_ORDER):
            ax = axes_grid[i, j]
            for axis_name in axes:
                sub = summary[
                    (summary["entity_id"] == ent_id)
                    & (summary["algorithm"] == alg)
                    & (summary["axis"] == axis_name)
                ].sort_values("level")
                if sub.empty:
                    continue
                ax.plot(sub["level"], sub["auc_mean"],
                        marker="o", markersize=4,
                        linestyle=AXIS_STYLE[axis_name],
                        color=ALG_COLOR[alg],
                        label=axis_name)
                ax.fill_between(sub["level"], sub["auc_lo"], sub["auc_hi"],
                                color=ALG_COLOR[alg], alpha=0.18)

            if i == 0:
                ax.set_title(ALG_LABEL[alg], fontsize=13, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{ent_label}\n(n={nexp})", fontsize=9.5,
                              rotation=0, ha="right", va="center", labelpad=12)
            if i == n_entities - 1:
                ax.set_xlabel("contamination level (%)")
            ax.set_xticks(LEVELS)
            ax.grid(True, alpha=0.3)

    if len(axes) > 1:
        handles = [
            plt.Line2D([0], [0], color="black", linestyle="-",  label="snapping axis"),
            plt.Line2D([0], [0], color="black", linestyle="--", label="lowacc axis"),
        ]
        axes_grid[0, -1].legend(handles=handles, loc="lower left", fontsize=9, frameon=True)

    fig.suptitle(
        f"Grid A — AUC degradation across {n_entities} {title_suffix} (combined track)\n"
        "30–50 replicates per cell, 5-fold spatial basin CV; bands = bootstrap 95 % CI",
        fontsize=13, y=0.998,
    )
    # More top margin for column titles, scaled to figure size
    top_margin = 1.0 - (0.07 if n_entities >= 6 else 0.10)
    fig.subplots_adjust(left=0.16, right=0.97, top=top_margin, bottom=0.04, hspace=0.30)

    png = OUT_DIR / f"{out_basename}.png"
    pdf = OUT_DIR / f"{out_basename}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    plt.close(fig)


# ---- Figure A1: DUAL-AXIS (both axes) ----
print("\n=== Building DUAL-AXIS appendix figure ===")
summary_dual, n_exp_dual = load_entity_summary(DUAL_ENTITIES, ["snapping", "lowacc"])
make_panel_figure(
    DUAL_ENTITIES, ["snapping", "lowacc"],
    "DUAL-AXIS entities",
    "appendix_A1_dual_axis_combined",
    summary_dual, n_exp_dual,
)

# ---- Figure A2: SNAPPING-ONLY ----
print("\n=== Building SNAPPING-ONLY appendix figure ===")
summary_snap, n_exp_snap = load_entity_summary(SNAP_ENTITIES, ["snapping"])
make_panel_figure(
    SNAP_ENTITIES, ["snapping"],
    "SNAPPING-ONLY entities",
    "appendix_A2_snapping_only_combined",
    summary_snap, n_exp_snap,
)

# Save tidy summary CSVs
summary_dual.to_csv(OUT_DIR / "appendix_A1_summary.csv", index=False)
summary_snap.to_csv(OUT_DIR / "appendix_A2_summary.csv", index=False)
print(f"\nWrote summary CSVs.")
