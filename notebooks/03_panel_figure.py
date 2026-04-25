"""
Panel-wide figure: AUC degradation curves for all 7 complete DUAL-AXIS entities,
3 algorithms × 2 contamination axes, on the combined track.
"""
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path("manuscript/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (id, short label for axis, full ecological label)
ENTITY_ORDER = [
    ("astacus_astacus",                       "Astacus\nastacus"),
    ("Pontastacus_leptodactylus_pooled",      "Pontastacus\nleptodactylus"),
    ("Austropotamobius_torrentium_pooled",    "Austropotamobius\ntorrentium"),
    ("Austropotamobius_fulcisianus_pooled",   "Austropotamobius\nfulcisianus"),
    ("Procambarus_clarkii_native",            "P. clarkii\n(native)"),
    ("Procambarus_clarkii_alien",             "P. clarkii\n(alien)"),
    ("Pacifastacus_leniusculus_alien",        "Pacifastacus\nleniusculus (alien)"),
]

ALG_ORDER  = ["random_forest", "xgboost", "maxent"]
ALG_LABEL  = {"random_forest": "RF", "xgboost": "XGBoost", "maxent": "Maxent"}
ALG_COLOR  = {"random_forest": "#1f77b4", "xgboost": "#ff7f0e", "maxent": "#2ca02c"}
AXIS_STYLE = {"snapping": "-", "lowacc": "--"}
LEVELS     = [0, 5, 10, 20, 35, 50]

RNG = np.random.default_rng(20260425)


def bootstrap_ci(values, n_boot=2000, ci=0.95):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    means = RNG.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(v.mean()), float(lo), float(hi))


# ---- Load ----
frames = []
n_exp_per_entity = {}
for ent_id, _ in ENTITY_ORDER:
    df = pd.read_parquet(f"results/task5_execution/{ent_id}/results_raw.parquet")
    df_ok = df[(df["status"] == "ok") & (df["track"] == "combined")].copy()
    df_ok["entity_id"] = ent_id
    frames.append(df_ok)
    n_exp_per_entity[ent_id] = int(df["n_experiment"].iloc[0])
all_df = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(all_df)} ok rows (combined track only) across {len(ENTITY_ORDER)} entities")

# ---- Summarise ----
rows = []
for (ent, alg, ax, lvl), sub in all_df.groupby(["entity_id","algorithm","axis","level"]):
    mean, lo, hi = bootstrap_ci(sub["auc"].values)
    rows.append({
        "entity_id": ent, "algorithm": alg, "axis": ax, "level": lvl,
        "n_reps_ok": len(sub),
        "auc_mean": mean, "auc_lo": lo, "auc_hi": hi,
    })
summary = pd.DataFrame(rows)
summary.to_csv(OUT_DIR / "panel_dual_axis_combined_summary.csv", index=False)

# ---- Figure ----
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
})

n_entities = len(ENTITY_ORDER)
n_algs = len(ALG_ORDER)
fig, axes = plt.subplots(n_entities, n_algs,
                         figsize=(11.5, 2.0 * n_entities),
                         sharex=True, sharey=False)

for i, (ent_id, ent_label) in enumerate(ENTITY_ORDER):
    nexp = n_exp_per_entity[ent_id]
    for j, alg in enumerate(ALG_ORDER):
        ax = axes[i, j]
        for axis_name in ["snapping", "lowacc"]:
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

handles = [
    plt.Line2D([0], [0], color="black", linestyle="-",  label="snapping axis"),
    plt.Line2D([0], [0], color="black", linestyle="--", label="lowacc axis"),
]
axes[0, -1].legend(handles=handles, loc="lower left", fontsize=9, frameon=True)

fig.suptitle(
    "Task 5 — AUC degradation across 7 DUAL-AXIS entities (combined track)\n"
    "30–50 replicates per cell, 5-fold spatial basin CV; bands = bootstrap 95 % CI",
    fontsize=13, y=0.995,
)
fig.subplots_adjust(left=0.16, right=0.97, top=0.945, bottom=0.04, hspace=0.30)

png = OUT_DIR / "panel_dual_axis_combined.png"
pdf = OUT_DIR / "panel_dual_axis_combined.pdf"
fig.savefig(png, dpi=180, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"Wrote {png}")
print(f"Wrote {pdf}")
plt.close(fig)
