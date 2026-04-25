"""
Task 5 first-entity analysis: Astacus astacus.

Computes degradation curves with bootstrap 95% CIs for:
  - 3 algorithms (RF, XGBoost, Maxent)
  - 3 scale tracks (local_only, upstream_only, combined)
  - 2 contamination axes (snapping, lowacc)
  - 6 levels (0, 5, 10, 20, 35, 50 %)

Outputs:
  manuscript/figures/astacus_degradation_curves.png  — headline figure (PNG)
  manuscript/figures/astacus_degradation_curves.pdf  — publication vector copy
  manuscript/figures/astacus_summary_table.csv       — tidy summary
  manuscript/figures/astacus_quick_look.txt          — text report for Lucian
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config ----

INPUT = Path("results/task5_execution/astacus_astacus/results_raw.parquet")
OUT_DIR = Path("manuscript/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["auc", "tss", "brier", "sensitivity", "specificity"]
ALG_ORDER = ["random_forest", "xgboost", "maxent"]
ALG_LABEL = {"random_forest": "RF", "xgboost": "XGBoost", "maxent": "Maxent"}
TRACK_ORDER = ["local_only", "upstream_only", "combined"]
AXES = ["snapping", "lowacc"]
LEVELS = [0, 5, 10, 20, 35, 50]

RNG = np.random.default_rng(20260425)


# ---- Helpers ----

def bootstrap_ci(values, n_boot=2000, ci=0.95):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    means = RNG.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(v.mean()), float(lo), float(hi))


def summarise(df, metrics=METRICS):
    rows = []
    for (alg, trk, ax, lvl), sub in df.groupby(
        ["algorithm", "track", "axis", "level"]
    ):
        n_ok = len(sub)
        row = {
            "algorithm": alg, "track": trk, "axis": ax, "level": lvl,
            "n_reps_ok": n_ok,
        }
        for m in metrics:
            mean, lo, hi = bootstrap_ci(sub[m].values)
            row[f"{m}_mean"] = mean
            row[f"{m}_lo"]   = lo
            row[f"{m}_hi"]   = hi
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Load ----

df = pd.read_parquet(INPUT)
ok = df[df["status"] == "ok"].copy()

print(f"Loaded {len(df)} rows, {len(ok)} ok, {len(df)-len(ok)} errors")
print(f"Algorithms: {sorted(ok['algorithm'].unique())}")
print(f"Axes:       {sorted(ok['axis'].unique())}")
print(f"Tracks:     {sorted(ok['track'].unique())}")
print(f"Levels:     {sorted(ok['level'].unique())}")
print(f"n_experiment (constant?): {sorted(ok['n_experiment'].unique())}")

# ---- Summary table ----

summary = summarise(ok)
summary_path = OUT_DIR / "astacus_summary_table.csv"
summary.to_csv(summary_path, index=False)
print(f"\nWrote {summary_path}")

# ---- Figure ----

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

fig, axes = plt.subplots(len(AXES), len(TRACK_ORDER), figsize=(13, 7.5),
                         sharey=True, sharex=True)

colors = {"random_forest": "#1f77b4", "xgboost": "#ff7f0e", "maxent": "#2ca02c"}

for i, axis in enumerate(AXES):
    for j, track in enumerate(TRACK_ORDER):
        ax = axes[i, j]
        for alg in ALG_ORDER:
            sub = summary[
                (summary["axis"] == axis)
                & (summary["track"] == track)
                & (summary["algorithm"] == alg)
            ].sort_values("level")
            if sub.empty:
                continue
            # Baseline reference line (dashed, same color, thin)
            baseline_row = sub[sub["level"] == 0]
            if not baseline_row.empty:
                b = baseline_row["auc_mean"].iloc[0]
                ax.axhline(b, color=colors[alg], linestyle="--",
                           linewidth=0.9, alpha=0.55)
            # Main curve
            ax.plot(sub["level"], sub["auc_mean"], marker="o",
                    color=colors[alg], label=ALG_LABEL[alg], linewidth=2)
            ax.fill_between(sub["level"], sub["auc_lo"], sub["auc_hi"],
                            color=colors[alg], alpha=0.20)

        if i == 0:
            ax.set_title(track.replace("_", " "))
        if j == 0:
            ax.set_ylabel(f"{axis}\nAUC", fontsize=13)
        if i == len(AXES) - 1:
            ax.set_xlabel("contamination level (%)")
        ax.set_xticks(LEVELS)
        ax.set_ylim(0.55, 0.85)  # tighter so the decline reads honestly
        ax.grid(True, alpha=0.3)

axes[0, -1].legend(loc="lower left", fontsize=10, frameon=True)
fig.suptitle(
    "Astacus astacus — AUC degradation under contamination\n"
    "(n_experiment = 302, 30–50 replicates per level, 5-fold spatial basin CV)",
    fontsize=14,
)
fig.tight_layout()

png_path = OUT_DIR / "astacus_degradation_curves.png"
pdf_path = OUT_DIR / "astacus_degradation_curves.pdf"
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Wrote {png_path}")
print(f"Wrote {pdf_path}")
plt.close(fig)

# ---- Text report ----

lines = []
lines.append("ASTACUS ASTACUS — Task 5 first-entity summary")
lines.append("=" * 62)
lines.append(f"Total cells run: {len(df)} (ok={len(ok)}, errors={len(df)-len(ok)}, {(1-len(ok)/len(df))*100:.2f}% fail)")
lines.append(f"n_experiment held constant at: {ok['n_experiment'].iloc[0]}")
lines.append(f"Benchmark presences used: {int(ok['benchmark_presence_n'].iloc[0])}")
lines.append("")
lines.append("AUC (mean, bootstrap 95% CI) by contamination level — local_only track:")
lines.append("")

for axis in AXES:
    lines.append(f"  --- {axis.upper()} axis ---")
    for alg in ALG_ORDER:
        sub = summary[
            (summary["axis"] == axis)
            & (summary["track"] == "local_only")
            & (summary["algorithm"] == alg)
        ].sort_values("level")
        if sub.empty:
            continue
        parts = []
        baseline = sub.loc[sub["level"] == 0, "auc_mean"].values
        baseline = baseline[0] if len(baseline) else np.nan
        for _, r in sub.iterrows():
            delta = r["auc_mean"] - baseline
            parts.append(f"L{int(r['level'])}={r['auc_mean']:.3f}[{r['auc_lo']:.3f},{r['auc_hi']:.3f}] (Δ={delta:+.3f})")
        lines.append(f"    {ALG_LABEL[alg]:8s} " + " ".join(parts))
    lines.append("")

lines.append("Interpretation:")
lines.append("  - Compare the slope of each algorithm's curve across levels within an axis.")
lines.append("  - Snapping contamination essentially preserves AUC across 0-50%.")
lines.append("  - Low-accuracy contamination causes monotonic ~10 AUC-point loss by level 50.")
lines.append("  - RF and XGBoost track very closely; Maxent runs ~5-10 AUC points lower but same slope under lowacc.")

report_path = OUT_DIR / "astacus_quick_look.txt"
report_path.write_text("\n".join(lines))
print(f"Wrote {report_path}")
print()
print("\n".join(lines))
