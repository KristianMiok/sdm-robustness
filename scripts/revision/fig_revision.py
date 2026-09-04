#!/usr/bin/env python3
"""Revision figures: full 13-entity panel, paired deltas + hierarchical bootstrap.

Changes vs the DUAL-8 round:
  - 13 entities, not 8; no track filter beyond `combined` for the main panels
  - performance metrics shown as PAIRED deltas (contaminated - matched benchmark
    replicate) with a hierarchical bootstrap interval, not raw values vs an envelope
  - benchmark-referenced metrics (schoener_d, warren_i, importance_spearman,
    niche_*, range_area_*) shown as levels: they already are comparisons
  - range metrics aggregated by entity-level MEDIAN (right-skewed)
  - Jaccard panel replaced by full-vector rank correlation
  - snapping series stops at 5%; level 10 drawn as a separate open marker
"""
import sys, os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results/figures"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20260904); NBOOT = 1000

ALG_COLOR = {"random_forest":"#1f77b4","xgboost":"#d62728","maxent":"#2ca02c"}
ALG_LABEL = {"random_forest":"Random Forest","xgboost":"XGBoost","maxent":"Maxent"}
AXIS_STYLE = {"snapping":"-","lowacc":"--"}
ALG_ORDER = ["random_forest","xgboost","maxent"]
CORE = {"snapping":[1,2,5], "lowacc":[3,10,20]}
SNAP_EXTRA = 10
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8,
                     "svg.fonttype":"none","pdf.fonttype":42})

D = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in
               sorted(glob.glob("results/campaign/*/results_raw.parquet"))], ignore_index=True)
OK = D[D.status == "ok"].copy()
print(f"rows ok={len(OK)}  entities={OK.entity.nunique()}  tracks={sorted(OK.track.unique())}")

PAIRED = [m for m in ("auc","tss","brier","boyce") if m in OK.columns]
KEY = ["entity","algorithm","track","replicate"]
B = OK[OK.axis == "benchmark"].groupby(KEY, as_index=False)[PAIRED].mean()
C = OK[OK.axis != "benchmark"].merge(B, on=KEY, suffixes=("","_b"))
for m in PAIRED:
    C[m + "_d"] = C[m] - C[m + "_b"]
print(f"paired rows={len(C)}  paired metrics={PAIRED}")

def hboot(g, col, stat="mean"):
    by = {e: v.dropna().to_numpy() for e, v in g.groupby("entity")[col]}
    by = {e: v for e, v in by.items() if len(v)}
    if len(by) < 2: return (np.nan, np.nan, np.nan, 0)
    ks = list(by); f = np.median if stat == "median" else np.mean
    point = float(f([by[k].mean() for k in ks]))
    out = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = RNG.choice(len(ks), len(ks), True)
        out[i] = f([by[ks[p]][RNG.integers(0, len(by[ks[p]]), len(by[ks[p]]))].mean()
                    for p in pick])
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(ks)

def panel(df, col, title, fname, ylab, stat="mean", zero=False, algs=ALG_ORDER):
    if col not in df.columns or df[col].notna().sum() == 0:
        print(f"[SKIP] {fname}: '{col}' absent or all-NaN"); return
    fig, ax = plt.subplots(figsize=(4.2, 3.4)); ns = {}
    for alg in algs:
        for axis, levels in CORE.items():
            xs, ms, los, his = [], [], [], []
            for lvl in levels:
                s = df[(df.axis == axis) & (df.level == lvl) & (df.algorithm == alg)]
                if s.empty: continue
                m, lo, hi, ne = hboot(s, col, stat)
                if np.isnan(m): continue
                xs.append(lvl); ms.append(m); los.append(lo); his.append(hi)
                ns[axis] = ne
            if not xs: continue
            ax.plot(xs, ms, AXIS_STYLE[axis], color=ALG_COLOR[alg], lw=1.6,
                    marker="o" if axis == "snapping" else "s", ms=4.5)
            ax.fill_between(xs, los, his, color=ALG_COLOR[alg], alpha=0.15, lw=0)
        s = df[(df.axis == "snapping") & (df.level == SNAP_EXTRA) & (df.algorithm == alg)]
        if not s.empty:
            m, lo, hi, ne = hboot(s, col, stat)
            if not np.isnan(m):
                ax.errorbar([SNAP_EXTRA], [m], yerr=[[m-lo],[hi-m]], fmt="o", mfc="white",
                            color=ALG_COLOR[alg], ms=5, lw=1.0, capsize=2, alpha=0.8)
    if zero: ax.axhline(0, color="black", lw=0.7, ls=":", alpha=0.6)
    sub = " · ".join(f"{k} n={v}" for k, v in sorted(ns.items()))
    ax.set_xlabel("Contamination level (%)"); ax.set_ylabel(ylab)
    ax.set_title(title, fontweight="bold", fontsize=12)
    if sub: ax.text(0.02, 0.02, sub, transform=ax.transAxes, fontsize=7, alpha=0.65)
    ax.grid(True, alpha=0.25); fig.tight_layout()
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"[{fname}] {col} stat={stat} {sub}")

CB = C[C.track == "combined"]
for col, title, fname, ylab in [
    ("auc_d",  "(a) AUC, paired delta", "fig2_auc", "Δ AUC vs benchmark"),
    ("tss_d",  "(b) TSS, paired delta", "fig2_tss", "Δ TSS vs benchmark"),
    ("boyce_d","(a) Boyce index, paired delta", "figS1_boyce", "Δ Boyce"),
    ("brier_d","(b) Brier score, paired delta", "figS1_brier", "Δ Brier")]:
    panel(CB, col, title, fname, ylab, zero=True)

panel(CB, "importance_spearman", "(Fig 3) Importance rank correlation",
      "fig3_spearman", "Spearman ρ vs benchmark", algs=["random_forest","xgboost"])
for col, title, fname, ylab, st in [
    ("niche_breadth_change","(a) Sampled niche breadth change","fig4_breadth","Dispersion change","mean"),
    ("niche_centroid_disp", "(b) Sampled centroid displacement","fig4_centroid","Displacement","mean"),
    ("schoener_d","(a) Schoener's D","fig5_schoener","Schoener's D","mean"),
    ("warren_i",  "(b) Warren's I","fig5_warren","Warren's I","mean"),
    ("range_area_pct_change_05","(c) Range-area change","fig5_range","Range-area change (%)","median")]:
    panel(CB, col, title, fname, ylab, stat=st, zero=(st=="median"))
for col, title, fname in [("CLI_shift","(S2) Climate-domain importance shift","figS2_cli"),
                          ("TOP_shift","(S3) Topography-domain importance shift","figS3_top"),
                          ("SOL_shift","(S4) Soil-domain importance shift","figS4_sol"),
                          ("LAC_shift","(S5) Land-cover-domain importance shift","figS5_lac")]:
    panel(CB, col, title, fname, "Importance-share difference", zero=True,
          algs=["random_forest","xgboost"])

# ---- Fig 6: T3 controlled displacement ----
T3f = [f for f in glob.glob("results/revision/t3*_*.csv") if "effective_dose" not in f]
if T3f:
    T3 = pd.concat([pd.read_csv(f) for f in T3f], ignore_index=True)
    T3 = T3[T3.dist_m > 0]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    dists = sorted(T3.dist_m.unique())
    cm = plt.cm.viridis(np.linspace(0.15, 0.85, len(dists)))
    for c, dm in zip(cm, dists):
        g = T3[T3.dist_m == dm].groupby("frac")["range_area_pct_change_05"].median()
        ax.plot(g.index, g.values, "-o", color=c, lw=1.6, ms=4.5, label=f"{int(dm)} m")
    ax.axhline(0, color="black", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("Records displaced (%)"); ax.set_ylabel("Range-area change (%)")
    ax.set_title("(Fig 6) Controlled displacement", fontweight="bold", fontsize=12)
    ax.legend(title="Distance", fontsize=8, title_fontsize=8, frameon=True)
    ax.grid(True, alpha=0.25); fig.tight_layout()
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT, f"fig6_displacement.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"[fig6_displacement] {T3.dist_m.nunique()} distances, {len(T3f)} files")

# ---- Fig 7: T1.5 stratified ----
T15f = glob.glob("results/revision/t1_5_*.csv")
if T15f:
    rows = []
    for f in T15f:
        stem = os.path.basename(f)[5:-4]
        for a in ("random_forest","xgboost","maxent"):
            if stem.endswith("_" + a):
                rows.append(pd.read_csv(f).assign(entity=stem[:-(len(a)+1)].replace("_"," "), algorithm=a))
                break
    T15 = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for mode, ls, mk in (("plain","-","o"), ("stratified","--","s")):
        g = T15[T15["mode"] == mode].groupby("level")["range_area_pct_change_05"].median()
        ax.plot(g.index, g.values, ls, color="#333333" if mode=="plain" else "#1f77b4",
                lw=1.8, marker=mk, ms=5, label=mode)
    ax.set_xlabel("Low-accuracy contamination (%)"); ax.set_ylabel("Range-area change (%)")
    ax.set_title("(Fig 7a) Strahler-stratified contamination", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); fig.tight_layout()
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT, f"fig7a_stratified.{ext}"), bbox_inches="tight")
    plt.close(fig)

    W = T15[T15.level == 20].groupby(["entity","mode"])["range_area_pct_change_05"].median().unstack()
    W = W.dropna().sort_values("plain")
    fig, ax = plt.subplots(figsize=(5.2, 0.42*len(W)+1.4))
    y = np.arange(len(W))
    ax.barh(y-0.19, W["plain"], height=0.36, color="#333333", label="plain")
    ax.barh(y+0.19, W["stratified"], height=0.36, color="#1f77b4", label="stratified")
    ax.set_yticks(y); ax.set_yticklabels(W.index, fontsize=8)
    ax.set_xlabel("Range-area change (%) at 20%")
    ax.set_title("(Fig 7b) Per entity", fontweight="bold", fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, axis="x", alpha=0.25); fig.tight_layout()
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT, f"fig7b_stratified_entity.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"[fig7] {T15.entity.nunique()} entities, {len(W)} in panel b")

# ---- legend ----
fig = plt.figure(figsize=(7, 0.8)); ax = fig.add_subplot(111); ax.axis("off")
h = [plt.Line2D([0],[0], color=ALG_COLOR[a], lw=2, label=ALG_LABEL[a]) for a in ALG_ORDER]
h += [plt.Line2D([0],[0], color="black", ls="-", marker="o", ms=4, label="snapping axis"),
      plt.Line2D([0],[0], color="black", ls="--", marker="s", ms=4, label="low-accuracy axis"),
      plt.Line2D([0],[0], color="black", ls="none", marker="o", mfc="white", ms=5,
                 label="snapping 10% (reduced panel)")]
ax.legend(handles=h, ncol=3, loc="center", frameon=True, fontsize=9)
for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT, f"legend.{ext}"), bbox_inches="tight")
plt.close(fig)
print("\nDONE ->", OUT, "|", len(glob.glob(os.path.join(OUT,"*.svg"))), "SVG")
