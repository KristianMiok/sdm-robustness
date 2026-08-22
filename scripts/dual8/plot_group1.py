#!/usr/bin/env python3
"""
Group 1 figures (Fig 2-5 + S1), DUAL-8, combined track, from the validated
grid_b_merged master + the 30-replicate benchmark envelope. Each sub-panel is
written as its own vector file (PDF + SVG), uniform styling, for uniform assembly.

Palette: Random Forest blue, XGBoost red, Maxent green.

Usage:
  python3 plot_group1.py \
    results/grid_b_merged/results_raw.csv \
    results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv \
    figures_dual8
"""
import sys, os, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv)>1 else "results/grid_b_merged/results_raw.csv"
ENV = sys.argv[2] if len(sys.argv)>2 else "results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv"
OUT = sys.argv[3] if len(sys.argv)>3 else "figures_dual8"
os.makedirs(OUT, exist_ok=True)

ALG_COLOR = {"random_forest":"#1f77b4", "xgboost":"#d62728", "maxent":"#2ca02c"}  # RF blue, XGB red, Mx green
ALG_LABEL = {"random_forest":"Random Forest", "xgboost":"XGBoost", "maxent":"Maxent"}
AXIS_STYLE = {"snapping":"-", "lowacc":"--"}
ALG_ORDER = ["random_forest","xgboost","maxent"]
CORE_LEVELS = {"snapping":[1,2,5], "lowacc":[3,10,20]}
RNG = np.random.default_rng(20260416)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8,
                     "svg.fonttype":"none","pdf.fonttype":42})

d = pd.read_csv(CSV, low_memory=False)
d = d[(d.get("grid_id","B")=="B") & (d["status"]=="ok") & (d["track"]=="combined") & (d["entity_type"]=="DUAL")].copy()
env = pd.read_csv(ENV, low_memory=False)
env = env[env["track"]=="combined"]
DUAL = sorted(d["entity"].unique())
# panel-wide benchmark reference per algorithm+metric (mean over available DUAL entities)
def bench_line(metric, alg):
    e = env[(env["metric_name"]==metric)&(env["algorithm"]==alg)&(env["entity"].isin(DUAL))]
    return e["benchmark_mean"].mean() if len(e) else None

def boot_ci(vals, n=2000):
    v=np.asarray(vals,float); v=v[~np.isnan(v)]
    if len(v)<2: return (np.nan,np.nan,np.nan)
    bs=RNG.choice(v,size=(n,len(v)),replace=True).mean(axis=1)
    return v.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

def panel(metric, title, fname, envelope=True):
    fig, ax = plt.subplots(figsize=(4.2,3.4))
    for alg in ALG_ORDER:
        for axis, levels in CORE_LEVELS.items():
            xs, ms, los, his = [], [], [], []
            for lvl in levels:
                sub = d[(d["axis"]==axis)&(d["level"]==lvl)&(d["algorithm"]==alg)]
                if sub.empty: continue
                m, lo, hi = boot_ci(sub[metric].values)
                xs.append(lvl); ms.append(m); los.append(lo); his.append(hi)
            if not xs: continue
            ax.plot(xs, ms, AXIS_STYLE[axis], color=ALG_COLOR[alg], lw=1.6,
                    marker="o" if axis=="snapping" else "s", ms=4.5)
            ax.fill_between(xs, los, his, color=ALG_COLOR[alg], alpha=0.15, lw=0)
        if envelope:
            b = bench_line(metric, alg)
            if b is not None:
                ax.axhline(b, color=ALG_COLOR[alg], lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("Contamination level (%)")
    ax.set_ylabel(title.split(") ",1)[-1] if ") " in title else title)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf","svg"):
        fig.savefig(os.path.join(OUT, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    # print panel-wide summary so it can be eyeballed vs the tables
    print(f"[{fname}] wrote {fname}.pdf/.svg")

# standalone legend
def legend_file():
    fig = plt.figure(figsize=(6,0.8)); ax=fig.add_subplot(111); ax.axis("off")
    h=[]
    for alg in ALG_ORDER:
        h.append(plt.Line2D([0],[0],color=ALG_COLOR[alg],lw=2,label=ALG_LABEL[alg]))
    h.append(plt.Line2D([0],[0],color="black",ls="-",marker="o",ms=4,label="snapping axis"))
    h.append(plt.Line2D([0],[0],color="black",ls="--",marker="s",ms=4,label="low-accuracy axis"))
    ax.legend(handles=h, ncol=5, loc="center", frameon=True, fontsize=9)
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT,f"legend.{ext}"),bbox_inches="tight")
    plt.close(fig); print("[legend] wrote legend.pdf/.svg")

PANELS = [
 ("auc","(a) AUC","fig2_auc",True),
 ("tss","(b) TSS","fig2_tss",True),
 ("importance_spearman","(a) Importance rank correlation","fig3_spearman",False),
 ("importance_jaccard_top10","(b) Top-10 Jaccard","fig3_jaccard",False),
 ("niche_breadth_change","(a) Niche breadth change","fig4_breadth",False),
 ("niche_centroid_disp","(b) Centroid displacement","fig4_centroid",False),
 ("schoener_d","(a) Schoener's D","fig5_schoener",False),
 ("warren_i","(b) Warren's I","fig5_warren",False),
 ("range_area_pct_change_05","(c) Range-area change (%)","fig5_range",False),
 ("boyce","(a) Boyce index","figS1_boyce",True),
 ("brier","(b) Brier score","figS1_brier",True),
]
for metric,title,fname,env_on in PANELS:
    if metric in d.columns: panel(metric,title,fname,env_on)
    else: print(f"[SKIP] column '{metric}' not found")
legend_file()
print("DONE ->", OUT)
