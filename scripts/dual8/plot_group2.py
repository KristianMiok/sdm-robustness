#!/usr/bin/env python3
"""
Group 2 figures S2-S5: domain-importance shift, DUAL-8, combined track, Grid B.
Signed difference in domain importance share (contaminated - benchmark),
Random Forest and XGBoost only (Maxent importance not extractable). Each domain
as its own vector file (PDF + SVG).

Usage:
  python3 plot_group2.py results/grid_b_merged/results_raw.csv figures_dual8
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv)>1 else "results/grid_b_merged/results_raw.csv"
OUT = sys.argv[2] if len(sys.argv)>2 else "figures_dual8"
os.makedirs(OUT, exist_ok=True)

ALG_COLOR = {"random_forest":"#1f77b4", "xgboost":"#d62728"}   # RF blue, XGB red (Maxent excluded)
AXIS_STYLE = {"snapping":"-", "lowacc":"--"}
ALG_ORDER = ["random_forest","xgboost"]
CORE_LEVELS = {"snapping":[1,2,5], "lowacc":[3,10,20]}
RNG = np.random.default_rng(20260416)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8,
                     "svg.fonttype":"none","pdf.fonttype":42})

d = pd.read_csv(CSV, low_memory=False)
d = d[(d.get("grid_id","B")=="B") & (d["status"]=="ok") & (d["track"]=="combined") & (d["entity_type"]=="DUAL")].copy()

def boot_ci(vals, n=2000):
    v=np.asarray(vals,float); v=v[~np.isnan(v)]
    if len(v)<2: return (np.nan,np.nan,np.nan)
    bs=RNG.choice(v,size=(n,len(v)),replace=True).mean(axis=1)
    return v.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

DOMAINS = [
 ("CLI_shift","(S2) Climate-domain importance shift","figS2_cli"),
 ("TOP_shift","(S3) Topography-domain importance shift","figS3_top"),
 ("SOL_shift","(S4) Soil-domain importance shift","figS4_sol"),
 ("LAC_shift","(S5) Land-cover-domain importance shift","figS5_lac"),
]

def panel(col, title, fname):
    if col not in d.columns:
        print(f"[SKIP] {col} not in columns"); return
    fig, ax = plt.subplots(figsize=(4.2,3.4))
    any_data=False
    for alg in ALG_ORDER:
        for axis, levels in CORE_LEVELS.items():
            xs, ms, los, his = [], [], [], []
            for lvl in levels:
                sub = d[(d["axis"]==axis)&(d["level"]==lvl)&(d["algorithm"]==alg)]
                vals = sub[col].dropna()
                if len(vals)<2: continue
                m,lo,hi = boot_ci(vals.values)
                xs.append(lvl); ms.append(m); los.append(lo); his.append(hi)
            if not xs: continue
            any_data=True
            ax.plot(xs, ms, AXIS_STYLE[axis], color=ALG_COLOR[alg], lw=1.6,
                    marker="o" if axis=="snapping" else "s", ms=4.5)
            ax.fill_between(xs, los, his, color=ALG_COLOR[alg], alpha=0.15, lw=0)
    ax.axhline(0, color="black", lw=0.7, ls=":", alpha=0.6)  # benchmark reference
    ax.set_xlabel("Contamination level (%)")
    ax.set_ylabel("Importance-share difference")
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if not any_data:
        print(f"[WARN] {fname}: no RF/XGB data for {col} (all NaN?)")
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT,f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig); print(f"[{fname}] wrote {fname}.pdf/.svg")

# legend for S2-S5 (RF/XGB only)
def legend_file():
    fig=plt.figure(figsize=(5,0.8)); ax=fig.add_subplot(111); ax.axis("off")
    h=[plt.Line2D([0],[0],color=ALG_COLOR["random_forest"],lw=2,label="Random Forest"),
       plt.Line2D([0],[0],color=ALG_COLOR["xgboost"],lw=2,label="XGBoost"),
       plt.Line2D([0],[0],color="black",ls="-",marker="o",ms=4,label="snapping axis"),
       plt.Line2D([0],[0],color="black",ls="--",marker="s",ms=4,label="low-accuracy axis")]
    ax.legend(handles=h,ncol=4,loc="center",frameon=True,fontsize=9)
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT,f"legend_domain.{ext}"),bbox_inches="tight")
    plt.close(fig); print("[legend_domain] wrote")

for col,title,fname in DOMAINS: panel(col,title,fname)
legend_file()
print("DONE ->", OUT)
