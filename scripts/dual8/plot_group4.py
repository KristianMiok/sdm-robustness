#!/usr/bin/env python3
"""
Group 4 figures S8-S9 (extended Grid A), from the clean grid_a_merged master.

S8 — extended aggregate AUC vs contamination (0-50%), combined track:
     figS8_upper : 8 DUAL entities (snapping solid + low-accuracy dashed)
     figS8_lower : 5 SNAPPING-ONLY entities (snapping only)
S9 — Astacus astacus per-track AUC vs contamination (0-50%), 3 tracks x 2 axes,
     each sub-panel drawn against the SHARED Grid B benchmark line (from the
     30-replicate envelope) instead of the axis-specific level-0 point.

Usage:
  python3 plot_group4.py \
    results/grid_a_merged/results_raw.csv \
    results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv \
    figures_dual8
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

GA  = sys.argv[1] if len(sys.argv)>1 else "results/grid_a_merged/results_raw.csv"
ENV = sys.argv[2] if len(sys.argv)>2 else "results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv"
OUT = sys.argv[3] if len(sys.argv)>3 else "figures_dual8"
os.makedirs(OUT, exist_ok=True)
ALG_COLOR={"random_forest":"#1f77b4","xgboost":"#d62728","maxent":"#2ca02c"}
ALG_LABEL={"random_forest":"Random Forest","xgboost":"XGBoost","maxent":"Maxent"}
AXIS_STYLE={"snapping":"-","lowacc":"--"}; ALG_ORDER=["random_forest","xgboost","maxent"]
RNG=np.random.default_rng(20260426)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.linewidth":0.8,
                     "svg.fonttype":"none","pdf.fonttype":42})

d=pd.read_csv(GA,low_memory=False)
d=d[d["status"]=="ok"].copy()
env=pd.read_csv(ENV,low_memory=False)
DUAL_LIST=['Astacus astacus','Pontastacus leptodactylus (pooled)','Austropotamobius torrentium (pooled)','Austropotamobius fulcisianus (pooled)','Procambarus clarkii (native)','Procambarus clarkii (alien)','Pacifastacus leniusculus (alien)','Faxonius limosus (alien)']
# entity-type membership: prefer column, else name list
def is_dual(df): return df["entity_type"]=="DUAL" if "entity_type" in df.columns else df["entity"].isin(DUAL_LIST)

def boot(vals,n=2000):
    v=np.asarray(vals,float); v=v[~np.isnan(v)]
    if len(v)<2: return (np.nan,np.nan,np.nan)
    bs=RNG.choice(v,size=(n,len(v)),replace=True).mean(axis=1)
    return v.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)

def agg_panel(df, axes, title, fname):
    fig,ax=plt.subplots(figsize=(5.2,3.6))
    for alg in ALG_ORDER:
        for axis in axes:
            sub_a=df[(df["axis"]==axis)&(df["algorithm"]==alg)&(df["track"]=="combined")]
            levels=sorted(sub_a["level"].unique())
            xs,ms,los,his=[],[],[],[]
            for lvl in levels:
                s=sub_a[sub_a["level"]==lvl]
                if s.empty: continue
                m,lo,hi=boot(s["auc"].values); xs.append(lvl); ms.append(m); los.append(lo); his.append(hi)
            if not xs: continue
            ax.plot(xs,ms,AXIS_STYLE[axis],color=ALG_COLOR[alg],lw=1.6,marker="o" if axis=="snapping" else "s",ms=4)
            ax.fill_between(xs,los,his,color=ALG_COLOR[alg],alpha=0.15,lw=0)
    ax.set_xlabel("Contamination level (%)"); ax.set_ylabel("AUC")
    ax.set_title(title,fontweight="bold",fontsize=12); ax.grid(True,alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT,f"{fname}.{ext}"),bbox_inches="tight")
    plt.close(fig); print(f"[{fname}] wrote")

# ---- S8 ----
dual=d[is_dual(d)]; snap=d[~is_dual(d)]
n_dual=dual["entity"].nunique(); n_snap=snap["entity"].nunique()
agg_panel(dual, ["snapping","lowacc"], f"(S8 upper) {n_dual} dual-axis entities", "figS8_upper")
agg_panel(snap, ["snapping"],          f"(S8 lower) {n_snap} snapping-only entities", "figS8_lower")

# ---- S9: A. astacus, per track x axis, shared Grid B benchmark line ----
aa = d[d["entity"].str.contains("Astacus astacus", case=False, na=False)]
ea=env[env["entity"].str.contains("Astacus astacus",case=False,na=False)&(env["metric_name"]=="auc")]
print("S9 envelope tracks for A. astacus:", sorted(ea["track"].unique()) if len(ea) else "NONE -> benchmark lines will not draw")
print("S9 entity match:", sorted(aa["entity"].unique()))
def bench(track, alg):
    e=env[(env["metric_name"]=="auc")&(env["track"]==track)&(env["algorithm"]==alg)&(env["entity"].str.contains("Astacus astacus",case=False,na=False))]
    return e["benchmark_mean"].mean() if len(e) else None
TRACKS=["local_only","upstream_only","combined"]
for track in TRACKS:
    for axis in ["snapping","lowacc"]:
        fig,ax=plt.subplots(figsize=(4.0,3.2))
        drew=False
        for alg in ALG_ORDER:
            sub=aa[(aa["track"]==track)&(aa["axis"]==axis)&(aa["algorithm"]==alg)]
            # drop level 0 (axis-specific baseline) -> use shared benchmark line instead
            sub=sub[sub["level"]>0]
            levels=sorted(sub["level"].unique())
            xs,ms=[],[]
            for lvl in levels:
                s=sub[sub["level"]==lvl]
                if s.empty: continue
                xs.append(lvl); ms.append(s["auc"].mean())
            if xs:
                ax.plot(xs,ms,AXIS_STYLE[axis],color=ALG_COLOR[alg],lw=1.6,marker="o" if axis=="snapping" else "s",ms=4); drew=True
            b=bench(track,alg)
            if b is not None: ax.axhline(b,color=ALG_COLOR[alg],lw=0.8,ls=":",alpha=0.7)
        ax.set_xlabel("Contamination level (%)"); ax.set_ylabel("AUC")
        ax.set_title(f"(S9) A. astacus — {track}, {axis}",fontweight="bold",fontsize=10)
        ax.grid(True,alpha=0.25); fig.tight_layout()
        fn=f"figS9_{track}_{axis}"
        for ext in ("pdf","svg"): fig.savefig(os.path.join(OUT,f"{fn}.{ext}"),bbox_inches="tight")
        plt.close(fig); print(f"[{fn}] wrote{'' if drew else '  (NO DATA)'}")
print("DONE ->",OUT)
