#!/usr/bin/env python3
"""
8-DUAL recompute of Tables S4 (envelope classification) and S5 (spatial metrics)
for Ecography Block 1.

Envelope (benchmark_mean, benchmark_sd) is read from the dedicated 30-replicate
stability file; contamination replicates are read from the merged grid-B file.

Usage:
  python3 aggregate_8dual.py \
    results/grid_b_merged/results_raw.csv \
    results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv
"""
import sys, pandas as pd, numpy as np
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 200)

CSV = sys.argv[1] if len(sys.argv) > 1 else "results/grid_b_merged/results_raw.csv"
ENV = sys.argv[2] if len(sys.argv) > 2 else "results/task5c_benchmark_stability_merged/task5c_benchmark_stability_partial.csv"
TRACK = "combined"
CORE_LEVELS = {"snapping":[1,2,5], "lowacc":[3,10,20]}
ALG_ORDER = ["random_forest","xgboost","maxent"]

print(">>> aggregate_8dual VERSION: v5-report-n_env <<<")

df = pd.read_csv(CSV, low_memory=False)
env_raw = pd.read_csv(ENV, low_memory=False)

# ---- 0. STRUCTURE ----
print("="*78); print("STRUCTURE CHECK"); print("="*78)
print("status values :", dict(df["status"].value_counts()))
d = df[(df.get("grid_id","B")=="B") & (df["status"]=="ok") & (df["track"]==TRACK)].copy()
print("axis x level (grid B, ok, combined):"); print(d.groupby(["axis","level"]).size().to_string())
DUAL = sorted(d.loc[d["entity_type"]=="DUAL","entity"].unique())
ALL  = sorted(d["entity"].unique())
print(f"\nDUAL (n={len(DUAL)}) / ALL (n={len(ALL)})")

# envelope: long format -> dict keyed by (entity, algorithm, metric)
e = env_raw[env_raw["track"]==TRACK]
print("envelope metrics available:", sorted(e["metric_name"].unique()))
print("envelope entities:", e["entity"].nunique(), "| replicates:", sorted(e["n_replicates_ok"].unique()))
ENVMAP = {(r.entity, r.algorithm, r.metric_name): (r.benchmark_mean, r.benchmark_sd)
          for r in e.itertuples()}

def s4_like(entities, metric="auc"):
    rows=[]
    for axis,levels in CORE_LEVELS.items():
        for lvl in levels:
            for alg in ALG_ORDER:
                sub = d[(d["axis"]==axis)&(d["level"]==lvl)&(d["algorithm"]==alg)&(d["entity"].isin(entities))]
                if sub.empty: continue
                # PER-ENTITY exceedance, then average across entities (matches published S4).
                # mean/median/iqr z computed on pooled replicates (equivalent for balanced cells).
                p2=[]; p3=[]; zpool=[]; used=[]; missing=[]
                for ent,grp in sub.groupby("entity"):
                    key=(ent,alg,metric)
                    if key not in ENVMAP:
                        missing.append(ent); continue
                    bm,bs=ENVMAP[key]
                    if not bs or bs<=0:
                        missing.append(ent); continue
                    # signed so that DEGRADATION is POSITIVE (matches published S4)
                    z=(bm-grp[metric])/bs
                    # ONE-SIDED: count only replicates degraded beyond the envelope
                    # (z>2 / z>3 in degradation direction); improvements (z<0) not counted.
                    p2.append(100*np.mean(z>2))
                    p3.append(100*np.mean(z>3))
                    zpool.extend(z.tolist())
                    used.append(ent)
                if not p2: continue
                zpool=np.array(zpool)
                rows.append(dict(axis=axis,level=lvl,algorithm=alg,
                    pct_out_2sd=round(float(np.mean(p2)),1),
                    pct_out_3sd=round(float(np.mean(p3)),1),
                    mean_z=round(zpool.mean(),2),
                    median_z=round(np.median(zpool),2),
                    iqr_z=round(np.percentile(zpool,75)-np.percentile(zpool,25),2),
                    n_ent=sub["entity"].nunique(),
                    n_env=len(used),
                    missing=";".join(sorted(set(missing))) if missing else ""))
    return pd.DataFrame(rows)

def s5_like(entities):
    rows=[]
    for axis,levels in CORE_LEVELS.items():
        for lvl in levels:
            for alg in ALG_ORDER:
                sub=d[(d["axis"]==axis)&(d["level"]==lvl)&(d["algorithm"]==alg)&(d["entity"].isin(entities))]
                if sub.empty: continue
                rows.append(dict(axis=axis,level=lvl,algorithm=alg,
                    schoener_d=round(sub["schoener_d"].mean(),3),
                    warren_i=round(sub["warren_i"].mean(),3),
                    range_area_pct=round(sub["range_area_pct_change_05"].mean(),1),
                    n_ent=sub["entity"].nunique()))
    return pd.DataFrame(rows)

def null_exceedance(metric="auc"):
    # benchmark reps (from grid file) vs their own stability envelope -> expect ~5%
    out=[]
    b_all = d[d["axis"]=="benchmark"]
    for alg in ALG_ORDER:
        b=b_all[b_all["algorithm"]==alg]; z=[]
        for ent,grp in b.groupby("entity"):
            key=(ent,alg,metric)
            if key not in ENVMAP: continue
            bm,bs=ENVMAP[key]
            if bs and bs>0: z.extend(((bm-grp[metric])/bs).tolist())
        z=np.array(z)
        out.append(dict(algorithm=alg,
            null_pct_out_2sd=(round(100*np.mean(z>2),1) if len(z) else float('nan')),
            n=len(z)))
    return pd.DataFrame(out)

print("\n"+"="*78); print("S4  —  CURRENT composition (validation: must match published S4)"); print("="*78)
print(s4_like(ALL).to_string(index=False))
print("\n"+"="*78); print("S4  —  8-DUAL ONLY, both axes (Block 1 deliverable)"); print("="*78)
print(s4_like(DUAL).to_string(index=False))
print("\n"+"="*78); print("S5  —  CURRENT composition (validation: must match published S5)"); print("="*78)
print(s5_like(ALL).to_string(index=False))
print("\n"+"="*78); print("S5  —  8-DUAL ONLY, both axes"); print("="*78)
print(s5_like(DUAL).to_string(index=False))
print("\n"+"="*78); print("(c) BENCHMARK NULL exceedance (~5% expected)"); print("="*78)
print(null_exceedance().to_string(index=False))
