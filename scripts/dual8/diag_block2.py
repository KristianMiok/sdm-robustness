import sys, glob, os, pandas as pd, numpy as np
GB=sys.argv[1] if len(sys.argv)>1 else "results/grid_b_merged/results_raw.csv"
d=pd.read_csv(GB,low_memory=False)
print("="*70); print("(d) REPLICATES PER CELL — Grid B"); print("="*70)
gb=d[d["status"]=="ok"]
rc=gb.groupby(["axis","level","algorithm","entity"]).size()
print("Grid B replicate counts per cell: min=%d max=%d (unique=%s)"%(rc.min(),rc.max(),sorted(rc.unique())))
print("=> if all 30, Grid B is uniform; the '30-50' label must come from the extended grid")

print("\n"+"="*70); print("(g) TOTAL MODEL RUNS"); print("="*70)
print("Grid B rows total=%d | ok=%d | error=%d"%(len(d),(d['status']=='ok').sum(),(d['status']!='ok').sum()))
# look for a merged Grid A file
cands=glob.glob("results/**/*.csv",recursive=True)
gridA=[c for c in cands if "task5_execution" in c or "grid_a" in c.lower()]
print("Grid A merged csv candidates:", gridA[:5] if gridA else "none found (task5_execution likely per-chunk parquet)")
pq=glob.glob("results/task5_execution/**/*.parquet",recursive=True)
print("task5_execution parquet files:", len(pq))

print("\n"+"="*70); print("(f) A. astacus benchmark per axis (RF combined)"); print("="*70)
a=gb[(gb["entity"]=="Astacus astacus")&(gb["algorithm"]=="random_forest")&(gb["track"]=="combined")]
print(a.groupby("axis")["auc"].agg(["mean","std","count"]).to_string())
print("=> Grid B has a single 'benchmark' axis; if S9 shows two different 0% lines per axis, S9 is the extended grid or plots axis-specific level-0")

print("\n"+"="*70); print("(c) RAW BENCHMARK PARQUET READ ATTEMPT"); print("="*70)
f=glob.glob("results/task5c_benchmark_stability_array/*maxent_combined/benchmark_stability.parquet")
if f:
    for eng in ["pyarrow","fastparquet"]:
        try:
            t=pd.read_parquet(f[0],engine=eng)
            print(f"  engine={eng}: OK shape={t.shape} cols={list(t.columns)[:8]}")
            break
        except Exception as e:
            print(f"  engine={eng}: FAIL {str(e)[:80]}")
else:
    print("  no maxent_combined benchmark_stability.parquet found")
