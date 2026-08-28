"""T1.5: low-accuracy contamination stratified by stream order.

Same dose, same pool, but each substituted record comes from the stream order
of the record it replaces. If range inflation collapses under stratification,
the published 29-36% is a stream-order composition effect rather than an effect
of locational accuracy.
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import fit_cv_cell, contaminate_presence_set
from sdm_robustness.utils.repro import derive_seed

MASTER = 20260426
ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True)
ap.add_argument("--alg", default="random_forest")
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--levels", type=int, nargs="+", default=[3, 10, 20])
a = ap.parse_args()

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
p = R._prepare_entity_data(M, a.entity)
B, pool, acc = p["benchmark"], p["lowacc_pool"], p["accessible_area"]
if not len(pool):
    raise SystemExit(f"{a.entity}: nema lowacc pool")
fm = assign_basin_folds(B.basin_id, n_splits=5, looo_threshold=15)
print(f"{a.entity} [{a.alg}]: benchmark {len(B)} | pool {len(pool)}", flush=True)

rows = []
for rep in range(a.reps):
    bg = derive_seed(MASTER, a.entity, a.alg, "combined", rep)
    base = fit_cv_cell(benchmark=B, contamination_pool=pool, accessible_area=acc,
                       entity=a.entity, algorithm=a.alg, track="combined",
                       axis="benchmark", level=0, replicate=rep,
                       seed=derive_seed(MASTER, a.entity, "benchmark", 0, rep),
                       n_experiment=len(B), fold_map=fm, bg_seed=bg,
                       return_artifacts=True)
    surf = base.pop("_run_surface", None); base.pop("_run_importance", None)
    for lv in a.levels:
        for mode, st in (("plain", None), ("stratified", "strahler")):
            s = derive_seed(MASTER, a.entity, "lowacc", lv, rep)
            C = contaminate_presence_set(benchmark=B, contamination_pool=pool,
                                         level_pct=lv, seed=s, stratify_by=st)
            r = fit_cv_cell(benchmark=C, contamination_pool=C, accessible_area=acc,
                            entity=a.entity, algorithm=a.alg, track="combined",
                            axis=f"lowacc_{mode}", level=lv, replicate=rep,
                            seed=s, n_experiment=len(C), fold_map=fm, bg_seed=bg,
                            benchmark_surface=surf, return_artifacts=True)
            r.pop("_run_surface", None); r.pop("_run_importance", None)
            rows.append(dict(rep=rep, level=lv, mode=mode,
                             d_str1=float((C.strahler == 1).mean()*100 -
                                          (B.strahler == 1).mean()*100),
                             n=len(C),
                             **{k: r.get(k) for k in
                                ("auc","cv_auc","tss","range_area_pct_change_05",
                                 "range_area_pct_change_maxsss","suitability_mad",
                                 "schoener_d","env_centroid_disp") if k in r}))
    print(f"  rep {rep+1}/{a.reps}", flush=True)

T = pd.DataFrame(rows)
slug = "".join(c for c in a.entity.replace(" ", "_") if c.isalnum() or c == "_")
Path("results/revision").mkdir(parents=True, exist_ok=True)
T.to_csv(f"results/revision/t1_5_{slug}_{a.alg}.csv", index=False)
print(f"\n=== {a.entity} [{a.alg}] ===")
g = T.groupby(["level","mode"]).agg(
    d_str1=("d_str1","mean"), range05=("range_area_pct_change_05","mean"),
    sd=("range_area_pct_change_05","std"), mad=("suitability_mad","mean"))
print(g.round(3).to_string())
