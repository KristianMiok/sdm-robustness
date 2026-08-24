"""How much does contamination actually change the fold structure?"""
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import contaminate_presence_set
from sdm_robustness.utils.repro import derive_seed

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False,
                usecols=["subc_id","basin_id","strahler","distance_m","Accuracy",
                         "Status","Crayfish_scientific_name"])
P = pd.read_csv("config/final_panel.csv")
MASTER = 20260416

print(f"{'entity':<38}{'osa':<9}{'L':>4}{'novih basena':>13}{'%zapisa novi':>13}"
      f"{'basena promenilo fold':>23}{'LOO flip':>10}")
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    pools = {"snapping": R._dedup_by_subc(d[hi & (d.distance_m > 200) & (d.distance_m <= 1000)]),
             "lowacc": R._dedup_by_subc(d[~hi])}
    f_clean = assign_basin_folds(b.basin_id, n_splits=5, looo_threshold=15)
    loo_clean = b.basin_id.nunique() < 15
    for ax, pool in pools.items():
        if not len(pool): continue
        if ax == "snapping" and int(r.run_snapping) != 1: continue
        if ax == "lowacc" and int(r.run_lowacc) != 1: continue
        for lv in ((1,2,5) if ax == "snapping" else (3,10,20)):
            need = int(round(len(b)*lv/100))
            if need > len(pool): continue
            s = derive_seed(MASTER, r.entity, ax, lv, 0)
            c = contaminate_presence_set(benchmark=b, contamination_pool=pool,
                                         level_pct=lv, seed=s)
            newb = set(c.basin_id.astype(str)) - set(b.basin_id.astype(str))
            pct_new = c.basin_id.astype(str).isin(newb).mean()*100
            f_cont = assign_basin_folds(c.basin_id, n_splits=5, looo_threshold=15)
            shared = set(f_clean) & set(f_cont)
            moved = sum(f_clean[k] != f_cont[k] for k in shared)
            loo_cont = c.basin_id.nunique() < 15
            print(f"{r.entity:<38}{ax:<9}{lv:>4}{len(newb):>13}{pct_new:>12.1f}%"
                  f"{f'{moved}/{len(shared)}':>23}{('DA' if loo_clean!=loo_cont else ''):>10}")
