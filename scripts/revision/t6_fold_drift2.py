"""Fold drift measured label-invariantly: adjusted Rand index plus
best-case agreement after optimal fold relabelling."""
import numpy as np, pandas as pd
from itertools import permutations
from sklearn.metrics import adjusted_rand_score
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import contaminate_presence_set
from sdm_robustness.utils.repro import derive_seed

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False,
                usecols=["subc_id","basin_id","strahler","distance_m","Accuracy",
                         "Status","Crayfish_scientific_name"])
P = pd.read_csv("config/final_panel.csv")
MASTER = 20260416

def best_agree(a, b):
    """Max fraction agreeing over all relabellings of b (<=6 folds)."""
    la, lb = sorted(set(a)), sorted(set(b))
    if len(lb) > 6 or len(la) > 6: return np.nan
    best = 0.0
    for p in permutations(range(len(lb))):
        mp = {lb[i]: la[p[i]] if p[i] < len(la) else -1 for i in range(len(lb))}
        best = max(best, np.mean([x == mp[y] for x, y in zip(a, b)]))
    return best

rows = []
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    if b.basin_id.nunique() < 15:
        continue  # LOO regime, folds are basins by definition
    f0 = assign_basin_folds(b.basin_id, n_splits=5, looo_threshold=15)
    pools = {"snapping": R._dedup_by_subc(d[hi & (d.distance_m>200) & (d.distance_m<=1000)]),
             "lowacc": R._dedup_by_subc(d[~hi])}
    for ax, pool in pools.items():
        if not len(pool): continue
        if ax == "snapping" and int(r.run_snapping) != 1: continue
        if ax == "lowacc" and int(r.run_lowacc) != 1: continue
        for lv in ((1,2,5) if ax=="snapping" else (3,10,20)):
            if int(round(len(b)*lv/100)) > len(pool): continue
            # two independent replicates of the SAME cell, plus clean-vs-contaminated
            fs = []
            for rep in (0,1):
                s = derive_seed(MASTER, r.entity, ax, lv, rep)
                c = contaminate_presence_set(benchmark=b, contamination_pool=pool,
                                            level_pct=lv, seed=s)
                fs.append(assign_basin_folds(c.basin_id, n_splits=5, looo_threshold=15))
            sh = sorted(set(f0) & set(fs[0]))
            a0, a1 = [f0[k] for k in sh], [fs[0][k] for k in sh]
            sh2 = sorted(set(fs[0]) & set(fs[1]))
            r0, r1 = [fs[0][k] for k in sh2], [fs[1][k] for k in sh2]
            rows.append(dict(entity=r.entity, axis=ax, level=lv, n_basins=len(sh),
                ari_vs_clean=adjusted_rand_score(a0, a1),
                agree_vs_clean=best_agree(a0, a1),
                ari_rep_vs_rep=adjusted_rand_score(r0, r1),
                agree_rep_vs_rep=best_agree(r0, r1)))
T = pd.DataFrame(rows)
T.to_csv("results/revision/t6_fold_drift.csv", index=False)
print(T.round(3).to_string(index=False))
print("\n=== prosek po osi i nivou ===")
print(T.groupby(["axis","level"])[["ari_vs_clean","agree_vs_clean",
      "ari_rep_vs_rep","agree_rep_vs_rep"]].mean().round(3).to_string())
