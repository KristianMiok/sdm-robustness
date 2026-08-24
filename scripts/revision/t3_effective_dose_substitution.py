"""Effective dose on the substitution axes.

Predictors are a subc_id lookup, so a substituted record only perturbs the
model if it brings a subcatchment the benchmark did not have. Substitution
also deletes: n_replace benchmark records leave, taking their subcatchments.
Three quantities per cell, averaged over replicates:

  p_new     fraction of the contaminated set whose subc_id is not in the benchmark
  p_dup     fraction of substituted records duplicating a retained subc_id
  p_lost    fraction of benchmark subcatchments absent from the contaminated set
  jaccard   set overlap of subc_ids, benchmark vs contaminated
"""
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R
from sdm_robustness.pipeline.core import contaminate_presence_set
from sdm_robustness.utils.repro import derive_seed

REPS, MASTER_SEED = 20, 20260416
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False,
                usecols=["subc_id","basin_id","distance_m","Accuracy","Status",
                         "Crayfish_scientific_name"])
P = pd.read_csv("config/final_panel.csv")

rows = []
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    pools = {"snapping": R._dedup_by_subc(d[hi & (d.distance_m>200) & (d.distance_m<=1000)]),
             "lowacc":   R._dedup_by_subc(d[~hi])}
    B = set(b.subc_id.astype(str))
    for axis, pool in pools.items():
        if axis == "snapping" and int(r.run_snapping) != 1: continue
        if axis == "lowacc" and int(r.run_lowacc) != 1: continue
        if not len(pool): continue
        Pl = set(pool.subc_id.astype(str))
        overlap = len(B & Pl) / len(Pl)
        for lv in ((1,2,5) if axis=="snapping" else (3,10,20)):
            if int(round(len(b)*lv/100)) > len(pool): continue
            acc = []
            for rep in range(REPS):
                s = derive_seed(MASTER_SEED, r.entity, axis, lv, rep)
                c = contaminate_presence_set(benchmark=b, contamination_pool=pool,
                                             level_pct=lv, seed=s)
                n_rep = int(round(len(b)*lv/100))
                kept_ids = c.subc_id.astype(str).iloc[:len(c)-n_rep]
                sub_ids = c.subc_id.astype(str).iloc[len(c)-n_rep:]
                C = set(c.subc_id.astype(str))
                acc.append(dict(
                    p_new=len([x for x in sub_ids if x not in B]) / len(c),
                    p_dup=np.mean([x in set(kept_ids) for x in sub_ids]),
                    p_lost=len(B - C) / len(B),
                    jaccard=len(B & C) / len(B | C)))
            a = pd.DataFrame(acc).mean()
            rows.append(dict(entity=r.entity, axis=axis, nominal=lv,
                             bench=len(b), pool=len(pool),
                             pool_overlap=round(overlap,3),
                             p_new=a.p_new*100, p_dup=a.p_dup*100,
                             p_lost=a.p_lost*100, jaccard=a.jaccard))
T = pd.DataFrame(rows)
T.to_csv("results/revision/t3_effective_dose_substitution.csv", index=False)

for ax in ("lowacc","snapping"):
    s = T[T.axis == ax]
    if s.empty: continue
    print(f"\n=== {ax} ===")
    print(s[["entity","nominal","bench","pool","pool_overlap",
             "p_new","p_dup","p_lost","jaccard"]].round(2).to_string(index=False))
    print("\n  nominalna -> efektivna (p_new), medijan panela:")
    for lv in sorted(s.nominal.unique()):
        q = s[s.nominal == lv]
        print(f"    {lv:>3}%  ->  {q.p_new.median():5.2f}%"
              f"   (raspon {q.p_new.min():.2f}-{q.p_new.max():.2f})"
              f"   izgubljeno {q.p_lost.median():5.2f}%")

print("\n=== poredjenje sa pomeranjem ===")
try:
    D = pd.read_csv("results/revision/t3_effective_dose.csv")
    med = D.groupby("dist_m").p_change.median()
    print("  pomeranje, efektivna doza pri nominalnih 100%:")
    for dm, p in med.items(): print(f"    {int(dm):>5} m -> {p*100:5.2f}%")
except FileNotFoundError:
    pass
