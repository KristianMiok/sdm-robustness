"""Per-entity first-order depletion vs range inflation at low-accuracy 20%."""
import glob, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from scipy.stats import spearmanr
from sdm_robustness.execution import runner as R
pd.set_option("display.width", 220)
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")
rows = []
for e in P.entity:
    d = R._prepare_entity_data(M, e)
    b, a = d["benchmark"], d["accessible_area"]
    bs = float((b.strahler == 1).mean()*100); as_ = float((a.strahler == 1).mean()*100)
    rows.append(dict(entity=e, bench_str1_pct=round(bs,2), acc_str1_pct=round(as_,2),
                     depletion_pp=round(as_-bs,2),
                     depletion_ratio=round(bs/as_,3) if as_ else np.nan))
T = pd.DataFrame(rows)
D = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in
               glob.glob("results/campaign/*/results_raw.parquet")], ignore_index=True)
C = D[(D.status=="ok") & (D.axis=="lowacc") & (D.level==20) & (D.track=="combined")]
INF = (C.groupby(["algorithm","entity"])["range_area_pct_change_05"].median()
        .unstack("algorithm").round(2))
T = T.merge(INF, left_on="entity", right_index=True, how="left")
T.to_csv("results/revision/tables/depletion_vs_inflation.csv", index=False)
print(T.to_string(index=False))
print("\n=== Spearman: depletion_pp vs inflacija ===")
res = []
for alg in ("random_forest","xgboost","maxent"):
    if alg not in T.columns: continue
    s = T[["depletion_pp", alg]].dropna()
    if len(s) < 3: print(f"  {alg:<14} n={len(s)} premalo"); continue
    r = spearmanr(s.depletion_pp, s[alg])
    res.append(dict(algorithm=alg, n=len(s), rho=round(r.statistic,3), p=round(r.pvalue,4)))
    print(f"  {alg:<14} n={len(s):<3} rho={r.statistic:+.3f}  p={r.pvalue:.4f}")
pd.DataFrame(res).to_csv("results/revision/tables/depletion_spearman.csv", index=False)
print("\n=== slaganje algoritama na rangiranju entiteta ===")
algs = [a for a in ("random_forest","xgboost","maxent") if a in T.columns]
for i, x in enumerate(algs):
    for y in algs[i+1:]:
        s = T[[x, y]].dropna()
        if len(s) < 3: continue
        r = spearmanr(s[x], s[y])
        print(f"  {x} vs {y:<14} n={len(s):<3} rho={r.statistic:+.3f}  p={r.pvalue:.4f}")
