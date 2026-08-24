"""Aggregate the per-entity variance pilots: does pairing help where the
contamination pool is large enough to contribute variance of its own?"""
import glob, re
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False,
                usecols=["subc_id","distance_m","Accuracy","Status","Crayfish_scientific_name"])
P = pd.read_csv("config/final_panel.csv")
pool_n = {}
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    pool_n[r.entity] = dict(bench=len(b),
        lowacc=len(R._dedup_by_subc(d[~hi])),
        snap=len(R._dedup_by_subc(d[hi & (d.distance_m>200) & (d.distance_m<=1000)])))

rows = []
for f in sorted(glob.glob("results/revision/var_*.csv")):
    D = pd.read_csv(f)
    if D.empty: continue
    m = re.match(r".*/var_(.+)_(random_forest|xgboost|maxent)_(lowacc|snapping)(\d+)\.csv", f)
    if not m: continue
    slug, alg, axis, lv = m.groups()
    ent = next((e for e in pool_n if re.sub(r"[^A-Za-z0-9_]", "", e.replace(" ", "_")) == slug), slug)
    g = D.groupby("arm").range_pct
    sd, mu = g.std(), g.mean()
    pi = pool_n.get(ent, {})
    need = int(round(pi.get("bench", 0) * int(lv) / 100)) if pi else np.nan
    avail = pi.get(axis, np.nan) if pi else np.nan
    rows.append(dict(entity=ent, alg=alg, axis=axis, level=int(lv), n=len(D)//4,
        bench=pi.get("bench"), pool=avail, need=need,
        pool_slack=round(avail/need, 2) if need else np.nan,
        sd_bg=sd.get("A_bg_only"), sd_model=sd.get("D_model_only"),
        sd_unpaired=sd.get("C_unpaired"), sd_paired=sd.get("B_paired"),
        gain=sd.get("C_unpaired")/sd.get("B_paired") if sd.get("B_paired") else np.nan,
        eff_paired=mu.get("B_paired"), eff_unpaired=mu.get("C_unpaired")))
T = pd.DataFrame(rows).sort_values(["axis","alg","entity"])
T.to_csv("results/revision/t5_pilot_summary.csv", index=False)
print(T.round(3).to_string(index=False))

print("\n=== dobitak od uparivanja vs zaliha pool-a ===")
from scipy.stats import spearmanr
for alg in T.alg.unique():
    s = T[(T.alg == alg)].dropna(subset=["gain","pool_slack"])
    if len(s) > 3:
        print(f"  {alg:<14} rho(pool_slack, gain) = {spearmanr(s.pool_slack, s.gain).statistic:>6.3f}"
              f"   gain median {s.gain.median():.1f}x   n={len(s)}")

print("\n=== efekat po entitetu, upareno, sa SE ===")
T["se"] = T.sd_paired / np.sqrt(T.n)
T["z"] = T.eff_paired / T.se
print(T[["entity","alg","eff_paired","se","z","eff_unpaired","sd_unpaired"]].round(3).to_string(index=False))
print("\n  celija gde |z| > 2:", int((T.z.abs() > 2).sum()), "od", len(T))
