"""Nominal vs effective contamination dose under displacement.

A displaced record only perturbs the model if it lands in a different
subcatchment - predictors are a subc_id lookup. So the effective dose is
nominal x P(subcatchment changes | displacement distance), and it is
distance-limited in a way substitution is not.
"""
import re
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sdm_robustness.execution import runner as R

Rk = 6371008.8
DIST = (100, 250, 500, 1000, 2000)
NOM = (3, 10, 20, 50, 100)

hdr = pd.read_csv("data/combined_data_true_master.csv", nrows=0).columns.tolist()
pred = [c for c in hdr if re.match(r"^[lu]_", c)]
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")
Z = M[pred].to_numpy(dtype=np.float32)
sd = np.nanstd(Z, 0); sd[sd == 0] = 1
Z = (Z - np.nanmean(Z, 0)) / sd
ref = None

rows = []
for _, r in P.iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    cand = M[M.basin_id.astype(str).isin(set(b.basin_id.astype(str)))]
    cand = cand.sort_values("subc_id").drop_duplicates("subc_id")
    if len(cand) < 50: continue
    T = BallTree(np.radians(cand[["lat_snap", "long_snap"]].to_numpy()), metric="haversine")
    X = np.radians(b[["lat_snap", "long_snap"]].to_numpy())
    own_sc = b.subc_id.to_numpy()
    own_ix = b.index.to_numpy()
    cand_sc = cand.subc_id.to_numpy()
    cand_ix = cand.index.to_numpy()
    for dm in DIST:
        idx = T.query_radius(X, r=dm / Rk)
        pch, denv = [], []
        for k, ii in enumerate(idx):
            if len(ii) == 0:
                pch.append(0.0); continue
            diff = cand_sc[ii] != own_sc[k]
            pch.append(diff.mean())
            if diff.any():
                denv.append(np.nanmean(np.abs(Z[cand_ix[ii][diff]] - Z[own_ix[k]])))
        rows.append(dict(entity=r.entity, dist_m=dm, n=len(b),
                         p_change=float(np.mean(pch)),
                         denv_given_change=float(np.mean(denv)) if denv else np.nan))
D = pd.DataFrame(rows)

print("=== P(subcatchment changes) po rastojanju ===")
piv = D.pivot(index="entity", columns="dist_m", values="p_change")
print(piv.round(3).to_string())
print("\n  panel median:", piv.median().round(3).to_dict())

print("\n=== efektivna doza = nominalna x P(promena) ===")
med = piv.median()
print(f"{'nominalna':>10}" + "".join(f"{f'{d} m':>11}" for d in DIST))
for nm in NOM:
    print(f"{nm:>9}%" + "".join(f"{nm*med[d]:>10.2f}%" for d in DIST))

print("\n=== koliko rastojanje treba za efektivnu dozu od 20% ===")
for nm in (50, 100):
    for d in DIST:
        if nm * med[d] >= 20:
            print(f"  nominalna {nm}% dostize 20% efektivne na {d} m")
            break
    else:
        print(f"  nominalna {nm}% NE dostize 20% efektivne ni na {max(DIST)} m"
              f" (max {nm*med[max(DIST)]:.1f}%)")

print("\n=== Denv | promena (uporedi sa nasumicnim parom 0.757) ===")
print(D.pivot(index="entity", columns="dist_m", values="denv_given_change").round(3).to_string())
D.to_csv("results/revision/t3_effective_dose.csv", index=False)
