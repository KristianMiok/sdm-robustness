"""Where would a displaced record land? Strahler composition of the
neighbourhood reachable at each displacement distance, per entity."""
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from sdm_robustness.execution import runner as R
from sdm_robustness.pipeline.core import prepare_accessible_area

Rk = 6371008.8
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")
DIST = (100, 250, 500, 1000)

for _, r in P[P.run_lowacc == 1].iterrows():
    sp = r.entity.split(" (")[0]
    d = M[M.Crayfish_scientific_name == sp]
    if r.treatment == "native_only": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif r.treatment == "alien_only": d = d[d.Status.isin(R.ALIEN_VALUES)]
    hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m <= 200)])
    acc = prepare_accessible_area(M, b)
    # candidate landing sites = everything in the occupied basins, incl. occupied subc
    cand = M[M.basin_id.astype(str).isin(set(b.basin_id.astype(str)))]
    cand = cand.sort_values("subc_id").drop_duplicates("subc_id")
    if len(cand) < 50: continue
    T = BallTree(np.radians(cand[["lat_snap","long_snap"]].to_numpy()), metric="haversine")
    X = np.radians(b[["lat_snap","long_snap"]].to_numpy())
    own = b.strahler.to_numpy()
    print(f"\n{r.entity}")
    print(f"  benchmark %1.red {(own==1).mean()*100:5.1f} | "
          f"pristupacno %1.red {(acc.strahler==1).mean()*100:5.1f} | "
          f"kandidati n={len(cand)} %1.red {(cand.strahler==1).mean()*100:5.1f}")
    for dm in DIST:
        idx = T.query_radius(X, r=dm/Rk)
        s1 = []; changed = []; lower = []
        for k, ii in enumerate(idx):
            if len(ii) == 0: continue
            st = cand.strahler.to_numpy()[ii]
            sc = cand.subc_id.to_numpy()[ii]
            s1.append((st == 1).mean())
            changed.append((sc != b.subc_id.to_numpy()[k]).mean())
            oth = st[sc != b.subc_id.to_numpy()[k]]
            if len(oth): lower.append((oth < own[k]).mean())
        if not s1: continue
        print(f"    {dm:>5} m: zapisa sa susedom {len(s1):>5}/{len(b)}"
              f"  susedstvo %1.red {np.mean(s1)*100:5.1f}"
              f"  P(drugi subc) {np.mean(changed):.3f}"
              f"  P(niži red | drugi) {np.mean(lower) if lower else float('nan'):.3f}")
