"""T3: controlled displacement at known distances.

Clean benchmark records are displaced by a known distance in a random
direction and re-assigned to the nearest subcatchment. Predictors follow the
subcatchment, so a record that lands in its own subcatchment is a no-op and the
EFFECTIVE dose is recorded alongside the nominal one.

Limitation, stated: candidate landing sites are subcatchments that appear in the
master table, i.e. those containing crayfish records. The real network has
headwater segments with no records, which this cannot reach. The test is
therefore conservative.
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import BallTree
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import fit_cv_cell, clean_predictors, get_track_columns
from sdm_robustness.utils.repro import derive_seed

RK, MASTER = 6371008.8, 20260426
ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True)
ap.add_argument("--alg", default="random_forest")
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--dists", type=int, nargs="+", default=[100, 250, 500])
ap.add_argument("--fracs", type=int, nargs="+", default=[10, 20, 50])
ap.add_argument("--out-suffix", default="")
a = ap.parse_args()

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
p = R._prepare_entity_data(M, a.entity)
B, acc = p["benchmark"], p["accessible_area"]
kept = clean_predictors(B, get_track_columns(B, "combined"))
fm = assign_basin_folds(B.basin_id, n_splits=5, looo_threshold=15)

cand = M[M.basin_id.astype(str).isin(set(B.basin_id.astype(str)))]
cand = cand.sort_values("subc_id").drop_duplicates("subc_id").reset_index(drop=True)
tree = BallTree(np.radians(cand[["lat_snap", "long_snap"]].to_numpy()), metric="haversine")
print(f"{a.entity} [{a.alg}]: benchmark {len(B)} | kandidata {len(cand)} | preds {len(kept)}",
      flush=True)

def displace(bench, dist_m, frac_pct, rng):
    """Move a fraction of records dist_m in a random direction, re-assign to
    the nearest candidate subcatchment, and return (new_set, effective_dose)."""
    n = int(round(len(bench) * frac_pct / 100))
    if n == 0: return bench.copy(), 0.0
    idx = rng.choice(len(bench), n, replace=False)
    lat = bench.lat_snap.to_numpy()[idx]; lon = bench.long_snap.to_numpy()[idx]
    th = rng.uniform(0, 2*np.pi, n); dd = dist_m / RK
    la = np.radians(lat); lo = np.radians(lon)
    la2 = np.arcsin(np.sin(la)*np.cos(dd) + np.cos(la)*np.sin(dd)*np.cos(th))
    lo2 = lo + np.arctan2(np.sin(th)*np.sin(dd)*np.cos(la),
                          np.cos(dd) - np.sin(la)*np.sin(la2))
    _, j = tree.query(np.c_[la2, lo2], k=1)
    land = cand.iloc[j.ravel()]
    out = bench.copy().reset_index(drop=True)
    cols = ["subc_id", "basin_id", "strahler"] + [c for c in kept if c in land.columns]
    moved = land.subc_id.to_numpy() != out.subc_id.to_numpy()[idx]
    # positional assignment: .loc would align land's own index against idx labels
    ci = [out.columns.get_loc(c) for c in cols]
    out.iloc[idx, ci] = land[cols].to_numpy()
    return out, float(moved.mean() * frac_pct)

rows = []
for rep in range(a.reps):
    rng = np.random.default_rng(derive_seed(MASTER, a.entity, "t3", 0, rep))
    bg = derive_seed(MASTER, a.entity, a.alg, "combined", rep)
    base = fit_cv_cell(benchmark=B, contamination_pool=B, accessible_area=acc,
                       entity=a.entity, algorithm=a.alg, track="combined",
                       axis="benchmark", level=0, replicate=rep,
                       seed=derive_seed(MASTER, a.entity, "benchmark", 0, rep),
                       n_experiment=len(B), fold_map=fm, bg_seed=bg,
                       return_artifacts=True)
    surf = base.pop("_run_surface", None)
    rows.append(dict(rep=rep, dist_m=0, frac=0, eff_dose=0.0,
                     **{k: base.get(k) for k in ("auc","tss","cv_auc") if k in base}))
    for d in a.dists:
        for f in a.fracs:
            D, eff = displace(B, d, f, rng)
            r = fit_cv_cell(benchmark=D, contamination_pool=D, accessible_area=acc,
                            entity=a.entity, algorithm=a.alg, track="combined",
                            axis="displacement", level=f, replicate=rep,
                            seed=derive_seed(MASTER, a.entity, "t3", f, rep),
                            n_experiment=len(D), fold_map=fm, bg_seed=bg,
                            benchmark_surface=surf, return_artifacts=True)
            r.pop("_run_surface", None); r.pop("_run_importance", None)
            rows.append(dict(rep=rep, dist_m=d, frac=f, eff_dose=eff,
                             **{k: r.get(k) for k in
                                ("auc","tss","cv_auc","range_area_pct_change_05",
                                 "range_area_pct_change_maxsss","suitability_mad",
                                 "schoener_d","env_centroid_disp") if k in r}))
    print(f"  rep {rep+1}/{a.reps}", flush=True)

T = pd.DataFrame(rows)
slug = "".join(c for c in a.entity.replace(" ", "_") if c.isalnum() or c == "_")
Path("results/revision").mkdir(parents=True, exist_ok=True)
T.to_csv(f"results/revision/t3{a.out_suffix}_{slug}_{a.alg}.csv", index=False)
print(f"\n=== {a.entity} [{a.alg}] ===")
g = T[T.dist_m > 0].groupby(["dist_m","frac"]).agg(
    eff=("eff_dose","mean"),
    range05=("range_area_pct_change_05","mean"),
    mad=("suitability_mad","mean"))
print(g.round(3).to_string())
