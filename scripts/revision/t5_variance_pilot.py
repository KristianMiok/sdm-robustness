"""Variance decomposition pilot: how much of the range-change noise is
background resampling, how much is model stochasticity, and what does the
floor become once benchmark and contaminated runs share the background draw?

Four arms, all on one entity:
  A bg_only      fixed presences, background varied      -> var(background)
  D model_only   fixed presences and background          -> var(model)
  C unpaired     benchmark vs contaminated, different bg -> current pipeline
  B paired       benchmark vs contaminated, SAME bg      -> what T5 buys us
"""
import argparse, time
import numpy as np, pandas as pd
from pathlib import Path

from sdm_robustness.execution.runner import _prepare_entity_data
from sdm_robustness.pipeline.core import (
    clean_predictors, get_track_columns, contaminate_presence_set,
    build_model, predict_suitability_surface)
from sdm_robustness.metrics import range_area_change, schoeners_d

ap = argparse.ArgumentParser()
ap.add_argument("--entity", default="Austropotamobius torrentium (pooled)")
ap.add_argument("--axis", default="snapping", choices=["snapping", "lowacc"])
ap.add_argument("--level", type=int, default=5)
ap.add_argument("--reps", type=int, default=15)
ap.add_argument("--track", default="combined")
ap.add_argument("--algorithms", default="random_forest,xgboost")
ap.add_argument("--out", default="results/revision/t5_variance_pilot.csv")
a = ap.parse_args()

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
prep = _prepare_entity_data(M, a.entity)
print("keys:", sorted(prep.keys()))
bench = prep["benchmark"]
pool = prep["snap_pool"] if a.axis == "snapping" else prep["lowacc_pool"]
acc = prep["accessible_area"]
print(f"{a.entity}: benchmark {len(bench)} | pool {len(pool)} | accessible {len(acc)}")
need = int(round(len(bench) * a.level / 100))
if need > len(pool):
    raise SystemExit(f"pool too small: need {need}, have {len(pool)}")

feat = get_track_columns(bench, a.track)
kept = clean_predictors(bench, feat)
medians = bench[kept].median(numeric_only=True)
acc = acc[["subc_id", "basin_id"] + kept].copy()
acc[kept] = acc[kept].fillna(medians)
print(f"predictors kept: {len(kept)}")

def prep_pres(df):
    d = df[["subc_id", "basin_id"] + kept].copy()
    d[kept] = d[kept].fillna(medians)
    return d

BENCH = prep_pres(bench)

def surface(pres, alg, bg_seed, model_seed):
    bg_n = min(10000, len(acc)) if alg == "maxent" else min(len(pres), len(acc))
    neg = acc.sample(n=bg_n, replace=False, random_state=int(bg_seed) % (2**31))
    X = pd.concat([pres[kept], neg[kept]], axis=0)
    y = np.array([1] * len(pres) + [0] * len(neg))
    m = build_model(alg, seed=int(model_seed) % (2**31), n_jobs=-1, maxent_n_cpus=1)
    m.fit(X, y)
    return predict_suitability_surface(m, acc[kept])

CONTAM = {i: prep_pres(contaminate_presence_set(
              benchmark=bench, contamination_pool=pool,
              level_pct=a.level, seed=90000 + i))
          for i in range(a.reps)}

rows, t0 = [], time.time()
for alg in a.algorithms.split(","):
    print(f"\n=== {alg} ===", flush=True)
    n_fit = 0
    for i in range(a.reps):
        # A: two independent background draws, same presences, same model seed
        s1 = surface(BENCH, alg, 20_000 + 2*i, 500)
        s2 = surface(BENCH, alg, 20_000 + 2*i + 1, 500)
        rows.append(dict(arm="A_bg_only", alg=alg, rep=i,
                         range_pct=range_area_change(s1, s2, threshold=0.5),
                         schoener=schoeners_d(s1, s2)))
        # D: two independent model seeds, same presences, same background
        s1 = surface(BENCH, alg, 10_000, 600 + 2*i)
        s2 = surface(BENCH, alg, 10_000, 600 + 2*i + 1)
        rows.append(dict(arm="D_model_only", alg=alg, rep=i,
                         range_pct=range_area_change(s1, s2, threshold=0.5),
                         schoener=schoeners_d(s1, s2)))
        # C: benchmark vs contaminated, different background (current pipeline)
        b = surface(BENCH, alg, 30_000 + i, 500)
        c = surface(CONTAM[i], alg, 40_000 + i, 500)
        rows.append(dict(arm="C_unpaired", alg=alg, rep=i,
                         range_pct=range_area_change(b, c, threshold=0.5),
                         schoener=schoeners_d(b, c)))
        # B: benchmark vs contaminated, SAME background
        b = surface(BENCH, alg, 50_000 + i, 500)
        c = surface(CONTAM[i], alg, 50_000 + i, 500)
        rows.append(dict(arm="B_paired", alg=alg, rep=i,
                         range_pct=range_area_change(b, c, threshold=0.5),
                         schoener=schoeners_d(b, c)))
        n_fit += 8
        if i == 0 or (i + 1) % 5 == 0:
            print(f"  rep {i+1}/{a.reps}  fits={n_fit}  "
                  f"{(time.time()-t0)/max(n_fit,1):.1f} s/fit", flush=True)

R = pd.DataFrame(rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
R.to_csv(a.out, index=False)

print(f"\n=== {a.entity} | {a.axis} L{a.level} | reps={a.reps} ===")
S = R.groupby(["alg", "arm"]).agg(
    mean=("range_pct", "mean"), sd=("range_pct", "std"),
    p5=("range_pct", lambda x: x.quantile(.05)),
    p95=("range_pct", lambda x: x.quantile(.95)),
    schoener=("schoener", "mean"))
print(S.round(3).to_string())

print("\n=== decomposition ===")
for alg in R.alg.unique():
    g = R[R.alg == alg].set_index("arm").groupby("arm").range_pct
    sd = {k: v for k, v in g.std().items()}
    unp, pai = sd.get("C_unpaired", np.nan), sd.get("B_paired", np.nan)
    print(f"  {alg}")
    print(f"    var(background)      SD {sd.get('A_bg_only', np.nan):.3f}")
    print(f"    var(model)           SD {sd.get('D_model_only', np.nan):.3f}")
    print(f"    unpaired (current)   SD {unp:.3f}")
    print(f"    paired               SD {pai:.3f}")
    A, D = sd.get("A_bg_only", np.nan), sd.get("D_model_only", np.nan)
    if np.isfinite(A) and np.isfinite(D):
        print(f"    consistency: sqrt(A^2+D^2) = {np.sqrt(A**2+D**2):.3f} vs unpaired {unp:.3f}")
    if np.isfinite(unp) and np.isfinite(pai) and pai > 0:
        print(f"    -> pairing reduces the floor {unp/pai:.1f}x")
        print(f"    -> effect detectable above ~{2*pai:.1f}% instead of ~{2*unp:.1f}%")
