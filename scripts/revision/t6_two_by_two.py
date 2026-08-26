"""Train clean / train contaminated x test clean / test contaminated.

Answers MC7 directly: does a declining AUC mean the model lost predictive
ability, or that the test set was degraded?

Reimplements the CV loop rather than patching fit_cv_cell - the pipeline is
tested and this is a three-entity diagnostic. Folds, background draw and
model seeds are shared across all four cells, so the only thing that varies
is which set trains and which set tests.

Also reports, free of extra fits, how the contaminated model scores on the
retained-benchmark part of its test fold versus the substituted part.
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import (
    clean_predictors, get_track_columns, contaminate_presence_set, build_model)
from sdm_robustness.metrics import compute_performance_metrics
from sdm_robustness.utils.repro import derive_seed

ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True)
ap.add_argument("--alg", default="random_forest")
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--track", default="combined")
a = ap.parse_args()
MASTER = 20260426

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
p = R._prepare_entity_data(M, a.entity)
train_b, _ref = R.split_reference_set(p["benchmark"], a.entity, MASTER)
B = train_b if _ref is not None else p["benchmark"]
acc = p["accessible_area"]
kept = clean_predictors(B, get_track_columns(B, a.track))
med = B[kept].median(numeric_only=True)
A = acc[["subc_id", "basin_id"] + kept].copy(); A[kept] = A[kept].fillna(med)
fold_map = assign_basin_folds(B.basin_id, n_splits=5, looo_threshold=15)
print(f"{a.entity} [{a.alg}]: train {len(B)} | acc {len(A)} | preds {len(kept)}", flush=True)

def prep(df):
    d = df[["subc_id", "basin_id"] + kept].copy()
    d[kept] = d[kept].fillna(med)
    d["fold"] = d["basin_id"].astype(str).map(fold_map)
    return d

def score(model, pres, neg):
    if not len(pres) or not len(neg): return None
    X = pd.concat([pres[kept], neg[kept]], axis=0)
    y = np.array([1]*len(pres) + [0]*len(neg))
    if hasattr(model, "predict_proba"):
        s = np.asarray(model.predict_proba(X)); s = s[:, -1] if s.ndim == 2 else s
    else:
        s = np.asarray(model.predict(X), dtype=float)
    return compute_performance_metrics(y, s, threshold=0.5)

rows = []
Bp = prep(B)
for rep in range(a.reps):
    for axis, levels in (("snapping", (1, 2, 5)), ("lowacc", (3, 10, 20))):
        pool = p["snap_pool"] if axis == "snapping" else p["lowacc_pool"]
        if not len(pool): continue
        if axis == "snapping" and int(p["panel_row"].get("run_snapping", 1)) != 1: continue
        if axis == "lowacc" and int(p["panel_row"].get("run_lowacc", 0)) != 1: continue
        for lv in levels:
            if int(round(len(B)*lv/100)) > len(pool): continue
            s = derive_seed(MASTER, a.entity, axis, lv, rep)
            Cr = contaminate_presence_set(benchmark=B, contamination_pool=pool,
                                          level_pct=lv, seed=s)
            Cp = prep(Cr)
            n_sub = int(round(len(B)*lv/100))
            sub_ids = set(Cr.subc_id.astype(str).iloc[len(Cr)-n_sub:])
            bg = derive_seed(MASTER, a.entity, a.alg, a.track, rep)
            cells = {k: [] for k in ("cc","cx","xc","xx","kept_part","sub_part")}
            for f in sorted(Bp.fold.dropna().unique()):
                nn = min(len(Cp), len(A))
                neg = A.sample(n=nn, replace=False, random_state=(bg + int(f)) % (2**31))
                neg["fold"] = neg.basin_id.astype(str).map(fold_map).fillna(0)
                ntr, nte = neg[neg.fold != f], neg[neg.fold == f]
                bt, bv = Bp[Bp.fold != f], Bp[Bp.fold == f]
                ct, cv = Cp[Cp.fold != f], Cp[Cp.fold == f]
                if bt.empty or bv.empty or ct.empty or cv.empty or nte.empty: continue
                for tag, tr in (("c", bt), ("x", ct)):
                    Xt = pd.concat([tr[kept], ntr[kept]], axis=0)
                    yt = np.array([1]*len(tr) + [0]*len(ntr))
                    if len(np.unique(yt)) < 2: continue
                    m = build_model(a.alg, seed=s, n_jobs=8, maxent_n_cpus=1)
                    m.fit(Xt, yt)
                    for tag2, te in (("c", bv), ("x", cv)):
                        r = score(m, te, nte)
                        if r: cells[tag+tag2].append(r["auc"])
                    if tag == "x":
                        kp = cv[~cv.subc_id.astype(str).isin(sub_ids)]
                        sp = cv[cv.subc_id.astype(str).isin(sub_ids)]
                        for nm, part in (("kept_part", kp), ("sub_part", sp)):
                            r = score(m, part, nte)
                            if r: cells[nm].append(r["auc"])
            rows.append(dict(rep=rep, axis=axis, level=lv,
                             **{k: (float(np.mean(v)) if v else np.nan)
                                for k, v in cells.items()}))
    print(f"  rep {rep+1}/{a.reps}", flush=True)

D = pd.DataFrame(rows)
slug = "".join(c for c in a.entity.replace(" ", "_") if c.isalnum() or c == "_")
Path("results/revision").mkdir(parents=True, exist_ok=True)
D.to_csv(f"results/revision/t6_2x2_{slug}_{a.alg}.csv", index=False)

print(f"\n=== {a.entity} [{a.alg}] {a.reps} replika ===")
print("  cc = train clean/test clean   cx = train clean/test contaminated")
print("  xc = train contam/test clean  xx = train contam/test contaminated\n")
g = D.groupby(["axis","level"])[["cc","cx","xc","xx","kept_part","sub_part"]].mean()
print(g.round(4).to_string())
print("\n  test-set effect (cx - cc) vs model effect (xc - cc):")
t = pd.DataFrame({"test_effect": g.cx - g.cc, "model_effect": g.xc - g.cc,
                  "observed_cv": g.xx - g.cc,
                  "kept_vs_sub": g.kept_part - g.sub_part})
print(t.round(4).to_string())
