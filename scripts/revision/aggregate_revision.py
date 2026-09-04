"""Aggregate the revision campaigns into the tables the manuscript needs."""
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path

OUT = Path("results/revision/tables"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260828); B = 3000
D = pd.concat([pd.read_parquet(x, engine="fastparquet")
               for x in sorted(glob.glob("results/campaign/*/results_raw.parquet"))],
              ignore_index=True)
D = D[D.status == "ok"].drop_duplicates(["entity","algorithm","track","axis","level","replicate"])
print(f"runova ok: {len(D)}")

def hboot(g, col):
    by = {e: v.dropna().to_numpy() for e, v in g.groupby("entity")[col]}
    by = {e: v for e, v in by.items() if len(v)}
    if len(by) < 2: return (np.nan,)*3
    ks = list(by)
    out = np.array([np.mean([by[ks[p]][rng.integers(0, len(by[ks[p]]), len(by[ks[p]]))].mean()
                             for p in rng.choice(len(ks), len(ks), True)]) for _ in range(B)])
    return (float(np.mean([v.mean() for v in by.values()])),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

# --- 1. delta-scale S4/S5 -------------------------------------------------
KEY = ["entity","algorithm","track"]
MET = [c for c in ("auc","tss","brier","sensitivity","specificity","omission_rate",
                   "importance_spearman","env_centroid_disp","env_dispersion_change")
       if c in D.columns]
b0 = D[D.axis=="benchmark"].groupby(KEY)[MET].mean().reset_index()
C = D[D.axis!="benchmark"].merge(b0, on=KEY, how="left", suffixes=("","_b0"))
for k in MET: C["d_"+k] = C[k] - C[k+"_b0"]
rows = []
for (ax, lv, al, tr), g in C.groupby(["axis","level","algorithm","track"]):
    for k in MET:
        e, lo, hi = hboot(g, "d_"+k)
        rows.append(dict(axis=ax, level=lv, algorithm=al, track=tr, metric=k,
                         delta=e, lo95=lo, hi95=hi, n_ent=g.entity.nunique(), n=len(g),
                         sig="" if not np.isfinite(lo) or lo <= 0 <= hi else "*"))
S = pd.DataFrame(rows); S.to_csv(OUT/"S4_S5_delta.csv", index=False)
print("\n=== S4/S5, combined track, AUC ===")
s = S[(S.metric=="auc") & (S.track=="combined")]
print(s.pivot_table(index=["axis","level"], columns="algorithm",
                    values="delta").round(4).to_string())

# --- 2. threshold sensitivity (T6.5) --------------------------------------
TH = [c for c in D.columns if c.startswith("range_area_pct_change_")]
T = D[D.axis!="benchmark"].groupby(["axis","level","algorithm"])[TH].mean()
T.to_csv(OUT/"T6_5_thresholds.csv")
print("\n=== T6.5 pragovi, combined ===")
print(D[(D.axis!="benchmark") & (D.track=="combined")]
      .groupby(["axis","level"])[TH].mean().round(2).to_string())

# --- 3. continuous, no threshold (T6.6) -----------------------------------
CO = [c for c in ("suitability_mad","suitability_mean_shift","schoener_d","warren_i")
      if c in D.columns]
D[D.axis!="benchmark"].groupby(["axis","level","algorithm"])[CO].mean().to_csv(
    OUT/"T6_6_continuous.csv")
print("\n=== T6.6 bez praga ===")
print(D[(D.axis!="benchmark") & (D.track=="combined")]
      .groupby(["axis","level"])[CO].mean().round(4).to_string())

# --- 4. per-entity effects ------------------------------------------------
P = D[(D.axis!="benchmark") & (D.track=="combined")].pivot_table(
    index="entity", columns=["axis","level"],
    values="range_area_pct_change_05", aggfunc="mean")
P.to_csv(OUT/"per_entity_range.csv")
print("\n=== po entitetu, range@0.5 ===")
print(P.round(1).to_string())

# --- 5. feasibility of the extended doses ---------------------------------
A = pd.concat([pd.read_parquet(x, engine="fastparquet")
               for x in sorted(glob.glob("results/campaign/*/results_raw.parquet"))],
              ignore_index=True)
F = (A.assign(ok=A.status.eq("ok")).groupby(["axis","level"])
       .ok.agg(["sum","size"]).rename(columns={"sum":"ok","size":"attempted"}))
F["pct"] = (F.ok/F.attempted*100).round(1)
F.to_csv(OUT/"dose_feasibility.csv")
print("\n=== izvodljivost doza ===")
print(F.to_string())
print(f"\nzapisano u {OUT}/")
