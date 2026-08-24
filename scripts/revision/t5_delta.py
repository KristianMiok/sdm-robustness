"""T5 fallback: delta estimator + hierarchical bootstrap on existing Grid B."""
import numpy as np, pandas as pd

rng = np.random.default_rng(20260822); B = 3000
KEY = ["entity","algorithm","track"]
df = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                     engine="fastparquet")
df = df[df.status == "ok"].drop_duplicates(KEY + ["axis","level","replicate"])
print("redova ok:", len(df))
print(pd.crosstab(df.axis, df.level).to_string())

MET = [c for c in ("auc","tss","brier","sensitivity","specificity","importance_spearman",
       "importance_jaccard_top5","niche_centroid_disp","niche_breadth_change",
       "schoener_d","warren_i","range_area_pct_change_05") if c in df.columns]
print("\nnenull po metrici, benchmark vs kontaminirani:")
bench = df[df.axis == "benchmark"]; cont = df[df.axis.isin(["snapping","lowacc"])]
print(pd.DataFrame({"benchmark": bench[MET].notna().sum(),
                    "kontam": cont[MET].notna().sum()}).to_string())

MET = [k for k in MET if bench[k].notna().sum() > 0 and cont[k].notna().sum() > 0]
print("\nupotrebljive metrike:", MET)
if not MET:
    raise SystemExit("nema metrike prisutne u oba kraka")

b0 = bench.groupby(KEY)[MET].mean().reset_index()
sd = bench.groupby(KEY)[MET].std().reset_index()
m = cont.merge(b0, on=KEY, how="left", suffixes=("", "_b0")) \
        .merge(sd, on=KEY, how="left", suffixes=("", "_sd"))
for k in MET:
    m["d_"+k] = m[k] - m[k+"_b0"]
print("\nnenull delta:", {k: int(m["d_"+k].notna().sum()) for k in MET})

def hboot(g, col):
    by = {e: v.dropna().to_numpy() for e, v in g.groupby("entity")[col]}
    by = {e: v for e, v in by.items() if len(v)}
    if len(by) < 2: return (np.nan,)*3
    ks = list(by)
    out = np.array([np.mean([by[ks[p]][rng.integers(0, len(by[ks[p]]), len(by[ks[p]]))].mean()
                             for p in rng.choice(len(ks), len(ks), True)]) for _ in range(B)])
    return (float(np.mean([v.mean() for v in by.values()])),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))

rows = []
for (ax, lv, al), g in m[m.track == "combined"].groupby(["axis","level","algorithm"]):
    for k in MET:
        e, lo, hi = hboot(g, "d_"+k)
        rows.append(dict(axis=ax, level=lv, algorithm=al, metric=k, delta=e,
                         lo95=lo, hi95=hi, n_ent=g.entity.nunique(), n=len(g),
                         sig="" if not np.isfinite(lo) or lo <= 0 <= hi else "*"))
R = pd.DataFrame(rows)
R.to_csv("results/revision/t5_delta_bootstrap.csv", index=False)
print("\nzapisano results/revision/t5_delta_bootstrap.csv")

for k in ("range_area_pct_change_05","auc","tss","schoener_d","importance_spearman"):
    s = R[R.metric == k]
    if s.empty or s.delta.isna().all(): continue
    print(f"\n=== {k} ===")
    for _, r in s.sort_values(["axis","level","algorithm"]).iterrows():
        print(f"  {r.axis:<9} L{int(r.level):<3} {r.algorithm:<14}{r.delta:>10.4f}"
              f"  [{r.lo95:>9.4f}, {r.hi95:>9.4f}] {r.sig}  ent={r.n_ent} n={r.n}")

print("\n=== stara +-2SD anvelopa, samo gde SD postoji ===")
for k in ("auc","tss"):
    if k+"_sd" not in m: continue
    v = m[["d_"+k, k+"_sd"]].dropna()
    if not len(v): continue
    ex = (v["d_"+k].abs() > 2*v[k+"_sd"]).mean()
    one = (v["d_"+k] < -2*v[k+"_sd"]).mean()
    print(f"  {k}: n={len(v)}  dvostrano {ex*100:5.2f}%   jednostrano(pad) {one*100:5.2f}%")
