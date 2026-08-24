"""Does the published delta change when the baseline is 30 replicates
instead of the single benchmark run?"""
import glob
import numpy as np, pandas as pd

KEY = ["entity","algorithm","track"]
B = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                    engine="fastparquet")
B = B[B.status == "ok"]
bm = B[B.axis == "benchmark"].drop_duplicates(KEY)
print("=== 1. benchmark krak ===")
print(f"  sirovih redova {len(B[B.axis=='benchmark'])} | jedinstvenih celija {len(bm)}"
      f" | replicate {sorted(B[B.axis=='benchmark'].replicate.unique())}"
      f" | seed-ova {B[B.axis=='benchmark'].seed.nunique()}")

print("\n=== 2. stability, svi fajlovi ===")
fs = sorted(set(glob.glob("results/task5c_benchmark_stability_array/**/benchmark_stability.parquet",
                          recursive=True)
              + glob.glob("results/task5_execution/**/benchmark_stability.parquet",
                          recursive=True)))
print("  fajlova:", len(fs))
S = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in fs], ignore_index=True)
S = S.drop_duplicates(KEY + ["metric_name"])
print("  redova:", len(S), "| entiteta:", S.entity.nunique(),
      "| metrika:", sorted(S.metric_name.unique()))
print("  replika:", S.n_replicates_ok.value_counts().to_dict())
print("\n  pokrivenost (combined track):")
print(pd.crosstab(S[S.track=="combined"].entity, S[S.track=="combined"].algorithm).to_string())

print("\n=== 3. baseline: jedan run vs 30 replika ===")
b30 = S[S.metric_name=="auc"][KEY+["benchmark_mean","benchmark_sd"]].rename(
        columns={"benchmark_mean":"b30","benchmark_sd":"sd30"})
b1 = bm[KEY+["auc"]].rename(columns={"auc":"b1"})
cmp = b1.merge(b30, on=KEY, how="inner")
cmp["diff"] = cmp.b1 - cmp.b30
cmp["z"] = cmp["diff"]/cmp.sd30
print(f"  celija sa oba: {len(cmp)}")
print("  |b1-b30|:", cmp["diff"].abs().describe(percentiles=[.5,.9]).round(5).to_dict())
print("  |z| (koliko SD je jedan run od sredine 30):",
      cmp.z.abs().describe(percentiles=[.5,.9]).round(2).to_dict())
print("  celija gde je jedan run >2 SD od sredine:", int((cmp.z.abs()>2).sum()), "od", len(cmp))
print(cmp.reindex(cmp.z.abs().sort_values(ascending=False).index)
        .head(6)[KEY+["b1","b30","sd30","z"]].round(4).to_string(index=False))

c = B[(B.axis!="benchmark") & (B.track=="combined")].merge(
        b1, on=KEY, how="left").merge(b30, on=KEY, how="left").dropna(subset=["b1","b30"])
c["d_one"], c["d_30"] = c.auc-c.b1, c.auc-c.b30
print(f"\n  kontaminiranih redova sa oba baseline-a: {len(c)} | entiteta {c.entity.nunique()}")
t = c.groupby(["axis","level","algorithm"])[["d_one","d_30"]].median()
t["razlika"] = t.d_one - t.d_30
print(t.round(4).to_string())

print("\n=== 4. Grid A, baseline = 30-50 replika po osi ===")
A = pd.concat([pd.read_parquet(p, engine="fastparquet")
               for p in glob.glob("results/task5_execution/**/results_raw.parquet", recursive=True)],
              ignore_index=True)
A = A[(A.status=="ok") & (A.track=="combined")].copy()
A["level"] = A.level.astype(int)
a0 = A[A.level==0].groupby(["entity","algorithm","axis"]).auc.agg(b0="mean", n="size")
print("  replika u level-0:", a0.n.describe()[["min","50%","max"]].to_dict())
m = A[A.level>0].merge(a0["b0"], on=["entity","algorithm","axis"], how="left")
m["d"] = m.auc - m.b0
print(m[m.level.isin([5,10,20])].pivot_table(index="level", columns=["axis","algorithm"],
      values="d", aggfunc="median").round(4).to_string())
print("\n  n_experiment po entitetu:")
print(pd.DataFrame({"gridA": A.groupby("entity").n_experiment.first(),
                    "gridB": B.groupby("entity").n_experiment.first()}).to_string())
