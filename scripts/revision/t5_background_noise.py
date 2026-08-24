"""How much of the range change is background resampling rather than contamination?"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr

RG = "range_area_pct_change_05"
B = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                    engine="fastparquet")
B = B[(B.status=="ok") & (B.track=="combined")].drop_duplicates(
    ["entity","algorithm","track","axis","level","replicate"])
C = B[B.axis != "benchmark"]

print("=== 1. signal i sum po osi/nivou ===")
t = C.groupby(["axis","level","algorithm"])[RG].agg(
    n="size", mean="mean", sd="std", median="median")
t["|mean|/sd"] = (t["mean"].abs()/t.sd).round(2)
print(t.round(3).to_string())

print("\n=== 2. sumni pod: snapping L1 (1% supstitucije) ===")
nf = C[(C.axis=="snapping") & (C.level==1)].groupby("algorithm")[RG].agg(["mean","std"])
print(nf.round(3).to_string())
print("\n  odnos efekta na lowacc L20 prema tom sumu:")
for alg in nf.index:
    e = C[(C.axis=="lowacc") & (C.level==20) & (C.algorithm==alg)][RG].mean()
    print(f"    {alg:<14} efekat {e:>7.2f}  sum(sd) {nf.loc[alg,'std']:>6.2f}"
          f"  odnos {e/nf.loc[alg,'std']:>6.1f}x")

print("\n=== 3. po entitetu: sumni pod vs efekat ===")
rows=[]
for e, g in C.groupby("entity"):
    acc = g.accessible_area_segment_count.median()
    npres = g.n_experiment.median()
    for alg in ("maxent","random_forest","xgboost"):
        n1 = g[(g.axis=="snapping") & (g.level==1) & (g.algorithm==alg)][RG]
        e20 = g[(g.axis=="lowacc") & (g.level==20) & (g.algorithm==alg)][RG]
        bg = min(10000, acc) if alg=="maxent" else npres
        rows.append(dict(entity=e, alg=alg, acc=int(acc), bg=int(bg),
            bg_frac=round(bg/acc,3), sum_mean=n1.mean(), sum_sd=n1.std(),
            eff20=e20.mean() if len(e20) else np.nan))
D = pd.DataFrame(rows)
print(D.round(3).to_string(index=False))

print("\n=== 4. da li manji udeo pozadine daje veci sum ===")
for alg in ("maxent","random_forest","xgboost"):
    s = D[(D.alg==alg)].dropna(subset=["sum_sd"])
    if len(s) > 4:
        print(f"  {alg:<14} rho(bg_frac, sum_sd) = "
              f"{spearmanr(s.bg_frac, s.sum_sd).statistic:>6.3f}  n={len(s)}")

print("\n=== 5. udeo runova gde je |efekat| manji od suma ===")
for ax, lv in (("snapping",1),("snapping",2),("snapping",5),
               ("lowacc",3),("lowacc",10),("lowacc",20)):
    s = C[(C.axis==ax) & (C.level==lv)]
    m = s.groupby(["entity","algorithm"])[RG].mean()
    sd = C[(C.axis=="snapping") & (C.level==1)].groupby(["entity","algorithm"])[RG].std()
    j = pd.DataFrame({"m":m,"sd":sd}).dropna()
    print(f"  {ax:<9} L{lv:<3} {(j.m.abs() < j.sd).mean()*100:>5.1f}%  (n={len(j)})")
