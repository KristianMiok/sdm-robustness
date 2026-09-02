"""Mean vs median za tabele osetljive na zakosenost."""
import glob, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
OUT = Path("results/revision/tables")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

def robust(df, cols, by):
    g = df.groupby(by)[cols]
    o = pd.concat({"mean": g.mean(), "med": g.median(),
                   "iqr": g.quantile(.75) - g.quantile(.25)}, axis=1).round(3)
    o.columns = [f"{m}|{s}" for s, m in o.columns]
    return o.reindex(sorted(o.columns), axis=1)

D = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in
               glob.glob("results/campaign/*/results_raw.parquet")], ignore_index=True)
C = D[(D.status == "ok") & (D.axis != "benchmark")]
THR = [c for c in C.columns if c.startswith("range_area_pct_change")]
CONT = [c for c in ("suitability_mad","suitability_mean_shift","schoener_d","warren_i") if c in C.columns]
robust(C, THR, ["axis","level"]).to_csv(OUT/"T6_5_thresholds_robust.csv")
robust(C, CONT, ["axis","level"]).to_csv(OUT/"T6_6_continuous_robust.csv")
print("=== T6.5 glavne kolone ===")
print(robust(C, ["range_area_pct_change_05","range_area_pct_change_03",
                 "range_area_pct_change_07","range_area_pct_change_maxsss"], ["axis","level"]))
print("\n=== T6.5 sporne kolone ===")
print(robust(C, ["range_area_pct_change_p10","range_area_pct_change_maxsss_fix"], ["axis","level"]))

T3 = pd.concat([pd.read_csv(f) for f in glob.glob("results/revision/t3*_*.csv")
                if "effective_dose" not in f], ignore_index=True)
T3 = T3[T3.dist_m > 0]
r3 = robust(T3, ["range_area_pct_change_05","range_area_pct_change_maxsss","schoener_d"], ["frac","dist_m"])
r3.to_csv(OUT/"T3_displacement_robust.csv"); print("\n=== T3 ===\n", r3)

T15 = pd.concat([pd.read_csv(f) for f in glob.glob("results/revision/t1_5_*.csv")], ignore_index=True)
r15 = robust(T15, ["range_area_pct_change_05","range_area_pct_change_maxsss","d_str1"], ["level","mode"])
r15.to_csv(OUT/"T1_5_stratified_robust.csv"); print("\n=== T1.5 ===\n", r15)
w = T15.groupby(["level","mode"])["range_area_pct_change_05"].median().unstack("mode")
w["pct_removed"] = (100*(w["plain"]-w["stratified"])/w["plain"]).round(1)
print("\n=== T1.5 Strahler, na medijanama ===\n", w.round(3))
