"""Dose-matched axis comparison on the metrics Grid A actually carries,
plus an inventory of where range-area metrics live."""
import numpy as np, pandas as pd
from pathlib import Path

M = ["auc","tss","brier","sensitivity","specificity"]
files = sorted(Path("results/task5_execution").rglob("results_raw.parquet"))
df = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in files], ignore_index=True)
d = df[df.status == "ok"].copy()
d["level"] = d.level.astype(int)
print(f"redova ok: {len(d)} / {len(df)}")

print("\n=== 0. pokrivenost celija ===")
print(pd.crosstab(d.level, d.axis).to_string())
print("\nreplika po celiji (entity x alg x track x axis x level):")
print(d.groupby(["entity","algorithm","track","axis","level"]).size()
       .describe()[["min","50%","max"]].round(1).to_dict())
print("\ncontrast_pool_n po osi (median):")
print(d.groupby("axis").contrast_pool_n.median().to_string())

print("\n=== 1. DOZNO UPARENO: promena u odnosu na nivo 0 ===")
key = ["entity","algorithm","track","axis"]
b = d[d.level == 0].groupby(key)[M].mean().add_suffix("_b0")
m = d.merge(b, left_on=key, right_index=True, how="left")
for met in M:
    m[met+"_d"] = m[met] - m[met+"_b0"]

for met in ("auc","tss"):
    print(f"\n--- {met.upper()}: median delta u odnosu na sopstveni nivo 0 ---")
    p = m[m.level > 0].pivot_table(index="level", columns=["axis","algorithm"],
                                   values=met+"_d", aggfunc="median")
    print(p.round(4).to_string())

print("\n=== 2. odnos efekata pri istoj dozi (lowacc / snapping) ===")
for met in ("auc","tss"):
    t = m[m.level > 0].groupby(["level","axis"])[met+"_d"].median().unstack()
    if {"lowacc","snapping"} <= set(t.columns):
        t["odnos"] = t.lowacc / t.snapping
        print(f"\n{met.upper()}:")
        print(t.round(4).to_string())

print("\n=== 3. po entitetu, samo nivo 20 ===")
t = m[m.level == 20].pivot_table(index="entity", columns="axis",
                                 values="tss_d", aggfunc="median")
print(t.round(4).to_string())

print("\n=== 4. gde uopste postoje range/niche metrike ===")
for p in sorted(Path("results").rglob("*.parquet")):
    try:
        c = pd.read_parquet(p, engine="fastparquet").columns.tolist()
    except Exception:
        continue
    hit = [x for x in c if any(k in x.lower() for k in
           ("range","area","schoener","warren","jaccard","centroid","breadth","spearman"))]
    if hit:
        print(f"  {p}\n     -> {hit}")
for p in sorted(Path("results").rglob("*.csv")):
    if p.stat().st_size > 5e8: continue
    try:
        c = pd.read_csv(p, nrows=0).columns.tolist()
    except Exception:
        continue
    hit = [x for x in c if any(k in x.lower() for k in
           ("range","area","schoener","warren","jaccard","centroid","breadth"))]
    if hit:
        print(f"  {p}\n     -> {hit[:12]}")
