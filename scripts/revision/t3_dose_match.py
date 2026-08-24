"""Dose-matched comparison of the two contamination axes, from Grid A."""
import numpy as np, pandas as pd
from pathlib import Path

files = sorted(Path("results/task5_execution").rglob("results_raw.parquet"))
print("fajlova:", len(files))
df = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in files], ignore_index=True)
print("redova:", len(df), "| kolona:", len(df.columns))
print("\n=== kolone ===")
print(list(df.columns))

print("\n=== niska kardinalnost ===")
for c in df.columns:
    try:
        n = df[c].nunique(dropna=False)
    except TypeError:
        continue
    if n <= 14:
        print(f"  {c:<28} n={n:<4} {sorted(map(str, df[c].dropna().unique()))[:14]}")

def find(*pats):
    return [c for c in df.columns if any(p in c.lower() for p in pats)]
AX, LV = find("axis"), find("level", "pct", "frac")
RG, NE = find("range"), find("n_exp", "n_pres", "n_train", "n_occ")
ST, AL = find("status", "error"), find("algorithm", "algo", "model")
print(f"\nosa={AX}\nnivo={LV}\nrange={RG}\nn={NE}\nstatus={ST}\nalgoritam={AL}")

if AX and LV and RG and AL:
    ax, lv, rg, al = AX[0], LV[0], RG[0], AL[0]
    d = df
    for s in ST:
        if d[s].dtype == object:
            d = d[d[s].astype(str).str.lower().isin(["ok", "success", "none", "nan"])]
    print(f"\nposle filtriranja gresaka: {len(d)}")

    print("\n=== A. baseline (nivo 0) po entitetu/algoritmu/osi ===")
    base = d[d[lv] == 0].groupby([al, ax])[rg].agg(["size", "mean"])
    print(base.round(2).to_string())

    print(f"\n=== B. promena range-a u odnosu na nivo 0, po osi i dozi ===")
    key = [c for c in ("entity", "entity_name", "species") if c in d.columns]
    key = key[0] if key else None
    if key:
        b = d[d[lv] == 0].groupby([key, al, ax])[rg].mean().rename("b0")
        m = d.merge(b, left_on=[key, al, ax], right_index=True, how="left")
        m["pct"] = (m[rg] - m.b0) / m.b0 * 100
        t = m.groupby([ax, lv, al]).pct.agg(n="size", median="median", mean="mean")
        print(t.round(2).to_string())
        print("\n--- sazeto: median % promene, ose jedna do druge ---")
        print(m.pivot_table(index=lv, columns=[ax, al], values="pct",
                            aggfunc="median").round(2).to_string())
    else:
        print("nema kolone entiteta; sirovi proseci:")
        print(d.groupby([ax, lv, al])[rg].mean().round(3).to_string())

    if NE:
        print(f"\n=== C. velicina uzorka po osi ({NE[0]}) ===")
        print(d.pivot_table(index=lv, columns=ax, values=NE[0],
                            aggfunc="median").round(1).to_string())
else:
    print("\nnisu prepoznate kolone — posalji listu kolona i idemo rucno")
