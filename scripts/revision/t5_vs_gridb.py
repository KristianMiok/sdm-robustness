"""Paired pilot vs published Grid B, with proper standard errors."""
import glob, re
import numpy as np, pandas as pd

G = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                    engine="fastparquet")
G = G[(G.status=="ok") & (G.track=="combined") & (G.axis=="lowacc") & (G.level==20)]
g = G.groupby(["entity","algorithm"]).range_area_pct_change_05.agg(
    gb_mean="mean", gb_sd="std", gb_n="size")

rows = []
for f in sorted(glob.glob("results/revision/var_*_lowacc20.csv")):
    m = re.match(r".*/var_(.+)_(random_forest|xgboost|maxent)_lowacc20\.csv", f)
    if not m: continue
    slug, alg = m.groups()
    D = pd.read_csv(f)
    for arm in ("B_paired","C_unpaired"):
        s = D[D.arm==arm].range_pct
        if not len(s): continue
        rows.append(dict(slug=slug, algorithm=alg, arm=arm,
                         mean=s.mean(), sd=s.std(), n=len(s)))
Pl = pd.DataFrame(rows)
ents = {re.sub(r"[^A-Za-z0-9_]","",e.replace(" ","_")): e for e in g.index.get_level_values(0).unique()}
Pl["entity"] = Pl.slug.map(ents)
J = Pl.merge(g, left_on=["entity","algorithm"], right_index=True, how="inner")
J["se"] = np.sqrt(J.sd**2/J.n + J.gb_sd**2/J.gb_n)
J["diff"] = J["mean"] - J.gb_mean
J["z"] = J["diff"]/J.se
J["pct_of_effect"] = (J["diff"]/J.gb_mean*100)

for arm in ("C_unpaired","B_paired"):
    s = J[J.arm==arm].sort_values("z")
    print(f"\n=== {arm} vs Grid B ===")
    print(s[["entity","algorithm","mean","gb_mean","diff","se","z","pct_of_effect",
             "sd","gb_sd"]].round(2).to_string(index=False))
    print(f"  |z|>2: {int((s.z.abs()>2).sum())} od {len(s)}"
          f" | median razlike {s['diff'].median():.2f}"
          f" | SD reprodukcija (pilot/gridB) median {(s.sd/s.gb_sd).median():.2f}")

print("\n=== uticaj na panel (samo entiteti sa oba) ===")
for alg in J.algorithm.unique():
    s = J[(J.arm=="B_paired") & (J.algorithm==alg)]
    if s.empty: continue
    full = G[G.algorithm==alg].groupby("entity").range_area_pct_change_05.mean()
    adj = full.copy()
    for _, r in s.iterrows(): adj[r.entity] = r["mean"]
    print(f"  {alg:<14} objavljeno {full.mean():>6.2f}%  -> upareno {adj.mean():>6.2f}%"
          f"  ({len(s)}/{len(full)} zamenjeno)")
