"""Domain-specific baselines, the coordinate/subc_id ambiguity, and pool composition."""
import re
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
rng = np.random.default_rng(20260822)
hdr = pd.read_csv(MASTER, nrows=0).columns.tolist()
pred = [c for c in hdr if re.match(r"^[lu]_", c)]
meta = ["WoCID","subc_id","strahler","basin_id","lat_or","long_or","lat_snap",
        "long_snap","distance_m","Accuracy","Crayfish_scientific_name","Status","Year_of_record"]
df = pd.read_csv(MASTER, usecols=meta+pred, low_memory=False)
Z = df[pred].to_numpy(dtype=np.float32)
sd = np.nanstd(Z,0); sd[sd==0]=1
Z = (Z-np.nanmean(Z,0))/sd

def dm(a,b,c=None):
    M = Z if c is None else Z[:,c]
    o = np.empty(len(a))
    for s in range(0,len(a),20000):
        e=min(s+20000,len(a)); o[s:e]=np.nanmean(np.abs(M[a[s:e]]-M[b[s:e]]),axis=1)
    return o

X = np.radians(df[["lat_snap","long_snap"]].to_numpy())
ind,dis = BallTree(X,metric="haversine").query_radius(X,r=1000/R,return_distance=True)
i = np.repeat(np.arange(len(ind)),[len(a) for a in ind])
j,d = np.concatenate(ind), np.concatenate(dis)*R
k=(j>i)&(d>500); i,j = i[k],j[k]
ra,rb = rng.integers(0,len(df),200000), rng.integers(0,len(df),200000)
mk = ra!=rb; ra,rb = ra[mk],rb[mk]

print("=== 1. Denv normalizovan referencom ISTOG podskupa ===")
print(f"{'grupa':<14}{'n':>5}{'denv 500-1k':>13}{'random':>10}{'udeo':>8}")
grp = [(g,[n for n,p in enumerate(pred) if g in p]) for g in ("CLI","TOP","SOL","LAC")]
grp += [("l_ lokalne",[n for n,p in enumerate(pred) if p.startswith("l_")]),
        ("u_ uzvodne",[n for n,p in enumerate(pred) if p.startswith("u_")])]
for g,c in grp:
    if not c: continue
    o,b = np.nanmean(dm(i,j,c)), np.nanmean(dm(ra,rb,c))
    print(f"{g:<14}{len(c):>5}{o:>13.4f}{b:>10.4f}{o/b:>8.3f}")

print("\n=== 2. iste koordinate, razlicit subc_id ===")
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
amb = df.groupby("_k").subc_id.nunique()
bad = df[df._k.isin(amb[amb>1].index)]
print(f"zapisa na ambigvitetnim pozicijama: {len(bad)} ({len(bad)/len(df)*100:.1f}%)")
g = bad.groupby("_k")
print("  isti WoCID?           ", round(g.WoCID.nunique().eq(1).mean(),3))
print("  ista vrsta?           ", round(g.Crayfish_scientific_name.nunique().eq(1).mean(),3))
print("  ista godina?          ", round(g.Year_of_record.nunique().eq(1).mean(),3))
print("  isti basin_id?        ", round(g.basin_id.nunique().eq(1).mean(),3))
print("  isti strahler?        ", round(g.strahler.nunique().eq(1).mean(),3))
print("  isto distance_m?      ", round(g.distance_m.nunique().eq(1).mean(),3))
print("\n  po vrstama (top 8):")
print(bad.Crayfish_scientific_name.value_counts().head(8).to_string())
print("\n  primer jedne pozicije:")
ex = amb.idxmax()
print(df[df._k==ex][["WoCID","Crayfish_scientific_name","Year_of_record","lat_or","long_or",
                     "distance_m","subc_id","strahler","basin_id"]].head(8).to_string(index=False))

print("\n=== 3. kako je pool zapravo definisan u kodu ===")
