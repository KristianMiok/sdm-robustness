"""Is the coordinate/subc_id ambiguity a locality-centroid artefact
or boundary assignment? And how exposed are the analytical pools?"""
import re
import numpy as np, pandas as pd
from itertools import combinations

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
ENT = ["Astacus astacus","Austropotamobius fulcisianus","Austropotamobius torrentium",
       "Cambarus latimanus","Cambarus striatus","Creaserinus fodiens","Faxonius limosus",
       "Lacunicambarus diogenes","Pacifastacus leniusculus","Pontastacus leptodactylus",
       "Procambarus clarkii"]
hdr = pd.read_csv(MASTER, nrows=0).columns.tolist()
pred = [c for c in hdr if re.match(r"^[lu]_", c)]
df = pd.read_csv(MASTER, low_memory=False, usecols=[
    "WoCID","subc_id","strahler","basin_id","lat_or","long_or","lat_snap","long_snap",
    "distance_m","Accuracy","Crayfish_scientific_name","Status","Year_of_record"]+pred)
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
amb_k = df.groupby("_k").subc_id.nunique()
df["_amb"] = df._k.map(amb_k) > 1
print(f"zapisa {len(df)} | ambigvitetnih {df._amb.sum()} ({df._amb.mean()*100:.1f}%)")

print("\n=== A. preciznost koordinata ===")
def dec(v, k=8):
    for n in range(k+1):
        if np.isclose(v, np.round(v, n), atol=1e-12): return n
    return k
for lab, s in (("ambigvitetni", df[df._amb]), ("ostali", df[~df._amb])):
    d = s.lat_or.head(20000).map(dec)
    print(f"  {lab:<13} median decimala {d.median():.0f} | <=4 dec: {(d<=4).mean():.3f} | <=2: {(d<=2).mean():.3f}")

print("\n=== B. koliko su medjusobno udaljeni subcatchmenti jedne koordinate ===")
loc = df[~df._amb].groupby("subc_id")[["lat_snap","long_snap"]].mean()
print(f"  subc_id lociranih preko cistih zapisa: {len(loc)}")
res = []
for k, g in df[df._amb].groupby("_k"):
    ids = [s for s in g.subc_id.unique() if s in loc.index]
    if len(ids) < 2: continue
    P = np.radians(loc.loc[ids].to_numpy())
    dd = [2*R*np.arcsin(np.sqrt(np.sin((a[0]-b[0])/2)**2 +
          np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2))
          for a, b in combinations(P, 2)]
    res.append((k, len(ids), np.max(dd)))
r = pd.DataFrame(res, columns=["k","n_subc","max_m"])
print(f"  grupa merljivo: {len(r)}")
print(r.max_m.describe(percentiles=[.1,.25,.5,.75,.9]).round(1).to_string())
print("  udeo grupa gde su subcatchmenti >1 km jedan od drugog:", round((r.max_m>1000).mean(),3))
print("  udeo >5 km:", round((r.max_m>5000).mean(),3))

print("\n=== C. koliko se prediktori razlikuju unutar ambigvitetne grupe ===")
Z = df[pred].to_numpy(dtype=np.float32)
sd = np.nanstd(Z,0); sd[sd==0]=1; Z=(Z-np.nanmean(Z,0))/sd
rng = np.random.default_rng(7)
pa, pb = [], []
for k, g in df[df._amb].groupby("_k"):
    ix = g.index.to_numpy()
    if len(ix) < 2: continue
    a, b = rng.choice(ix, 2, replace=False)
    pa.append(a); pb.append(b)
pa, pb = np.array(pa), np.array(pb)
print(f"  parova unutar grupe: {len(pa)}")
print(f"  E|dz| unutar ambigvitetne grupe: {np.nanmean(np.abs(Z[pa]-Z[pb])):.4f}")
sb = df.basin_id.to_numpy()
ra, rb = rng.integers(0,len(df),200000), rng.integers(0,len(df),200000)
m = (ra!=rb)&(sb[ra]==sb[rb])
print(f"  E|dz| nasumicno u istom slivu:   {np.nanmean(np.abs(Z[ra[m]]-Z[rb[m]])):.4f}")

print("\n=== D. izlozenost pool-ova ===")
hi = df.Accuracy.astype(str).str.strip().str.lower().eq("high")
for lab, s in (("benchmark  High <=200m", df[hi & (df.distance_m<=200)]),
               ("snap_pool  High 200-1000", df[hi & (df.distance_m>200) & (df.distance_m<=1000)]),
               ("lowacc_pool  ~High", df[~hi])):
    print(f"  {lab:<26} n={len(s):>6}  ambigvitetnih {s._amb.mean()*100:5.1f}%")
print("\n  po analitickim vrstama (udeo ambigvitetnih):")
sub = df[df.Crayfish_scientific_name.isin(ENT)]
print(sub.groupby("Crayfish_scientific_name")._amb.agg(["size","mean"]).round(3).to_string())

print("\n=== E. da li su ambigvitetni vezani za WoCID serije ===")
df["_pfx"] = df.WoCID.str[:2]
t = df.groupby("_pfx")._amb.agg(["size","mean"])
print(t[t["size"]>500].sort_values("mean", ascending=False).head(10).round(3).to_string())
