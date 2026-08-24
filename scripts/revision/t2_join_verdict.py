"""area_sqm gives an internal scale check that needs no centroid reference."""
import numpy as np, pandas as pd
from itertools import combinations

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
df = pd.read_csv(MASTER, low_memory=False, usecols=[
    "WoCID","subc_id","reg_id","basin_id","strahler","area_sqm","lat_or","long_or",
    "lat_snap","long_snap","distance_m","Accuracy","Crayfish_scientific_name"])
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
df["_amb"] = df._k.map(df.groupby("_k").subc_id.nunique()) > 1

def hav(a1,o1,a2,o2):
    a1,o1,a2,o2 = (np.radians(np.asarray(x,float)) for x in (a1,o1,a2,o2))
    return 2*R*np.arcsin(np.sqrt(np.sin((a1-a2)/2)**2+np.cos(a1)*np.cos(a2)*np.sin((o1-o2)/2)**2))

g = df.groupby("subc_id")
print("=== 1. area_sqm kao lookup ===")
print("  subc_id sa >1 area_sqm:", int((g.area_sqm.nunique()>1).sum()))
a = g.area_sqm.median()
print("  area_sqm:", a.describe(percentiles=[.1,.5,.9]).round(0).to_dict())
print("  implicirani precnik median:", round(float(2*np.sqrt(a.median()/np.pi)),1), "m")

print("\n=== 2. rasap zapisa unutar subc_id vs implicirana velicina ===")
agg = g.agg(n=("WoCID","size"), la0=("lat_snap","min"), la1=("lat_snap","max"),
            lo0=("long_snap","min"), lo1=("long_snap","max"),
            area=("area_sqm","median"), amb=("_amb","max"))
agg["diag"] = hav(agg.la0, agg.lo0, agg.la1, agg.lo1)
agg["impl"] = 2*np.sqrt(agg.area/np.pi)
m = agg.n >= 2
print(f"  subc_id sa >=2 zapisa: {int(m.sum())}")
for lab, s in (("bez ambigvitetnih", agg[m & ~agg.amb]), ("sa ambigvitetnim", agg[m & agg.amb])):
    print(f"  {lab:<19} n={len(s):>6}  median diag {s.diag.median():9.1f} m"
          f"  >2km {np.mean(s.diag>2000):.3f}  >50km {np.mean(s.diag>50000):.3f}"
          f"  diag/impl med {np.median(s.diag/s.impl):.2f}")

print("\n=== 3. stvarna udaljenost subc_id u grupi (samo pouzdani anchor-i) ===")
tight = agg[(agg.n>=2) & (agg.diag<2000)].index
loc = df[df.subc_id.isin(tight)].groupby("subc_id")[["lat_snap","long_snap"]].median()
print(f"  pouzdanih subc_id (rasap <2 km): {len(loc)}")
res = []
for k, s in df[df._amb].groupby("_k"):
    ids = [i for i in s.subc_id.unique() if i in loc.index]
    if len(ids) < 2: continue
    P = loc.loc[ids].to_numpy()
    res.append(max(hav(x[0],x[1],y[0],y[1]) for x,y in combinations(P,2)))
r = pd.Series(res)
print(f"  grupa merljivo: {len(r)}")
if len(r):
    print(r.describe(percentiles=[.25,.5,.75,.9]).round(1).to_string())
    print("  >2 km:", round(float((r>2000).mean()),3), "| >50 km:", round(float((r>50000).mean()),3))

print("\n=== 4. numericka blizina subc_id u grupi ===")
sp = pd.Series([np.ptp(s.subc_id.unique()) for _, s in df[df._amb].groupby("_k")
                if s.subc_id.nunique() > 1])
print(sp.describe(percentiles=[.5,.9]).round(0).to_string())

print("\n=== 5. preciznost koordinata ===")
def dec(v, k=8):
    for n in range(k+1):
        if abs(v-round(v,n)) < 1e-12: return n
    return k
for lab, s in (("ambig", df[df._amb]), ("cisti", df[~df._amb])):
    for c in ("lat_or","lat_snap"):
        d = s[c].head(20000).map(dec)
        print(f"  {lab:<6} {c:<9} median dec {d.median():.0f}  <=3: {(d<=3).mean():.3f}")
