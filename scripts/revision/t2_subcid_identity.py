"""Is subc_id globally unique, or unique only within reg_id?
And does the coordinate match the subcatchment the environment came from?"""
import numpy as np, pandas as pd

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
df = pd.read_csv(MASTER, low_memory=False, usecols=[
    "WoCID","subc_id","reg_id","basin_id","strahler","lat_or","long_or",
    "lat_snap","long_snap","distance_m","Accuracy","Crayfish_scientific_name","Year_of_record"])
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
df["_amb"] = df._k.map(df.groupby("_k").subc_id.nunique()) > 1

print("=== A. je li subc_id globalno jedinstven ===")
g = df.groupby("subc_id")
print("  subc_id ukupno:", df.subc_id.nunique())
for col in ("reg_id","basin_id"):
    n = g[col].nunique()
    print(f"  subc_id sa >1 {col:<9}: {(n>1).sum():>6}  ({(n>1).mean()*100:5.2f}%)  max={n.max()}")
print("  reg_id vrednosti:", sorted(df.reg_id.unique())[:20], "... n =", df.reg_id.nunique())
print("\n  prostorni rasap unutar subc_id (m):")
def spread(s):
    if len(s) < 2: return 0.0
    P = np.radians(s[["lat_snap","long_snap"]].to_numpy())
    la, lo = P[:,0], P[:,1]
    return float(np.max(2*R*np.arcsin(np.sqrt(
        np.sin((la[:,None]-la[None,:])/2)**2 +
        np.cos(la)[:,None]*np.cos(la)[None,:]*np.sin((lo[:,None]-lo[None,:])/2)**2))))
big = [s for s, n in g.size().items() if 2 <= n <= 60]
sp = pd.Series({s: spread(df[df.subc_id==s]) for s in big[:4000]})
print(sp.describe(percentiles=[.5,.9,.99]).round(1).to_string())
print("  udeo subc_id koji se prostiru >1 km:", round((sp>1000).mean(),4))
print("  udeo >100 km:", round((sp>100000).mean(),4))

print("\n=== B. unutar ambigvitetne grupe: isti region? ===")
a = df[df._amb].groupby("_k")
for col in ("reg_id","basin_id","strahler"):
    print(f"  isti {col:<9} u grupi: {a[col].nunique().eq(1).mean():.3f}")
print("  broj razlicitih subc_id po grupi:", a.subc_id.nunique().describe()[["mean","50%","max"]].round(2).to_dict())

print("\n=== C. da li koordinata odgovara subcatchmentu (par subc_id+reg_id) ===")
df["_sr"] = df.subc_id.astype(str)+"|"+df.reg_id.astype(str)
cen = df[~df._amb].groupby("_sr")[["lat_snap","long_snap"]].mean()
def dist_to_cen(s):
    m = s._sr.map(cen.lat_snap).notna()
    s = s[m]
    la1, lo1 = np.radians(s.lat_snap), np.radians(s.long_snap)
    la2, lo2 = np.radians(s._sr.map(cen.lat_snap)), np.radians(s._sr.map(cen.long_snap))
    return 2*R*np.arcsin(np.sqrt(np.sin((la1-la2)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo1-lo2)/2)**2))
for lab, s in (("ambigvitetni", df[df._amb]), ("cisti", df[~df._amb])):
    d = dist_to_cen(s)
    print(f"  {lab:<13} n={len(d):>6}  median {np.nanmedian(d):9.1f} m   >1km {np.nanmean(d>1000):.3f}")

print("\n=== D. sta dedup_by_subc radi ===")
hi = df.Accuracy.astype(str).str.strip().str.lower().eq("high")
bench = df[hi & (df.distance_m<=200)]
print(f"  benchmark pre dedup:            {len(bench)}")
print(f"  posle drop_duplicates(subc_id): {bench.subc_id.nunique()}")
print(f"  posle drop_duplicates(subc+reg): {bench._sr.nunique()}")
print(f"  razlika (izgubljeni zapisi):    {bench._sr.nunique() - bench.subc_id.nunique()}")
print("\n  po analitickim vrstama:")
ENT = ["Astacus astacus","Pontastacus leptodactylus","Procambarus clarkii",
       "Austropotamobius torrentium","Faxonius limosus","Pacifastacus leniusculus"]
for e in ENT:
    s = bench[bench.Crayfish_scientific_name==e]
    if len(s): print(f"    {e:<30} n={len(s):>6}  subc={s.subc_id.nunique():>5}  subc+reg={s._sr.nunique():>5}")

print("\n=== E. WJ/WK/WI blokovi ===")
df["_pfx"] = df.WoCID.str[:2]
t = df.groupby("_pfx").agg(n=("WoCID","size"), amb=("_amb","mean"),
        god_med=("Year_of_record","median"), acc_high=("Accuracy", lambda s: s.eq("High").mean()),
        dist_med=("distance_m","median"), nreg=("reg_id","nunique"))
print(t[t.n>500].sort_values("amb", ascending=False).round(3).to_string())
