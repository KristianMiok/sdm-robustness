"""Which column is wrong — the coordinate or the subc_id?"""
import numpy as np, pandas as pd

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
df = pd.read_csv(MASTER, low_memory=False, usecols=[
    "WoCID","subc_id","reg_id","basin_id","strahler","lat_or","long_or",
    "lat_snap","long_snap","distance_m","Accuracy","Crayfish_scientific_name","Year_of_record"])
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
df["_amb"] = df._k.map(df.groupby("_k").subc_id.nunique()) > 1
A, C = df[df._amb], df[~df._amb]

def hav(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    return 2*R*np.arcsin(np.sqrt(np.sin((la1-la2)/2)**2 +
           np.cos(la1)*np.cos(la2)*np.sin((lo1-lo2)/2)**2))

print("=== 1. TEST: da li distance_m odgovara sopstvenim koordinatama ===")
for lab, s in (("ambigvitetni", A), ("cisti", C)):
    h = hav(s.lat_or, s.long_or, s.lat_snap, s.long_snap)
    e = (h - s.distance_m).abs()
    print(f"  {lab:<13} n={len(s):>6}  median |greska| {np.nanmedian(e):10.3f} m"
          f"   <1m: {np.nanmean(e<1):.4f}   >100m: {np.nanmean(e>100):.4f}")

print("\n=== 2. duplirani koordinatni blokovi ===")
for cols, lab in ((["lat_or","long_or"],"lat_or+long_or"),
                  (["lat_snap","long_snap"],"lat_snap+long_snap"),
                  (["lat_or","long_or","lat_snap","long_snap","distance_m"],"ceo blok")):
    d = df.duplicated(cols, keep=False)
    print(f"  {lab:<22} duplirano: {d.sum():>6} ({d.mean()*100:5.1f}%)"
          f"  |  medju ambig: {d[df._amb].mean()*100:5.1f}%  medju cistim: {d[~df._amb].mean()*100:5.1f}%")

print("\n=== 3. forward-fill? da li ambigvitetni kopiraju prethodni red ===")
s = df.sort_values("WoCID").reset_index(drop=True)
same_prev = (s.lat_or == s.lat_or.shift(1)) & (s.long_or == s.long_or.shift(1))
print(f"  zapisa sa koordinatom identicnom prethodnom WoCID-u: {same_prev.sum()} ({same_prev.mean()*100:.1f}%)")
print(f"    medju ambigvitetnim: {same_prev[s._amb].mean()*100:.1f}%")
print(f"    medju cistim:        {same_prev[~s._amb].mean()*100:.1f}%")
print(f"  isti test za subc_id: {(s.subc_id==s.subc_id.shift(1)).mean()*100:.1f}%"
      f"  (ambig {(s.subc_id==s.subc_id.shift(1))[s._amb].mean()*100:.1f}%)")

print("\n=== 4. kontinentalna provera ===")
cen = C.groupby("subc_id")[["lat_snap","long_snap"]].mean()
A2 = A[A.subc_id.isin(cen.index)].copy()
A2["sub_lon"] = A2.subc_id.map(cen.long_snap); A2["sub_lat"] = A2.subc_id.map(cen.lat_snap)
def cont(lon): return np.where(lon < -30, "NA", "EU")
A2["c_coord"], A2["c_subc"] = cont(A2.long_snap), cont(A2.sub_lon)
print(f"  merljivo: {len(A2)}  |  kontinent se NE poklapa: {(A2.c_coord!=A2.c_subc).sum()}"
      f" ({(A2.c_coord!=A2.c_subc).mean()*100:.1f}%)")
mis = A2[A2.c_coord != A2.c_subc]
if len(mis):
    print("\n  vrste sa neusaglasenim kontinentom (top 8):")
    print(mis.groupby(["Crayfish_scientific_name","c_coord","c_subc"]).size()
            .sort_values(ascending=False).head(8).to_string())

print("\n=== 5. ima li u grupi tacno jedan 'tacan' zapis ===")
cl = C.groupby("subc_id")[["lat_snap","long_snap"]].mean()
g = A[A.subc_id.isin(cl.index)].copy()
g["d_to_own"] = hav(g.lat_snap, g.long_snap, g.subc_id.map(cl.lat_snap), g.subc_id.map(cl.long_snap))
h = g.groupby("_k").d_to_own.agg(n="size", ok=lambda s: (s < 1000).sum())
print(h.ok.value_counts().head(6).to_string())
print(f"  grupa sa tacno jednim usaglasenim zapisom: {(h.ok==1).mean():.3f}")
print(f"  grupa bez nijednog usaglasenog:            {(h.ok==0).mean():.3f}")

print("\n=== 6. WJ/WK/WI: koliko jedinstvenih koordinata ===")
df["_pfx"] = df.WoCID.str[:2]
t = df.groupby("_pfx").agg(n=("WoCID","size"),
      uniq_coord=("_k","nunique"), uniq_subc=("subc_id","nunique"), amb=("_amb","mean"))
t["zapisa_po_koordinati"] = (t.n/t.uniq_coord).round(2)
print(t[t.n>500].sort_values("amb", ascending=False).round(3).to_string())
