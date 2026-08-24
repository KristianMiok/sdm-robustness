"""Non-circular test: does each record's assigned environment match the
climate of its own spatial neighbourhood? Local mis-join vs long-range mis-join."""
import re
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
hdr = pd.read_csv(MASTER, nrows=0).columns.tolist()
CLI = [c for c in hdr if re.match(r"^[lu]_CLI", c)]
TOP = [c for c in hdr if re.match(r"^[lu]_TOP", c)]
meta = ["WoCID","subc_id","reg_id","basin_id","strahler","area_sqm",
        "lat_snap","long_snap","distance_m","Accuracy","Crayfish_scientific_name"]
df = pd.read_csv(MASTER, usecols=meta+CLI+TOP, low_memory=False).reset_index(drop=True)
df["_k"] = df.lat_snap.round(6).astype(str)+","+df.long_snap.round(6).astype(str)
df["_amb"] = df._k.map(df.groupby("_k").subc_id.nunique()) > 1
ptp = df[df._amb].groupby("_k").subc_id.agg(lambda s: np.ptp(s.unique()))
df["_far"] = df._k.map(ptp) > 100000

def zed(cols):
    Z = df[cols].to_numpy(dtype=np.float32)
    sd = np.nanstd(Z,0); sd[sd==0] = 1
    return (Z-np.nanmean(Z,0))/sd
Zc, Zt = zed(CLI), zed(TOP)

X = np.radians(df[["lat_snap","long_snap"]].to_numpy())
tree = BallTree(X, metric="haversine")
K = 30
nd, ni = tree.query(X, k=K)
nd = nd*R
sc = df.subc_id.to_numpy()

def local_mismatch(Z, radius):
    out = np.full(len(df), np.nan)
    for a in range(0, len(df), 5000):
        b = min(a+5000, len(df))
        idx, dd = ni[a:b], nd[a:b]
        ok = (dd <= radius) & (sc[idx] != sc[a:b, None])
        for r in range(b-a):
            sel = idx[r][ok[r]]
            if len(sel) < 3: continue
            out[a+r] = np.nanmean(np.abs(Z[a+r] - np.nanmedian(Z[sel], axis=0)))
    return out

for lab, Z, radius in (("CLI @5km", Zc, 5000), ("CLI @20km", Zc, 20000), ("TOP @5km", Zt, 5000)):
    mm = local_mismatch(Z, radius)
    print(f"\n=== {lab} — odstupanje od lokalnog susedstva ===")
    grp = {"cisti": ~df._amb, "ambig BLIZU id": df._amb & ~df._far,
           "ambig DALEKO id": df._amb & df._far}
    for g, m in grp.items():
        v = mm[m.to_numpy()]
        v = v[~np.isnan(v)]
        if not len(v): continue
        print(f"  {g:<16} n={len(v):>6}  median {np.median(v):.4f}"
              f"  p90 {np.percentile(v,90):.4f}  udeo>1SD {np.mean(v>1):.3f}")
    ref = mm[(~df._amb).to_numpy()]; ref = ref[~np.isnan(ref)]
    thr = np.percentile(ref, 99)
    bad = (mm > thr)
    print(f"  prag = p99 cistih = {thr:.4f}")
    print(f"  zapisa iznad praga: {np.nansum(bad)} ({np.nanmean(bad)*100:.2f}%)"
          f"  |  medju ambig: {np.nanmean(bad[df._amb.to_numpy()])*100:.1f}%"
          f"  medju cistim: {np.nanmean(bad[(~df._amb).to_numpy()])*100:.1f}%")
    if lab == "CLI @20km":
        df["_bad"] = bad
        hi = df.Accuracy.astype(str).str.strip().str.lower().eq("high")
        print("\n  --- izlozenost pool-ova (CLI @20km) ---")
        for pl, s in (("benchmark High<=200m", df[hi & (df.distance_m<=200)]),
                      ("snap_pool High 200-1k", df[hi & (df.distance_m>200) & (df.distance_m<=1000)]),
                      ("lowacc_pool", df[~hi])):
            print(f"   {pl:<23} n={len(s):>6}  sumnjivih {s._bad.mean()*100:5.2f}%")
        ENT = ["Astacus astacus","Pontastacus leptodactylus","Procambarus clarkii",
               "Austropotamobius torrentium","Austropotamobius fulcisianus","Faxonius limosus",
               "Pacifastacus leniusculus","Cambarus latimanus","Cambarus striatus",
               "Creaserinus fodiens","Lacunicambarus diogenes"]
        e = df[df.Crayfish_scientific_name.isin(ENT) & hi & (df.distance_m<=200)]
        print("\n  --- po analitickim vrstama, samo benchmark ---")
        print(e.groupby("Crayfish_scientific_name")._bad.agg(["size","mean"]).round(4).to_string())
