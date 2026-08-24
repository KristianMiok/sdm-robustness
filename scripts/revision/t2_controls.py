"""Controls before the Denv(d) decomposition can be reported:
random-pair baseline, the zero-distance subc_id ambiguity, full 398-predictor
version, domain breakdown, and pool membership of pathological snaps."""
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
rng = np.random.default_rng(20260822)
hdr = pd.read_csv(MASTER, nrows=0).columns.tolist()
pred = [c for c in hdr if re.match(r"^[lu]_", c)]
meta = ["WoCID", "subc_id", "strahler", "basin_id", "lat_or", "long_or",
        "lat_snap", "long_snap", "distance_m", "Accuracy",
        "Crayfish_scientific_name", "Status"]
df = pd.read_csv(MASTER, usecols=meta + pred, low_memory=False)

Z = df[pred].to_numpy(dtype=np.float32)
sd = np.nanstd(Z, 0); sd[sd == 0] = 1
Z = (Z - np.nanmean(Z, 0)) / sd
keep = ~np.all(np.isnan(Z), axis=0)
Z, predk = Z[:, keep], [p for p, k in zip(pred, keep) if k]
print(f"zapisa {len(df)} | prediktora koriscenih {Z.shape[1]} od {len(pred)}")

def dmean(a, b, cols=None):
    M = Z if cols is None else Z[:, cols]
    out = np.empty(len(a), dtype=np.float64)
    for s in range(0, len(a), 20000):
        e = min(s + 20000, len(a))
        out[s:e] = np.nanmean(np.abs(M[a[s:e]] - M[b[s:e]]), axis=1)
    return out

print("\n=== 1. referenca: nasumicni parovi ===")
n = 300000
ra, rb = rng.integers(0, len(df), n), rng.integers(0, len(df), n)
m = ra != rb
base = dmean(ra[m], rb[m])
print(f"  E|dz| nasumicni par:            {np.nanmean(base):.4f}")
sb = df.basin_id.to_numpy()
mb = m & (sb[ra] == sb[rb])
print(f"  E|dz| nasumicno, isti basin:    {np.nanmean(dmean(ra[mb], rb[mb])):.4f}  (n={mb.sum()})")

print("\n=== 2. nulti pomeraj, razlicit subc_id ===")
key = df.lat_snap.round(6).astype(str) + "," + df.long_snap.round(6).astype(str)
g = df.assign(_k=key).groupby("_k")
amb = g.subc_id.nunique()
print(f"  jedinstvenih snapped pozicija: {len(amb)}")
print(f"  pozicija sa >1 subc_id:        {(amb > 1).sum()}  ({(amb>1).mean()*100:.2f}%)")
print(f"  max subc_id na jednoj poziciji:{amb.max()}")
sus = amb[amb > 1].index[:200]
sub = df[df._k.isin(sus)] if "_k" in df else df.assign(_k=key).query("_k in @sus")
print("  strahler tih pozicija:", sub.strahler.value_counts().head(4).to_dict())
print("  distance_m tih pozicija — median %.1f, max %.1f"
      % (sub.distance_m.median(), sub.distance_m.max()))
print("  da li im se i originalne koordinate poklapaju:",
      round(sub.assign(ko=sub.lat_or.round(6).astype(str)+","+sub.long_or.round(6).astype(str))
            .groupby("_k").ko.nunique().eq(1).mean(), 3))

print("\n=== 3. puna dekompozicija, 398 prediktora, bez nultog opsega ===")
X = np.radians(df[["lat_snap", "long_snap"]].to_numpy())
ind, dis = BallTree(X, metric="haversine").query_radius(X, r=1000/R, return_distance=True)
i = np.repeat(np.arange(len(ind)), [len(a) for a in ind])
j, d = np.concatenate(ind), np.concatenate(dis) * R
k = (j > i) & (d > 0)
i, j, d = i[k], j[k], d[k]
sc, st = df.subc_id.to_numpy(), df.strahler.to_numpy()
P = pd.DataFrame({"band": pd.cut(d, [0, 50, 100, 250, 500, 1000]),
                  "denv": dmean(i, j), "subc": sc[i] != sc[j], "stra": st[i] != st[j],
                  "bas": sb[i] != sb[j]})
t = P.groupby("band", observed=True).agg(
    n=("denv", "size"), P_subc=("subc", "mean"), P_strahler=("stra", "mean"),
    P_basin=("bas", "mean"), denv=("denv", "mean"))
t["denv_uslovno"] = P[P.subc].groupby("band", observed=True).denv.mean()
t["udeo_reference"] = (t.denv / np.nanmean(base))
print(t.round(4).to_string())

print("\n=== 4. po domenu i po lokalno/uzvodno ===")
for lab, sel in (("CLI", "CLI"), ("TOP", "TOP"), ("SOL", "SOL"), ("LAC", "LAC")):
    c = [n for n, p in enumerate(predk) if sel in p]
    if c:
        print(f"  {lab:<4} n={len(c):>3}  denv(500-1000m)="
              f"{np.nanmean(dmean(i[(d>500)], j[(d>500)], c)):.4f}")
for lab, pre in (("lokalne l_", "l_"), ("uzvodne u_", "u_")):
    c = [n for n, p in enumerate(predk) if p.startswith(pre)]
    print(f"  {lab:<11} n={len(c):>3}  denv(500-1000m)="
          f"{np.nanmean(dmean(i[(d>500)], j[(d>500)], c)):.4f}")

print("\n=== 5. gde zavrsavaju zapisi sa snapping > 1000 m ===")
o = df[df.distance_m > 1000]
ENT = df.Crayfish_scientific_name.value_counts()
print(o.groupby(["Crayfish_scientific_name", "Accuracy", "Status"]).size()
       .sort_values(ascending=False).head(10).to_string())
print("\n  raspodela po opsezima koji definisu pool-ove:")
for lo, hi in ((0,200),(200,1000),(1000,10**9)):
    s = df[(df.distance_m > lo) & (df.distance_m <= hi)]
    print(f"   {lo:>5}-{hi if hi<10**9 else 'inf':>5} m: n={len(s):>6}  "
          f"High={int((s.Accuracy=='High').sum()):>6}  Low={int((s.Accuracy=='Low').sum()):>6}")
