"""T2 structure probe: what the master file alone establishes about
environmental change under displacement, before any GIS layer arrives."""
import re
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

MASTER, R = "data/combined_data_true_master.csv", 6371008.8
hdr = pd.read_csv(MASTER, nrows=0).columns.tolist()
pred_all = [c for c in hdr if re.match(r"^[lu]_", c)]
probe = pred_all[::16][:25]
meta = ["WoCID", "subc_id", "strahler", "basin_id", "lat_or", "long_or",
        "lat_snap", "long_snap", "distance_m", "Accuracy",
        "Crayfish_scientific_name"]
df = pd.read_csv(MASTER, usecols=meta + probe, low_memory=False)
print(f"zapisa {len(df)} | prediktora ukupno {len(pred_all)} | probe {len(probe)}")

print("\n=== A. jesu li prediktori konstantni unutar subc_id ===")
dup = df[df.duplicated("subc_id", keep=False)]
print(f"zapisa u deljenim subc_id: {len(dup)} ({len(dup)/len(df)*100:.1f}%)")
nu = dup.groupby("subc_id")[probe].nunique()
print(f"udeo grupa gde su SVI probe-prediktori konstantni: {(nu <= 1).all(axis=1).mean():.4f}")
bad = nu[(nu > 1).any(axis=1)]
print(f"grupa sa varijacijom: {len(bad)}")
if len(bad):
    print(bad.head(3).to_string())
print("strahler konstantan unutar subc_id:",
      round((dup.groupby('subc_id').strahler.nunique() <= 1).mean(), 4))

print("\n=== B. duplirane pozicije ===")
for tag, la, lo in (("original", "lat_or", "long_or"), ("snapped", "lat_snap", "long_snap")):
    Xc = np.radians(df[[la, lo]].to_numpy())
    d2, _ = BallTree(Xc, metric="haversine").query(Xc, k=2)
    nn = d2[:, 1] * R
    print(f"  {tag:<9} NN=0: {(nn < 1e-6).mean()*100:5.1f}%   median {np.median(nn):8.1f} m")

print("\n=== C. retention <=200 m po Accuracy ===")
print(df.groupby("Accuracy").distance_m.apply(lambda s: (s <= 200).mean()).round(4).to_string())
print("svi zajedno:", round((df.distance_m <= 200).mean(), 4))

X = np.radians(df[["lat_snap", "long_snap"]].to_numpy())
ind, dis = BallTree(X, metric="haversine").query_radius(X, r=1000/R, return_distance=True)
i = np.repeat(np.arange(len(ind)), [len(a) for a in ind])
j, d = np.concatenate(ind), np.concatenate(dis) * R
m = j > i
i, j, d = i[m], j[m], d[m]
print(f"\n=== D. parovi <=1000 m: {len(i)} ===")

Z = df[probe].to_numpy(dtype=float)
Z = (Z - np.nanmean(Z, 0)) / np.where(np.nanstd(Z, 0) == 0, 1, np.nanstd(Z, 0))
denv = np.nanmean(np.abs(Z[i] - Z[j]), axis=1)
subc, stra, bas = df.subc_id.to_numpy(), df.strahler.to_numpy(), df.basin_id.to_numpy()
band = pd.cut(d, [-.1, 0, 1, 50, 100, 250, 500, 1000])
P = pd.DataFrame({"band": band, "denv": denv, "subc": subc[i] != subc[j],
                  "stra": stra[i] != stra[j], "bas": bas[i] != bas[j]})
print(P.groupby("band", observed=True).agg(
    n=("denv", "size"), P_subc=("subc", "mean"), P_strahler=("stra", "mean"),
    P_basin=("bas", "mean"), denv_mean=("denv", "mean")).round(4).to_string())
print("\n--- Denv uslovno na promenu subcatchmenta ---")
print(P.groupby(["subc", "band"], observed=True).denv.agg(["size", "mean"]).round(4).to_string())

print("\n=== E. patoloski snapping > 1000 m ===")
o = df[df.distance_m > 1000]
print(f"n={len(o)} | High={(o.Accuracy == 'High').sum()} | Low={(o.Accuracy == 'Low').sum()}")
print(o.Crayfish_scientific_name.value_counts().head(6).to_string())
print(o.nlargest(8, "distance_m")[
    ["WoCID", "Accuracy", "Crayfish_scientific_name", "distance_m", "strahler"]
].to_string(index=False))
