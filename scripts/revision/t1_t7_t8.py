"""T1.1/T1.2 snapping distributions and breakpoint, T7.2 redundancy, T8.5 full-vector importance."""
import glob
import numpy as np, pandas as pd
from sklearn.neighbors import BallTree
from scipy.stats import spearmanr
from sdm_robustness.execution import runner as R

Rk = 6371008.8
ENT = [("Astacus astacus",""),("Austropotamobius fulcisianus","pooled"),
       ("Austropotamobius torrentium","pooled"),("Cambarus latimanus",""),
       ("Cambarus striatus",""),("Creaserinus fodiens",""),("Faxonius limosus","alien"),
       ("Faxonius limosus","native"),("Lacunicambarus diogenes",""),
       ("Pacifastacus leniusculus","alien"),("Pontastacus leptodactylus","pooled"),
       ("Procambarus clarkii","alien"),("Procambarus clarkii","native")]
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False, usecols=[
    "WoCID","subc_id","basin_id","strahler","lat_snap","long_snap","distance_m",
    "Accuracy","Status","Crayfish_scientific_name"])
hi = R._is_high_accuracy(M.Accuracy)

def bench(sp, tr):
    d = M[M.Crayfish_scientific_name == sp]
    if tr == "native": d = d[d.Status.isin(R.NATIVE_VALUES)]
    elif tr == "alien": d = d[d.Status.isin(R.ALIEN_VALUES)]
    return R._dedup_by_subc(d[R._is_high_accuracy(d.Accuracy) & (d.distance_m <= 200)])

print("=== T1.1 snapping rastojanja, High-accuracy ===")
H = M[hi]
print(H.distance_m.describe(percentiles=[.25,.5,.75,.9,.95,.99]).round(1).to_string())
print("\nudeo ispod praga:", {t: round(float((H.distance_m<=t).mean()),4) for t in (50,100,200,500,1000)})
print("\npo entitetu (median / p90 / %<=200):")
for sp, tr in ENT:
    b = bench(sp, tr)
    d = M[(M.Crayfish_scientific_name==sp)]
    d = d[R._is_high_accuracy(d.Accuracy)]
    print(f"  {sp+' '+tr:<36} n={len(d):>6} med={d.distance_m.median():>6.1f}"
          f" p90={d.distance_m.quantile(.9):>7.1f} <=200:{(d.distance_m<=200).mean()*100:>5.1f}%")

print("\n=== T1.2 prelom u P(promena subcatchmenta) vs rastojanje ===")
X = np.radians(M[["lat_snap","long_snap"]].to_numpy())
ind, dis = BallTree(X, metric="haversine").query_radius(X, r=1500/Rk, return_distance=True)
i = np.repeat(np.arange(len(ind)), [len(a) for a in ind])
j, d = np.concatenate(ind), np.concatenate(dis)*Rk
k = (j > i) & (d > 0); i, j, d = i[k], j[k], d[k]
sc = M.subc_id.to_numpy()
ch = sc[i] != sc[j]
bins = np.array([0,25,50,75,100,150,200,300,400,500,750,1000,1500])
mid = (bins[:-1]+bins[1:])/2
p = np.array([ch[(d>=a)&(d<b)].mean() if ((d>=a)&(d<b)).sum()>200 else np.nan
              for a,b in zip(bins[:-1],bins[1:])])
n = np.array([int(((d>=a)&(d<b)).sum()) for a,b in zip(bins[:-1],bins[1:])])
print(pd.DataFrame({"od":bins[:-1],"do":bins[1:],"n":n,"P(promena)":p.round(4)}).to_string(index=False))
ok = np.isfinite(p)
best = min(((np.sum((p[ok][mid[ok]<=c]-p[ok][mid[ok]<=c].mean())**2)
             + np.sum((p[ok][mid[ok]>c]-p[ok][mid[ok]>c].mean())**2), c)
            for c in mid[ok][1:-1]))
print(f"  najbolji prelom (dvosegmentni SSE): {best[1]:.0f} m")

print("\n=== T7.2 prostorna redundansa po entitetu ===")
rows = []
for sp, tr in ENT:
    b = bench(sp, tr)
    if len(b) < 3: continue
    P = np.radians(b[["lat_snap","long_snap"]].to_numpy())
    nn = BallTree(P, metric="haversine").query(P, k=2)[0][:,1]*Rk
    rows.append({"entity": f"{sp} {tr}".strip(), "n": len(b), "subc": b.subc_id.nunique(),
                 "basena": b.basin_id.nunique(), "n/basen": round(len(b)/b.basin_id.nunique(),1),
                 "NN_med_m": round(float(np.median(nn)),1),
                 "NN_p10_m": round(float(np.percentile(nn,10)),1),
                 "%NN<1km": round(float((nn<1000).mean()),3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== T8.5 puni vektor vs top-k Jaccard ===")
fs = glob.glob("results/grid_b_full/**/variable_importance_vectors.parquet", recursive=True)
print("fajlova:", len(fs))
V = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in fs], ignore_index=True)
print("redova:", len(V), "| kolone:", list(V.columns))
print("po osi:", V.axis.value_counts().to_dict(), "| algoritmi:", V.algorithm.unique().tolist())
V = V[V.algorithm.isin(["random_forest","xgboost"]) & (V.track=="combined")]
base = V[V.axis=="benchmark"].groupby(["entity","algorithm","variable"]).importance.mean()
if base.empty:
    print("  nema benchmark vektora — koristi se level 0 iste ose")
    base = V[V.level==0].groupby(["entity","algorithm","variable"]).importance.mean()
out = []
for (e,a,ax,lv,rep), g in V[V.level>0].groupby(["entity","algorithm","axis","level","replicate"]):
    try: b = base.loc[(e,a)]
    except KeyError: continue
    m = g.set_index("variable").importance.reindex(b.index)
    if m.notna().sum() < 50: continue
    rho = spearmanr(m.fillna(0), b).statistic
    t5 = len(set(m.nlargest(5).index) & set(b.nlargest(5).index))/len(set(m.nlargest(5).index)|set(b.nlargest(5).index))
    t10 = len(set(m.nlargest(10).index) & set(b.nlargest(10).index))/len(set(m.nlargest(10).index)|set(b.nlargest(10).index))
    out.append(dict(entity=e, algorithm=a, axis=ax, level=lv, rho_full=rho, jac5=t5, jac10=t10))
O = pd.DataFrame(out)
if len(O):
    O.to_csv("results/revision/t8_importance_full.csv", index=False)
    print(O.groupby(["axis","level","algorithm"])[["rho_full","jac5","jac10"]].mean().round(4).to_string())
    print("\n  varijabilnost (SD kroz replike, prosek po celiji):")
    print(O.groupby(["axis","level","algorithm"])[["rho_full","jac5","jac10"]].std().round(4).to_string())
