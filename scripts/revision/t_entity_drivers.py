"""Corrected: what predicts the per-entity range inflation at lowacc L20?"""
import re
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R

RG = "range_area_pct_change_05"
B = pd.read_parquet("results/grid_b_merged/grid_b_results_raw_merged.parquet",
                    engine="fastparquet")
B = B[B.status == "ok"].drop_duplicates(["entity","algorithm","track","axis","level","replicate"])
low20 = B[(B.axis=="lowacc") & (B.level==20) & (B.track=="combined")]

hdr = pd.read_csv("data/combined_data_true_master.csv", nrows=0).columns.tolist()
pred = [c for c in hdr if re.match(r"^[lu]_", c)]
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False, usecols=[
    "subc_id","basin_id","strahler","distance_m","Accuracy","Status",
    "Crayfish_scientific_name","hylak_id","Year_of_record"]+pred)
Z = M[pred].to_numpy(dtype=np.float32)
sd = np.nanstd(Z,0); sd[sd==0]=1
Z = (Z-np.nanmean(Z,0))/sd
P = pd.read_csv("config/final_panel.csv")

def cv(a,b):
    t = pd.crosstab(pd.concat([a,b]), ["A"]*len(a)+["B"]*len(b))
    if t.shape[0] < 2: return np.nan
    n = t.values.sum(); e = np.outer(t.sum(1), t.sum(0))/n
    return float(np.sqrt(((t.values-e)**2/np.where(e==0,np.nan,e)).sum()/(n*(min(t.shape)-1))))

rows = []
for _, r in P[P.run_lowacc==1].iterrows():
    sp = r.entity.split(" (")[0]
    m = M.Crayfish_scientific_name == sp
    if r.treatment=="native_only": m &= M.Status.isin(R.NATIVE_VALUES)
    elif r.treatment=="alien_only": m &= M.Status.isin(R.ALIEN_VALUES)
    d = M[m]; hi = R._is_high_accuracy(d.Accuracy)
    b = R._dedup_by_subc(d[hi & (d.distance_m<=200)])
    l = R._dedup_by_subc(d[~hi])
    if not len(l): continue
    denv = float(np.nanmean(np.abs(np.nanmean(Z[b.index],0) - np.nanmean(Z[l.index],0))))
    rows.append(dict(entity=r.entity, bench=len(b), low=len(l), low_pct=len(l)/len(b)*100,
        basena=b.basin_id.nunique(), po_basenu=len(b)/b.basin_id.nunique(),
        V_strahler=cv(b.strahler, l.strahler), denv_pool=denv,
        d_str1=(l.strahler==1).mean()*100-(b.strahler==1).mean()*100,
        d_lake=(l.hylak_id.notna().mean()-b.hylak_id.notna().mean())*100,
        d_god=float(l.Year_of_record.median()-b.Year_of_record.median())))
E = pd.DataFrame(rows).set_index("entity")

for alg in ("maxent","random_forest","xgboost"):
    E["eff_"+alg] = low20[low20.algorithm==alg].groupby("entity")[RG].mean()
print(E.round(2).to_string())
print("\n=== Spearman: sta predvidja efekat (n=8, indikativno) ===")
c = E.corr(method="spearman")[[f"eff_{a}" for a in ("maxent","random_forest","xgboost")]]
print(c.drop(index=[f"eff_{a}" for a in ("maxent","random_forest","xgboost")]).round(3).to_string())

print("\n=== bez A. torrentium ===")
E2 = E.drop(index=[i for i in E.index if "torrentium" in i])
c2 = E2.corr(method="spearman")[[f"eff_{a}" for a in ("maxent","random_forest","xgboost")]]
print(c2.drop(index=[f"eff_{a}" for a in ("maxent","random_forest","xgboost")]).round(3).to_string())

print("\n=== po nivou: raste li heterogenost sa dozom ===")
for lv in (3,10,20):
    s = B[(B.axis=="lowacc") & (B.level==lv) & (B.track=="combined") & (B.algorithm=="xgboost")]
    e = s.groupby("entity")[RG].mean()
    print(f"  L{lv:<3} mean {e.mean():>7.2f}  median {e.median():>7.2f}"
          f"  CV {e.std()/e.mean():>5.2f}  min {e.min():>6.2f}  max {e.max():>7.2f}")
