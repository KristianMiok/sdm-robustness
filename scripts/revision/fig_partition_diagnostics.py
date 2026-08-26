"""Display-item data for the evaluation-design section:
(a) largest-basin share vs fold balance, per entity
(b) what unweighted fold averaging does, per entity
(c) reference vs CV effect estimates where available
"""
import glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")
out = Path("results/revision"); out.mkdir(parents=True, exist_ok=True)

rows = []
for _, r in P.iterrows():
    b = R._prepare_entity_data(M, r.entity)["benchmark"]
    s = b.basin_id.astype(str)
    fm = assign_basin_folds(b.basin_id, n_splits=5, looo_threshold=15)
    c = s.map(fm).value_counts()
    tr, ref = R.split_reference_set(b, r.entity, 20260426)
    rows.append(dict(
        entity=r.entity, n=len(b), basins=b.basin_id.nunique(),
        largest_basin_pct=round(s.value_counts().max()/len(b)*100, 1),
        top3_basin_pct=round(s.value_counts().head(3).sum()/len(b)*100, 1),
        fold_min_max=round(c.min()/c.max(), 3),
        fold_min_n=int(c.min()), fold_max_n=int(c.max()),
        gini=round(float(np.abs(np.subtract.outer(c.values, c.values)).mean()
                         / (2*c.values.mean())), 3),
        reference_possible=ref is not None,
        reference_n=len(ref) if ref is not None else 0,
        entity_type=r.type))
T = pd.DataFrame(rows).sort_values("largest_basin_pct", ascending=False)
T.to_csv(out/"fig_partition_diagnostics.csv", index=False)
print("=== (a) najveci basen vs balans foldova ===")
print(T[["entity","n","basins","largest_basin_pct","fold_min_max",
         "fold_min_n","fold_max_n","reference_possible"]].to_string(index=False))
from scipy.stats import spearmanr
print(f"\n  rho(largest_basin_pct, fold_min_max) = "
      f"{spearmanr(T.largest_basin_pct, T.fold_min_max).statistic:+.3f}  n={len(T)}")

print("\n=== (c) referentni vs CV, gde postoji ===")
for f in sorted(glob.glob("results/revision/refcv_*.csv")):
    D = pd.read_csv(f)
    ent = Path(f).stem.replace("refcv_","").rsplit("_random_forest",1)[0]
    ent = ent.rsplit("_xgboost",1)[0].replace("_"," ")
    alg = "xgboost" if "xgboost" in f else "random_forest"
    b = D[D.axis=="benchmark"]
    g = D[D.axis!="benchmark"].groupby(["axis","level"])[["auc","cv_auc"]].mean()
    g["d_ref"] = g.auc - b.auc.mean()
    g["d_cv"] = g.cv_auc - b.cv_auc.mean()
    g["gap"] = g.d_ref - g.d_cv
    print(f"\n  {ent} [{alg}]  bench ref {b.auc.mean():.4f} cv {b.cv_auc.mean():.4f}")
    print(g[["d_ref","d_cv","gap"]].round(4).to_string())
