"""Does fixing the fold map change the measured contamination effect,
or is the single-replicate difference just noise?"""
import argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sdm_robustness.execution import runner as R
from sdm_robustness.execution.cv import assign_basin_folds
from sdm_robustness.pipeline.core import fit_cv_cell
from sdm_robustness.utils.repro import derive_seed

ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True)
ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--alg", default="random_forest")
a = ap.parse_args()

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
p = R._prepare_entity_data(M, a.entity)
fm = assign_basin_folds(p["benchmark"].basin_id, n_splits=5, looo_threshold=15)
print(f"{a.entity}: bench {len(p['benchmark'])} basena {p['benchmark'].basin_id.nunique()}")

rows = []
for rep in range(a.reps):
    for lv in (0, 3, 10, 20):
        kw = dict(benchmark=p["benchmark"], contamination_pool=p["lowacc_pool"],
                  accessible_area=p["accessible_area"], entity=a.entity,
                  algorithm=a.alg, track="combined",
                  axis="benchmark" if lv == 0 else "lowacc", level=lv, replicate=rep,
                  seed=derive_seed(20260416, a.entity, "lowacc", lv, rep),
                  n_experiment=len(p["benchmark"]))
        for tag, extra in (("floating", {}), ("fixed", {"fold_map": fm})):
            r = fit_cv_cell(**kw, **extra)
            rows.append(dict(rep=rep, level=lv, regime=tag,
                             auc=r["auc"], tss=r["tss"]))
    print(f"  rep {rep+1}/{a.reps}", flush=True)

D = pd.DataFrame(rows)
slug = "".join(c for c in a.entity.replace(" ", "_") if c.isalnum() or c == "_")
D.to_csv(f"results/revision/t6_fold_{slug}_{a.alg}.csv", index=False)

print("\n=== AUC po nivou i rezimu ===")
print(D.pivot_table(index="level", columns="regime", values="auc",
                    aggfunc=["mean","std"]).round(4).to_string())
print("\n=== pad u odnosu na L0 iste replike ===")
b0 = D[D.level == 0].set_index(["rep","regime"]).auc.rename("b0")
D = D.merge(b0, left_on=["rep","regime"], right_index=True, how="left")
D["drop"] = D.auc - D.b0
assert D["drop"].notna().all(), "baseline join failed"
t = D[D.level > 0].groupby(["level","regime"]).drop.agg(["mean","std","size"])
print(t.round(4).to_string())
print("\n  razlika (fixed - floating) u izmerenom padu:")
for lv in (3, 10, 20):
    f = D[(D.level==lv)&(D.regime=="fixed")].drop
    g = D[(D.level==lv)&(D.regime=="floating")].drop
    se = np.sqrt(f.var()/len(f) + g.var()/len(g))
    print(f"    L{lv:<3} {f.mean()-g.mean():+.4f}  SE {se:.4f}  z {(f.mean()-g.mean())/se:+.2f}")
