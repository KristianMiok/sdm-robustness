import glob
import numpy as np, pandas as pd
fs = sorted(glob.glob("results/revision/t6_fold_*_random_forest.csv"))
if not fs:
    raise SystemExit("jos nema rezultata")
for f in fs:
    D = pd.read_csv(f)
    ent = f.split("t6_fold_")[1].rsplit("_random_forest", 1)[0].replace("_", " ")
    b0 = D[D.level == 0].set_index(["rep", "regime"]).auc.rename("b0")
    D = D.merge(b0, left_on=["rep", "regime"], right_index=True, how="left")
    D["drop"] = D.auc - D.b0
    print(f"\n=== {ent}  ({D.rep.nunique()} replika) ===")
    l0 = D[D.level == 0].groupby("regime").auc.agg(["mean", "std"])
    print("  AUC L0:", {k: (round(v['mean'], 4), round(v['std'], 4)) for k, v in l0.iterrows()})
    for lv in (3, 10, 20):
        a = D[(D.level == lv) & (D.regime == "fixed")]["drop"]
        b = D[(D.level == lv) & (D.regime == "floating")]["drop"]
        if not len(a) or not len(b): continue
        se = np.sqrt(a.var()/len(a) + b.var()/len(b))
        z = (a.mean() - b.mean())/se if se else np.nan
        print(f"  L{lv:<3} floating {b.mean():+.4f} (sd {b.std():.4f}) |"
              f" fixed {a.mean():+.4f} (sd {a.std():.4f}) |"
              f" razlika {a.mean()-b.mean():+.4f}  z {z:+.2f}")
