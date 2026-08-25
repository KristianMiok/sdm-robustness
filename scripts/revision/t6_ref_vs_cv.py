"""Reference set vs cross-validation: does the disagreement track basin count?"""
import argparse, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pathlib import Path
from sdm_robustness.execution.runner import run_grid_b_factorial

ap = argparse.ArgumentParser()
ap.add_argument("--entity", required=True)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--alg", default="random_forest")
a = ap.parse_args()

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")
slug = "".join(c for c in a.entity.replace(" ", "_") if c.isalnum() or c == "_")
out = Path(f"results/revision/refcv_{slug}_{a.alg}")
path = run_grid_b_factorial(
    species_panel=P[P.entity == a.entity], master_table=M, output_dir=out,
    algorithms=(a.alg,), scale_tracks=("combined",),
    snap_levels_pct=(0, 1, 2, 5), lowacc_levels_pct=(0, 3, 10, 20),
    n_replicates=a.reps, save_surfaces=False, use_reference_set=True)
D = pd.read_parquet(path, engine="fastparquet")
D.to_csv(out.with_suffix(".csv"), index=False)

b = D[D.axis == "benchmark"]
print(f"\n=== {a.entity} [{a.alg}] {a.reps} replika ===")
print(f"  benchmark: auc {b.auc.mean():.4f} (sd {b.auc.std():.4f}) | "
      f"cv_auc {b.cv_auc.mean():.4f} (sd {b.cv_auc.std():.4f})")
print(f"\n{'osa':<10}{'lvl':>4}{'d_auc_ref':>12}{'sd':>8}{'d_auc_cv':>11}{'sd':>8}")
for (ax, lv), g in D[D.axis != "benchmark"].groupby(["axis", "level"]):
    dr = g.auc.mean() - b.auc.mean()
    dc = g.cv_auc.mean() - b.cv_auc.mean()
    print(f"{ax:<10}{int(lv):>4}{dr:>+12.4f}{g.auc.std():>8.4f}"
          f"{dc:>+11.4f}{g.cv_auc.std():>8.4f}")
