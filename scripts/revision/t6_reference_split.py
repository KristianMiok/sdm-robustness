"""Basin-blocked 20% withholding: can every entity actually support it?"""
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R
from sdm_robustness.utils.repro import derive_seed

TARGET = 0.20
M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
P = pd.read_csv("config/final_panel.csv")

BASIN_CAP = 0.05

def split(bench, entity, target=TARGET, cap=BASIN_CAP):
    """Basin-blocked withholding for T6.

    Greedy over a random permutation of basins, keeping the achieved fraction
    closest to target. Basins holding more than `cap` of the entity's records
    are excluded: without that, the reference set can collapse onto a handful
    of catchments (A. fulcisianus reached 20% in 3 basins of 79), which makes
    the metric a statement about those catchments rather than about predictive
    ability. Deterministic given the entity name.
    """
    sizes = bench.basin_id.astype(str).value_counts()
    rng = np.random.default_rng(derive_seed(20260416, entity, "reference", 0, 0))
    elig = sizes.index[sizes <= cap * len(bench)] if cap else sizes.index
    if len(elig) == 0:
        elig = sizes.index
    bas = rng.permutation(np.asarray(elig))
    held, cum, goal = [], 0, target * len(bench)
    for bid in bas:
        n = int(sizes[bid])
        if abs(cum + n - goal) < abs(cum - goal):
            held.append(bid); cum += n
    return set(held), cum

rows = []
for _, r in P.iterrows():
    b = R._prepare_entity_data(M, r.entity)["benchmark"]
    held, cum = split(b, r.entity)
    nb = b.basin_id.nunique()
    sz = b.basin_id.astype(str).value_counts()
    rows.append(dict(entity=r.entity, n=len(b), basena=nb,
                     max_basen_pct=round(sz.max()/len(b)*100,1),
                     held_n=cum, held_pct=round(cum/len(b)*100,1),
                     held_basena=len(held), train_n=len(b)-cum))
T = pd.DataFrame(rows)
print(T.to_string(index=False))
print(f"\n  odstupanje od 20%: median {abs(T.held_pct-20).median():.1f} pp,"
      f" max {abs(T.held_pct-20).max():.1f} pp")
print("  entiteti gde jedan basen nosi >20% zapisa:",
      T[T.max_basen_pct > 20].entity.tolist())
