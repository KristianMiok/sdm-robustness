"""Per-fit Maxent time across the entity size range."""
import time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sdm_robustness.execution import runner as R
from sdm_robustness.pipeline.core import (
    clean_predictors, get_track_columns, build_model, predict_suitability_surface)

M = pd.read_csv("data/combined_data_true_master.csv", low_memory=False)
ENTS = ["Faxonius limosus (native)", "Austropotamobius torrentium (pooled)",
        "Astacus astacus", "Procambarus clarkii (alien)"]
print(f"{'entity':<38}{'pres':>7}{'acc':>8}{'preds':>7}{'fit s':>8}{'pred s':>8}{'run s':>8}")
for ent in ENTS:
    p = R._prepare_entity_data(M, ent)
    b, acc = p["benchmark"], p["accessible_area"]
    kept = clean_predictors(b, get_track_columns(b, "combined"))
    med = b[kept].median(numeric_only=True)
    A = acc[["subc_id"] + kept].copy(); A[kept] = A[kept].fillna(med)
    X = b[kept].fillna(med)
    neg = A.sample(n=min(10000, len(A)), random_state=1)
    xt = pd.concat([X, neg[kept]], axis=0)
    yt = np.array([1]*len(X) + [0]*len(neg))
    m = build_model("maxent", seed=1, n_jobs=-1, maxent_n_cpus=1)
    t = time.time(); m.fit(xt, yt); tf = time.time() - t
    t = time.time(); predict_suitability_surface(m, A[kept]); tp = time.time() - t
    print(f"{ent:<38}{len(b):>7}{len(acc):>8}{len(kept):>7}"
          f"{tf:>8.1f}{tp:>8.1f}{(tf+tp)*6:>8.0f}")
