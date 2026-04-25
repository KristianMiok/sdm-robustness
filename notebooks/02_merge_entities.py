"""
Merge per-entity results: combine RF+XGB + 36 Maxent sub-jobs into
results/task5_execution/<entity>/results_raw.parquet.

Skips entities that are incomplete (missing any of the 36 maxent sub-jobs).
"""
import pandas as pd
import glob
from pathlib import Path

ENTITIES = [
    "astacus_astacus",
    "Pontastacus_leptodactylus_pooled",
    "Austropotamobius_torrentium_pooled",
    "Austropotamobius_fulcisianus_pooled",
    "Procambarus_clarkii_native",
    "Procambarus_clarkii_alien",
    "Pacifastacus_leniusculus_alien",
    "Faxonius_limosus_alien",
]

base = Path("results/task5_execution")
print(f"{'Entity':45s} {'RF+XGB':>8s} {'Maxent':>8s} {'Status':>10s}")
print("-" * 75)

for ent in ENTITIES:
    rfxgb_path = base / f"{ent}_rfxgb" / "results_raw.parquet"
    mx_paths = sorted(glob.glob(str(base / f"{ent}_maxent_*" / "results_raw.parquet")))

    has_rfxgb = rfxgb_path.exists()
    n_mx = len(mx_paths)
    expected_mx = 36
    complete = has_rfxgb and n_mx == expected_mx
    status = "OK" if complete else f"PARTIAL ({n_mx}/{expected_mx})"

    print(f"{ent:45s} {'yes' if has_rfxgb else 'no':>8s} {n_mx:>8d} {status:>10s}")

    if not complete:
        continue

    rfxgb = pd.read_parquet(rfxgb_path)
    mx_frames = [pd.read_parquet(p) for p in mx_paths]
    maxent = pd.concat(mx_frames, ignore_index=True)
    merged = pd.concat([rfxgb, maxent], ignore_index=True)

    out = base / ent / "results_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
