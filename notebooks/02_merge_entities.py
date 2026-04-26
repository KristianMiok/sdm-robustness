"""
Merge per-entity results: combine RF+XGB + Maxent sub-jobs into a single parquet.

Handles both DUAL-AXIS (8 entities, 36 maxent sub-jobs) and SNAPPING-ONLY 
(5 entities, 18 maxent sub-jobs) types.
"""
import pandas as pd
import glob
from pathlib import Path

# 8 DUAL + 5 SNAP = 13 entities total
ENTITIES = [
    ("astacus_astacus", "DUAL"),
    ("Pontastacus_leptodactylus_pooled", "DUAL"),
    ("Austropotamobius_torrentium_pooled", "DUAL"),
    ("Austropotamobius_fulcisianus_pooled", "DUAL"),
    ("Procambarus_clarkii_native", "DUAL"),
    ("Procambarus_clarkii_alien", "DUAL"),
    ("Pacifastacus_leniusculus_alien", "DUAL"),
    ("Faxonius_limosus_alien", "DUAL"),
    ("Lacunicambarus_diogenes", "SNAP"),
    ("Cambarus_latimanus", "SNAP"),
    ("Cambarus_striatus", "SNAP"),
    ("Creaserinus_fodiens", "SNAP"),
    ("Faxonius_limosus_native", "SNAP"),
]

base = Path("results/task5_execution")
print(f"{'Entity':45s} {'Type':>6s} {'RF+XGB':>8s} {'Maxent':>8s} {'Status':>15s}")
print("-" * 90)

for ent, ent_type in ENTITIES:
    rfxgb_path = base / f"{ent}_rfxgb" / "results_raw.parquet"
    mx_paths = sorted(glob.glob(str(base / f"{ent}_maxent_*" / "results_raw.parquet")))

    has_rfxgb = rfxgb_path.exists()
    n_mx = len(mx_paths)
    expected_mx = 36 if ent_type == "DUAL" else 18  # SNAP has 1 axis × 3 tracks × 6 levels = 18
    complete = has_rfxgb and n_mx == expected_mx
    status = "OK" if complete else f"PARTIAL ({n_mx}/{expected_mx})"

    print(f"{ent:45s} {ent_type:>6s} {'yes' if has_rfxgb else 'no':>8s} {n_mx:>8d} {status:>15s}")

    if not complete:
        continue

    rfxgb = pd.read_parquet(rfxgb_path)
    mx_frames = [pd.read_parquet(p) for p in mx_paths]
    maxent = pd.concat(mx_frames, ignore_index=True)
    merged = pd.concat([rfxgb, maxent], ignore_index=True)

    out = base / ent / "results_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
