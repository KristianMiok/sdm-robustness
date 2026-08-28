"""T7.1: reproducible screening workflow, taxa to analytical entities.

R1 could not reconstruct how 456 taxa became 13 entities because the criteria
were in code and configuration but not in the manuscript. This writes the
cascade with every threshold stated numerically and every drop counted.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, yaml
from pathlib import Path

A = Path("results/task1_audit/revision")
inv = pd.read_csv(A / "species_audit_full.csv")
sh = pd.read_csv(A / "candidate_shortlist.csv")
g = yaml.safe_load(open("configs/task1_gates.yaml"))
panel = pd.read_csv("config/final_panel.csv")

print(f"=== ulaz: {len(inv)} taksona ===\n")
print("kriterijumi (configs/task1_gates.yaml):")
m = g["gate_1_minimum_benchmark"]
print(f"  gate 1  benchmark >= {m['widespread']} (widespread) / "
      f"{m['regional']} (regional) / {m['endemic']} (endemic)")
print(f"  gate 2  snapping pool supports 5% contamination")
print(f"  gate 3  low-accuracy pool supports 20% contamination")
print(f"  gate 4  >= {g['gate_4_basin_spread']['min_basins']} basins")
print(f"  gate 5  >= {g['gate_5_strahler_spread']['min_distinct_orders']} distinct Strahler orders")

cols = [c for c in sh.columns if c.startswith("gate_")]
print(f"\n=== kaskada ===")
alive = pd.Series(True, index=sh.index)
rows = []
for c in cols:
    passed = sh[c] == 1
    drop = int((alive & ~passed).sum())
    alive = alive & passed
    rows.append(dict(gate=c, dropped=drop, remaining=int(alive.sum())))
    print(f"  {c:<24} odbaceno {drop:>4}  ostalo {int(alive.sum()):>4}")
C = pd.DataFrame(rows)

print(f"\n=== klasifikacija ===")
print(sh.classification.value_counts().to_string())
print(f"\n=== panel: {len(panel)} entiteta iz "
      f"{panel.entity.str.split(' (', regex=False).str[0].nunique()} vrsta ===")
print(panel[["entity","type","category","run_snapping","run_lowacc","notes"]].to_string(index=False))

out = Path("results/revision"); out.mkdir(parents=True, exist_ok=True)
C.to_csv(out / "t7_1_cascade.csv", index=False)
sh.to_csv(out / "t7_1_shortlist.csv", index=False)
print(f"\nzapisano {out}/t7_1_cascade.csv")

print("\n=== od shortlist do panela ===")
elig = sh[sh.classification != "INELIGIBLE"]
print(f"  eligible {len(elig)}, panel {len(panel)}")
pn = set(panel.entity.str.split(" (", regex=False).str[0])
extra = sorted(set(elig.species) - pn)
print(f"  eligible ali ne u panelu: {len(extra)}")
for s in extra[:20]:
    r = elig[elig.species == s].iloc[0]
    print(f"    {s:<38} {r.classification:<16} n={r.n_clean_dedup_200m:.0f} "
          f"basena={r.n_basins:.0f}")
