import os, sys, json, glob
ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
print("ROOT:", os.path.abspath(ROOT))
print("=== top-level result dirs (name: file count) ===")
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p):
        n = sum(len(f) for _, _, f in os.walk(p))
        print(f"  {d}: {n} files")
# find candidate Grid B core files (S4/S5 feed): look for anything with 'grid' or per-entity metric files
cands = []
for ext in ("*.csv","*.json","*.parquet","*.pkl","*.npz"):
    cands += glob.glob(os.path.join(ROOT, "**", ext), recursive=True)
print(f"\n=== {len(cands)} tabular/serialized files found; sampling up to 5 paths ===")
for c in cands[:5]:
    print(" ", os.path.relpath(c, ROOT))
# print schema of first CSV/JSON we can read
print("\n=== schema of first readable file ===")
for c in cands:
    try:
        if c.endswith(".csv"):
            import csv
            with open(c) as fh:
                head = fh.readline().strip()
            print("CSV:", os.path.relpath(c, ROOT))
            print("  columns:", head)
            with open(c) as fh:
                fh.readline(); print("  row1:", fh.readline().strip())
            break
        elif c.endswith(".json"):
            with open(c) as fh:
                obj = json.load(fh)
            print("JSON:", os.path.relpath(c, ROOT))
            print("  type:", type(obj).__name__)
            if isinstance(obj, dict): print("  keys:", list(obj.keys())[:20])
            elif isinstance(obj, list) and obj: print("  first elem keys:", list(obj[0].keys())[:20] if isinstance(obj[0], dict) else obj[0])
            break
    except Exception as e:
        print("  (skip", os.path.relpath(c, ROOT), "->", e, ")")
