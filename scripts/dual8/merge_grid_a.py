#!/usr/bin/env python3
"""
Merge all extended-grid (Grid A) per-entity parquets into one clean CSV,
reading with fastparquet to bypass the pyarrow 'Repetition level histogram
size mismatch' error. Reports read success/failure per file.

Usage:
  python3 merge_grid_a.py results/task5_execution results/grid_a_merged/results_raw.csv
"""
import sys, os, glob, pandas as pd
SRC = sys.argv[1] if len(sys.argv)>1 else "results/task5_execution"
OUT = sys.argv[2] if len(sys.argv)>2 else "results/grid_a_merged/results_raw.csv"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC, "**", "results_raw.parquet"), recursive=True))
print(f"found {len(files)} parquet files under {SRC}")
ok, bad, frames = 0, [], []
for f in files:
    try:
        df = pd.read_parquet(f, engine="fastparquet")
        frames.append(df); ok += 1
    except Exception as e:
        bad.append((f, str(e)[:60]))
print(f"read OK: {ok} | failed: {len(bad)}")
for f,e in bad[:10]:
    print("  FAIL", os.path.relpath(f, SRC), "->", e)
if not frames:
    print("nothing read; aborting"); sys.exit(1)
big = pd.concat(frames, ignore_index=True)
big.to_csv(OUT, index=False)
print(f"\nwrote {OUT}  ({len(big)} rows)")
# structure so we can see the extended range and replicate counts
if "axis" in big and "level" in big:
    print("\naxis x level (all rows):")
    print(big.groupby(["axis","level"]).size().to_string())
if "grid_id" in big:
    print("\ngrid_id:", dict(big["grid_id"].value_counts()))
if "status" in big:
    print("status:", dict(big["status"].value_counts()))
