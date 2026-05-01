"""Merge Task 5c chunks: combine 3 chunks of 10 reps each into one 30-rep aggregate.

Uses pooled-variance formula to produce exact mean and SD estimates equivalent to
what would have been computed had all 30 replicates been run as a single batch.

Output goes into results/task5c_benchmark_stability_array/ alongside the 114
already-completed entries, producing the full 117-entry Task 5c dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path

CHUNK_DIR = Path("results/task5c_benchmark_stability_chunks")
ARRAY_DIR = Path("results/task5c_benchmark_stability_array")

# Find chunks and group by (safe_entity, algorithm, track)
chunks = sorted(CHUNK_DIR.glob("*/benchmark_stability.parquet"))
print(f"Found {len(chunks)} chunk files")

# Group: dirname pattern is {safe_entity}_{algorithm}_{track}_chunk{N}
groups = {}
for c in chunks:
    name = c.parent.name
    # Strip trailing _chunkN
    if "_chunk" not in name:
        print(f"WARNING: {name} doesn't match _chunk pattern, skipping")
        continue
    group_key = name.rsplit("_chunk", 1)[0]
    groups.setdefault(group_key, []).append(c)

print(f"Found {len(groups)} unique (entity, algorithm, track) groups\n")

for group_key, files in sorted(groups.items()):
    print(f"=== {group_key} ({len(files)} chunks) ===")
    if len(files) != 3:
        print(f"  WARNING: expected 3 chunks, got {len(files)} — skipping")
        continue

    dfs = [pd.read_parquet(f) for f in sorted(files)]

    # All 3 chunks should share entity, species, category, entity_type, algorithm, track
    # and have one row per metric_name. Pool across chunks per metric.
    combined_rows = []
    metrics = sorted(dfs[0]["metric_name"].unique())

    for metric in metrics:
        rows = [d[d["metric_name"] == metric].iloc[0] for d in dfs]
        ns = np.array([r["n_replicates_ok"] for r in rows], dtype=float)
        ms = np.array([r["benchmark_mean"] for r in rows], dtype=float)
        ss = np.array([r["benchmark_sd"] for r in rows], dtype=float)

        N = ns.sum()
        if N <= 1:
            print(f"  {metric}: N={N}, skipping (degenerate)")
            continue

        pooled_mean = (ns * ms).sum() / N

        ss_within = ((ns - 1) * ss**2).sum()
        ss_between = (ns * (ms - pooled_mean)**2).sum()
        pooled_sd = np.sqrt((ss_within + ss_between) / (N - 1))

        new_row = rows[0].to_dict()
        new_row["benchmark_mean"] = pooled_mean
        new_row["benchmark_sd"] = pooled_sd
        new_row["n_replicates_ok"] = int(N)
        combined_rows.append(new_row)
        print(f"  {metric}: pooled_mean={pooled_mean:.4f}, pooled_sd={pooled_sd:.4f}, n={int(N)}")

    final = pd.DataFrame(combined_rows)

    # Place the merged file in the array dir to fill the gap
    out_dir = ARRAY_DIR / group_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_stability.parquet"
    final.to_parquet(out_file, index=False)
    print(f"  -> wrote {out_file}")
    print()

# Verify final coverage
print("=" * 60)
print("=== Coverage check after merge ===")
all_files = sorted(ARRAY_DIR.glob("*/benchmark_stability.parquet"))
print(f"Total benchmark_stability.parquet files in array dir: {len(all_files)}")
print(f"Expected: 117 (39 entities × 3 algorithms × 1 track? no, 13 × 3 × 3 = 117)")
