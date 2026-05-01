#!/bin/bash
#SBATCH --job-name=task5c_arr
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-116%20
#SBATCH --output=logs/slurm/task5c_arr_%A_%a.out
#SBATCH --error=logs/slurm/task5c_arr_%A_%a.err

set -euo pipefail

cd /ceph/hpc/home/miokk/sdm-robustness
source .venv/bin/activate

mkdir -p results/task5c_benchmark_stability_array logs/slurm

python - <<'PY' > /tmp/task5c_jobs.tsv
import pandas as pd
from pathlib import Path

panel = pd.read_csv("config/final_panel.csv")

entities = panel["entity"].dropna().tolist()
algorithms = ["random_forest", "xgboost", "maxent"]
tracks = ["local_only", "upstream_only", "combined"]

rows = []
for entity in entities:
    safe = (
        entity.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )
    for algorithm in algorithms:
        for track in tracks:
            rows.append((entity, safe, algorithm, track))

for i, row in enumerate(rows):
    print(i, *row, sep="\t")
PY

line=$(awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print}' /tmp/task5c_jobs.tsv)

if [ -z "$line" ]; then
  echo "No task for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

entity=$(echo "$line" | cut -f2)
safe_entity=$(echo "$line" | cut -f3)
algorithm=$(echo "$line" | cut -f4)
track=$(echo "$line" | cut -f5)

outdir="results/task5c_benchmark_stability_array/${safe_entity}_${algorithm}_${track}"

echo "Running Task 5c benchmark stability"
echo "entity=$entity"
echo "algorithm=$algorithm"
echo "track=$track"
echo "outdir=$outdir"

python scripts/run_task5_entity.py \
  --entity "$entity" \
  --data-path data/combined_data_true_master.csv \
  --output-dir "$outdir" \
  --algorithms "$algorithm" \
  --tracks "$track" \
  --benchmark-only \
  --n-replicates-default 30

echo "Finished: $outdir"
