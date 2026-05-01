#!/bin/bash
#SBATCH --job-name=task5c_fc
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm/task5c_fc_%A_%a.out
#SBATCH --error=logs/slurm/task5c_fc_%A_%a.err

set -euo pipefail
cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12
source .venv/bin/activate

line=$(awk -v id="$SLURM_ARRAY_TASK_ID" '$1 == id {print}' results/task5c_failed_chunks.tsv)

if [ -z "$line" ]; then
  echo "No task for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

entity=$(echo "$line" | cut -f2)
safe_entity=$(echo "$line" | cut -f3)
algorithm=$(echo "$line" | cut -f4)
track=$(echo "$line" | cut -f5)
chunk=$(echo "$line" | cut -f6)
seed=$(echo "$line" | cut -f7)

outdir="results/task5c_benchmark_stability_chunks/${safe_entity}_${algorithm}_${track}_chunk${chunk}"

echo "entity=$entity"
echo "track=$track"
echo "chunk=$chunk"
echo "seed=$seed"

python scripts/run_task5_entity.py \
  --entity "$entity" \
  --data-path data/combined_data_true_master.csv \
  --output-dir "$outdir" \
  --algorithms "$algorithm" \
  --tracks "$track" \
  --benchmark-only \
  --n-replicates-default 10 \
  --master-seed "$seed"
