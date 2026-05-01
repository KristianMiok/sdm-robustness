#!/bin/bash
#SBATCH --job-name=task5c_miss
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm/task5c_missing_%A_%a.out
#SBATCH --error=logs/slurm/task5c_missing_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12
source .venv/bin/activate

case "$SLURM_ARRAY_TASK_ID" in
  0)
    entity="Pacifastacus leniusculus (alien)"
    safe_entity="Pacifastacus_leniusculus_alien"
    algorithm="maxent"
    track="combined"
    ;;
  1)
    entity="Faxonius limosus (alien)"
    safe_entity="Faxonius_limosus_alien"
    algorithm="maxent"
    track="local_only"
    ;;
  2)
    entity="Faxonius limosus (alien)"
    safe_entity="Faxonius_limosus_alien"
    algorithm="maxent"
    track="combined"
    ;;
esac

outdir="results/task5c_benchmark_stability_array/${safe_entity}_${algorithm}_${track}"

echo "Running missing Task 5c job"
echo "entity=$entity"
echo "algorithm=$algorithm"
echo "track=$track"
echo "outdir=$outdir"

rm -rf "$outdir"
mkdir -p "$outdir"

python scripts/run_task5_entity.py \
  --entity "$entity" \
  --data-path data/combined_data_true_master.csv \
  --output-dir "$outdir" \
  --algorithms "$algorithm" \
  --tracks "$track" \
  --benchmark-only \
  --n-replicates-default 30

echo "Finished: $outdir"
