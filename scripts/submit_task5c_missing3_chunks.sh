#!/bin/bash
#SBATCH --job-name=task5c_mchunk
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-8
#SBATCH --output=logs/slurm/task5c_mchunk_%A_%a.out
#SBATCH --error=logs/slurm/task5c_mchunk_%A_%a.err

set -euo pipefail

cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12
source .venv/bin/activate

combo=$((SLURM_ARRAY_TASK_ID / 3))
chunk=$((SLURM_ARRAY_TASK_ID % 3))

case "$combo" in
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

seed=$((20260426 + 1000 + chunk))
outdir="results/task5c_benchmark_stability_chunks/${safe_entity}_${algorithm}_${track}_chunk${chunk}"

rm -rf "$outdir"
mkdir -p "$outdir"

echo "Running Task 5c chunk"
echo "entity=$entity"
echo "algorithm=$algorithm"
echo "track=$track"
echo "chunk=$chunk"
echo "seed=$seed"
echo "outdir=$outdir"

python scripts/run_task5_entity.py \
  --entity "$entity" \
  --data-path data/combined_data_true_master.csv \
  --output-dir "$outdir" \
  --algorithms "$algorithm" \
  --tracks "$track" \
  --benchmark-only \
  --n-replicates-default 10 \
  --master-seed "$seed"

echo "Finished: $outdir"
