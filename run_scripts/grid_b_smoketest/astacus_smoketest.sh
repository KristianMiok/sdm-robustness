#!/bin/bash
#SBATCH --job-name=gridB_astacus_smoke
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/gridB_astacus_smoke_%j.out
#SBATCH --error=logs/slurm/gridB_astacus_smoke_%j.err

set -e
cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

echo "=== Grid B smoketest: Astacus, 2 replicates, all 3 algorithms × 3 tracks ==="
date

python scripts/run_task5_entity.py \
  --entity "Astacus astacus" \
  --grid B \
  --algorithms random_forest xgboost maxent \
  --tracks local_only upstream_only combined \
  --n-replicates-default 2 \
  --output-dir results/grid_b_smoketest/astacus \
  2>&1

echo "=== Done ==="
date
