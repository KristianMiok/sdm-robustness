#!/bin/bash
#SBATCH --job-name=gridB_Astacus_astacus
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm/gridB_Astacus_astacus_%j.out
#SBATCH --error=logs/slurm/gridB_Astacus_astacus_%j.err

set -e
cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

echo "=== Grid B full: Astacus astacus ==="
date

python scripts/run_task5_entity.py \
  --entity "Astacus astacus" \
  --grid B \
  --algorithms random_forest xgboost maxent \
  --tracks local_only upstream_only combined \
  --n-replicates-default 30 \
  --output-dir results/grid_b_full/Astacus_astacus \
  2>&1

echo "=== Done ==="
date
