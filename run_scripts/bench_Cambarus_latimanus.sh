#!/bin/bash
#SBATCH --job-name=bench_Cambarus_latimanus
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

cd /ceph/hpc/home/miokk/sdm-robustness
module load Python/3.12.3-GCCcore-13.3.0
source .venv/bin/activate

python scripts/run_task5_entity.py \
  --entity "Cambarus latimanus" \
  --data-path /ceph/hpc/home/miokk/sdm-robustness/data/combined_data_true_master.csv \
  --output-dir results/task5_execution/Cambarus_latimanus_benchmark \
  --benchmark-only \
  --algorithms random_forest xgboost maxent \
  --tracks local_only upstream_only combined
