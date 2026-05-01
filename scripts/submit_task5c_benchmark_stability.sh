#!/bin/bash
#SBATCH --job-name=task5c_bench
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm/task5c_bench_%j.out
#SBATCH --error=logs/slurm/task5c_bench_%j.err

set -euo pipefail

cd /ceph/hpc/home/miokk/sdm-robustness

source .venv/bin/activate

python scripts/run_task5_entity.py \
  --data-path data/combined_data_true_master.csv \
  --output-dir results/task5c_benchmark_stability/full_panel \
  --algorithms random_forest xgboost maxent \
  --tracks local_only upstream_only combined \
  --benchmark-only \
  --n-replicates-default 30
