#!/bin/bash
#SBATCH --job-name=rfxgb_Pontastacus_leptodactylus_pooled
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=12:00:00
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
  --entity "Pontastacus leptodactylus (pooled)" \
  --data-path /ceph/hpc/home/miokk/sdm-robustness/data/combined_data_true_master.csv \
  --output-dir results/task5_execution/Pontastacus_leptodactylus_pooled_rfxgb \
  --skip-benchmark \
  --algorithms random_forest xgboost \
  --tracks local_only upstream_only combined \
  --levels 0 5 10 20 35 50 \
  --n-replicates-default 30 \
  --n-replicates-low-levels 50
