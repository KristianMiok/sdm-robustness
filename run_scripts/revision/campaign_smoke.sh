#!/bin/bash
#SBATCH --job-name=camp_smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=06:00:00
#SBATCH --output=logs/camp_smoke_%j.out
#SBATCH --error=logs/camp_smoke_%j.err
set -e
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
python scripts/run_task5_entity.py \
  --entity "Pontastacus leptodactylus (pooled)" \
  --grid B \
  --algorithms random_forest xgboost \
  --tracks combined \
  --snap-levels 0 1 2 5 10 20 \
  --lowacc-levels 0 3 10 20 \
  --n-replicates-default 5 \
  --no-surfaces \
  --output-dir results/campaign_smoke/Pontastacus_leptodactylus_pooled
