#!/bin/bash
#SBATCH --job-name=var_pilot3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=12:00:00
#SBATCH --output=logs/var_pilot3_%j.out
#SBATCH --error=logs/var_pilot3_%j.err
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
python scripts/revision/t5_variance_pilot.py --out results/revision/t5_var_clarkii_xgb.csv --reps 15 --algorithms xgboost \
  --entity "Procambarus clarkii (alien)" --axis lowacc --level 20
