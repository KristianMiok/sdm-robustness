#!/bin/bash
#SBATCH --job-name=var_pilot
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=04:00:00
#SBATCH --output=logs/var_pilot_%j.out
#SBATCH --error=logs/var_pilot_%j.err
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
python scripts/revision/t5_variance_pilot.py --out results/revision/t5_var_torrentium.csv --reps 40 \
  --entity "Austropotamobius torrentium (pooled)" --axis snapping --level 5
