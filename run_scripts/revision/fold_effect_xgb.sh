#!/bin/bash
#SBATCH --job-name=fold_effx
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/fold_effx_%A_%a.out
#SBATCH --error=logs/fold_effx_%A_%a.err
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
ENTS=("Austropotamobius torrentium (pooled)" "Astacus astacus" "Procambarus clarkii (alien)")
python scripts/revision/t6_fold_effect.py --entity "${ENTS[$SLURM_ARRAY_TASK_ID]}" --reps 12 --alg xgboost
