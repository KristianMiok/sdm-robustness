#!/bin/bash
#SBATCH --job-name=t3
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=24:00:00
#SBATCH --array=0-25%10
#SBATCH --output=logs/t3_%A_%a.out
#SBATCH --error=logs/t3_%A_%a.err
set -uo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
L=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" run_scripts/revision/t3_tasks.tsv)
ENT=$(cut -f1 <<< "$L"); ALG=$(cut -f2 <<< "$L")
echo "task=$SLURM_ARRAY_TASK_ID $ENT [$ALG]"
python scripts/revision/t3_displacement.py --entity "$ENT" --alg "$ALG" --reps 15
