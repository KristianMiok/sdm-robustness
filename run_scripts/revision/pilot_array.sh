#!/bin/bash
#SBATCH --job-name=var_arr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3000M
#SBATCH --time=08:00:00
#SBATCH --array=0-38%12
#SBATCH --output=logs/var_arr_%A_%a.out
#SBATCH --error=logs/var_arr_%A_%a.err
set -uo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate

TASKS=run_scripts/revision/pilot_tasks.tsv
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TASKS")
ENTITY=$(cut -f1 <<< "$LINE")
ALG=$(cut -f2 <<< "$LINE")
SLUG=$(tr -cd '[:alnum:]_' <<< "${ENTITY// /_}")

echo "task=$SLURM_ARRAY_TASK_ID entity=$ENTITY alg=$ALG"
python scripts/revision/t5_variance_pilot.py \
  --entity "$ENTITY" --algorithms "$ALG" \
  --axis "${AXIS:-lowacc}" --level "${LEVEL:-20}" --reps "${REPS:-15}" \
  --out "results/revision/var_${SLUG}_${ALG}_${AXIS:-lowacc}${LEVEL:-20}.csv"
