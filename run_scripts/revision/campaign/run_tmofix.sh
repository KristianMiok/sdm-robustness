#!/bin/bash
#SBATCH --job-name=camptmo
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=48:00:00
#SBATCH --array=43,44
#SBATCH --output=logs/camptmo_%A_%a.out
#SBATCH --error=logs/camptmo_%A_%a.err
set -uo pipefail
module load Python/3.12.3-GCCcore-13.3.0
cd $HOME/sdm-robustness && source .venv/bin/activate
L=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" run_scripts/revision/campaign/tasks.tsv)
ENT=$(cut -f1 <<< "$L"); SLUG=$(cut -f2 <<< "$L"); GRP=$(cut -f3 <<< "$L")
ALGS=$(cut -f4 <<< "$L"); TRACKS=$(cut -f5 <<< "$L")
SNAP=$(cut -f6 <<< "$L"); LOW=$(cut -f7 <<< "$L"); REPS=$(cut -f8 <<< "$L")
echo "task=$SLURM_ARRAY_TASK_ID $ENT [$GRP]"
ARGS=(--entity "$ENT" --grid B --algorithms $ALGS
      --tracks $TRACKS
      --n-replicates-default $REPS --no-surfaces
      --output-dir "results/campaign/${SLUG}_${GRP}")
[ -n "$SNAP" ] && ARGS+=(--snap-levels $SNAP)
[ -n "$LOW" ] && ARGS+=(--lowacc-levels $LOW)
[ -z "$SNAP" ] && ARGS+=(--axes lowacc)
[ -z "$LOW" ] && ARGS+=(--axes snapping)
python scripts/run_task5_entity.py "${ARGS[@]}"
