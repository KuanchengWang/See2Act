#!/usr/bin/env bash
# End-to-end reproduction of the Ravens experiments: demonstrations -> training -> checkpoint sweep.
# Usage: bash scripts/reproduce.sh [task ...]      (default: all four tasks; one GPU is enough, trainings run one at a time)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(place-red-in-green bin-picking put-within-shelf bin-search)
export PYTHONUNBUFFERED=1
for t in "${TASKS[@]}"; do
  [ -f "data/$t.hdf5" ] || $PY scripts/collect_demos.py --task "$t" --n_train 100 --n_valid 20 --out "data/$t.hdf5"
done
for t in "${TASKS[@]}"; do
  $PY scripts/train.py --config configs/see2act.json --dataset "data/$t.hdf5" --name "see2act_$t" --output_dir runs
  RUN=$(ls -d runs/see2act_$t/* | tail -1)
  $PY scripts/sweep_checkpoints.py --run_dir "$RUN" --protocol single --n_episodes 50 --out_dir results
done
