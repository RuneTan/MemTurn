#!/usr/bin/env bash
# Copy this file, fill resource fields with the cluster owner, then submit.
#SBATCH --job-name=memturn
#SBATCH --partition=TO_SET_PARTITION
#SBATCH --gres=gpu:TO_SET_GPU_COUNT
#SBATCH --cpus-per-task=TO_SET_CPU_COUNT
#SBATCH --mem=TO_SET_RAM
#SBATCH --time=TO_SET_WALLTIME
set -euo pipefail
: "${MEMTURN_PYTHON:?Set the Python executable in the project environment}"
: "${MEMTURN_PROFILE:?Set the verified profile path}"
: "${MEMTURN_OUTPUT:?Set a new run directory or an explicit resume directory}"
# Start inference servers separately using their verified, pinned launch configs.
# Credentials are inherited from the job environment, never written here.
"$MEMTURN_PYTHON" -X utf8 scripts/cluster_inventory.py
"$MEMTURN_PYTHON" -X utf8 run_memturn.py check-profile --profile "$MEMTURN_PROFILE"
if [[ "${MEMTURN_RESUME:-0}" == "1" ]]; then
  "$MEMTURN_PYTHON" -X utf8 run_memturn.py run --profile "$MEMTURN_PROFILE" --output "$MEMTURN_OUTPUT" --resume
else
  "$MEMTURN_PYTHON" -X utf8 run_memturn.py run --profile "$MEMTURN_PROFILE" --output "$MEMTURN_OUTPUT"
fi
