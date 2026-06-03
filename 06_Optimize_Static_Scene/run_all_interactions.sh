#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-4dhsi}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/run_all_interactions}"
PYTHON_SCRIPT="${SCRIPT_DIR}/01_optimize_static_scene.py"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

mapfile -t INTERACTIONS < <(
  find "${PROJECT_DIR}/01_Generate_SIG/output" \
    -maxdepth 1 \
    -type d \
    -name 'interaction_*' \
    -printf '%f\n' \
  | sort
)

if [[ "${#INTERACTIONS[@]}" -eq 0 ]]; then
  echo "No interaction_* directories found under ${PROJECT_DIR}/01_Generate_SIG/output" >&2
  exit 1
fi

echo "Workspace: ${WORKSPACE_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Device argument: cuda:0"
echo "Logs: ${LOG_DIR}"
echo "Interactions: ${INTERACTIONS[*]}"
echo

cd "${WORKSPACE_DIR}"

for interaction_name in "${INTERACTIONS[@]}"; do
  started_at="$(date -Is)"
  log_path="${LOG_DIR}/${interaction_name}.log"
  echo "[$started_at] Starting ${interaction_name}"
  python "${PYTHON_SCRIPT}" \
    --interaction_name "${interaction_name}" \
    --device cuda:0 \
    2>&1 | tee "${log_path}"
  finished_at="$(date -Is)"
  echo "[$finished_at] Finished ${interaction_name}"
  echo
done

echo "All interactions completed at $(date -Is)."
