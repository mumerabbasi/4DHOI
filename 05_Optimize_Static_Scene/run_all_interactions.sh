#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-4dhsi}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/run_all_interactions}"
PYTHON_SCRIPT="${SCRIPT_DIR}/01_optimize_static_scene.py"
EXCLUDED_INTERACTIONS_RAW="${EXCLUDED_INTERACTIONS:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--exclude interaction_03,interaction_08] [--exclude interaction_15 interaction_16]

Environment:
  CONDA_ENV              Conda environment to activate. Default: 4dhsi
  LOG_DIR                Directory for per-interaction logs.
  CUDA_VISIBLE_DEVICES   Visible GPU list. Default: all GPUs reported by nvidia-smi, or 0.
  EXCLUDED_INTERACTIONS  Space- or comma-separated interactions to skip.

All optimizer invocations use --device cuda:0. CUDA_VISIBLE_DEVICES controls
which physical GPUs are used. Each interaction runs with one visible GPU, so
cuda:0 maps to that job's assigned physical GPU.
EOF
}

normalize_interaction_name() {
  local raw="$1"
  raw="${raw#interaction_}"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    printf "interaction_%02d" "$((10#${raw}))"
  else
    printf "%s" "$1"
  fi
}

append_exclusions() {
  local raw="$1"
  raw="${raw//,/ }"
  EXCLUDED_INTERACTIONS_RAW+=" ${raw}"
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --exclude|--exclude-interactions)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "Missing value after --exclude" >&2
        exit 1
      fi
      while [[ "$#" -gt 0 && "$1" != --* ]]; do
        append_exclusions "$1"
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda activation script not found: ${CONDA_SH}" >&2
  exit 1
fi

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

mkdir -p "${LOG_DIR}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  export CUDA_VISIBLE_DEVICES
fi
IFS=',' read -r -a RAW_GPU_SLOTS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_SLOTS=()
for raw_gpu_slot in "${RAW_GPU_SLOTS[@]}"; do
  gpu_slot="${raw_gpu_slot//[[:space:]]/}"
  [[ -z "${gpu_slot}" ]] && continue
  GPU_SLOTS+=("${gpu_slot}")
done
if [[ "${#GPU_SLOTS[@]}" -eq 0 ]]; then
  GPU_SLOTS=("0")
fi

EXCLUDED_INTERACTIONS=()
if [[ -n "${EXCLUDED_INTERACTIONS_RAW// /}" ]]; then
  read -r -a RAW_EXCLUDED_INTERACTIONS <<< "${EXCLUDED_INTERACTIONS_RAW//,/ }"
  for raw_interaction_name in "${RAW_EXCLUDED_INTERACTIONS[@]}"; do
    [[ -z "${raw_interaction_name}" ]] && continue
    EXCLUDED_INTERACTIONS+=("$(normalize_interaction_name "${raw_interaction_name}")")
  done
fi

is_excluded_interaction() {
  local interaction_name="$1"
  local excluded_name
  for excluded_name in "${EXCLUDED_INTERACTIONS[@]}"; do
    if [[ "${interaction_name}" == "${excluded_name}" ]]; then
      return 0
    fi
  done
  return 1
}

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

FILTERED_INTERACTIONS=()
for interaction_name in "${INTERACTIONS[@]}"; do
  if is_excluded_interaction "${interaction_name}"; then
    continue
  fi
  FILTERED_INTERACTIONS+=("${interaction_name}")
done

if [[ "${#FILTERED_INTERACTIONS[@]}" -eq 0 ]]; then
  echo "No interactions left after exclusions." >&2
  exit 1
fi

echo "Workspace: ${WORKSPACE_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "GPU slots: ${GPU_SLOTS[*]}"
echo "Device argument: cuda:0"
echo "Logs: ${LOG_DIR}"
if [[ "${#EXCLUDED_INTERACTIONS[@]}" -gt 0 ]]; then
  echo "Excluded interactions: ${EXCLUDED_INTERACTIONS[*]}"
fi
echo "Interactions: ${FILTERED_INTERACTIONS[*]}"
echo

cd "${WORKSPACE_DIR}"

ACTIVE_PIDS=()
ACTIVE_INTERACTIONS=()
ACTIVE_GPUS=()
FAILED_INTERACTIONS=()
SKIPPED_INTERACTIONS=()
NEXT_INTERACTION_INDEX=0
POLL_SECONDS="${POLL_SECONDS:-2}"

is_gpu_in_use() {
  local gpu_id="$1"
  local active_gpu
  for active_gpu in "${ACTIVE_GPUS[@]}"; do
    if [[ "${gpu_id}" == "${active_gpu}" ]]; then
      return 0
    fi
  done
  return 1
}

next_available_gpu() {
  local gpu_id
  for gpu_id in "${GPU_SLOTS[@]}"; do
    if ! is_gpu_in_use "${gpu_id}"; then
      printf "%s" "${gpu_id}"
      return 0
    fi
  done
  return 1
}

is_pid_running() {
  local pid="$1"
  local running_pid
  for running_pid in "${RUNNING_PIDS[@]}"; do
    if [[ "${pid}" == "${running_pid}" ]]; then
      return 0
    fi
  done
  return 1
}

launch_interaction() {
  local interaction_name="$1"
  local gpu_id="$2"
  local log_path="${LOG_DIR}/${interaction_name}.log"
  (
    set +e
    {
      started_at="$(date -Is)"
      echo "[$started_at] Starting ${interaction_name} on physical GPU ${gpu_id} as cuda:0"
      CUDA_VISIBLE_DEVICES="${gpu_id}" python "${PYTHON_SCRIPT}" \
        --interaction_name "${interaction_name}" \
        --device cuda:0
      run_status="$?"
      finished_at="$(date -Is)"
      if [[ "${run_status}" -eq 0 ]]; then
        echo "[$finished_at] Finished ${interaction_name} on physical GPU ${gpu_id}"
      else
        echo "[$finished_at] Failed ${interaction_name} on physical GPU ${gpu_id} with exit code ${run_status}"
      fi
      exit "${run_status}"
    } 2>&1 | tee "${log_path}"
    exit "${PIPESTATUS[0]}"
  ) &
  local pid="$!"
  ACTIVE_PIDS+=("${pid}")
  ACTIVE_INTERACTIONS+=("${interaction_name}")
  ACTIVE_GPUS+=("${gpu_id}")
  echo "Launched ${interaction_name} on physical GPU ${gpu_id} as cuda:0 (pid ${pid}, log ${log_path})"
}

missing_required_inputs() {
  local interaction_name="$1"
  local missing=()

  local generated_root="${PROJECT_DIR}/02_Generate_Human_Frame/output/${interaction_name}"
  local sig_input_root="${PROJECT_DIR}/01_Generate_SIG/input_prompts/${interaction_name}"
  local sig_output_root="${PROJECT_DIR}/01_Generate_SIG/output/${interaction_name}"
  local human_pose_root="${PROJECT_DIR}/04_Estimate_Human_Pose/output/${interaction_name}"
  local contact_root="${PROJECT_DIR}/03_Estimate_Contact/output/${interaction_name}"

  [[ -f "${generated_root}/inpainted_frame_resized.png" ]] || missing+=("${generated_root}/inpainted_frame_resized.png")
  [[ -f "${sig_input_root}/input_scene.json" ]] || missing+=("${sig_input_root}/input_scene.json")
  [[ -f "${sig_output_root}/sig.json" ]] || missing+=("${sig_output_root}/sig.json")
  [[ -f "${human_pose_root}/hmr4d_results.pt" ]] || missing+=("${human_pose_root}/hmr4d_results.pt")
  [[ -d "${contact_root}/contact_masks" ]] || missing+=("${contact_root}/contact_masks/")
  [[ -f "${contact_root}/prompt/target_scene_crop.png" ]] || missing+=("${contact_root}/prompt/target_scene_crop.png")
  [[ -f "${contact_root}/contact_spec.json" ]] || missing+=("${contact_root}/contact_spec.json")

  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 1
  fi

  printf "%s\n" "${missing[@]}"
  return 0
}

skip_interaction() {
  local interaction_name="$1"
  local reason="$2"
  local log_path="${LOG_DIR}/${interaction_name}.log"

  SKIPPED_INTERACTIONS+=("${interaction_name}")
  {
    echo "[$(date -Is)] Skipping ${interaction_name}"
    echo "${reason}"
  } | tee "${log_path}"
}

reap_completed_jobs() {
  mapfile -t RUNNING_PIDS < <(jobs -r -p)
  local remaining_pids=()
  local remaining_interactions=()
  local remaining_gpus=()
  local index
  for index in "${!ACTIVE_PIDS[@]}"; do
    local pid="${ACTIVE_PIDS[${index}]}"
    local interaction_name="${ACTIVE_INTERACTIONS[${index}]}"
    local gpu_id="${ACTIVE_GPUS[${index}]}"
    if is_pid_running "${pid}"; then
      remaining_pids+=("${pid}")
      remaining_interactions+=("${interaction_name}")
      remaining_gpus+=("${gpu_id}")
      continue
    fi
    if wait "${pid}"; then
      echo "Completed ${interaction_name} on physical GPU ${gpu_id}"
    else
      echo "Failed ${interaction_name} on physical GPU ${gpu_id}; see ${LOG_DIR}/${interaction_name}.log" >&2
      FAILED_INTERACTIONS+=("${interaction_name}")
    fi
  done
  ACTIVE_PIDS=("${remaining_pids[@]}")
  ACTIVE_INTERACTIONS=("${remaining_interactions[@]}")
  ACTIVE_GPUS=("${remaining_gpus[@]}")
}

terminate_active_jobs() {
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    echo "Terminating active optimization jobs..." >&2
    kill "${ACTIVE_PIDS[@]}" 2>/dev/null || true
  fi
}
trap terminate_active_jobs INT TERM

while [[ "${NEXT_INTERACTION_INDEX}" -lt "${#FILTERED_INTERACTIONS[@]}" || "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
  while [[ "${NEXT_INTERACTION_INDEX}" -lt "${#FILTERED_INTERACTIONS[@]}" ]]; do
    interaction_name="${FILTERED_INTERACTIONS[${NEXT_INTERACTION_INDEX}]}"
    if missing="$(missing_required_inputs "${interaction_name}")"; then
      skip_interaction "${interaction_name}" "Missing required inputs:
${missing}"
      NEXT_INTERACTION_INDEX="$((NEXT_INTERACTION_INDEX + 1))"
      continue
    fi
    if ! gpu_id="$(next_available_gpu)"; then
      break
    fi
    launch_interaction "${interaction_name}" "${gpu_id}"
    NEXT_INTERACTION_INDEX="$((NEXT_INTERACTION_INDEX + 1))"
  done
  if [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; then
    sleep "${POLL_SECONDS}"
    reap_completed_jobs
  fi
done

trap - INT TERM

if [[ "${#FAILED_INTERACTIONS[@]}" -gt 0 ]]; then
  echo "Failed interactions: ${FAILED_INTERACTIONS[*]}" >&2
  exit 1
fi

if [[ "${#SKIPPED_INTERACTIONS[@]}" -gt 0 ]]; then
  echo "Skipped interactions with missing inputs: ${SKIPPED_INTERACTIONS[*]}"
fi
echo "All interactions completed at $(date -Is)."
