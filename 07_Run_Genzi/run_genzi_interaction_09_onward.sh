#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/01_run_genzi.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
LOG_DIR="${SCRIPT_DIR}/logs/run_genzi_parallel"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Each worker is sequential on its assigned physical GPU; both workers run in
# parallel. CUDA_VISIBLE_DEVICES remaps that physical GPU to logical cuda:0.
GPU0_INTERACTIONS=(interaction_26 interaction_27)
GPU1_INTERACTIONS=(interaction_28 interaction_30)

mkdir -p "${LOG_DIR}"

run_worker() {
  local physical_gpu="$1"
  shift
  local interaction_name
  local config_path
  local summary_path
  local interaction_log
  local worker_failed=()

  echo "[*] GPU ${physical_gpu} queue: $*"
  for interaction_name in "$@"; do
    config_path="${OUTPUT_DIR}/_scene_configs/${interaction_name}_v1.yml"
    summary_path="${OUTPUT_DIR}/${interaction_name}/genzi_run_summary.json"
    interaction_log="${LOG_DIR}/${interaction_name}.log"

    if [[ ! -f "${config_path}" ]]; then
      echo "[!] GPU ${physical_gpu}: missing prepared config for ${interaction_name}; skipping." >&2
      worker_failed+=("${interaction_name}:unprepared")
      continue
    fi
    if [[ -f "${summary_path}" ]]; then
      echo "[*] GPU ${physical_gpu}: skipping completed ${interaction_name}"
      continue
    fi

    echo "[*] GPU ${physical_gpu}: running ${interaction_name}"
    command=(
      "${PYTHON_BIN}"
      "${RUN_SCRIPT}"
      --interaction-name "${interaction_name}"
      --device cuda:0
    )

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '    CUDA_VISIBLE_DEVICES=%q ' "${physical_gpu}"
      printf '%q ' "${command[@]}"
      printf '\n'
    elif ! CUDA_VISIBLE_DEVICES="${physical_gpu}" "${command[@]}" \
      > >(tee -a "${interaction_log}") \
      2> >(tee -a "${interaction_log}" >&2); then
      echo "[!] GPU ${physical_gpu}: GenZI failed for ${interaction_name}; continuing." >&2
      worker_failed+=("${interaction_name}:failed")
    else
      echo "[*] GPU ${physical_gpu}: finished ${interaction_name}"
    fi
  done

  if ((${#worker_failed[@]} > 0)); then
    echo "[!] GPU ${physical_gpu} failures: ${worker_failed[*]}" >&2
    return 1
  fi
  echo "[*] GPU ${physical_gpu} queue finished"
}

run_worker 0 "${GPU0_INTERACTIONS[@]}" &
gpu0_pid=$!
run_worker 1 "${GPU1_INTERACTIONS[@]}" &
gpu1_pid=$!

gpu0_status=0
gpu1_status=0
wait "${gpu0_pid}" || gpu0_status=$?
wait "${gpu1_pid}" || gpu1_status=$?

if ((gpu0_status != 0 || gpu1_status != 0)); then
  echo "[!] Parallel GenZI run finished with failures " \
    "(GPU 0 status=${gpu0_status}, GPU 1 status=${gpu1_status})." >&2
  exit 1
fi

echo "[*] Parallel GenZI run finished successfully."
