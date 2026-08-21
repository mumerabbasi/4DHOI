#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODULE05_OUTPUT="${PROJECT_DIR}/05_Optimize_Static_Scene/output"
PREP_SCRIPT="${SCRIPT_DIR}/00_prepare_genzi.py"
OUTPUT_DIR="${SCRIPT_DIR}/output"
PYTHON_BIN="${PYTHON_BIN:-python}"

mapfile -t interactions < <(
  find "${MODULE05_OUTPUT}" \
    -mindepth 2 -maxdepth 2 \
    -type f -name alignment_summary.json \
    -printf '%h\n' \
  | sed 's#.*/##' \
  | sort -V
)

failed=()
selected=0

for interaction_name in "${interactions[@]}"; do
  interaction_number="${interaction_name#interaction_}"
  if ((10#${interaction_number} < 9)); then
    continue
  fi
  selected=$((selected + 1))

  summary_path="${OUTPUT_DIR}/${interaction_name}/preparation_summary.json"
  config_path="${OUTPUT_DIR}/_scene_configs/${interaction_name}_v1.yml"
  if [[ -f "${summary_path}" && -f "${config_path}" ]]; then
    echo "[*] Skipping completed ${interaction_name}"
    continue
  fi

  command=(
    "${PYTHON_BIN}"
    "${PREP_SCRIPT}"
    --interaction-name "${interaction_name}"
  )
  if [[ -e "${OUTPUT_DIR}/${interaction_name}" || -e "${config_path}" ]]; then
    command+=(--overwrite)
  fi

  echo
  echo "[*] Preparing ${interaction_name}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '    '
    printf '%q ' "${command[@]}"
    printf '\n'
  elif ! "${command[@]}"; then
    echo "[!] Preparation failed for ${interaction_name}; continuing." >&2
    failed+=("${interaction_name}")
  fi
done

if ((selected == 0)); then
  echo "[!] No module-05-completed interactions numbered 09 or above were found." >&2
  exit 1
fi

if ((${#failed[@]} > 0)); then
  echo
  echo "[!] Failed interactions: ${failed[*]}" >&2
  exit 1
fi

echo
echo "[*] Finished preparing all available interactions numbered 09 and above."
