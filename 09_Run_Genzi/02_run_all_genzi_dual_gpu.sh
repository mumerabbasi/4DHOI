#!/usr/bin/env bash
set -Eeuo pipefail

MODULE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${MODULE_DIR}/.." && pwd)"
MODULE05_OUTPUT="${PROJECT_DIR}/05_Optimize_Static_Scene/output"
PREP_SCRIPT="${MODULE_DIR}/00_prepare_genzi.py"
RUN_SCRIPT="${MODULE_DIR}/01_run_genzi.py"

GENZI_PYTHON="/root/miniconda3/envs/genzi/bin/python"
OUTPUT_BASE="${MODULE_DIR}/output"
GPU_IDS=(0 1)
SEED=1
STAGES="all"
MAX_STEPS=""
WANDB_MODE="offline"
OVERWRITE_PREP=0
SKIP_COMPLETED=0
SAVE_DEBUG_RENDERS=1
DRY_RUN=0

usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Prepares every Module 05 interaction on two GPUs, then runs native GenZI."
    echo "Preparation requires and reuses Module 08's validated crop mesh and TSDF."
    echo "If default-distance preparation fails, interaction 08 retries at 1.8m and"
    echo "interaction 26 retries at 1.0m. No other interaction receives a fallback."
    echo
    echo "Options:"
    echo "  --gpu-ids ID0,ID1       Physical GPU IDs (default: 0,1)"
    echo "  --output-base PATH      Module 09 output directory"
    echo "  --genzi-python PATH     GenZI environment Python"
    echo "  --seed N                Base random seed (default: 1)"
    echo "  --stages all|stage0     GenZI stages (default: all)"
    echo "  --max-steps N           Limit optimization steps"
    echo "  --wandb-mode MODE       W&B mode (default: offline)"
    echo "  --overwrite-prep        Recreate existing preparation outputs"
    echo "  --skip-completed        Skip interactions with a GenZI run summary"
    echo "  --no-debug-renders      Do not save preparation debug renders"
    echo "  --dry-run               Print the GPU assignment without running"
    echo "  -h, --help              Show this help"
}

while (($# > 0)); do
    case "$1" in
        --gpu-ids)
            IFS=',' read -r -a GPU_IDS <<< "$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --genzi-python)
            GENZI_PYTHON="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --stages)
            STAGES="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --wandb-mode)
            WANDB_MODE="$2"
            shift 2
            ;;
        --overwrite-prep)
            OVERWRITE_PREP=1
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --no-debug-renders)
            SAVE_DEBUG_RENDERS=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ((${#GPU_IDS[@]} != 2)); then
    echo "--gpu-ids must contain exactly two comma-separated GPU IDs." >&2
    exit 2
fi
if [[ "${STAGES}" != "all" && "${STAGES}" != "stage0" ]]; then
    echo "--stages must be 'all' or 'stage0'." >&2
    exit 2
fi
if [[ ! -x "${GENZI_PYTHON}" ]]; then
    echo "GenZI Python is not executable: ${GENZI_PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${PREP_SCRIPT}" || ! -f "${RUN_SCRIPT}" ]]; then
    echo "Module 09 Python entry points are missing." >&2
    exit 2
fi

shopt -s nullglob
alignment_summaries=("${MODULE05_OUTPUT}"/interaction_*/alignment_summary.json)
if ((${#alignment_summaries[@]} == 0)); then
    echo "No completed Module 05 interactions found under ${MODULE05_OUTPUT}." >&2
    exit 1
fi

mapfile -t INTERACTIONS < <(
    for summary in "${alignment_summaries[@]}"; do
        basename -- "$(dirname -- "${summary}")"
    done | sort -V
)

declare -a SHARD0=()
declare -a SHARD1=()
for ((index = 0; index < ${#INTERACTIONS[@]}; index++)); do
    if ((index % 2 == 0)); then
        SHARD0+=("${INTERACTIONS[index]}")
    else
        SHARD1+=("${INTERACTIONS[index]}")
    fi
done

BATCH_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${MODULE_DIR}/logs/dual_gpu_${BATCH_ID}"
mkdir -p -- "${OUTPUT_BASE}" "${LOG_DIR}"

configure_worker_gpu() {
    local physical_gpu="$1"

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="${physical_gpu}"
    export PYOPENGL_PLATFORM=egl
    export TOKENIZERS_PARALLELISM=false
}

prepare_worker() {
    local worker_id="$1"
    local physical_gpu="$2"
    shift 2
    local interactions=("$@")
    local interaction
    local interaction_output
    local scene_config
    local worker_seed="${SEED}"
    local fallback_distance
    local preparation_is_current

    configure_worker_gpu "${physical_gpu}"

    echo "[prep worker ${worker_id}] physical GPU ${physical_gpu}; ${#interactions[@]} interaction(s)"
    for interaction in "${interactions[@]}"; do
        interaction_output="${OUTPUT_BASE}/${interaction}"
        scene_config="${OUTPUT_BASE}/_scene_configs/${interaction}_v1.yml"

        echo "[prep worker ${worker_id}] ${interaction}: starting"
        preparation_is_current=0
        if [[ -f "${interaction_output}/preparation_summary.json" \
              && -f "${scene_config}" ]] \
              && grep -Fq \
                  '"mode": "module08_prox_tsdf_with_module09_color_crop"' \
                  "${interaction_output}/preparation_summary.json"; then
            preparation_is_current=1
        fi
        if [[ "${OVERWRITE_PREP}" == "0" && "${preparation_is_current}" == "1" ]]; then
            echo "[prep worker ${worker_id}] ${interaction}: preparation already complete"
        else
            prep_args=(
                --interaction-name "${interaction}"
                --output-base "${OUTPUT_BASE}"
                --scene-source module08
                --sam3-device cuda:0
                --seed "${worker_seed}"
                --no-root-summary
            )
            if [[ "${OVERWRITE_PREP}" == "1" || -d "${interaction_output}" ]]; then
                if [[ "${OVERWRITE_PREP}" == "0" ]]; then
                    echo "[prep worker ${worker_id}] ${interaction}: replacing stale or incomplete preparation"
                fi
                prep_args+=(--overwrite)
            fi
            if [[ "${SAVE_DEBUG_RENDERS}" == "0" ]]; then
                prep_args+=(--no-save-debug-renders)
            fi

            if "${GENZI_PYTHON}" "${PREP_SCRIPT}" "${prep_args[@]}"; then
                :
            else
                case "${interaction}" in
                    interaction_08)
                        fallback_distance="1.8"
                        ;;
                    interaction_26)
                        fallback_distance="1.0"
                        ;;
                    *)
                        echo "[prep worker ${worker_id}] ${interaction}: preparation failed; no distance fallback configured" >&2
                        return 1
                        ;;
                esac
                echo "[prep worker ${worker_id}] ${interaction}: default-distance preparation failed; retrying at ${fallback_distance}m"
                "${GENZI_PYTHON}" "${PREP_SCRIPT}" \
                    "${prep_args[@]}" \
                    --overwrite \
                    --view-distance-m "${fallback_distance}"
            fi
        fi
        echo "[prep worker ${worker_id}] ${interaction}: complete"
    done
}

run_genzi_worker() {
    local worker_id="$1"
    local physical_gpu="$2"
    shift 2
    local interactions=("$@")
    local interaction
    local run_summary
    local worker_seed="${SEED}"

    configure_worker_gpu "${physical_gpu}"

    echo "[GenZI worker ${worker_id}] physical GPU ${physical_gpu}; ${#interactions[@]} interaction(s)"
    for interaction in "${interactions[@]}"; do
        run_summary="${OUTPUT_BASE}/${interaction}/genzi_run_summary.json"

        if [[ "${SKIP_COMPLETED}" == "1" && -f "${run_summary}" ]]; then
            echo "[GenZI worker ${worker_id}] ${interaction}: already complete; skipping"
            continue
        fi

        echo "[GenZI worker ${worker_id}] ${interaction}: starting"
        run_args=(
            --interaction-name "${interaction}"
            --output-base "${OUTPUT_BASE}"
            --genzi-python "${GENZI_PYTHON}"
            --device cuda:0
            --seed "${worker_seed}"
            --stages "${STAGES}"
            --exp-name "dual_${BATCH_ID}_gpu${physical_gpu}_${interaction}"
            --wandb-mode "${WANDB_MODE}"
            --no-root-summary
        )
        if [[ -n "${MAX_STEPS}" ]]; then
            run_args+=(--max-steps "${MAX_STEPS}")
        fi
        "${GENZI_PYTHON}" "${RUN_SCRIPT}" "${run_args[@]}"
        echo "[GenZI worker ${worker_id}] ${interaction}: complete"
    done
}

wait_for_workers() {
    local phase="$1"
    shift
    local pids=("$@")
    local pid
    local status=0

    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            status=1
        fi
    done
    if ((status != 0)); then
        echo "${phase} failed on one or both GPU workers. Check ${LOG_DIR}." >&2
        return 1
    fi
}

echo "Batch: ${BATCH_ID}"
echo "Interactions: ${#INTERACTIONS[@]}"
echo "GPU ${GPU_IDS[0]}: ${SHARD0[*]}"
echo "GPU ${GPU_IDS[1]}: ${SHARD1[*]}"
echo "Logs: ${LOG_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Phase 1: prepare all interactions"
    echo "Phase 2: run GenZI only after all preparation workers succeed"
    exit 0
fi

declare -a WORKER_PIDS=()
prepare_worker 0 "${GPU_IDS[0]}" "${SHARD0[@]}" \
    > >(tee "${LOG_DIR}/prep_gpu_${GPU_IDS[0]}.log") 2>&1 &
WORKER_PIDS+=("$!")
prepare_worker 1 "${GPU_IDS[1]}" "${SHARD1[@]}" \
    > >(tee "${LOG_DIR}/prep_gpu_${GPU_IDS[1]}.log") 2>&1 &
WORKER_PIDS+=("$!")

terminate_workers() {
    kill "${WORKER_PIDS[@]}" 2>/dev/null || true
}
trap terminate_workers INT TERM

echo "Phase 1/2: preparing all interactions"
wait_for_workers "Preparation" "${WORKER_PIDS[@]}"
trap - INT TERM

echo "All preparations completed successfully. Starting GenZI."

WORKER_PIDS=()
run_genzi_worker 0 "${GPU_IDS[0]}" "${SHARD0[@]}" \
    > >(tee "${LOG_DIR}/genzi_gpu_${GPU_IDS[0]}.log") 2>&1 &
WORKER_PIDS+=("$!")
run_genzi_worker 1 "${GPU_IDS[1]}" "${SHARD1[@]}" \
    > >(tee "${LOG_DIR}/genzi_gpu_${GPU_IDS[1]}.log") 2>&1 &
WORKER_PIDS+=("$!")

trap terminate_workers INT TERM
echo "Phase 2/2: running GenZI for all interactions"
wait_for_workers "GenZI" "${WORKER_PIDS[@]}"
trap - INT TERM

echo "All ${#INTERACTIONS[@]} interactions completed successfully."
