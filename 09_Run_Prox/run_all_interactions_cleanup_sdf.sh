#!/usr/bin/env bash
set -euo pipefail

module_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
eligible_root="${module_dir}/../03_Estimate_Contact_Agentic/output"
output_root="${module_dir}/output"
prox_python="/root/miniconda3/envs/prox/bin/python"

mkdir -p "${output_root}"

mapfile -t interactions < <(
    find "${eligible_root}" -mindepth 1 -maxdepth 1 -type d \
        -name 'interaction_*' -printf '%f\n' | sort -V
)

echo "[*] Running ${#interactions[@]} interactions sequentially with dense SDF cleanup"

for interaction in "${interactions[@]}"; do
    interaction_output="${output_root}/${interaction}"
    tsdf_debug="${interaction_output}/debug/tsdf"

    echo
    echo "[*] Running ${interaction}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONUNBUFFERED=1 \
        "${prox_python}" "${module_dir}/00_run_prox.py" \
        --interaction_name "${interaction}" \
        2>&1 | tee "${output_root}/run_${interaction}_384.log"

    test -s "${interaction_output}/result.pkl"
    test -s "${interaction_output}/final_smplx_camera.ply"
    test -s "${interaction_output}/final_smplx_world.ply"
    test -s "${interaction_output}/overlay.png"
    test -s "${interaction_output}/metadata.json"

    echo "[*] ${interaction} succeeded; deleting reproducible dense SDF artifacts"
    rm -rf -- "${interaction_output}/sdf"
    rm -f -- \
        "${tsdf_debug}/tsdf_full.npz" \
        "${tsdf_debug}/tsdf_full_colored.ply" \
        "${tsdf_debug}/tsdf_negative.ply" \
        "${tsdf_debug}/tsdf_positive_observed.ply" \
        "${tsdf_debug}/tsdf_boundary_crossings.ply" \
        "${tsdf_debug}/tsdf_unknown.ply"
done

echo
echo "[*] Finished all interactions"
