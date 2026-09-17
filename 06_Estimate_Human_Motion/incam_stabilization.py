from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any


def _ensure_gvhmr_on_path(gvhmr_path: Path) -> None:
    gvhmr_root = str(gvhmr_path.resolve())
    if gvhmr_root not in sys.path:
        sys.path.insert(0, gvhmr_root)


def _clone_param_dict(params: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in params.items():
        cloned[key] = value.clone() if hasattr(value, "clone") else copy.deepcopy(value)
    return cloned


def compute_stabilized_incam_params(
    result_data: dict[str, Any],
    gvhmr_path: Path,
) -> dict[str, Any]:
    """Map the post-processed global motion back into a static camera frame.

    GVHMR's anti-foot-sliding post-process updates the global trajectory, while the
    raw incam trajectory is kept as-predicted. For static-camera videos, the
    corrected global motion can be projected back into camera coordinates through a
    single world-to-camera transform estimated from frame 0.
    """
    _ensure_gvhmr_on_path(gvhmr_path)

    import torch
    from hmr4d.utils.geo.hmr_global import get_T_w2c_from_wcparams, get_c_rootparam
    from hmr4d.utils.smplx_utils import make_smplx

    global_params = result_data["smpl_params_global"]
    raw_incam_params = result_data.get(
        "smpl_params_incam_raw", result_data["smpl_params_incam"]
    )

    stabilized = {
        "body_pose": global_params["body_pose"].clone(),
        "betas": global_params["betas"].clone(),
    }

    with torch.no_grad():
        betas = stabilized["betas"]
        beta_frame0 = betas[:1] if betas.shape[0] > 1 else betas
        smplx_model = make_smplx("supermotion")
        offset = smplx_model.get_skeleton(beta_frame0)[0, 0]

        T_w2c = get_T_w2c_from_wcparams(
            global_orient_w=global_params["global_orient"][:1],
            transl_w=global_params["transl"][:1],
            global_orient_c=raw_incam_params["global_orient"][:1],
            transl_c=raw_incam_params["transl"][:1],
            offset=offset,
        )[0]

        global_orient_c, transl_c = get_c_rootparam(
            global_orient=global_params["global_orient"],
            transl=global_params["transl"],
            T_w2c=T_w2c,
            offset=offset,
        )

    stabilized["global_orient"] = global_orient_c
    stabilized["transl"] = transl_c
    return stabilized


def stabilize_result_file(
    result_path: Path,
    gvhmr_path: Path,
    overwrite_incam: bool = True,
) -> bool:
    """Overwrite smpl_params_incam with a stabilized camera-frame motion."""
    _ensure_gvhmr_on_path(gvhmr_path)

    import torch

    result_data = torch.load(result_path, map_location="cpu")
    stabilized = compute_stabilized_incam_params(result_data, gvhmr_path)
    if overwrite_incam:
        result_data["smpl_params_incam"] = _clone_param_dict(stabilized)
    result_data.pop("smpl_params_incam_raw", None)
    result_data.pop("smpl_params_incam_stabilized", None)

    torch.save(result_data, result_path)
    return True
