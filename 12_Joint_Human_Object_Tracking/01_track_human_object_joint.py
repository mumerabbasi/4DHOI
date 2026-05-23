"""Unified object-trajectory and human-object contact optimization.

This module intentionally does not depend on module-10 outputs.  It reads the
same object point tracks that module 10 used, builds the same frame-0
mesh-surface correspondences, and then runs two stages over one object state:

1. robust object trajectory tracking from 2D tracks;
2. contact-aware refinement with the module-11 human/object losses.

Humans are frozen GVHMR SMPL-X sequences transformed by module-09's alignment
matrix.  Only object SE(3) trajectories and bounded object scales are optimized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODULE10_DIR = PROJECT_DIR / "10_Track_Object_Mesh"
MODULE11_DIR = PROJECT_DIR / "11_Track_Human_Object_Mesh"

for _path in (str(MODULE10_DIR), str(MODULE11_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import tracking_utils as track_utils  # noqa: E402
from data_loading import (  # noqa: E402
    OBJECT_COLORS_BGR,
    SURFACE_SAMPLE_SEED,
    _build_sdf_grid,
    _extract_vertex_colors,
    _infer_image_size,
    _load_human_data,
    _load_object_mask_targets,
    _load_object_part_segments,
    _load_smpl_body_and_contact_seg,
    _parse_pag,
    _resolve_interaction_node,
    _sample_surface_points,
)
from geometry import bounded_log_scale_delta  # noqa: E402
from losses import (  # noqa: E402
    compute_all_losses,
    compute_final_loss_diagnostics,
    get_scaled_loss_terms,
    rotation_smoothness_loss,
    rotation_static_loss,
    simple_smoothness_loss,
    simple_static_loss,
)
from models import (  # noqa: E402
    DiagnosticLossResult,
    HumanData,
    InteractionEdge,
    ObjectData,
    ObjectPartSegments,
    OptimizationResult,
    ProblemContext,
)
from utils import (  # noqa: E402
    close_ffmpeg,
    draw_overlay,
    ensure_dir,
    list_images,
    start_ffmpeg_writer,
)


_MODULE10_SPEC = importlib.util.spec_from_file_location(
    "module10_object_tracker",
    MODULE10_DIR / "01_track_object_mesh.py",
)
if _MODULE10_SPEC is None or _MODULE10_SPEC.loader is None:
    raise ImportError("Could not load module-10 object tracker helpers.")
module10 = importlib.util.module_from_spec(_MODULE10_SPEC)
sys.modules[_MODULE10_SPEC.name] = module10
_MODULE10_SPEC.loader.exec_module(module10)


LOSS_KEYS_STAGE2 = (
    "track_reproj",
    "object_cd2d",
    "object_part_cd2d",
    "object_smooth_trans",
    "object_smooth_rot",
    "object_scale",
    "intersect",
    "nocontact",
    "contact_drift",
)


@dataclass
class ObjectTrackState:
    name: str
    slug: str
    is_translational: bool
    is_rotational: bool
    mesh: trimesh.Trimesh
    verts: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    x0: torch.Tensor
    obs_uv: torch.Tensor
    vis: torch.Tensor
    masks: torch.Tensor
    rotvecs: torch.nn.Parameter
    trans: torch.nn.Parameter
    raw_scale_delta: torch.nn.Parameter
    num_input_tracks: int
    num_valid_seed_tracks: int
    num_dropped_invalid_face: int
    num_dropped_outside_mask0: int
    num_dropped_nonfinite_seed: int
    pnp_info: list[dict[str, Any]]


@dataclass
class TrackingLossBundle:
    total: torch.Tensor
    track_reproj: torch.Tensor
    smooth_trans: torch.Tensor
    smooth_rot: torch.Tensor
    scale_prior: torch.Tensor
    per_object: dict[str, dict[str, torch.Tensor]]


@dataclass
class Stage2LossBundle:
    total: torch.Tensor
    track_reproj: torch.Tensor
    contact_result: Any


def _resolve_path(path_str: str | None, base: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def _save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    track_utils._save_csv(path, rows)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pag_path(args: argparse.Namespace) -> Path:
    if args.pag_file is not None:
        pag_path = _resolve_path(args.pag_file, SCRIPT_DIR)
        if pag_path is None or not pag_path.exists():
            raise FileNotFoundError(f"PAG file not found: {pag_path}")
        return pag_path

    pag_dir = PROJECT_DIR / "01_Generate_PAG" / "output" / args.interaction_name
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in {pag_dir}")
    return candidates[0].resolve()


def _resolve_smpl_seg_path(args: argparse.Namespace) -> Path:
    if args.smpl_seg_json is not None:
        seg_path = _resolve_path(args.smpl_seg_json, SCRIPT_DIR)
        if seg_path is None or not seg_path.exists():
            raise FileNotFoundError(f"SMPL-X segmentation JSON not found: {seg_path}")
        return seg_path
    return (
        PROJECT_DIR
        / "06_Estimate_Human_Motion"
        / "assets"
        / "smplx_vert_segmentation.json"
    ).resolve()


def _resolve_dirs(args: argparse.Namespace) -> dict[str, Path]:
    interaction = args.interaction_name
    output_root = _resolve_path(args.output_root, SCRIPT_DIR)
    assert output_root is not None
    return {
        "object_tracks": (
            _resolve_path(args.object_point_tracks_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "07_Track_Object_Points" / "output" / interaction)
        ).resolve(),
        "aligned": (
            _resolve_path(args.aligned_mesh_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "09_Align_Meshes" / "output" / interaction)
        ).resolve(),
        "human_motion": (
            _resolve_path(args.human_motion_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "06_Estimate_Human_Motion" / "output" / interaction)
        ).resolve(),
        "seg_vid": (
            _resolve_path(args.segment_video_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "03_Segment_Video" / "output" / interaction)
        ).resolve(),
        "seg_obj": (
            _resolve_path(args.segment_object_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "08_Segment_Object_Mesh" / "output" / interaction)
        ).resolve(),
        "output": (output_root / interaction).resolve(),
    }


def _to_device(device_name: str) -> torch.device:
    return track_utils._to_device(device_name)


def _load_alignment_transforms(aligned_dir: Path) -> dict[str, dict[str, Any]]:
    path = aligned_dir / "meshes" / "transforms.json"
    payload = _load_json(path)
    rows = payload["transforms"]
    return {str(row["slug"]): row for row in rows}


def _human_result_path(
    human_slug: str,
    row: dict[str, Any],
    human_motion_dir: Path,
) -> Path:
    input_path = row.get("input_mesh_path")
    if isinstance(input_path, str) and input_path:
        path = Path(input_path)
        if path.exists():
            return path.resolve()
    return (
        human_motion_dir / "humans" / human_slug / "hmr4d_results.pt"
    ).resolve()


def _apply_transform_np(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rot_scale = matrix[:3, :3].astype(np.float32)
    trans = matrix[:3, 3].astype(np.float32)
    return (verts.astype(np.float32) @ rot_scale.T) + trans[None, None, :]


def _load_smplx_human_sequence(
    result_path: Path,
    smplx_layer: Any,
    alignment_matrix: np.ndarray,
    num_frames: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(str(result_path), map_location="cpu")
    params = data["smpl_params_incam"]
    body_pose = params["body_pose"].detach().clone().float()[:num_frames]
    global_orient = params["global_orient"].detach().clone().float()[:num_frames]
    transl = params["transl"].detach().clone().float()[:num_frames]
    betas_raw = params["betas"].detach().clone().float()
    beta0 = betas_raw[:1] if betas_raw.ndim > 1 else betas_raw.view(1, -1)

    verts_batches: list[np.ndarray] = []
    for start in range(0, body_pose.shape[0], batch_size):
        end = min(start + batch_size, body_pose.shape[0])
        count = end - start
        zeros_3 = torch.zeros(count, 3, dtype=torch.float32)
        zeros_hand = torch.zeros(
            count,
            int(getattr(smplx_layer, "num_pca_comps", 12)),
            dtype=torch.float32,
        )
        zeros_expr = torch.zeros(
            count,
            int(getattr(smplx_layer, "num_expression_coeffs", 10)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = smplx_layer(
                betas=beta0.repeat(count, 1),
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                transl=transl[start:end],
                left_hand_pose=zeros_hand,
                right_hand_pose=zeros_hand,
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
                expression=zeros_expr,
            )
        verts_batches.append(output.vertices.detach().cpu().numpy().astype(np.float32))

    verts = np.concatenate(verts_batches, axis=0)
    verts = _apply_transform_np(verts, alignment_matrix)
    faces = np.asarray(smplx_layer.faces, dtype=np.int32)
    return verts.astype(np.float32), faces


def _load_frozen_humans(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    transforms_by_slug: dict[str, dict[str, Any]],
    body_seg: dict[str, np.ndarray],
    contact_seg: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, HumanData], list[str], dict[str, dict[str, Any]], int]:
    import smplx

    smpl_folder = _resolve_path(args.smpl_folder, SCRIPT_DIR)
    if smpl_folder is None or not smpl_folder.exists():
        raise FileNotFoundError(f"SMPL-X body model folder not found: {smpl_folder}")

    print(f"Loading frozen SMPL-X humans from GVHMR using: {smpl_folder}")
    smplx_layer = smplx.create(
        str(smpl_folder),
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
    )
    smplx_layer.eval()

    human_rows = [
        row for row in transforms_by_slug.values() if row.get("kind") == "human"
    ]
    if not human_rows:
        raise RuntimeError("No human rows found in module-09 transforms.json.")

    humans: dict[str, HumanData] = {}
    human_keys: list[str] = []
    human_metadata: dict[str, dict[str, Any]] = {}
    num_frames: int | None = None

    for row in sorted(human_rows, key=lambda item: str(item["slug"])):
        slug = str(row["slug"])
        result_path = _human_result_path(slug, row, dirs["human_motion"])
        if not result_path.exists():
            raise FileNotFoundError(f"GVHMR result file not found: {result_path}")

        data = torch.load(str(result_path), map_location="cpu")
        params = data["smpl_params_incam"]
        available_frames = int(params["body_pose"].shape[0])
        if num_frames is None:
            num_frames = available_frames
        else:
            num_frames = min(num_frames, available_frames)

    if args.end_frame >= 0:
        num_frames = min(int(num_frames), int(args.end_frame) + 1)
    assert num_frames is not None and num_frames > 0

    for row in sorted(human_rows, key=lambda item: str(item["slug"])):
        slug = str(row["slug"])
        name = str(row.get("name", slug)).replace("_", " ")
        result_path = _human_result_path(slug, row, dirs["human_motion"])
        matrix = np.asarray(
            row["source_to_output_matrix_4x4"],
            dtype=np.float32,
        )
        verts_np, faces_np = _load_smplx_human_sequence(
            result_path=result_path,
            smplx_layer=smplx_layer,
            alignment_matrix=matrix,
            num_frames=num_frames,
            batch_size=max(1, int(args.smplx_batch_size)),
        )
        humans[slug] = _load_human_data(
            human_name=name,
            human_slug=slug,
            human_verts_np=verts_np,
            human_faces=faces_np,
            body_seg=body_seg,
            contact_seg=contact_seg,
            device=device,
        )
        human_keys.append(slug)
        human_metadata[slug] = {
            "source": "module_06_gvhmr_smpl_params_incam",
            "hmr4d_results": str(result_path),
            "module09_transform": matrix.tolist(),
            "num_frames": int(verts_np.shape[0]),
            "num_verts": int(verts_np.shape[1]),
            "num_faces": int(faces_np.shape[0]),
            "future_human_optimization_hooks": {
                "root_translation_delta": False,
                "root_rotation_delta": False,
                "body_pose_delta": False,
                "gvhmr_root_orient_prior": False,
                "gvhmr_body_pose_prior": False,
                "human_temporal_smoothness": False,
            },
        }
        print(
            f"  Human {slug}: {verts_np.shape[0]} frames, "
            f"{verts_np.shape[1]} verts, {faces_np.shape[0]} faces"
        )

    return humans, human_keys, human_metadata, int(num_frames)


def _tracks_total_frames(
    tracks_path: Path,
    vis_path: Path,
    mask_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    tracks_raw = np.load(str(tracks_path))
    vis_raw = np.load(str(vis_path))
    tracks_nt2, vis_nt = track_utils._normalize_tracks_vis_with_mask_length(
        tracks_raw,
        vis_raw,
        mask_count,
    )
    total = min(int(mask_count), int(tracks_nt2.shape[1]), int(vis_nt.shape[1]))
    return tracks_nt2, vis_nt, total


def _discover_num_frames(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag: Any,
    human_num_frames: int,
) -> int:
    num_frames = human_num_frames
    for state in pag.object_states:
        slug = state.slug
        mask_dir = track_utils._resolve_object_mask_dir(dirs["seg_vid"], slug)
        mask_paths = track_utils._list_mask_files(mask_dir)
        tracks_path = dirs["object_tracks"] / slug / "tracks.npy"
        vis_path = dirs["object_tracks"] / slug / "visibility.npy"
        if not tracks_path.exists() or not vis_path.exists() or not mask_paths:
            continue
        _, _, total = _tracks_total_frames(tracks_path, vis_path, len(mask_paths))
        num_frames = min(num_frames, total)

    if args.end_frame >= 0:
        num_frames = min(num_frames, int(args.end_frame) + 1)
    if num_frames <= 0:
        raise RuntimeError("No frames available after clipping inputs.")
    return int(num_frames)


def _initial_object_params(
    x0: torch.Tensor,
    tracks_valid: np.ndarray,
    vis_valid: np.ndarray,
    k: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    n_params = max(tracks_valid.shape[1] - 1, 0)
    if args.disable_pnp_init or n_params == 0:
        return (
            torch.zeros(n_params, 3, device=device),
            torch.zeros(n_params, 3, device=device),
            [],
        )

    obs_uv_tm = tracks_valid.transpose(1, 0, 2)
    vis_tm = vis_valid.transpose(1, 0)
    rv_init, tr_init, pnp_info = module10._pnp_sequential_init(
        x0.cpu().numpy(),
        obs_uv_tm,
        vis_tm,
        k,
        float(args.pnp_ransac_thresh),
    )
    return (
        torch.from_numpy(rv_init).to(device, torch.float32),
        torch.from_numpy(tr_init).to(device, torch.float32),
        pnp_info,
    )


def _load_track_object_state(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    state: Any,
    k: np.ndarray,
    num_frames: int,
    device: torch.device,
) -> ObjectTrackState | None:
    slug = state.slug
    mesh_path = dirs["aligned"] / "meshes" / f"{slug}.ply"
    tracks_path = dirs["object_tracks"] / slug / "tracks.npy"
    vis_path = dirs["object_tracks"] / slug / "visibility.npy"
    mask_dir = track_utils._resolve_object_mask_dir(dirs["seg_vid"], slug)

    missing = [
        str(path)
        for path in (mesh_path, tracks_path, vis_path, mask_dir)
        if not path.exists()
    ]
    if missing:
        print(f"  [SKIP] {slug}: missing {', '.join(missing)}")
        return None

    mesh = trimesh.load(str(mesh_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Cannot load mesh: {mesh_path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_colors = _extract_vertex_colors(mesh)

    mask_paths = track_utils._list_mask_files(mask_dir)
    masks_np, h_mask, w_mask = track_utils._load_mask_stack(
        mask_paths[:num_frames],
        int(args.mask_threshold),
    )
    tracks_nt2, vis_nt, _ = _tracks_total_frames(
        tracks_path,
        vis_path,
        len(mask_paths),
    )
    tracks_nt2 = tracks_nt2[:, :num_frames].astype(np.float32)
    vis_nt = vis_nt[:, :num_frames].astype(np.float32)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    mapping = module10._map_seed_points_to_mesh(
        tracks_nt2[:, 0, :],
        verts,
        faces,
        masks_np[0],
        fx,
        fy,
        cx,
        cy,
        h_mask,
        w_mask,
        device,
        int(args.bin_size),
        float(args.mask_gate_threshold),
    )
    valid = mapping.valid_seed_mask
    tracks_valid = tracks_nt2[valid]
    vis_valid = vis_nt[valid]
    if tracks_valid.shape[0] < int(args.min_valid_tracks):
        raise RuntimeError(
            f"{slug}: too few valid tracks {tracks_valid.shape[0]} "
            f"(need {args.min_valid_tracks})"
        )

    obs_uv = torch.from_numpy(tracks_valid).to(device, torch.float32).permute(1, 0, 2)
    vis_tm = torch.from_numpy(vis_valid).to(device, torch.float32).permute(1, 0)
    masks_t = torch.from_numpy(masks_np).to(device, torch.float32)

    rv_init, tr_init, pnp_info = _initial_object_params(
        mapping.points_cv,
        tracks_valid,
        vis_valid,
        k,
        args,
        device,
    )
    if pnp_info:
        pnp_ok = sum(1 for item in pnp_info if item["pnp_ok"])
        print(f"  {slug}: PnP init {pnp_ok}/{len(pnp_info)} frames OK")

    if float(args.outlier_reproj_thresh_px) > 0.0 and rv_init.numel() > 0:
        outlier_mask = module10._identify_outlier_tracks(
            mapping.points_cv,
            obs_uv,
            vis_tm,
            rv_init,
            tr_init,
            fx,
            fy,
            cx,
            cy,
            float(args.outlier_reproj_thresh_px),
            float(args.outlier_max_fraction),
        )
        n_outlier = int(outlier_mask.sum().item())
        if n_outlier > 0:
            vis_tm[:, outlier_mask] = 0.0
            print(f"  {slug}: removed {n_outlier} global track outliers")

    return ObjectTrackState(
        name=state.name,
        slug=slug,
        is_translational=bool(state.is_translational),
        is_rotational=bool(state.is_rotational),
        mesh=mesh,
        verts=verts,
        faces=faces,
        vertex_colors=vertex_colors,
        x0=mapping.points_cv,
        obs_uv=obs_uv,
        vis=vis_tm,
        masks=masks_t,
        rotvecs=torch.nn.Parameter(rv_init.clone()),
        trans=torch.nn.Parameter(tr_init.clone()),
        raw_scale_delta=torch.nn.Parameter(torch.zeros(1, device=device)),
        num_input_tracks=int(tracks_nt2.shape[0]),
        num_valid_seed_tracks=int(tracks_valid.shape[0]),
        num_dropped_invalid_face=int(mapping.invalid_face_count),
        num_dropped_outside_mask0=int(mapping.outside_mask0_count),
        num_dropped_nonfinite_seed=int(mapping.nonfinite_seed_count),
        pnp_info=pnp_info,
    )


def _scale_from_raw(raw_scale_delta: torch.Tensor, max_log_scale_delta: float) -> torch.Tensor:
    return torch.exp(bounded_log_scale_delta(raw_scale_delta, max_log_scale_delta))


def _full_T(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    return module10._build_T_matrices(rotvecs, trans, num_frames, device)


def _tracking_object_loss(
    obj: ObjectTrackState,
    args: argparse.Namespace,
    k: np.ndarray,
    huber_delta: float,
) -> dict[str, torch.Tensor]:
    device = obj.x0.device
    num_frames = obj.obs_uv.shape[0]
    T_mats = _full_T(obj.rotvecs, obj.trans, num_frames, device)
    scale = _scale_from_raw(obj.raw_scale_delta, float(args.max_log_scale_delta))
    points = obj.x0 * scale
    rot = T_mats[:, :3, :3]
    trans = T_mats[:, :3, 3]
    xt = torch.einsum("tij,mj->tmi", rot, points) + trans.unsqueeze(1)

    z = xt[..., 2]
    z_valid = z > 1e-6
    z_safe = torch.where(z_valid, z, torch.ones_like(z))
    pred_u = float(k[0, 0]) * xt[..., 0] / z_safe + float(k[0, 2])
    pred_v = float(k[1, 1]) * xt[..., 1] / z_safe + float(k[1, 2])
    pred_uv = torch.stack([pred_u, pred_v], dim=-1)

    finite_obs = torch.isfinite(obj.obs_uv).all(dim=-1)
    mask_vals = module10._sample_masks_bilinear_seq(obj.masks, obj.obs_uv)
    mask_gate = mask_vals >= float(args.mask_gate_threshold)
    vis_w = (
        obj.vis
        if float(args.visibility_threshold) <= 0.0
        else torch.where(
            obj.vis >= float(args.visibility_threshold),
            obj.vis,
            torch.zeros_like(obj.vis),
        )
    )
    weights = vis_w * mask_gate.float() * finite_obs.float() * z_valid.float()
    r2 = ((obj.obs_uv - pred_uv) ** 2).sum(dim=-1)
    robust = module10._huber_on_squared(r2, float(huber_delta))
    reproj = (weights * robust).sum() / weights.sum().clamp(min=1.0)

    if obj.is_translational:
        smooth_trans = simple_smoothness_loss(trans)
    else:
        smooth_trans = simple_static_loss(trans) * 10.0

    if obj.is_rotational:
        smooth_rot = rotation_smoothness_loss(rot)
    else:
        smooth_rot = rotation_static_loss(rot) * 10.0

    scale_prior = F.relu(torch.abs(scale - 1.0) - 0.1)

    return {
        "reproj": reproj,
        "smooth_trans": smooth_trans,
        "smooth_rot": smooth_rot,
        "scale_prior": scale_prior.reshape(()),
        "r2": r2,
        "weights": weights,
        "pred_uv": pred_uv,
        "T_mats": T_mats,
        "scale": scale.reshape(()),
    }


def _compute_stage1_loss(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
    huber_delta: float,
) -> TrackingLossBundle:
    per_object = {
        slug: _tracking_object_loss(obj, args, k, huber_delta)
        for slug, obj in objects.items()
    }
    reproj = torch.stack([item["reproj"] for item in per_object.values()]).mean()
    smooth_trans = torch.stack(
        [item["smooth_trans"] for item in per_object.values()]
    ).mean()
    smooth_rot = torch.stack(
        [item["smooth_rot"] for item in per_object.values()]
    ).mean()
    scale_prior = torch.stack(
        [item["scale_prior"] for item in per_object.values()]
    ).mean()
    total = (
        float(args.lambda_track_reproj) * reproj
        + float(args.lambda_smooth_trans) * smooth_trans
        + float(args.lambda_smooth_rot) * smooth_rot
        + float(args.lambda_scale) * scale_prior
    )
    return TrackingLossBundle(
        total=total,
        track_reproj=reproj,
        smooth_trans=smooth_trans,
        smooth_rot=smooth_rot,
        scale_prior=scale_prior,
        per_object=per_object,
    )


def _stage1_row(iteration: int, bundle: TrackingLossBundle, args: argparse.Namespace) -> dict[str, Any]:
    active_pairs = sum(
        int(item["weights"].detach().gt(0.0).sum().item())
        for item in bundle.per_object.values()
    )
    return {
        "iter": int(iteration),
        "total": float(bundle.total.detach().item()),
        "track_reproj_raw": float(bundle.track_reproj.detach().item()),
        "track_reproj_scaled": float(
            float(args.lambda_track_reproj) * bundle.track_reproj.detach().item()
        ),
        "smooth_trans_raw": float(bundle.smooth_trans.detach().item()),
        "smooth_trans_scaled": float(
            float(args.lambda_smooth_trans) * bundle.smooth_trans.detach().item()
        ),
        "smooth_rot_raw": float(bundle.smooth_rot.detach().item()),
        "smooth_rot_scaled": float(
            float(args.lambda_smooth_rot) * bundle.smooth_rot.detach().item()
        ),
        "scale_prior_raw": float(bundle.scale_prior.detach().item()),
        "scale_prior_scaled": float(
            float(args.lambda_scale) * bundle.scale_prior.detach().item()
        ),
        "active_pairs": active_pairs,
    }


def _retrim_tracks(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
) -> int:
    trimmed_total = 0
    with torch.no_grad():
        for obj in objects.values():
            item = _tracking_object_loss(
                obj,
                args,
                k,
                huber_delta=float(args.huber_delta_px),
            )
            err = torch.sqrt(item["r2"].clamp(min=1e-12))
            for frame_idx in range(obj.vis.shape[0]):
                active = obj.vis[frame_idx] > 0.0
                if int(active.sum().item()) < 20:
                    continue
                threshold = torch.quantile(
                    err[frame_idx, active],
                    float(args.retrim_percentile) / 100.0,
                )
                to_zero = active & (err[frame_idx] > threshold)
                n_zero = int(to_zero.sum().item())
                if n_zero > 0:
                    obj.vis[frame_idx, to_zero] = 0.0
                    trimmed_total += n_zero
    return trimmed_total


def _run_stage1(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
) -> list[dict[str, Any]]:
    params: list[torch.nn.Parameter] = []
    for obj in objects.values():
        params.extend([obj.rotvecs, obj.trans, obj.raw_scale_delta])
    if not params:
        raise RuntimeError("No object parameters to optimize.")

    iter_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    opt = torch.optim.Adam(params, lr=float(args.stage1_lr))
    scheduler = None
    if args.stage1_lr_schedule == "cosine" and args.stage1_iters > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(args.stage1_iters),
            eta_min=float(args.stage1_lr) / 20.0,
        )

    huber_target = float(args.huber_delta_px)
    huber_start = huber_target * 3.0 if args.graduated_huber else huber_target

    print(f"\n[Stage 1] object tracking init: {args.stage1_iters} Adam iterations")
    if args.stage1_iters <= 0:
        with torch.no_grad():
            bundle = _compute_stage1_loss(objects, args, k, huber_target)
        iter_rows.append(_stage1_row(0, bundle, args))
        return iter_rows

    for iteration in range(1, int(args.stage1_iters) + 1):
        if args.graduated_huber:
            frac = min(iteration / max(args.stage1_iters * 0.5, 1), 1.0)
            huber = huber_start + (huber_target - huber_start) * frac
        else:
            huber = huber_target

        opt.zero_grad(set_to_none=True)
        bundle = _compute_stage1_loss(objects, args, k, huber)
        if not torch.isfinite(bundle.total):
            raise RuntimeError(f"Non-finite Stage-1 loss at iter {iteration}")
        bundle.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
        opt.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            eval_bundle = _compute_stage1_loss(objects, args, k, huber_target)
            cur = float(eval_bundle.total.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {
                    slug: {
                        "rot": obj.rotvecs.detach().clone(),
                        "trans": obj.trans.detach().clone(),
                        "scale": obj.raw_scale_delta.detach().clone(),
                    }
                    for slug, obj in objects.items()
                }

            if (
                args.retrim_interval > 0
                and iteration % int(args.retrim_interval) == 0
                and iteration < int(args.stage1_iters)
            ):
                n_trimmed = _retrim_tracks(objects, args, k)
                if n_trimmed > 0:
                    print(
                        f"  Stage1 retrim@{iteration}: zeroed {n_trimmed} "
                        "track-frame pairs"
                    )

            if (
                args.debug_save_interval <= 1
                or iteration % int(args.debug_save_interval) == 0
                or iteration == int(args.stage1_iters)
            ):
                iter_rows.append(_stage1_row(iteration, eval_bundle, args))

            if (
                args.log_every > 0
                and (iteration % int(args.log_every) == 0 or iteration == args.stage1_iters)
            ):
                print(
                    f"  stage1 {iteration:05d} total={eval_bundle.total.item():.6f} "
                    f"track={eval_bundle.track_reproj.item():.6f} "
                    f"smooth_t={eval_bundle.smooth_trans.item():.6f} "
                    f"smooth_r={eval_bundle.smooth_rot.item():.6f} "
                    f"scale={eval_bundle.scale_prior.item():.6f}"
                )

    if best_state:
        with torch.no_grad():
            for slug, state in best_state.items():
                objects[slug].rotvecs.copy_(state["rot"])
                objects[slug].trans.copy_(state["trans"])
                objects[slug].raw_scale_delta.copy_(state["scale"])
    return iter_rows


def _full_delta_vars(
    objects: dict[str, ObjectTrackState],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full_rot: dict[str, torch.Tensor] = {}
    full_trans: dict[str, torch.Tensor] = {}
    raw_scales: dict[str, torch.Tensor] = {}
    for slug, obj in objects.items():
        zero_rot = torch.zeros(1, 3, device=obj.rotvecs.device, dtype=obj.rotvecs.dtype)
        zero_trans = torch.zeros(1, 3, device=obj.trans.device, dtype=obj.trans.dtype)
        full_rot[slug] = torch.cat([zero_rot, obj.rotvecs], dim=0)
        full_trans[slug] = torch.cat([zero_trans, obj.trans], dim=0)
        raw_scales[slug] = obj.raw_scale_delta
    return full_rot, full_trans, raw_scales


def _stage2_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        tracking_weight=0.0,
        object_cd2d_weight_start=float(args.object_cd2d_weight),
        object_cd2d_weight_end=float(args.object_cd2d_weight),
        object_part_cd2d_weight_start=float(args.object_part_cd2d_weight),
        object_part_cd2d_weight_end=float(args.object_part_cd2d_weight),
        object_smooth_trans_weight_start=float(args.object_smooth_trans_weight),
        object_smooth_trans_weight_end=float(args.object_smooth_trans_weight),
        object_smooth_rot_weight_start=float(args.object_smooth_rot_weight),
        object_smooth_rot_weight_end=float(args.object_smooth_rot_weight),
        object_scale_weight_start=float(args.object_scale_weight),
        object_scale_weight_end=float(args.object_scale_weight),
        intersect_weight_start=float(args.intersect_weight_start),
        intersect_weight_end=float(args.intersect_weight_end),
        nocontact_weight_start=float(args.nocontact_weight),
        nocontact_weight_end=float(args.nocontact_weight),
        contact_drift_weight_start=float(args.contact_drift_weight_start),
        contact_drift_weight_end=float(args.contact_drift_weight_end),
        optimize_object_scale=True,
        max_log_scale_delta=float(args.max_log_scale_delta),
    )


def _compute_stage2_loss(
    objects: dict[str, ObjectTrackState],
    context: ProblemContext,
    args: argparse.Namespace,
    iteration: int,
) -> Stage2LossBundle:
    track_bundle = _compute_stage1_loss(
        objects,
        args,
        context.k,
        float(args.huber_delta_px),
    )
    full_rot, full_trans, raw_scales = _full_delta_vars(objects)
    contact_result = compute_all_losses(
        full_rot,
        full_trans,
        raw_scales,
        context.objects,
        context.humans,
        context.interaction_edges,
        context.obj_keys,
        _stage2_args(args),
        iteration=iteration,
        total_iters=max(int(args.stage2_iters), 1),
        k=context.k_torch,
        width=context.width,
        height=context.height,
    )
    total = float(args.lambda_track_reproj) * track_bundle.track_reproj
    total = total + contact_result.total
    return Stage2LossBundle(
        total=total,
        track_reproj=track_bundle.track_reproj,
        contact_result=contact_result,
    )


def _stage2_row(iteration: int, bundle: Stage2LossBundle, args: argparse.Namespace) -> dict[str, Any]:
    scaled_terms = get_scaled_loss_terms(bundle.contact_result)
    row: dict[str, Any] = {
        "iter": int(iteration),
        "total": float(bundle.total.detach().item()),
        "track_reproj_weight": float(args.lambda_track_reproj),
        "track_reproj_raw": float(bundle.track_reproj.detach().item()),
        "track_reproj_scaled": float(
            float(args.lambda_track_reproj) * bundle.track_reproj.detach().item()
        ),
    }
    for key in (
        "object_cd2d",
        "object_part_cd2d",
        "object_smooth_trans",
        "object_smooth_rot",
        "object_scale",
        "intersect",
        "nocontact",
        "contact_drift",
    ):
        row[f"{key}_weight"] = float(bundle.contact_result.weights[key])
        row[f"{key}_raw"] = float(getattr(bundle.contact_result, key).detach().item())
        row[f"{key}_scaled"] = float(scaled_terms[key].detach().item())
    return row


def _run_stage2(
    objects: dict[str, ObjectTrackState],
    context: ProblemContext,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    iter_rows: list[dict[str, Any]] = []
    if int(args.stage2_iters) <= 0:
        print("\n[Stage 2] skipped")
        return iter_rows

    params: list[torch.nn.Parameter] = []
    for obj in objects.values():
        params.extend([obj.rotvecs, obj.trans, obj.raw_scale_delta])

    opt = torch.optim.Adam(params, lr=float(args.stage2_lr))
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}

    print(f"\n[Stage 2] contact refinement: {args.stage2_iters} Adam iterations")
    for iteration in range(1, int(args.stage2_iters) + 1):
        opt.zero_grad(set_to_none=True)
        bundle = _compute_stage2_loss(objects, context, args, iteration)
        if not torch.isfinite(bundle.total):
            raise RuntimeError(f"Non-finite Stage-2 loss at iter {iteration}")
        bundle.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
        opt.step()

        with torch.no_grad():
            eval_bundle = _compute_stage2_loss(objects, context, args, iteration)
            cur = float(eval_bundle.total.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {
                    slug: {
                        "rot": obj.rotvecs.detach().clone(),
                        "trans": obj.trans.detach().clone(),
                        "scale": obj.raw_scale_delta.detach().clone(),
                    }
                    for slug, obj in objects.items()
                }
            if (
                args.debug_save_interval <= 1
                or iteration % int(args.debug_save_interval) == 0
                or iteration == int(args.stage2_iters)
            ):
                iter_rows.append(_stage2_row(iteration, eval_bundle, args))
            if (
                args.log_every > 0
                and (iteration % int(args.log_every) == 0 or iteration == args.stage2_iters)
            ):
                print(
                    f"  stage2 {iteration:05d} total={eval_bundle.total.item():.6f} "
                    f"track={eval_bundle.track_reproj.item():.6f} "
                    f"intersect={eval_bundle.contact_result.intersect.item():.6f} "
                    f"nocontact={eval_bundle.contact_result.nocontact.item():.6f} "
                    f"drift={eval_bundle.contact_result.contact_drift.item():.6f}"
                )

    if best_state:
        with torch.no_grad():
            for slug, state in best_state.items():
                objects[slug].rotvecs.copy_(state["rot"])
                objects[slug].trans.copy_(state["trans"])
                objects[slug].raw_scale_delta.copy_(state["scale"])
    return iter_rows


def _identity_poses(num_frames: int) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (num_frames, 1, 1))
    return poses


def _build_contact_context(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag_path: Path,
    smpl_seg_path: Path,
    k: np.ndarray,
    humans: dict[str, HumanData],
    human_keys: list[str],
    objects: dict[str, ObjectTrackState],
    num_frames: int,
    device: torch.device,
) -> ProblemContext:
    pag = _parse_pag(pag_path)
    body_seg, _ = _load_smpl_body_and_contact_seg(smpl_seg_path)
    width, height = _infer_image_size(dirs)
    k_torch = torch.from_numpy(k.astype(np.float32)).to(device)

    object_data: dict[str, ObjectData] = {}
    obj_keys: list[str] = []
    for idx, state in enumerate(pag.object_states):
        slug = state.slug
        if slug not in objects:
            continue
        track_obj = objects[slug]
        faces_i32 = track_obj.faces.astype(np.int32)

        try:
            part_segments = _load_object_part_segments(dirs["seg_obj"], slug, faces_i32)
        except FileNotFoundError:
            print(f"  [WARN] {slug}: no part segmentation, using whole mesh")
            part_segments = ObjectPartSegments(vert_ids={}, face_ids={})

        sampled_points = torch.from_numpy(
            _sample_surface_points(
                track_obj.verts,
                faces_i32,
                int(args.num_object_surface_points),
                SURFACE_SAMPLE_SEED + idx,
            )
        ).float().to(device)

        part_sampled_points: dict[str, torch.Tensor] = {}
        for part_idx, (part_name, face_ids) in enumerate(
            sorted(part_segments.face_ids.items())
        ):
            part_faces = faces_i32[face_ids]
            part_points = _sample_surface_points(
                track_obj.verts,
                part_faces,
                int(args.num_part_surface_points),
                SURFACE_SAMPLE_SEED + 97 * idx + part_idx + 1,
            )
            part_sampled_points[part_name] = torch.from_numpy(part_points).float().to(device)

        object_mask_points, part_mask_points = _load_object_mask_targets(
            dirs["seg_vid"],
            slug,
            list(part_segments.face_ids.keys()),
            num_frames,
            width,
            height,
            device,
            int(args.num_mask_points_2d),
        )

        print(f"  Building SDF for {slug} (res={args.sdf_resolution})")
        sdf_grid = _build_sdf_grid(
            track_obj.verts,
            faces_i32,
            int(args.sdf_resolution),
            device,
        )

        tracked_poses = _identity_poses(num_frames)
        zeros_rot = torch.zeros(num_frames, 3, device=device)
        zeros_trans = torch.zeros(num_frames, 3, device=device)
        object_data[slug] = ObjectData(
            name=state.name,
            slug=slug,
            state=state,
            template_verts=torch.from_numpy(track_obj.verts).float().to(device),
            faces=faces_i32,
            vertex_colors=track_obj.vertex_colors,
            faces_torch=torch.from_numpy(faces_i32.astype(np.int64)).to(device),
            tracked_poses=tracked_poses,
            tracked_poses_torch=torch.from_numpy(tracked_poses).float().to(device),
            tracked_rotvecs=zeros_rot,
            tracked_trans=zeros_trans,
            part_vert_ids=part_segments.vert_ids,
            part_face_ids=part_segments.face_ids,
            sampled_points=sampled_points,
            part_sampled_points=part_sampled_points,
            mask_points_2d=object_mask_points,
            part_mask_points_2d=part_mask_points,
            sdf_grid=sdf_grid,
            color_bgr=OBJECT_COLORS_BGR[idx % len(OBJECT_COLORS_BGR)],
        )
        obj_keys.append(slug)

    interaction_edges: list[InteractionEdge] = []
    print("\nResolving PAG interaction edges for Stage 2...")
    for edge in pag.edges:
        try:
            node_a = _resolve_interaction_node(edge.node_a, humans, object_data, body_seg)
            node_b = _resolve_interaction_node(edge.node_b, humans, object_data, body_seg)
        except (KeyError, ValueError) as exc:
            print(f"  [WARN] Skipping edge: {exc}")
            continue
        interaction_edges.append(
            InteractionEdge(
                node_a=node_a,
                node_b=node_b,
                is_continuous=edge.is_continuous,
                is_rel_static=edge.is_rel_static,
            )
        )
        print(
            f"  Edge: {node_a.raw_node} <-> {node_b.raw_node} "
            f"(continuous={edge.is_continuous}, static={edge.is_rel_static})"
        )

    return ProblemContext(
        dirs=dirs,
        out_dir=dirs["output"],
        pag_path=pag_path,
        smpl_seg_path=smpl_seg_path,
        intr_path=dirs["aligned"] / "alignment_summary.json",
        device=device,
        k=k,
        k_torch=k_torch,
        width=width,
        height=height,
        num_frames=num_frames,
        pag=pag,
        humans=humans,
        human_keys=human_keys,
        objects=object_data,
        obj_keys=obj_keys,
        interaction_edges=interaction_edges,
    )


def _final_T_mats(
    objects: dict[str, ObjectTrackState],
    num_frames: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    return {
        slug: _full_T(obj.rotvecs, obj.trans, num_frames, device)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for slug, obj in objects.items()
    }


def _final_scales(
    objects: dict[str, ObjectTrackState],
    max_log_scale_delta: float,
) -> dict[str, float]:
    return {
        slug: float(
            _scale_from_raw(obj.raw_scale_delta.detach(), max_log_scale_delta).item()
        )
        for slug, obj in objects.items()
    }


def _save_pose_json(path: Path, T_mats: np.ndarray) -> None:
    rows = [
        {"frame": int(idx), "T_4x4": T_mats[idx].tolist()}
        for idx in range(T_mats.shape[0])
    ]
    _save_json(path, rows)


def _save_transform_refined(path: Path, T_mats: np.ndarray, scale: float) -> None:
    rows = [
        {"frame": int(idx), "T_4x4": T_mats[idx].tolist()}
        for idx in range(T_mats.shape[0])
    ]
    _save_json(path, {"global_scale": float(scale), "frames": rows})


def _save_object_meshes(
    out_dir: Path,
    obj: ObjectTrackState,
    T_mats: np.ndarray,
    scale: float,
) -> None:
    ensure_dir(out_dir)
    for frame_idx in range(T_mats.shape[0]):
        rot = T_mats[frame_idx, :3, :3]
        trans = T_mats[frame_idx, :3, 3]
        verts = (obj.verts * scale) @ rot.T + trans[None, :]
        mesh = obj.mesh.copy()
        mesh.vertices = verts.astype(np.float32)
        mesh.export(str(out_dir / f"frame_{frame_idx:04d}.ply"))


def _save_human_meshes(
    out_dir: Path,
    human: HumanData,
) -> None:
    ensure_dir(out_dir)
    verts = human.base_verts.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts[0], faces=human.faces, process=False)
    for frame_idx in range(verts.shape[0]):
        mesh_t = mesh.copy()
        mesh_t.vertices = verts[frame_idx].astype(np.float32)
        mesh_t.export(str(out_dir / f"frame_{frame_idx:04d}.ply"))


def _build_delta_stats(
    obj: ObjectTrackState,
    scale: float,
) -> dict[str, Any]:
    with torch.no_grad():
        rot_norm = obj.rotvecs.norm(dim=-1)
        trans_norm = obj.trans.norm(dim=-1)
    return {
        "slug": obj.slug,
        "state_type": "absolute_object_trajectory_from_module12",
        "frame0_fixed_identity": True,
        "max_rot_deg": float(rot_norm.max().item() * 180.0 / math.pi)
        if rot_norm.numel() else 0.0,
        "mean_rot_deg": float(rot_norm.mean().item() * 180.0 / math.pi)
        if rot_norm.numel() else 0.0,
        "max_trans_m": float(trans_norm.max().item()) if trans_norm.numel() else 0.0,
        "mean_trans_m": float(trans_norm.mean().item()) if trans_norm.numel() else 0.0,
        "global_scale": float(scale),
        "num_input_tracks": obj.num_input_tracks,
        "num_valid_seed_tracks": obj.num_valid_seed_tracks,
        "num_dropped_invalid_face": obj.num_dropped_invalid_face,
        "num_dropped_outside_mask0": obj.num_dropped_outside_mask0,
        "num_dropped_nonfinite_seed": obj.num_dropped_nonfinite_seed,
        "active_track_frame_pairs": int(obj.vis.gt(0.0).sum().item()),
    }


def _frame_metrics(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
    num_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        per_object = {
            slug: _tracking_object_loss(obj, args, k, float(args.huber_delta_px))
            for slug, obj in objects.items()
        }
        for frame_idx in range(num_frames):
            row: dict[str, Any] = {"frame_idx": int(frame_idx)}
            reproj_vals = []
            active_total = 0
            for slug, item in per_object.items():
                active = item["weights"][frame_idx] > 0.0
                active_count = int(active.sum().item())
                active_total += active_count
                if active_count > 0:
                    reproj = torch.sqrt(item["r2"][frame_idx, active].clamp(min=1e-12))
                    row[f"{slug}_reproj_mean_px"] = float(reproj.mean().item())
                    reproj_vals.append(reproj.mean())
                else:
                    row[f"{slug}_reproj_mean_px"] = float("nan")
            row["active_pairs"] = active_total
            if reproj_vals:
                row["mean_reproj_px"] = float(torch.stack(reproj_vals).mean().item())
            else:
                row["mean_reproj_px"] = float("nan")
            rows.append(row)
    return rows


def _plot_loss_csv(csv_path: Path, out_path: Path, title: str) -> None:
    if not csv_path.exists():
        return
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = pd.read_csv(csv_path)
    if "iter" not in data or "total" not in data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["iter"], data["total"], label="total", linewidth=1.8)
    for col in data.columns:
        if col.endswith("_scaled") and col != "total_scaled":
            ax.plot(data["iter"], data[col], linewidth=1.0, alpha=0.75, label=col)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)


def _render_overlay(
    out_dir: Path,
    frame_paths: list[Path],
    humans: dict[str, HumanData],
    human_keys: list[str],
    objects: dict[str, ObjectTrackState],
    T_mats_by_slug: dict[str, np.ndarray],
    scales_by_slug: dict[str, float],
    k: np.ndarray,
    fps: float,
    save_pngs: bool,
    num_frames: int,
) -> None:
    if not frame_paths:
        print("  [WARN] No frames available for overlay")
        return
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print("  [WARN] Cannot read first frame for overlay")
        return
    h, w = first.shape[:2]
    writer = start_ffmpeg_writer(out_dir / "overlay.mp4", float(fps), (h, w))
    overlays_dir = out_dir / "overlays"
    if save_pngs:
        ensure_dir(overlays_dir)
    human_colors = [(255, 200, 100), (100, 220, 255), (180, 255, 140)]

    try:
        for frame_idx in range(min(num_frames, len(frame_paths))):
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                continue
            overlay = frame
            for hidx, slug in enumerate(human_keys):
                verts = humans[slug].base_verts[frame_idx].detach().cpu().numpy()
                overlay = draw_overlay(
                    overlay,
                    verts,
                    humans[slug].faces,
                    k,
                    fill_alpha=0.32,
                    contour_thickness=0,
                    color_bgr=human_colors[hidx % len(human_colors)],
                )
            for oidx, (slug, obj) in enumerate(objects.items()):
                T = T_mats_by_slug[slug][frame_idx]
                scale = scales_by_slug[slug]
                verts = (obj.verts * scale) @ T[:3, :3].T + T[:3, 3][None, :]
                overlay = draw_overlay(
                    overlay,
                    verts.astype(np.float32),
                    obj.faces,
                    k,
                    fill_alpha=0.55,
                    contour_thickness=0,
                    color_bgr=OBJECT_COLORS_BGR[oidx % len(OBJECT_COLORS_BGR)],
                )
            if save_pngs:
                cv2.imwrite(str(overlays_dir / f"overlay_{frame_idx:04d}.png"), overlay)
            if writer.stdin is not None:
                writer.stdin.write(np.ascontiguousarray(overlay).tobytes())
    finally:
        close_ffmpeg(writer)


def _save_outputs(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag_path: Path,
    smpl_seg_path: Path,
    k: np.ndarray,
    intr_path: Path,
    objects: dict[str, ObjectTrackState],
    humans: dict[str, HumanData],
    human_keys: list[str],
    human_metadata: dict[str, dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    context: ProblemContext | None,
    start_time: float,
) -> None:
    out_dir = dirs["output"]
    ensure_dir(out_dir)
    num_frames = next(iter(humans.values())).base_verts.shape[0]
    device = next(iter(objects.values())).x0.device
    T_mats = _final_T_mats(objects, num_frames, device)
    scales = _final_scales(objects, float(args.max_log_scale_delta))

    print("\nSaving module-12 outputs...")
    object_summaries: dict[str, Any] = {}
    for slug, obj in objects.items():
        obj_dir = out_dir / slug
        ensure_dir(obj_dir)
        _save_pose_json(obj_dir / "poses.json", T_mats[slug])
        _save_transform_refined(obj_dir / "transform_refined.json", T_mats[slug], scales[slug])
        _save_object_meshes(obj_dir / "meshes", obj, T_mats[slug], scales[slug])
        delta_stats = _build_delta_stats(obj, scales[slug])
        _save_json(obj_dir / "delta_stats.json", delta_stats)
        object_summaries[slug] = {
            "name": obj.name,
            "num_verts": int(obj.verts.shape[0]),
            "num_faces": int(obj.faces.shape[0]),
            "final_scale": float(scales[slug]),
            "delta_stats": delta_stats,
        }
        print(f"  {slug}: scale={scales[slug]:.4f}, meshes/ poses.json")

    human_summaries: dict[str, Any] = {}
    for slug in human_keys:
        human_dir = out_dir / slug
        _save_human_meshes(human_dir / "meshes", humans[slug])
        stats = {
            "status": "fixed_smplx_sequence",
            "name": humans[slug].name,
            "slug": slug,
            "num_frames": int(humans[slug].base_verts.shape[0]),
            "num_verts": int(humans[slug].base_verts.shape[1]),
            "num_faces": int(humans[slug].faces.shape[0]),
            "source_metadata": human_metadata[slug],
        }
        _save_json(human_dir / "fixed_input_stats.json", stats)
        human_summaries[slug] = stats

    debug_csv_dir = out_dir / "debug" / "csv"
    debug_plot_dir = out_dir / "debug" / "plots"
    _save_csv(debug_csv_dir / "stage1_iter_metrics.csv", stage1_rows)
    _save_csv(debug_csv_dir / "stage2_iter_metrics.csv", stage2_rows)
    frame_rows = _frame_metrics(objects, args, k, num_frames)
    _save_csv(debug_csv_dir / "frame_loss_metrics.csv", frame_rows)

    if context is not None:
        full_rot, full_trans, raw_scales = _full_delta_vars(objects)
        with torch.no_grad():
            diagnostic = compute_final_loss_diagnostics(
                full_rot,
                full_trans,
                raw_scales,
                context.objects,
                context.humans,
                context.interaction_edges,
                context.obj_keys,
                _stage2_args(args),
                iteration=max(int(args.stage2_iters), 1),
                total_iters=max(int(args.stage2_iters), 1),
                k=context.k_torch,
                width=context.width,
                height=context.height,
            )
        final_contact = _diagnostic_summary(diagnostic)
        _save_json(out_dir / "debug" / "final_contact_diagnostic.json", final_contact)
    else:
        final_contact = None

    _plot_loss_csv(
        debug_csv_dir / "stage1_iter_metrics.csv",
        debug_plot_dir / "stage1_loss.png",
        "Stage 1 Loss",
    )
    _plot_loss_csv(
        debug_csv_dir / "stage2_iter_metrics.csv",
        debug_plot_dir / "stage2_loss.png",
        "Stage 2 Loss",
    )

    frames_dir = track_utils._resolve_frames_dir(dirs["object_tracks"], dirs["seg_vid"])
    frame_paths = list_images(frames_dir) if frames_dir and frames_dir.exists() else []
    if frame_paths:
        print("  Rendering joint overlay...")
        _render_overlay(
            out_dir,
            frame_paths,
            humans,
            human_keys,
            objects,
            T_mats,
            scales,
            k,
            float(args.overlay_fps),
            bool(args.overlay_save_pngs),
            num_frames,
        )

    summary = {
        "interaction_name": args.interaction_name,
        "status": "completed",
        "script": "01_track_human_object_joint.py",
        "num_frames": int(num_frames),
        "num_objects": len(objects),
        "num_humans": len(human_keys),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "inputs": {
            "object_point_tracks_dir": str(dirs["object_tracks"]),
            "aligned_mesh_dir": str(dirs["aligned"]),
            "human_motion_dir": str(dirs["human_motion"]),
            "segment_video_dir": str(dirs["seg_vid"]),
            "segment_object_dir": str(dirs["seg_obj"]),
            "pag_file": str(pag_path),
            "smpl_seg_json": str(smpl_seg_path),
            "intrinsics_source": str(intr_path),
        },
        "settings": {
            "stage1_iters": int(args.stage1_iters),
            "stage1_lr": float(args.stage1_lr),
            "stage2_iters": int(args.stage2_iters),
            "stage2_lr": float(args.stage2_lr),
            "huber_delta_px": float(args.huber_delta_px),
            "lambda_track_reproj": float(args.lambda_track_reproj),
            "lambda_smooth_trans": float(args.lambda_smooth_trans),
            "lambda_smooth_rot": float(args.lambda_smooth_rot),
            "lambda_scale": float(args.lambda_scale),
            "max_log_scale_delta": float(args.max_log_scale_delta),
            "retrim_interval": int(args.retrim_interval),
            "retrim_percentile": float(args.retrim_percentile),
            "stage2_contact_weights": {
                "object_cd2d_weight": float(args.object_cd2d_weight),
                "object_part_cd2d_weight": float(args.object_part_cd2d_weight),
                "object_smooth_trans_weight": float(args.object_smooth_trans_weight),
                "object_smooth_rot_weight": float(args.object_smooth_rot_weight),
                "object_scale_weight": float(args.object_scale_weight),
                "intersect_weight_start": float(args.intersect_weight_start),
                "intersect_weight_end": float(args.intersect_weight_end),
                "nocontact_weight": float(args.nocontact_weight),
                "contact_drift_weight_start": float(args.contact_drift_weight_start),
                "contact_drift_weight_end": float(args.contact_drift_weight_end),
            },
        },
        "objects": object_summaries,
        "humans": human_summaries,
        "final_contact_diagnostic": final_contact,
        "conventions": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "object_T_4x4": "standard column-vector [[R,t],[0,1]], p'=R@p+t",
            "object_scale": "bounded global scale applied before object T_4x4",
            "frame0_object_pose": "identity relative to module-09 aligned mesh",
            "human": "frozen GVHMR SMPL-X sequence transformed by module-09 similarity",
        },
    }
    _save_json(out_dir / "run_summary.json", summary)
    print(f"Done. Output: {out_dir}")


def _diagnostic_summary(diagnostic: DiagnosticLossResult) -> dict[str, Any]:
    sequence = diagnostic.sequence
    scaled = get_scaled_loss_terms(sequence)
    out: dict[str, Any] = {
        "total": float(sequence.total.item()),
        "terms": {},
    }
    for key, value in scaled.items():
        out["terms"][key] = {
            "weight": float(sequence.weights[key]),
            "raw": float(getattr(sequence, key).item()),
            "scaled": float(value.item()),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified human-object trajectory optimization.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--object_point_tracks_dir", default=None)
    parser.add_argument("--aligned_mesh_dir", default=None)
    parser.add_argument("--human_motion_dir", default=None)
    parser.add_argument("--segment_video_dir", default=None)
    parser.add_argument("--segment_object_dir", default=None)
    parser.add_argument("--pag_file", default=None)
    parser.add_argument("--smpl_seg_json", default=None)
    parser.add_argument(
        "--smpl_folder",
        default="../../GVHMR/inputs/checkpoints/body_models/",
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--stage1_iters", type=int, default=4000)
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument(
        "--stage1_lr_schedule",
        choices=["none", "cosine"],
        default="cosine",
    )
    parser.add_argument("--stage2_iters", type=int, default=1000)
    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)

    parser.add_argument("--huber_delta_px", type=float, default=3.0)
    parser.add_argument("--lambda_track_reproj", type=float, default=1.0)
    parser.add_argument("--lambda_smooth_trans", type=float, default=300.0)
    parser.add_argument("--lambda_smooth_rot", type=float, default=80.0)
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--max_log_scale_delta", type=float, default=0.22)

    parser.add_argument("--object_cd2d_weight", type=float, default=1e-4)
    parser.add_argument("--object_part_cd2d_weight", type=float, default=1e-4)
    parser.add_argument("--object_smooth_trans_weight", type=float, default=1e3)
    parser.add_argument("--object_smooth_rot_weight", type=float, default=1e3)
    parser.add_argument("--object_scale_weight", type=float, default=1.0)
    parser.add_argument("--intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--intersect_weight_end", type=float, default=10.0)
    parser.add_argument("--nocontact_weight", type=float, default=1e3)
    parser.add_argument("--contact_drift_weight_start", type=float, default=10.0)
    parser.add_argument("--contact_drift_weight_end", type=float, default=1e3)

    parser.add_argument("--bin_size", type=int, default=0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--mask_gate_threshold", type=float, default=0.5)
    parser.add_argument("--visibility_threshold", type=float, default=0.0)
    parser.add_argument("--min_valid_tracks", type=int, default=50)
    parser.add_argument("--disable_pnp_init", action="store_true")
    parser.add_argument("--pnp_ransac_thresh", type=float, default=8.0)
    parser.add_argument("--outlier_reproj_thresh_px", type=float, default=20.0)
    parser.add_argument("--outlier_max_fraction", type=float, default=0.4)
    parser.add_argument(
        "--graduated_huber",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retrim_interval", type=int, default=1000)
    parser.add_argument("--retrim_percentile", type=float, default=90.0)

    parser.add_argument("--num_object_surface_points", type=int, default=3000)
    parser.add_argument("--num_part_surface_points", type=int, default=1000)
    parser.add_argument("--num_mask_points_2d", type=int, default=2500)
    parser.add_argument("--sdf_resolution", type=int, default=48)
    parser.add_argument("--smplx_batch_size", type=int, default=32)

    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--debug_save_interval", type=int, default=20)
    parser.add_argument("--overlay_fps", type=float, default=6.0)
    parser.add_argument(
        "--overlay_save_pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    dirs = _resolve_dirs(args)
    pag_path = _resolve_pag_path(args)
    smpl_seg_path = _resolve_smpl_seg_path(args)
    device = _to_device(args.device)
    ensure_dir(dirs["output"])

    for label, path in (
        ("Object point tracks", dirs["object_tracks"]),
        ("Aligned meshes", dirs["aligned"]),
        ("Human motion", dirs["human_motion"]),
        ("Video segmentation", dirs["seg_vid"]),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} dir not found: {path}")

    k, intr_path = track_utils._load_intrinsics_from_alignment_summary(dirs["aligned"])
    transforms_by_slug = _load_alignment_transforms(dirs["aligned"])
    body_seg, contact_seg = _load_smpl_body_and_contact_seg(smpl_seg_path)
    pag = _parse_pag(pag_path)
    if contact_seg:
        print(
            "Loaded SMPL-X contact subsets from "
            f"{smpl_seg_path.name}: {', '.join(sorted(contact_seg))}"
        )

    print("=" * 68)
    print("Module 12: Unified Human-Object Trajectory Optimization")
    print(f"  interaction: {args.interaction_name}")
    print(f"  device:      {device}")
    print(f"  K:           fx={k[0,0]:.1f} fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}")
    print(f"  PAG:         {pag_path.name}")
    print("=" * 68)

    humans, human_keys, human_metadata, human_num_frames = _load_frozen_humans(
        args,
        dirs,
        transforms_by_slug,
        body_seg,
        contact_seg,
        device,
    )
    num_frames = _discover_num_frames(args, dirs, pag, human_num_frames)
    if num_frames < human_num_frames:
        for human in humans.values():
            human.base_verts = human.base_verts[:num_frames]
            for key in list(human.part_points):
                human.part_points[key] = human.part_points[key][:num_frames]
            for key in list(human.contact_part_points):
                human.contact_part_points[key] = human.contact_part_points[key][:num_frames]
        for metadata in human_metadata.values():
            metadata["num_frames"] = int(num_frames)
    print(f"Using {num_frames} frames")

    objects: dict[str, ObjectTrackState] = {}
    print("\nLoading object tracks and frame-0 surface anchors...")
    for state in pag.object_states:
        obj = _load_track_object_state(args, dirs, state, k, num_frames, device)
        if obj is not None:
            objects[obj.slug] = obj
            print(
                f"  {obj.slug}: {obj.num_valid_seed_tracks}/{obj.num_input_tracks} "
                "valid seed tracks"
            )
    if not objects:
        raise RuntimeError("No objects loaded for module-12 optimization.")

    stage1_rows = _run_stage1(objects, args, k)

    context: ProblemContext | None = None
    stage2_rows: list[dict[str, Any]] = []
    if int(args.stage2_iters) > 0:
        context = _build_contact_context(
            args=args,
            dirs=dirs,
            pag_path=pag_path,
            smpl_seg_path=smpl_seg_path,
            k=k,
            humans=humans,
            human_keys=human_keys,
            objects=objects,
            num_frames=num_frames,
            device=device,
        )
        stage2_rows = _run_stage2(objects, context, args)
    else:
        print("\n[Stage 2] disabled by --stage2_iters 0")

    _save_outputs(
        args=args,
        dirs=dirs,
        pag_path=pag_path,
        smpl_seg_path=smpl_seg_path,
        k=k,
        intr_path=intr_path,
        objects=objects,
        humans=humans,
        human_keys=human_keys,
        human_metadata=human_metadata,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        context=context,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
