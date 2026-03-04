"""Multi-frame SE(3) object mesh tracking from CoTracker tracks.

This script estimates a smooth rigid pose trajectory for each PAG-listed object
using:
- aligned frame-0 mesh in camera coordinates (OpenCV convention),
- CoTracker tracks + visibility,
- per-frame segmentation masks.

The optimization is end-to-end (no post-smoothing):
    E = E_img + lambda_a * E_smooth + lambda_v * E_vel

Where E_img is a robust (Huber) reprojection term and the temporal terms are
computed in SE(3) via Lie log maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.transforms import se3_exp_map, se3_log_map

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False

from tracking_utils import (
    close_ffmpeg,
    draw_overlay,
    ensure_dir,
    list_images,
    resolve_path,
    start_ffmpeg_writer,
)


F_CV_TO_P3D = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
F_P3D_TO_CV = F_CV_TO_P3D.copy()


@dataclass
class SeedMappingResult:
    points_cv: torch.Tensor  # [M, 3]
    valid_seed_mask: np.ndarray  # [N] bool, keeps track ids that survive mapping
    invalid_face_count: int
    outside_mask0_count: int
    nonfinite_seed_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track aligned object meshes with a multi-frame SE(3) optimizer from "
            "CoTracker tracks."
        )
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video identifier, e.g. video_01.",
    )

    parser.add_argument(
        "--cotracker_video_dir",
        type=str,
        default=None,
        help="Override CoTracker output dir. Default: ../Estimate_Optical_Flow/output_cotracker/<video_name>",
    )
    parser.add_argument(
        "--aligned_mesh_video_dir",
        type=str,
        default=None,
        help="Override aligned mesh dir. Default: ../Align_Meshes/output/<video_name>",
    )
    parser.add_argument(
        "--segment_video_dir",
        type=str,
        default=None,
        help="Override Segment_Video dir. Default: ../Segment_Video/output/<video_name>",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="Override PAG file. Default: first output_pag_*.json in ../Generate_PAG/output/<video_name>/",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_cotracker_se3",
        help="Root directory for tracking outputs.",
    )

    parser.add_argument(
        "--output_coord",
        type=str,
        choices=["opencv", "pytorch3d"],
        default="opencv",
        help="Coordinate convention for saved .ply sequence.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bin_size", type=int, default=0, help="Rasterizer bin size. 0 means naive.")
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--mask_gate_threshold", type=float, default=0.5)
    parser.add_argument("--visibility_threshold", type=float, default=0.0)
    parser.add_argument("--min_valid_tracks", type=int, default=50)

    parser.add_argument("--huber_delta_px", type=float, default=3.0)
    parser.add_argument("--lambda_a", type=float, default=10.0)
    parser.add_argument("--lambda_v", type=float, default=0.1)
    parser.add_argument("--adam_iters", type=int, default=2000)
    parser.add_argument("--adam_lr", type=float, default=1e-2)
    parser.add_argument("--disable_lbfgs", action="store_true")
    parser.add_argument("--lbfgs_iters", type=int, default=120)
    parser.add_argument("--lbfgs_lr", type=float, default=0.5)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--overlay_fps", type=float, default=6.0)
    parser.add_argument("--overlay_max_verts", type=int, default=20000)
    parser.add_argument("--overlay_point_radius", type=int, default=1)
    parser.add_argument("--debug_save_interval", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _extract_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def _sanitize_object_name(name: str) -> str:
    # Segment_Video uses this exact convention for folder names.
    return name.strip().replace(" ", "_")


def _resolve_default_dirs(args: argparse.Namespace, script_dir: Path) -> tuple[Path, Path, Path, Path]:
    video_name = args.video_name

    if args.cotracker_video_dir is None:
        cotracker_video_dir = (
            script_dir.parent / "Estimate_Optical_Flow" / "output_cotracker" / video_name
        ).resolve()
    else:
        cotracker_video_dir = resolve_path(args.cotracker_video_dir, script_dir)

    if args.aligned_mesh_video_dir is None:
        aligned_mesh_video_dir = (script_dir.parent / "Align_Meshes" / "output" / video_name).resolve()
    else:
        aligned_mesh_video_dir = resolve_path(args.aligned_mesh_video_dir, script_dir)

    if args.segment_video_dir is None:
        segment_video_dir = (script_dir.parent / "Segment_Video" / "output" / video_name).resolve()
    else:
        segment_video_dir = resolve_path(args.segment_video_dir, script_dir)

    output_root = resolve_path(args.output_root, script_dir)
    return cotracker_video_dir, aligned_mesh_video_dir, segment_video_dir, output_root


def _resolve_pag_path(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.pag_file is not None:
        pag_path = resolve_path(args.pag_file, script_dir)
        if not pag_path.exists():
            raise FileNotFoundError(f"PAG file not found: {pag_path}")
        return pag_path

    pag_dir = (script_dir.parent / "Generate_PAG" / "output" / args.video_name).resolve()
    if not pag_dir.exists():
        raise FileNotFoundError(f"PAG directory not found: {pag_dir}")
    pag_candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not pag_candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return pag_candidates[0]


def _load_intrinsics_from_alignment_summary(aligned_mesh_video_dir: Path) -> tuple[np.ndarray, Path]:
    summary_path = (aligned_mesh_video_dir / "alignment_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"alignment_summary.json not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise KeyError(
            f"Missing 'camera' dictionary in alignment summary: {summary_path}"
        )

    k_raw = camera.get("intrinsics_3x3")
    if k_raw is None:
        raise KeyError(
            f"Missing 'camera.intrinsics_3x3' in alignment summary: {summary_path}"
        )

    k = np.array(k_raw, dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(
            "Expected camera.intrinsics_3x3 to resolve to shape (3, 3), "
            f"got {k.shape} in {summary_path}"
        )
    return k.astype(np.float32), summary_path


def _load_pag_objects_from_states_only(pag_path: Path) -> list[tuple[str, str]]:
    with pag_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    object_states = payload.get("object states")
    if not isinstance(object_states, list):
        raise RuntimeError(
            "PAG must contain a list in 'object states'. "
            f"Got: {type(object_states).__name__}"
        )

    objects: list[tuple[str, str]] = []
    seen = set()
    for item in object_states:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        objects.append((name, _sanitize_object_name(name)))

    if not objects:
        raise RuntimeError(
            "No valid object names found in PAG 'object states'. "
            f"File: {pag_path}"
        )
    return objects


def _resolve_frames_dir(cotracker_video_dir: Path, segment_video_dir: Path) -> Path | None:
    candidates = [
        cotracker_video_dir / "_frames",
        segment_video_dir / "_frames",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return None


def _resolve_object_mask_dir(segment_video_dir: Path, object_slug: str) -> Path:
    return (
        segment_video_dir
        / "objects"
        / object_slug
        / "object_segmentation"
        / "masks"
    ).resolve()


def _list_mask_files(mask_dir: Path) -> list[Path]:
    mask_paths = sorted(mask_dir.glob("frame_*.png"), key=_extract_index)
    if not mask_paths:
        mask_paths = sorted(mask_dir.glob("*.png"), key=_extract_index)
    return mask_paths


def _load_mask_stack(mask_paths: list[Path], mask_threshold: int) -> tuple[np.ndarray, int, int]:
    if not mask_paths:
        raise RuntimeError("mask_paths is empty")
    masks = []
    h_ref, w_ref = -1, -1
    for idx, path in enumerate(mask_paths):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        if idx == 0:
            h_ref, w_ref = mask.shape[:2]
        if mask.shape[:2] != (h_ref, w_ref):
            mask = cv2.resize(mask, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > int(mask_threshold)).astype(np.float32))
    return np.stack(masks, axis=0).astype(np.float32), h_ref, w_ref


def _normalize_tracks_vis_with_mask_length(
    tracks_raw: np.ndarray,
    vis_raw: np.ndarray,
    expected_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize to tracks [N, T, 2], vis [N, T] based on expected_t from masks."""
    tracks = np.asarray(tracks_raw)
    vis = np.asarray(vis_raw)

    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError(f"Expected tracks shape [*, *, 2], got {tracks.shape}")

    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]
    if vis.ndim != 2:
        raise ValueError(f"Expected visibility shape [*, *], got {vis.shape}")

    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    # Candidate A: tracks are [T, N, 2].
    if tracks.shape[0] == expected_t:
        t, n = tracks.shape[0], tracks.shape[1]
        if vis.shape == (t, n):
            candidates.append((tracks.transpose(1, 0, 2), vis.transpose(1, 0)))
        elif vis.shape == (n, t):
            candidates.append((tracks.transpose(1, 0, 2), vis))

    # Candidate B: tracks are [N, T, 2].
    if tracks.shape[1] == expected_t:
        n, t = tracks.shape[0], tracks.shape[1]
        if vis.shape == (n, t):
            candidates.append((tracks, vis))
        elif vis.shape == (t, n):
            candidates.append((tracks, vis.transpose(1, 0)))

    # Fallback independent of expected_t for slightly inconsistent datasets.
    if not candidates:
        if vis.shape == tracks.shape[:2]:
            # If dim0 is smaller, treat dim0 as time ([T,N,2]) and transpose.
            if tracks.shape[0] < tracks.shape[1]:
                candidates.append((tracks.transpose(1, 0, 2), vis.transpose(1, 0)))
            else:
                candidates.append((tracks, vis))
        elif vis.shape == (tracks.shape[1], tracks.shape[0]):
            # If dim0 is smaller, tracks are likely [T,N,2] and vis [N,T].
            if tracks.shape[0] < tracks.shape[1]:
                candidates.append((tracks.transpose(1, 0, 2), vis))
            else:
                candidates.append((tracks, vis.transpose(1, 0)))

    if not candidates:
        raise ValueError(
            "Could not infer tracks/visibility orientation. "
            f"tracks={tracks.shape}, vis={vis.shape}, expected_t={expected_t}"
        )

    tracks_nt2, vis_nt = candidates[0]
    tracks_nt2 = tracks_nt2.astype(np.float32)
    vis_nt = vis_nt.astype(np.float32)
    return tracks_nt2, vis_nt


def _to_device(args_device: str) -> torch.device:
    try:
        dev = torch.device(args_device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {args_device}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {args_device}, but only {torch.cuda.device_count()} CUDA device(s) available."
        )
    return dev


def _build_rasterizer(
    device: torch.device,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    h: int,
    w: int,
    bin_size: int,
) -> MeshRasterizer:
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[h, w]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),
        max_faces_per_bin=300000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def _cv_to_p3d_torch(points_cv: torch.Tensor) -> torch.Tensor:
    out = points_cv.clone()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _cv_to_p3d_np(points_cv: np.ndarray) -> np.ndarray:
    out = points_cv.copy().astype(np.float32)
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _sample_mask_bilinear_single(mask_hw: torch.Tensor, uv_n2: torch.Tensor) -> torch.Tensor:
    """Sample one mask with sub-pixel bilinear interpolation.

    Returns:
        values [N], in [0, 1].
    """
    if uv_n2.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32, device=mask_hw.device)

    h, w = int(mask_hw.shape[0]), int(mask_hw.shape[1])
    mask_nchw = mask_hw.view(1, 1, h, w)
    x_norm = (2.0 * uv_n2[:, 0] / max(float(w - 1), 1.0)) - 1.0
    y_norm = (2.0 * uv_n2[:, 1] / max(float(h - 1), 1.0)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def _sample_masks_bilinear_sequence(masks_thw: torch.Tensor, uv_tm2: torch.Tensor) -> torch.Tensor:
    """Sample masks for all frames.

    Args:
        masks_thw: [T, H, W]
        uv_tm2: [T, M, 2] in pixel coordinates
    Returns:
        sampled values [T, M]
    """
    if uv_tm2.numel() == 0:
        t = int(masks_thw.shape[0])
        return torch.zeros((t, 0), dtype=torch.float32, device=masks_thw.device)

    t, h, w = masks_thw.shape
    if uv_tm2.shape[0] != t:
        raise ValueError(f"Mask frame count {t} != uv frame count {uv_tm2.shape[0]}")

    masks_tchw = masks_thw.unsqueeze(1)  # [T,1,H,W]
    x_norm = (2.0 * uv_tm2[..., 0] / max(float(w - 1), 1.0)) - 1.0
    y_norm = (2.0 * uv_tm2[..., 1] / max(float(h - 1), 1.0)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(2)  # [T,M,1,2]
    sampled = F.grid_sample(
        masks_tchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)  # [T,M]


def _map_seed_points_to_mesh(
    seed_uv_n2: np.ndarray,
    verts_cv_np: np.ndarray,
    faces_np: np.ndarray,
    mask0_hw_np: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    h: int,
    w: int,
    device: torch.device,
    bin_size: int,
    mask_gate_threshold: float,
) -> SeedMappingResult:
    rasterizer = _build_rasterizer(
        device=device, fx=fx, fy=fy, cx=cx, cy=cy, h=h, w=w, bin_size=bin_size
    )

    verts_cv = torch.from_numpy(verts_cv_np).to(device=device, dtype=torch.float32)
    verts_p3d = _cv_to_p3d_torch(verts_cv)
    faces = torch.from_numpy(faces_np.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces])
    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]  # [H,W]
    bary = fragments.bary_coords[0, ..., 0, :]  # [H,W,3]

    seed_uv = torch.from_numpy(seed_uv_n2).to(device=device, dtype=torch.float32)
    finite_seed = torch.isfinite(seed_uv).all(dim=1)

    x_idx = torch.clamp(torch.round(seed_uv[:, 0]).long(), 0, w - 1)
    y_idx = torch.clamp(torch.round(seed_uv[:, 1]).long(), 0, h - 1)
    face_id = pix_to_face[y_idx, x_idx]  # [N]
    bary_seed = bary[y_idx, x_idx, :]  # [N,3]

    mask0 = torch.from_numpy(mask0_hw_np.astype(np.float32)).to(device=device)
    mask_vals = _sample_mask_bilinear_single(mask0, seed_uv)

    valid = finite_seed & (face_id >= 0) & (mask_vals >= float(mask_gate_threshold))

    points_cv_all = torch.zeros((seed_uv.shape[0], 3), dtype=torch.float32, device=device)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if idx.numel() > 0:
        tri = faces[face_id[idx].long()]  # [M,3]
        tri_verts = verts_cv[tri]  # [M,3,3]
        b = bary_seed[idx].unsqueeze(-1)  # [M,3,1]
        points_cv_all[idx] = (b * tri_verts).sum(dim=1)

    invalid_face_count = int(((face_id < 0) & finite_seed).sum().item())
    outside_mask0_count = int(
        ((mask_vals < float(mask_gate_threshold)) & (face_id >= 0) & finite_seed).sum().item()
    )
    nonfinite_seed_count = int((~finite_seed).sum().item())

    return SeedMappingResult(
        points_cv=points_cv_all[idx],
        valid_seed_mask=valid.detach().cpu().numpy().astype(bool),
        invalid_face_count=invalid_face_count,
        outside_mask0_count=outside_mask0_count,
        nonfinite_seed_count=nonfinite_seed_count,
    )


def _build_T_mats_from_xi(xi_tm1_6: torch.Tensor, t_frames: int, device: torch.device) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    if t_frames <= 1:
        return eye
    mats = se3_exp_map(xi_tm1_6)  # [T-1,4,4]
    return torch.cat([eye, mats], dim=0)


def _huber_on_squared(s: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber-like robustifier operating on squared residual norm."""
    d2 = float(delta) * float(delta)
    sqrt_s = torch.sqrt(torch.clamp(s, min=1e-12))
    return torch.where(s <= d2, s, 2.0 * float(delta) * sqrt_s - d2)


@dataclass
class LossBundle:
    total: torch.Tensor
    e_img: torch.Tensor
    e_smooth: torch.Tensor
    e_vel: torch.Tensor
    T_mats: torch.Tensor
    pred_uv_tm2: torch.Tensor
    obs_uv_tm2: torch.Tensor
    weights_tm: torch.Tensor
    r2_tm: torch.Tensor
    mask_values_tm: torch.Tensor
    vis_weights_tm: torch.Tensor


def _compute_loss_bundle(
    xi_tm1_6: torch.Tensor,
    x0_m3: torch.Tensor,
    obs_uv_tm2: torch.Tensor,
    vis_tm: torch.Tensor,
    masks_thw: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    lambda_a: float,
    lambda_v: float,
    huber_delta_px: float,
    visibility_threshold: float,
    mask_gate_threshold: float,
) -> LossBundle:
    device = x0_m3.device
    t_frames = int(obs_uv_tm2.shape[0])
    t_mats = _build_T_mats_from_xi(xi_tm1_6, t_frames=t_frames, device=device)

    ones = torch.ones((x0_m3.shape[0], 1), dtype=torch.float32, device=device)
    x0_h = torch.cat([x0_m3, ones], dim=1)  # [M,4]

    # X_t = T_t @ X_0 in homogeneous coordinates.
    xt_h_tm4 = torch.einsum("tij,mj->tmi", t_mats, x0_h)
    xt_tm3 = xt_h_tm4[..., :3]

    z = xt_tm3[..., 2]
    z_valid = z > 1e-6
    z_safe = torch.where(z_valid, z, torch.ones_like(z))
    pred_u = fx * (xt_tm3[..., 0] / z_safe) + cx
    pred_v = fy * (xt_tm3[..., 1] / z_safe) + cy
    pred_uv = torch.stack([pred_u, pred_v], dim=-1)

    finite_obs = torch.isfinite(obs_uv_tm2).all(dim=-1)
    mask_values = _sample_masks_bilinear_sequence(masks_thw, obs_uv_tm2)
    mask_gate = mask_values >= float(mask_gate_threshold)

    if visibility_threshold > 0.0:
        vis_weight = torch.where(vis_tm >= float(visibility_threshold), vis_tm, torch.zeros_like(vis_tm))
    else:
        vis_weight = vis_tm

    weights = (
        vis_weight
        * mask_gate.float()
        * finite_obs.float()
        * z_valid.float()
    )

    residual = obs_uv_tm2 - pred_uv
    r2 = (residual ** 2).sum(dim=-1)
    robust = _huber_on_squared(r2, delta=huber_delta_px)
    e_img = (weights * robust).sum()

    if t_frames > 1:
        t_rel = torch.matmul(torch.linalg.inv(t_mats[:-1]), t_mats[1:])
        delta = se3_log_map(t_rel)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        e_vel = (delta ** 2).sum()
        # Follow requested index range t=2..T-2 (1-indexed).
        if delta.shape[0] >= 3:
            e_smooth = ((delta[1:-1] - delta[:-2]) ** 2).sum()
        else:
            e_smooth = torch.zeros((), dtype=torch.float32, device=device)
    else:
        e_vel = torch.zeros((), dtype=torch.float32, device=device)
        e_smooth = torch.zeros((), dtype=torch.float32, device=device)

    total = e_img + float(lambda_a) * e_smooth + float(lambda_v) * e_vel
    return LossBundle(
        total=total,
        e_img=e_img,
        e_smooth=e_smooth,
        e_vel=e_vel,
        T_mats=t_mats,
        pred_uv_tm2=pred_uv,
        obs_uv_tm2=obs_uv_tm2,
        weights_tm=weights,
        r2_tm=r2,
        mask_values_tm=mask_values,
        vis_weights_tm=vis_weight,
    )


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_loss_plot(path: Path, iter_rows: list[dict[str, Any]]) -> None:
    if not _HAS_MPL or not iter_rows:
        return
    it = np.array([int(r["iter"]) for r in iter_rows], dtype=np.int32)
    total = np.array([float(r["total"]) for r in iter_rows], dtype=np.float32)
    e_img = np.array([float(r["e_img"]) for r in iter_rows], dtype=np.float32)
    e_smooth = np.array([float(r["e_smooth"]) for r in iter_rows], dtype=np.float32)
    e_vel = np.array([float(r["e_vel"]) for r in iter_rows], dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(it, total, label="E_total", linewidth=2.0)
    ax.plot(it, e_img, label="E_img")
    ax.plot(it, e_smooth, label="E_smooth")
    ax.plot(it, e_vel, label="E_vel")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Optimization Loss Terms")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def _draw_frame0_correspondence(
    frame_bgr: np.ndarray,
    obs_uv_m2: np.ndarray,
    pred_uv_m2: np.ndarray,
    max_points: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    n = int(obs_uv_m2.shape[0])
    if n <= 0:
        return out
    max_points = max(1, int(max_points))
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int32)
    else:
        idx = np.arange(n, dtype=np.int32)

    for i in idx:
        ox = int(np.clip(np.round(obs_uv_m2[i, 0]), 0, w - 1))
        oy = int(np.clip(np.round(obs_uv_m2[i, 1]), 0, h - 1))
        px = int(np.clip(np.round(pred_uv_m2[i, 0]), 0, w - 1))
        py = int(np.clip(np.round(pred_uv_m2[i, 1]), 0, h - 1))
        cv2.circle(out, (ox, oy), 2, (0, 255, 0), -1)
        cv2.circle(out, (px, py), 2, (0, 165, 255), -1)
        cv2.arrowedLine(out, (px, py), (ox, oy), (0, 255, 255), 1, tipLength=0.25)
    return out


def _transform_points_cv_np(points_cv: np.ndarray, t_44: np.ndarray) -> np.ndarray:
    r = t_44[:3, :3].astype(np.float32)
    t = t_44[:3, 3].astype(np.float32)
    return points_cv.astype(np.float32) @ r.T + t[None, :]


def _save_pose_json(
    out_path: Path,
    t_mats_np: np.ndarray,
    frame_offset: int,
) -> None:
    rows = []
    for i in range(int(t_mats_np.shape[0])):
        rows.append(
            {
                "frame": int(frame_offset + i),
                "T_4x4": t_mats_np[i].astype(np.float32).tolist(),
            }
        )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _save_mesh_sequences(
    mesh_template: trimesh.Trimesh,
    verts0_cv: np.ndarray,
    t_mats_np: np.ndarray,
    meshes_dir: Path,
    output_coord: str,
    frame_offset: int,
) -> None:
    ensure_dir(meshes_dir)
    for i in range(t_mats_np.shape[0]):
        frame_idx = frame_offset + i
        verts_t_cv = _transform_points_cv_np(verts0_cv, t_mats_np[i])
        if output_coord == "opencv":
            verts_save = verts_t_cv
        else:
            verts_save = _cv_to_p3d_np(verts_t_cv)

        mesh = mesh_template.copy()
        mesh.vertices = verts_save.astype(np.float32)
        mesh.export(str(meshes_dir / f"frame_{frame_idx:04d}.ply"))


def _render_overlays(
    frame_paths: list[Path],
    verts0_cv: np.ndarray,
    t_mats_np: np.ndarray,
    k: np.ndarray,
    out_dir: Path,
    frame_offset: int,
    fps: float,
    max_verts: int,
    point_radius: int,
) -> tuple[bool, str]:
    overlays_dir = out_dir / "overlays"
    ensure_dir(overlays_dir)

    if not frame_paths:
        return False, "No frame images available; skipped overlay rendering."
    if len(frame_paths) <= frame_offset:
        return False, "Frame list shorter than frame_offset; skipped overlays."

    end = min(len(frame_paths), frame_offset + t_mats_np.shape[0])
    if end <= frame_offset:
        return False, "No frames in selected range; skipped overlays."

    first = cv2.imread(str(frame_paths[frame_offset]))
    if first is None:
        return False, f"Failed to read first frame for overlays: {frame_paths[frame_offset]}"
    h, w = first.shape[:2]

    overlay_writer = start_ffmpeg_writer(out_dir / "overlay.mp4", float(fps), (h, w))

    try:
        local_idx = 0
        for frame_idx in range(frame_offset, end):
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                continue
            verts_t_cv = _transform_points_cv_np(verts0_cv, t_mats_np[local_idx])
            verts_t_p3d = _cv_to_p3d_np(verts_t_cv)
            overlay = draw_overlay(
                frame_bgr=frame,
                verts_cam_p3d=verts_t_p3d,
                k=k,
                max_verts=int(max_verts),
                radius=int(point_radius),
            )

            out_png = overlays_dir / f"overlay_{frame_idx:04d}.png"
            cv2.imwrite(str(out_png), overlay)

            if overlay_writer.stdin is not None:
                overlay_writer.stdin.write(np.ascontiguousarray(overlay.astype(np.uint8)).tobytes())
            local_idx += 1
    finally:
        close_ffmpeg(overlay_writer)

    return True, "Rendered overlays."


def _build_frame_metrics(
    frame_offset: int,
    bundle: LossBundle,
    mask_gate_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    t_frames = int(bundle.r2_tm.shape[0])
    for t in range(t_frames):
        w_t = bundle.weights_tm[t]
        active = w_t > 0.0
        num_active = int(active.sum().item())
        if num_active > 0:
            reproj = torch.sqrt(torch.clamp(bundle.r2_tm[t][active], min=1e-12))
            reproj_mean = float(reproj.mean().item())
            reproj_p50 = float(torch.quantile(reproj, q=0.50).item())
            reproj_p90 = float(torch.quantile(reproj, q=0.90).item())
        else:
            reproj_mean = float("nan")
            reproj_p50 = float("nan")
            reproj_p90 = float("nan")
        rows.append(
            {
                "frame_idx": int(frame_offset + t),
                "num_active": num_active,
                "sum_weight": float(w_t.sum().item()),
                "visibility_mean": float(bundle.vis_weights_tm[t].mean().item()),
                "mask_gate_mean": float(
                    (bundle.mask_values_tm[t] >= float(mask_gate_threshold)).float().mean().item()
                ),
                "reproj_mean_px": reproj_mean,
                "reproj_p50_px": reproj_p50,
                "reproj_p90_px": reproj_p90,
            }
        )
    return rows


def _run_single_object(
    object_name: str,
    object_slug: str,
    mesh_path: Path,
    tracks_path: Path,
    vis_path: Path,
    mask_dir: Path,
    frame_paths: list[Path],
    k: np.ndarray,
    args: argparse.Namespace,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    start_time = time.time()

    ensure_dir(out_dir)
    debug_dir = out_dir / "debug"
    meshes_dir = out_dir / "meshes"
    ensure_dir(debug_dir)
    ensure_dir(meshes_dir)

    mesh_template = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh_template, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as trimesh.Trimesh: {mesh_path}")
    if mesh_template.faces is None or len(mesh_template.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    verts_cv_np = np.asarray(mesh_template.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_template.faces, dtype=np.int64)

    mask_paths = _list_mask_files(mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No mask frames found in: {mask_dir}")
    masks_thw_np, h_mask, w_mask = _load_mask_stack(mask_paths, mask_threshold=int(args.mask_threshold))

    tracks_raw = np.load(str(tracks_path))
    vis_raw = np.load(str(vis_path))
    tracks_nt2_np, vis_nt_np = _normalize_tracks_vis_with_mask_length(
        tracks_raw=tracks_raw,
        vis_raw=vis_raw,
        expected_t=int(masks_thw_np.shape[0]),
    )

    if int(args.start_frame) != 0:
        raise ValueError(
            "--start_frame must be 0 for this script because the aligned mesh is posed for frame 0."
        )

    t_total = min(int(masks_thw_np.shape[0]), int(tracks_nt2_np.shape[1]), int(vis_nt_np.shape[1]))
    if int(args.end_frame) >= 0:
        t_use = min(t_total, int(args.end_frame) + 1)
    else:
        t_use = t_total
    if t_use <= 0:
        raise RuntimeError("No valid frames available after range clipping.")

    masks_thw_np = masks_thw_np[:t_use]
    tracks_nt2_np = tracks_nt2_np[:, :t_use, :].astype(np.float32)
    vis_nt_np = vis_nt_np[:, :t_use].astype(np.float32)

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    seed_uv_n2 = tracks_nt2_np[:, 0, :]
    mapping = _map_seed_points_to_mesh(
        seed_uv_n2=seed_uv_n2,
        verts_cv_np=verts_cv_np,
        faces_np=faces_np,
        mask0_hw_np=masks_thw_np[0],
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        h=h_mask,
        w=w_mask,
        device=device,
        bin_size=int(args.bin_size),
        mask_gate_threshold=float(args.mask_gate_threshold),
    )

    valid_seed = mapping.valid_seed_mask
    tracks_valid = tracks_nt2_np[valid_seed]
    vis_valid = vis_nt_np[valid_seed]
    x0_m3 = mapping.points_cv

    if tracks_valid.shape[0] < int(args.min_valid_tracks):
        raise RuntimeError(
            f"Too few valid tracks after frame-0 mesh mapping: {tracks_valid.shape[0]} "
            f"(min={int(args.min_valid_tracks)})."
        )

    obs_uv_tm2 = torch.from_numpy(tracks_valid).to(device=device, dtype=torch.float32).permute(1, 0, 2)
    vis_tm = torch.from_numpy(vis_valid).to(device=device, dtype=torch.float32).permute(1, 0)
    masks_thw = torch.from_numpy(masks_thw_np).to(device=device, dtype=torch.float32)

    xi_tm1_6 = torch.nn.Parameter(torch.zeros((max(t_use - 1, 0), 6), device=device, dtype=torch.float32))

    iter_rows: list[dict[str, Any]] = []

    def _record_iter(it: int, stage: str, bundle: LossBundle) -> None:
        active = bundle.weights_tm > 0.0
        if bool(active.any()):
            reproj = torch.sqrt(torch.clamp(bundle.r2_tm[active], min=1e-12))
            mean_reproj = float(reproj.mean().item())
        else:
            mean_reproj = float("nan")
        iter_rows.append(
            {
                "iter": int(it),
                "stage": stage,
                "total": float(bundle.total.detach().cpu().item()),
                "e_img": float(bundle.e_img.detach().cpu().item()),
                "e_smooth": float(bundle.e_smooth.detach().cpu().item()),
                "e_vel": float(bundle.e_vel.detach().cpu().item()),
                "active_pairs": int(active.sum().item()),
                "sum_weight": float(bundle.weights_tm.sum().item()),
                "mean_reproj_px": mean_reproj,
            }
        )

    # Adam stage.
    if t_use > 1 and int(args.adam_iters) > 0:
        adam = torch.optim.Adam([xi_tm1_6], lr=float(args.adam_lr))
        for it in range(1, int(args.adam_iters) + 1):
            adam.zero_grad(set_to_none=True)
            bundle = _compute_loss_bundle(
                xi_tm1_6=xi_tm1_6,
                x0_m3=x0_m3,
                obs_uv_tm2=obs_uv_tm2,
                vis_tm=vis_tm,
                masks_thw=masks_thw,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                lambda_a=float(args.lambda_a),
                lambda_v=float(args.lambda_v),
                huber_delta_px=float(args.huber_delta_px),
                visibility_threshold=float(args.visibility_threshold),
                mask_gate_threshold=float(args.mask_gate_threshold),
            )
            if not torch.isfinite(bundle.total):
                raise RuntimeError(f"Loss became non-finite at Adam iter={it}.")
            bundle.total.backward()
            adam.step()

            if (
                int(args.debug_save_interval) <= 1
                or it % int(args.debug_save_interval) == 0
                or it == int(args.adam_iters)
            ):
                _record_iter(it=it, stage="adam", bundle=bundle)
            if int(args.log_every) > 0 and (it % int(args.log_every) == 0 or it == int(args.adam_iters)):
                print(
                    f"[{object_slug}] adam iter={it:05d} "
                    f"total={float(bundle.total.detach().cpu().item()):.6f} "
                    f"img={float(bundle.e_img.detach().cpu().item()):.6f} "
                    f"smooth={float(bundle.e_smooth.detach().cpu().item()):.6f} "
                    f"vel={float(bundle.e_vel.detach().cpu().item()):.6f}"
                )

    # Optional LBFGS refinement.
    if t_use > 1 and (not args.disable_lbfgs) and int(args.lbfgs_iters) > 0:
        lbfgs = torch.optim.LBFGS(
            [xi_tm1_6],
            lr=float(args.lbfgs_lr),
            max_iter=int(args.lbfgs_iters),
            line_search_fn="strong_wolfe",
        )
        closure_calls = {"n": 0}

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            bundle_local = _compute_loss_bundle(
                xi_tm1_6=xi_tm1_6,
                x0_m3=x0_m3,
                obs_uv_tm2=obs_uv_tm2,
                vis_tm=vis_tm,
                masks_thw=masks_thw,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                lambda_a=float(args.lambda_a),
                lambda_v=float(args.lambda_v),
                huber_delta_px=float(args.huber_delta_px),
                visibility_threshold=float(args.visibility_threshold),
                mask_gate_threshold=float(args.mask_gate_threshold),
            )
            if not torch.isfinite(bundle_local.total):
                raise RuntimeError("Loss became non-finite during LBFGS closure.")
            bundle_local.total.backward()
            closure_calls["n"] += 1
            return bundle_local.total

        lbfgs.step(closure)
        bundle_lbfgs = _compute_loss_bundle(
            xi_tm1_6=xi_tm1_6,
            x0_m3=x0_m3,
            obs_uv_tm2=obs_uv_tm2,
            vis_tm=vis_tm,
            masks_thw=masks_thw,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            lambda_a=float(args.lambda_a),
            lambda_v=float(args.lambda_v),
            huber_delta_px=float(args.huber_delta_px),
            visibility_threshold=float(args.visibility_threshold),
            mask_gate_threshold=float(args.mask_gate_threshold),
        )
        _record_iter(it=(iter_rows[-1]["iter"] + 1) if iter_rows else 1, stage="lbfgs_final", bundle=bundle_lbfgs)
        print(f"[{object_slug}] lbfgs closure_calls={closure_calls['n']} total={iter_rows[-1]['total']:.6f}")

    # Final evaluation bundle.
    final_bundle = _compute_loss_bundle(
        xi_tm1_6=xi_tm1_6,
        x0_m3=x0_m3,
        obs_uv_tm2=obs_uv_tm2,
        vis_tm=vis_tm,
        masks_thw=masks_thw,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        lambda_a=float(args.lambda_a),
        lambda_v=float(args.lambda_v),
        huber_delta_px=float(args.huber_delta_px),
        visibility_threshold=float(args.visibility_threshold),
        mask_gate_threshold=float(args.mask_gate_threshold),
    )
    _record_iter(
        it=(iter_rows[-1]["iter"] + 1) if iter_rows else 0,
        stage="final",
        bundle=final_bundle,
    )
    t_mats_np = final_bundle.T_mats.detach().cpu().numpy().astype(np.float32)

    _save_pose_json(
        out_path=out_dir / "poses.json",
        t_mats_np=t_mats_np,
        frame_offset=int(args.start_frame),
    )
    _save_mesh_sequences(
        mesh_template=mesh_template,
        verts0_cv=verts_cv_np,
        t_mats_np=t_mats_np,
        meshes_dir=meshes_dir,
        output_coord=str(args.output_coord),
        frame_offset=int(args.start_frame),
    )

    overlay_ok, overlay_msg = _render_overlays(
        frame_paths=frame_paths,
        verts0_cv=verts_cv_np,
        t_mats_np=t_mats_np,
        k=k,
        out_dir=out_dir,
        frame_offset=int(args.start_frame),
        fps=float(args.overlay_fps),
        max_verts=int(args.overlay_max_verts),
        point_radius=int(args.overlay_point_radius),
    )

    iter_csv = debug_dir / "iter_metrics.csv"
    frame_csv = debug_dir / "frame_metrics.csv"
    _save_csv(iter_csv, iter_rows)
    frame_rows = _build_frame_metrics(
        frame_offset=int(args.start_frame),
        bundle=final_bundle,
        mask_gate_threshold=float(args.mask_gate_threshold),
    )
    _save_csv(frame_csv, frame_rows)
    _save_loss_plot(debug_dir / "loss_curves.png", iter_rows)

    # Frame-0 correspondence visualization.
    if frame_paths and len(frame_paths) > int(args.start_frame):
        frame0 = cv2.imread(str(frame_paths[int(args.start_frame)]))
        if frame0 is None:
            frame0 = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)
    else:
        frame0 = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)

    obs0 = final_bundle.obs_uv_tm2[0].detach().cpu().numpy()
    pred0 = final_bundle.pred_uv_tm2[0].detach().cpu().numpy()
    corr_img = _draw_frame0_correspondence(
        frame_bgr=frame0,
        obs_uv_m2=obs0,
        pred_uv_m2=pred0,
        max_points=2000,
    )
    cv2.putText(
        corr_img,
        f"{object_slug}: frame0 reprojection correspondences",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_dir / "frame0_correspondence.png"), corr_img)

    valid_seed_count = int(valid_seed.sum())
    elapsed_s = float(time.time() - start_time)

    object_summary = {
        "object_name": object_name,
        "slug": object_slug,
        "status": "processed",
        "mesh_path": str(mesh_path),
        "tracks_path": str(tracks_path),
        "visibility_path": str(vis_path),
        "mask_dir": str(mask_dir),
        "num_input_tracks": int(tracks_nt2_np.shape[0]),
        "num_valid_seed_tracks": valid_seed_count,
        "num_dropped_invalid_face": int(mapping.invalid_face_count),
        "num_dropped_outside_mask0": int(mapping.outside_mask0_count),
        "num_dropped_nonfinite_seed": int(mapping.nonfinite_seed_count),
        "num_frames": int(t_use),
        "huber_delta_px": float(args.huber_delta_px),
        "lambda_a": float(args.lambda_a),
        "lambda_v": float(args.lambda_v),
        "adam_iters": int(args.adam_iters),
        "lbfgs_enabled": bool(not args.disable_lbfgs),
        "lbfgs_iters": int(args.lbfgs_iters),
        "final_total_loss": float(final_bundle.total.detach().cpu().item()),
        "final_e_img": float(final_bundle.e_img.detach().cpu().item()),
        "final_e_smooth": float(final_bundle.e_smooth.detach().cpu().item()),
        "final_e_vel": float(final_bundle.e_vel.detach().cpu().item()),
        "overlay_rendered": bool(overlay_ok),
        "overlay_message": overlay_msg,
        "iter_metrics_csv": str(iter_csv),
        "frame_metrics_csv": str(frame_csv),
        "elapsed_seconds": elapsed_s,
    }
    return object_summary


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    cotracker_video_dir, aligned_mesh_video_dir, segment_video_dir, output_root = _resolve_default_dirs(
        args=args, script_dir=script_dir
    )
    pag_path = _resolve_pag_path(args=args, script_dir=script_dir)
    k, intrinsics_summary_path = _load_intrinsics_from_alignment_summary(aligned_mesh_video_dir)
    output_video_dir = (output_root / args.video_name).resolve()
    ensure_dir(output_video_dir)

    if not cotracker_video_dir.exists() or not cotracker_video_dir.is_dir():
        raise NotADirectoryError(f"CoTracker video dir not found: {cotracker_video_dir}")
    if not aligned_mesh_video_dir.exists() or not aligned_mesh_video_dir.is_dir():
        raise NotADirectoryError(f"Aligned mesh video dir not found: {aligned_mesh_video_dir}")
    if not segment_video_dir.exists() or not segment_video_dir.is_dir():
        raise NotADirectoryError(f"Segment video dir not found: {segment_video_dir}")

    device = _to_device(args.device)
    pag_objects = _load_pag_objects_from_states_only(pag_path)

    frames_dir = _resolve_frames_dir(cotracker_video_dir, segment_video_dir)
    frame_paths: list[Path] = []
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
    else:
        print("[WARN] No _frames directory found in CoTracker or Segment_Video outputs.")

    run_summary: dict[str, Any] = {
        "video_name": args.video_name,
        "status": "completed",
        "inputs": {
            "cotracker_video_dir": str(cotracker_video_dir),
            "aligned_mesh_video_dir": str(aligned_mesh_video_dir),
            "segment_video_dir": str(segment_video_dir),
            "pag_file": str(pag_path),
            "intrinsics_source": str(intrinsics_summary_path),
            "frames_dir": None if frames_dir is None else str(frames_dir),
        },
        "optimization_settings": {
            "huber_delta_px": float(args.huber_delta_px),
            "lambda_a": float(args.lambda_a),
            "lambda_v": float(args.lambda_v),
            "visibility_threshold": float(args.visibility_threshold),
            "mask_gate_threshold": float(args.mask_gate_threshold),
            "adam_iters": int(args.adam_iters),
            "adam_lr": float(args.adam_lr),
            "lbfgs_enabled": bool(not args.disable_lbfgs),
            "lbfgs_iters": int(args.lbfgs_iters),
            "lbfgs_lr": float(args.lbfgs_lr),
        },
        "energy_terms": {
            "pose_parameterization": "T_1 = I (fixed), T_t = exp(xi_t) for t=2..T",
            "seed_mapping": (
                "X_i = barycentric interpolation of frame-0 aligned mesh triangle at seed pixel; "
                "drop seeds with invalid face id or outside frame-0 mask"
            ),
            "projection": "u = fx * X/Z + cx, v = fy * Y/Z + cy",
            "mask_gate": (
                "mask_gate(i,t)=1 if bilinear_sample(mask_t, u_obs(i,t)) >= mask_gate_threshold else 0"
            ),
            "weights": "w_{i,t} = vis_{i,t} * mask_gate(i,t) * z_valid(i,t)",
            "rho_huber": "rho(s)=s if s<=d^2 else 2*d*sqrt(s)-d^2, with d=huber_delta_px",
            "E_img": "sum_{i,t} w_{i,t} * rho(||u_obs(i,t)-u_pred(i,t)||_2^2)",
            "delta_t": "delta_t = log(inv(T_t) @ T_{t+1})",
            "E_smooth": "sum_{t=2..T-2} ||delta_t - delta_{t-1}||_2^2",
            "E_vel": "sum_{t=1..T-1} ||delta_t||_2^2",
            "E_total": "E_img + lambda_a * E_smooth + lambda_v * E_vel",
        },
        "device": str(device),
        "output_dir": str(output_video_dir),
        "objects_from_pag_states": [
            {"name": name, "slug": slug} for name, slug in pag_objects
        ],
        "objects_processed": [],
        "objects_skipped": [],
        "objects_failed": [],
    }

    print(f"[INFO] video_name: {args.video_name}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] pag_file: {pag_path}")
    print(f"[INFO] num_objects_from_pag_states: {len(pag_objects)}")

    for obj_name, obj_slug in pag_objects:
        mesh_path = aligned_mesh_video_dir / "meshes" / f"{obj_slug}.ply"
        tracks_path = cotracker_video_dir / obj_slug / "tracks.npy"
        vis_path = cotracker_video_dir / obj_slug / "visibility.npy"
        mask_dir = _resolve_object_mask_dir(segment_video_dir, obj_slug)
        out_dir = output_video_dir / obj_slug

        missing = []
        if not mesh_path.exists():
            missing.append(f"mesh missing: {mesh_path}")
        if not tracks_path.exists():
            missing.append(f"tracks missing: {tracks_path}")
        if not vis_path.exists():
            missing.append(f"visibility missing: {vis_path}")
        if not mask_dir.exists():
            missing.append(f"mask dir missing: {mask_dir}")

        if missing:
            reason = "; ".join(missing)
            print(f"[WARN] Skipping {obj_slug}: {reason}")
            run_summary["objects_skipped"].append(
                {"name": obj_name, "slug": obj_slug, "reason": reason}
            )
            continue

        print(f"\n[OBJECT] {obj_name} ({obj_slug})")
        try:
            obj_summary = _run_single_object(
                object_name=obj_name,
                object_slug=obj_slug,
                mesh_path=mesh_path,
                tracks_path=tracks_path,
                vis_path=vis_path,
                mask_dir=mask_dir,
                frame_paths=frame_paths,
                k=k,
                args=args,
                out_dir=out_dir,
                device=device,
            )
            run_summary["objects_processed"].append(obj_summary)
            print(f"[OK] Saved tracking outputs: {out_dir}")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] Failed {obj_slug}: {reason}")
            run_summary["objects_failed"].append(
                {"name": obj_name, "slug": obj_slug, "reason": reason}
            )

    with (output_video_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
