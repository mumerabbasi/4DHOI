"""Multi-frame SE(3) object mesh tracking from CoTracker tracks (corrected).

Fixes over original track_object_mesh.py:
  1. **Convention mismatch**: PyTorch3D's se3_exp_map returns 4x4 matrices in
     *row-vector* convention (translation in bottom row, R^T in top-left).
     The original code treated them as standard *column-vector* matrices,
     causing all translations to be zero and rotations transposed.
  2. **Pose parameterization**: Uses axis_angle_to_matrix (standard convention)
     + explicit translation vector. This avoids any convention ambiguity.
  3. **SE(3) velocity/smoothness**: Computes body-frame relative transforms
     properly using matrix operations, not mixed conventions.

Coordinate convention used throughout: **OpenCV** (X-right, Y-down, Z-forward).
The 4x4 matrices are in standard column-vector convention: [[R, t], [0, 1]].
Projection: u = fx * X/Z + cx,  v = fy * Y/Z + cy.

Usage is identical to the original script.
"""

from __future__ import annotations

import argparse
import json
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
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from tracking_utils import (
    _draw_frame0_correspondence,
    _list_mask_files,
    _load_intrinsics_from_alignment_summary,
    _load_mask_stack,
    _load_pag_objects_from_states_only,
    _normalize_tracks_vis_with_mask_length,
    _resolve_default_dirs,
    _resolve_frames_dir,
    _resolve_object_mask_dir,
    _resolve_pag_path,
    _save_csv,
    _save_loss_plots,
    _to_device,
    close_ffmpeg,
    draw_overlay,
    ensure_dir,
    list_images,
    start_ffmpeg_writer,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAY_FILL_ALPHA = 0.60
OVERLAY_CONTOUR_THICKNESS = 0
OVERLAY_COLOR_BGR = (0, 255, 255)  # yellow-cyan


# ---------------------------------------------------------------------------
# Argument parsing (same interface as original)
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track aligned object meshes with corrected SE(3) optimizer."
    )
    p.add_argument("--video_name", type=str, default="video_01")
    p.add_argument("--cotracker_video_dir", type=str, default=None)
    p.add_argument("--aligned_mesh_video_dir", type=str, default=None)
    p.add_argument("--segment_video_dir", type=str, default=None)
    p.add_argument("--pag_file", type=str, default=None)
    p.add_argument("--output_root", type=str, default="./output")

    p.add_argument("--output_coord", type=str, choices=["opencv", "pytorch3d"], default="opencv")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bin_size", type=int, default=0)
    p.add_argument("--mask_threshold", type=int, default=127)
    p.add_argument("--mask_gate_threshold", type=float, default=0.5)
    p.add_argument("--visibility_threshold", type=float, default=0.0)
    p.add_argument("--min_valid_tracks", type=int, default=50)

    p.add_argument("--huber_delta_px", type=float, default=3.0)
    p.add_argument("--lambda_img", type=float, default=1.0)
    p.add_argument("--lambda_a", type=float, default=10.0)
    p.add_argument("--lambda_v", type=float, default=10.0)
    p.add_argument("--adam_iters", type=int, default=4000)
    p.add_argument("--adam_lr", type=float, default=1e-2)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_rel_min_delta", type=float, default=1e-4)
    p.add_argument("--early_stop_min_iter", type=int, default=300)
    p.add_argument("--disable_lbfgs", action="store_true")
    p.add_argument("--lbfgs_iters", type=int, default=120)
    p.add_argument("--lbfgs_lr", type=float, default=0.5)
    p.add_argument("--log_every", type=int, default=20)

    # --- Optimisation improvements ---
    p.add_argument("--disable_pnp_init", action="store_true",
                   help="Disable PnP-RANSAC sequential initialization (use zeros)")
    p.add_argument("--pnp_ransac_thresh", type=float, default=8.0,
                   help="RANSAC reprojection threshold in PnP init (px)")
    p.add_argument("--outlier_reproj_thresh_px", type=float, default=20.0,
                   help="Tracks with mean reproj > this after PnP init are outliers (0=disable)")
    p.add_argument("--outlier_max_fraction", type=float, default=0.4,
                   help="Max fraction of tracks to reject as outliers")
    p.add_argument("--lr_schedule", type=str, choices=["none", "cosine"], default="cosine",
                   help="LR schedule for Adam optimizer")
    p.add_argument("--graduated_huber", action=argparse.BooleanOptionalAction, default=True,
                   help="Anneal Huber delta from 3x to 1x over first half of Adam")
    p.add_argument("--retrim_interval", type=int, default=1000,
                   help="Every N Adam iters, zero out per-frame outlier tracks (0=disable)")
    p.add_argument("--retrim_percentile", type=float, default=90.0,
                   help="Per-frame residual percentile threshold for retrimming (e.g., 90=keep best 90%%)")

    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--overlay_fps", type=float, default=6.0)
    p.add_argument("--debug_save_interval", type=int, default=20)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def _cv_to_p3d_torch(pts: torch.Tensor) -> torch.Tensor:
    """OpenCV (X-right Y-down Z-fwd) → PyTorch3D (X-left Y-up Z-fwd)."""
    out = pts.clone()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _cv_to_p3d_np(pts: np.ndarray) -> np.ndarray:
    out = pts.copy().astype(np.float32)
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


# ---------------------------------------------------------------------------
# Rasterizer (for seed-point mapping at frame 0 only)
# ---------------------------------------------------------------------------
def _build_rasterizer(
    device: torch.device,
    fx: float, fy: float, cx: float, cy: float,
    h: int, w: int, bin_size: int,
) -> MeshRasterizer:
    """PyTorch3D rasterizer using screen-space (non-NDC) cameras."""
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
        max_faces_per_bin=300_000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


# ---------------------------------------------------------------------------
# Mask sampling
# ---------------------------------------------------------------------------
def _sample_mask_bilinear_single(mask_hw: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample one [H,W] mask at sub-pixel positions uv [N,2]. Returns [N]."""
    if uv.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=mask_hw.device)
    h, w = mask_hw.shape
    grid_x = (2.0 * uv[:, 0] / max(w - 1, 1)) - 1.0
    grid_y = (2.0 * uv[:, 1] / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask_hw.view(1, 1, h, w), grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return sampled.view(-1)


def _sample_masks_bilinear_seq(masks: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample masks [T,H,W] at positions uv [T,M,2]. Returns [T,M]."""
    if uv.numel() == 0:
        return torch.zeros(masks.shape[0], 0, dtype=torch.float32, device=masks.device)
    t, h, w = masks.shape
    grid_x = (2.0 * uv[..., 0] / max(w - 1, 1)) - 1.0
    grid_y = (2.0 * uv[..., 1] / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)  # [T,M,1,2]
    sampled = F.grid_sample(
        masks.unsqueeze(1), grid,  # [T,1,H,W]
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)  # [T,M]


# ---------------------------------------------------------------------------
# Seed-point mapping (frame 0)
# ---------------------------------------------------------------------------
@dataclass
class SeedMappingResult:
    points_cv: torch.Tensor       # [M, 3] in OpenCV coords
    valid_seed_mask: np.ndarray    # [N_total] bool
    invalid_face_count: int
    outside_mask0_count: int
    nonfinite_seed_count: int


def _map_seed_points_to_mesh(
    seed_uv: np.ndarray,         # [N, 2] pixel coords at frame 0
    verts_cv: np.ndarray,        # [V, 3] in OpenCV coords
    faces: np.ndarray,           # [F, 3]
    mask0: np.ndarray,           # [H, W] float 0/1
    fx: float, fy: float, cx: float, cy: float,
    h: int, w: int,
    device: torch.device,
    bin_size: int,
    mask_gate_threshold: float,
) -> SeedMappingResult:
    """Map 2D seed points to 3D via rasterised depth + barycentric interp."""
    rasterizer = _build_rasterizer(device, fx, fy, cx, cy, h, w, bin_size)

    verts_cv_t = torch.from_numpy(verts_cv).to(device=device, dtype=torch.float32)
    # PyTorch3D rasterizer needs PyTorch3D-convention vertices
    verts_p3d = _cv_to_p3d_torch(verts_cv_t)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])

    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]   # [H, W]
    bary_coords = fragments.bary_coords[0, ..., 0, :]  # [H, W, 3]

    seed_t = torch.from_numpy(seed_uv).to(device=device, dtype=torch.float32)
    finite = torch.isfinite(seed_t).all(dim=1)

    x_idx = torch.clamp(torch.round(seed_t[:, 0]).long(), 0, w - 1)
    y_idx = torch.clamp(torch.round(seed_t[:, 1]).long(), 0, h - 1)
    face_id = pix_to_face[y_idx, x_idx]
    bary_seed = bary_coords[y_idx, x_idx]

    mask0_t = torch.from_numpy(mask0.astype(np.float32)).to(device)
    mask_vals = _sample_mask_bilinear_single(mask0_t, seed_t)

    valid = finite & (face_id >= 0) & (mask_vals >= mask_gate_threshold)

    # Barycentric interpolation in **OpenCV** coords (not PyTorch3D)
    points_all = torch.zeros(seed_t.shape[0], 3, dtype=torch.float32, device=device)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if idx.numel() > 0:
        tri_verts = verts_cv_t[faces_t[face_id[idx].long()]]  # [M, 3, 3]
        b = bary_seed[idx].unsqueeze(-1)                       # [M, 3, 1]
        points_all[idx] = (b * tri_verts).sum(dim=1)

    return SeedMappingResult(
        points_cv=points_all[idx],
        valid_seed_mask=valid.cpu().numpy().astype(bool),
        invalid_face_count=int(((face_id < 0) & finite).sum().item()),
        outside_mask0_count=int(
            ((mask_vals < mask_gate_threshold) & (face_id >= 0) & finite).sum().item()
        ),
        nonfinite_seed_count=int((~finite).sum().item()),
    )


# ---------------------------------------------------------------------------
# Pose helpers – **standard column-vector convention** [[R t],[0 1]]
# ---------------------------------------------------------------------------
def _build_T_matrices(
    rotvecs: torch.Tensor,   # [T-1, 3] axis-angle
    trans: torch.Tensor,     # [T-1, 3] translation
    t_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Build [T, 4, 4] standard-convention SE(3) matrices.

    Frame 0 is identity.  Frame t: T = [[R(rotvec_t), trans_t], [0, 1]].

    Convention: p_t = R_t @ p_0 + t_t  (column-vector).
    """
    eye = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0)
    if t_frames <= 1:
        return eye

    R = axis_angle_to_matrix(rotvecs)  # [T-1, 3, 3] standard rotation
    T = torch.zeros(t_frames - 1, 4, 4, device=device, dtype=torch.float32)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    T[:, 3, 3] = 1.0
    return torch.cat([eye, T], dim=0)  # [T, 4, 4]


def _relative_body_velocity(
    R_all: torch.Tensor,     # [T, 3, 3]
    t_all: torch.Tensor,     # [T, 3]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute body-frame velocity for t = 0 .. T-2.

    R_rel_t = R_t^T @ R_{t+1}
    omega_t = axis_angle(R_rel_t)      (3-vector)
    v_t     = R_t^T @ (t_{t+1} - t_t)  (body-frame translation vel)

    Returns:
        omega [T-1, 3], v [T-1, 3]
    """
    R_rel = torch.bmm(R_all[:-1].transpose(-1, -2), R_all[1:])  # [T-1,3,3]
    omega = matrix_to_axis_angle(R_rel)                          # [T-1,3]

    dt = t_all[1:] - t_all[:-1]   # [T-1, 3]
    # Body-frame: rotate back by current R
    v = torch.bmm(
        R_all[:-1].transpose(-1, -2),
        dt.unsqueeze(-1),
    ).squeeze(-1)  # [T-1, 3]

    return omega, v


# ---------------------------------------------------------------------------
# Huber robustifier on squared residuals
# ---------------------------------------------------------------------------
def _huber_on_squared(s: torch.Tensor, delta: float) -> torch.Tensor:
    d2 = delta * delta
    sqrt_s = torch.sqrt(s.clamp(min=1e-12))
    return torch.where(s <= d2, s, 2.0 * delta * sqrt_s - d2)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
@dataclass
class LossBundle:
    total: torch.Tensor
    e_img: torch.Tensor
    e_smooth: torch.Tensor
    e_vel: torch.Tensor
    e_img_raw: torch.Tensor
    e_smooth_raw: torch.Tensor
    e_vel_raw: torch.Tensor
    e_img_denom: torch.Tensor
    e_smooth_denom: torch.Tensor
    e_vel_denom: torch.Tensor
    T_mats: torch.Tensor        # [T, 4, 4] standard convention
    pred_uv: torch.Tensor       # [T, M, 2]
    obs_uv: torch.Tensor        # [T, M, 2]
    weights: torch.Tensor       # [T, M]
    r2: torch.Tensor             # [T, M]
    mask_values: torch.Tensor   # [T, M]
    vis_weights: torch.Tensor   # [T, M]


def _compute_loss(
    rotvecs: torch.Tensor,     # [T-1, 3]
    trans: torch.Tensor,       # [T-1, 3]
    x0: torch.Tensor,         # [M, 3] frame-0 3D points in OpenCV
    obs_uv: torch.Tensor,     # [T, M, 2]
    vis: torch.Tensor,         # [T, M]
    masks: torch.Tensor,      # [T, H, W]
    fx: float, fy: float, cx: float, cy: float,
    lambda_img: float, lambda_a: float, lambda_v: float,
    huber_delta: float,
    vis_threshold: float,
    mask_gate_threshold: float,
) -> LossBundle:
    device = x0.device
    t_frames = obs_uv.shape[0]

    # --- Build SE(3) matrices in standard convention ---
    T_mats = _build_T_matrices(rotvecs, trans, t_frames, device)
    R_all = T_mats[:, :3, :3]   # [T, 3, 3]
    t_all = T_mats[:, :3, 3]    # [T, 3]

    # --- Transform 3D points: x_t = R_t @ x_0 + t_t ---
    # x0 [M,3], R_all [T,3,3], t_all [T,3]
    xt = torch.einsum("tij,mj->tmi", R_all, x0) + t_all.unsqueeze(1)  # [T,M,3]

    # --- Project to pixel coords (OpenCV) ---
    z = xt[..., 2]
    z_valid = z > 1e-6
    z_safe = torch.where(z_valid, z, torch.ones_like(z))
    pred_u = fx * xt[..., 0] / z_safe + cx
    pred_v = fy * xt[..., 1] / z_safe + cy
    pred_uv = torch.stack([pred_u, pred_v], dim=-1)  # [T,M,2]

    # --- Weights: visibility * mask_gate * finite_obs * z_valid ---
    finite_obs = torch.isfinite(obs_uv).all(dim=-1)
    mask_vals = _sample_masks_bilinear_seq(masks, obs_uv)
    mask_gate = mask_vals >= mask_gate_threshold

    vis_w = vis if vis_threshold <= 0.0 else torch.where(
        vis >= vis_threshold, vis, torch.zeros_like(vis)
    )
    weights = vis_w * mask_gate.float() * finite_obs.float() * z_valid.float()

    # --- Image reprojection loss (Huber) ---
    r2 = ((obs_uv - pred_uv) ** 2).sum(dim=-1)       # [T,M]
    robust = _huber_on_squared(r2, huber_delta)
    e_img_raw = (weights * robust).sum()
    e_img_denom = weights.sum().clamp(min=1.0)
    e_img = e_img_raw / e_img_denom

    # --- Velocity & smoothness (body-frame SE(3)) ---
    if t_frames > 1:
        omega_vel, trans_vel = _relative_body_velocity(R_all, t_all)
        vel = torch.cat([omega_vel, trans_vel], dim=-1)  # [T-1, 6]
        e_vel_raw = (vel ** 2).sum()
        e_vel_denom = torch.tensor(max(vel.numel(), 1), dtype=torch.float32, device=device)
        e_vel = e_vel_raw / e_vel_denom

        if vel.shape[0] >= 2:
            accel = vel[1:] - vel[:-1]
            e_smooth_raw = (accel ** 2).sum()
            e_smooth_denom = torch.tensor(max(accel.numel(), 1), dtype=torch.float32, device=device)
            e_smooth = e_smooth_raw / e_smooth_denom
        else:
            e_smooth_raw = torch.zeros((), device=device)
            e_smooth_denom = torch.ones((), device=device)
            e_smooth = torch.zeros((), device=device)
    else:
        e_vel_raw = torch.zeros((), device=device)
        e_vel_denom = torch.ones((), device=device)
        e_vel = torch.zeros((), device=device)
        e_smooth_raw = torch.zeros((), device=device)
        e_smooth_denom = torch.ones((), device=device)
        e_smooth = torch.zeros((), device=device)

    total = lambda_img * e_img + lambda_a * e_smooth + lambda_v * e_vel

    return LossBundle(
        total=total,
        e_img=e_img, e_smooth=e_smooth, e_vel=e_vel,
        e_img_raw=e_img_raw, e_smooth_raw=e_smooth_raw, e_vel_raw=e_vel_raw,
        e_img_denom=e_img_denom, e_smooth_denom=e_smooth_denom, e_vel_denom=e_vel_denom,
        T_mats=T_mats, pred_uv=pred_uv, obs_uv=obs_uv,
        weights=weights, r2=r2,
        mask_values=mask_vals, vis_weights=vis_w,
    )


# ---------------------------------------------------------------------------
# Pose application & saving – all in OpenCV standard convention
# ---------------------------------------------------------------------------
def _transform_points_cv(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply standard [[R,t],[0,1]] to points.  p' = R @ p + t."""
    R = T[:3, :3].astype(np.float32)
    t = T[:3, 3].astype(np.float32)
    return (pts.astype(np.float32) @ R.T) + t[None, :]


def _save_pose_json(path: Path, T_mats: np.ndarray, frame_offset: int) -> None:
    rows = []
    for i in range(T_mats.shape[0]):
        rows.append({
            "frame": int(frame_offset + i),
            "T_4x4": T_mats[i].tolist(),
        })
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def _save_mesh_sequence(
    mesh_template: trimesh.Trimesh,
    verts0_cv: np.ndarray,
    T_mats: np.ndarray,
    meshes_dir: Path,
    output_coord: str,
    frame_offset: int,
) -> None:
    ensure_dir(meshes_dir)
    for i in range(T_mats.shape[0]):
        verts_t = _transform_points_cv(verts0_cv, T_mats[i])
        if output_coord == "pytorch3d":
            verts_t = _cv_to_p3d_np(verts_t)
        mesh = mesh_template.copy()
        mesh.vertices = verts_t
        mesh.export(str(meshes_dir / f"frame_{frame_offset + i:04d}.ply"))


def _render_overlays(
    frame_paths: list[Path],
    verts0_cv: np.ndarray,
    faces: np.ndarray,
    T_mats: np.ndarray,
    k: np.ndarray,
    out_dir: Path,
    frame_offset: int,
    fps: float,
) -> tuple[bool, str]:
    overlays_dir = out_dir / "overlays"
    ensure_dir(overlays_dir)
    if not frame_paths or len(frame_paths) <= frame_offset or faces.shape[0] == 0:
        return False, "Skipped overlays (no frames or faces)."

    end = min(len(frame_paths), frame_offset + T_mats.shape[0])
    first = cv2.imread(str(frame_paths[frame_offset]))
    if first is None:
        return False, f"Cannot read first frame: {frame_paths[frame_offset]}"
    h, w = first.shape[:2]

    writer = start_ffmpeg_writer(out_dir / "overlay.mp4", fps, (h, w))
    try:
        for local_i, fi in enumerate(range(frame_offset, end)):
            frame = cv2.imread(str(frame_paths[fi]))
            if frame is None:
                continue
            verts_t = _transform_points_cv(verts0_cv, T_mats[local_i])
            overlay = draw_overlay(
                frame_bgr=frame, verts_cv=verts_t, faces=faces, k=k,
                fill_alpha=OVERLAY_FILL_ALPHA, contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                color_bgr=OVERLAY_COLOR_BGR,
            )
            cv2.imwrite(str(overlays_dir / f"overlay_{fi:04d}.png"), overlay)
            if writer.stdin is not None:
                writer.stdin.write(np.ascontiguousarray(overlay).tobytes())
    finally:
        close_ffmpeg(writer)
    return True, f"Rendered overlays ({faces.shape[0]} faces)."


# ---------------------------------------------------------------------------
# Per-frame metrics
# ---------------------------------------------------------------------------
def _build_frame_metrics(
    frame_offset: int,
    bundle: LossBundle,
    mask_gate_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    T = bundle.r2.shape[0]
    for t in range(T):
        active = bundle.weights[t] > 0.0
        n_active = int(active.sum().item())
        if n_active > 0:
            reproj = torch.sqrt(bundle.r2[t][active].clamp(min=1e-12))
            rmean = float(reproj.mean().item())
            rp50 = float(torch.quantile(reproj, 0.5).item())
            rp90 = float(torch.quantile(reproj, 0.9).item())
        else:
            rmean = rp50 = rp90 = float("nan")

        # Extract rotation angle and translation magnitude for this frame
        T_mat = bundle.T_mats[t]
        R_t = T_mat[:3, :3]
        t_t = T_mat[:3, 3]
        # Rotation angle from trace: cos(theta) = (trace(R) - 1) / 2
        cos_angle = ((R_t.trace() - 1.0) / 2.0).clamp(-1.0, 1.0)
        rot_angle_deg = float(torch.acos(cos_angle).item()) * 180.0 / np.pi
        trans_mag = float(t_t.norm().item())

        rows.append({
            "frame_idx": frame_offset + t,
            "num_active": n_active,
            "sum_weight": float(bundle.weights[t].sum().item()),
            "reproj_mean_px": rmean,
            "reproj_p50_px": rp50,
            "reproj_p90_px": rp90,
            "rotation_angle_deg": rot_angle_deg,
            "translation_magnitude": trans_mag,
            "tx": float(t_t[0].item()),
            "ty": float(t_t[1].item()),
            "tz": float(t_t[2].item()),
        })
    return rows


# ---------------------------------------------------------------------------
# Debug: save per-frame pose summary image
# ---------------------------------------------------------------------------
def _save_pose_debug_plots(debug_dir: Path, frame_rows: list[dict]) -> None:
    """Save translation & rotation magnitude plots using cv2."""
    if not frame_rows:
        return
    frames = np.array([r["frame_idx"] for r in frame_rows])
    tx = np.array([r["tx"] for r in frame_rows])
    ty = np.array([r["ty"] for r in frame_rows])
    tz = np.array([r["tz"] for r in frame_rows])
    rot_deg = np.array([r["rotation_angle_deg"] for r in frame_rows])
    trans_mag = np.array([r["translation_magnitude"] for r in frame_rows])

    def _simple_cv2_plot(xs, ys_list, labels, title, out_path, ylabel="Value"):
        h, w = 480, 960
        pad_l, pad_r, pad_t, pad_b = 80, 30, 50, 60
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        x0, x1, y0, y1 = pad_l, w - pad_r, pad_t, h - pad_b

        all_y = np.concatenate(ys_list)
        ymin, ymax = float(np.nanmin(all_y)), float(np.nanmax(all_y))
        if abs(ymax - ymin) < 1e-12:
            ymin -= 0.01
            ymax += 0.01
        xmin, xmax = float(xs.min()), float(xs.max())
        if xmax == xmin:
            xmax = xmin + 1

        colors = [(0, 0, 200), (0, 160, 0), (200, 0, 0), (200, 0, 200)]
        for ci, (ys, lbl) in enumerate(zip(ys_list, labels)):
            pts = []
            for i in range(len(xs)):
                px = int(x0 + (xs[i] - xmin) / (xmax - xmin) * (x1 - x0))
                py = int(y1 - (ys[i] - ymin) / (ymax - ymin) * (y1 - y0))
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts_arr], False, colors[ci % len(colors)], 2, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (x1 - 180, y0 + 18 + ci * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[ci % len(colors)], 1, cv2.LINE_AA)

        cv2.putText(canvas, title, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Frame", (w // 2 - 20, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Y ticks
        for i in range(5):
            frac = i / 4.0
            yy = int(y1 - frac * (y1 - y0))
            val = ymin + frac * (ymax - ymin)
            cv2.putText(canvas, f"{val:.4f}", (5, yy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA)
            cv2.line(canvas, (x0, yy), (x1, yy), (230, 230, 230), 1)

        cv2.imwrite(str(out_path), canvas)

    _simple_cv2_plot(
        frames, [tx, ty, tz], ["tx", "ty", "tz"],
        "Translation Components", debug_dir / "pose_translation.png",
    )
    _simple_cv2_plot(
        frames, [trans_mag], ["||t||"],
        "Translation Magnitude", debug_dir / "pose_trans_magnitude.png",
    )
    _simple_cv2_plot(
        frames, [rot_deg], ["angle (deg)"],
        "Rotation Angle", debug_dir / "pose_rotation_angle.png",
    )


# ---------------------------------------------------------------------------
# PnP-RANSAC sequential initialization
# ---------------------------------------------------------------------------
def _pnp_sequential_init(
    x0_cv: np.ndarray,         # [M, 3] frame-0 3D in OpenCV cam coords
    obs_uv: np.ndarray,        # [T, M, 2] pixel observations
    vis: np.ndarray,           # [T, M] visibility
    k: np.ndarray,             # [3, 3] intrinsics
    ransac_thresh: float,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Per-frame PnP-RANSAC chained from previous frame.

    Returns (rotvecs [T-1,3], trans [T-1,3], per_frame_info).
    Coordinate convention matches our pose parameterization: p_t = R_t @ p_0 + t_t.
    PnP (OpenCV) returns exactly this when input points are in camera frame.
    """
    T, M = obs_uv.shape[:2]
    rotvecs_out = np.zeros((T - 1, 3), dtype=np.float32)
    trans_out = np.zeros((T - 1, 3), dtype=np.float32)
    info: list[dict] = []

    prev_rvec = np.zeros((3, 1), dtype=np.float64)
    prev_tvec = np.zeros((3, 1), dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)
    k64 = k.astype(np.float64)

    for t in range(1, T):
        valid = vis[t] > 0.5
        n_valid = int(valid.sum())
        if n_valid < 6:
            rotvecs_out[t - 1] = prev_rvec.ravel().astype(np.float32)
            trans_out[t - 1] = prev_tvec.ravel().astype(np.float32)
            info.append({"frame": t, "n_valid": n_valid, "n_inliers": 0, "pnp_ok": False})
            continue

        pts3d = x0_cv[valid].astype(np.float64)
        pts2d = obs_uv[t, valid].astype(np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, k64, dist_coeffs,
            rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
            useExtrinsicGuess=(t > 1),
            iterationsCount=200,
            reprojectionError=ransac_thresh,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        n_inliers = len(inliers) if (success and inliers is not None) else 0
        if success and n_inliers >= 6:
            # Refine with inliers
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    pts3d[inliers.ravel()], pts2d[inliers.ravel()],
                    k64, dist_coeffs, rvec, tvec,
                )
            except cv2.error:
                pass  # keep RANSAC result

            rotvecs_out[t - 1] = rvec.ravel().astype(np.float32)
            trans_out[t - 1] = tvec.ravel().astype(np.float32)
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()
        elif success:
            rotvecs_out[t - 1] = rvec.ravel().astype(np.float32)
            trans_out[t - 1] = tvec.ravel().astype(np.float32)
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()
        else:
            rotvecs_out[t - 1] = prev_rvec.ravel().astype(np.float32)
            trans_out[t - 1] = prev_tvec.ravel().astype(np.float32)

        info.append({"frame": t, "n_valid": n_valid, "n_inliers": n_inliers, "pnp_ok": bool(success)})

    return rotvecs_out, trans_out, info


# ---------------------------------------------------------------------------
# Outlier track rejection after PnP init
# ---------------------------------------------------------------------------
def _identify_outlier_tracks(
    x0: torch.Tensor,         # [M, 3]
    obs_uv: torch.Tensor,     # [T, M, 2]
    vis: torch.Tensor,        # [T, M]
    rotvecs: torch.Tensor,    # [T-1, 3]
    trans: torch.Tensor,      # [T-1, 3]
    fx: float, fy: float, cx: float, cy: float,
    threshold_px: float,
    max_fraction: float,
) -> torch.Tensor:
    """Return per-track boolean mask [M] where True = outlier.

    A track is an outlier if its mean reprojection error (over visible frames)
    after PnP initialization exceeds threshold_px.  At most max_fraction of
    tracks will be rejected.
    """
    with torch.no_grad():
        T = obs_uv.shape[0]
        M = x0.shape[0]
        T_mats = _build_T_matrices(rotvecs, trans, T, x0.device)
        R_all = T_mats[:, :3, :3]
        t_all = T_mats[:, :3, 3]

        xt = torch.einsum("tij,mj->tmi", R_all, x0) + t_all.unsqueeze(1)
        z = xt[..., 2].clamp(min=1e-6)
        pred_u = fx * xt[..., 0] / z + cx
        pred_v = fy * xt[..., 1] / z + cy
        pred_uv = torch.stack([pred_u, pred_v], dim=-1)

        err = torch.sqrt(((obs_uv - pred_uv) ** 2).sum(dim=-1).clamp(min=1e-12))

        # Mean error per track over visible frames
        vis_binary = (vis > 0.5).float()
        vis_count = vis_binary.sum(dim=0).clamp(min=1.0)
        mean_err = (err * vis_binary).sum(dim=0) / vis_count  # [M]

        outlier = mean_err > threshold_px

        # Cap outlier fraction
        n_outlier = int(outlier.sum().item())
        max_reject = int(M * max_fraction)
        if n_outlier > max_reject and max_reject > 0:
            # Keep only the worst max_reject tracks as outliers
            _, sorted_idx = mean_err.sort(descending=True)
            outlier = torch.zeros(M, dtype=torch.bool, device=x0.device)
            outlier[sorted_idx[:max_reject]] = True

    return outlier


# ---------------------------------------------------------------------------
# Main per-object optimisation
# ---------------------------------------------------------------------------
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
    t0 = time.time()
    ensure_dir(out_dir)
    debug_dir = out_dir / "debug"
    meshes_dir = out_dir / "meshes"
    ensure_dir(debug_dir)
    ensure_dir(meshes_dir)

    # --- Load mesh (OpenCV coords from alignment stage) ---
    mesh_template = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh_template, trimesh.Trimesh):
        raise ValueError(f"Cannot load mesh: {mesh_path}")
    verts_cv = np.asarray(mesh_template.vertices, dtype=np.float32)
    faces_np = np.asarray(mesh_template.faces, dtype=np.int64)

    # --- Load masks ---
    mask_paths = _list_mask_files(mask_dir)
    if not mask_paths:
        raise RuntimeError(f"No masks in: {mask_dir}")
    masks_np, h_mask, w_mask = _load_mask_stack(mask_paths, int(args.mask_threshold))

    # --- Load tracks / visibility ---
    tracks_raw = np.load(str(tracks_path))
    vis_raw = np.load(str(vis_path))
    tracks_nt2, vis_nt = _normalize_tracks_vis_with_mask_length(
        tracks_raw, vis_raw, int(masks_np.shape[0]),
    )

    if int(args.start_frame) != 0:
        raise ValueError("--start_frame must be 0 (mesh is aligned for frame 0).")

    t_total = min(masks_np.shape[0], tracks_nt2.shape[1], vis_nt.shape[1])
    t_use = min(t_total, int(args.end_frame) + 1) if args.end_frame >= 0 else t_total
    if t_use <= 0:
        raise RuntimeError("No frames after clipping.")

    masks_np = masks_np[:t_use]
    tracks_nt2 = tracks_nt2[:, :t_use].astype(np.float32)
    vis_nt = vis_nt[:, :t_use].astype(np.float32)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    # --- Map seed pixels to 3D mesh surface ---
    seed_uv = tracks_nt2[:, 0, :]
    mapping = _map_seed_points_to_mesh(
        seed_uv, verts_cv, faces_np, masks_np[0],
        fx, fy, cx, cy, h_mask, w_mask,
        device, int(args.bin_size), float(args.mask_gate_threshold),
    )
    valid = mapping.valid_seed_mask
    tracks_valid = tracks_nt2[valid]
    vis_valid = vis_nt[valid]
    x0_m3 = mapping.points_cv  # [M, 3] in OpenCV coords on GPU

    n_valid = tracks_valid.shape[0]
    print(f"  Seed mapping: {n_valid}/{seed_uv.shape[0]} valid "
          f"(dropped face={mapping.invalid_face_count}, "
          f"mask={mapping.outside_mask0_count}, "
          f"nan={mapping.nonfinite_seed_count})")

    if n_valid < int(args.min_valid_tracks):
        raise RuntimeError(
            f"Too few valid tracks: {n_valid} (need {args.min_valid_tracks})"
        )

    # --- Debug: verify frame-0 reprojection before optimization ---
    with torch.no_grad():
        z0 = x0_m3[:, 2]
        pred_u0 = fx * x0_m3[:, 0] / z0 + cx
        pred_v0 = fy * x0_m3[:, 1] / z0 + cy
        obs_u0 = torch.from_numpy(tracks_valid[:, 0, 0]).to(device)
        obs_v0 = torch.from_numpy(tracks_valid[:, 0, 1]).to(device)
        err0 = torch.sqrt((pred_u0 - obs_u0) ** 2 + (pred_v0 - obs_v0) ** 2)
        print(f"  Frame-0 reprojection sanity: mean={err0.mean():.2f}px "
              f"median={err0.median():.2f}px max={err0.max():.2f}px")

    # --- Move data to GPU ---
    obs_uv = torch.from_numpy(tracks_valid).to(device, torch.float32).permute(1, 0, 2)  # [T,M,2]
    vis_tm = torch.from_numpy(vis_valid).to(device, torch.float32).permute(1, 0)          # [T,M]
    masks_t = torch.from_numpy(masks_np).to(device, torch.float32)                        # [T,H,W]

    # --- PnP-RANSAC sequential initialization ---
    n_params = max(t_use - 1, 0)
    if not args.disable_pnp_init and n_params > 0:
        print(f"  [{object_slug}] PnP-RANSAC sequential initialization...")
        # tracks_valid is [M, T, 2]; PnP needs [T, M, 2]
        obs_uv_np_tm = tracks_valid.transpose(1, 0, 2)
        vis_np_tm = vis_valid.transpose(1, 0)
        rv_init, tr_init, pnp_info = _pnp_sequential_init(
            x0_m3.cpu().numpy(), obs_uv_np_tm, vis_np_tm,
            k, float(args.pnp_ransac_thresh),
        )
        pnp_max_trans = np.linalg.norm(tr_init, axis=1).max() if tr_init.shape[0] > 0 else 0.0
        pnp_max_rot = np.linalg.norm(rv_init, axis=1).max() if rv_init.shape[0] > 0 else 0.0
        pnp_ok_count = sum(1 for p in pnp_info if p["pnp_ok"])
        print(f"  [{object_slug}] PnP init: {pnp_ok_count}/{len(pnp_info)} frames OK, "
              f"max_trans={pnp_max_trans:.4f}m, max_rot={np.degrees(pnp_max_rot):.1f}deg")

        rotvecs_init = torch.from_numpy(rv_init).to(device, torch.float32)
        trans_init = torch.from_numpy(tr_init).to(device, torch.float32)

        # --- Outlier track rejection based on PnP poses ---
        if args.outlier_reproj_thresh_px > 0:
            outlier_mask = _identify_outlier_tracks(
                x0_m3, obs_uv, vis_tm, rotvecs_init, trans_init,
                fx, fy, cx, cy,
                float(args.outlier_reproj_thresh_px),
                float(args.outlier_max_fraction),
            )
            n_outlier = int(outlier_mask.sum().item())
            if n_outlier > 0:
                print(f"  [{object_slug}] Outlier rejection: {n_outlier}/{x0_m3.shape[0]} tracks removed")
                vis_tm[:, outlier_mask] = 0.0
    else:
        rotvecs_init = torch.zeros(n_params, 3, device=device)
        trans_init = torch.zeros(n_params, 3, device=device)

    # --- Optimisable parameters: initialized from PnP or zeros ---
    rotvecs = torch.nn.Parameter(rotvecs_init.clone())
    translations = torch.nn.Parameter(trans_init.clone())

    iter_rows: list[dict] = []
    best_total: float | None = None
    best_iter: int | None = None
    best_rotvecs: torch.Tensor | None = None
    best_trans: torch.Tensor | None = None
    early_stopped = False
    early_stop_iter: int | None = None

    def _loss_kwargs(huber_delta_override=None):
        return dict(
            fx=fx, fy=fy, cx=cx, cy=cy,
            lambda_img=float(args.lambda_img),
            lambda_a=float(args.lambda_a),
            lambda_v=float(args.lambda_v),
            huber_delta=huber_delta_override if huber_delta_override is not None else float(args.huber_delta_px),
            vis_threshold=float(args.visibility_threshold),
            mask_gate_threshold=float(args.mask_gate_threshold),
        )

    def _record(it: int, stage: str, b: LossBundle):
        active = b.weights > 0
        if active.any():
            reproj = torch.sqrt(b.r2[active].clamp(min=1e-12))
            mean_rp = float(reproj.mean().item())
        else:
            mean_rp = float("nan")
        ei = float(b.e_img.detach().item())
        es = float(b.e_smooth.detach().item())
        ev = float(b.e_vel.detach().item())
        ti = args.lambda_img * ei
        ts = args.lambda_a * es
        tv = args.lambda_v * ev
        # Pose magnitude summary
        with torch.no_grad():
            t_norms = b.T_mats[:, :3, 3].norm(dim=-1)
            max_trans = float(t_norms.max().item())
        iter_rows.append({
            "iter": it, "stage": stage,
            "total": float(b.total.detach().item()),
            "e_img": ei, "e_smooth": es, "e_vel": ev,
            "term_img": ti, "term_smooth_weighted": ts, "term_vel_weighted": tv,
            "total_from_terms": ti + ts + tv,
            "e_img_raw": float(b.e_img_raw.detach().item()),
            "e_smooth_raw": float(b.e_smooth_raw.detach().item()),
            "e_vel_raw": float(b.e_vel_raw.detach().item()),
            "e_img_denom": float(b.e_img_denom.detach().item()),
            "e_smooth_denom": float(b.e_smooth_denom.detach().item()),
            "e_vel_denom": float(b.e_vel_denom.detach().item()),
            "active_pairs": int(active.sum().item()),
            "sum_weight": float(b.weights.sum().item()),
            "mean_reproj_px": mean_rp,
            "max_translation": max_trans,
        })

    # ==================== Adam ====================
    if t_use > 1 and args.adam_iters > 0:
        opt = torch.optim.Adam([rotvecs, translations], lr=float(args.adam_lr))

        # LR scheduling
        scheduler = None
        if args.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.adam_iters, eta_min=float(args.adam_lr) / 20.0,
            )

        # Graduated Huber delta: start lenient (3x), anneal to target over first 50%
        huber_target = float(args.huber_delta_px)
        huber_start = huber_target * 3.0 if args.graduated_huber else huber_target

        for it in range(1, int(args.adam_iters) + 1):
            # Current Huber delta (graduated non-convexity)
            if args.graduated_huber:
                frac = min(it / max(args.adam_iters * 0.5, 1), 1.0)
                current_huber = huber_start + (huber_target - huber_start) * frac
            else:
                current_huber = huber_target

            opt.zero_grad(set_to_none=True)
            b = _compute_loss(
                rotvecs, translations, x0_m3, obs_uv, vis_tm, masks_t,
                **_loss_kwargs(current_huber),
            )
            if not torch.isfinite(b.total):
                raise RuntimeError(f"Non-finite loss at Adam iter {it}")
            b.total.backward()
            opt.step()

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                # --- Periodic outlier re-trimming ---
                if (
                    args.retrim_interval > 0
                    and it > 0
                    and it % args.retrim_interval == 0
                    and it < args.adam_iters  # don't retrim at the very end
                ):
                    T_mats_trim = _build_T_matrices(rotvecs, translations, t_use, device)
                    R_tr = T_mats_trim[:, :3, :3]
                    t_tr = T_mats_trim[:, :3, 3]
                    xt_tr = torch.einsum("tij,mj->tmi", R_tr, x0_m3) + t_tr.unsqueeze(1)
                    z_tr = xt_tr[..., 2].clamp(min=1e-6)
                    pred_u_tr = fx * xt_tr[..., 0] / z_tr + cx
                    pred_v_tr = fy * xt_tr[..., 1] / z_tr + cy
                    err_tr = torch.sqrt(
                        ((obs_uv[..., 0] - pred_u_tr) ** 2 +
                         (obs_uv[..., 1] - pred_v_tr) ** 2).clamp(min=1e-12)
                    )  # [T, M]
                    n_trimmed_total = 0
                    for ft in range(t_use):
                        active_ft = vis_tm[ft] > 0
                        n_active_ft = int(active_ft.sum().item())
                        if n_active_ft < 20:
                            continue
                        err_active = err_tr[ft, active_ft]
                        thresh = torch.quantile(err_active, args.retrim_percentile / 100.0)
                        to_zero = active_ft & (err_tr[ft] > thresh)
                        n_zero = int(to_zero.sum().item())
                        if n_zero > 0:
                            vis_tm[ft, to_zero] = 0.0
                            n_trimmed_total += n_zero
                    if n_trimmed_total > 0:
                        print(f"  [{object_slug}] retrim@{it}: zeroed {n_trimmed_total} track-frame pairs "
                              f"(p{args.retrim_percentile:.0f} threshold)")

                # Evaluate with TARGET huber (not graduated) for fair comparison
                b_eval = _compute_loss(
                    rotvecs, translations, x0_m3, obs_uv, vis_tm, masks_t,
                    **_loss_kwargs(),
                )
                cur = float(b_eval.total.item())
                improved = False
                if best_total is None or not np.isfinite(best_total):
                    improved = True
                else:
                    rel = (best_total - cur) / max(abs(best_total), 1e-12)
                    improved = rel >= float(args.early_stop_rel_min_delta)
                if improved:
                    best_total = cur
                    best_iter = it
                    best_rotvecs = rotvecs.detach().clone()
                    best_trans = translations.detach().clone()

                if args.debug_save_interval <= 1 or it % args.debug_save_interval == 0 or it == args.adam_iters:
                    _record(it, "adam", b_eval)

                if args.log_every > 0 and (it % args.log_every == 0 or it == args.adam_iters):
                    t_norms = b_eval.T_mats[:, :3, 3].norm(dim=-1)
                    cur_lr = scheduler.get_last_lr()[0] if scheduler else float(args.adam_lr)
                    print(
                        f"  [{object_slug}] adam {it:05d}  "
                        f"total={b_eval.total.item():.6f}  "
                        f"img={args.lambda_img * b_eval.e_img.item():.6f}  "
                        f"smooth={args.lambda_a * b_eval.e_smooth.item():.6f}  "
                        f"vel={args.lambda_v * b_eval.e_vel.item():.6f}  "
                        f"max_t={t_norms.max().item():.5f}  "
                        f"lr={cur_lr:.6f}  huber={current_huber:.2f}"
                    )

                if (
                    args.early_stop_patience > 0
                    and it >= args.early_stop_min_iter
                    and best_iter is not None
                    and it - best_iter >= args.early_stop_patience
                ):
                    early_stopped = True
                    early_stop_iter = it
                    print(f"  [{object_slug}] early stop at iter {it} "
                          f"(best={best_iter}, loss={best_total:.6f})")
                    break

        if early_stopped and best_rotvecs is not None:
            with torch.no_grad():
                rotvecs.copy_(best_rotvecs)
                translations.copy_(best_trans)

    # ==================== L-BFGS refinement ====================
    if t_use > 1 and not args.disable_lbfgs and args.lbfgs_iters > 0 and not early_stopped:
        lbfgs = torch.optim.LBFGS(
            [rotvecs, translations],
            lr=float(args.lbfgs_lr),
            max_iter=int(args.lbfgs_iters),
            line_search_fn="strong_wolfe",
        )
        n_closure = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            b = _compute_loss(
                rotvecs, translations, x0_m3, obs_uv, vis_tm, masks_t,
                **_loss_kwargs(),
            )
            if not torch.isfinite(b.total):
                raise RuntimeError("Non-finite loss in L-BFGS")
            b.total.backward()
            n_closure["n"] += 1
            return b.total

        lbfgs.step(closure)
        with torch.no_grad():
            b_lbfgs = _compute_loss(
                rotvecs, translations, x0_m3, obs_uv, vis_tm, masks_t,
                **_loss_kwargs(),
            )
        _record(
            (iter_rows[-1]["iter"] + 1) if iter_rows else 1,
            "lbfgs_final", b_lbfgs,
        )
        print(f"  [{object_slug}] lbfgs closures={n_closure['n']} "
              f"total={b_lbfgs.total.item():.6f}")
    elif early_stopped:
        print(f"  [{object_slug}] skipping L-BFGS (early stop)")

    # ==================== Final evaluation ====================
    with torch.no_grad():
        final = _compute_loss(
            rotvecs, translations, x0_m3, obs_uv, vis_tm, masks_t,
            **_loss_kwargs(),
        )
    _record(
        (iter_rows[-1]["iter"] + 1) if iter_rows else 0,
        "final", final,
    )
    T_mats_np = final.T_mats.detach().cpu().numpy().astype(np.float32)

    # --- Debug: print pose summary ---
    print(f"\n  [{object_slug}] Final pose summary:")
    for i in [0, 1, t_use // 2, t_use - 1]:
        if 0 <= i < t_use:
            t_vec = T_mats_np[i, :3, 3]
            R_mat = T_mats_np[i, :3, :3]
            angle = np.arccos(np.clip((np.trace(R_mat) - 1) / 2, -1, 1))
            print(f"    frame {i}: t=[{t_vec[0]:.5f}, {t_vec[1]:.5f}, {t_vec[2]:.5f}]  "
                  f"rot={np.degrees(angle):.2f}deg")

    # --- Save outputs ---
    _save_pose_json(out_dir / "poses.json", T_mats_np, args.start_frame)
    _save_mesh_sequence(
        mesh_template, verts_cv, T_mats_np, meshes_dir,
        str(args.output_coord), args.start_frame,
    )

    overlay_ok, overlay_msg = _render_overlays(
        frame_paths, verts_cv, faces_np, T_mats_np, k,
        out_dir, args.start_frame, float(args.overlay_fps),
    )

    # --- Save metrics ---
    _save_csv(debug_dir / "iter_metrics.csv", iter_rows)
    frame_rows = _build_frame_metrics(args.start_frame, final, float(args.mask_gate_threshold))
    _save_csv(debug_dir / "frame_metrics.csv", frame_rows)
    loss_pngs = _save_loss_plots(debug_dir, iter_rows)
    _save_pose_debug_plots(debug_dir, frame_rows)

    # --- Frame-0 correspondence vis ---
    if frame_paths and len(frame_paths) > args.start_frame:
        frame0_img = cv2.imread(str(frame_paths[args.start_frame]))
        if frame0_img is None:
            frame0_img = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)
    else:
        frame0_img = np.zeros((h_mask, w_mask, 3), dtype=np.uint8)

    obs0 = final.obs_uv[0].cpu().numpy()
    pred0 = final.pred_uv[0].cpu().numpy()
    corr = _draw_frame0_correspondence(frame0_img, obs0, pred0, max_points=2000)
    cv2.putText(corr, f"{object_slug}: frame-0 correspondences",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / "frame0_correspondence.png"), corr)

    # --- Build summary dict ---
    elapsed = time.time() - t0
    active_mask = final.weights > 0
    final_active = int(active_mask.sum().item())
    mean_reproj = float("nan")
    if final_active > 0:
        mean_reproj = float(torch.sqrt(final.r2[active_mask].clamp(min=0)).mean().item())

    return {
        "object_name": object_name,
        "slug": object_slug,
        "status": "processed",
        "mesh_path": str(mesh_path),
        "tracks_path": str(tracks_path),
        "visibility_path": str(vis_path),
        "mask_dir": str(mask_dir),
        "num_input_tracks": int(tracks_nt2.shape[0]),
        "num_valid_seed_tracks": n_valid,
        "num_dropped_invalid_face": mapping.invalid_face_count,
        "num_dropped_outside_mask0": mapping.outside_mask0_count,
        "num_dropped_nonfinite_seed": mapping.nonfinite_seed_count,
        "num_frames": t_use,
        "huber_delta_px": float(args.huber_delta_px),
        "lambda_img": float(args.lambda_img),
        "lambda_a": float(args.lambda_a),
        "lambda_v": float(args.lambda_v),
        "adam_iters": int(args.adam_iters),
        "early_stop_patience": int(args.early_stop_patience),
        "early_stop_rel_min_delta": float(args.early_stop_rel_min_delta),
        "early_stop_min_iter": int(args.early_stop_min_iter),
        "early_stopped": early_stopped,
        "early_stop_iter": early_stop_iter,
        "best_iter": best_iter,
        "best_total_loss": best_total,
        "lbfgs_enabled": not args.disable_lbfgs,
        "lbfgs_iters": int(args.lbfgs_iters),
        "final_total_loss": float(final.total.item()),
        "final_e_img": float(final.e_img.item()),
        "final_e_smooth": float(final.e_smooth.item()),
        "final_e_vel": float(final.e_vel.item()),
        "final_e_img_raw": float(final.e_img_raw.item()),
        "final_e_smooth_raw": float(final.e_smooth_raw.item()),
        "final_e_vel_raw": float(final.e_vel_raw.item()),
        "final_e_img_denom": float(final.e_img_denom.item()),
        "final_e_smooth_denom": float(final.e_smooth_denom.item()),
        "final_e_vel_denom": float(final.e_vel_denom.item()),
        "final_term_img": float(args.lambda_img) * float(final.e_img.item()),
        "final_term_smooth_weighted": float(args.lambda_a) * float(final.e_smooth.item()),
        "final_term_vel_weighted": float(args.lambda_v) * float(final.e_vel.item()),
        "final_active_pairs": final_active,
        "final_sum_weight": float(final.weights.sum().item()),
        "final_mean_reproj_px": mean_reproj,
        "overlay_rendered": overlay_ok,
        "overlay_message": overlay_msg,
        "iter_metrics_csv": str(debug_dir / "iter_metrics.csv"),
        "frame_metrics_csv": str(debug_dir / "frame_metrics.csv"),
        "loss_curve_pngs": loss_pngs,
        "elapsed_seconds": elapsed,
        "convention_notes": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "T_4x4_convention": "standard column-vector: [[R,t],[0,1]], p'=R@p+t",
            "rotation_param": "axis-angle via pytorch3d.axis_angle_to_matrix (standard convention)",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Validate args
    for name, val, lo in [
        ("early_stop_patience", args.early_stop_patience, 0),
        ("early_stop_min_iter", args.early_stop_min_iter, 0),
    ]:
        if val < lo:
            raise ValueError(f"--{name} must be >= {lo}")
    for name in ("lambda_img", "lambda_a", "lambda_v", "early_stop_rel_min_delta"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")

    script_dir = Path(__file__).resolve().parent

    cotracker_dir, aligned_dir, segment_dir, output_root = _resolve_default_dirs(args, script_dir)
    pag_path = _resolve_pag_path(args, script_dir)
    k, intr_path = _load_intrinsics_from_alignment_summary(aligned_dir)
    out_video_dir = (output_root / args.video_name).resolve()
    ensure_dir(out_video_dir)

    for label, d in [
        ("CoTracker", cotracker_dir),
        ("Aligned", aligned_dir),
        ("Segment", segment_dir),
    ]:
        if not d.exists():
            raise NotADirectoryError(f"{label} dir missing: {d}")

    device = _to_device(args.device)
    pag_objects = _load_pag_objects_from_states_only(pag_path)

    frames_dir = _resolve_frames_dir(cotracker_dir, segment_dir)
    frame_paths: list[Path] = list_images(frames_dir) if frames_dir else []
    if not frame_paths:
        print("[WARN] No frames directory found.")

    # --- Print important info ---
    print("=" * 60)
    print("track_object_mesh_claude.py — corrected SE(3) tracker")
    print(f"  video:  {args.video_name}")
    print(f"  device: {device}")
    print(f"  K:      fx={k[0,0]:.1f}  fy={k[1,1]:.1f}  cx={k[0,2]:.1f}  cy={k[1,2]:.1f}")
    print(f"  pag:    {pag_path.name} ({len(pag_objects)} objects)")
    print("  convention: OpenCV coords, standard 4x4 [[R,t],[0,1]]")
    print("=" * 60)

    summary: dict[str, Any] = {
        "video_name": args.video_name,
        "status": "completed",
        "script": "track_object_mesh_claude.py",
        "inputs": {
            "cotracker_video_dir": str(cotracker_dir),
            "aligned_mesh_video_dir": str(aligned_dir),
            "segment_video_dir": str(segment_dir),
            "pag_file": str(pag_path),
            "intrinsics_source": str(intr_path),
            "frames_dir": str(frames_dir) if frames_dir else None,
        },
        "optimization_settings": {
            "huber_delta_px": float(args.huber_delta_px),
            "lambda_img": float(args.lambda_img),
            "lambda_a": float(args.lambda_a),
            "lambda_v": float(args.lambda_v),
            "adam_iters": int(args.adam_iters),
            "adam_lr": float(args.adam_lr),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_rel_min_delta": float(args.early_stop_rel_min_delta),
            "early_stop_min_iter": int(args.early_stop_min_iter),
            "lbfgs_enabled": not args.disable_lbfgs,
            "lbfgs_iters": int(args.lbfgs_iters),
            "lbfgs_lr": float(args.lbfgs_lr),
            "pnp_init_enabled": not args.disable_pnp_init,
            "pnp_ransac_thresh": float(args.pnp_ransac_thresh),
            "outlier_reproj_thresh_px": float(args.outlier_reproj_thresh_px),
            "outlier_max_fraction": float(args.outlier_max_fraction),
            "lr_schedule": args.lr_schedule,
            "graduated_huber": args.graduated_huber,
        },
        "conventions": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "T_4x4": "standard column-vector [[R,t],[0,1]], p_world = R @ p_local + t",
            "rotation": "axis_angle_to_matrix (standard, R @ p)",
            "projection": "u = fx*X/Z + cx,  v = fy*Y/Z + cy",
            "bugs_fixed": [
                "PyTorch3D se3_exp_map row-vector convention → translation was always zero",
                "Rotation was transposed (R^T instead of R)",
                "Body-frame velocity/smoothness now computed properly",
            ],
        },
        "device": str(device),
        "output_dir": str(out_video_dir),
        "objects_from_pag_states": [
            {"name": n, "slug": s} for n, s in pag_objects
        ],
        "objects_processed": [],
        "objects_skipped": [],
        "objects_failed": [],
    }

    for obj_name, obj_slug in pag_objects:
        mesh_path = aligned_dir / "meshes" / f"{obj_slug}.ply"
        tracks_path = cotracker_dir / obj_slug / "tracks.npy"
        vis_path = cotracker_dir / obj_slug / "visibility.npy"
        mask_dir = _resolve_object_mask_dir(segment_dir, obj_slug)
        obj_out = out_video_dir / obj_slug

        missing = []
        if not mesh_path.exists():
            missing.append(f"mesh: {mesh_path}")
        if not tracks_path.exists():
            missing.append(f"tracks: {tracks_path}")
        if not vis_path.exists():
            missing.append(f"vis: {vis_path}")
        if not mask_dir.exists():
            missing.append(f"masks: {mask_dir}")
        if missing:
            reason = "; ".join(missing)
            print(f"\n[SKIP] {obj_slug}: {reason}")
            summary["objects_skipped"].append(
                {"name": obj_name, "slug": obj_slug, "reason": reason}
            )
            continue

        print(f"\n{'─' * 50}")
        print(f"[OBJECT] {obj_name} ({obj_slug})")
        print(f"{'─' * 50}")
        try:
            obj_summary = _run_single_object(
                obj_name, obj_slug, mesh_path, tracks_path, vis_path,
                mask_dir, frame_paths, k, args, obj_out, device,
            )
            summary["objects_processed"].append(obj_summary)
            print(f"[OK] {obj_slug} → {obj_out}")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {obj_slug}: {reason}")
            import traceback
            traceback.print_exc()
            summary["objects_failed"].append(
                {"name": obj_name, "slug": obj_slug, "reason": reason}
            )

    with (out_video_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. Summary: {out_video_dir / 'run_summary.json'}")
    print(f"  processed: {len(summary['objects_processed'])}")
    print(f"  skipped:   {len(summary['objects_skipped'])}")
    print(f"  failed:    {len(summary['objects_failed'])}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
