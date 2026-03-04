"""Mesh-to-depth alignment using bidirectional chamfer losses on a fixed visible subset.

Overview
--------
This script aligns human/object meshes to frame_00 depth in camera coordinates
with bidirectional chamfer terms in 3D and 2D:
- 3D: observed depth points <-> transformed mesh surface samples
- 2D: observed mask pixels <-> projected transformed mesh surface samples

Compared to the base chamfer script, visibility is computed once at initialization:
1. Sample mesh points on surface (triangle-area weighted).
2. Render once at init pose and keep only visible sampled points as fixed subset.
3. Optimize on that fixed subset for all subsequent iterations.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt  # noqa: E402
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes

from utils_align_meshes import (
    F_CV_TO_P3D,
    F_P3D_TO_CV,
    MeshAsset,
    build_cameras,
    colorize_points_by_xyz,
    erode_mask,
    ensure_3x3_intrinsics,
    find_first_human_ply,
    load_binary_mask,
    load_json,
    load_mesh,
    load_object_intrinsics,
    maybe_resize_for_optimization,
    parse_device,
    render_quality_overlay_from_cv_meshes,
    resolve_frame_0000_mask,
    resolve_path,
    save_colored_point_cloud,
    save_mesh_ply,
    slugify,
)


@dataclass
class ObservationSet:
    obs_points_3d: np.ndarray  # (N3,3)
    obs_pixels_2d: np.ndarray  # (N2,2), (u,v)
    mask_pixels_total: int
    depth_valid_pixels_total: int
    obs_3d_used: int
    obs_2d_used: int


@dataclass
class OptimizationResultChamfer:
    status: str
    message: str | None
    obs_3d_count: int
    obs_2d_count: int
    model_sample_count: int
    model_sample_count_total: int
    model_sample_count_visible_fixed: int
    model_sample_visible_ratio: float
    visible_subset_mode: str
    scale: float
    log_scale: float
    tx_init: float
    ty_init: float
    tz_init: float
    delta_tx: float
    delta_ty: float
    delta_tz: float
    tx: float
    ty: float
    tz: float
    history: dict[str, list[float]]
    final_total_loss: float | None
    final_cd3d_loss: float | None
    final_cd2d_loss: float | None
    resumed: bool
    resume_source: str
    start_iter: int
    end_iter: int
    iters_executed_this_run: int
    early_stopped: bool
    early_stop_iter: int | None
    best_iter: int | None
    best_total: float | None
    checkpoint_path: str | None


@dataclass
class ResumeStateChamfer:
    signature: dict[str, Any]
    start_iter: int
    history: dict[str, list[float]]
    log_scale: float
    delta_tx: float
    delta_ty: float
    delta_tz: float
    tx_init: float
    ty_init: float
    tz_init: float
    best_total: float | None
    best_iter: int | None
    best_log_scale: float | None
    best_delta_tx: float | None
    best_delta_ty: float | None
    best_delta_tz: float | None
    visible_subset_indices: np.ndarray
    visible_subset_mode: str
    visible_subset_z_abs_tol_m: float
    visible_subset_z_rel_tol: float
    min_visible_subset_points_per_mesh: int
    visible_subset_focal_scale: float
    optimizer_state_dict: dict[str, Any] | None


@dataclass
class FixedVisibleSubset:
    indices: np.ndarray
    mode: str
    strict_visible_count: int
    in_frame_positive_count: int
    positive_depth_count: int


def geman_mcclure_func(residual: torch.Tensor, rho: float) -> torch.Tensor:
    r2 = residual * residual
    rho2 = float(rho) * float(rho)
    dist = torch.div(r2, r2 + rho2)
    return rho2 * dist


def sample_mesh_surface_points(
    verts_cv: np.ndarray,
    faces: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample mesh surface points in OpenCV camera coordinates."""
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0")
    if verts_cv.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    tri = verts_cv[faces]  # (F,3,3)
    v0 = tri[:, 0, :]
    v1 = tri[:, 1, :]
    v2 = tri[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    rng = np.random.default_rng(seed)
    area_sum = float(np.sum(areas))
    if not np.isfinite(area_sum) or area_sum <= 1e-12:
        # Degenerate fallback: sample vertices.
        vidx = rng.choice(verts_cv.shape[0], size=int(n_samples), replace=True)
        return verts_cv[vidx].astype(np.float32)

    probs = areas / area_sum
    fidx = rng.choice(faces.shape[0], size=int(n_samples), replace=True, p=probs)
    tri_sel = tri[fidx]  # (N,3,3)

    r1 = rng.random(int(n_samples), dtype=np.float32)
    r2 = rng.random(int(n_samples), dtype=np.float32)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2

    pts = (
        w0[:, None] * tri_sel[:, 0, :]
        + w1[:, None] * tri_sel[:, 1, :]
        + w2[:, None] * tri_sel[:, 2, :]
    )
    return pts.astype(np.float32)


def build_fixed_visible_subset(
    verts_cv: np.ndarray,
    faces: np.ndarray,
    sampled_points_base: np.ndarray,
    intrinsics: np.ndarray,
    image_h: int,
    image_w: int,
    device: torch.device,
    scale_init: float,
    tx_init: float,
    ty_init: float,
    tz_init: float,
    z_abs_tol_m: float,
    z_rel_tol: float,
    min_visible_subset_points_per_mesh: int,
    focal_scale_for_visibility: float,
) -> FixedVisibleSubset:
    n_samples = int(sampled_points_base.shape[0])
    if n_samples == 0:
        return FixedVisibleSubset(
            indices=np.zeros((0,), dtype=np.int64),
            mode="no_samples",
            strict_visible_count=0,
            in_frame_positive_count=0,
            positive_depth_count=0,
        )

    verts_init = (float(scale_init) * verts_cv).astype(np.float32)
    verts_init[:, 0] += float(tx_init)
    verts_init[:, 1] += float(ty_init)
    verts_init[:, 2] += float(tz_init)

    sampled_init = (float(scale_init) * sampled_points_base).astype(np.float32)
    sampled_init[:, 0] += float(tx_init)
    sampled_init[:, 1] += float(ty_init)
    sampled_init[:, 2] += float(tz_init)

    k_vis = intrinsics.astype(np.float32).copy()
    focal_scale = float(focal_scale_for_visibility)
    k_vis[0, 0] *= focal_scale
    k_vis[1, 1] *= focal_scale

    cams = build_cameras(k_vis, image_w, image_h, device)
    raster_settings = RasterizationSettings(
        image_size=(image_h, image_w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=300000,
    )
    rasterizer = MeshRasterizer(cameras=cams, raster_settings=raster_settings)

    verts_t = torch.from_numpy(verts_init).to(device=device, dtype=torch.float32)
    cv_to_p3d = torch.from_numpy(F_CV_TO_P3D).to(device=device, dtype=torch.float32)
    verts_p3d = verts_t @ cv_to_p3d.transpose(0, 1)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])
    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    zbuf = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)

    fx = float(k_vis[0, 0])
    fy = float(k_vis[1, 1])
    cx = float(k_vis[0, 2])
    cy = float(k_vis[1, 2])

    z = sampled_init[:, 2]
    valid_depth = np.isfinite(z) & (z > 1e-6)

    u = np.zeros((n_samples,), dtype=np.float32)
    v = np.zeros((n_samples,), dtype=np.float32)
    if np.any(valid_depth):
        zv = z[valid_depth]
        u[valid_depth] = fx * sampled_init[valid_depth, 0] / zv + cx - 0.5
        v[valid_depth] = fy * sampled_init[valid_depth, 1] / zv + cy - 0.5

    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    in_frame = (
        valid_depth
        & (ui >= 0)
        & (ui < int(image_w))
        & (vi >= 0)
        & (vi < int(image_h))
    )

    render_valid = np.zeros((n_samples,), dtype=bool)
    if np.any(in_frame):
        ii = np.nonzero(in_frame)[0]
        pix = pix_to_face[vi[ii], ui[ii]]
        zr = zbuf[vi[ii], ui[ii]]
        render_valid_i = (pix >= 0) & np.isfinite(zr) & (zr > 1e-6)
        render_valid[ii] = render_valid_i

    strict_visible = np.zeros((n_samples,), dtype=bool)
    if np.any(render_valid):
        ii = np.nonzero(render_valid)[0]
        zr = zbuf[vi[ii], ui[ii]]
        ztol = float(z_abs_tol_m) + float(z_rel_tol) * np.maximum(zr, 1e-6)
        strict_visible[ii] = np.abs(z[ii] - zr) <= ztol

    strict_idx = np.nonzero(strict_visible)[0].astype(np.int64)
    in_frame_positive_idx = np.nonzero(in_frame)[0].astype(np.int64)
    positive_depth_idx = np.nonzero(valid_depth)[0].astype(np.int64)

    min_count = int(min_visible_subset_points_per_mesh)
    if strict_idx.shape[0] >= min_count:
        idx = strict_idx
        mode = "strict_visible"
    elif in_frame_positive_idx.shape[0] >= min_count:
        idx = in_frame_positive_idx
        mode = "fallback_in_frame_positive_depth"
    else:
        idx = positive_depth_idx
        mode = "fallback_positive_depth"

    return FixedVisibleSubset(
        indices=idx.astype(np.int64),
        mode=mode,
        strict_visible_count=int(strict_idx.shape[0]),
        in_frame_positive_count=int(in_frame_positive_idx.shape[0]),
        positive_depth_count=int(positive_depth_idx.shape[0]),
    )


def colorize_points_by_xy(points_2d: np.ndarray) -> np.ndarray:
    if points_2d.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    pmin = points_2d.min(axis=0)
    pmax = points_2d.max(axis=0)
    denom = np.maximum(pmax - pmin, 1e-8)
    norm = (points_2d - pmin) / denom
    r = (255.0 * norm[:, 0]).clip(0.0, 255.0)
    g = (255.0 * norm[:, 1]).clip(0.0, 255.0)
    b = (255.0 * (1.0 - 0.5 * (norm[:, 0] + norm[:, 1]))).clip(0.0, 255.0)
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def build_observation_set(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray | None,
    max_obs_3d_points: int,
    max_obs_2d_points: int,
    seed: int,
) -> ObservationSet:
    if mask is None:
        return ObservationSet(
            obs_points_3d=np.zeros((0, 3), dtype=np.float32),
            obs_pixels_2d=np.zeros((0, 2), dtype=np.float32),
            mask_pixels_total=0,
            depth_valid_pixels_total=0,
            obs_3d_used=0,
            obs_2d_used=0,
        )

    mask_bool = mask > 0.5
    ys_mask, xs_mask = np.nonzero(mask_bool)
    n_mask_total = int(xs_mask.shape[0])
    if n_mask_total == 0:
        return ObservationSet(
            obs_points_3d=np.zeros((0, 3), dtype=np.float32),
            obs_pixels_2d=np.zeros((0, 2), dtype=np.float32),
            mask_pixels_total=0,
            depth_valid_pixels_total=0,
            obs_3d_used=0,
            obs_2d_used=0,
        )

    rng = np.random.default_rng(seed)

    # O2D: all mask pixels (with optional cap).
    idx2d = np.arange(n_mask_total, dtype=np.int64)
    if max_obs_2d_points > 0 and n_mask_total > int(max_obs_2d_points):
        idx2d = rng.choice(n_mask_total, size=int(max_obs_2d_points), replace=False)
    xs2d = xs_mask[idx2d].astype(np.float32)
    ys2d = ys_mask[idx2d].astype(np.float32)
    obs2d = np.stack([xs2d, ys2d], axis=1).astype(np.float32)

    # O3D: depth-valid pixels within mask, backprojected.
    z_all = depth[ys_mask, xs_mask]
    valid_depth = np.isfinite(z_all) & (z_all > 0.0)
    ys3_all = ys_mask[valid_depth]
    xs3_all = xs_mask[valid_depth]
    z3_all = z_all[valid_depth].astype(np.float32)
    n_depth_valid_total = int(z3_all.shape[0])

    if n_depth_valid_total > 0:
        idx3d = np.arange(n_depth_valid_total, dtype=np.int64)
        if max_obs_3d_points > 0 and n_depth_valid_total > int(max_obs_3d_points):
            idx3d = rng.choice(
                n_depth_valid_total, size=int(max_obs_3d_points), replace=False
            )
        xs3 = xs3_all[idx3d].astype(np.float32)
        ys3 = ys3_all[idx3d].astype(np.float32)
        z3 = z3_all[idx3d]

        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        x3 = ((xs3 - cx) / fx) * z3
        y3 = ((ys3 - cy) / fy) * z3
        obs3d = np.stack([x3, y3, z3], axis=1).astype(np.float32)
    else:
        obs3d = np.zeros((0, 3), dtype=np.float32)

    return ObservationSet(
        obs_points_3d=obs3d,
        obs_pixels_2d=obs2d,
        mask_pixels_total=n_mask_total,
        depth_valid_pixels_total=n_depth_valid_total,
        obs_3d_used=int(obs3d.shape[0]),
        obs_2d_used=int(obs2d.shape[0]),
    )


def pairwise_squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (Na,D), b: (Nb,D) -> (Na,Nb)
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)
    d2 = a2 + b2 - 2.0 * (a @ b.transpose(0, 1))
    return torch.clamp(d2, min=0.0)


def min_distances_obs_to_model_chunked(
    obs: torch.Tensor,
    model: torch.Tensor,
    chunk_size: int,
    return_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute min_{m in model} ||obs - m||^2 for each obs point with chunking."""
    if obs.ndim != 2 or model.ndim != 2:
        raise ValueError("obs/model must be 2D tensors")
    if obs.shape[0] == 0:
        empty = torch.zeros((0,), device=obs.device, dtype=obs.dtype)
        return empty, (empty.long() if return_indices else None)
    if model.shape[0] == 0:
        infv = torch.full((obs.shape[0],), float("inf"), device=obs.device, dtype=obs.dtype)
        return infv, (torch.full((obs.shape[0],), -1, device=obs.device, dtype=torch.long) if return_indices else None)

    csz = max(1, int(chunk_size))
    best_d2_chunks: list[torch.Tensor] = []
    best_idx_chunks: list[torch.Tensor] = []

    for o0 in range(0, obs.shape[0], csz):
        o1 = min(obs.shape[0], o0 + csz)
        och = obs[o0:o1]

        best_d2: torch.Tensor | None = None
        best_idx: torch.Tensor | None = None
        for m0 in range(0, model.shape[0], csz):
            m1 = min(model.shape[0], m0 + csz)
            mch = model[m0:m1]

            d2 = pairwise_squared_l2(och, mch)
            cur_d2, cur_idx_local = torch.min(d2, dim=1)
            cur_idx_global = cur_idx_local + int(m0)

            if best_d2 is None:
                best_d2 = cur_d2
                if return_indices:
                    best_idx = cur_idx_global
            else:
                better = cur_d2 < best_d2
                best_d2 = torch.where(better, cur_d2, best_d2)
                if return_indices:
                    assert best_idx is not None
                    best_idx = torch.where(better, cur_idx_global, best_idx)

        assert best_d2 is not None
        best_d2_chunks.append(best_d2)
        if return_indices:
            assert best_idx is not None
            best_idx_chunks.append(best_idx)

    all_d2 = torch.cat(best_d2_chunks, dim=0)
    if return_indices:
        all_idx = torch.cat(best_idx_chunks, dim=0)
    else:
        all_idx = None
    return all_d2, all_idx


def draw_points_uniform(
    canvas_bgr: np.ndarray,
    uv: np.ndarray,
    color_bgr: tuple[int, int, int],
    radius: int,
    max_points: int,
) -> None:
    h, w = canvas_bgr.shape[:2]
    if uv.shape[0] == 0:
        return
    if max_points > 0 and uv.shape[0] > max_points:
        keep = np.linspace(0, uv.shape[0] - 1, num=max_points, dtype=np.int64)
        pts = uv[keep]
    else:
        pts = uv

    for u, v in pts:
        x = int(round(float(u)))
        y = int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas_bgr, (x, y), int(radius), color_bgr, -1)


def draw_points_per_point_color(
    canvas_bgr: np.ndarray,
    uv: np.ndarray,
    colors_rgb: np.ndarray,
    radius: int,
    max_points: int,
) -> None:
    h, w = canvas_bgr.shape[:2]
    if uv.shape[0] == 0 or colors_rgb.shape[0] == 0:
        return

    n_points = min(uv.shape[0], colors_rgb.shape[0])
    if max_points > 0 and n_points > max_points:
        keep = np.linspace(0, n_points - 1, num=max_points, dtype=np.int64)
        pts = uv[keep]
        cols = colors_rgb[keep]
    else:
        pts = uv[:n_points]
        cols = colors_rgb[:n_points]

    for (u, v), rgb in zip(pts, cols):
        x = int(round(float(u)))
        y = int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.circle(canvas_bgr, (x, y), int(radius), bgr, -1)


def draw_nn_edges(
    canvas_bgr: np.ndarray,
    obs2d: np.ndarray,
    model2d: np.ndarray,
    nn_idx: np.ndarray,
    nn_d2: np.ndarray,
    max_lines: int,
    seed: int,
) -> None:
    if obs2d.shape[0] == 0 or model2d.shape[0] == 0 or nn_idx.shape[0] == 0:
        return

    valid = (
        np.isfinite(obs2d).all(axis=1)
        & np.isfinite(nn_d2)
        & (nn_idx >= 0)
        & (nn_idx < model2d.shape[0])
    )
    if not np.any(valid):
        return

    idx = np.nonzero(valid)[0]
    if max_lines > 0 and idx.shape[0] > max_lines:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=int(max_lines), replace=False)

    obs_sel = obs2d[idx]
    model_sel = model2d[nn_idx[idx]]

    h, w = canvas_bgr.shape[:2]
    for (uo, vo), (um, vm) in zip(obs_sel, model_sel):
        xo = int(round(float(uo)))
        yo = int(round(float(vo)))
        xm = int(round(float(um)))
        ym = int(round(float(vm)))

        if 0 <= xo < w and 0 <= yo < h and 0 <= xm < w and 0 <= ym < h:
            cv2.line(canvas_bgr, (xo, yo), (xm, ym), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas_bgr, (xo, yo), 1, (0, 255, 0), -1)
            cv2.circle(canvas_bgr, (xm, ym), 1, (0, 0, 255), -1)


def save_distance_histogram(
    out_path: Path,
    d2_3d: np.ndarray,
    d2_2d: np.ndarray,
    mesh_name: str,
    iter_idx: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    ax1 = plt.subplot(1, 2, 1)
    if d2_3d.size > 0:
        d3 = np.sqrt(np.maximum(d2_3d.astype(np.float64), 0.0))
        ax1.hist(d3, bins=50, color="#1f77b4", alpha=0.85)
        ax1.set_xlabel("3D NN distance (m)")
    else:
        ax1.text(0.5, 0.5, "No 3D observations", ha="center", va="center")
        ax1.set_xticks([])
        ax1.set_yticks([])
    ax1.set_ylabel("Count")
    ax1.set_title("Obs3D -> Model3D")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    if d2_2d.size > 0:
        d2 = np.sqrt(np.maximum(d2_2d.astype(np.float64), 0.0))
        ax2.hist(d2, bins=50, color="#ff7f0e", alpha=0.85)
        ax2.set_xlabel("2D NN distance (px)")
    else:
        ax2.text(0.5, 0.5, "No 2D observations", ha="center", va="center")
        ax2.set_xticks([])
        ax2.set_yticks([])
    ax2.set_ylabel("Count")
    ax2.set_title("Obs2D -> Model2D")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{mesh_name} - Iter {iter_idx:05d}")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def save_chamfer_debug_snapshot(
    out_dir: Path,
    mesh_name: str,
    iter_idx: int,
    frame_bgr: np.ndarray,
    obs3d: np.ndarray,
    obs3d_colors: np.ndarray,
    obs2d: np.ndarray,
    obs2d_anchor_colors_rgb: np.ndarray,
    model3d_all: np.ndarray,
    model3d_all_colors: np.ndarray,
    model2d_visible: np.ndarray,
    model2d_visible_colors_rgb: np.ndarray,
    nn2d_idx: np.ndarray,
    nn2d_d2: np.ndarray,
    nn3d_d2: np.ndarray,
    visible_subset_mode: str,
    visible_subset_count: int,
    total_model_samples: int,
    point_radius: int,
    max_points_vis: int,
    max_nn_lines: int,
    seed: int,
    save_obs3d_fixed: bool,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if save_obs3d_fixed:
        obs3d_path = out_dir / f"iter_{iter_idx:05d}_obs3d_fixed.ply"
        save_colored_point_cloud(obs3d_path, obs3d, obs3d_colors)
        paths["obs3d_fixed_ply"] = str(obs3d_path)

    model3d_path = out_dir / f"iter_{iter_idx:05d}_model3d_transformed.ply"
    save_colored_point_cloud(model3d_path, model3d_all, model3d_all_colors)
    paths["model3d_transformed_ply"] = str(model3d_path)

    left = frame_bgr.copy()
    right = frame_bgr.copy()
    draw_points_per_point_color(
        canvas_bgr=left,
        uv=obs2d,
        colors_rgb=obs2d_anchor_colors_rgb,
        radius=point_radius,
        max_points=max_points_vis,
    )
    draw_points_per_point_color(
        canvas_bgr=right,
        uv=model2d_visible,
        colors_rgb=model2d_visible_colors_rgb,
        radius=point_radius,
        max_points=max_points_vis,
    )

    cv2.putText(
        left,
        "Observed mask pixels (O2D, fixed colors)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        "Projected visible subset (colored only)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        left,
        f"N={obs2d.shape[0]}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        f"N={model2d_visible.shape[0]} visible={visible_subset_count} mode={visible_subset_mode}",
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        right,
        f"visible_ratio={visible_subset_count}/{total_model_samples}",
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    points_panel = np.concatenate([left, right], axis=1)
    points_panel_path = out_dir / f"iter_{iter_idx:05d}_points2d_panel.png"
    cv2.imwrite(str(points_panel_path), points_panel)
    paths["points2d_panel_png"] = str(points_panel_path)

    nn_img = frame_bgr.copy()
    draw_nn_edges(
        canvas_bgr=nn_img,
        obs2d=obs2d,
        model2d=model2d_visible,
        nn_idx=nn2d_idx,
        nn_d2=nn2d_d2,
        max_lines=max_nn_lines,
        seed=seed,
    )
    cv2.putText(
        nn_img,
        "2D NN edges: O2D -> nearest M2D'",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    nn_path = out_dir / f"iter_{iter_idx:05d}_nn2d.png"
    cv2.imwrite(str(nn_path), nn_img)
    paths["nn2d_png"] = str(nn_path)

    hist_path = out_dir / f"iter_{iter_idx:05d}_nn_hist.png"
    save_distance_histogram(
        out_path=hist_path,
        d2_3d=nn3d_d2,
        d2_2d=nn2d_d2,
        mesh_name=mesh_name,
        iter_idx=iter_idx,
    )
    paths["nn_hist_png"] = str(hist_path)

    return paths


def new_history_dict_chamfer() -> dict[str, list[float]]:
    return {
        "iter": [],
        "total": [],
        "cd3d": [],
        "cd3d_fwd": [],
        "cd3d_bwd": [],
        "cd3d_raw": [],
        "cd3d_raw_fwd": [],
        "cd3d_raw_bwd": [],
        "cd2d": [],
        "cd2d_fwd": [],
        "cd2d_bwd": [],
        "scale_reg": [],
        "t_reg": [],
        "scale": [],
        "tx": [],
        "ty": [],
        "tz": [],
    }


def append_history_chamfer(
    history: dict[str, list[float]],
    iter_idx: int,
    losses: dict[str, torch.Tensor],
) -> None:
    history["iter"].append(int(iter_idx))
    history["total"].append(float(losses["total"].detach().cpu().item()))
    history["cd3d"].append(float(losses["cd3d"].detach().cpu().item()))
    history["cd3d_fwd"].append(float(losses["cd3d_fwd"].detach().cpu().item()))
    history["cd3d_bwd"].append(float(losses["cd3d_bwd"].detach().cpu().item()))
    history["cd3d_raw"].append(float(losses["cd3d_raw"].detach().cpu().item()))
    history["cd3d_raw_fwd"].append(float(losses["cd3d_raw_fwd"].detach().cpu().item()))
    history["cd3d_raw_bwd"].append(float(losses["cd3d_raw_bwd"].detach().cpu().item()))
    history["cd2d"].append(float(losses["cd2d"].detach().cpu().item()))
    history["cd2d_fwd"].append(float(losses["cd2d_fwd"].detach().cpu().item()))
    history["cd2d_bwd"].append(float(losses["cd2d_bwd"].detach().cpu().item()))
    history["scale_reg"].append(float(losses["scale_reg"].detach().cpu().item()))
    history["t_reg"].append(float(losses["t_reg"].detach().cpu().item()))
    history["scale"].append(float(losses["scale"].detach().cpu().item()))
    history["tx"].append(float(losses["tx"].detach().cpu().item()))
    history["ty"].append(float(losses["ty"].detach().cpu().item()))
    history["tz"].append(float(losses["tz"].detach().cpu().item()))


def normalize_history_chamfer(
    history: dict[str, list[Any]],
) -> dict[str, list[float]]:
    out = new_history_dict_chamfer()
    for key in out:
        vals = history.get(key, [])
        if not isinstance(vals, list):
            raise ValueError(f"History key '{key}' must be a list.")
        if key == "iter":
            out[key] = [int(v) for v in vals]
        else:
            out[key] = [float(v) for v in vals]

    n = len(out["iter"])
    for key, vals in out.items():
        if len(vals) != n:
            raise ValueError(f"History length mismatch for key '{key}'.")
    return out


def build_resume_signature(
    *,
    video_name: str,
    mesh_slug: str,
    mesh_name: str,
    intrinsics_source: str,
    opt_max_side: int,
    sam3_mask_erode_iters: int,
    seed: int,
    lr: float,
    w_cd3d: float,
    w_cd2d: float,
    w_scale_reg: float,
    w_t_reg: float,
    rho_geman_3d: float,
    mesh_sample_points: int,
    max_obs_3d_points_per_mesh: int,
    max_obs_2d_points_per_mesh: int,
    nn_chunk_size: int,
    min_obs_3d_points_per_mesh: int,
    min_obs_2d_points_per_mesh: int,
    min_scale: float,
    max_scale: float,
    max_abs_delta_tx: float,
    max_abs_delta_ty: float,
    max_abs_delta_tz: float,
    visible_subset_z_abs_tol_m: float,
    visible_subset_z_rel_tol: float,
    min_visible_subset_points_per_mesh: int,
    visible_subset_focal_scale: float,
) -> dict[str, Any]:
    return {
        "video_name": video_name,
        "mesh_slug": mesh_slug,
        "mesh_name": mesh_name,
        "intrinsics_source": intrinsics_source,
        "opt_max_side": int(opt_max_side),
        "sam3_mask_erode_iters": int(sam3_mask_erode_iters),
        "seed": int(seed),
        "lr": float(lr),
        "w_cd3d": float(w_cd3d),
        "w_cd2d": float(w_cd2d),
        "w_scale_reg": float(w_scale_reg),
        "w_t_reg": float(w_t_reg),
        "rho_geman_3d": float(rho_geman_3d),
        "mesh_sample_points": int(mesh_sample_points),
        "max_obs_3d_points_per_mesh": int(max_obs_3d_points_per_mesh),
        "max_obs_2d_points_per_mesh": int(max_obs_2d_points_per_mesh),
        "nn_chunk_size": int(nn_chunk_size),
        "min_obs_3d_points_per_mesh": int(min_obs_3d_points_per_mesh),
        "min_obs_2d_points_per_mesh": int(min_obs_2d_points_per_mesh),
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "max_abs_delta_tx": float(max_abs_delta_tx),
        "max_abs_delta_ty": float(max_abs_delta_ty),
        "max_abs_delta_tz": float(max_abs_delta_tz),
        "visible_subset_z_abs_tol_m": float(visible_subset_z_abs_tol_m),
        "visible_subset_z_rel_tol": float(visible_subset_z_rel_tol),
        "min_visible_subset_points_per_mesh": int(min_visible_subset_points_per_mesh),
        "visible_subset_focal_scale": float(visible_subset_focal_scale),
    }


def compare_resume_signature_strict(
    saved_signature: dict[str, Any],
    current_signature: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    all_keys = sorted(set(saved_signature.keys()) | set(current_signature.keys()))
    for key in all_keys:
        if key not in saved_signature:
            mismatches.append(f"{key}: missing in checkpoint")
            continue
        if key not in current_signature:
            mismatches.append(f"{key}: missing in current config")
            continue

        a = saved_signature[key]
        b = current_signature[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
                mismatches.append(f"{key}: checkpoint={a} current={b}")
        else:
            if a != b:
                mismatches.append(f"{key}: checkpoint={a!r} current={b!r}")
    return mismatches


def optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def save_resume_checkpoint(
    checkpoint_path: Path,
    *,
    signature: dict[str, Any],
    start_iter: int,
    history: dict[str, list[float]],
    log_scale: torch.nn.Parameter,
    delta_tx: torch.nn.Parameter,
    delta_ty: torch.nn.Parameter,
    delta_tz: torch.nn.Parameter,
    tx_init: float,
    ty_init: float,
    tz_init: float,
    optimizer: torch.optim.Optimizer,
    best_total: float | None,
    best_iter: int | None,
    best_log_scale: float | None,
    best_delta_tx: float | None,
    best_delta_ty: float | None,
    best_delta_tz: float | None,
    visible_subset_indices: np.ndarray,
    visible_subset_mode: str,
    visible_subset_z_abs_tol_m: float,
    visible_subset_z_rel_tol: float,
    min_visible_subset_points_per_mesh: int,
    visible_subset_focal_scale: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "signature": signature,
        "start_iter": int(start_iter),
        "history": normalize_history_chamfer(history),
        "log_scale": float(log_scale.detach().cpu().item()),
        "delta_tx": float(delta_tx.detach().cpu().item()),
        "delta_ty": float(delta_ty.detach().cpu().item()),
        "delta_tz": float(delta_tz.detach().cpu().item()),
        "tx_init": float(tx_init),
        "ty_init": float(ty_init),
        "tz_init": float(tz_init),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_total": None if best_total is None else float(best_total),
        "best_iter": None if best_iter is None else int(best_iter),
        "best_log_scale": None if best_log_scale is None else float(best_log_scale),
        "best_delta_tx": None if best_delta_tx is None else float(best_delta_tx),
        "best_delta_ty": None if best_delta_ty is None else float(best_delta_ty),
        "best_delta_tz": None if best_delta_tz is None else float(best_delta_tz),
        "visible_subset_indices": visible_subset_indices.astype(np.int64).tolist(),
        "visible_subset_mode": str(visible_subset_mode),
        "visible_subset_z_abs_tol_m": float(visible_subset_z_abs_tol_m),
        "visible_subset_z_rel_tol": float(visible_subset_z_rel_tol),
        "min_visible_subset_points_per_mesh": int(min_visible_subset_points_per_mesh),
        "visible_subset_focal_scale": float(visible_subset_focal_scale),
    }
    torch.save(payload, str(checkpoint_path))


def load_resume_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> ResumeStateChamfer:
    payload = torch.load(str(checkpoint_path), map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
    if int(payload.get("version", -1)) != 2:
        raise ValueError(
            f"Unsupported checkpoint version for {checkpoint_path}: {payload.get('version')}"
        )

    history_raw = payload.get("history")
    if not isinstance(history_raw, dict):
        raise ValueError(f"Checkpoint missing history: {checkpoint_path}")

    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"Checkpoint missing signature: {checkpoint_path}")
    visible_subset_raw = payload.get("visible_subset_indices")
    if not isinstance(visible_subset_raw, list):
        raise ValueError(f"Checkpoint missing visible_subset_indices: {checkpoint_path}")
    visible_subset_indices = np.asarray(visible_subset_raw, dtype=np.int64)

    return ResumeStateChamfer(
        signature=signature,
        start_iter=int(payload.get("start_iter", 0)),
        history=normalize_history_chamfer(history_raw),
        log_scale=float(payload.get("log_scale", 0.0)),
        delta_tx=float(payload.get("delta_tx", 0.0)),
        delta_ty=float(payload.get("delta_ty", 0.0)),
        delta_tz=float(payload.get("delta_tz", 0.0)),
        tx_init=float(payload.get("tx_init", 0.0)),
        ty_init=float(payload.get("ty_init", 0.0)),
        tz_init=float(payload.get("tz_init", 0.0)),
        best_total=(
            None if payload.get("best_total") is None else float(payload["best_total"])
        ),
        best_iter=None if payload.get("best_iter") is None else int(payload["best_iter"]),
        best_log_scale=(
            None
            if payload.get("best_log_scale") is None
            else float(payload["best_log_scale"])
        ),
        best_delta_tx=(
            None if payload.get("best_delta_tx") is None else float(payload["best_delta_tx"])
        ),
        best_delta_ty=(
            None if payload.get("best_delta_ty") is None else float(payload["best_delta_ty"])
        ),
        best_delta_tz=(
            None if payload.get("best_delta_tz") is None else float(payload["best_delta_tz"])
        ),
        visible_subset_indices=visible_subset_indices,
        visible_subset_mode=str(payload.get("visible_subset_mode", "unknown")),
        visible_subset_z_abs_tol_m=float(payload.get("visible_subset_z_abs_tol_m", 0.02)),
        visible_subset_z_rel_tol=float(payload.get("visible_subset_z_rel_tol", 0.01)),
        min_visible_subset_points_per_mesh=int(
            payload.get("min_visible_subset_points_per_mesh", 128)
        ),
        visible_subset_focal_scale=float(payload.get("visible_subset_focal_scale", 0.6)),
        optimizer_state_dict=payload.get("optimizer_state_dict"),
    )


def save_loss_history_csv_chamfer(
    history: dict[str, list[float]],
    out_path: Path,
) -> None:
    if len(history["iter"]) == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [
            np.array(history["iter"], dtype=np.int32),
            np.array(history["total"], dtype=np.float32),
            np.array(history["cd3d"], dtype=np.float32),
            np.array(history["cd3d_fwd"], dtype=np.float32),
            np.array(history["cd3d_bwd"], dtype=np.float32),
            np.array(history["cd3d_raw"], dtype=np.float32),
            np.array(history["cd3d_raw_fwd"], dtype=np.float32),
            np.array(history["cd3d_raw_bwd"], dtype=np.float32),
            np.array(history["cd2d"], dtype=np.float32),
            np.array(history["cd2d_fwd"], dtype=np.float32),
            np.array(history["cd2d_bwd"], dtype=np.float32),
            np.array(history["scale_reg"], dtype=np.float32),
            np.array(history["t_reg"], dtype=np.float32),
            np.array(history["scale"], dtype=np.float32),
            np.array(history["tx"], dtype=np.float32),
            np.array(history["ty"], dtype=np.float32),
            np.array(history["tz"], dtype=np.float32),
        ]
    )
    np.savetxt(
        str(out_path),
        arr,
        fmt=[
            "%d",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
            "%.8f",
        ],
        delimiter=",",
        header=(
            "iter,total,cd3d,cd3d_fwd,cd3d_bwd,cd3d_raw,cd3d_raw_fwd,cd3d_raw_bwd,"
            "cd2d,cd2d_fwd,cd2d_bwd,scale_reg,t_reg,scale,tx,ty,tz"
        ),
        comments="",
    )


def plot_single_loss_curve(
    history: dict[str, list[float]],
    key: str,
    label: str,
    out_path: Path,
    title: str,
) -> None:
    if len(history["iter"]) == 0:
        return
    iters = np.array(history["iter"], dtype=np.int32)
    values = np.array(history[key], dtype=np.float32)
    plt.figure(figsize=(9, 5))
    plt.plot(iters, values, label=label, linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def plot_loss_curves_chamfer(
    history: dict[str, list[float]],
    out_dir: Path,
    slug: str,
    mesh_name: str,
) -> dict[str, str]:
    keys = {
        "total": "E_total",
        "cd3d": "E_cd3d_robust",
        "cd3d_fwd": "E_cd3d_fwd_robust",
        "cd3d_bwd": "E_cd3d_bwd_robust",
        "cd3d_raw": "E_cd3d_raw",
        "cd3d_raw_fwd": "E_cd3d_raw_fwd",
        "cd3d_raw_bwd": "E_cd3d_raw_bwd",
        "cd2d": "E_cd2d",
        "cd2d_fwd": "E_cd2d_fwd",
        "cd2d_bwd": "E_cd2d_bwd",
        "scale_reg": "E_scale_reg",
        "t_reg": "E_t_reg",
    }
    paths: dict[str, str] = {}
    for key, label in keys.items():
        out_path = out_dir / f"{slug}_loss_{key}.png"
        plot_single_loss_curve(
            history=history,
            key=key,
            label=label,
            out_path=out_path,
            title=f"{mesh_name}: {label}",
        )
        paths[key] = str(out_path)
    return paths


def compute_losses_chamfer_torch(
    model_points_base_all: torch.Tensor,
    model_points_base_visible: torch.Tensor,
    obs3d: torch.Tensor,
    obs2d: torch.Tensor,
    intrinsics: torch.Tensor,
    log_scale: torch.Tensor,
    delta_tx: torch.Tensor,
    delta_ty: torch.Tensor,
    delta_tz: torch.Tensor,
    tx_init: torch.Tensor,
    ty_init: torch.Tensor,
    tz_init: torch.Tensor,
    w_cd3d: float,
    w_cd2d: float,
    w_scale_reg: float,
    w_t_reg: float,
    rho_geman_3d: float,
    nn_chunk_size: int,
) -> dict[str, torch.Tensor]:
    scale = torch.exp(log_scale)
    tx = tx_init + delta_tx
    ty = ty_init + delta_ty
    tz = tz_init + delta_tz

    model3d_all = scale * model_points_base_all
    model3d_all = model3d_all + torch.stack([tx, ty, tz]).view(1, 3)
    model3d_visible = scale * model_points_base_visible
    model3d_visible = model3d_visible + torch.stack([tx, ty, tz]).view(1, 3)

    z_all = torch.clamp(model3d_all[:, 2], min=1e-6)
    u_all = intrinsics[0, 0] * model3d_all[:, 0] / z_all + intrinsics[0, 2] - 0.5
    v_all = intrinsics[1, 1] * model3d_all[:, 1] / z_all + intrinsics[1, 2] - 0.5
    model2d_all = torch.stack([u_all, v_all], dim=1)

    z_visible = torch.clamp(model3d_visible[:, 2], min=1e-6)
    u_visible = (
        intrinsics[0, 0] * model3d_visible[:, 0] / z_visible + intrinsics[0, 2] - 0.5
    )
    v_visible = (
        intrinsics[1, 1] * model3d_visible[:, 1] / z_visible + intrinsics[1, 2] - 0.5
    )
    model2d_visible = torch.stack([u_visible, v_visible], dim=1)

    zero = torch.zeros(
        (), device=model_points_base_visible.device, dtype=model_points_base_visible.dtype
    )

    if obs3d.shape[0] > 0 and model3d_visible.shape[0] > 0:
        d2_3d_fwd, _ = min_distances_obs_to_model_chunked(
            obs=obs3d,
            model=model3d_visible,
            chunk_size=nn_chunk_size,
            return_indices=False,
        )
        d2_3d_bwd, _ = min_distances_obs_to_model_chunked(
            obs=model3d_visible,
            model=obs3d,
            chunk_size=nn_chunk_size,
            return_indices=False,
        )
        cd3d_raw_fwd = d2_3d_fwd.mean()
        cd3d_raw_bwd = d2_3d_bwd.mean()
        cd3d_fwd = geman_mcclure_func(cd3d_raw_fwd, rho=float(rho_geman_3d))
        cd3d_bwd = geman_mcclure_func(cd3d_raw_bwd, rho=float(rho_geman_3d))
        cd3d_raw = 0.5 * (cd3d_raw_fwd + cd3d_raw_bwd)
        cd3d = 0.5 * (cd3d_fwd + cd3d_bwd)
    else:
        cd3d_raw_fwd = zero
        cd3d_raw_bwd = zero
        cd3d_fwd = zero
        cd3d_bwd = zero
        cd3d_raw = zero
        cd3d = zero

    if obs2d.shape[0] > 0 and model2d_visible.shape[0] > 0:
        d2_2d_fwd, _ = min_distances_obs_to_model_chunked(
            obs=obs2d,
            model=model2d_visible,
            chunk_size=nn_chunk_size,
            return_indices=False,
        )
        d2_2d_bwd, _ = min_distances_obs_to_model_chunked(
            obs=model2d_visible,
            model=obs2d,
            chunk_size=nn_chunk_size,
            return_indices=False,
        )
        cd2d_fwd = d2_2d_fwd.mean()
        cd2d_bwd = d2_2d_bwd.mean()
        cd2d = 0.5 * (cd2d_fwd + cd2d_bwd)
    else:
        cd2d_fwd = zero
        cd2d_bwd = zero
        cd2d = zero

    scale_reg = log_scale.pow(2)
    t_reg = delta_tx.pow(2) + delta_ty.pow(2) + delta_tz.pow(2)

    total = (
        float(w_cd3d) * cd3d
        + float(w_cd2d) * cd2d
        + float(w_scale_reg) * scale_reg
        + float(w_t_reg) * t_reg
    )

    return {
        "total": total,
        "cd3d": cd3d,
        "cd3d_fwd": cd3d_fwd,
        "cd3d_bwd": cd3d_bwd,
        "cd3d_raw": cd3d_raw,
        "cd3d_raw_fwd": cd3d_raw_fwd,
        "cd3d_raw_bwd": cd3d_raw_bwd,
        "cd2d": cd2d,
        "cd2d_fwd": cd2d_fwd,
        "cd2d_bwd": cd2d_bwd,
        "scale_reg": scale_reg,
        "t_reg": t_reg,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "model3d_all": model3d_all,
        "model2d_all": model2d_all,
        "model3d_visible": model3d_visible,
        "model2d_visible": model2d_visible,
    }


def optimize_scale_txyz_chamfer(
    mesh_name: str,
    obs: ObservationSet,
    model_points_base_np: np.ndarray,
    verts_cv_np: np.ndarray,
    faces_np: np.ndarray,
    intrinsics: np.ndarray,
    frame_bgr_for_debug: np.ndarray,
    debug_dir: Path,
    device: torch.device,
    iters: int,
    lr: float,
    w_cd3d: float,
    w_cd2d: float,
    w_scale_reg: float,
    w_t_reg: float,
    rho_geman_3d: float,
    nn_chunk_size: int,
    min_scale: float,
    max_scale: float,
    max_abs_delta_tx: float,
    max_abs_delta_ty: float,
    max_abs_delta_tz: float,
    log_every: int,
    debug_save_every: int,
    debug_point_radius: int,
    debug_max_points_vis: int,
    debug_max_nn_lines: int,
    debug_seed: int,
    min_obs_3d_points_per_mesh: int,
    min_obs_2d_points_per_mesh: int,
    checkpoint_path: Path,
    resume_signature: dict[str, Any],
    resume_state: ResumeStateChamfer | None,
    checkpoint_every: int,
    early_stop_patience: int,
    early_stop_rel_min_delta: float,
    early_stop_min_iter: int,
    visible_subset_z_abs_tol_m: float,
    visible_subset_z_rel_tol: float,
    min_visible_subset_points_per_mesh: int,
    visible_subset_focal_scale: float,
) -> OptimizationResultChamfer:
    n_obs3d = int(obs.obs_points_3d.shape[0])
    n_obs2d = int(obs.obs_pixels_2d.shape[0])
    n_model_total = int(model_points_base_np.shape[0])
    resumed = resume_state is not None
    resume_source = "checkpoint" if resumed else "fresh"
    start_iter = 0 if resume_state is None else int(resume_state.start_iter)

    if n_model_total == 0:
        return OptimizationResultChamfer(
            status="skipped_no_model_samples",
            message="No model surface samples available.",
            obs_3d_count=n_obs3d,
            obs_2d_count=n_obs2d,
            model_sample_count=0,
            model_sample_count_total=0,
            model_sample_count_visible_fixed=0,
            model_sample_visible_ratio=0.0,
            visible_subset_mode="no_samples",
            scale=1.0,
            log_scale=0.0,
            tx_init=0.0,
            ty_init=0.0,
            tz_init=0.0,
            delta_tx=0.0,
            delta_ty=0.0,
            delta_tz=0.0,
            tx=0.0,
            ty=0.0,
            tz=0.0,
            history=new_history_dict_chamfer(),
            final_total_loss=None,
            final_cd3d_loss=None,
            final_cd2d_loss=None,
            resumed=resumed,
            resume_source=resume_source,
            start_iter=start_iter,
            end_iter=start_iter,
            iters_executed_this_run=0,
            early_stopped=False,
            early_stop_iter=None,
            best_iter=None,
            best_total=None,
            checkpoint_path=str(checkpoint_path),
        )

    enough_3d = n_obs3d >= int(min_obs_3d_points_per_mesh)
    enough_2d = n_obs2d >= int(min_obs_2d_points_per_mesh)
    if not enough_3d and not enough_2d:
        msg = (
            "Too few observations for both 3D and 2D: "
            f"obs3d={n_obs3d} (<{int(min_obs_3d_points_per_mesh)}), "
            f"obs2d={n_obs2d} (<{int(min_obs_2d_points_per_mesh)})."
        )
        return OptimizationResultChamfer(
            status="skipped_too_few_observations",
            message=msg,
            obs_3d_count=n_obs3d,
            obs_2d_count=n_obs2d,
            model_sample_count=0,
            model_sample_count_total=n_model_total,
            model_sample_count_visible_fixed=0,
            model_sample_visible_ratio=0.0,
            visible_subset_mode="too_few_observations",
            scale=1.0,
            log_scale=0.0,
            tx_init=0.0,
            ty_init=0.0,
            tz_init=0.0,
            delta_tx=0.0,
            delta_ty=0.0,
            delta_tz=0.0,
            tx=0.0,
            ty=0.0,
            tz=0.0,
            history=new_history_dict_chamfer(),
            final_total_loss=None,
            final_cd3d_loss=None,
            final_cd2d_loss=None,
            resumed=resumed,
            resume_source=resume_source,
            start_iter=start_iter,
            end_iter=start_iter,
            iters_executed_this_run=0,
            early_stopped=False,
            early_stop_iter=None,
            best_iter=None,
            best_total=None,
            checkpoint_path=str(checkpoint_path),
        )

    obs3d_t = torch.from_numpy(obs.obs_points_3d).to(device=device, dtype=torch.float32)
    obs2d_t = torch.from_numpy(obs.obs_pixels_2d).to(device=device, dtype=torch.float32)

    if resumed:
        tx_init_val = float(resume_state.tx_init)
        ty_init_val = float(resume_state.ty_init)
        tz_init_val = float(resume_state.tz_init)
    else:
        tx_init_val = 0.0
        ty_init_val = 0.0
        if n_obs3d > 0:
            obs_z_median = float(np.median(obs.obs_points_3d[:, 2]))
            model_z_median = float(np.median(model_points_base_np[:, 2]))
            tz_init_val = obs_z_median - model_z_median
            tz_init_val = 0.0
        else:
            tz_init_val = 0.0

    log_scale_init = 0.0 if resume_state is None else float(resume_state.log_scale)
    delta_tx_init = 0.0 if resume_state is None else float(resume_state.delta_tx)
    delta_ty_init = 0.0 if resume_state is None else float(resume_state.delta_ty)
    delta_tz_init = 0.0 if resume_state is None else float(resume_state.delta_tz)

    if resumed:
        if not math.isclose(
            float(resume_state.visible_subset_z_abs_tol_m),
            float(visible_subset_z_abs_tol_m),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[{mesh_name}] checkpoint visible_subset_z_abs_tol_m mismatch: "
                f"checkpoint={resume_state.visible_subset_z_abs_tol_m} "
                f"current={visible_subset_z_abs_tol_m}"
            )
        if not math.isclose(
            float(resume_state.visible_subset_z_rel_tol),
            float(visible_subset_z_rel_tol),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[{mesh_name}] checkpoint visible_subset_z_rel_tol mismatch: "
                f"checkpoint={resume_state.visible_subset_z_rel_tol} "
                f"current={visible_subset_z_rel_tol}"
            )
        if int(resume_state.min_visible_subset_points_per_mesh) != int(
            min_visible_subset_points_per_mesh
        ):
            raise RuntimeError(
                f"[{mesh_name}] checkpoint min_visible_subset_points_per_mesh mismatch: "
                f"checkpoint={resume_state.min_visible_subset_points_per_mesh} "
                f"current={min_visible_subset_points_per_mesh}"
            )
        if not math.isclose(
            float(resume_state.visible_subset_focal_scale),
            float(visible_subset_focal_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[{mesh_name}] checkpoint visible_subset_focal_scale mismatch: "
                f"checkpoint={resume_state.visible_subset_focal_scale} "
                f"current={visible_subset_focal_scale}"
            )

    if resumed:
        visible_subset_indices = resume_state.visible_subset_indices.astype(np.int64)
        visible_subset_mode = str(resume_state.visible_subset_mode)
        strict_visible_count = int(visible_subset_indices.shape[0])
        in_frame_positive_count = int(visible_subset_indices.shape[0])
        positive_depth_count = int(visible_subset_indices.shape[0])
    else:
        subset = build_fixed_visible_subset(
            verts_cv=verts_cv_np,
            faces=faces_np,
            sampled_points_base=model_points_base_np,
            intrinsics=intrinsics,
            image_h=int(frame_bgr_for_debug.shape[0]),
            image_w=int(frame_bgr_for_debug.shape[1]),
            device=device,
            scale_init=float(math.exp(log_scale_init)),
            tx_init=float(tx_init_val + delta_tx_init),
            ty_init=float(ty_init_val + delta_ty_init),
            tz_init=float(tz_init_val + delta_tz_init),
            z_abs_tol_m=float(visible_subset_z_abs_tol_m),
            z_rel_tol=float(visible_subset_z_rel_tol),
            min_visible_subset_points_per_mesh=int(min_visible_subset_points_per_mesh),
            focal_scale_for_visibility=float(visible_subset_focal_scale),
        )
        visible_subset_indices = subset.indices.astype(np.int64)
        visible_subset_mode = subset.mode
        strict_visible_count = int(subset.strict_visible_count)
        in_frame_positive_count = int(subset.in_frame_positive_count)
        positive_depth_count = int(subset.positive_depth_count)

    valid_idx = (
        (visible_subset_indices >= 0) & (visible_subset_indices < int(n_model_total))
    )
    visible_subset_indices = visible_subset_indices[valid_idx]
    if visible_subset_indices.shape[0] == 0:
        msg = (
            "No visible fixed model subset points after one-time visibility selection "
            f"(mode={visible_subset_mode})."
        )
        return OptimizationResultChamfer(
            status="skipped_no_visible_subset",
            message=msg,
            obs_3d_count=n_obs3d,
            obs_2d_count=n_obs2d,
            model_sample_count=0,
            model_sample_count_total=n_model_total,
            model_sample_count_visible_fixed=0,
            model_sample_visible_ratio=0.0,
            visible_subset_mode=visible_subset_mode,
            scale=1.0,
            log_scale=0.0,
            tx_init=tx_init_val,
            ty_init=ty_init_val,
            tz_init=tz_init_val,
            delta_tx=0.0,
            delta_ty=0.0,
            delta_tz=0.0,
            tx=float(tx_init_val),
            ty=float(ty_init_val),
            tz=float(tz_init_val),
            history=new_history_dict_chamfer(),
            final_total_loss=None,
            final_cd3d_loss=None,
            final_cd2d_loss=None,
            resumed=resumed,
            resume_source=resume_source,
            start_iter=start_iter,
            end_iter=start_iter,
            iters_executed_this_run=0,
            early_stopped=False,
            early_stop_iter=None,
            best_iter=None,
            best_total=None,
            checkpoint_path=str(checkpoint_path),
        )

    n_model_visible = int(visible_subset_indices.shape[0])
    visible_ratio = float(n_model_visible) / float(max(1, n_model_total))
    print(
        f"[{mesh_name}] fixed visible subset: {n_model_visible}/{n_model_total} "
        f"(ratio={visible_ratio:.4f}, mode={visible_subset_mode}, "
        f"strict={strict_visible_count}, in_frame_zpos={in_frame_positive_count}, "
        f"zpos={positive_depth_count})"
    )

    model_visible_base_np = model_points_base_np[visible_subset_indices]
    model_base_all_t = torch.from_numpy(model_points_base_np).to(
        device=device, dtype=torch.float32
    )
    model_base_visible_t = torch.from_numpy(model_visible_base_np).to(
        device=device, dtype=torch.float32
    )
    k_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)
    tx_init_t = torch.tensor(tx_init_val, device=device, dtype=torch.float32)
    ty_init_t = torch.tensor(ty_init_val, device=device, dtype=torch.float32)
    tz_init_t = torch.tensor(tz_init_val, device=device, dtype=torch.float32)

    log_scale = torch.nn.Parameter(
        torch.tensor(log_scale_init, device=device, dtype=torch.float32)
    )
    delta_tx = torch.nn.Parameter(
        torch.tensor(delta_tx_init, device=device, dtype=torch.float32)
    )
    delta_ty = torch.nn.Parameter(
        torch.tensor(delta_ty_init, device=device, dtype=torch.float32)
    )
    delta_tz = torch.nn.Parameter(
        torch.tensor(delta_tz_init, device=device, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam([log_scale, delta_tx, delta_ty, delta_tz], lr=float(lr))
    if resumed and resume_state.optimizer_state_dict is not None:
        optimizer.load_state_dict(resume_state.optimizer_state_dict)
        optimizer_state_to_device(optimizer, device)

    min_log_scale = math.log(float(min_scale))
    max_log_scale = math.log(float(max_scale))

    if resumed:
        history = normalize_history_chamfer(resume_state.history)
        if len(history["iter"]) > 0 and int(history["iter"][-1]) != start_iter:
            raise ValueError(
                f"{mesh_name}: checkpoint start_iter={start_iter} does not match "
                f"history last iter={history['iter'][-1]}."
            )
        if len(history["iter"]) == 0 and start_iter != 0:
            raise ValueError(
                f"{mesh_name}: checkpoint start_iter={start_iter} with empty history."
            )
    else:
        history = new_history_dict_chamfer()
    obs3d_anchor_colors = colorize_points_by_xyz(obs.obs_points_3d)
    obs2d_anchor_colors = colorize_points_by_xy(obs.obs_pixels_2d)
    gray_rgb = np.array([160, 160, 160], dtype=np.uint8)

    def _save_debug(iter_idx: int, losses_eval: dict[str, torch.Tensor], save_fixed: bool) -> None:
        with torch.no_grad():
            model3d_all_t = losses_eval["model3d_all"]
            model3d_visible_t = losses_eval["model3d_visible"]
            model2d_visible_t = losses_eval["model2d_visible"]
            model3d_all_np = model3d_all_t.detach().cpu().numpy()
            model3d_visible_np = model3d_visible_t.detach().cpu().numpy()
            model2d_visible_np = model2d_visible_t.detach().cpu().numpy()

            d2_3d_t, _ = min_distances_obs_to_model_chunked(
                obs=obs3d_t,
                model=model3d_visible_t,
                chunk_size=nn_chunk_size,
                return_indices=False,
            )
            _, model_to_obs_idx_t = min_distances_obs_to_model_chunked(
                obs=model3d_visible_t,
                model=obs3d_t,
                chunk_size=nn_chunk_size,
                return_indices=True,
            )
            d2_2d_t, idx2d_t = min_distances_obs_to_model_chunked(
                obs=obs2d_t,
                model=model2d_visible_t,
                chunk_size=nn_chunk_size,
                return_indices=True,
            )
            _, model2d_to_obs2d_idx_t = min_distances_obs_to_model_chunked(
                obs=model2d_visible_t,
                model=obs2d_t,
                chunk_size=nn_chunk_size,
                return_indices=True,
            )

            model3d_all_colors = np.tile(gray_rgb[None, :], (model3d_all_np.shape[0], 1))
            if model3d_visible_np.shape[0] > 0:
                visible3d_colors = colorize_points_by_xyz(model3d_visible_np)
                if model_to_obs_idx_t is not None and obs3d_anchor_colors.shape[0] > 0:
                    model_to_obs_idx = (
                        model_to_obs_idx_t.detach().cpu().numpy().astype(np.int64)
                    )
                    valid_idx = (
                        (model_to_obs_idx >= 0)
                        & (model_to_obs_idx < obs3d_anchor_colors.shape[0])
                    )
                    if np.any(valid_idx):
                        visible3d_colors[valid_idx] = obs3d_anchor_colors[
                            model_to_obs_idx[valid_idx]
                        ]
                model3d_all_colors[visible_subset_indices] = visible3d_colors

            model2d_visible_colors_rgb = colorize_points_by_xy(model2d_visible_np)
            if model2d_visible_np.shape[0] > 0:
                if model2d_to_obs2d_idx_t is not None and obs2d_anchor_colors.shape[0] > 0:
                    model2d_to_obs2d_idx = (
                        model2d_to_obs2d_idx_t.detach().cpu().numpy().astype(np.int64)
                    )
                    valid_model2d_idx = (
                        (model2d_to_obs2d_idx >= 0)
                        & (model2d_to_obs2d_idx < obs2d_anchor_colors.shape[0])
                    )
                    if np.any(valid_model2d_idx):
                        model2d_visible_colors_rgb[valid_model2d_idx] = obs2d_anchor_colors[
                            model2d_to_obs2d_idx[valid_model2d_idx]
                        ]

            save_chamfer_debug_snapshot(
                out_dir=debug_dir,
                mesh_name=mesh_name,
                iter_idx=iter_idx,
                frame_bgr=frame_bgr_for_debug,
                obs3d=obs.obs_points_3d,
                obs3d_colors=obs3d_anchor_colors,
                obs2d=obs.obs_pixels_2d,
                obs2d_anchor_colors_rgb=obs2d_anchor_colors,
                model3d_all=model3d_all_np,
                model3d_all_colors=model3d_all_colors,
                model2d_visible=model2d_visible_np,
                model2d_visible_colors_rgb=model2d_visible_colors_rgb,
                nn2d_idx=np.zeros((0,), dtype=np.int64)
                if idx2d_t is None
                else idx2d_t.detach().cpu().numpy().astype(np.int64),
                nn2d_d2=d2_2d_t.detach().cpu().numpy(),
                nn3d_d2=d2_3d_t.detach().cpu().numpy(),
                visible_subset_mode=visible_subset_mode,
                visible_subset_count=int(n_model_visible),
                total_model_samples=int(n_model_total),
                point_radius=debug_point_radius,
                max_points_vis=debug_max_points_vis,
                max_nn_lines=debug_max_nn_lines,
                seed=int(debug_seed + 7919 * (iter_idx + 1)),
                save_obs3d_fixed=save_fixed,
            )

    best_total: float | None = None
    best_iter: int | None = None
    best_log_scale: float | None = None
    best_delta_tx: float | None = None
    best_delta_ty: float | None = None
    best_delta_tz: float | None = None

    if resumed:
        best_total = (
            None if resume_state.best_total is None else float(resume_state.best_total)
        )
        best_iter = None if resume_state.best_iter is None else int(resume_state.best_iter)
        best_log_scale = (
            None
            if resume_state.best_log_scale is None
            else float(resume_state.best_log_scale)
        )
        best_delta_tx = (
            None if resume_state.best_delta_tx is None else float(resume_state.best_delta_tx)
        )
        best_delta_ty = (
            None if resume_state.best_delta_ty is None else float(resume_state.best_delta_ty)
        )
        best_delta_tz = (
            None if resume_state.best_delta_tz is None else float(resume_state.best_delta_tz)
        )
    else:
        with torch.no_grad():
            init_losses = compute_losses_chamfer_torch(
                model_points_base_all=model_base_all_t,
                model_points_base_visible=model_base_visible_t,
                obs3d=obs3d_t,
                obs2d=obs2d_t,
                intrinsics=k_t,
                log_scale=log_scale,
                delta_tx=delta_tx,
                delta_ty=delta_ty,
                delta_tz=delta_tz,
                tx_init=tx_init_t,
                ty_init=ty_init_t,
                tz_init=tz_init_t,
                w_cd3d=w_cd3d,
                w_cd2d=w_cd2d,
                w_scale_reg=w_scale_reg,
                w_t_reg=w_t_reg,
                rho_geman_3d=rho_geman_3d,
                nn_chunk_size=nn_chunk_size,
            )
            append_history_chamfer(history, 0, init_losses)
            _save_debug(iter_idx=0, losses_eval=init_losses, save_fixed=True)

            best_total = float(init_losses["total"].detach().cpu().item())
            best_iter = 0
            best_log_scale = float(log_scale.detach().cpu().item())
            best_delta_tx = float(delta_tx.detach().cpu().item())
            best_delta_ty = float(delta_ty.detach().cpu().item())
            best_delta_tz = float(delta_tz.detach().cpu().item())

    if best_total is None:
        if len(history["total"]) > 0:
            best_total = float(np.min(np.array(history["total"], dtype=np.float64)))
            best_idx_local = int(np.argmin(np.array(history["total"], dtype=np.float64)))
            best_iter = int(history["iter"][best_idx_local])
        else:
            best_total = None
            best_iter = None
    if best_log_scale is None:
        best_log_scale = float(log_scale.detach().cpu().item())
    if best_delta_tx is None:
        best_delta_tx = float(delta_tx.detach().cpu().item())
    if best_delta_ty is None:
        best_delta_ty = float(delta_ty.detach().cpu().item())
    if best_delta_tz is None:
        best_delta_tz = float(delta_tz.detach().cpu().item())

    def _save_checkpoint(iter_done: int) -> None:
        save_resume_checkpoint(
            checkpoint_path=checkpoint_path,
            signature=resume_signature,
            start_iter=int(iter_done),
            history=history,
            log_scale=log_scale,
            delta_tx=delta_tx,
            delta_ty=delta_ty,
            delta_tz=delta_tz,
            tx_init=tx_init_val,
            ty_init=ty_init_val,
            tz_init=tz_init_val,
            optimizer=optimizer,
            best_total=best_total,
            best_iter=best_iter,
            best_log_scale=best_log_scale,
            best_delta_tx=best_delta_tx,
            best_delta_ty=best_delta_ty,
            best_delta_tz=best_delta_tz,
            visible_subset_indices=visible_subset_indices,
            visible_subset_mode=visible_subset_mode,
            visible_subset_z_abs_tol_m=float(visible_subset_z_abs_tol_m),
            visible_subset_z_rel_tol=float(visible_subset_z_rel_tol),
            min_visible_subset_points_per_mesh=int(min_visible_subset_points_per_mesh),
            visible_subset_focal_scale=float(visible_subset_focal_scale),
        )

    _save_checkpoint(iter_done=start_iter)

    target_additional_iters = int(iters)
    early_stopped = False
    early_stop_iter: int | None = None
    executed_iters = 0
    end_iter = start_iter

    for local_iter in range(1, target_additional_iters + 1):
        global_iter = start_iter + local_iter
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses_chamfer_torch(
            model_points_base_all=model_base_all_t,
            model_points_base_visible=model_base_visible_t,
            obs3d=obs3d_t,
            obs2d=obs2d_t,
            intrinsics=k_t,
            log_scale=log_scale,
            delta_tx=delta_tx,
            delta_ty=delta_ty,
            delta_tz=delta_tz,
            tx_init=tx_init_t,
            ty_init=ty_init_t,
            tz_init=tz_init_t,
            w_cd3d=w_cd3d,
            w_cd2d=w_cd2d,
            w_scale_reg=w_scale_reg,
            w_t_reg=w_t_reg,
            rho_geman_3d=rho_geman_3d,
            nn_chunk_size=nn_chunk_size,
        )
        losses["total"].backward()
        optimizer.step()

        with torch.no_grad():
            log_scale.clamp_(min_log_scale, max_log_scale)
            delta_tx.clamp_(-float(max_abs_delta_tx), float(max_abs_delta_tx))
            delta_ty.clamp_(-float(max_abs_delta_ty), float(max_abs_delta_ty))
            delta_tz.clamp_(-float(max_abs_delta_tz), float(max_abs_delta_tz))

            eval_losses = compute_losses_chamfer_torch(
                model_points_base_all=model_base_all_t,
                model_points_base_visible=model_base_visible_t,
                obs3d=obs3d_t,
                obs2d=obs2d_t,
                intrinsics=k_t,
                log_scale=log_scale,
                delta_tx=delta_tx,
                delta_ty=delta_ty,
                delta_tz=delta_tz,
                tx_init=tx_init_t,
                ty_init=ty_init_t,
                tz_init=tz_init_t,
                w_cd3d=w_cd3d,
                w_cd2d=w_cd2d,
                w_scale_reg=w_scale_reg,
                w_t_reg=w_t_reg,
                rho_geman_3d=rho_geman_3d,
                nn_chunk_size=nn_chunk_size,
            )
            append_history_chamfer(history, global_iter, eval_losses)
            executed_iters = local_iter
            end_iter = global_iter

            current_total = float(eval_losses["total"].detach().cpu().item())
            improved = False
            if best_total is None or not np.isfinite(best_total):
                improved = True
            else:
                rel_gain = (best_total - current_total) / max(abs(best_total), 1e-12)
                improved = rel_gain >= float(early_stop_rel_min_delta)
            if improved:
                best_total = current_total
                best_iter = int(global_iter)
                best_log_scale = float(log_scale.detach().cpu().item())
                best_delta_tx = float(delta_tx.detach().cpu().item())
                best_delta_ty = float(delta_ty.detach().cpu().item())
                best_delta_tz = float(delta_tz.detach().cpu().item())

            if log_every > 0 and (
                global_iter % int(log_every) == 0
                or local_iter == int(target_additional_iters)
            ):
                print(
                    f"iter={global_iter:05d} total={history['total'][-1]:.6f} "
                    f"cd3d={history['cd3d'][-1]:.6f} "
                    f"cd2d={history['cd2d'][-1]:.6f} "
                    f"scale={history['scale'][-1]:.6f} "
                    f"tx={history['tx'][-1]:.6f} "
                    f"ty={history['ty'][-1]:.6f} "
                    f"tz={history['tz'][-1]:.6f}"
                )

            if debug_save_every > 0 and (
                global_iter % int(debug_save_every) == 0
                or local_iter == int(target_additional_iters)
            ):
                _save_debug(iter_idx=global_iter, losses_eval=eval_losses, save_fixed=False)

            if checkpoint_every > 0 and global_iter % int(checkpoint_every) == 0:
                _save_checkpoint(iter_done=global_iter)

            if (
                int(early_stop_patience) > 0
                and global_iter >= int(early_stop_min_iter)
                and best_iter is not None
                and global_iter - int(best_iter) >= int(early_stop_patience)
            ):
                early_stopped = True
                early_stop_iter = int(global_iter)
                print(
                    f"[{mesh_name}] early stopping at iter={global_iter:05d} "
                    f"(best_iter={best_iter:05d}, best_total={best_total:.6f})"
                )
                break

    if early_stopped:
        with torch.no_grad():
            assert best_log_scale is not None
            assert best_delta_tx is not None
            assert best_delta_ty is not None
            assert best_delta_tz is not None
            log_scale.copy_(torch.tensor(best_log_scale, device=device, dtype=torch.float32))
            delta_tx.copy_(torch.tensor(best_delta_tx, device=device, dtype=torch.float32))
            delta_ty.copy_(torch.tensor(best_delta_ty, device=device, dtype=torch.float32))
            delta_tz.copy_(torch.tensor(best_delta_tz, device=device, dtype=torch.float32))
            log_scale.clamp_(min_log_scale, max_log_scale)
            delta_tx.clamp_(-float(max_abs_delta_tx), float(max_abs_delta_tx))
            delta_ty.clamp_(-float(max_abs_delta_ty), float(max_abs_delta_ty))
            delta_tz.clamp_(-float(max_abs_delta_tz), float(max_abs_delta_tz))

    with torch.no_grad():
        final_losses = compute_losses_chamfer_torch(
            model_points_base_all=model_base_all_t,
            model_points_base_visible=model_base_visible_t,
            obs3d=obs3d_t,
            obs2d=obs2d_t,
            intrinsics=k_t,
            log_scale=log_scale,
            delta_tx=delta_tx,
            delta_ty=delta_ty,
            delta_tz=delta_tz,
            tx_init=tx_init_t,
            ty_init=ty_init_t,
            tz_init=tz_init_t,
            w_cd3d=w_cd3d,
            w_cd2d=w_cd2d,
            w_scale_reg=w_scale_reg,
            w_t_reg=w_t_reg,
            rho_geman_3d=rho_geman_3d,
            nn_chunk_size=nn_chunk_size,
        )

    scale_final = float(math.exp(float(log_scale.detach().cpu().item())))
    log_scale_final = float(log_scale.detach().cpu().item())
    delta_tx_final = float(delta_tx.detach().cpu().item())
    delta_ty_final = float(delta_ty.detach().cpu().item())
    delta_tz_final = float(delta_tz.detach().cpu().item())
    tx_final = float(tx_init_val + delta_tx_final)
    ty_final = float(ty_init_val + delta_ty_final)
    tz_final = float(tz_init_val + delta_tz_final)

    _save_checkpoint(iter_done=end_iter)

    return OptimizationResultChamfer(
        status="optimized",
        message=None,
        obs_3d_count=n_obs3d,
        obs_2d_count=n_obs2d,
        model_sample_count=n_model_visible,
        model_sample_count_total=n_model_total,
        model_sample_count_visible_fixed=n_model_visible,
        model_sample_visible_ratio=visible_ratio,
        visible_subset_mode=visible_subset_mode,
        scale=scale_final,
        log_scale=log_scale_final,
        tx_init=tx_init_val,
        ty_init=ty_init_val,
        tz_init=tz_init_val,
        delta_tx=delta_tx_final,
        delta_ty=delta_ty_final,
        delta_tz=delta_tz_final,
        tx=tx_final,
        ty=ty_final,
        tz=tz_final,
        history=history,
        final_total_loss=float(final_losses["total"].detach().cpu().item()),
        final_cd3d_loss=float(final_losses["cd3d"].detach().cpu().item()),
        final_cd2d_loss=float(final_losses["cd2d"].detach().cpu().item()),
        resumed=resumed,
        resume_source=resume_source,
        start_iter=start_iter,
        end_iter=end_iter,
        iters_executed_this_run=int(executed_iters),
        early_stopped=early_stopped,
        early_stop_iter=early_stop_iter,
        best_iter=best_iter,
        best_total=best_total,
        checkpoint_path=str(checkpoint_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mesh-depth alignment with bidirectional 3D/2D chamfer losses and "
            "scale+t_xyz optimization."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_02")

    parser.add_argument(
        "--object_video_dir",
        type=str,
        default=None,
        help=(
            "Directory like ../Generate_Object_Mesh/output/video_xx "
            "(mesh + intrinsics)."
        ),
    )
    parser.add_argument(
        "--segmentation_video_dir",
        type=str,
        default=None,
        help="Directory like ../Segment_Video/output/video_xx (tracked masks).",
    )
    parser.add_argument(
        "--depth_video_dir",
        type=str,
        default=None,
        help="Directory like ../Estimate_Depth/output/video_xx",
    )
    parser.add_argument(
        "--human_video_dir",
        type=str,
        default=None,
        help="Directory like ../Estimate_Human_Motion/output/video_xx",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_scale_txyz_chamfer_visible_bidir",
        help="Root output directory; results are written to output_root/video_name.",
    )

    parser.add_argument(
        "--output_coord",
        type=str,
        choices=["opencv", "pytorch3d"],
        default="opencv",
        help="Coordinate frame for exported aligned .ply meshes.",
    )
    parser.add_argument(
        "--intrinsics_source",
        type=str,
        choices=["object", "depth"],
        default="object",
        help=(
            "'object': camera_intrinsics.json intrinsics_pixels_3x3. "
            "'depth': camera_intrinsics.json intrinsics_pixels_3x3 (DA3 side)."
        ),
    )
    parser.add_argument(
        "--intrinsics_warn_threshold_px",
        type=float,
        default=100.0,
        help=(
            "Warn if max |object - depth intrinsics| exceeds this threshold "
            "in pixels."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opt_max_side", type=int, default=1280)

    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--w_cd3d", type=float, default=1e3)
    parser.add_argument("--w_cd2d", type=float, default=1e-4)
    parser.add_argument("--w_scale_reg", type=float, default=1e-3)
    parser.add_argument("--w_t_reg", type=float, default=1e-3)
    parser.add_argument("--rho_geman_3d", type=float, default=0.2)

    # Note: Original defaults were 20000, 60000, 80000. Reduced for faster debugging.
    parser.add_argument("--mesh_sample_points", type=int, default=6000)
    parser.add_argument("--max_obs_3d_points_per_mesh", type=int, default=6000)
    parser.add_argument("--max_obs_2d_points_per_mesh", type=int, default=6000)
    parser.add_argument("--nn_chunk_size", type=int, default=1024)

    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=5.0)
    parser.add_argument("--max_abs_delta_tx", type=float, default=2.0)
    parser.add_argument("--max_abs_delta_ty", type=float, default=2.0)
    parser.add_argument("--max_abs_delta_tz", type=float, default=2.0)

    parser.add_argument("--min_obs_3d_points_per_mesh", type=int, default=32)
    parser.add_argument("--min_obs_2d_points_per_mesh", type=int, default=64)
    parser.add_argument("--visible_subset_z_abs_tol_m", type=float, default=0.02)
    parser.add_argument("--visible_subset_z_rel_tol", type=float, default=0.01)
    parser.add_argument("--min_visible_subset_points_per_mesh", type=int, default=128)
    parser.add_argument(
        "--visible_subset_focal_scale",
        type=float,
        default=0.6,
        help=(
            "Scale factor applied to fx/fy only during one-time visibility subset "
            "selection (render + sample projection)."
        ),
    )

    parser.add_argument("--debug_save_every", type=int, default=50)
    parser.add_argument("--debug_point_radius", type=int, default=1)
    parser.add_argument("--debug_max_points_vis", type=int, default=50000)
    parser.add_argument("--debug_max_nn_lines", type=int, default=2000)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable automatic checkpoint resume and start fresh optimization.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="Save per-mesh optimization checkpoint every N global iterations.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=100,
        help="Patience in global iterations for early stopping (0 disables).",
    )
    parser.add_argument(
        "--early_stop_rel_min_delta",
        type=float,
        default=1e-4,
        help="Minimum relative improvement in total loss to reset early-stop patience.",
    )
    parser.add_argument(
        "--early_stop_min_iter",
        type=int,
        default=300,
        help="Earliest global iteration to start applying early stopping checks.",
    )
    parser.add_argument(
        "--sam3_mask_erode_iters",
        type=int,
        default=3,
        help=(
            "Number of 3x3 erosion iterations applied to SAM3 masks "
            "(objects + human) before observation extraction."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_scale <= 0.0:
        raise ValueError("--min_scale must be > 0.")
    if args.max_scale <= args.min_scale:
        raise ValueError("--max_scale must be > --min_scale.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.mesh_sample_points <= 0:
        raise ValueError("--mesh_sample_points must be > 0.")
    if args.max_obs_3d_points_per_mesh <= 0:
        raise ValueError("--max_obs_3d_points_per_mesh must be > 0.")
    if args.max_obs_2d_points_per_mesh <= 0:
        raise ValueError("--max_obs_2d_points_per_mesh must be > 0.")
    if args.nn_chunk_size <= 0:
        raise ValueError("--nn_chunk_size must be > 0.")
    if args.max_abs_delta_tx <= 0.0:
        raise ValueError("--max_abs_delta_tx must be > 0.")
    if args.max_abs_delta_ty <= 0.0:
        raise ValueError("--max_abs_delta_ty must be > 0.")
    if args.max_abs_delta_tz <= 0.0:
        raise ValueError("--max_abs_delta_tz must be > 0.")
    if args.rho_geman_3d <= 0.0:
        raise ValueError("--rho_geman_3d must be > 0.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint_every must be > 0.")
    if args.early_stop_patience < 0:
        raise ValueError("--early_stop_patience must be >= 0.")
    if args.early_stop_rel_min_delta < 0.0:
        raise ValueError("--early_stop_rel_min_delta must be >= 0.")
    if args.early_stop_min_iter < 0:
        raise ValueError("--early_stop_min_iter must be >= 0.")
    if args.sam3_mask_erode_iters < 0:
        raise ValueError("--sam3_mask_erode_iters must be >= 0.")
    if args.visible_subset_z_abs_tol_m < 0.0:
        raise ValueError("--visible_subset_z_abs_tol_m must be >= 0.")
    if args.visible_subset_z_rel_tol < 0.0:
        raise ValueError("--visible_subset_z_rel_tol must be >= 0.")
    if args.min_visible_subset_points_per_mesh <= 0:
        raise ValueError("--min_visible_subset_points_per_mesh must be > 0.")
    if args.visible_subset_focal_scale <= 0.0:
        raise ValueError("--visible_subset_focal_scale must be > 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    object_video_dir = resolve_path(args.object_video_dir, script_dir) or (
        script_dir.parent / "Generate_Object_Mesh" / "output" / args.video_name
    ).resolve()
    segmentation_video_dir = resolve_path(args.segmentation_video_dir, script_dir) or (
        script_dir.parent / "Segment_Video" / "output" / args.video_name
    ).resolve()
    depth_video_dir = resolve_path(args.depth_video_dir, script_dir) or (
        script_dir.parent / "Estimate_Depth" / "output" / args.video_name
    ).resolve()
    human_video_dir = resolve_path(args.human_video_dir, script_dir) or (
        script_dir.parent / "Estimate_Human_Motion" / "output" / args.video_name
    ).resolve()

    output_root = resolve_path(args.output_root, script_dir)
    assert output_root is not None
    output_dir = output_root / args.video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not object_video_dir.exists():
        raise FileNotFoundError(f"Object dir not found: {object_video_dir}")
    if not segmentation_video_dir.exists():
        raise FileNotFoundError(f"Segmentation dir not found: {segmentation_video_dir}")
    if not depth_video_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_video_dir}")
    if not human_video_dir.exists():
        raise FileNotFoundError(f"Human dir not found: {human_video_dir}")

    depth_intrinsics_json_path = depth_video_dir / "camera_intrinsics.json"
    run_summary_path = depth_video_dir / "run_summary.json"
    object_intrinsics_json_path = object_video_dir / "camera_intrinsics.json"
    depth_npy_path = depth_video_dir / "metric_depth" / "metric_depth.npy"

    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found: {run_summary_path}")
    if not depth_intrinsics_json_path.exists():
        raise FileNotFoundError(
            f"camera_intrinsics.json not found: {depth_intrinsics_json_path}"
        )
    if not depth_npy_path.exists():
        raise FileNotFoundError(f"metric_depth.npy not found: {depth_npy_path}")

    run_summary = load_json(run_summary_path)
    frame_00_raw = None
    outputs = run_summary.get("outputs")
    if isinstance(outputs, dict):
        frame_00_raw = outputs.get("frame_00")
    if frame_00_raw is None:
        frame_00_raw = run_summary.get("frame_00")
    if not isinstance(frame_00_raw, str) or not frame_00_raw.strip():
        raise KeyError(
            f"'frame_00' is missing or invalid in run_summary.json: {run_summary_path}"
        )

    frame_path = Path(frame_00_raw)
    if not frame_path.is_absolute():
        frame_path = (depth_video_dir / frame_path).resolve()
    else:
        frame_path = frame_path.resolve()
    if not frame_path.exists():
        raise FileNotFoundError(f"frame_00 image not found: {frame_path}")

    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read frame_00 image: {frame_path}")

    depth_obs = np.load(depth_npy_path).astype(np.float32)
    depth_h, depth_w = depth_obs.shape

    depth_intr = load_json(depth_intrinsics_json_path)
    k_object_full = load_object_intrinsics(object_intrinsics_json_path)
    k_depth_full = ensure_3x3_intrinsics(depth_intr.get("intrinsics_pixels_3x3"))
    if args.intrinsics_source == "object":
        k_full = k_object_full
    else:
        k_full = k_depth_full

    k_diff = np.abs(k_object_full - k_depth_full)
    max_k_diff = float(np.max(k_diff))

    assets: list[MeshAsset] = []
    for obj_dir in sorted(object_video_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        mesh_path = obj_dir / "mesh_posed.ply"
        if not mesh_path.exists():
            continue

        object_dir_name = obj_dir.name
        obj_name = object_dir_name.replace("_", " ")
        mask_dir = (
            segmentation_video_dir
            / "objects"
            / object_dir_name
            / "object_segmentation"
            / "masks"
        ).resolve()
        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Object mask dir not found for '{obj_name}': {mask_dir}"
            )
        mask_path = resolve_frame_0000_mask(mask_dir)

        verts_src, faces = load_mesh(mesh_path)
        mask = load_binary_mask(mask_path, (depth_h, depth_w))
        mask = erode_mask(mask, int(args.sam3_mask_erode_iters))
        source_to_cv = F_P3D_TO_CV.copy().astype(np.float32)
        assets.append(
            MeshAsset(
                name=obj_name,
                slug=slugify(obj_name),
                kind="object",
                source_mesh_path=mesh_path,
                source_coord="pytorch3d_camera",
                verts_source=verts_src,
                faces=faces,
                source_to_cv=source_to_cv,
                mask_path=mask_path,
                mask=mask,
            )
        )

    if len(assets) == 0:
        raise RuntimeError("No object meshes found to align.")

    human_mesh_path = find_first_human_ply(human_video_dir / "output_plys")
    if not human_mesh_path.exists():
        raise FileNotFoundError(f"Human mesh not found: {human_mesh_path}")
    if human_mesh_path.suffix.lower() != ".ply":
        raise ValueError(f"Human mesh must be a .ply file, got: {human_mesh_path}")

    humans_dir = (segmentation_video_dir / "humans").resolve()
    if not humans_dir.exists():
        raise FileNotFoundError(f"Humans mask dir not found: {humans_dir}")
    human_seg_dir = (humans_dir / "person_1").resolve()
    if not human_seg_dir.exists():
        human_candidates = sorted(d for d in humans_dir.iterdir() if d.is_dir())
        if len(human_candidates) == 0:
            raise FileNotFoundError(f"No human mask folders found in: {humans_dir}")
        human_seg_dir = human_candidates[0].resolve()

    human_mask_dir = (human_seg_dir / "masks").resolve()
    if not human_mask_dir.exists():
        raise FileNotFoundError(f"Human mask dir not found: {human_mask_dir}")

    human_mask_path = resolve_frame_0000_mask(human_mask_dir)
    human_mask = load_binary_mask(human_mask_path, (depth_h, depth_w))
    human_mask = erode_mask(human_mask, int(args.sam3_mask_erode_iters))

    human_verts_src, human_faces = load_mesh(human_mesh_path)
    assets = [
        MeshAsset(
            name="human",
            slug="human",
            kind="human",
            source_mesh_path=human_mesh_path,
            source_coord="opencv_camera",
            verts_source=human_verts_src,
            faces=human_faces,
            source_to_cv=np.eye(3, dtype=np.float32),
            mask_path=human_mask_path,
            mask=human_mask,
        )
    ] + assets

    names = [a.name for a in assets]
    print(f"Loaded meshes: {names}")

    masks_full = [a.mask for a in assets]
    depth_opt, masks_opt, frame_opt, k_opt, resize_scale = maybe_resize_for_optimization(
        depth=depth_obs,
        masks=masks_full,
        frame=frame_bgr,
        k=k_full,
        opt_max_side=int(args.opt_max_side),
    )

    for asset, m in zip(assets, masks_opt):
        asset.mask = m

    print(
        f"Optimization resolution: {depth_opt.shape[1]}x{depth_opt.shape[0]} "
        f"(scale={resize_scale:.4f})"
    )

    device = parse_device(args.device)
    print(f"Using device: {device}")

    verts_base_cv_np: list[np.ndarray] = []
    for asset in assets:
        verts_cv = (asset.verts_source @ asset.source_to_cv.transpose(0, 1)).astype(
            np.float32
        )
        verts_base_cv_np.append(verts_cv)

    overlay_before = render_quality_overlay_from_cv_meshes(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_base_cv_np,
        faces_list=[a.faces for a in assets],
        names=names,
        k=k_full,
        device=device,
    )
    cv2.imwrite(str(output_dir / "overlay_before.png"), overlay_before)

    observations: list[ObservationSet] = []
    model_samples_list: list[np.ndarray] = []
    observation_stats: list[dict[str, Any]] = []

    for idx, (asset, verts_cv) in enumerate(zip(assets, verts_base_cv_np)):
        obs = build_observation_set(
            depth=depth_opt,
            intrinsics=k_opt,
            mask=asset.mask,
            max_obs_3d_points=int(args.max_obs_3d_points_per_mesh),
            max_obs_2d_points=int(args.max_obs_2d_points_per_mesh),
            seed=int(args.seed + 1009 * (idx + 1)),
        )
        observations.append(obs)

        model_samples = sample_mesh_surface_points(
            verts_cv=verts_cv,
            faces=asset.faces,
            n_samples=int(args.mesh_sample_points),
            seed=int(args.seed + 4049 * (idx + 1)),
        )
        model_samples_list.append(model_samples)

        observation_stats.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "mask_path": None if asset.mask_path is None else str(asset.mask_path),
                "mask_pixels_total": int(obs.mask_pixels_total),
                "depth_valid_pixels_total": int(obs.depth_valid_pixels_total),
                "obs_3d_used": int(obs.obs_3d_used),
                "obs_2d_used": int(obs.obs_2d_used),
                "model_surface_samples": int(model_samples.shape[0]),
                "model_sample_count_total": int(model_samples.shape[0]),
            }
        )

        print(
            f"[{asset.name}] obs3d={obs.obs_3d_used}/{obs.depth_valid_pixels_total} "
            f"obs2d={obs.obs_2d_used}/{obs.mask_pixels_total} "
            f"model_samples={model_samples.shape[0]}"
        )

    losses_dir = output_dir / "loss_curves"
    debug_root = output_dir / "debug"
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    optimization_results: list[OptimizationResultChamfer] = []
    loss_plot_paths_by_slug: dict[str, dict[str, str]] = {}

    for idx, (asset, obs, model_samples, verts_cv) in enumerate(
        zip(assets, observations, model_samples_list, verts_base_cv_np)
    ):
        debug_dir = debug_root / asset.slug
        checkpoint_path = checkpoints_dir / f"{asset.slug}_resume.pt"
        resume_signature = build_resume_signature(
            video_name=args.video_name,
            mesh_slug=asset.slug,
            mesh_name=asset.name,
            intrinsics_source=args.intrinsics_source,
            opt_max_side=int(args.opt_max_side),
            sam3_mask_erode_iters=int(args.sam3_mask_erode_iters),
            seed=int(args.seed),
            lr=float(args.lr),
            w_cd3d=float(args.w_cd3d),
            w_cd2d=float(args.w_cd2d),
            w_scale_reg=float(args.w_scale_reg),
            w_t_reg=float(args.w_t_reg),
            rho_geman_3d=float(args.rho_geman_3d),
            mesh_sample_points=int(args.mesh_sample_points),
            max_obs_3d_points_per_mesh=int(args.max_obs_3d_points_per_mesh),
            max_obs_2d_points_per_mesh=int(args.max_obs_2d_points_per_mesh),
            nn_chunk_size=int(args.nn_chunk_size),
            min_obs_3d_points_per_mesh=int(args.min_obs_3d_points_per_mesh),
            min_obs_2d_points_per_mesh=int(args.min_obs_2d_points_per_mesh),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
            max_abs_delta_tx=float(args.max_abs_delta_tx),
            max_abs_delta_ty=float(args.max_abs_delta_ty),
            max_abs_delta_tz=float(args.max_abs_delta_tz),
            visible_subset_z_abs_tol_m=float(args.visible_subset_z_abs_tol_m),
            visible_subset_z_rel_tol=float(args.visible_subset_z_rel_tol),
            min_visible_subset_points_per_mesh=int(
                args.min_visible_subset_points_per_mesh
            ),
            visible_subset_focal_scale=float(args.visible_subset_focal_scale),
        )

        resume_state: ResumeStateChamfer | None = None
        if not args.no_resume and checkpoint_path.exists():
            resume_state = load_resume_checkpoint(checkpoint_path, device=device)
            mismatches = compare_resume_signature_strict(
                saved_signature=resume_state.signature,
                current_signature=resume_signature,
            )
            if len(mismatches) > 0:
                details = "\n  - ".join([""] + mismatches)
                raise RuntimeError(
                    f"[{asset.name}] checkpoint signature mismatch for {checkpoint_path}:{details}"
                )
            print(
                f"[{asset.name}] resuming from iter {resume_state.start_iter} "
                f"for +{int(args.iters)} iterations ..."
            )
        else:
            print(f"[{asset.name}] optimizing from scratch for {int(args.iters)} iterations ...")

        result = optimize_scale_txyz_chamfer(
            mesh_name=asset.name,
            obs=obs,
            model_points_base_np=model_samples,
            verts_cv_np=verts_cv,
            faces_np=asset.faces,
            intrinsics=k_opt,
            frame_bgr_for_debug=frame_opt,
            debug_dir=debug_dir,
            device=device,
            iters=int(args.iters),
            lr=float(args.lr),
            w_cd3d=float(args.w_cd3d),
            w_cd2d=float(args.w_cd2d),
            w_scale_reg=float(args.w_scale_reg),
            w_t_reg=float(args.w_t_reg),
            rho_geman_3d=float(args.rho_geman_3d),
            nn_chunk_size=int(args.nn_chunk_size),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
            max_abs_delta_tx=float(args.max_abs_delta_tx),
            max_abs_delta_ty=float(args.max_abs_delta_ty),
            max_abs_delta_tz=float(args.max_abs_delta_tz),
            log_every=int(args.log_every),
            debug_save_every=int(args.debug_save_every),
            debug_point_radius=int(args.debug_point_radius),
            debug_max_points_vis=int(args.debug_max_points_vis),
            debug_max_nn_lines=int(args.debug_max_nn_lines),
            debug_seed=int(args.seed + 2029 * (idx + 1)),
            min_obs_3d_points_per_mesh=int(args.min_obs_3d_points_per_mesh),
            min_obs_2d_points_per_mesh=int(args.min_obs_2d_points_per_mesh),
            checkpoint_path=checkpoint_path,
            resume_signature=resume_signature,
            resume_state=resume_state,
            checkpoint_every=int(args.checkpoint_every),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_rel_min_delta=float(args.early_stop_rel_min_delta),
            early_stop_min_iter=int(args.early_stop_min_iter),
            visible_subset_z_abs_tol_m=float(args.visible_subset_z_abs_tol_m),
            visible_subset_z_rel_tol=float(args.visible_subset_z_rel_tol),
            min_visible_subset_points_per_mesh=int(
                args.min_visible_subset_points_per_mesh
            ),
            visible_subset_focal_scale=float(args.visible_subset_focal_scale),
        )
        optimization_results.append(result)

        if len(result.history["iter"]) > 0:
            loss_plot_paths_by_slug[asset.slug] = plot_loss_curves_chamfer(
                history=result.history,
                out_dir=losses_dir,
                slug=asset.slug,
                mesh_name=asset.name,
            )
            save_loss_history_csv_chamfer(
                history=result.history,
                out_path=losses_dir / f"{asset.slug}_loss.csv",
            )

    meshes_out_dir = output_dir / "meshes"
    meshes_out_dir.mkdir(parents=True, exist_ok=True)

    transforms_out: list[dict[str, Any]] = []
    verts_after_np: list[np.ndarray] = []

    cv_to_output = (
        np.eye(3, dtype=np.float32)
        if args.output_coord == "opencv"
        else F_CV_TO_P3D.copy()
    )

    for asset, verts_cv, result in zip(assets, verts_base_cv_np, optimization_results):
        scale = float(result.scale)
        tx = float(result.tx)
        ty = float(result.ty)
        tz = float(result.tz)

        verts_aligned_cv = (scale * verts_cv).astype(np.float32)
        verts_aligned_cv[:, 0] += tx
        verts_aligned_cv[:, 1] += ty
        verts_aligned_cv[:, 2] += tz
        verts_after_np.append(verts_aligned_cv)

        verts_aligned_out = (verts_aligned_cv @ cv_to_output.transpose(0, 1)).astype(
            np.float32
        )
        out_mesh_path = meshes_out_dir / f"{asset.slug}.ply"
        save_mesh_ply(out_mesh_path, verts_aligned_out, asset.faces)

        source_to_output_4x4 = np.eye(4, dtype=np.float32)
        source_to_output_4x4[:3, :3] = (
            scale * (cv_to_output @ asset.source_to_cv.astype(np.float32))
        )
        t_cv = np.array([tx, ty, tz], dtype=np.float32)
        source_to_output_4x4[:3, 3] = t_cv @ cv_to_output.transpose(0, 1)

        transforms_out.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "source_mesh_path": str(asset.source_mesh_path),
                "source_coordinate": asset.source_coord,
                "output_coordinate": args.output_coord,
                "aligned_mesh_ply": str(out_mesh_path),
                "optimized_parameters": {
                    "status": result.status,
                    "message": result.message,
                    "log_scale_alpha": float(result.log_scale),
                    "scale_exp_alpha": float(scale),
                    "tx_init_m": float(result.tx_init),
                    "ty_init_m": float(result.ty_init),
                    "tz_init_m": float(result.tz_init),
                    "delta_tx_m": float(result.delta_tx),
                    "delta_ty_m": float(result.delta_ty),
                    "delta_tz_m": float(result.delta_tz),
                    "tx_total_m": float(tx),
                    "ty_total_m": float(ty),
                    "tz_total_m": float(tz),
                    "obs_3d_count": int(result.obs_3d_count),
                    "obs_2d_count": int(result.obs_2d_count),
                    "model_sample_count": int(result.model_sample_count),
                    "model_sample_count_total": int(result.model_sample_count_total),
                    "model_sample_count_visible_fixed": int(
                        result.model_sample_count_visible_fixed
                    ),
                    "model_sample_visible_ratio": float(result.model_sample_visible_ratio),
                    "visible_subset_mode": result.visible_subset_mode,
                },
                "source_to_output_matrix_4x4": source_to_output_4x4.tolist(),
            }
        )

    overlay_after = render_quality_overlay_from_cv_meshes(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_after_np,
        faces_list=[a.faces for a in assets],
        names=names,
        k=k_full,
        device=device,
    )
    cv2.imwrite(str(output_dir / "overlay_after.png"), overlay_after)

    summary_out = {
        "inputs": {
            "video_name": args.video_name,
            "object_video_dir": str(object_video_dir),
            "segmentation_video_dir": str(segmentation_video_dir),
            "depth_video_dir": str(depth_video_dir),
            "human_video_dir": str(human_video_dir),
            "depth_npy": str(depth_npy_path),
            "depth_intrinsics_json": str(depth_intrinsics_json_path),
            "frame_00": str(frame_path),
        },
        "camera": {
            "intrinsics_source": args.intrinsics_source,
            "intrinsics_3x3": k_full.tolist(),
            "max_abs_object_depth_intrinsics_diff_px": max_k_diff,
        },
        "optimization_settings": {
            "iters": int(args.iters),
            "iters_semantics_with_resume": "additive",
            "lr": float(args.lr),
            "w_cd3d": float(args.w_cd3d),
            "w_cd2d": float(args.w_cd2d),
            "w_scale_reg": float(args.w_scale_reg),
            "w_t_reg": float(args.w_t_reg),
            "rho_geman_3d": float(args.rho_geman_3d),
            "mesh_sample_points": int(args.mesh_sample_points),
            "max_obs_3d_points_per_mesh": int(args.max_obs_3d_points_per_mesh),
            "max_obs_2d_points_per_mesh": int(args.max_obs_2d_points_per_mesh),
            "nn_chunk_size": int(args.nn_chunk_size),
            "min_obs_3d_points_per_mesh": int(args.min_obs_3d_points_per_mesh),
            "min_obs_2d_points_per_mesh": int(args.min_obs_2d_points_per_mesh),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
            "max_abs_delta_tx": float(args.max_abs_delta_tx),
            "max_abs_delta_ty": float(args.max_abs_delta_ty),
            "max_abs_delta_tz": float(args.max_abs_delta_tz),
            "visible_subset_z_abs_tol_m": float(args.visible_subset_z_abs_tol_m),
            "visible_subset_z_rel_tol": float(args.visible_subset_z_rel_tol),
            "min_visible_subset_points_per_mesh": int(
                args.min_visible_subset_points_per_mesh
            ),
            "visible_subset_focal_scale": float(args.visible_subset_focal_scale),
            "no_resume": bool(args.no_resume),
            "checkpoint_every": int(args.checkpoint_every),
            "early_stop_patience": int(args.early_stop_patience),
            "early_stop_rel_min_delta": float(args.early_stop_rel_min_delta),
            "early_stop_min_iter": int(args.early_stop_min_iter),
            "debug_save_every": int(args.debug_save_every),
            "debug_point_radius": int(args.debug_point_radius),
            "debug_max_points_vis": int(args.debug_max_points_vis),
            "debug_max_nn_lines": int(args.debug_max_nn_lines),
            "opt_max_side": int(args.opt_max_side),
            "resize_scale_for_optimization": float(resize_scale),
            "output_coordinate": args.output_coord,
            "sam3_mask_erode_iters": int(args.sam3_mask_erode_iters),
        },
        "energy_terms": {
            "transform": "p' = exp(alpha) * p + [tx, ty, tz]^T",
            "obs3d": "O3D = depth back-projected points inside asset mask",
            "obs2d": "O2D = mask pixels (u,v)",
            "model3d_all": "M3D_all = sampled mesh surface points",
            "model3d_visible_fixed": "M3D_vis = fixed visible subset sampled once from one-time render at init",
            "model2d_visible_fixed": "M2D_vis' = project(M3D_vis')",
            "L_cd3d_raw_fwd": "mean_{o in O3D} min_{m in M3D_vis'} ||o-m||_2^2",
            "L_cd3d_raw_bwd": "mean_{m in M3D_vis'} min_{o in O3D} ||m-o||_2^2",
            "L_cd3d_fwd": "GemanMcClure(L_cd3d_raw_fwd)",
            "L_cd3d_bwd": "GemanMcClure(L_cd3d_raw_bwd)",
            "L_cd3d": "0.5 * (L_cd3d_fwd + L_cd3d_bwd)",
            "L_cd2d_fwd": "mean_{o in O2D} min_{m in M2D_vis'} ||o-m||_2^2",
            "L_cd2d_bwd": "mean_{m in M2D_vis'} min_{o in O2D} ||m-o||_2^2",
            "L_cd2d": "0.5 * (L_cd2d_fwd + L_cd2d_bwd)",
            "L_scale_reg": "alpha^2",
            "L_t_reg": "tx^2 + ty^2 + tz^2",
            "L_total": "w_cd3d*L_cd3d + w_cd2d*L_cd2d + w_scale_reg*L_scale_reg + w_t_reg*L_t_reg",
        },
        "observation_stats": observation_stats,
        "per_mesh_optimization": [
            {
                "name": asset.name,
                "slug": asset.slug,
                "status": result.status,
                "message": result.message,
                "obs_3d_count": int(result.obs_3d_count),
                "obs_2d_count": int(result.obs_2d_count),
                "model_sample_count": int(result.model_sample_count),
                "model_sample_count_total": int(result.model_sample_count_total),
                "model_sample_count_visible_fixed": int(
                    result.model_sample_count_visible_fixed
                ),
                "model_sample_visible_ratio": float(result.model_sample_visible_ratio),
                "visible_subset_mode": result.visible_subset_mode,
                "final_scale": float(result.scale),
                "final_tx_m": float(result.tx),
                "final_ty_m": float(result.ty),
                "final_tz_m": float(result.tz),
                "final_total_loss": result.final_total_loss,
                "final_cd3d_loss": result.final_cd3d_loss,
                "final_cd2d_loss": result.final_cd2d_loss,
                "final_cd3d_fwd_loss": None
                if len(result.history["cd3d_fwd"]) == 0
                else float(result.history["cd3d_fwd"][-1]),
                "final_cd3d_bwd_loss": None
                if len(result.history["cd3d_bwd"]) == 0
                else float(result.history["cd3d_bwd"][-1]),
                "final_cd2d_fwd_loss": None
                if len(result.history["cd2d_fwd"]) == 0
                else float(result.history["cd2d_fwd"][-1]),
                "final_cd2d_bwd_loss": None
                if len(result.history["cd2d_bwd"]) == 0
                else float(result.history["cd2d_bwd"][-1]),
                "resumed": bool(result.resumed),
                "resume_source": result.resume_source,
                "start_iter": int(result.start_iter),
                "end_iter": int(result.end_iter),
                "iters_executed_this_run": int(result.iters_executed_this_run),
                "early_stopped": bool(result.early_stopped),
                "early_stop_iter": result.early_stop_iter,
                "best_iter": result.best_iter,
                "best_total": result.best_total,
                "checkpoint_path": result.checkpoint_path,
                "loss_curve_png": (
                    loss_plot_paths_by_slug.get(asset.slug, {}).get("total")
                    if len(result.history["iter"]) > 0
                    else None
                ),
                "loss_curve_pngs": (
                    loss_plot_paths_by_slug.get(asset.slug, {})
                    if len(result.history["iter"]) > 0
                    else None
                ),
                "loss_curve_csv": str(losses_dir / f"{asset.slug}_loss.csv")
                if len(result.history["iter"]) > 0
                else None,
                "debug_dir": str(debug_root / asset.slug),
            }
            for asset, result in zip(assets, optimization_results)
        ],
        "outputs": {
            "output_dir": str(output_dir),
            "meshes_dir": str(meshes_out_dir),
            "mesh_format": "ply",
            "output_coordinate": args.output_coord,
            "overlay_before": str(output_dir / "overlay_before.png"),
            "overlay_after": str(output_dir / "overlay_after.png"),
            "debug_root": str(debug_root),
            "checkpoints_dir": str(checkpoints_dir),
        },
    }

    with open(output_dir / "alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)
    with open(meshes_out_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump({"transforms": transforms_out}, f, indent=2)

    print(f"Saved outputs to: {output_dir}")
    print(f"Saved aligned meshes to: {meshes_out_dir}")
    print(f"Saved summary to: {output_dir / 'alignment_summary.json'}")


if __name__ == "__main__":
    main()
