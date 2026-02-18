"""Align human and object meshes to masked depth point clouds in camera space.

This script is a 3D-only alternative to align_meshes.py.
It aligns each mesh independently to its own masked depth point cloud using
trimmed bidirectional Chamfer distance with robust losses.

High-level pipeline:
1. Load object meshes/masks from Generate_Object_Mesh outputs and human mesh/mask.
2. Load camera intrinsics and observed depth for frame_00.
3. Build per-mesh target point clouds from depth + masks.
4. Run staged per-mesh similarity optimization in OpenCV camera coordinates.
5. Export aligned OBJ meshes, transforms, overlays, and JSON summaries.
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
import trimesh
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

# GLB meshes are Y-up by convention; SAM3D camera transforms are effectively in Z-up.
R_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# PyTorch3D camera (+X left, +Y up, +Z forward) <-> OpenCV (+X right, +Y down, +Z forward)
F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


@dataclass
class MeshAsset:
    name: str
    slug: str
    kind: str
    source_mesh_path: Path
    source_coord: str
    verts_source: np.ndarray
    faces: np.ndarray
    source_to_cv: np.ndarray
    mask_path: Path | None
    mask: np.ndarray | None


@dataclass
class MeshState:
    log_s: torch.nn.Parameter
    rotvec: torch.nn.Parameter
    tvec: torch.nn.Parameter
    tvec_init: torch.Tensor
    sample_points_base: torch.Tensor
    target_points: torch.Tensor | None
    target_points_total: int
    target_points_used: int
    target_z_median: torch.Tensor | None
    active: bool
    status: str
    message: str | None


def slugify(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "mesh"


def resolve_path(path_str: str | None, base_dir: Path) -> Path | None:
    if path_str is None:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_3x3_intrinsics(raw: Any) -> np.ndarray:
    arr = np.array(raw, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {arr.shape}")
    return arr


def ensure_3x4_extrinsics(raw: Any | None) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.array(raw, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.shape != (3, 4):
        raise ValueError(f"Expected extrinsics shape (3,4), got {arr.shape}")
    return arr


def load_object_intrinsics(camera_intrinsics_json_path: Path) -> np.ndarray:
    if not camera_intrinsics_json_path.exists():
        raise FileNotFoundError(
            f"camera_intrinsics.json not found: {camera_intrinsics_json_path}"
        )
    camera_intr = load_json(camera_intrinsics_json_path)
    if "intrinsics_pixels_3x3" not in camera_intr:
        raise KeyError(
            f"Missing 'intrinsics_pixels_3x3' in {camera_intrinsics_json_path}"
        )
    return ensure_3x3_intrinsics(camera_intr.get("intrinsics_pixels_3x3"))


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def load_binary_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    target_h, target_w = target_hw
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def find_first_human_obj(output_objs_dir: Path) -> Path:
    obj_paths = sorted(output_objs_dir.glob("*.obj"))
    if not obj_paths:
        raise FileNotFoundError(f"No .obj files found in {output_objs_dir}")
    return obj_paths[0]


def parse_device(device_str: str) -> torch.device:
    device_str = device_str.strip()
    if not device_str:
        return torch.device("cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested but torch.cuda.is_available() is False."
        )
    return torch.device(device_str)


def build_cameras(
    k: np.ndarray, width: int, height: int, device: torch.device
) -> PerspectiveCameras:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    return PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size=torch.tensor([[height, width]], dtype=torch.float32, device=device),
        in_ndc=False,
        device=device,
    )


def build_hard_rasterizer(
    cameras: PerspectiveCameras,
    image_size: tuple[int, int],
    bin_size: int,
) -> MeshRasterizer:
    hard_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),
        max_faces_per_bin=300000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=hard_settings)


def maybe_resize_for_optimization(
    depth: np.ndarray,
    masks: list[np.ndarray | None],
    k: np.ndarray,
    opt_max_side: int,
) -> tuple[np.ndarray, list[np.ndarray | None], np.ndarray, float]:
    h, w = depth.shape
    if opt_max_side <= 0 or max(h, w) <= opt_max_side:
        return depth, masks, k, 1.0

    scale = float(opt_max_side) / float(max(h, w))
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))

    depth_rs = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    masks_rs: list[np.ndarray | None] = []
    for m in masks:
        if m is None:
            masks_rs.append(None)
            continue
        mr = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        masks_rs.append((mr > 0.5).astype(np.float32))

    k_rs = k.copy()
    k_rs[0, :] *= scale
    k_rs[1, :] *= scale
    return depth_rs.astype(np.float32), masks_rs, k_rs.astype(np.float32), scale


def transform_vertices_list(
    verts_base_cv: list[torch.Tensor],
    log_s: torch.Tensor,
    rotvec: torch.Tensor,
    tvec: torch.Tensor,
) -> list[torch.Tensor]:
    rots = axis_angle_to_matrix(rotvec)  # (J,3,3)
    scales = torch.exp(log_s)  # (J,)
    out = []
    for j, verts in enumerate(verts_base_cv):
        v = scales[j] * (verts @ rots[j].transpose(0, 1)) + tvec[j]
        out.append(v)
    return out


def transform_points_single(
    points_base: torch.Tensor,
    log_s: torch.Tensor,
    rotvec: torch.Tensor,
    tvec: torch.Tensor,
) -> torch.Tensor:
    rot = axis_angle_to_matrix(rotvec.unsqueeze(0))[0]
    scale = torch.exp(log_s)
    return scale * (points_base @ rot.transpose(0, 1)) + tvec


def render_scene_depth_and_mesh_id(
    hard_rasterizer: MeshRasterizer,
    verts_cv: list[torch.Tensor],
    faces: list[torch.Tensor],
    cv_to_p3d: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    verts_p3d = [v @ cv_to_p3d.transpose(0, 1) for v in verts_cv]

    verts_cat = []
    faces_cat = []
    face_to_mesh = []
    offset = 0
    for mesh_idx, (v, f) in enumerate(zip(verts_p3d, faces)):
        verts_cat.append(v)
        faces_cat.append(f + offset)
        face_to_mesh.append(
            torch.full((f.shape[0],), mesh_idx, dtype=torch.long, device=device)
        )
        offset += int(v.shape[0])

    scene_mesh = Meshes(
        verts=[torch.cat(verts_cat, dim=0)],
        faces=[torch.cat(faces_cat, dim=0)],
    )
    fragments = hard_rasterizer(scene_mesh)
    pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H,W)
    zbuf = fragments.zbuf[0, ..., 0]  # (H,W)

    valid = pix_to_face >= 0
    depth = torch.zeros_like(zbuf)
    depth[valid] = zbuf[valid]

    mesh_id = torch.full_like(pix_to_face, -1)
    if bool(valid.any()):
        face_to_mesh_cat = torch.cat(face_to_mesh, dim=0)
        mesh_id[valid] = face_to_mesh_cat[pix_to_face[valid]]
    return depth, mesh_id


def huber_loss(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_r = residual.abs()
    return torch.where(
        abs_r <= delta,
        0.5 * residual * residual,
        delta * (abs_r - 0.5 * delta),
    )


def save_mesh_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(path))


def export_meshes_and_transforms(
    assets: list[MeshAsset],
    verts_aligned: list[np.ndarray],
    rotvec_axis_angle: np.ndarray,
    rots_np: np.ndarray,
    scales_np: np.ndarray,
    t_np: np.ndarray,
    meshes_out_dir: Path,
    transforms_out_path: Path,
) -> list[dict[str, Any]]:
    meshes_out_dir.mkdir(parents=True, exist_ok=True)

    transforms_out: list[dict[str, Any]] = []
    for j, asset in enumerate(assets):
        out_mesh_path = meshes_out_dir / f"{asset.slug}.obj"
        save_mesh_obj(out_mesh_path, verts_aligned[j], asset.faces)

        c = asset.source_to_cv.astype(np.float32)
        r = rots_np[j].astype(np.float32)
        s = float(scales_np[j])
        t = t_np[j].astype(np.float32)

        source_to_cv_4x4 = np.eye(4, dtype=np.float32)
        source_to_cv_4x4[:3, :3] = c

        source_to_aligned_4x4 = np.eye(4, dtype=np.float32)
        source_to_aligned_4x4[:3, :3] = (s * r @ c).astype(np.float32)
        source_to_aligned_4x4[:3, 3] = t

        transforms_out.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "source_mesh_path": str(asset.source_mesh_path),
                "source_coordinate": asset.source_coord,
                "source_to_cv_rotation_3x3": c.tolist(),
                "source_to_cv_matrix_4x4": source_to_cv_4x4.tolist(),
                "optimized_similarity_in_cv": {
                    "scale": s,
                    "rotation_axis_angle": rotvec_axis_angle[j].tolist(),
                    "rotation_matrix_3x3": r.tolist(),
                    "translation_xyz_m": t.tolist(),
                },
                "source_to_aligned_matrix_4x4": source_to_aligned_4x4.tolist(),
                "aligned_mesh_obj": str(out_mesh_path),
            }
        )

    with open(transforms_out_path, "w", encoding="utf-8") as f:
        json.dump({"transforms": transforms_out}, f, indent=2)
    return transforms_out


def project_points_cv(
    points_cv: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_cv[:, 2]
    valid = z > 1e-6
    pts = points_cv[valid]
    z = z[valid]
    u = (pts[:, 0] * k[0, 0]) / z + k[0, 2]
    v = (pts[:, 1] * k[1, 1]) / z + k[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui[inb], vi[inb], valid


def draw_overlay_points(
    frame_bgr: np.ndarray,
    verts_cv_list: list[np.ndarray],
    names: list[str],
    k: np.ndarray,
    max_points_per_mesh: int = 25000,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    palette = [
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 128, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    for idx, (verts, name) in enumerate(zip(verts_cv_list, names)):
        if len(verts) > max_points_per_mesh:
            stride = max(1, int(len(verts) / max_points_per_mesh))
            pts = verts[::stride]
        else:
            pts = verts

        u, v, _ = project_points_cv(pts, k, w, h)
        color = palette[idx % len(palette)]
        for x, y in zip(u, v):
            cv2.circle(out, (int(x), int(y)), 1, color, -1)
        cv2.putText(
            out,
            name,
            (12, 28 + 24 * idx),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def per_mesh_depth_stats(
    depth_render: torch.Tensor,
    mesh_id: torch.Tensor,
    depth_obs: torch.Tensor,
    masks: list[torch.Tensor | None],
    names: list[str],
    use_visibility: bool,
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    valid_depth = depth_obs > 0.0

    for j, name in enumerate(names):
        mask = masks[j]
        if mask is None:
            stats.append(
                {
                    "name": name,
                    "pixels": 0,
                    "mean_residual_m": None,
                    "median_residual_m": None,
                    "mad_residual_m": None,
                }
            )
            continue

        pix = valid_depth & (mask > 0.5)
        if use_visibility:
            pix = pix & (mesh_id == j)

        n_pix = int(pix.sum().item())
        if n_pix == 0:
            stats.append(
                {
                    "name": name,
                    "pixels": 0,
                    "mean_residual_m": None,
                    "median_residual_m": None,
                    "mad_residual_m": None,
                }
            )
            continue

        r = (depth_render[pix] - depth_obs[pix]).detach().cpu().numpy()
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)))
        stats.append(
            {
                "name": name,
                "pixels": n_pix,
                "mean_residual_m": float(np.mean(r)),
                "median_residual_m": med,
                "mad_residual_m": mad,
            }
        )
    return stats


def pointcloud_stats(points: np.ndarray) -> dict[str, Any]:
    if points.shape[0] == 0:
        return {
            "num_points": 0,
            "bounds_min_xyz": [0.0, 0.0, 0.0],
            "bounds_max_xyz": [0.0, 0.0, 0.0],
            "median_z_m": None,
        }
    bmin = points.min(axis=0).astype(np.float32)
    bmax = points.max(axis=0).astype(np.float32)
    return {
        "num_points": int(points.shape[0]),
        "bounds_min_xyz": bmin.tolist(),
        "bounds_max_xyz": bmax.tolist(),
        "median_z_m": float(np.median(points[:, 2])),
    }


def masked_depth_to_pointcloud_cv(
    depth: np.ndarray,
    mask: np.ndarray | None,
    intrinsics: np.ndarray,
) -> np.ndarray:
    if mask is None:
        return np.zeros((0, 3), dtype=np.float32)

    h, w = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    yy, xx = np.indices((h, w), dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0) & (mask > 0.5)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    z = depth[valid]
    x = ((xx[valid] - cx) / fx) * z
    y = ((yy[valid] - cy) / fy) * z
    return np.stack((x, y, z), axis=-1).astype(np.float32)


def downsample_points(
    points: np.ndarray, max_points: int, rng: np.random.Generator
) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    if faces.shape[0] == 0:
        raise ValueError("Mesh has no faces for surface sampling.")

    tri = verts[faces]  # (F,3,3)
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
    area_sum = float(np.sum(areas))

    if not np.isfinite(area_sum) or area_sum <= 1e-12:
        vidx = rng.choice(verts.shape[0], size=num_samples, replace=True)
        return verts[vidx].astype(np.float32)

    probs = areas / area_sum
    face_idx = rng.choice(faces.shape[0], size=num_samples, replace=True, p=probs)
    chosen = tri[face_idx]

    u = rng.random((num_samples, 1), dtype=np.float32)
    v = rng.random((num_samples, 1), dtype=np.float32)
    su = np.sqrt(u)
    bary0 = 1.0 - su
    bary1 = su * (1.0 - v)
    bary2 = su * v
    pts = (
        bary0 * chosen[:, 0]
        + bary1 * chosen[:, 1]
        + bary2 * chosen[:, 2]
    )
    return pts.astype(np.float32)


def nearest_neighbor_distances(
    src: torch.Tensor,
    dst: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 3 or dst.shape[1] != 3:
        raise ValueError("src and dst must be Nx3 / Mx3 tensors.")
    if src.shape[0] == 0:
        return torch.zeros((0,), device=src.device, dtype=src.dtype)
    if dst.shape[0] == 0:
        return torch.full((src.shape[0],), float("inf"), device=src.device, dtype=src.dtype)

    if chunk_size <= 0:
        chunk_size = src.shape[0]

    out = []
    dst_b = dst.unsqueeze(0)
    for start in range(0, src.shape[0], chunk_size):
        s = src[start: start + chunk_size]
        d = torch.cdist(s.unsqueeze(0), dst_b, p=2)[0]
        out.append(d.min(dim=1).values)
    return torch.cat(out, dim=0)


def trimmed_huber(
    distances: torch.Tensor,
    trim_quantile: float,
    delta: float,
) -> tuple[torch.Tensor, int]:
    if distances.numel() == 0:
        return distances.new_zeros(()), 0

    trim_q = float(trim_quantile)
    if trim_q <= 0.0:
        trim_q = 1.0
    if trim_q < 1.0 and distances.numel() > 1:
        keep = max(1, int(math.ceil(trim_q * distances.numel())))
        kept = torch.topk(distances, k=keep, largest=False).values
    else:
        kept = distances
        keep = int(distances.numel())

    return huber_loss(kept, float(delta)).mean(), keep


def compute_mesh_losses(
    state: MeshState,
    trim_quantile: float,
    chamfer_forward_weight: float,
    chamfer_backward_weight: float,
    depth_weight: float,
    reg_weight: float,
    reg_scale: float,
    reg_rot: float,
    reg_trans: float,
    reg_trans_reference: str,
    depth_huber_delta_3d: float,
    depth_anchor_weight: float,
    nn_chunk_size: int,
) -> dict[str, Any]:
    pred = transform_points_single(
        state.sample_points_base, state.log_s, state.rotvec, state.tvec
    )
    if reg_trans_reference == "warmstart":
        trans_reg_vec = state.tvec - state.tvec_init
    else:
        trans_reg_vec = state.tvec

    if state.target_points is None or state.target_points.shape[0] == 0:
        z = pred.new_zeros(())
        l_reg = (
            float(reg_scale) * (state.log_s.pow(2))
            + float(reg_rot) * (state.rotvec.pow(2).mean())
            + float(reg_trans) * (trans_reg_vec.pow(2).mean())
        )
        total = float(reg_weight) * l_reg
        return {
            "total": total,
            "chamfer": z,
            "forward": z,
            "backward": z,
            "anchor": z,
            "reg": l_reg,
            "forward_kept": 0,
            "backward_kept": 0,
        }

    d_fw = nearest_neighbor_distances(pred, state.target_points, nn_chunk_size)
    d_bw = nearest_neighbor_distances(state.target_points, pred, nn_chunk_size)
    l_fw, kept_fw = trimmed_huber(d_fw, trim_quantile, depth_huber_delta_3d)
    l_bw, kept_bw = trimmed_huber(d_bw, trim_quantile, depth_huber_delta_3d)

    l_ch = float(chamfer_forward_weight) * l_fw + float(chamfer_backward_weight) * l_bw
    l_reg = (
        float(reg_scale) * (state.log_s.pow(2))
        + float(reg_rot) * (state.rotvec.pow(2).mean())
        + float(reg_trans) * (trans_reg_vec.pow(2).mean())
    )

    if state.target_z_median is None:
        l_anchor = pred.new_zeros(())
    else:
        z_residual = torch.median(pred[:, 2]) - state.target_z_median
        l_anchor = huber_loss(z_residual, float(depth_huber_delta_3d))

    total = float(depth_weight) * l_ch + float(reg_weight) * l_reg + float(
        depth_anchor_weight
    ) * l_anchor
    return {
        "total": total,
        "chamfer": l_ch,
        "forward": l_fw,
        "backward": l_bw,
        "anchor": l_anchor,
        "reg": l_reg,
        "forward_kept": kept_fw,
        "backward_kept": kept_bw,
    }


def build_intrinsics_mismatch_report(
    k_object: np.ndarray,
    k_depth: np.ndarray,
    warn_threshold_px: float,
) -> dict[str, Any]:
    diff = (k_object - k_depth).astype(np.float32)
    fx_diff = float(abs(diff[0, 0]))
    fy_diff = float(abs(diff[1, 1]))
    cx_diff = float(abs(diff[0, 2]))
    cy_diff = float(abs(diff[1, 2]))
    max_abs = float(np.max(np.abs(diff)))
    warn = (
        fx_diff > float(warn_threshold_px)
        or fy_diff > float(warn_threshold_px)
        or cx_diff > float(warn_threshold_px)
        or cy_diff > float(warn_threshold_px)
    )
    return {
        "object_intrinsics_3x3": k_object.tolist(),
        "depth_intrinsics_3x3": k_depth.tolist(),
        "difference_object_minus_depth_3x3": diff.tolist(),
        "abs_fx_diff_px": fx_diff,
        "abs_fy_diff_px": fy_diff,
        "abs_cx_diff_px": cx_diff,
        "abs_cy_diff_px": cy_diff,
        "max_abs_diff": max_abs,
        "warn_threshold_px": float(warn_threshold_px),
        "warning": bool(warn),
    }


def stacked_params(states: list[MeshState]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_s = torch.stack([s.log_s for s in states], dim=0)
    rotvec = torch.stack([s.rotvec for s in states], dim=0)
    tvec = torch.stack([s.tvec for s in states], dim=0)
    return log_s, rotvec, tvec


def get_transform_arrays(states: list[MeshState]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    log_s_np = np.array([float(s.log_s.detach().cpu().item()) for s in states], dtype=np.float32)
    scales_np = np.exp(log_s_np)
    rotvec_np = np.stack([s.rotvec.detach().cpu().numpy() for s in states], axis=0).astype(
        np.float32
    )
    t_np = np.stack([s.tvec.detach().cpu().numpy() for s in states], axis=0).astype(np.float32)
    rots_np = axis_angle_to_matrix(torch.from_numpy(rotvec_np)).cpu().numpy().astype(np.float32)
    return log_s_np, scales_np, rotvec_np, rots_np, t_np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align human and object meshes into a single OpenCV camera frame "
            "using masked 3D depth point clouds (first frame)."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_03")

    parser.add_argument(
        "--object_video_dir",
        type=str,
        default=None,
        help="Directory like ../Generate_Object_Mesh/output/video_xx",
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
        default="./output_3d",
        help="Output root for aligned results.",
    )

    parser.add_argument(
        "--human_obj",
        type=str,
        default=None,
        help="Optional explicit human OBJ path. Default: first OBJ in output_objs.",
    )
    parser.add_argument(
        "--human_mask",
        type=str,
        default=None,
        help="Optional human binary mask for frame_00. If omitted, script renders an initial pseudo-mask.",
    )
    parser.add_argument(
        "--human_coord",
        type=str,
        choices=["opencv", "pytorch3d"],
        default="opencv",
        help="Coordinate frame of the input human OBJ.",
    )
    parser.add_argument(
        "--intrinsics_source",
        type=str,
        choices=["object", "depth"],
        default="object",
        help=(
            "Camera intrinsics source for alignment point-cloud back-projection. "
            "'object' uses Generate_Object_Mesh/output/video_xx/camera_intrinsics.json "
            "(intrinsics_pixels_3x3). "
            "'depth' uses Estimate_Depth/output/video_xx/pose_estimation.json intrinsics."
        ),
    )
    parser.add_argument(
        "--intrinsics_warn_threshold_px",
        type=float,
        default=100.0,
        help="Warn when |object-depth intrinsics element diff| exceeds this threshold.",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opt_max_side", type=int, default=1280)
    parser.add_argument("--bin_size", type=int, default=0)

    parser.add_argument("--iters", type=int, nargs=3, default=[2500, 2500, 3000])
    parser.add_argument("--stage_lr", type=float, nargs=3, default=[5e-3, 2e-3, 1e-3])
    parser.add_argument(
        "--stage_early_stop_patience",
        type=int,
        nargs=3,
        default=[80, 100, 120],
        help=(
            "Per-stage per-mesh early stopping patience. "
            "Stop stage for a mesh when loss does not improve for this many iterations. "
            "Set <= 0 to disable."
        ),
    )
    parser.add_argument(
        "--stage_early_stop_min_delta",
        type=float,
        nargs=3,
        default=[1e-4, 5e-5, 1e-5],
        help=(
            "Per-stage minimum loss improvement required to reset early-stop patience. "
            "Improvement means eval_total < (best_total - min_delta)."
        ),
    )
    parser.add_argument("--trim_quantile", type=float, default=0.7)

    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    parser.add_argument("--reg_scale", type=float, default=0.05)
    parser.add_argument("--reg_rot", type=float, default=0.005)
    parser.add_argument("--reg_trans", type=float, default=0.01)
    parser.add_argument(
        "--reg_trans_reference",
        type=str,
        choices=["zero", "warmstart"],
        default="warmstart",
        help="Reference for translation regularization; warmstart penalizes drift from initialized translation.",
    )
    parser.add_argument("--depth_huber_delta_3d", type=float, default=0.05)
    parser.add_argument("--depth_anchor_weight", type=float, default=0.15)

    parser.add_argument("--chamfer_forward_weight", type=float, default=0.5)
    parser.add_argument("--chamfer_backward_weight", type=float, default=1.0)
    parser.add_argument("--pc_max_points_per_mesh", type=int, default=5000)
    parser.add_argument("--mesh_sample_points", type=int, default=5000)
    parser.add_argument("--min_points_per_mesh", type=int, default=256)
    parser.add_argument("--nn_chunk_size", type=int, default=1024)

    parser.add_argument("--human_mask_dilate_px", type=int, default=6)
    parser.add_argument(
        "--use_mesh_visibility_for_depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use visible mesh_id from full-scene rasterization for depth residual stats.",
    )

    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=5.0)
    parser.add_argument("--max_rot_deg", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--save_stage_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save per-stage outputs (OBJ meshes + combined transforms.json + overlay.png) "
            "in meshes_stage_1, meshes_stage_2, ..."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent

    object_video_dir = resolve_path(
        args.object_video_dir,
        script_dir,
    ) or (script_dir.parent / "Generate_Object_Mesh" / "output" / args.video_name).resolve()
    depth_video_dir = resolve_path(
        args.depth_video_dir,
        script_dir,
    ) or (script_dir.parent / "Estimate_Depth" / "output" / args.video_name).resolve()
    human_video_dir = resolve_path(
        args.human_video_dir,
        script_dir,
    ) or (script_dir.parent / "Estimate_Human_Motion" / "output" / args.video_name).resolve()
    output_root = resolve_path(args.output_root, script_dir)
    assert output_root is not None
    output_dir = output_root / args.video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not object_video_dir.exists():
        raise FileNotFoundError(f"Object dir not found: {object_video_dir}")
    if not depth_video_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_video_dir}")
    if not human_video_dir.exists():
        raise FileNotFoundError(f"Human dir not found: {human_video_dir}")

    summary_path = object_video_dir / "frame_00_segmentation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Segmentation summary not found: {summary_path}")
    summary = load_json(summary_path)

    pose_json_path = depth_video_dir / "pose_estimation.json"
    object_intrinsics_json_path = object_video_dir / "camera_intrinsics.json"
    run_summary_path = depth_video_dir / "run_summary.json"
    metric_depth_dir = depth_video_dir / "metric_depth"
    depth_npy_path = metric_depth_dir / "metric_depth.npy"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found: {run_summary_path}")
    if not depth_npy_path.exists():
        raise FileNotFoundError(f"metric_depth.npy not found: {depth_npy_path}")
    if not pose_json_path.exists():
        raise FileNotFoundError(f"pose_estimation.json not found: {pose_json_path}")

    run_summary = load_json(run_summary_path)
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

    pose = load_json(pose_json_path)
    k_object_full = load_object_intrinsics(object_intrinsics_json_path)
    k_depth_full = ensure_3x3_intrinsics(pose.get("intrinsics"))
    if args.intrinsics_source == "object":
        k_full = k_object_full
    else:
        k_full = k_depth_full
    extrinsics = ensure_3x4_extrinsics(pose.get("extrinsics"))
    print(f"Using intrinsics source: {args.intrinsics_source}")

    mismatch_report = build_intrinsics_mismatch_report(
        k_object=k_object_full,
        k_depth=k_depth_full,
        warn_threshold_px=float(args.intrinsics_warn_threshold_px),
    )
    selected_intrinsics_consistent = (
        (args.intrinsics_source == "depth") or (not bool(mismatch_report["warning"]))
    )
    selected_intrinsics_message = "Selected intrinsics are consistent with 3D depth back-projection."
    if mismatch_report["warning"]:
        if args.intrinsics_source == "depth":
            print(
                "WARNING: object/depth intrinsics mismatch exceeds threshold; "
                f"max_abs_diff={mismatch_report['max_abs_diff']:.3f}px "
                "(using depth intrinsics for 3D back-projection)."
            )
            selected_intrinsics_message = (
                "Object/depth intrinsics differ, but depth intrinsics were selected for back-projection."
            )
        else:
            print(
                "WARNING: object/depth intrinsics mismatch exceeds threshold; "
                f"max_abs_diff={mismatch_report['max_abs_diff']:.3f}px "
                "(non-depth intrinsics may bias 3D point-cloud alignment)."
            )
            selected_intrinsics_message = (
                "Object/depth intrinsics differ and non-depth intrinsics were selected; "
                "this can bias back-projected target point clouds."
            )

    assets: list[MeshAsset] = []
    object_count = 0
    for obj in summary.get("objects", []):
        if not obj.get("success", False):
            continue
        obj_name = str(obj["object"])
        obj_dir = Path(obj["output_dir"]).resolve()
        mesh_path = obj_dir / "mesh_posed.glb"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Missing mesh_posed.glb for object '{obj_name}': {mesh_path}"
            )

        mask_rel = obj.get("mask_file")
        if mask_rel is None:
            raise KeyError(
                f"Missing 'mask_file' for object '{obj_name}' in {summary_path}"
            )
        mask_path = (obj_dir / mask_rel).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for object '{obj_name}': {mask_path}"
            )

        verts_src, faces = load_mesh(mesh_path)
        mask = load_binary_mask(mask_path, (depth_h, depth_w))
        source_to_cv = F_P3D_TO_CV @ R_Y_UP_TO_Z_UP
        asset = MeshAsset(
            name=obj_name,
            slug=slugify(obj_name),
            kind="object",
            source_mesh_path=mesh_path,
            source_coord="mesh_posed_glb_y_up",
            verts_source=verts_src,
            faces=faces,
            source_to_cv=source_to_cv.astype(np.float32),
            mask_path=mask_path,
            mask=mask,
        )
        assets.append(asset)
        object_count += 1

    if object_count == 0:
        raise RuntimeError(
            f"No usable object meshes found in summary: {summary_path}"
        )

    human_obj_path = resolve_path(args.human_obj, script_dir)
    if human_obj_path is None:
        output_objs_dir = human_video_dir / "output_objs"
        human_obj_path = find_first_human_obj(output_objs_dir)
    if not human_obj_path.exists():
        raise FileNotFoundError(f"Human OBJ not found: {human_obj_path}")

    human_mask_path = resolve_path(args.human_mask, script_dir)
    human_mask: np.ndarray | None = None
    if human_mask_path is not None:
        human_mask = load_binary_mask(human_mask_path, (depth_h, depth_w))

    human_verts_src, human_faces = load_mesh(human_obj_path)
    human_source_to_cv = (
        np.eye(3, dtype=np.float32)
        if args.human_coord == "opencv"
        else F_P3D_TO_CV.copy()
    )
    human_asset = MeshAsset(
        name="human",
        slug="human",
        kind="human",
        source_mesh_path=human_obj_path,
        source_coord="opencv_camera"
        if args.human_coord == "opencv"
        else "pytorch3d_camera",
        verts_source=human_verts_src,
        faces=human_faces,
        source_to_cv=human_source_to_cv.astype(np.float32),
        mask_path=human_mask_path,
        mask=human_mask,
    )
    assets = [human_asset] + assets

    names = [a.name for a in assets]
    j_count = len(assets)
    print(f"Loaded {j_count} meshes: {names}")

    masks_full = [a.mask for a in assets]
    depth_opt, masks_opt, k_opt, resize_scale = maybe_resize_for_optimization(
        depth_obs, masks_full, k_full, int(args.opt_max_side)
    )
    opt_h, opt_w = depth_opt.shape
    print(f"Optimization resolution: {opt_w}x{opt_h} (scale={resize_scale:.4f})")

    for asset, mask_opt in zip(assets, masks_opt):
        asset.mask = mask_opt

    device = parse_device(args.device)
    print(f"Using device: {device}")

    cams = build_cameras(k_opt, opt_w, opt_h, device)
    hard_rasterizer = build_hard_rasterizer(
        cameras=cams,
        image_size=(opt_h, opt_w),
        bin_size=int(args.bin_size),
    )

    cv_to_p3d = torch.tensor(F_P3D_TO_CV, dtype=torch.float32, device=device)
    depth_obs_t = torch.from_numpy(depth_opt).to(device=device, dtype=torch.float32)

    verts_base_cv: list[torch.Tensor] = []
    faces_t: list[torch.Tensor] = []
    masks_t: list[torch.Tensor | None] = []
    verts_base_cv_np: list[np.ndarray] = []
    for asset in assets:
        verts_cv_np = asset.verts_source @ asset.source_to_cv.transpose(0, 1)
        verts_base_cv_np.append(verts_cv_np.astype(np.float32))
        verts_base_cv.append(
            torch.from_numpy(verts_cv_np).to(device=device, dtype=torch.float32)
        )
        faces_t.append(torch.from_numpy(asset.faces.astype(np.int64)).to(device=device))
        masks_t.append(
            None
            if asset.mask is None
            else torch.from_numpy(asset.mask).to(device=device, dtype=torch.float32)
        )

    verts_before = [v.detach().cpu().numpy() for v in verts_base_cv]
    overlay_before = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_before,
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_before.png"), overlay_before)
    print(f"Saved initial overlay to: {output_dir / 'overlay_before.png'}")

    # If no human mask is provided, use initial rendered silhouette as a pseudo target.
    if masks_t[0] is None:
        with torch.no_grad():
            human_depth, human_mesh_id = render_scene_depth_and_mesh_id(
                hard_rasterizer=hard_rasterizer,
                verts_cv=[verts_base_cv[0]],
                faces=[faces_t[0]],
                cv_to_p3d=cv_to_p3d,
                device=device,
            )
            del human_depth
            human_mask = (human_mesh_id == 0).detach().cpu().numpy().astype(np.uint8)
            dpx = int(args.human_mask_dilate_px)
            if dpx > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * dpx + 1, 2 * dpx + 1),
                )
                human_mask = cv2.dilate(human_mask, kernel, iterations=1)
            masks_t[0] = torch.from_numpy(human_mask.astype(np.float32)).to(
                device=device, dtype=torch.float32
            )
            assets[0].mask = human_mask.astype(np.float32)
            assets[0].mask_path = None
        print("Human mask: generated from initial rendered silhouette.")
    else:
        print("Human mask: loaded from file.")

    # Build target point clouds + fixed mesh surface samples.
    # rng = np.random.default_rng(args.seed)
    states: list[MeshState] = []
    per_mesh_pointcloud_stats: list[dict[str, Any]] = []
    mesh_status: list[dict[str, Any]] = []
    for j, asset in enumerate(assets):
        mesh_rng = np.random.default_rng(args.seed + 10007 * (j + 1))

        target_full = masked_depth_to_pointcloud_cv(
            depth=depth_opt,
            mask=asset.mask,
            intrinsics=k_opt,
        )
        target_used = downsample_points(
            target_full, int(args.pc_max_points_per_mesh), mesh_rng
        )

        sample_points_np = sample_mesh_surface_points(
            verts=verts_base_cv_np[j],
            faces=asset.faces,
            num_samples=int(args.mesh_sample_points),
            rng=mesh_rng,
        )
        sample_points_t = torch.from_numpy(sample_points_np).to(
            device=device, dtype=torch.float32
        )

        target_points_t: torch.Tensor | None = None
        target_z_median_t: torch.Tensor | None = None
        if target_used.shape[0] > 0:
            target_points_t = torch.from_numpy(target_used).to(
                device=device, dtype=torch.float32
            )
            target_z_median_t = torch.median(target_points_t[:, 2])

        state = MeshState(
            log_s=torch.nn.Parameter(
                torch.zeros((), device=device, dtype=torch.float32)
            ),
            rotvec=torch.nn.Parameter(
                torch.zeros((3,), device=device, dtype=torch.float32)
            ),
            tvec=torch.nn.Parameter(
                torch.zeros((3,), device=device, dtype=torch.float32)
            ),
            tvec_init=torch.zeros((3,), device=device, dtype=torch.float32),
            sample_points_base=sample_points_t,
            target_points=target_points_t,
            target_points_total=int(target_full.shape[0]),
            target_points_used=int(target_used.shape[0]),
            target_z_median=target_z_median_t,
            active=True,
            status="pending",
            message=None,
        )

        # Warm-start tz from median depth.
        if target_points_t is not None and target_points_t.shape[0] > 0:
            with torch.no_grad():
                state.tvec[2] += torch.median(target_points_t[:, 2]) - torch.median(
                    sample_points_t[:, 2]
                )
        with torch.no_grad():
            state.tvec_init.copy_(state.tvec.detach())

        if state.target_points_used < int(args.min_points_per_mesh):
            state.active = False
            state.status = "skipped_insufficient_points"
            state.message = (
                f"target points {state.target_points_used} < min_points_per_mesh {int(args.min_points_per_mesh)}"
            )

        states.append(state)
        per_mesh_pointcloud_stats.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "mask_path": None if asset.mask_path is None else str(asset.mask_path),
                "target_points_full": int(target_full.shape[0]),
                "target_points_used": int(target_used.shape[0]),
                "stats_full": pointcloud_stats(target_full),
                "stats_used": pointcloud_stats(target_used),
                "mesh_sample_points": int(sample_points_np.shape[0]),
                "active_for_optimization": bool(state.active),
            }
        )
        mesh_status.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "status": state.status,
                "message": state.message,
                "target_points_total": int(state.target_points_total),
                "target_points_used": int(state.target_points_used),
            }
        )

    loss_history: list[dict[str, Any]] = []
    per_stage_mesh_losses_3d: list[dict[str, Any]] = []
    stage_output_records: list[dict[str, Any]] = []
    scale_clamped_count = 0
    rotation_clamped_count = 0
    low_overlap_pixel_threshold = 128

    max_rot_rad = math.radians(float(args.max_rot_deg))
    max_rot_t = torch.tensor(float(max_rot_rad), device=device, dtype=torch.float32)
    min_log_scale = math.log(float(args.min_scale))
    max_log_scale = math.log(float(args.max_scale))
    stages = [
        {"use_scale": True, "use_rot": False, "use_txy": False, "use_tz": True},
        {"use_scale": True, "use_rot": False, "use_txy": True, "use_tz": True},
        {"use_scale": True, "use_rot": True, "use_txy": True, "use_tz": True},
    ]
    if (
        len(args.iters) != 3
        or len(args.stage_lr) != 3
        or len(args.stage_early_stop_patience) != 3
        or len(args.stage_early_stop_min_delta) != 3
    ):
        raise ValueError(
            "--iters, --stage_lr, --stage_early_stop_patience, "
            "and --stage_early_stop_min_delta must each provide exactly 3 values."
        )

    stage_iters = [int(v) for v in args.iters]
    stage_lrs = [float(v) for v in args.stage_lr]
    stage_early_stop_patience = [int(v) for v in args.stage_early_stop_patience]
    stage_early_stop_min_delta = [float(v) for v in args.stage_early_stop_min_delta]
    for stage_i, n_iter in enumerate(stage_iters):
        if n_iter <= 0:
            raise ValueError(f"--iters[{stage_i}] must be > 0.")
    for stage_i, patience in enumerate(stage_early_stop_patience):
        if patience < 0:
            raise ValueError(
                f"--stage_early_stop_patience[{stage_i}] must be >= 0."
            )
    for stage_i, min_delta in enumerate(stage_early_stop_min_delta):
        if min_delta < 0.0:
            raise ValueError(
                f"--stage_early_stop_min_delta[{stage_i}] must be >= 0."
            )

    trim_q = float(args.trim_quantile)
    for stage_idx, stage_cfg in enumerate(stages):
        n_iter = stage_iters[stage_idx]
        lr_stage = stage_lrs[stage_idx]
        stage_patience = stage_early_stop_patience[stage_idx]
        stage_min_delta = stage_early_stop_min_delta[stage_idx]
        print(
            f"[Stage {stage_idx}] iters={n_iter}, lr={lr_stage}, trim_q={trim_q}, "
            f"patience={stage_patience}, min_delta={stage_min_delta}, cfg={stage_cfg}"
        )

        stage_mesh_losses: list[dict[str, Any]] = []
        for j, state in enumerate(states):
            mesh_name = names[j]
            if not state.active:
                stage_mesh_losses.append(
                    {
                        "name": mesh_name,
                        "status": state.status,
                        "message": state.message,
                        "iters_requested": n_iter,
                        "iters_ran": 0,
                        "initial_total": None,
                        "best_total": None,
                        "final_total": None,
                        "final_chamfer": None,
                        "final_forward": None,
                        "final_backward": None,
                        "final_anchor": None,
                        "final_reg": None,
                        "forward_kept": 0,
                        "backward_kept": 0,
                        "target_points_used": int(state.target_points_used),
                        "mesh_sample_points": int(state.sample_points_base.shape[0]),
                        "trim_quantile": trim_q,
                        "early_stopped": False,
                        "early_stop_patience": int(stage_patience),
                        "early_stop_min_delta": float(stage_min_delta),
                        "early_stop_iter": None,
                    }
                )
                continue

            with torch.no_grad():
                init_losses = compute_mesh_losses(
                    state=state,
                    trim_quantile=trim_q,
                    chamfer_forward_weight=float(args.chamfer_forward_weight),
                    chamfer_backward_weight=float(args.chamfer_backward_weight),
                    depth_weight=float(args.depth_weight),
                    reg_weight=float(args.reg_weight),
                    reg_scale=float(args.reg_scale),
                    reg_rot=float(args.reg_rot),
                    reg_trans=float(args.reg_trans),
                    reg_trans_reference=str(args.reg_trans_reference),
                    depth_huber_delta_3d=float(args.depth_huber_delta_3d),
                    depth_anchor_weight=float(args.depth_anchor_weight),
                    nn_chunk_size=int(args.nn_chunk_size),
                )
                init_total = float(init_losses["total"].item())

            if not math.isfinite(init_total):
                state.active = False
                state.status = "failed_nonfinite_initial"
                state.message = "Initial 3D loss is non-finite."
                stage_mesh_losses.append(
                    {
                        "name": mesh_name,
                        "status": state.status,
                        "message": state.message,
                        "iters_requested": n_iter,
                        "iters_ran": 0,
                        "initial_total": init_total,
                        "best_total": None,
                        "final_total": None,
                        "final_chamfer": None,
                        "final_forward": None,
                        "final_backward": None,
                        "final_anchor": None,
                        "final_reg": None,
                        "forward_kept": 0,
                        "backward_kept": 0,
                        "target_points_used": int(state.target_points_used),
                        "mesh_sample_points": int(state.sample_points_base.shape[0]),
                        "trim_quantile": trim_q,
                        "early_stopped": False,
                        "early_stop_patience": int(stage_patience),
                        "early_stop_min_delta": float(stage_min_delta),
                        "early_stop_iter": None,
                    }
                )
                continue

            optimizer = torch.optim.Adam(
                [state.log_s, state.rotvec, state.tvec], lr=lr_stage
            )
            best_total = init_total
            best_state = (
                state.log_s.detach().clone(),
                state.rotvec.detach().clone(),
                state.tvec.detach().clone(),
            )
            diverged = False
            early_stopped = False
            early_stop_iter: int | None = None
            no_improve_iters = 0
            iters_ran = 0

            for it in range(n_iter):
                losses = compute_mesh_losses(
                    state=state,
                    trim_quantile=trim_q,
                    chamfer_forward_weight=float(args.chamfer_forward_weight),
                    chamfer_backward_weight=float(args.chamfer_backward_weight),
                    depth_weight=float(args.depth_weight),
                    reg_weight=float(args.reg_weight),
                    reg_scale=float(args.reg_scale),
                    reg_rot=float(args.reg_rot),
                    reg_trans=float(args.reg_trans),
                    reg_trans_reference=str(args.reg_trans_reference),
                    depth_huber_delta_3d=float(args.depth_huber_delta_3d),
                    depth_anchor_weight=float(args.depth_anchor_weight),
                    nn_chunk_size=int(args.nn_chunk_size),
                )
                total = losses["total"]
                if not torch.isfinite(total):
                    diverged = True
                    break

                optimizer.zero_grad()
                total.backward()

                if not stage_cfg["use_scale"] and state.log_s.grad is not None:
                    state.log_s.grad.zero_()
                if not stage_cfg["use_rot"] and state.rotvec.grad is not None:
                    state.rotvec.grad.zero_()
                if state.tvec.grad is not None:
                    if not stage_cfg["use_txy"]:
                        state.tvec.grad[0:2].zero_()
                    if not stage_cfg["use_tz"]:
                        state.tvec.grad[2].zero_()

                optimizer.step()
                iters_ran = it + 1

                with torch.no_grad():
                    prev_log_s = state.log_s.detach().clone()
                    prev_rot_norm = float(torch.linalg.norm(state.rotvec).item())
                    state.log_s.clamp_(min=min_log_scale, max=max_log_scale)
                    if float((state.log_s - prev_log_s).abs().item()) > 1e-10:
                        scale_clamped_count += 1
                    norm = torch.linalg.norm(state.rotvec).clamp_min(1e-8)
                    fac = torch.clamp(max_rot_t / norm, max=1.0)
                    state.rotvec.mul_(fac)
                    if prev_rot_norm > float(max_rot_rad) + 1e-8:
                        rotation_clamped_count += 1

                    eval_losses = compute_mesh_losses(
                        state=state,
                        trim_quantile=trim_q,
                        chamfer_forward_weight=float(args.chamfer_forward_weight),
                        chamfer_backward_weight=float(args.chamfer_backward_weight),
                        depth_weight=float(args.depth_weight),
                        reg_weight=float(args.reg_weight),
                        reg_scale=float(args.reg_scale),
                        reg_rot=float(args.reg_rot),
                        reg_trans=float(args.reg_trans),
                        reg_trans_reference=str(args.reg_trans_reference),
                        depth_huber_delta_3d=float(args.depth_huber_delta_3d),
                        depth_anchor_weight=float(args.depth_anchor_weight),
                        nn_chunk_size=int(args.nn_chunk_size),
                    )
                    eval_total = float(eval_losses["total"].item())
                    if not math.isfinite(eval_total):
                        diverged = True
                        break
                    if eval_total < (best_total - stage_min_delta):
                        best_total = eval_total
                        best_state = (
                            state.log_s.detach().clone(),
                            state.rotvec.detach().clone(),
                            state.tvec.detach().clone(),
                        )
                        no_improve_iters = 0
                    else:
                        no_improve_iters += 1

                    if stage_patience > 0 and no_improve_iters >= stage_patience:
                        early_stopped = True
                        early_stop_iter = it + 1
                        print(
                            f"stage={stage_idx:02d} mesh={mesh_name} "
                            f"early_stop iter={early_stop_iter:04d}/{n_iter:04d} "
                            f"best_loss={best_total:.6f} patience={stage_patience} "
                            f"min_delta={stage_min_delta:.6g}"
                        )
                        break

                    if (it + 1) % int(args.log_every) == 0 or it == 0 or (it + 1) == n_iter:
                        print(
                            f"stage={stage_idx:02d} mesh={mesh_name} "
                            f"iter={it + 1:04d}/{n_iter:04d} "
                            f"loss={eval_total:.6f} ch={float(eval_losses['chamfer'].item()):.6f} "
                            f"fw={float(eval_losses['forward'].item()):.6f} "
                            f"bw={float(eval_losses['backward'].item()):.6f} "
                            f"anchor={float(eval_losses['anchor'].item()):.6f} "
                            f"reg={float(eval_losses['reg'].item()):.6f}"
                        )
                        loss_history.append(
                            {
                                "stage": stage_idx,
                                "mesh": mesh_name,
                                "mesh_index": j,
                                "iter": it + 1,
                                "loss_total": eval_total,
                                "loss_chamfer": float(eval_losses["chamfer"].item()),
                                "loss_forward": float(eval_losses["forward"].item()),
                                "loss_backward": float(eval_losses["backward"].item()),
                                "loss_anchor": float(eval_losses["anchor"].item()),
                                "loss_regularization": float(eval_losses["reg"].item()),
                                "forward_kept": int(eval_losses["forward_kept"]),
                                "backward_kept": int(eval_losses["backward_kept"]),
                            }
                        )

            if diverged:
                with torch.no_grad():
                    state.log_s.copy_(best_state[0])
                    state.rotvec.copy_(best_state[1])
                    state.tvec.copy_(best_state[2])
                state.status = "diverged_recovered"
                state.message = (
                    "Encountered non-finite loss; reverted to best finite checkpoint."
                )
            elif early_stopped:
                with torch.no_grad():
                    state.log_s.copy_(best_state[0])
                    state.rotvec.copy_(best_state[1])
                    state.tvec.copy_(best_state[2])
                state.status = "optimized_early_stopped"
                state.message = (
                    f"Early stopped at iter {int(early_stop_iter or 0)} "
                    f"(patience={stage_patience}, min_delta={stage_min_delta}); "
                    "reverted to best checkpoint."
                )
            else:
                state.status = "optimized"
                state.message = None

            with torch.no_grad():
                final_losses = compute_mesh_losses(
                    state=state,
                    trim_quantile=trim_q,
                    chamfer_forward_weight=float(args.chamfer_forward_weight),
                    chamfer_backward_weight=float(args.chamfer_backward_weight),
                    depth_weight=float(args.depth_weight),
                    reg_weight=float(args.reg_weight),
                    reg_scale=float(args.reg_scale),
                    reg_rot=float(args.reg_rot),
                    reg_trans=float(args.reg_trans),
                    reg_trans_reference=str(args.reg_trans_reference),
                    depth_huber_delta_3d=float(args.depth_huber_delta_3d),
                    depth_anchor_weight=float(args.depth_anchor_weight),
                    nn_chunk_size=int(args.nn_chunk_size),
                )

            stage_mesh_losses.append(
                {
                    "name": mesh_name,
                    "status": state.status,
                    "message": state.message,
                    "iters_requested": n_iter,
                    "iters_ran": int(iters_ran),
                    "initial_total": init_total,
                    "best_total": float(best_total),
                    "final_total": float(final_losses["total"].item()),
                    "final_chamfer": float(final_losses["chamfer"].item()),
                    "final_forward": float(final_losses["forward"].item()),
                    "final_backward": float(final_losses["backward"].item()),
                    "final_anchor": float(final_losses["anchor"].item()),
                    "final_reg": float(final_losses["reg"].item()),
                    "forward_kept": int(final_losses["forward_kept"]),
                    "backward_kept": int(final_losses["backward_kept"]),
                    "target_points_used": int(state.target_points_used),
                    "mesh_sample_points": int(state.sample_points_base.shape[0]),
                    "trim_quantile": trim_q,
                    "early_stopped": bool(early_stopped),
                    "early_stop_patience": int(stage_patience),
                    "early_stop_min_delta": float(stage_min_delta),
                    "early_stop_iter": None
                    if early_stop_iter is None
                    else int(early_stop_iter),
                }
            )

            mesh_status[j] = {
                "name": assets[j].name,
                "slug": assets[j].slug,
                "kind": assets[j].kind,
                "status": state.status,
                "message": state.message,
                "target_points_total": int(state.target_points_total),
                "target_points_used": int(state.target_points_used),
            }

        per_stage_mesh_losses_3d.append(
            {
                "stage": stage_idx + 1,
                "stage_index": stage_idx,
                "trim_quantile": trim_q,
                "mesh_losses": stage_mesh_losses,
                "mode": "joint",
                "stage_config": stage_cfg,
                "iters": int(n_iter),
                "lr": float(lr_stage),
                "early_stop_patience": int(stage_patience),
                "early_stop_min_delta": float(stage_min_delta),
            }
        )

        if bool(args.save_stage_outputs):
            stage_num = stage_idx + 1
            stage_meshes_dir = output_dir / f"meshes_stage_{stage_num}"
            stage_transforms_path = stage_meshes_dir / "transforms.json"
            stage_overlay_path = stage_meshes_dir / "overlay.png"
            with torch.no_grad():
                verts_stage_t = [
                    transform_points_single(v, st.log_s, st.rotvec, st.tvec)
                    for v, st in zip(verts_base_cv, states)
                ]
                verts_stage = [v.detach().cpu().numpy() for v in verts_stage_t]
                _, scales_np_stage, rotvec_np_stage, rots_np_stage, t_np_stage = (
                    get_transform_arrays(states)
                )
            export_meshes_and_transforms(
                assets=assets,
                verts_aligned=verts_stage,
                rotvec_axis_angle=rotvec_np_stage,
                rots_np=rots_np_stage,
                scales_np=scales_np_stage,
                t_np=t_np_stage,
                meshes_out_dir=stage_meshes_dir,
                transforms_out_path=stage_transforms_path,
            )
            overlay_stage = draw_overlay_points(
                frame_bgr=frame_bgr,
                verts_cv_list=verts_stage,
                names=names,
                k=k_full,
            )
            cv2.imwrite(str(stage_overlay_path), overlay_stage)
            stage_output_records.append(
                {
                    "stage": stage_num,
                    "meshes_dir": str(stage_meshes_dir),
                    "transforms_json": str(stage_transforms_path),
                    "overlay_png": str(stage_overlay_path),
                }
            )
            print(f"Saved stage {stage_num} outputs to: {stage_meshes_dir}")

    with torch.no_grad():
        verts_final_t = [
            transform_points_single(v, st.log_s, st.rotvec, st.tvec)
            for v, st in zip(verts_base_cv, states)
        ]
        verts_final = [v.detach().cpu().numpy() for v in verts_final_t]
        depth_before, mesh_id_before = render_scene_depth_and_mesh_id(
            hard_rasterizer=hard_rasterizer,
            verts_cv=verts_base_cv,
            faces=faces_t,
            cv_to_p3d=cv_to_p3d,
            device=device,
        )
        depth_after, mesh_id_after = render_scene_depth_and_mesh_id(
            hard_rasterizer=hard_rasterizer,
            verts_cv=verts_final_t,
            faces=faces_t,
            cv_to_p3d=cv_to_p3d,
            device=device,
        )

    before_stats = per_mesh_depth_stats(
        depth_render=depth_before,
        mesh_id=mesh_id_before,
        depth_obs=depth_obs_t,
        masks=masks_t,
        names=names,
        use_visibility=bool(args.use_mesh_visibility_for_depth),
    )
    after_stats = per_mesh_depth_stats(
        depth_render=depth_after,
        mesh_id=mesh_id_after,
        depth_obs=depth_obs_t,
        masks=masks_t,
        names=names,
        use_visibility=bool(args.use_mesh_visibility_for_depth),
    )
    mesh_target_points_used = {
        str(m["name"]): int(m.get("target_points_used", 0)) for m in mesh_status
    }
    low_overlap_meshes = []
    for s in after_stats:
        n = str(s["name"])
        pix = int(s.get("pixels", 0))
        tgt_used = int(mesh_target_points_used.get(n, 0))
        if tgt_used >= int(args.min_points_per_mesh) and pix < low_overlap_pixel_threshold:
            low_overlap_meshes.append(
                {
                    "name": n,
                    "pixels_after": pix,
                    "target_points_used": tgt_used,
                }
            )
            for m in mesh_status:
                if str(m["name"]) != n:
                    continue
                cur_status = str(m.get("status", ""))
                cur_msg = m.get("message")
                if cur_status == "optimized":
                    m["status"] = "optimized_low_overlap"
                warn_msg = (
                    f"Low final overlap: pixels_after={pix} < "
                    f"low_overlap_pixel_threshold={low_overlap_pixel_threshold}."
                )
                if cur_msg:
                    m["message"] = f"{cur_msg} {warn_msg}"
                else:
                    m["message"] = warn_msg
                break

    meshes_out_dir = output_dir / "meshes"
    transforms_json_path = meshes_out_dir / "transforms.json"
    _, scales_np, rotvec_np, rots_np, t_np = get_transform_arrays(states)
    transforms_out = export_meshes_and_transforms(
        assets=assets,
        verts_aligned=verts_final,
        rotvec_axis_angle=rotvec_np,
        rots_np=rots_np,
        scales_np=scales_np,
        t_np=t_np,
        meshes_out_dir=meshes_out_dir,
        transforms_out_path=transforms_json_path,
    )

    overlay_after = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_final,
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_after.png"), overlay_after)

    result = {
        "video_name": args.video_name,
        "coordinate_frame": "opencv_camera_frame0",
        "paths": {
            "object_video_dir": str(object_video_dir),
            "depth_video_dir": str(depth_video_dir),
            "metric_depth_dir": str(depth_npy_path.parent),
            "human_video_dir": str(human_video_dir),
            "summary_json": str(summary_path),
            "pose_json": str(pose_json_path),
            "object_intrinsics_json": str(object_intrinsics_json_path),
            "depth_npy": str(depth_npy_path),
            "output_dir": str(output_dir),
            "meshes_dir": str(meshes_out_dir),
            "transforms_json": str(transforms_json_path),
        },
        "camera": {
            "intrinsics_source": str(args.intrinsics_source),
            "intrinsics_3x3": k_full.tolist(),
            "object_intrinsics_3x3": k_object_full.tolist(),
            "depth_intrinsics_3x3": k_depth_full.tolist(),
            "extrinsics_3x4_from_depth": None if extrinsics is None else extrinsics.tolist(),
            "depth_is_metric": bool(int(pose.get("is_metric", 0))),
            "depth_scale_factor": pose.get("scale_factor", None),
        },
        "intrinsics_mismatch_report": mismatch_report,
        "optimization": {
            "device": str(device),
            "resolution_input_hw": [int(depth_h), int(depth_w)],
            "resolution_optimized_hw": [int(opt_h), int(opt_w)],
            "iters": [int(v) for v in args.iters],
            "stage_lr": [float(v) for v in args.stage_lr],
            "stage_early_stop_patience": [
                int(v) for v in args.stage_early_stop_patience
            ],
            "stage_early_stop_min_delta": [
                float(v) for v in args.stage_early_stop_min_delta
            ],
            "depth_weight": float(args.depth_weight),
            "reg_weight": float(args.reg_weight),
            "reg_scale": float(args.reg_scale),
            "reg_rot": float(args.reg_rot),
            "reg_trans": float(args.reg_trans),
            "reg_trans_reference": str(args.reg_trans_reference),
            "use_mesh_visibility_for_depth": bool(args.use_mesh_visibility_for_depth),
            "stage_dof_schedule": stages,
        },
        "optimization_3d": {
            "mode": "staged_joint",
            "num_stages": 3,
            "iters": [int(v) for v in args.iters],
            "stage_lr": [float(v) for v in args.stage_lr],
            "stage_early_stop_patience": [
                int(v) for v in args.stage_early_stop_patience
            ],
            "stage_early_stop_min_delta": [
                float(v) for v in args.stage_early_stop_min_delta
            ],
            "stage_dof_schedule": stages,
            "trim_quantile": float(args.trim_quantile),
            "pc_max_points_per_mesh": int(args.pc_max_points_per_mesh),
            "mesh_sample_points": int(args.mesh_sample_points),
            "min_points_per_mesh": int(args.min_points_per_mesh),
            "nn_chunk_size": int(args.nn_chunk_size),
            "chamfer_forward_weight": float(args.chamfer_forward_weight),
            "chamfer_backward_weight": float(args.chamfer_backward_weight),
            "depth_huber_delta_3d": float(args.depth_huber_delta_3d),
            "depth_anchor_weight": float(args.depth_anchor_weight),
            "reg_trans_reference": str(args.reg_trans_reference),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
            "max_rot_deg": float(args.max_rot_deg),
        },
        "optimizer_health": {
            "scale_clamped_count": int(scale_clamped_count),
            "rotation_clamped_count": int(rotation_clamped_count),
            "low_overlap_pixel_threshold": int(low_overlap_pixel_threshold),
            "low_overlap_warning": bool(len(low_overlap_meshes) > 0),
            "low_overlap_meshes": low_overlap_meshes,
            "selected_intrinsics_consistency": {
                "intrinsics_source": str(args.intrinsics_source),
                "mismatch_warning": bool(mismatch_report["warning"]),
                "consistent": bool(selected_intrinsics_consistent),
                "message": str(selected_intrinsics_message),
            },
        },
        "per_mesh_pointcloud_stats": per_mesh_pointcloud_stats,
        "per_stage_mesh_losses_3d": per_stage_mesh_losses_3d,
        "stage_outputs": stage_output_records,
        "mesh_status": mesh_status,
        "depth_residual_stats_before": before_stats,
        "depth_residual_stats_after": after_stats,
        "transforms": transforms_out,
        "loss_history": loss_history,
    }

    result_path = output_dir / "alignment_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved alignment result: {result_path}")
    print(f"Saved aligned meshes to: {meshes_out_dir}")
    print(f"Saved combined transforms JSON to: {transforms_json_path}")


if __name__ == "__main__":
    main()
