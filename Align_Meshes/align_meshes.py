"""Mesh-to-depth alignment with fixed correspondences.

Overview
--------
This script aligns meshes (human and/or objects) to frame_00 depth in camera
coordinates using fixed point-to-point correspondences.

The pipeline is intentionally simple:
1. Load frame_00 RGB, metric depth, intrinsics, meshes, and masks.
2. Convert each mesh to OpenCV camera coordinates.
3. Render each mesh once to establish mesh<->depth correspondences.
4. Keep correspondences fixed and optimize only global mesh scale + z translation.
5. Save aligned meshes and diagnostics (loss curves + correspondence snapshots).


Detailed Algorithm
------------------
A) Inputs and coordinate preparation
  - Depth and frame:
    - `metric_depth.npy` from Estimate_Depth.
    - `frame_00` path from `run_summary.json`.
  - Intrinsics:
    - Either object intrinsics (`camera_intrinsics.json`, SAM3D side)
      or depth intrinsics (`pose_estimation.json`, DA3 side), selected by
      `--intrinsics_source`.
  - Meshes:
    - Objects: `mesh_posed.glb` and per-object mask from
      `frame_00_segmentation_summary.json`.
    - Human: first OBJ in `output_objs` (or `--human_obj`) and optional mask.
  - Coordinate conversion:
    - Object GLB vertices are mapped to OpenCV camera coordinates using
      `F_P3D_TO_CV @ R_Y_UP_TO_Z_UP`.
    - Human vertices use identity or `F_P3D_TO_CV` depending on `--human_coord`.

B) Render mesh and build fixed correspondences
  For each mesh independently:
  1. Rasterize once with PyTorch3D:
     - `faces_per_pixel=1`, `blur_radius=0`, image size = optimization resolution.
     - Retrieve:
       - `pix_to_face[y,x]`: visible face index at pixel.
       - `bary_coords[y,x]`: barycentric coordinates inside that face.
  2. Keep only valid pixels where:
     - mesh is visible (`pix_to_face >= 0`)
     - depth is valid (`depth > 0` and finite)
     - mask is active (if mask exists).
  3. Convert each valid rendered pixel into a mesh-surface 3D point:
     - Use face vertices + barycentric interpolation.
     - This produces `m_i` (mesh point cloud sampled from rendered visibility).
  4. Convert the same pixel into a depth 3D point:
     - Back-project depth with intrinsics:
       - `x = (u - cx) * z / fx`
       - `y = (v - cy) * z / fy`
       - `z = depth[u,v]`
     - This produces `d_i`.
  5. Store fixed pair `(m_i, d_i)` and fixed reference pixel `p_i^0=(u_i^0, v_i^0)`.

  Important: correspondences are created once and remain fixed for the full
  optimization. The script does not rerender/reassociate correspondences at each
  iteration.

C) Color mapping for correspondence visualization
  - Assign RGB color to each mesh correspondence point `m_i` based on normalized
    XYZ in the sampled mesh-point set.
  - Use the exact same color for its paired depth point `d_i`.
  - This makes matching pairs across mesh/depth clouds visually consistent.

D) Optimization variables and transform
  Per mesh optimize:
    - `alpha` (log-scale), where `s = exp(alpha)`
    - `delta_t_z`
  Initialization:
    - `alpha = 0` -> `s = 1`
    - `t_z_init = 0`
    - `t_z = t_z_init + delta_t_z`

  Apply transform to the same fixed mesh correspondence points every iteration:
    `m_i'(alpha, delta_t_z) = s * m_i + [0, 0, t_z]^T`

  This is how the script "keeps correspondences":
    - The point identities `m_i` are fixed.
    - At each iteration, the exact same points are transformed analytically.
    - No new mesh points are sampled; no new pixel associations are formed.

E) Losses (exact energy terms)
  Let K = (fx, fy, cx, cy), and projection of transformed point:
    - `u_i = fx * m'_ix / m'_iz + cx`
    - `v_i = fy * m'_iy / m'_iz + cy`

  Terms:
    - `E_corr      = (1/N) * sum_i || m_i' - d_i ||_2^2`
    - `E_reproj    = (1/N) * sum_i || [u_i, v_i] - [u_i^0, v_i^0] ||_2^2`
    - `E_scale_reg = alpha^2`
    - `E_tz_reg    = delta_t_z^2`

  Combined objective:
    - `E_total = w_corr * E_corr`
    - `        + w_reproj * E_reproj`
    - `        + w_scale_reg * E_scale_reg`
    - `        + w_tz_reg * E_tz_reg`

F) What is saved
  - `overlay_before.png`: projection of raw input meshes before optimization.
  - `overlay_after.png`: projection after optimized scale + t_z.
  - `meshes/<slug>.obj`: aligned meshes.
  - `meshes/transforms.json`: source-to-aligned transform metadata.
  - `loss_curves/<slug>_loss_*.png`: separate loss plots for each term and total.
  - `loss_curves/<slug>_loss.csv`: numeric loss history.
  - `correspondences/<slug>/iter_XXXXX_depth_fixed.ply`:
    fixed depth correspondence cloud with colors.
  - `correspondences/<slug>/iter_XXXXX_mesh_transformed.ply`:
    transformed mesh correspondence cloud with same colors.
  - `correspondences/<slug>/iter_XXXXX_correspondence_projection.png`:
    side-by-side 2D projection visualization of fixed depth points and transformed
    mesh points at that iteration.
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
import matplotlib
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# GLB meshes are typically Y-up while SAM3D transforms are Z-up.
R_Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# OpenCV (+X right, +Y down, +Z forward) <-> PyTorch3D (+X left, +Y up, +Z forward)
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
class CorrespondenceSet:
    mesh_points_base: np.ndarray  # (N,3)
    depth_points: np.ndarray  # (N,3)
    uv_ref: np.ndarray  # (N,2)
    colors_rgb: np.ndarray  # (N,3) uint8
    pixels_considered: int
    pixels_used: int


@dataclass
class OptimizationResult:
    status: str
    message: str | None
    correspondences: int
    scale: float
    log_scale: float
    tz_init: float
    delta_tz: float
    tz: float
    history: dict[str, list[float]]


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


def save_mesh_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(path))


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
        print(
            "CUDA device requested but unavailable; falling back to CPU for alignment."
        )
        return torch.device("cpu")
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


def maybe_resize_for_optimization(
    depth: np.ndarray,
    masks: list[np.ndarray | None],
    frame: np.ndarray,
    k: np.ndarray,
    opt_max_side: int,
) -> tuple[np.ndarray, list[np.ndarray | None], np.ndarray, np.ndarray, float]:
    h, w = depth.shape
    if opt_max_side <= 0 or max(h, w) <= opt_max_side:
        return depth, masks, frame, k, 1.0

    scale = float(opt_max_side) / float(max(h, w))
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))

    depth_rs = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    frame_rs = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
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
    return (
        depth_rs.astype(np.float32),
        masks_rs,
        frame_rs,
        k_rs.astype(np.float32),
        scale,
    )


def project_points_cv(points_cv: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = points_cv[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((points_cv.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        pts = points_cv[valid]
        z_valid = pts[:, 2]
        uv_valid = np.empty((pts.shape[0], 2), dtype=np.float32)
        uv_valid[:, 0] = (pts[:, 0] * k[0, 0]) / z_valid + k[0, 2]
        uv_valid[:, 1] = (pts[:, 1] * k[1, 1]) / z_valid + k[1, 2]
        uv[valid] = uv_valid
    return uv, valid


def draw_overlay_points(
    frame_bgr: np.ndarray,
    verts_cv_list: list[np.ndarray],
    names: list[str],
    k: np.ndarray,
    max_points_per_mesh: int = 30000,
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
        if len(verts) == 0:
            continue
        if max_points_per_mesh > 0 and len(verts) > max_points_per_mesh:
            stride = max(1, int(len(verts) / max_points_per_mesh))
            pts = verts[::stride]
        else:
            pts = verts
        uv, valid = project_points_cv(pts, k)
        uv_i = np.round(uv[valid]).astype(np.int32)
        inb = (
            (uv_i[:, 0] >= 0)
            & (uv_i[:, 0] < w)
            & (uv_i[:, 1] >= 0)
            & (uv_i[:, 1] < h)
        )
        uv_i = uv_i[inb]
        color = palette[idx % len(palette)]
        for x, y in uv_i:
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


def colorize_points_by_xyz(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    denom = np.maximum(pmax - pmin, 1e-8)
    rgb = (255.0 * (points - pmin) / denom).clip(0.0, 255.0)
    return rgb.astype(np.uint8)


def save_colored_point_cloud(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    if points.shape[0] == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((0, 3), dtype=np.float32))
    else:
        colors_rgba = np.concatenate(
            [colors_rgb.astype(np.uint8), 255 * np.ones((points.shape[0], 1), dtype=np.uint8)],
            axis=1,
        )
        cloud = trimesh.points.PointCloud(
            vertices=points.astype(np.float32), colors=colors_rgba
        )
    cloud.export(str(path))


def draw_colored_uv_points(
    canvas_bgr: np.ndarray, uv: np.ndarray, colors_rgb: np.ndarray, radius: int
) -> None:
    h, w = canvas_bgr.shape[:2]
    for (u, v), rgb in zip(uv, colors_rgb):
        x = int(round(float(u)))
        y = int(round(float(v)))
        if 0 <= x < w and 0 <= y < h:
            color_bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
            cv2.circle(canvas_bgr, (x, y), radius, color_bgr, -1)


def save_correspondence_snapshot(
    out_dir: Path,
    iter_idx: int,
    depth_points: np.ndarray,
    transformed_mesh_points: np.ndarray,
    uv_ref: np.ndarray,
    colors_rgb: np.ndarray,
    intrinsics: np.ndarray,
    frame_bgr: np.ndarray,
    point_radius: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_colored_point_cloud(
        out_dir / f"iter_{iter_idx:05d}_depth_fixed.ply",
        depth_points,
        colors_rgb,
    )
    save_colored_point_cloud(
        out_dir / f"iter_{iter_idx:05d}_mesh_transformed.ply",
        transformed_mesh_points,
        colors_rgb,
    )

    depth_vis = frame_bgr.copy()
    draw_colored_uv_points(
        canvas_bgr=depth_vis,
        uv=uv_ref,
        colors_rgb=colors_rgb,
        radius=point_radius,
    )
    cv2.putText(
        depth_vis,
        "Depth points (fixed correspondences)",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    mesh_uv, mesh_valid = project_points_cv(transformed_mesh_points, intrinsics)
    mesh_vis = frame_bgr.copy()
    draw_colored_uv_points(
        canvas_bgr=mesh_vis,
        uv=mesh_uv[mesh_valid],
        colors_rgb=colors_rgb[mesh_valid],
        radius=point_radius,
    )
    cv2.putText(
        mesh_vis,
        "Transformed mesh points",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    combined = np.concatenate([depth_vis, mesh_vis], axis=1)
    cv2.imwrite(
        str(out_dir / f"iter_{iter_idx:05d}_correspondence_projection.png"), combined
    )


def build_mesh_correspondences(
    verts_cv: np.ndarray,
    faces: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    bin_size: int,
    max_points: int,
    seed: int,
) -> CorrespondenceSet:
    h, w = depth.shape
    cams = build_cameras(intrinsics, w, h, device)
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),
        max_faces_per_bin=300000,
    )
    rasterizer = MeshRasterizer(cameras=cams, raster_settings=raster_settings)

    cv_to_p3d = torch.from_numpy(F_P3D_TO_CV).to(device=device, dtype=torch.float32)
    verts_t = torch.from_numpy(verts_cv).to(device=device, dtype=torch.float32)
    verts_p3d = verts_t @ cv_to_p3d.transpose(0, 1)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])

    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    bary = fragments.bary_coords[0, ..., 0, :].detach().cpu().numpy().astype(np.float32)

    valid = pix_to_face >= 0
    valid &= np.isfinite(depth) & (depth > 0.0)
    if mask is not None:
        valid &= mask > 0.5

    ys, xs = np.nonzero(valid)
    pixels_considered = int(ys.shape[0])
    if pixels_considered == 0:
        return CorrespondenceSet(
            mesh_points_base=np.zeros((0, 3), dtype=np.float32),
            depth_points=np.zeros((0, 3), dtype=np.float32),
            uv_ref=np.zeros((0, 2), dtype=np.float32),
            colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            pixels_considered=0,
            pixels_used=0,
        )

    if max_points > 0 and pixels_considered > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(pixels_considered, size=max_points, replace=False)
        ys = ys[keep]
        xs = xs[keep]

    face_idx = pix_to_face[ys, xs].astype(np.int64)
    bary_sel = bary[ys, xs]  # (N,3)
    tri = verts_cv[faces[face_idx]]  # (N,3,3)

    mesh_points = (
        bary_sel[:, [0]] * tri[:, 0, :]
        + bary_sel[:, [1]] * tri[:, 1, :]
        + bary_sel[:, [2]] * tri[:, 2, :]
    ).astype(np.float32)

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    z = depth[ys, xs].astype(np.float32)
    x = ((xs.astype(np.float32) - cx) / fx) * z
    y = ((ys.astype(np.float32) - cy) / fy) * z
    depth_points = np.stack([x, y, z], axis=1).astype(np.float32)

    uv_ref = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    colors_rgb = colorize_points_by_xyz(mesh_points)

    return CorrespondenceSet(
        mesh_points_base=mesh_points,
        depth_points=depth_points,
        uv_ref=uv_ref.astype(np.float32),
        colors_rgb=colors_rgb,
        pixels_considered=pixels_considered,
        pixels_used=int(mesh_points.shape[0]),
    )


def compute_losses_torch(
    mesh_points_base: torch.Tensor,
    depth_points: torch.Tensor,
    uv_ref: torch.Tensor,
    intrinsics: torch.Tensor,
    log_scale: torch.Tensor,
    delta_tz: torch.Tensor,
    tz_init: torch.Tensor,
    w_corr: float,
    w_reproj: float,
    w_scale_reg: float,
    w_tz_reg: float,
) -> dict[str, torch.Tensor]:
    scale = torch.exp(log_scale)
    tz = tz_init + delta_tz

    transformed = scale * mesh_points_base
    transformed = torch.stack(
        [transformed[:, 0], transformed[:, 1], transformed[:, 2] + tz], dim=1
    )

    z = torch.clamp(transformed[:, 2], min=1e-6)
    u = intrinsics[0, 0] * transformed[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * transformed[:, 1] / z + intrinsics[1, 2]
    uv = torch.stack([u, v], dim=1)

    e_corr = ((transformed - depth_points) ** 2).sum(dim=1).mean()
    e_reproj = ((uv - uv_ref) ** 2).sum(dim=1).mean()
    e_scale_reg = log_scale.pow(2)
    e_tz_reg = delta_tz.pow(2)
    e_total = (
        float(w_corr) * e_corr
        + float(w_reproj) * e_reproj
        + float(w_scale_reg) * e_scale_reg
        + float(w_tz_reg) * e_tz_reg
    )
    return {
        "total": e_total,
        "corr": e_corr,
        "reproj": e_reproj,
        "scale_reg": e_scale_reg,
        "tz_reg": e_tz_reg,
        "scale": scale,
        "tz": tz,
        "transformed": transformed,
    }


def new_history_dict() -> dict[str, list[float]]:
    return {
        "iter": [],
        "total": [],
        "corr": [],
        "reproj": [],
        "scale_reg": [],
        "tz_reg": [],
        "scale": [],
        "tz": [],
    }


def append_history(
    history: dict[str, list[float]], iter_idx: int, losses: dict[str, torch.Tensor]
) -> None:
    history["iter"].append(int(iter_idx))
    history["total"].append(float(losses["total"].detach().cpu().item()))
    history["corr"].append(float(losses["corr"].detach().cpu().item()))
    history["reproj"].append(float(losses["reproj"].detach().cpu().item()))
    history["scale_reg"].append(float(losses["scale_reg"].detach().cpu().item()))
    history["tz_reg"].append(float(losses["tz_reg"].detach().cpu().item()))
    history["scale"].append(float(losses["scale"].detach().cpu().item()))
    history["tz"].append(float(losses["tz"].detach().cpu().item()))


def optimize_scale_tz(
    corr: CorrespondenceSet,
    intrinsics: np.ndarray,
    device: torch.device,
    iters: int,
    lr: float,
    w_corr: float,
    w_reproj: float,
    w_scale_reg: float,
    w_tz_reg: float,
    min_scale: float,
    max_scale: float,
    max_abs_delta_tz: float,
    log_every: int,
    corr_save_every: int,
    corr_vis_dir: Path,
    corr_vis_frame: np.ndarray,
    corr_vis_point_radius: int,
) -> OptimizationResult:
    n = int(corr.mesh_points_base.shape[0])
    if n == 0:
        return OptimizationResult(
            status="skipped_no_correspondence",
            message="No valid pixel correspondences after rendering/depth/mask filtering.",
            correspondences=0,
            scale=1.0,
            log_scale=0.0,
            tz_init=0.0,
            delta_tz=0.0,
            tz=0.0,
            history=new_history_dict(),
        )

    mesh_points_t = torch.from_numpy(corr.mesh_points_base).to(
        device=device, dtype=torch.float32
    )
    depth_points_t = torch.from_numpy(corr.depth_points).to(
        device=device, dtype=torch.float32
    )
    uv_ref_t = torch.from_numpy(corr.uv_ref).to(device=device, dtype=torch.float32)
    k_t = torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32)

    # tz_init_val = float(np.median(corr.depth_points[:, 2] - corr.mesh_points_base[:, 2]))
    # Start from the mesh's current 3D placement: scale=1 and no extra z-translation.
    tz_init_val = 0.0
    tz_init_t = torch.tensor(tz_init_val, device=device, dtype=torch.float32)

    log_scale = torch.nn.Parameter(
        torch.zeros((), device=device, dtype=torch.float32)
    )
    delta_tz = torch.nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([log_scale, delta_tz], lr=float(lr))

    min_log_scale = math.log(float(min_scale))
    max_log_scale = math.log(float(max_scale))
    history = new_history_dict()

    with torch.no_grad():
        init_losses = compute_losses_torch(
            mesh_points_base=mesh_points_t,
            depth_points=depth_points_t,
            uv_ref=uv_ref_t,
            intrinsics=k_t,
            log_scale=log_scale,
            delta_tz=delta_tz,
            tz_init=tz_init_t,
            w_corr=w_corr,
            w_reproj=w_reproj,
            w_scale_reg=w_scale_reg,
            w_tz_reg=w_tz_reg,
        )
        append_history(history, 0, init_losses)
        save_correspondence_snapshot(
            out_dir=corr_vis_dir,
            iter_idx=0,
            depth_points=corr.depth_points,
            transformed_mesh_points=init_losses["transformed"].detach().cpu().numpy(),
            uv_ref=corr.uv_ref,
            colors_rgb=corr.colors_rgb,
            intrinsics=intrinsics,
            frame_bgr=corr_vis_frame,
            point_radius=corr_vis_point_radius,
        )

    for iter_idx in range(1, int(iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses_torch(
            mesh_points_base=mesh_points_t,
            depth_points=depth_points_t,
            uv_ref=uv_ref_t,
            intrinsics=k_t,
            log_scale=log_scale,
            delta_tz=delta_tz,
            tz_init=tz_init_t,
            w_corr=w_corr,
            w_reproj=w_reproj,
            w_scale_reg=w_scale_reg,
            w_tz_reg=w_tz_reg,
        )
        losses["total"].backward()
        optimizer.step()

        with torch.no_grad():
            log_scale.clamp_(min_log_scale, max_log_scale)
            delta_tz.clamp_(-float(max_abs_delta_tz), float(max_abs_delta_tz))

            eval_losses = compute_losses_torch(
                mesh_points_base=mesh_points_t,
                depth_points=depth_points_t,
                uv_ref=uv_ref_t,
                intrinsics=k_t,
                log_scale=log_scale,
                delta_tz=delta_tz,
                tz_init=tz_init_t,
                w_corr=w_corr,
                w_reproj=w_reproj,
                w_scale_reg=w_scale_reg,
                w_tz_reg=w_tz_reg,
            )
            append_history(history, iter_idx, eval_losses)

            if log_every > 0 and (iter_idx % int(log_every) == 0 or iter_idx == int(iters)):
                print(
                    f"iter={iter_idx:04d} total={history['total'][-1]:.6f} "
                    f"corr={history['corr'][-1]:.6f} "
                    f"reproj={history['reproj'][-1]:.6f} "
                    f"scale={history['scale'][-1]:.6f} "
                    f"tz={history['tz'][-1]:.6f}"
                )

            if (
                corr_save_every > 0
                and (iter_idx % int(corr_save_every) == 0 or iter_idx == int(iters))
            ):
                save_correspondence_snapshot(
                    out_dir=corr_vis_dir,
                    iter_idx=iter_idx,
                    depth_points=corr.depth_points,
                    transformed_mesh_points=eval_losses["transformed"]
                    .detach()
                    .cpu()
                    .numpy(),
                    uv_ref=corr.uv_ref,
                    colors_rgb=corr.colors_rgb,
                    intrinsics=intrinsics,
                    frame_bgr=corr_vis_frame,
                    point_radius=corr_vis_point_radius,
                )

    scale_final = float(math.exp(float(log_scale.detach().cpu().item())))
    log_scale_final = float(log_scale.detach().cpu().item())
    delta_tz_final = float(delta_tz.detach().cpu().item())
    tz_final = float(tz_init_val + delta_tz_final)

    return OptimizationResult(
        status="optimized",
        message=None,
        correspondences=n,
        scale=scale_final,
        log_scale=log_scale_final,
        tz_init=tz_init_val,
        delta_tz=delta_tz_final,
        tz=tz_final,
        history=history,
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
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=160)
    plt.close()


def plot_loss_curves_separate(
    history: dict[str, list[float]],
    out_dir: Path,
    slug: str,
    mesh_name: str,
) -> dict[str, str]:
    loss_keys = {
        "total": "E_total",
        "corr": "E_corr",
        "reproj": "E_reproj",
        "scale_reg": "E_scale_reg",
        "tz_reg": "E_tz_reg",
    }
    paths: dict[str, str] = {}
    for key, label in loss_keys.items():
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


def save_loss_history_csv(history: dict[str, list[float]], out_path: Path) -> None:
    if len(history["iter"]) == 0:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack(
        [
            np.array(history["iter"], dtype=np.int32),
            np.array(history["total"], dtype=np.float32),
            np.array(history["corr"], dtype=np.float32),
            np.array(history["reproj"], dtype=np.float32),
            np.array(history["scale_reg"], dtype=np.float32),
            np.array(history["tz_reg"], dtype=np.float32),
            np.array(history["scale"], dtype=np.float32),
            np.array(history["tz"], dtype=np.float32),
        ]
    )
    np.savetxt(
        str(out_path),
        arr,
        fmt=["%d", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f", "%.8f"],
        delimiter=",",
        header="iter,total,corr,reproj,scale_reg,tz_reg,scale,tz",
        comments="",
    )


def save_energy_terms_file(path: Path) -> None:
    content = """Mesh-wise optimization variables:
  alpha (log-scale), delta_t_z
  s = exp(alpha), t_z = t_z_init + delta_t_z

Per-correspondence transform:
  m_i' = s * m_i + [0, 0, t_z]^T

Projection:
  u_i = fx * m'_ix / m'_iz + cx
  v_i = fy * m'_iy / m'_iz + cy

Loss terms:
  E_corr      = (1/N) * sum_i || m_i' - d_i ||_2^2
  E_reproj    = (1/N) * sum_i || [u_i, v_i] - [u_i^0, v_i^0] ||_2^2
  E_scale_reg = alpha^2
  E_tz_reg    = delta_t_z^2

Combined:
  E_total = w_corr * E_corr
          + w_reproj * E_reproj
          + w_scale_reg * E_scale_reg
          + w_tz_reg * E_tz_reg
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple mesh-depth alignment with fixed render-to-depth correspondences "
            "and scale+t_z optimization."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_01")

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
        default="./output",
        help="Root output directory; results are written to output_root/video_name.",
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
        help="Optional binary human mask at frame_00 resolution.",
    )
    parser.add_argument(
        "--human_coord",
        type=str,
        choices=["opencv", "pytorch3d"],
        default="opencv",
        help="Coordinate frame of the input human OBJ.",
    )
    parser.add_argument(
        "--include_human",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include human mesh in alignment.",
    )

    parser.add_argument(
        "--intrinsics_source",
        type=str,
        choices=["object", "depth"],
        default="object",
        help=(
            "'object': camera_intrinsics.json intrinsics_pixels_3x3. "
            "'depth': pose_estimation.json intrinsics."
        ),
    )
    parser.add_argument(
        "--intrinsics_warn_threshold_px",
        type=float,
        default=100.0,
        help="Warn if max |object - depth intrinsics| exceeds this threshold in pixels.",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opt_max_side", type=int, default=1280)
    parser.add_argument("--bin_size", type=int, default=0)

    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--w_corr", type=float, default=1.0)
    parser.add_argument("--w_reproj", type=float, default=1e-3)
    parser.add_argument("--w_scale_reg", type=float, default=1e-3)
    parser.add_argument("--w_tz_reg", type=float, default=1e-3)

    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=5.0)
    parser.add_argument("--max_abs_delta_tz", type=float, default=2.0)

    parser.add_argument("--max_correspondences_per_mesh", type=int, default=80000)
    parser.add_argument("--min_correspondences_per_mesh", type=int, default=128)
    parser.add_argument("--corr_save_every", type=int, default=50)
    parser.add_argument("--corr_vis_point_radius", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_scale <= 0.0:
        raise ValueError("--min_scale must be > 0.")
    if args.max_scale <= args.min_scale:
        raise ValueError("--max_scale must be > --min_scale.")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0.")
    if args.max_correspondences_per_mesh <= 0:
        raise ValueError("--max_correspondences_per_mesh must be > 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    object_video_dir = resolve_path(args.object_video_dir, script_dir) or (
        script_dir.parent / "Generate_Object_Mesh" / "output" / args.video_name
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
    if not depth_video_dir.exists():
        raise FileNotFoundError(f"Depth dir not found: {depth_video_dir}")
    if args.include_human and not human_video_dir.exists():
        raise FileNotFoundError(f"Human dir not found: {human_video_dir}")

    summary_path = object_video_dir / "frame_00_segmentation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Segmentation summary not found: {summary_path}")
    summary = load_json(summary_path)

    pose_json_path = depth_video_dir / "pose_estimation.json"
    run_summary_path = depth_video_dir / "run_summary.json"
    object_intrinsics_json_path = object_video_dir / "camera_intrinsics.json"
    depth_npy_path = depth_video_dir / "metric_depth" / "metric_depth.npy"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found: {run_summary_path}")
    if not pose_json_path.exists():
        raise FileNotFoundError(f"pose_estimation.json not found: {pose_json_path}")
    if not depth_npy_path.exists():
        raise FileNotFoundError(f"metric_depth.npy not found: {depth_npy_path}")

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

    k_diff = np.abs(k_object_full - k_depth_full)
    max_k_diff = float(np.max(k_diff))
    if max_k_diff > float(args.intrinsics_warn_threshold_px):
        print(
            "WARNING: object/depth intrinsics mismatch is large "
            f"(max abs diff: {max_k_diff:.3f}px)."
        )
        print(
            f"Using '{args.intrinsics_source}' intrinsics for correspondence back-projection."
        )

    assets: list[MeshAsset] = []
    for obj in summary.get("objects", []):
        if not obj.get("success", False):
            continue

        obj_name = str(obj["object"])
        out_dir_raw = Path(str(obj["output_dir"]))
        if out_dir_raw.is_absolute():
            obj_dir = out_dir_raw.resolve()
        else:
            obj_dir = (object_video_dir / out_dir_raw).resolve()

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
        mask_path = (obj_dir / str(mask_rel)).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for object '{obj_name}': {mask_path}"
            )

        verts_src, faces = load_mesh(mesh_path)
        mask = load_binary_mask(mask_path, (depth_h, depth_w))
        source_to_cv = (F_P3D_TO_CV @ R_Y_UP_TO_Z_UP).astype(np.float32)
        assets.append(
            MeshAsset(
                name=obj_name,
                slug=slugify(obj_name),
                kind="object",
                source_mesh_path=mesh_path,
                source_coord="mesh_posed_glb_y_up",
                verts_source=verts_src,
                faces=faces,
                source_to_cv=source_to_cv,
                mask_path=mask_path,
                mask=mask,
            )
        )

    if args.include_human:
        human_obj_path = resolve_path(args.human_obj, script_dir)
        if human_obj_path is None:
            human_obj_path = find_first_human_obj(human_video_dir / "output_objs")
        if not human_obj_path.exists():
            raise FileNotFoundError(f"Human OBJ not found: {human_obj_path}")

        human_mask_path = resolve_path(args.human_mask, script_dir)
        human_mask = None
        if human_mask_path is not None:
            human_mask = load_binary_mask(human_mask_path, (depth_h, depth_w))

        human_verts_src, human_faces = load_mesh(human_obj_path)
        human_source_to_cv = (
            np.eye(3, dtype=np.float32)
            if args.human_coord == "opencv"
            else F_P3D_TO_CV.copy()
        )
        assets = [
            MeshAsset(
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
        ] + assets

    if len(assets) == 0:
        raise RuntimeError("No meshes found to align.")

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

    overlay_before = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_base_cv_np,
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_before.png"), overlay_before)

    correspondences: list[CorrespondenceSet] = []
    correspondence_stats: list[dict[str, Any]] = []
    for idx, (asset, verts_cv) in enumerate(zip(assets, verts_base_cv_np)):
        corr = build_mesh_correspondences(
            verts_cv=verts_cv,
            faces=asset.faces,
            depth=depth_opt,
            intrinsics=k_opt,
            mask=asset.mask,
            device=device,
            bin_size=int(args.bin_size),
            max_points=int(args.max_correspondences_per_mesh),
            seed=int(args.seed + 7919 * (idx + 1)),
        )
        correspondences.append(corr)
        correspondence_stats.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "mask_path": None if asset.mask_path is None else str(asset.mask_path),
                "pixels_considered": int(corr.pixels_considered),
                "pixels_used": int(corr.pixels_used),
            }
        )
        print(
            f"[{asset.name}] correspondences: used={corr.pixels_used} "
            f"(considered={corr.pixels_considered})"
        )

    losses_dir = output_dir / "loss_curves"
    corr_dir_root = output_dir / "correspondences"
    optimization_results: list[OptimizationResult] = []
    loss_plot_paths_by_slug: dict[str, dict[str, str]] = {}
    for asset, corr in zip(assets, correspondences):
        mesh_corr_dir = corr_dir_root / asset.slug
        if corr.mesh_points_base.shape[0] < int(args.min_correspondences_per_mesh):
            msg = (
                f"Too few correspondences ({corr.mesh_points_base.shape[0]}) < "
                f"{int(args.min_correspondences_per_mesh)}"
            )
            result = OptimizationResult(
                status="skipped_too_few_correspondences",
                message=msg,
                correspondences=int(corr.mesh_points_base.shape[0]),
                scale=1.0,
                log_scale=0.0,
                tz_init=0.0,
                delta_tz=0.0,
                tz=0.0,
                history=new_history_dict(),
            )
            optimization_results.append(result)
            print(f"[{asset.name}] {msg}")
            continue

        print(f"[{asset.name}] optimizing scale + t_z ...")
        result = optimize_scale_tz(
            corr=corr,
            intrinsics=k_opt,
            device=device,
            iters=int(args.iters),
            lr=float(args.lr),
            w_corr=float(args.w_corr),
            w_reproj=float(args.w_reproj),
            w_scale_reg=float(args.w_scale_reg),
            w_tz_reg=float(args.w_tz_reg),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
            max_abs_delta_tz=float(args.max_abs_delta_tz),
            log_every=int(args.log_every),
            corr_save_every=int(args.corr_save_every),
            corr_vis_dir=mesh_corr_dir,
            corr_vis_frame=frame_opt,
            corr_vis_point_radius=int(args.corr_vis_point_radius),
        )
        optimization_results.append(result)

        if len(result.history["iter"]) > 0:
            loss_plot_paths_by_slug[asset.slug] = plot_loss_curves_separate(
                history=result.history,
                out_dir=losses_dir,
                slug=asset.slug,
                mesh_name=asset.name,
            )
            save_loss_history_csv(
                history=result.history,
                out_path=losses_dir / f"{asset.slug}_loss.csv",
            )

    meshes_out_dir = output_dir / "meshes"
    meshes_out_dir.mkdir(parents=True, exist_ok=True)
    transforms_out: list[dict[str, Any]] = []
    verts_after_np: list[np.ndarray] = []
    for asset, verts_cv, result in zip(assets, verts_base_cv_np, optimization_results):
        scale = float(result.scale)
        tz = float(result.tz)
        verts_aligned = (scale * verts_cv).astype(np.float32)
        verts_aligned[:, 2] += tz
        verts_after_np.append(verts_aligned)

        out_mesh_path = meshes_out_dir / f"{asset.slug}.obj"
        save_mesh_obj(out_mesh_path, verts_aligned, asset.faces)

        source_to_cv_4x4 = np.eye(4, dtype=np.float32)
        source_to_cv_4x4[:3, :3] = asset.source_to_cv.astype(np.float32)

        source_to_aligned_4x4 = np.eye(4, dtype=np.float32)
        source_to_aligned_4x4[:3, :3] = (
            scale * asset.source_to_cv.astype(np.float32)
        )
        source_to_aligned_4x4[:3, 3] = np.array([0.0, 0.0, tz], dtype=np.float32)

        transforms_out.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "source_mesh_path": str(asset.source_mesh_path),
                "source_coordinate": asset.source_coord,
                "source_to_cv_rotation_3x3": asset.source_to_cv.tolist(),
                "source_to_cv_matrix_4x4": source_to_cv_4x4.tolist(),
                "optimized_parameters": {
                    "status": result.status,
                    "message": result.message,
                    "log_scale_alpha": float(result.log_scale),
                    "scale_exp_alpha": float(scale),
                    "tz_init_m": float(result.tz_init),
                    "delta_tz_m": float(result.delta_tz),
                    "tz_total_m": float(tz),
                    "num_correspondences": int(result.correspondences),
                },
                "source_to_aligned_matrix_4x4": source_to_aligned_4x4.tolist(),
                "aligned_mesh_obj": str(out_mesh_path),
            }
        )

    overlay_after = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_after_np,
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_after.png"), overlay_after)

    save_energy_terms_file(output_dir / "energy_terms.txt")

    summary_out = {
        "inputs": {
            "video_name": args.video_name,
            "object_video_dir": str(object_video_dir),
            "depth_video_dir": str(depth_video_dir),
            "human_video_dir": str(human_video_dir),
            "summary_json": str(summary_path),
            "depth_npy": str(depth_npy_path),
            "pose_json": str(pose_json_path),
            "frame_00": str(frame_path),
        },
        "camera": {
            "intrinsics_source": args.intrinsics_source,
            "intrinsics_3x3": k_full.tolist(),
            "object_intrinsics_3x3": k_object_full.tolist(),
            "depth_intrinsics_3x3": k_depth_full.tolist(),
            "max_abs_object_depth_intrinsics_diff_px": max_k_diff,
        },
        "optimization_settings": {
            "iters": int(args.iters),
            "lr": float(args.lr),
            "w_corr": float(args.w_corr),
            "w_reproj": float(args.w_reproj),
            "w_scale_reg": float(args.w_scale_reg),
            "w_tz_reg": float(args.w_tz_reg),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
            "max_abs_delta_tz": float(args.max_abs_delta_tz),
            "max_correspondences_per_mesh": int(args.max_correspondences_per_mesh),
            "min_correspondences_per_mesh": int(args.min_correspondences_per_mesh),
            "corr_save_every": int(args.corr_save_every),
            "opt_max_side": int(args.opt_max_side),
            "resize_scale_for_optimization": float(resize_scale),
        },
        "energy_terms": {
            "transform": "m_i' = exp(alpha) * m_i + [0, 0, t_z_init + delta_t_z]^T",
            "projection_u": "u_i = fx * m'_ix / m'_iz + cx",
            "projection_v": "v_i = fy * m'_iy / m'_iz + cy",
            "E_corr": "(1/N) * sum_i ||m_i' - d_i||_2^2",
            "E_reproj": "(1/N) * sum_i ||[u_i,v_i]-[u_i^0,v_i^0]||_2^2",
            "E_scale_reg": "alpha^2",
            "E_tz_reg": "delta_t_z^2",
            "E_total": "w_corr*E_corr + w_reproj*E_reproj + w_scale_reg*E_scale_reg + w_tz_reg*E_tz_reg",
        },
        "correspondence_stats": correspondence_stats,
        "per_mesh_optimization": [
            {
                "name": asset.name,
                "slug": asset.slug,
                "status": result.status,
                "message": result.message,
                "num_correspondences": int(result.correspondences),
                "final_scale": float(result.scale),
                "final_tz_m": float(result.tz),
                "final_total_loss": None
                if len(result.history["total"]) == 0
                else float(result.history["total"][-1]),
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
                "correspondence_dir": str(corr_dir_root / asset.slug),
            }
            for asset, result in zip(assets, optimization_results)
        ],
        "outputs": {
            "output_dir": str(output_dir),
            "meshes_dir": str(meshes_out_dir),
            "overlay_before": str(output_dir / "overlay_before.png"),
            "overlay_after": str(output_dir / "overlay_after.png"),
            "energy_terms_txt": str(output_dir / "energy_terms.txt"),
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
