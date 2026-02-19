"""Align first-frame human/object meshes via overlap correspondences.

Pipeline:
1. Load meshes, masks, depth, and intrinsics using the repo's fixed layout.
2. For each mesh, rasterize into the camera and keep overlap pixels:
   rendered mesh ∩ mask ∩ valid metric depth.
3. Convert overlap pixels to:
   - mesh surface points (face id + barycentric interpolation)
   - depth point cloud points (pixel + depth + intrinsics)
4. Optimize per-mesh scale + 3D translation so correspondences align.
5. Export aligned meshes, transforms, overlays, and a result JSON.
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
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

# GLB meshes are Y-up by convention; SAM3 camera transforms are effectively Z-up.
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
    tvec: torch.nn.Parameter
    tvec_init: torch.Tensor
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


def transform_vertices_scale_t(
    verts_base_cv: torch.Tensor,
    log_s: torch.Tensor,
    tvec: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(log_s) * verts_base_cv + tvec


def save_mesh_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(path))


def export_meshes_and_transforms(
    assets: list[MeshAsset],
    verts_aligned: list[np.ndarray],
    scales_np: np.ndarray,
    t_np: np.ndarray,
    meshes_out_dir: Path,
    transforms_out_path: Path,
) -> list[dict[str, Any]]:
    meshes_out_dir.mkdir(parents=True, exist_ok=True)

    transforms_out: list[dict[str, Any]] = []
    eye3 = np.eye(3, dtype=np.float32)
    rot_axis_angle = np.zeros((3,), dtype=np.float32)

    for j, asset in enumerate(assets):
        out_mesh_path = meshes_out_dir / f"{asset.slug}.obj"
        save_mesh_obj(out_mesh_path, verts_aligned[j], asset.faces)

        c = asset.source_to_cv.astype(np.float32)
        s = float(scales_np[j])
        t = t_np[j].astype(np.float32)

        source_to_cv_4x4 = np.eye(4, dtype=np.float32)
        source_to_cv_4x4[:3, :3] = c

        source_to_aligned_4x4 = np.eye(4, dtype=np.float32)
        source_to_aligned_4x4[:3, :3] = (s * eye3 @ c).astype(np.float32)
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
                    "rotation_axis_angle": rot_axis_angle.tolist(),
                    "rotation_matrix_3x3": eye3.tolist(),
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
) -> tuple[np.ndarray, np.ndarray]:
    z = points_cv[:, 2]
    valid = z > 1e-6
    pts = points_cv[valid]
    z = z[valid]
    u = (pts[:, 0] * k[0, 0]) / z + k[0, 2]
    v = (pts[:, 1] * k[1, 1]) / z + k[1, 2]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui[inb], vi[inb]


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

        u, v = project_points_cv(pts, k, w, h)
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


def depth_to_xyz_map(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth.astype(np.float32)
    x = ((xx - cx) / fx) * z
    y = ((yy - cy) / fy) * z
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32)
    invalid = ~np.isfinite(z) | (z <= 0.0)
    xyz[invalid] = 0.0
    return xyz


def huber_loss(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_r = residual.abs()
    return torch.where(
        abs_r <= delta,
        0.5 * residual * residual,
        delta * (abs_r - 0.5 * delta),
    )


def trim_distances(
    distances: torch.Tensor,
    trim_quantile: float,
) -> tuple[torch.Tensor, int]:
    if distances.numel() == 0:
        return distances, 0
    q = float(trim_quantile)
    if q <= 0.0 or q >= 1.0 or distances.numel() == 1:
        return distances, int(distances.numel())
    keep = max(1, int(math.ceil(q * distances.numel())))
    return torch.topk(distances, k=keep, largest=False).values, keep


def rasterize_single_mesh(
    rasterizer: MeshRasterizer,
    verts_cv: torch.Tensor,
    faces_t: torch.Tensor,
    cv_to_p3d: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    verts_p3d = verts_cv @ cv_to_p3d.transpose(0, 1)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])
    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0, :]
    zbuf = fragments.zbuf[0, ..., 0]
    return pix_to_face, bary_coords, zbuf


def gather_overlap_correspondences(
    pix_to_face: torch.Tensor,
    bary_coords: torch.Tensor,
    mask_t: torch.Tensor,
    depth_valid_t: torch.Tensor,
    xyz_map_t: torch.Tensor,
    verts_base_cv_t: torch.Tensor,
    faces_t: torch.Tensor,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    rendered = pix_to_face >= 0
    overlap = rendered & (mask_t > 0.5) & depth_valid_t
    n_rendered = int(rendered.sum().item())
    n_overlap = int(overlap.sum().item())

    if n_overlap <= 0:
        empty = torch.zeros((0, 3), device=verts_base_cv_t.device, dtype=torch.float32)
        return empty, empty, {
            "rendered_pixels": n_rendered,
            "overlap_pixels": 0,
            "used_correspondences": 0,
        }

    yx = torch.nonzero(overlap, as_tuple=False)
    if max_points > 0 and yx.shape[0] > max_points:
        perm = torch.randperm(yx.shape[0], device=yx.device)[:max_points]
        yx = yx[perm]
    yy = yx[:, 0]
    xx = yx[:, 1]

    face_idx = pix_to_face[yy, xx].long()
    bary = bary_coords[yy, xx]  # (N,3)
    tri_idx = faces_t[face_idx]  # (N,3)
    tri_base = verts_base_cv_t[tri_idx]  # (N,3,3)
    mesh_points_base = (tri_base * bary.unsqueeze(-1)).sum(dim=1)
    depth_points = xyz_map_t[yy, xx]

    return mesh_points_base, depth_points, {
        "rendered_pixels": n_rendered,
        "overlap_pixels": n_overlap,
        "used_correspondences": int(mesh_points_base.shape[0]),
    }


def correspondence_loss(
    mesh_points_base: torch.Tensor,
    depth_points: torch.Tensor,
    log_s: torch.Tensor,
    tvec: torch.Tensor,
    trim_quantile: float,
    huber_delta_3d: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if mesh_points_base.shape[0] == 0:
        z = log_s.new_zeros(())
        return z, {
            "corr_used": 0,
            "corr_kept": 0,
            "mean_dist_m": None,
            "median_dist_m": None,
        }

    pred_points = torch.exp(log_s) * mesh_points_base + tvec
    diff = pred_points - depth_points
    dist = torch.linalg.norm(diff, dim=1)
    dist_kept, kept = trim_distances(dist, trim_quantile=trim_quantile)
    loss_data = huber_loss(dist_kept, float(huber_delta_3d)).mean()

    dist_np = dist.detach().cpu().numpy()
    return loss_data, {
        "corr_used": int(dist.shape[0]),
        "corr_kept": int(kept),
        "mean_dist_m": float(np.mean(dist_np)),
        "median_dist_m": float(np.median(dist_np)),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align human and object meshes using first-frame overlap correspondences "
            "(rendered mesh pixels overlapped with masks and metric depth)."
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
        default="./output_v1",
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
        help=(
            "Optional human binary mask for frame_00. If omitted, use rendered "
            "human silhouette from initial pose."
        ),
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
            "'object': camera_intrinsics.json intrinsics_pixels_3x3, "
            "'depth': pose_estimation.json intrinsics."
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

    parser.add_argument("--iters", type=int, nargs=2, default=[1200, 1800])
    parser.add_argument("--stage_lr", type=float, nargs=2, default=[3e-3, 1e-3])
    parser.add_argument(
        "--stage_early_stop_patience",
        type=int,
        nargs=2,
        default=[80, 120],
        help="Per-stage early stop patience per mesh. <=0 disables early stop.",
    )
    parser.add_argument(
        "--stage_early_stop_min_delta",
        type=float,
        nargs=2,
        default=[1e-5, 5e-6],
        help="Per-stage minimum improvement for resetting early-stop patience.",
    )

    parser.add_argument("--corr_max_points", type=int, default=12000)
    parser.add_argument("--min_corr_points", type=int, default=128)
    parser.add_argument("--trim_quantile", type=float, default=0.85)
    parser.add_argument("--huber_delta_3d", type=float, default=0.03)
    parser.add_argument("--reg_scale", type=float, default=1e-3)
    parser.add_argument("--reg_trans", type=float, default=1e-4)
    parser.add_argument(
        "--reg_trans_reference",
        type=str,
        choices=["zero", "warmstart"],
        default="warmstart",
        help="Reference for translation regularization.",
    )
    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=5.0)

    parser.add_argument("--human_mask_dilate_px", type=int, default=6)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if (
        len(args.iters) != 2
        or len(args.stage_lr) != 2
        or len(args.stage_early_stop_patience) != 2
        or len(args.stage_early_stop_min_delta) != 2
    ):
        raise ValueError(
            "--iters, --stage_lr, --stage_early_stop_patience, and "
            "--stage_early_stop_min_delta must provide exactly 2 values each."
        )

    script_dir = Path(__file__).resolve().parent
    object_video_dir = resolve_path(
        args.object_video_dir, script_dir
    ) or (script_dir.parent / "Generate_Object_Mesh" / "output" / args.video_name).resolve()
    depth_video_dir = resolve_path(
        args.depth_video_dir, script_dir
    ) or (script_dir.parent / "Estimate_Depth" / "output" / args.video_name).resolve()
    human_video_dir = resolve_path(
        args.human_video_dir, script_dir
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
    k_full = k_object_full if args.intrinsics_source == "object" else k_depth_full
    extrinsics = ensure_3x4_extrinsics(pose.get("extrinsics"))
    print(f"Using intrinsics source: {args.intrinsics_source}")

    mismatch_report = build_intrinsics_mismatch_report(
        k_object=k_object_full,
        k_depth=k_depth_full,
        warn_threshold_px=float(args.intrinsics_warn_threshold_px),
    )
    if mismatch_report["warning"]:
        print(
            "WARNING: object/depth intrinsics mismatch exceeds threshold; "
            f"max_abs_diff={mismatch_report['max_abs_diff']:.3f}px"
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
        assets.append(
            MeshAsset(
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
        )
        object_count += 1

    if object_count == 0:
        raise RuntimeError(f"No usable object meshes found in summary: {summary_path}")

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
    print(f"Loaded {len(assets)} meshes: {names}")

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
        cameras=cams, image_size=(opt_h, opt_w), bin_size=int(args.bin_size)
    )

    cv_to_p3d = torch.tensor(F_P3D_TO_CV, dtype=torch.float32, device=device)
    depth_valid_t = torch.from_numpy(
        np.isfinite(depth_opt) & (depth_opt > 0.0)
    ).to(device=device)
    xyz_map_t = torch.from_numpy(depth_to_xyz_map(depth_opt, k_opt)).to(
        device=device, dtype=torch.float32
    )

    verts_base_cv_t: list[torch.Tensor] = []
    verts_base_cv_np: list[np.ndarray] = []
    faces_t: list[torch.Tensor] = []
    masks_t: list[torch.Tensor | None] = []
    for asset in assets:
        verts_cv_np = asset.verts_source @ asset.source_to_cv.transpose(0, 1)
        verts_base_cv_np.append(verts_cv_np.astype(np.float32))
        verts_base_cv_t.append(
            torch.from_numpy(verts_cv_np).to(device=device, dtype=torch.float32)
        )
        faces_t.append(torch.from_numpy(asset.faces.astype(np.int64)).to(device=device))
        masks_t.append(
            None
            if asset.mask is None
            else torch.from_numpy(asset.mask).to(device=device, dtype=torch.float32)
        )

    if masks_t[0] is None:
        with torch.no_grad():
            pix_to_face_h, _, _ = rasterize_single_mesh(
                rasterizer=hard_rasterizer,
                verts_cv=verts_base_cv_t[0],
                faces_t=faces_t[0],
                cv_to_p3d=cv_to_p3d,
            )
            human_mask_gen = (pix_to_face_h >= 0).detach().cpu().numpy().astype(np.uint8)
            dpx = int(args.human_mask_dilate_px)
            if dpx > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (2 * dpx + 1, 2 * dpx + 1)
                )
                human_mask_gen = cv2.dilate(human_mask_gen, kernel, iterations=1)
            human_mask_f = human_mask_gen.astype(np.float32)
            masks_t[0] = torch.from_numpy(human_mask_f).to(device=device, dtype=torch.float32)
            assets[0].mask = human_mask_f
            assets[0].mask_path = None
        print("Human mask: generated from rendered human silhouette.")
    else:
        print("Human mask: loaded from file.")

    overlay_before = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=[v.copy() for v in verts_base_cv_np],
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_before.png"), overlay_before)

    states: list[MeshState] = []
    mesh_status: list[dict[str, Any]] = []
    per_mesh_setup: list[dict[str, Any]] = []

    min_log_scale = math.log(float(args.min_scale))
    max_log_scale = math.log(float(args.max_scale))

    for j, asset in enumerate(assets):
        mask_t = masks_t[j]
        if mask_t is None:
            states.append(
                MeshState(
                    log_s=torch.nn.Parameter(torch.zeros((), device=device)),
                    tvec=torch.nn.Parameter(torch.zeros((3,), device=device)),
                    tvec_init=torch.zeros((3,), device=device),
                    active=False,
                    status="skipped_missing_mask",
                    message="Mask is missing for this mesh.",
                )
            )
            mesh_status.append(
                {
                    "name": asset.name,
                    "slug": asset.slug,
                    "kind": asset.kind,
                    "status": "skipped_missing_mask",
                    "message": "Mask is missing for this mesh.",
                }
            )
            per_mesh_setup.append(
                {
                    "name": asset.name,
                    "mask_path": None if asset.mask_path is None else str(asset.mask_path),
                    "active_for_optimization": False,
                    "initial_correspondences": 0,
                }
            )
            continue

        log_s = torch.nn.Parameter(torch.zeros((), device=device, dtype=torch.float32))
        tvec = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
        tvec_init = torch.zeros((3,), device=device, dtype=torch.float32)

        with torch.no_grad():
            verts_init = transform_vertices_scale_t(verts_base_cv_t[j], log_s, tvec)
            pix_to_face, bary_coords, _ = rasterize_single_mesh(
                rasterizer=hard_rasterizer,
                verts_cv=verts_init,
                faces_t=faces_t[j],
                cv_to_p3d=cv_to_p3d,
            )
            mesh_pts_base, depth_pts, stats = gather_overlap_correspondences(
                pix_to_face=pix_to_face,
                bary_coords=bary_coords,
                mask_t=mask_t,
                depth_valid_t=depth_valid_t,
                xyz_map_t=xyz_map_t,
                verts_base_cv_t=verts_base_cv_t[j],
                faces_t=faces_t[j],
                max_points=int(args.corr_max_points),
            )

            if mesh_pts_base.shape[0] > 0:
                tvec[2] += torch.median(depth_pts[:, 2]) - torch.median(mesh_pts_base[:, 2])
                if mesh_pts_base.shape[0] >= int(args.min_corr_points):
                    pred_init = torch.exp(log_s) * mesh_pts_base + tvec
                    centroid_delta = depth_pts.mean(dim=0) - pred_init.mean(dim=0)
                    tvec += centroid_delta
            tvec_init.copy_(tvec)
            log_s.clamp_(min=min_log_scale, max=max_log_scale)

        active = bool(stats["used_correspondences"] >= int(args.min_corr_points))
        status = "pending" if active else "skipped_insufficient_correspondences"
        message = None
        if not active:
            message = (
                f"Initial overlap correspondences {stats['used_correspondences']} < "
                f"min_corr_points {int(args.min_corr_points)}."
            )

        states.append(
            MeshState(
                log_s=log_s,
                tvec=tvec,
                tvec_init=tvec_init,
                active=active,
                status=status,
                message=message,
            )
        )
        mesh_status.append(
            {
                "name": asset.name,
                "slug": asset.slug,
                "kind": asset.kind,
                "status": status,
                "message": message,
            }
        )
        per_mesh_setup.append(
            {
                "name": asset.name,
                "mask_path": None if asset.mask_path is None else str(asset.mask_path),
                "active_for_optimization": bool(active),
                "initial_correspondences": int(stats["used_correspondences"]),
                "initial_overlap_pixels": int(stats["overlap_pixels"]),
                "initial_rendered_pixels": int(stats["rendered_pixels"]),
            }
        )

    stages = [
        {"name": "scale_tz", "use_txy": False, "use_tz": True},
        {"name": "scale_txyz", "use_txy": True, "use_tz": True},
    ]

    for stage_i, n_iter in enumerate(args.iters):
        if int(n_iter) <= 0:
            raise ValueError(f"--iters[{stage_i}] must be > 0.")
    for stage_i, p in enumerate(args.stage_early_stop_patience):
        if int(p) < 0:
            raise ValueError(f"--stage_early_stop_patience[{stage_i}] must be >= 0.")
    for stage_i, md in enumerate(args.stage_early_stop_min_delta):
        if float(md) < 0.0:
            raise ValueError(f"--stage_early_stop_min_delta[{stage_i}] must be >= 0.")

    loss_history: list[dict[str, Any]] = []
    per_stage_mesh_logs: list[dict[str, Any]] = []
    scale_clamped_count = 0

    for stage_idx, stage_cfg in enumerate(stages):
        n_iter = int(args.iters[stage_idx])
        lr = float(args.stage_lr[stage_idx])
        patience = int(args.stage_early_stop_patience[stage_idx])
        min_delta = float(args.stage_early_stop_min_delta[stage_idx])
        print(
            f"[Stage {stage_idx}] {stage_cfg['name']} "
            f"iters={n_iter}, lr={lr}, patience={patience}, min_delta={min_delta}"
        )

        stage_mesh_entries: list[dict[str, Any]] = []
        for j, state in enumerate(states):
            mesh_name = assets[j].name
            mask_t = masks_t[j]
            if (not state.active) or (mask_t is None):
                stage_mesh_entries.append(
                    {
                        "name": mesh_name,
                        "status": state.status,
                        "message": state.message,
                        "iters_requested": n_iter,
                        "iters_ran": 0,
                        "best_total": None,
                        "final_total": None,
                        "final_corr_used": 0,
                    }
                )
                continue

            optimizer = torch.optim.Adam([state.log_s, state.tvec], lr=lr)
            best_total = float("inf")
            best_state = (
                state.log_s.detach().clone(),
                state.tvec.detach().clone(),
            )
            no_improve = 0
            early_stop_iter: int | None = None
            iters_ran = 0
            last_total: float | None = None
            last_corr_used = 0
            last_overlap_pixels = 0
            last_rendered_pixels = 0

            for it in range(n_iter):
                with torch.no_grad():
                    verts_now = transform_vertices_scale_t(
                        verts_base_cv_t[j], state.log_s, state.tvec
                    )
                    pix_to_face, bary_coords, _ = rasterize_single_mesh(
                        rasterizer=hard_rasterizer,
                        verts_cv=verts_now,
                        faces_t=faces_t[j],
                        cv_to_p3d=cv_to_p3d,
                    )
                    mesh_pts_base, depth_pts, corr_stats = gather_overlap_correspondences(
                        pix_to_face=pix_to_face,
                        bary_coords=bary_coords,
                        mask_t=mask_t,
                        depth_valid_t=depth_valid_t,
                        xyz_map_t=xyz_map_t,
                        verts_base_cv_t=verts_base_cv_t[j],
                        faces_t=faces_t[j],
                        max_points=int(args.corr_max_points),
                    )

                last_corr_used = int(corr_stats["used_correspondences"])
                last_overlap_pixels = int(corr_stats["overlap_pixels"])
                last_rendered_pixels = int(corr_stats["rendered_pixels"])
                if last_corr_used < int(args.min_corr_points):
                    if it == 0:
                        state.active = False
                        state.status = "skipped_insufficient_correspondences"
                        state.message = (
                            f"Stage {stage_idx}: correspondences {last_corr_used} < "
                            f"min_corr_points {int(args.min_corr_points)}."
                        )
                    else:
                        state.active = False
                        state.status = "optimized_low_overlap"
                        state.message = (
                            f"Stage {stage_idx}: correspondences dropped to "
                            f"{last_corr_used} < min_corr_points {int(args.min_corr_points)}."
                        )
                    break

                loss_data, corr_loss_stats = correspondence_loss(
                    mesh_points_base=mesh_pts_base,
                    depth_points=depth_pts,
                    log_s=state.log_s,
                    tvec=state.tvec,
                    trim_quantile=float(args.trim_quantile),
                    huber_delta_3d=float(args.huber_delta_3d),
                )
                trans_ref = state.tvec - state.tvec_init
                if args.reg_trans_reference == "zero":
                    trans_ref = state.tvec
                loss_reg = float(args.reg_scale) * (state.log_s.pow(2)) + float(
                    args.reg_trans
                ) * (trans_ref.pow(2).mean())
                total = loss_data + loss_reg

                if not torch.isfinite(total):
                    state.active = False
                    state.status = "failed_nonfinite_loss"
                    state.message = "Encountered non-finite loss."
                    break

                optimizer.zero_grad()
                total.backward()
                if state.tvec.grad is not None:
                    if not stage_cfg["use_txy"]:
                        state.tvec.grad[0:2].zero_()
                    if not stage_cfg["use_tz"]:
                        state.tvec.grad[2].zero_()
                optimizer.step()
                iters_ran = it + 1

                with torch.no_grad():
                    prev_log_s = state.log_s.detach().clone()
                    state.log_s.clamp_(min=min_log_scale, max=max_log_scale)
                    if float((state.log_s - prev_log_s).abs().item()) > 1e-10:
                        scale_clamped_count += 1

                total_val = float(total.item())
                last_total = total_val
                if total_val < (best_total - min_delta):
                    best_total = total_val
                    best_state = (
                        state.log_s.detach().clone(),
                        state.tvec.detach().clone(),
                    )
                    no_improve = 0
                else:
                    no_improve += 1

                if (it + 1) % int(args.log_every) == 0 or it == 0 or (it + 1) == n_iter:
                    print(
                        f"stage={stage_idx:02d} mesh={mesh_name} iter={it + 1:04d}/{n_iter:04d} "
                        f"loss={total_val:.6f} data={float(loss_data.item()):.6f} "
                        f"reg={float(loss_reg.item()):.6f} corr={last_corr_used} "
                        f"trim_kept={corr_loss_stats['corr_kept']}"
                    )
                    loss_history.append(
                        {
                            "stage": stage_idx,
                            "mesh": mesh_name,
                            "mesh_index": j,
                            "iter": it + 1,
                            "loss_total": total_val,
                            "loss_data": float(loss_data.item()),
                            "loss_reg": float(loss_reg.item()),
                            "corr_used": int(corr_loss_stats["corr_used"]),
                            "corr_kept": int(corr_loss_stats["corr_kept"]),
                            "mean_dist_m": corr_loss_stats["mean_dist_m"],
                            "median_dist_m": corr_loss_stats["median_dist_m"],
                            "overlap_pixels": int(last_overlap_pixels),
                            "rendered_pixels": int(last_rendered_pixels),
                        }
                    )

                if patience > 0 and no_improve >= patience:
                    early_stop_iter = it + 1
                    print(
                        f"stage={stage_idx:02d} mesh={mesh_name} early_stop "
                        f"iter={early_stop_iter:04d}/{n_iter:04d} best_loss={best_total:.6f}"
                    )
                    break

            with torch.no_grad():
                if math.isfinite(best_total):
                    state.log_s.copy_(best_state[0])
                    state.tvec.copy_(best_state[1])

            if not state.active and state.status.startswith("skipped"):
                pass
            elif not state.active:
                pass
            elif early_stop_iter is not None:
                state.status = "optimized_early_stopped"
                state.message = (
                    f"Early stopped at iter {int(early_stop_iter)} "
                    f"(patience={patience}, min_delta={min_delta})."
                )
            else:
                state.status = "optimized"
                state.message = None

            mesh_status[j] = {
                "name": assets[j].name,
                "slug": assets[j].slug,
                "kind": assets[j].kind,
                "status": state.status,
                "message": state.message,
            }

            stage_mesh_entries.append(
                {
                    "name": mesh_name,
                    "status": state.status,
                    "message": state.message,
                    "iters_requested": n_iter,
                    "iters_ran": int(iters_ran),
                    "best_total": None if not math.isfinite(best_total) else float(best_total),
                    "final_total": None if last_total is None else float(last_total),
                    "final_corr_used": int(last_corr_used),
                    "final_overlap_pixels": int(last_overlap_pixels),
                    "final_rendered_pixels": int(last_rendered_pixels),
                    "early_stop_iter": None if early_stop_iter is None else int(early_stop_iter),
                    "early_stop_patience": int(patience),
                    "early_stop_min_delta": float(min_delta),
                }
            )

        per_stage_mesh_logs.append(
            {
                "stage": stage_idx + 1,
                "stage_index": stage_idx,
                "stage_name": stage_cfg["name"],
                "iters": int(n_iter),
                "lr": float(lr),
                "early_stop_patience": int(patience),
                "early_stop_min_delta": float(min_delta),
                "mesh_logs": stage_mesh_entries,
            }
        )

    verts_final: list[np.ndarray] = []
    scales_np = np.zeros((len(assets),), dtype=np.float32)
    t_np = np.zeros((len(assets), 3), dtype=np.float32)
    for j, state in enumerate(states):
        with torch.no_grad():
            verts_t = transform_vertices_scale_t(verts_base_cv_t[j], state.log_s, state.tvec)
            verts_final.append(verts_t.detach().cpu().numpy().astype(np.float32))
            scales_np[j] = float(torch.exp(state.log_s).detach().cpu().item())
            t_np[j] = state.tvec.detach().cpu().numpy().astype(np.float32)

    meshes_out_dir = output_dir / "meshes"
    transforms_json_path = meshes_out_dir / "transforms.json"
    transforms_out = export_meshes_and_transforms(
        assets=assets,
        verts_aligned=verts_final,
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

    final_overlap_stats: list[dict[str, Any]] = []
    for j, asset in enumerate(assets):
        mask_t = masks_t[j]
        if mask_t is None:
            final_overlap_stats.append(
                {
                    "name": asset.name,
                    "overlap_pixels": 0,
                    "rendered_pixels": 0,
                    "used_correspondences": 0,
                    "mean_dist_m": None,
                    "median_dist_m": None,
                }
            )
            continue
        with torch.no_grad():
            verts_now = transform_vertices_scale_t(
                verts_base_cv_t[j], states[j].log_s, states[j].tvec
            )
            pix_to_face, bary_coords, _ = rasterize_single_mesh(
                rasterizer=hard_rasterizer,
                verts_cv=verts_now,
                faces_t=faces_t[j],
                cv_to_p3d=cv_to_p3d,
            )
            mesh_pts_base, depth_pts, corr_stats = gather_overlap_correspondences(
                pix_to_face=pix_to_face,
                bary_coords=bary_coords,
                mask_t=mask_t,
                depth_valid_t=depth_valid_t,
                xyz_map_t=xyz_map_t,
                verts_base_cv_t=verts_base_cv_t[j],
                faces_t=faces_t[j],
                max_points=int(args.corr_max_points),
            )
            _, corr_loss_stats = correspondence_loss(
                mesh_points_base=mesh_pts_base,
                depth_points=depth_pts,
                log_s=states[j].log_s,
                tvec=states[j].tvec,
                trim_quantile=float(args.trim_quantile),
                huber_delta_3d=float(args.huber_delta_3d),
            )
        final_overlap_stats.append(
            {
                "name": asset.name,
                "overlap_pixels": int(corr_stats["overlap_pixels"]),
                "rendered_pixels": int(corr_stats["rendered_pixels"]),
                "used_correspondences": int(corr_stats["used_correspondences"]),
                "mean_dist_m": corr_loss_stats["mean_dist_m"],
                "median_dist_m": corr_loss_stats["median_dist_m"],
            }
        )

    result = {
        "video_name": args.video_name,
        "coordinate_frame": "opencv_camera_frame0",
        "method": "overlap_pixel_correspondence_v1",
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
            "overlay_before": str(output_dir / "overlay_before.png"),
            "overlay_after": str(output_dir / "overlay_after.png"),
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
            "stage_early_stop_patience": [int(v) for v in args.stage_early_stop_patience],
            "stage_early_stop_min_delta": [
                float(v) for v in args.stage_early_stop_min_delta
            ],
            "stages": stages,
            "corr_max_points": int(args.corr_max_points),
            "min_corr_points": int(args.min_corr_points),
            "trim_quantile": float(args.trim_quantile),
            "huber_delta_3d": float(args.huber_delta_3d),
            "reg_scale": float(args.reg_scale),
            "reg_trans": float(args.reg_trans),
            "reg_trans_reference": str(args.reg_trans_reference),
            "min_scale": float(args.min_scale),
            "max_scale": float(args.max_scale),
            "scale_clamped_count": int(scale_clamped_count),
        },
        "per_mesh_setup": per_mesh_setup,
        "per_stage_mesh_logs": per_stage_mesh_logs,
        "mesh_status": mesh_status,
        "final_overlap_stats": final_overlap_stats,
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
