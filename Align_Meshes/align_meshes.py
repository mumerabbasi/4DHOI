"""Align human and object meshes in a shared camera frame using depth + silhouettes.

This script takes posed object meshes and a human mesh for a video, then optimizes
per-mesh similarity transforms (scale, rotation, translation) so all meshes align
to the first-frame camera in OpenCV coordinates.

High-level pipeline:
1. Load object meshes/masks from Generate_Object_Mesh outputs and human mesh/mask.
2. Load camera intrinsics and observed depth for frame_00.
3. Run 3-stage optimization with depth residual + silhouette + regularization losses.
4. Export aligned OBJ meshes (in OpenCV coordinates), transforms, overlays, and a JSON result summary.
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
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
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


def build_cameras(k: np.ndarray, width: int, height: int, device: torch.device) -> PerspectiveCameras:
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


def build_rasterizers_and_silhouette_renderer(
    cameras: PerspectiveCameras,
    image_size: tuple[int, int],
    bin_size: int,
    sil_sigma: float,
    sil_gamma: float,
    sil_faces_per_pixel: int,
) -> tuple[MeshRasterizer, MeshRenderer]:
    hard_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),
        max_faces_per_bin=300000,
    )
    hard_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=hard_settings)

    blur_radius = math.log(1.0 / 1e-4 - 1.0) * float(sil_sigma)
    sil_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=blur_radius,
        faces_per_pixel=int(sil_faces_per_pixel),
        bin_size=int(bin_size),
        max_faces_per_bin=300000,
    )
    blend_params = BlendParams(sigma=float(sil_sigma), gamma=float(sil_gamma))
    sil_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=sil_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )
    return hard_rasterizer, sil_renderer


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


def render_soft_silhouettes(
    sil_renderer: MeshRenderer,
    verts_cv: list[torch.Tensor],
    faces: list[torch.Tensor],
    cv_to_p3d: torch.Tensor,
) -> torch.Tensor:
    verts_p3d = [v @ cv_to_p3d.transpose(0, 1) for v in verts_cv]
    meshes = Meshes(verts=verts_p3d, faces=faces)
    rgba = sil_renderer(meshes)
    return rgba[..., 3]  # (J,H,W)


def huber_loss(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_r = residual.abs()
    return torch.where(
        abs_r <= delta,
        0.5 * residual * residual,
        delta * (abs_r - 0.5 * delta),
    )


def balanced_bce_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    p = pred.clamp(min=eps, max=1.0 - eps)
    gt = target > 0.5
    losses = []
    if bool(gt.any()):
        losses.append(-torch.log(p[gt]).mean())
    neg = ~gt
    if bool(neg.any()):
        losses.append(-torch.log(1.0 - p[neg]).mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align human and object meshes into a single OpenCV camera frame "
            "using observed depth + silhouettes (first frame)."
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
            "Camera intrinsics source for rendering/alignment. "
            "'object' uses Generate_Object_Mesh/output/video_xx/camera_intrinsics.json "
            "(intrinsics_pixels_3x3). "
            "'depth' uses Estimate_Depth/output/video_xx/pose_estimation.json intrinsics."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--opt_max_side", type=int, default=640)
    parser.add_argument("--bin_size", type=int, default=0)

    parser.add_argument("--iters", type=int, nargs=3, default=[100, 120, 160])
    parser.add_argument("--stage_lr", type=float, nargs=3, default=[5e-3, 2e-3, 1e-3])
    parser.add_argument("--stage_sil_weight", type=float, nargs=3, default=[0.05, 0.2, 0.5])

    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--human_sil_weight", type=float, default=0.25)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--reg_scale", type=float, default=0.2)
    parser.add_argument("--reg_rot", type=float, default=0.01)
    parser.add_argument("--reg_trans", type=float, default=0.1)
    parser.add_argument("--depth_huber_delta", type=float, default=0.1)

    parser.add_argument("--sil_sigma", type=float, default=1e-4)
    parser.add_argument("--sil_gamma", type=float, default=1e-4)
    parser.add_argument("--sil_faces_per_pixel", type=int, default=50)
    parser.add_argument("--human_mask_dilate_px", type=int, default=6)
    parser.add_argument(
        "--use_mesh_visibility_for_depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use visible mesh_id from full-scene rasterization for per-mesh depth residuals.",
    )

    parser.add_argument("--min_scale", type=float, default=0.2)
    parser.add_argument("--max_scale", type=float, default=5.0)
    parser.add_argument("--max_rot_deg", type=float, default=5.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_stage_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save per-stage outputs (OBJ meshes + combined transforms.json) "
            "in meshes_stage_1, meshes_stage_2, ..."
        ),
    )
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
    if args.intrinsics_source == "object":
        k_full = load_object_intrinsics(object_intrinsics_json_path)
    else:
        k_full = ensure_3x3_intrinsics(pose.get("intrinsics"))
    extrinsics = ensure_3x4_extrinsics(pose.get("extrinsics"))
    print(f"Using intrinsics source: {args.intrinsics_source}")

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
        source_coord="opencv_camera" if args.human_coord == "opencv" else "pytorch3d_camera",
        verts_source=human_verts_src,
        faces=human_faces,
        source_to_cv=human_source_to_cv.astype(np.float32),
        mask_path=human_mask_path,
        mask=human_mask,
    )
    assets = [human_asset] + assets

    names = [a.name for a in assets]
    kinds = [a.kind for a in assets]
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
    hard_rasterizer, sil_renderer = build_rasterizers_and_silhouette_renderer(
        cameras=cams,
        image_size=(opt_h, opt_w),
        bin_size=int(args.bin_size),
        sil_sigma=float(args.sil_sigma),
        sil_gamma=float(args.sil_gamma),
        sil_faces_per_pixel=int(args.sil_faces_per_pixel),
    )

    cv_to_p3d = torch.tensor(F_P3D_TO_CV, dtype=torch.float32, device=device)
    depth_obs_t = torch.from_numpy(depth_opt).to(device=device, dtype=torch.float32)

    verts_base_cv: list[torch.Tensor] = []
    faces_t: list[torch.Tensor] = []
    masks_t: list[torch.Tensor | None] = []
    for asset in assets:
        verts_cv_np = asset.verts_source @ asset.source_to_cv.transpose(0, 1)
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

    log_s = torch.nn.Parameter(torch.zeros((j_count,), device=device, dtype=torch.float32))
    rotvec = torch.nn.Parameter(torch.zeros((j_count, 3), device=device, dtype=torch.float32))
    tvec = torch.nn.Parameter(torch.zeros((j_count, 3), device=device, dtype=torch.float32))

    # Warm-start z translation from masked depth median offset.
    with torch.no_grad():
        verts_init = transform_vertices_list(verts_base_cv, log_s, rotvec, tvec)
        depth_init, mesh_id_init = render_scene_depth_and_mesh_id(
            hard_rasterizer=hard_rasterizer,
            verts_cv=verts_init,
            faces=faces_t,
            cv_to_p3d=cv_to_p3d,
            device=device,
        )
        for j in range(j_count):
            mask = masks_t[j]
            if mask is None:
                continue
            pix = (depth_obs_t > 0.0) & (mask > 0.5)
            if bool(args.use_mesh_visibility_for_depth):
                pix = pix & (mesh_id_init == j)
            if int(pix.sum().item()) < 64:
                pix = (depth_obs_t > 0.0) & (mask > 0.5)
            if int(pix.sum().item()) < 64:
                continue
            z_obs = torch.median(depth_obs_t[pix])
            z_rnd = torch.median(depth_init[pix])
            if torch.isfinite(z_obs) and torch.isfinite(z_rnd):
                tvec[j, 2] += (z_obs - z_rnd)
    print("Warm-started tz from depth medians.")

    stages = [
        {"use_scale": True, "use_rot": False, "use_txy": False, "use_tz": True},
        {"use_scale": True, "use_rot": False, "use_txy": True, "use_tz": True},
        {"use_scale": True, "use_rot": True, "use_txy": True, "use_tz": True},
    ]
    if len(args.iters) != 3 or len(args.stage_lr) != 3 or len(args.stage_sil_weight) != 3:
        raise ValueError("--iters, --stage_lr, --stage_sil_weight must each provide exactly 3 values.")

    loss_history: list[dict[str, Any]] = []
    stage_output_records: list[dict[str, Any]] = []
    max_rot_rad = math.radians(float(args.max_rot_deg))
    min_log_scale = math.log(float(args.min_scale))
    max_log_scale = math.log(float(args.max_scale))

    for stage_idx, stage_cfg in enumerate(stages):
        n_iter = int(args.iters[stage_idx])
        if n_iter <= 0:
            continue

        lr = float(args.stage_lr[stage_idx])
        sil_weight_stage = float(args.stage_sil_weight[stage_idx])
        optimizer = torch.optim.Adam([log_s, rotvec, tvec], lr=lr)

        print(
            f"[Stage {stage_idx}] iters={n_iter}, lr={lr}, "
            f"sil_w={sil_weight_stage}, cfg={stage_cfg}"
        )

        for it in range(n_iter):
            verts_cur = transform_vertices_list(verts_base_cv, log_s, rotvec, tvec)
            depth_rend, mesh_id = render_scene_depth_and_mesh_id(
                hard_rasterizer=hard_rasterizer,
                verts_cv=verts_cur,
                faces=faces_t,
                cv_to_p3d=cv_to_p3d,
                device=device,
            )
            sil_rend = render_soft_silhouettes(
                sil_renderer=sil_renderer,
                verts_cv=verts_cur,
                faces=faces_t,
                cv_to_p3d=cv_to_p3d,
            )

            l_depth = depth_obs_t.new_zeros(())
            depth_terms = 0
            valid_depth = depth_obs_t > 0.0
            for j in range(j_count):
                mask = masks_t[j]
                if mask is None:
                    continue
                pix = valid_depth & (mask > 0.5)
                if bool(args.use_mesh_visibility_for_depth):
                    pix = pix & (mesh_id == j)
                if int(pix.sum().item()) < 32 and bool(args.use_mesh_visibility_for_depth):
                    # Fallback to weaker supervision if occlusion/visibility labeling is empty.
                    pix = valid_depth & (mask > 0.5)
                if int(pix.sum().item()) == 0:
                    continue
                r = depth_rend[pix] - depth_obs_t[pix]
                l_depth = l_depth + huber_loss(r, float(args.depth_huber_delta)).mean()
                depth_terms += 1
            if depth_terms > 0:
                l_depth = l_depth / float(depth_terms)

            l_sil = depth_obs_t.new_zeros(())
            sil_weight_sum = 0.0
            for j in range(j_count):
                mask = masks_t[j]
                if mask is None:
                    continue
                mesh_w = 1.0 if kinds[j] == "object" else float(args.human_sil_weight)
                if mesh_w <= 0.0:
                    continue
                l_sil = l_sil + mesh_w * balanced_bce_loss(sil_rend[j], mask)
                sil_weight_sum += mesh_w
            if sil_weight_sum > 0.0:
                l_sil = l_sil / sil_weight_sum

            l_reg = (
                float(args.reg_scale) * (log_s.pow(2).mean())
                + float(args.reg_rot) * (rotvec.pow(2).mean())
                + float(args.reg_trans) * (tvec.pow(2).mean())
            )
            total = (
                float(args.depth_weight) * l_depth
                + sil_weight_stage * l_sil
                + float(args.reg_weight) * l_reg
            )

            optimizer.zero_grad()
            total.backward()

            if not stage_cfg["use_scale"] and log_s.grad is not None:
                log_s.grad.zero_()
            if not stage_cfg["use_rot"] and rotvec.grad is not None:
                rotvec.grad.zero_()
            if tvec.grad is not None:
                if not stage_cfg["use_txy"]:
                    tvec.grad[:, 0:2].zero_()
                if not stage_cfg["use_tz"]:
                    tvec.grad[:, 2].zero_()

            optimizer.step()

            with torch.no_grad():
                log_s.clamp_(min=min_log_scale, max=max_log_scale)
                norms = torch.linalg.norm(rotvec, dim=1, keepdim=True).clamp_min(1e-8)
                factor = torch.clamp(max_rot_rad / norms, max=1.0)
                rotvec.mul_(factor)

            if (it + 1) % int(args.log_every) == 0 or it == 0 or (it + 1) == n_iter:
                msg = (
                    f"stage={stage_idx:02d} iter={it + 1:04d}/{n_iter:04d} "
                    f"loss={float(total.item()):.6f} "
                    f"depth={float(l_depth.item()):.6f} "
                    f"sil={float(l_sil.item()):.6f} "
                    f"reg={float(l_reg.item()):.6f}"
                )
                print(msg)
                loss_history.append(
                    {
                        "stage": stage_idx,
                        "iter": it + 1,
                        "loss_total": float(total.item()),
                        "loss_depth": float(l_depth.item()),
                        "loss_silhouette": float(l_sil.item()),
                        "loss_regularization": float(l_reg.item()),
                    }
                )

        if bool(args.save_stage_outputs):
            stage_num = stage_idx + 1
            stage_meshes_dir = output_dir / f"meshes_stage_{stage_num}"
            stage_transforms_path = stage_meshes_dir / "transforms.json"
            with torch.no_grad():
                verts_stage_t = transform_vertices_list(verts_base_cv, log_s, rotvec, tvec)
                verts_stage = [v.detach().cpu().numpy() for v in verts_stage_t]
                rotvec_np_stage = rotvec.detach().cpu().numpy()
                rots_np_stage = axis_angle_to_matrix(rotvec.detach()).cpu().numpy()
                scales_np_stage = np.exp(log_s.detach().cpu().numpy())
                t_np_stage = tvec.detach().cpu().numpy()

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
            stage_output_records.append(
                {
                    "stage": stage_num,
                    "meshes_dir": str(stage_meshes_dir),
                    "transforms_json": str(stage_transforms_path),
                }
            )
            print(f"Saved stage {stage_num} outputs to: {stage_meshes_dir}")

    with torch.no_grad():
        verts_final_t = transform_vertices_list(verts_base_cv, log_s, rotvec, tvec)
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

    meshes_out_dir = output_dir / "meshes"
    transforms_json_path = meshes_out_dir / "transforms.json"
    rotvec_np = rotvec.detach().cpu().numpy()
    rots_np = axis_angle_to_matrix(rotvec.detach()).cpu().numpy()
    scales_np = np.exp(log_s.detach().cpu().numpy())
    t_np = tvec.detach().cpu().numpy()
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
            "extrinsics_3x4_from_depth": None if extrinsics is None else extrinsics.tolist(),
            "depth_is_metric": bool(int(pose.get("is_metric", 0))),
            "depth_scale_factor": pose.get("scale_factor", None),
        },
        "optimization": {
            "device": str(device),
            "resolution_input_hw": [int(depth_h), int(depth_w)],
            "resolution_optimized_hw": [int(opt_h), int(opt_w)],
            "iters": [int(x) for x in args.iters],
            "stage_lr": [float(x) for x in args.stage_lr],
            "stage_sil_weight": [float(x) for x in args.stage_sil_weight],
            "depth_weight": float(args.depth_weight),
            "human_sil_weight": float(args.human_sil_weight),
            "reg_weight": float(args.reg_weight),
            "reg_scale": float(args.reg_scale),
            "reg_rot": float(args.reg_rot),
            "reg_trans": float(args.reg_trans),
            "depth_huber_delta": float(args.depth_huber_delta),
            "use_mesh_visibility_for_depth": bool(args.use_mesh_visibility_for_depth),
        },
        "depth_residual_stats_before": before_stats,
        "depth_residual_stats_after": after_stats,
        "stage_outputs": stage_output_records,
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
