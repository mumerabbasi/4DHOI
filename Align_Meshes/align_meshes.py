"""Mesh-to-depth alignment with fixed correspondences.

Overview
--------
This script aligns human and object meshes to frame_00 depth in camera
coordinates using fixed point-to-point correspondences.

The pipeline is intentionally simple:
1. Load frame_00 RGB, metric depth, intrinsics, human/object meshes, and masks.
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
    - Objects: `mesh_posed.ply` from Generate_Object_Mesh output and per-object
      mask from Segment_First_Frame `frame_00_segmentation_summary.json`.
    - Human: first PLY in `output_plys` and human SAM3 mask from the same summary.
  - Coordinate conversion:
    - Object PLY vertices are treated as PyTorch3D camera coordinates and mapped
      to OpenCV camera coordinates using `F_P3D_TO_CV`.
    - Human vertices are assumed OpenCV camera coordinates (identity transform).
    - No Y-up to Z-up rotation is applied for objects.

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
  - `meshes/<slug>.ply`: aligned meshes (coordinate frame selected by
    `--output_coord`, default `opencv`).
  - `meshes/transforms.json`: source-to-aligned transform metadata.
  - `loss_curves/<slug>_loss_*.png`: separate loss plots for each term and total.
  - `loss_curves/<slug>_loss.csv`: numeric loss history.
  - `correspondences/<slug>/iter_00000_depth_fixed.ply`:
    fixed depth correspondence cloud with colors, saved once.
  - `correspondences/<slug>/iter_XXXXX_mesh_transformed.ply`:
    transformed full mesh cloud (grey) + transformed correspondence points
    (same correspondence colors as depth points).
  - `correspondences/<slug>/iter_XXXXX_correspondence_projection.png`:
    side-by-side 2D projection visualization of fixed depth points and transformed
    mesh points at that iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes

from utils_align_meshes import (
    F_CV_TO_P3D,
    F_P3D_TO_CV,
    CorrespondenceSet,
    MeshAsset,
    OptimizationResult,
    append_history,
    build_cameras,
    colorize_points_by_xyz,
    draw_overlay_points,
    ensure_3x3_intrinsics,
    find_first_human_ply,
    load_binary_mask,
    load_json,
    load_mesh,
    load_object_intrinsics,
    maybe_resize_for_optimization,
    new_history_dict,
    parse_device,
    plot_loss_curves_separate,
    resolve_path,
    save_correspondence_snapshot,
    save_loss_history_csv,
    save_mesh_ply,
    slugify,
)


def _resolve_summary_relative_path(path_like: str | None, base_dir: Path) -> Path | None:
    """Resolve potentially-relative summary path against base_dir."""
    if path_like is None:
        return None
    p = Path(str(path_like))
    if not p.is_absolute():
        p = base_dir / p
    return p.resolve()


def _object_dir_name_from_summary(obj: dict[str, Any]) -> str:
    """Resolve object directory slug used for mesh output subfolders."""
    out_dir_raw = obj.get("output_dir")
    if out_dir_raw:
        return Path(str(out_dir_raw)).name
    return str(obj.get("object", "object")).replace(" ", "_")


def _erode_mask(mask: np.ndarray | None, erode_iters: int) -> np.ndarray | None:
    """Erode binary mask with a 3x3 kernel for tighter correspondence filtering."""
    if mask is None:
        return None
    if erode_iters <= 0:
        return (mask > 0.5).astype(np.float32)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = ((mask > 0.5).astype(np.uint8) * 255)
    eroded = cv2.erode(mask_u8, kernel, iterations=int(erode_iters))
    return (eroded > 127).astype(np.float32)


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
        # cull_backfaces=True,
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

    ys_all, xs_all = np.nonzero(valid)
    pixels_considered = int(ys_all.shape[0])
    if pixels_considered == 0:
        return CorrespondenceSet(
            mesh_points_base=np.zeros((0, 3), dtype=np.float32),
            depth_points=np.zeros((0, 3), dtype=np.float32),
            uv_ref=np.zeros((0, 2), dtype=np.float32),
            colors_rgb=np.zeros((0, 3), dtype=np.uint8),
            pixels_considered=0,
            pixels_used=0,
        )

    keep_idx = np.arange(pixels_considered, dtype=np.int64)
    if max_points > 0 and pixels_considered > max_points:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(pixels_considered, size=max_points, replace=False)

    ys = ys_all[keep_idx]
    xs = xs_all[keep_idx]

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
    # PyTorch3D rasterization uses pixel centers; uv_ref is stored as integer pixel
    # indices. Subtract 0.5 so projected UVs are in the same index convention.
    u = intrinsics[0, 0] * transformed[:, 0] / z + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * transformed[:, 1] / z + intrinsics[1, 2] - 0.5
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


def optimize_scale_tz(
    corr: CorrespondenceSet,
    full_mesh_points_base: np.ndarray,
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
        init_full_mesh_points = (float(init_losses["scale"].detach().cpu().item()) * full_mesh_points_base).astype(
            np.float32
        )
        init_full_mesh_points[:, 2] += float(init_losses["tz"].detach().cpu().item())
        save_correspondence_snapshot(
            out_dir=corr_vis_dir,
            iter_idx=0,
            depth_points=corr.depth_points,
            transformed_mesh_points=init_losses["transformed"].detach().cpu().numpy(),
            full_transformed_mesh_points=init_full_mesh_points,
            uv_ref=corr.uv_ref,
            colors_rgb=corr.colors_rgb,
            intrinsics=intrinsics,
            frame_bgr=corr_vis_frame,
            point_radius=corr_vis_point_radius,
            save_depth_fixed=True,
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
                eval_full_mesh_points = (
                    float(eval_losses["scale"].detach().cpu().item()) * full_mesh_points_base
                ).astype(np.float32)
                eval_full_mesh_points[:, 2] += float(
                    eval_losses["tz"].detach().cpu().item()
                )
                save_correspondence_snapshot(
                    out_dir=corr_vis_dir,
                    iter_idx=iter_idx,
                    depth_points=corr.depth_points,
                    transformed_mesh_points=eval_losses["transformed"]
                    .detach()
                    .cpu()
                    .numpy(),
                    full_transformed_mesh_points=eval_full_mesh_points,
                    uv_ref=corr.uv_ref,
                    colors_rgb=corr.colors_rgb,
                    intrinsics=intrinsics,
                    frame_bgr=corr_vis_frame,
                    point_radius=corr_vis_point_radius,
                    save_depth_fixed=False,
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
        help="Directory like ../Generate_Object_Mesh/output/video_xx (mesh + intrinsics).",
    )
    parser.add_argument(
        "--segmentation_video_dir",
        type=str,
        default=None,
        help="Directory like ../Segment_First_Frame/output/video_xx (summary + masks).",
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

    parser.add_argument("--iters", type=int, default=1500)
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
    parser.add_argument(
        "--sam3_mask_erode_iters",
        type=int,
        default=3,
        help=(
            "Number of 3x3 erosion iterations applied to SAM3 masks "
            "(objects + human) before correspondence filtering."
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
    if args.max_correspondences_per_mesh <= 0:
        raise ValueError("--max_correspondences_per_mesh must be > 0.")
    if args.sam3_mask_erode_iters < 0:
        raise ValueError("--sam3_mask_erode_iters must be >= 0.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    object_video_dir = resolve_path(args.object_video_dir, script_dir) or (
        script_dir.parent / "Generate_Object_Mesh" / "output" / args.video_name
    ).resolve()
    segmentation_video_dir = resolve_path(args.segmentation_video_dir, script_dir) or (
        script_dir.parent / "Segment_First_Frame" / "output" / args.video_name
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

    summary_path = segmentation_video_dir / "frame_00_segmentation_summary.json"
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
        object_dir_name = _object_dir_name_from_summary(obj)
        obj_dir = (object_video_dir / object_dir_name).resolve()

        mesh_path = obj_dir / "mesh_posed.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Missing mesh_posed.ply for object '{obj_name}': {mesh_path}"
            )
        mask_rel = obj.get("mask_file")
        if mask_rel is None:
            raise KeyError(
                f"Missing 'mask_file' for object '{obj_name}' in {summary_path}"
            )
        seg_obj_dir = _resolve_summary_relative_path(
            obj.get("output_dir"),
            segmentation_video_dir,
        ) or (segmentation_video_dir / object_dir_name).resolve()
        mask_path = _resolve_summary_relative_path(str(mask_rel), seg_obj_dir)
        assert mask_path is not None
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing mask for object '{obj_name}': {mask_path}"
            )

        verts_src, faces = load_mesh(mesh_path)
        mask = load_binary_mask(mask_path, (depth_h, depth_w))
        mask = _erode_mask(mask, int(args.sam3_mask_erode_iters))
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
        raise ValueError(
            f"Human mesh must be a .ply file, got: {human_mesh_path}"
        )

    human_summary = summary.get("human")
    if not isinstance(human_summary, dict):
        raise KeyError(
            f"Missing top-level 'human' entry in segmentation summary: {summary_path}"
        )
    if not human_summary.get("success", False):
        raise RuntimeError(
            f"Human segmentation failed according to summary: {summary_path}"
        )
    human_mask_rel = human_summary.get("mask_file")
    if human_mask_rel is None:
        raise KeyError(f"Missing human.mask_file in segmentation summary: {summary_path}")
    human_seg_dir = _resolve_summary_relative_path(
        human_summary.get("output_dir"),
        segmentation_video_dir,
    ) or (segmentation_video_dir / "human").resolve()
    human_mask_path = _resolve_summary_relative_path(str(human_mask_rel), human_seg_dir)
    if human_mask_path is None or not human_mask_path.exists():
        raise FileNotFoundError(f"Human mask not found: {human_mask_path}")
    human_mask = load_binary_mask(human_mask_path, (depth_h, depth_w))
    human_mask = _erode_mask(human_mask, int(args.sam3_mask_erode_iters))

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
    for asset, corr, verts_cv in zip(assets, correspondences, verts_base_cv_np):
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
            full_mesh_points_base=verts_cv,
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
    cv_to_output = (
        np.eye(3, dtype=np.float32)
        if args.output_coord == "opencv"
        else F_CV_TO_P3D.copy()
    )
    for asset, verts_cv, result in zip(assets, verts_base_cv_np, optimization_results):
        scale = float(result.scale)
        tz = float(result.tz)
        verts_aligned_cv = (scale * verts_cv).astype(np.float32)
        verts_aligned_cv[:, 2] += tz
        verts_after_np.append(verts_aligned_cv)

        verts_aligned_out = (
            verts_aligned_cv @ cv_to_output.transpose(0, 1)
        ).astype(np.float32)
        out_mesh_path = meshes_out_dir / f"{asset.slug}.ply"
        save_mesh_ply(out_mesh_path, verts_aligned_out, asset.faces)

        source_to_aligned_cv_4x4 = np.eye(4, dtype=np.float32)
        source_to_aligned_cv_4x4[:3, :3] = (
            scale * asset.source_to_cv.astype(np.float32)
        )
        source_to_aligned_cv_4x4[:3, 3] = np.array([0.0, 0.0, tz], dtype=np.float32)

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
                    "tz_init_m": float(result.tz_init),
                    "delta_tz_m": float(result.delta_tz),
                    "tz_total_m": float(tz),
                    "num_correspondences": int(result.correspondences),
                },
                "source_to_aligned_cv_matrix_4x4": source_to_aligned_cv_4x4.tolist(),
            }
        )

    overlay_after = draw_overlay_points(
        frame_bgr=frame_bgr,
        verts_cv_list=verts_after_np,
        names=names,
        k=k_full,
    )
    cv2.imwrite(str(output_dir / "overlay_after.png"), overlay_after)

    summary_out = {
        "inputs": {
            "video_name": args.video_name,
            "object_video_dir": str(object_video_dir),
            "segmentation_video_dir": str(segmentation_video_dir),
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
            "output_coordinate": args.output_coord,
            "sam3_mask_erode_iters": int(args.sam3_mask_erode_iters),
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
            "mesh_format": "ply",
            "output_coordinate": args.output_coord,
            "overlay_before": str(output_dir / "overlay_before.png"),
            "overlay_after": str(output_dir / "overlay_after.png"),
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
