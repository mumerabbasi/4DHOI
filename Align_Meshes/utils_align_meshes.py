from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# OpenCV (+X right, +Y down, +Z forward) <-> PyTorch3D (+X left, +Y up, +Z forward)
F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
F_CV_TO_P3D = F_P3D_TO_CV.copy()

OVERLAY_PALETTE_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 140, 0),
    (0, 255, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (0, 0, 255),
    (255, 0, 0),
]

DEFAULT_OVERLAY_FILL_ALPHA = 0.60
DEFAULT_OVERLAY_CONTOUR_THICKNESS = 0


@dataclass
class MeshAsset:
    name: str
    slug: str
    kind: str
    source_mesh_path: Path
    source_coord: str
    verts_source: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
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
    return text.strip().replace(" ", "_").replace("-", "_")


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


def _normalize_vertex_colors_rgba(colors: np.ndarray) -> np.ndarray | None:
    """Normalize vertex colors to uint8 RGBA."""
    arr = np.asarray(colors)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    if arr.shape[1] == 3:
        alpha = np.full((arr.shape[0], 1), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=1)
    else:
        arr = arr[:, :4]

    if np.issubdtype(arr.dtype, np.floating):
        # Trimesh colors are typically [0,255], but support normalized [0,1].
        max_val = float(np.nanmax(arr)) if arr.size else 0.0
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.round(arr), 0.0, 255.0).astype(np.uint8)
    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mesh = trimesh.load(str(path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_colors: np.ndarray | None = None
    visual = getattr(mesh, "visual", None)
    if visual is not None:
        raw_colors = getattr(visual, "vertex_colors", None)
        if raw_colors is not None and len(raw_colors) == verts.shape[0]:
            vertex_colors = _normalize_vertex_colors_rgba(np.asarray(raw_colors))
    return verts, faces, vertex_colors


def save_mesh_ply(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None = None,
) -> None:
    visual = None
    if vertex_colors is not None:
        colors_rgba = _normalize_vertex_colors_rgba(vertex_colors)
        if colors_rgba is not None and colors_rgba.shape[0] == verts.shape[0]:
            visual = trimesh.visual.ColorVisuals(vertex_colors=colors_rgba)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, visual=visual)
    mesh.export(str(path))


def load_binary_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    target_h, target_w = target_hw
    if mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.float32)


def resolve_frame_0000_mask(mask_dir: Path) -> Path:
    """Resolve frame_0000 mask path in a SAM3 video mask directory."""
    frame_0000 = (mask_dir / "frame_0000.png").resolve()
    if frame_0000.exists():
        return frame_0000
    first_mask = next(iter(sorted(mask_dir.glob("*.png"))), None)
    if first_mask is None:
        raise FileNotFoundError(f"No PNG masks found in: {mask_dir}")
    return first_mask.resolve()


def erode_mask(mask: np.ndarray | None, erode_iters: int) -> np.ndarray | None:
    """Erode binary mask with a 3x3 kernel for tighter correspondence filtering."""
    if mask is None:
        return None
    if erode_iters <= 0:
        return (mask > 0.5).astype(np.float32)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = ((mask > 0.5).astype(np.uint8) * 255)
    eroded = cv2.erode(mask_u8, kernel, iterations=int(erode_iters))
    return (eroded > 127).astype(np.float32)


def find_first_human_ply(output_plys_dir: Path) -> Path:
    ply_paths = sorted(output_plys_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files found in {output_plys_dir}")
    return ply_paths[0]


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


def project_points_cv(
    points_cv: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = points_cv[:, 2]
    valid = np.isfinite(z) & (z > 1e-6)
    uv = np.zeros((points_cv.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        pts = points_cv[valid]
        z_valid = pts[:, 2]
        uv_valid = np.empty((pts.shape[0], 2), dtype=np.float32)
        # Keep projection in pixel-index coordinates (not pixel-center coordinates)
        # to match rasterized pixel indices used by correspondences.
        uv_valid[:, 0] = (pts[:, 0] * k[0, 0]) / z_valid + k[0, 2] - 0.5
        uv_valid[:, 1] = (pts[:, 1] * k[1, 1]) / z_valid + k[1, 2] - 0.5
        uv[valid] = uv_valid
    return uv, valid


def build_overlay_color_map(names: list[str]) -> dict[str, tuple[int, int, int]]:
    color_map: dict[str, tuple[int, int, int]] = {}
    for idx, name in enumerate(names):
        color_map[name] = OVERLAY_PALETTE_BGR[idx % len(OVERLAY_PALETTE_BGR)]
    return color_map


def add_overlay_legend(
    image_bgr: np.ndarray,
    legend_items: Sequence[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    """Draw a compact overlay legend using the local overlay palette."""
    deduped_items: list[tuple[str, tuple[int, int, int]]] = []
    seen_names: set[str] = set()
    for name, color in legend_items:
        label = str(name).strip()
        if not label or label in seen_names:
            continue
        seen_names.add(label)
        deduped_items.append((label, color))

    if not deduped_items:
        return image_bgr.copy()

    result = image_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_line_type = cv2.LINE_AA
    line_height = 24
    x_start = 10
    y_start = 24

    max_text_w = max(
        cv2.getTextSize(name, font, font_scale, thickness)[0][0]
        for name, _ in deduped_items
    )
    legend_w = 20 + max_text_w + 10
    overlay_bg = result.copy()
    cv2.rectangle(
        overlay_bg,
        (x_start - 5, y_start - line_height + 2),
        (x_start + legend_w, y_start + (len(deduped_items) - 1) * line_height + 10),
        (0, 0, 0),
        -1,
    )
    result = cv2.addWeighted(overlay_bg, 0.5, result, 0.5, 0)

    for idx, (name, color) in enumerate(deduped_items):
        y = y_start + idx * line_height
        cv2.rectangle(result, (x_start, y - 10), (x_start + 14, y + 4), color, -1)
        cv2.putText(
            result,
            name,
            (x_start + 20, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            text_line_type,
        )

    return result


def draw_mask_outline_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    fill_alpha: float = DEFAULT_OVERLAY_FILL_ALPHA,
    contour_thickness: int = DEFAULT_OVERLAY_CONTOUR_THICKNESS,
) -> np.ndarray:
    if mask is None:
        return image_bgr.copy()

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return image_bgr.copy()

    out = image_bgr.astype(np.float32).copy()
    color_arr = np.array(color_bgr, dtype=np.float32)
    alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours_info = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if int(contour_thickness) > 0:
        outline = tuple(int(np.clip(c + 48, 0, 255)) for c in color_bgr)
        cv2.drawContours(
            out_u8,
            contours,
            contourIdx=-1,
            color=outline,
            thickness=int(contour_thickness),
            lineType=cv2.LINE_AA,
        )
    return out_u8


def rasterize_silhouette_mask_cv_mesh(
    verts_cv: np.ndarray,
    faces: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> np.ndarray:
    cameras = build_cameras(k, width, height, device)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=300000,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    verts_t = torch.from_numpy(verts_cv.astype(np.float32)).to(
        device=device,
        dtype=torch.float32,
    )
    cv_to_p3d = torch.from_numpy(F_CV_TO_P3D).to(device=device, dtype=torch.float32)
    verts_p3d = verts_t @ cv_to_p3d.transpose(0, 1)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])

    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]
    zbuf = fragments.zbuf[0, ..., 0]
    valid = (pix_to_face >= 0) & torch.isfinite(zbuf) & (zbuf > 1e-6)
    return valid.detach().cpu().numpy().astype(np.uint8)


def render_quality_overlay_from_cv_meshes(
    frame_bgr: np.ndarray,
    verts_cv_list: list[np.ndarray],
    faces_list: list[np.ndarray],
    names: list[str],
    k: np.ndarray,
    device: torch.device,
    fill_alpha: float = DEFAULT_OVERLAY_FILL_ALPHA,
    contour_thickness: int = DEFAULT_OVERLAY_CONTOUR_THICKNESS,
) -> np.ndarray:
    """Render multi-mesh silhouette+outline overlay from OpenCV-camera meshes."""
    if len(verts_cv_list) != len(faces_list) or len(verts_cv_list) != len(names):
        raise ValueError("verts/faces/names lengths must match for overlay rendering.")

    h, w = frame_bgr.shape[:2]
    color_map = build_overlay_color_map(names)
    out = frame_bgr.copy()
    for name, verts_cv, faces in zip(names, verts_cv_list, faces_list):
        if verts_cv.size == 0 or faces.size == 0:
            continue
        mask = rasterize_silhouette_mask_cv_mesh(
            verts_cv=verts_cv,
            faces=faces,
            k=k,
            width=w,
            height=h,
            device=device,
        )
        out = draw_mask_outline_overlay(
            image_bgr=out,
            mask=mask,
            color_bgr=color_map[name],
            fill_alpha=fill_alpha,
            contour_thickness=contour_thickness,
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


def save_colored_point_cloud(
    path: Path,
    points: np.ndarray,
    colors_rgb: np.ndarray,
) -> None:
    if points.shape[0] == 0:
        cloud = trimesh.points.PointCloud(vertices=np.zeros((0, 3), dtype=np.float32))
    else:
        colors_rgba = np.concatenate(
            [
                colors_rgb.astype(np.uint8),
                255 * np.ones((points.shape[0], 1), dtype=np.uint8),
            ],
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
    full_transformed_mesh_points: np.ndarray | None,
    uv_ref: np.ndarray,
    colors_rgb: np.ndarray,
    intrinsics: np.ndarray,
    frame_bgr: np.ndarray,
    point_radius: int,
    save_depth_fixed: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_depth_fixed:
        save_colored_point_cloud(
            out_dir / f"iter_{iter_idx:05d}_depth_fixed.ply",
            depth_points,
            colors_rgb,
        )

    mesh_points_for_export = transformed_mesh_points
    mesh_colors_for_export = colors_rgb
    if (
        full_transformed_mesh_points is not None
        and full_transformed_mesh_points.shape[0] > 0
    ):
        gray_points = full_transformed_mesh_points.astype(np.float32)
        num_corr = int(transformed_mesh_points.shape[0])
        # Keep grey context points but prevent them from visually overwhelming
        # the colored correspondence points in dense meshes.
        if num_corr > 0:
            max_gray = max(1, 2 * num_corr)
            if gray_points.shape[0] > max_gray:
                keep_idx = np.linspace(
                    0, gray_points.shape[0] - 1, num=max_gray, dtype=np.int64
                )
                gray_points = gray_points[keep_idx]

        gray_rgb = np.full(
            (gray_points.shape[0], 3), 160, dtype=np.uint8
        )
        mesh_points_for_export = np.concatenate(
            [gray_points, transformed_mesh_points],
            axis=0,
        )
        mesh_colors_for_export = np.concatenate([gray_rgb, colors_rgb], axis=0)

    save_colored_point_cloud(
        out_dir / f"iter_{iter_idx:05d}_mesh_transformed.ply",
        mesh_points_for_export,
        mesh_colors_for_export,
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
