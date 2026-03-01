"""Shared overlay rendering utilities for Generate_Object_Mesh scripts."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import trimesh

# P3D camera coords (+X left, +Y up, +Z forward) -> OpenCV (+X right, +Y down, +Z forward)
F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
SENSOR_WIDTH_MM = 36.0

# High-contrast palette for object overlays (BGR).
_OBJECT_PALETTE_BRG: List[Tuple[int, int, int]] = [
    (0, 255, 255),   # yellow
    (255, 140, 0),   # orange
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 165, 255),   # amber
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
]


def ensure_quality_backend_available() -> None:
    """Raise a clear error if PyTorch3D quality backend is unavailable."""
    try:
        _import_pytorch3d()
    except Exception as exc:
        raise RuntimeError(
            "Overlay renderer 'quality' requires PyTorch3D. "
            "Install pytorch3d or run with --overlay_quality legacy."
        ) from exc


def camera_k_from_info(camera_info: Dict[str, Any], focal_scale: float = 1.0) -> np.ndarray:
    """Extract and optionally focal-scale intrinsics matrix K (3x3)."""
    k = np.asarray(camera_info.get("intrinsics_pixels_3x3"), dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsics_pixels_3x3 shape (3, 3), got {k.shape}")
    if focal_scale <= 0:
        raise ValueError(f"focal_scale must be > 0, got {focal_scale}")

    k_out = k.copy()
    k_out[0, 0] *= float(focal_scale)
    k_out[1, 1] *= float(focal_scale)
    return k_out


def build_object_color_map(object_names: Sequence[str]) -> Dict[str, Tuple[int, int, int]]:
    """Assign stable high-contrast BGR colors to object names."""
    color_map: Dict[str, Tuple[int, int, int]] = {}
    for i, name in enumerate(object_names):
        color_map[str(name)] = _OBJECT_PALETTE_BRG[i % len(_OBJECT_PALETTE_BRG)]
    return color_map


def build_p3d_rasterizer_from_k(
    k: np.ndarray,
    width: int,
    height: int,
    device="cuda",
):
    """Build a PyTorch3D rasterizer using pixel-space intrinsics K."""
    torch, PerspectiveCameras, MeshRasterizer, RasterizationSettings, _ = _import_pytorch3d()
    dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=dev, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=dev, dtype=torch.float32),
        image_size=torch.tensor([[height, width]], device=dev, dtype=torch.float32),
        in_ndc=False,
        device=dev,
    )
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
        max_faces_per_bin=300000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def rasterize_mesh_silhouette(
    verts_p3d: np.ndarray,
    faces: np.ndarray,
    rasterizer,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize mesh and return binary mask and z-buffer in camera depth units."""
    torch, _, _, _, Meshes = _import_pytorch3d()
    device = rasterizer.cameras.device

    if verts_p3d.size == 0 or faces.size == 0:
        image_size = rasterizer.raster_settings.image_size
        h, w = int(image_size[0]), int(image_size[1])
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), np.inf, dtype=np.float32)

    verts_t = torch.from_numpy(verts_p3d.astype(np.float32)).to(device=device)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]
    zbuf = fragments.zbuf[0, ..., 0]
    valid = (pix_to_face >= 0) & torch.isfinite(zbuf) & (zbuf > 1e-6)

    z_out = torch.where(valid, zbuf, torch.full_like(zbuf, float("inf")))
    mask_np = valid.detach().cpu().numpy().astype(np.uint8)
    zbuf_np = z_out.detach().cpu().numpy().astype(np.float32)
    return mask_np, zbuf_np


def draw_mask_outline_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    fill_alpha: float = 0.35,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Blend a colored silhouette and draw contour on top."""
    if mask is None:
        return image_bgr.copy()

    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return image_bgr.copy()

    out = image_bgr.astype(np.float32).copy()
    color_arr = np.array(color_bgr, dtype=np.float32)
    a = float(np.clip(fill_alpha, 0.0, 1.0))
    out[mask_bool] = (1.0 - a) * out[mask_bool] + a * color_arr
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours_info = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    outline = tuple(int(np.clip(c + 48, 0, 255)) for c in color_bgr)
    cv2.drawContours(
        out_u8,
        contours,
        contourIdx=-1,
        color=outline,
        thickness=max(1, int(contour_thickness)),
        lineType=cv2.LINE_AA,
    )
    return out_u8


def render_single_object_overlay_quality(
    image_rgb: np.ndarray,
    posed_mesh: trimesh.Trimesh,
    camera_k: np.ndarray,
    color_bgr: Tuple[int, int, int],
    fill_alpha: float = 0.35,
    contour_thickness: int = 2,
    device="cuda",
) -> np.ndarray:
    """Quality overlay: triangle rasterization + silhouette fill + contour."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    verts = np.asarray(posed_mesh.vertices, dtype=np.float32)
    faces = np.asarray(posed_mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return result

    rasterizer = build_p3d_rasterizer_from_k(camera_k, w, h, device=device)
    mask, _ = rasterize_mesh_silhouette(verts, faces, rasterizer)
    return draw_mask_outline_overlay(
        result,
        mask,
        color_bgr=color_bgr,
        fill_alpha=fill_alpha,
        contour_thickness=contour_thickness,
    )


def render_multi_object_overlay_quality(
    image_rgb: np.ndarray,
    posed_meshes: Sequence[trimesh.Trimesh],
    camera_k: np.ndarray,
    colors_bgr: Sequence[Tuple[int, int, int]],
    fill_alpha: float = 0.35,
    contour_thickness: int = 2,
    device="cuda",
) -> np.ndarray:
    """Quality multi-object overlay with depth-aware compositing."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    if not posed_meshes:
        return result

    rasterizer = build_p3d_rasterizer_from_k(camera_k, w, h, device=device)
    all_masks: List[np.ndarray] = []
    all_zbufs: List[np.ndarray] = []

    for mesh in posed_meshes:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        mask, zbuf = rasterize_mesh_silhouette(verts, faces, rasterizer)
        all_masks.append(mask)
        all_zbufs.append(zbuf)

    z_stack = np.stack(all_zbufs, axis=0)
    valid_any = np.isfinite(z_stack).any(axis=0)
    nearest_idx = np.argmin(z_stack, axis=0)

    out = result
    for i, _ in enumerate(posed_meshes):
        visible_mask = valid_any & (nearest_idx == i)
        color = colors_bgr[i] if i < len(colors_bgr) else _OBJECT_PALETTE_BRG[i % len(_OBJECT_PALETTE_BRG)]
        out = draw_mask_outline_overlay(
            out,
            visible_mask.astype(np.uint8),
            color_bgr=color,
            fill_alpha=fill_alpha,
            contour_thickness=contour_thickness,
        )
    return out


def render_single_object_overlay_legacy(
    image_rgb: np.ndarray,
    posed_mesh: trimesh.Trimesh,
    focal_length_mm: float,
    point_radius: int = 1,
) -> np.ndarray:
    """Legacy overlay: project colored vertices as points."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    proj = _project_mesh_vertices_legacy(posed_mesh, h, w, focal_length_mm)
    if proj is None:
        return result
    u, v, vm, iv = proj
    _draw_projected_vertices_legacy(result, u, v, point_radius, _get_vertex_colors(posed_mesh), vm, iv)
    return result


def render_multi_object_overlay_legacy(
    image_rgb: np.ndarray,
    posed_meshes: Sequence[trimesh.Trimesh],
    focal_length_mm: float,
    point_radius: int = 1,
) -> np.ndarray:
    """Legacy overlay for multiple meshes with per-vertex colors."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    for mesh in posed_meshes:
        proj = _project_mesh_vertices_legacy(mesh, h, w, focal_length_mm)
        if proj is None:
            continue
        u, v, vm, iv = proj
        _draw_projected_vertices_legacy(result, u, v, point_radius, _get_vertex_colors(mesh), vm, iv)
    return result


def _project_mesh_vertices_legacy(
    mesh: trimesh.Trimesh,
    image_height: int,
    image_width: int,
    focal_length_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        return None

    valid_mask = vertices[:, 2] > 0.1
    if not np.any(valid_mask):
        return None

    pts_cv = vertices[valid_mask] @ F_P3D_TO_CV.T
    z = pts_cv[:, 2]
    focal_px = float(focal_length_mm) * float(image_width) / SENSOR_WIDTH_MM
    cx, cy = image_width / 2.0, image_height / 2.0
    u = ((pts_cv[:, 0] * focal_px) / z + cx).astype(np.int32)
    v = ((pts_cv[:, 1] * focal_px) / z + cy).astype(np.int32)
    in_view = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    if not np.any(in_view):
        return None
    return u[in_view], v[in_view], valid_mask, in_view


def _get_vertex_colors(mesh: trimesh.Trimesh) -> Optional[np.ndarray]:
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None:
        return None
    colors_np = np.asarray(colors)
    if colors_np.ndim != 2 or colors_np.shape[0] != len(mesh.vertices) or colors_np.shape[1] < 3:
        return None
    return colors_np[:, :3]


def _draw_projected_vertices_legacy(
    canvas_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    point_radius: int,
    vertex_colors: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    in_view: Optional[np.ndarray],
) -> None:
    if vertex_colors is None or valid_mask is None or in_view is None:
        raise ValueError("Mesh is missing vertex colors; fallback coloring is disabled.")
    colors = np.asarray(vertex_colors)[valid_mask][in_view]
    for x, y, c in zip(u, v, colors):
        cv2.circle(
            canvas_bgr,
            (int(x), int(y)),
            int(point_radius),
            (int(c[2]), int(c[1]), int(c[0])),
            -1,
        )


def _import_pytorch3d():
    import torch
    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    return torch, PerspectiveCameras, MeshRasterizer, RasterizationSettings, Meshes
