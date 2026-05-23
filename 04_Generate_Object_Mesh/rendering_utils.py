"""Shared overlay rendering utilities for 04_Generate_Object_Mesh scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import trimesh

# High-contrast palette for object overlays (BGR).
_OBJECT_PALETTE_BGR: List[Tuple[int, int, int]] = [
    (0, 255, 255),   # yellow
    (255, 140, 0),   # orange
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 165, 255),   # amber
    (0, 0, 255),     # red
    (255, 0, 0),     # blue
]

DEFAULT_OVERLAY_FILL_ALPHA = 0.60
DEFAULT_OVERLAY_CONTOUR_THICKNESS = 0
ColorBGR = Tuple[int, int, int]


@dataclass(frozen=True)
class QualityRenderBackend:
    """PyTorch3D backend objects supplied by the entry script."""
    torch: Any
    PerspectiveCameras: Any
    MeshRasterizer: Any
    RasterizationSettings: Any
    Meshes: Any


def camera_k_from_info(camera_info: Dict[str, Any]) -> np.ndarray:
    """Extract intrinsics matrix K (3x3)."""
    k = np.asarray(camera_info.get("intrinsics_pixels_3x3"), dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsics_pixels_3x3 shape (3, 3), got {k.shape}")
    return k.copy()


def build_object_color_map(object_names: Sequence[str]) -> Dict[str, ColorBGR]:
    """Assign stable high-contrast BGR colors to object names."""
    color_map: Dict[str, ColorBGR] = {}
    for i, name in enumerate(object_names):
        color_map[str(name)] = _OBJECT_PALETTE_BGR[i % len(_OBJECT_PALETTE_BGR)]
    return color_map


def build_p3d_rasterizer_from_k(
    k: np.ndarray,
    width: int,
    height: int,
    backend: QualityRenderBackend,
    device: str = "cuda",
):
    """Build a PyTorch3D rasterizer using pixel-space intrinsics K."""
    torch = backend.torch
    PerspectiveCameras = backend.PerspectiveCameras
    MeshRasterizer = backend.MeshRasterizer
    RasterizationSettings = backend.RasterizationSettings

    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
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
    backend: QualityRenderBackend,
) -> np.ndarray:
    """Rasterize mesh and return a binary silhouette mask."""
    torch = backend.torch
    Meshes = backend.Meshes
    device = rasterizer.cameras.device

    if verts_p3d.size == 0 or faces.size == 0:
        image_size = rasterizer.raster_settings.image_size
        h, w = int(image_size[0]), int(image_size[1])
        return np.zeros((h, w), dtype=np.uint8)

    verts_t = torch.from_numpy(verts_p3d.astype(np.float32)).to(device=device)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    with torch.no_grad():
        fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]
    zbuf = fragments.zbuf[0, ..., 0]
    valid = (pix_to_face >= 0) & torch.isfinite(zbuf) & (zbuf > 1e-6)
    return valid.detach().cpu().numpy().astype(np.uint8)


def draw_mask_outline_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: ColorBGR,
    fill_alpha: float = DEFAULT_OVERLAY_FILL_ALPHA,
    contour_thickness: int = DEFAULT_OVERLAY_CONTOUR_THICKNESS,
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


def add_overlay_legend(
    image_bgr: np.ndarray,
    legend_items: Sequence[Tuple[str, ColorBGR]],
) -> np.ndarray:
    """Draw a 03_Segment_Video-style legend onto an overlay image."""
    deduped_items: List[Tuple[str, ColorBGR]] = []
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


def render_single_object_overlay_quality(
    image_rgb: np.ndarray,
    posed_mesh: trimesh.Trimesh,
    camera_k: np.ndarray,
    color_bgr: ColorBGR,
    backend: QualityRenderBackend,
    fill_alpha: float = DEFAULT_OVERLAY_FILL_ALPHA,
    contour_thickness: int = DEFAULT_OVERLAY_CONTOUR_THICKNESS,
    device: str = "cuda",
) -> np.ndarray:
    """Quality overlay: triangle rasterization + silhouette fill + contour."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    verts = np.asarray(posed_mesh.vertices, dtype=np.float32)
    faces = np.asarray(posed_mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return result

    rasterizer = build_p3d_rasterizer_from_k(camera_k, w, h, backend=backend, device=device)
    mask = rasterize_mesh_silhouette(verts, faces, rasterizer, backend=backend)
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
    colors_bgr: Sequence[ColorBGR],
    backend: QualityRenderBackend,
    fill_alpha: float = DEFAULT_OVERLAY_FILL_ALPHA,
    contour_thickness: int = DEFAULT_OVERLAY_CONTOUR_THICKNESS,
    device: str = "cuda",
) -> np.ndarray:
    """Quality multi-object overlay with sequential mask compositing."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    if not posed_meshes:
        return result

    rasterizer = build_p3d_rasterizer_from_k(camera_k, w, h, backend=backend, device=device)
    out = result
    for i, mesh in enumerate(posed_meshes):
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        mask = rasterize_mesh_silhouette(verts, faces, rasterizer, backend=backend)
        color = colors_bgr[i] if i < len(colors_bgr) else _OBJECT_PALETTE_BGR[i % len(_OBJECT_PALETTE_BGR)]
        out = draw_mask_outline_overlay(
            out,
            mask.astype(np.uint8),
            color_bgr=color,
            fill_alpha=fill_alpha,
            contour_thickness=contour_thickness,
        )
    return out
