"""Shared mesh-generation helpers for Generate_Object_Mesh scripts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

SAM3D_PATH = "/my_workspace/4DHHOI/sam-3d-objects"

SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0


def normalize_object_slug(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def estimate_camera_intrinsics(sam3d: Any, image_rgb: np.ndarray) -> Dict[str, Any]:
    """Estimate camera intrinsics from MoGe (via SAM3D depth model)."""
    h, w = image_rgb.shape[:2]
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        depth_output = sam3d._pipeline.depth_model(image_tensor)

    intrinsics = depth_output["intrinsics"]
    if hasattr(intrinsics, "detach"):
        intrinsics = intrinsics.detach().cpu().numpy()
    else:
        intrinsics = np.asarray(intrinsics)
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[0]

    fx_px = float(intrinsics[0, 0]) * w
    fy_px = float(intrinsics[1, 1]) * h
    cx_px = float(intrinsics[0, 2]) * w
    cy_px = float(intrinsics[1, 2]) * h
    focal_mm = fx_px * SENSOR_WIDTH_MM / float(w)

    print(f"  MoGe intrinsics: fx={fx_px:.1f}px, fy={fy_px:.1f}px | lens={focal_mm:.2f}mm")
    return {
        "source": "moge_from_sam3d_depth_model",
        "intrinsics_pixels_3x3": [
            [fx_px, 0.0, cx_px],
            [0.0, fy_px, cy_px],
            [0.0, 0.0, 1.0],
        ],
        "blender_recommendation": {
            "sensor_fit": "HORIZONTAL",
            "lens_mm": focal_mm,
            "sensor_width_mm": SENSOR_WIDTH_MM,
            "sensor_height_mm": SENSOR_HEIGHT_MM,
            "note": "Lens uses fx with full-frame horizontal fit.",
        },
    }


def to_numpy(value: Any) -> Any:
    """Convert torch-like values to numpy."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def create_posed_mesh(
    canonical_mesh: trimesh.Trimesh,
    rotation_quat: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> trimesh.Trimesh:
    """Apply scale -> rotate -> translate to a Z-up canonical mesh."""
    verts = np.asarray(canonical_mesh.vertices, dtype=np.float32)
    r_mat = quaternion_to_rotation_matrix(rotation_quat)
    verts = (verts * scale) @ r_mat + translation

    posed = canonical_mesh.copy()
    posed.vertices = verts
    return posed


def load_sam3d() -> Any:
    """Load SAM 3D Objects inference pipeline."""
    notebook_path = Path(SAM3D_PATH) / "notebook"
    if str(notebook_path) not in sys.path:
        sys.path.insert(0, str(notebook_path))

    from inference import Inference

    possible_paths = [
        Path(SAM3D_PATH) / "checkpoints" / "hf" / "pipeline.yaml",
        Path(SAM3D_PATH) / "checkpoints" / "hf-download" / "checkpoints" / "pipeline.yaml",
    ]
    config_path = next((p for p in possible_paths if p.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"SAM3D config not found. Tried: {[str(p) for p in possible_paths]}")

    print(f"Loading SAM 3D Objects from: {config_path}")
    return Inference(str(config_path), compile=False)


def load_sam3d_postprocess_mesh() -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """Load the SAM3D mesh post-processing utility."""
    sam3d_root = Path(SAM3D_PATH)
    if str(sam3d_root) not in sys.path:
        sys.path.insert(0, str(sam3d_root))

    from sam3d_objects.model.backbone.tdfy_dit.utils.postprocessing_utils import (
        postprocess_mesh,
    )

    return postprocess_mesh


def generate_mesh(inference: Any, image: np.ndarray, mask: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    """Generate SAM 3D Objects output with mesh and pose fields."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if mask.dtype != bool:
        mask = mask > 0
    return inference(image, mask, seed=int(seed))


def sam3d_mesh_to_trimesh(sam3d_mesh: Any) -> trimesh.Trimesh:
    """Convert SAM3D mesh output to trimesh (Z-up PLY)."""
    vertices = to_numpy(sam3d_mesh.vertices)
    faces = to_numpy(sam3d_mesh.faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    vertex_attrs = to_numpy(getattr(sam3d_mesh, "vertex_attrs", None))
    if vertex_attrs is not None and getattr(vertex_attrs, "ndim", 0) == 2 and vertex_attrs.shape[1] >= 3:
        mesh.visual.vertex_colors = np.clip(vertex_attrs[:, :3] * 255.0, 0, 255).astype(np.uint8)
    return mesh


def transfer_vertex_colors_by_distance(
    source_vertices: np.ndarray,
    source_vertex_colors: np.ndarray,
    target_vertices: np.ndarray,
    *,
    k_neighbors: int = 4,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Transfer colors to target vertices via inverse-distance weighting."""
    if source_vertices.size == 0 or target_vertices.size == 0:
        return np.zeros((target_vertices.shape[0], source_vertex_colors.shape[1]), dtype=np.uint8)

    source_xyz = torch.as_tensor(np.asarray(source_vertices, dtype=np.float32), dtype=torch.float32)
    target_xyz = torch.as_tensor(np.asarray(target_vertices, dtype=np.float32), dtype=torch.float32)
    source_colors = torch.as_tensor(
        np.asarray(source_vertex_colors, dtype=np.float32),
        dtype=torch.float32,
    )

    k_neighbors = max(1, min(int(k_neighbors), source_xyz.shape[0]))
    transferred_chunks: List[np.ndarray] = []

    for start_idx in range(0, target_xyz.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, target_xyz.shape[0])
        distances = torch.cdist(target_xyz[start_idx:end_idx], source_xyz)
        nearest_distances, nearest_indices = torch.topk(
            distances,
            k=k_neighbors,
            largest=False,
            dim=1,
        )

        nearest_colors = source_colors[nearest_indices]
        weights = 1.0 / torch.clamp(nearest_distances, min=1e-8)
        blended = (nearest_colors * weights.unsqueeze(-1)).sum(dim=1)
        blended = blended / weights.sum(dim=1, keepdim=True)
        transferred_chunks.append(blended.cpu().numpy())

    transferred = np.concatenate(transferred_chunks, axis=0)
    return np.clip(np.rint(transferred), 0, 255).astype(np.uint8)


def postprocess_trimesh_with_sam3d(
    mesh: trimesh.Trimesh,
    simplify_ratio: float,
    *,
    verbose: bool = False,
) -> trimesh.Trimesh:
    """Run SAM3D mesh postprocess and restore vertex colors on simplified vertices."""
    postprocess_mesh = load_sam3d_postprocess_mesh()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    processed_vertices, processed_faces = postprocess_mesh(
        vertices,
        faces,
        simplify=simplify_ratio > 0.0,
        simplify_ratio=float(simplify_ratio),
        fill_holes=False,
    )
    processed_vertices = np.asarray(processed_vertices, dtype=np.float32)
    processed_faces = np.asarray(processed_faces, dtype=np.int32)

    processed_mesh = trimesh.Trimesh(
        vertices=processed_vertices,
        faces=processed_faces,
        process=False,
    )

    source_colors = np.asarray(getattr(mesh.visual, "vertex_colors", None))
    if source_colors.ndim == 2 and source_colors.shape[0] == vertices.shape[0]:
        processed_mesh.visual.vertex_colors = transfer_vertex_colors_by_distance(
            vertices,
            source_colors,
            np.asarray(processed_mesh.vertices, dtype=np.float32),
        )

    return processed_mesh


def extract_pose_components(output: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract rotation, translation, and scale arrays from SAM3D output."""
    rotation_quat = np.asarray(to_numpy(output["rotation"]), dtype=np.float32).reshape(-1)
    translation = np.asarray(to_numpy(output["translation"]), dtype=np.float32).reshape(-1)
    scale = np.asarray(to_numpy(output["scale"]), dtype=np.float32).reshape(-1)
    return rotation_quat, translation, scale


def discover_frames(input_dir: Path) -> List[str]:
    """List frame stems from <input_dir>/_frames/."""
    frames_dir = input_dir / "_frames"
    stems = sorted(
        p.stem for p in frames_dir.iterdir() if p.is_file() and p.stem.startswith("frame_")
    )
    if not stems:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
    return stems


def find_frame_image_path(frames_dir: Path, frame_stem: str) -> Optional[Path]:
    """Resolve frame image path by trying supported extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = frames_dir / f"{frame_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def discover_first_frame_stem(input_dir: Path) -> str:
    """Discover first frame stem from <input_dir>/_frames/."""
    stems = discover_frames(input_dir)
    return stems[0]


def discover_objects_with_first_frame_masks(
    input_dir: Path,
    first_frame_stem: str,
) -> List[Tuple[str, Path]]:
    """Discover object first-frame masks from Segment_Video fixed structure."""
    objects_root = input_dir / "objects"
    if not objects_root.is_dir():
        raise FileNotFoundError(f"Missing objects directory: {objects_root}")

    results: List[Tuple[str, Path]] = []
    for child in sorted(objects_root.iterdir()):
        if not child.is_dir():
            continue
        mask_path = child / "object_segmentation" / "masks" / f"{first_frame_stem}.png"
        if mask_path.exists():
            results.append((normalize_object_slug(child.name), mask_path))

    if not results:
        raise FileNotFoundError(
            f"No first-frame masks found under {objects_root} for frame '{first_frame_stem}'"
        )
    return results
