"""Shared mesh-generation helpers for Generate_Object_Mesh scripts."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh

SAM3D_PATH = "/my_workspace/4DHHOI/sam-3d-objects"

KEYS_TO_SAVE = {
    "6drotation_normalized",
    "rotation",
    "translation",
    "scale",
    "translation_scale",
}

SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0


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


def scale_camera_intrinsics(camera_info: Dict[str, Any], f_scale: float) -> Dict[str, Any]:
    """Scale fx/fy/lens for auto-estimated intrinsics."""
    if f_scale <= 0:
        raise ValueError(f"--f_scale must be > 0, got {f_scale}")

    scaled = copy.deepcopy(camera_info)
    k = scaled.get("intrinsics_pixels_3x3")
    if not isinstance(k, list) or len(k) != 3:
        raise ValueError("camera_info['intrinsics_pixels_3x3'] is missing or invalid")

    k[0][0] = float(k[0][0]) * float(f_scale)
    k[1][1] = float(k[1][1]) * float(f_scale)

    blender_rec = scaled.get("blender_recommendation", {})
    if "lens_mm" in blender_rec:
        blender_rec["lens_mm"] = float(blender_rec["lens_mm"]) * float(f_scale)
    blender_rec["focal_scale_factor"] = float(f_scale)
    scaled["blender_recommendation"] = blender_rec

    print(
        "  Scaled intrinsics: "
        f"fx={k[0][0]:.1f}px, fy={k[1][1]:.1f}px | lens={blender_rec.get('lens_mm', 0.0):.2f}mm"
    )
    return scaled


def compute_overlay_focal_scale(
    camera_info: Dict[str, Any],
    focal_length_mm_override: Optional[float],
) -> float:
    """Convert focal override in mm into an fx/fy scale for pixel-space K."""
    if focal_length_mm_override is None:
        return 1.0

    blender_rec = camera_info.get("blender_recommendation", {})
    base_lens_mm = float(blender_rec.get("lens_mm", 0.0))
    if base_lens_mm <= 0.0:
        raise ValueError(
            "camera_info.blender_recommendation.lens_mm must be > 0 to use --focal_length override."
        )
    return float(focal_length_mm_override) / base_lens_mm


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


def to_jsonable(value: Any) -> Any:
    """Convert tensor/numpy types to JSON-serializable Python types."""
    if value is None:
        return None
    value = to_numpy(value)
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0]) if value.size == 1 else value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
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


def quaternion_to_euler_xyz_degrees(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) -> Euler XYZ angles in degrees."""
    w, x, y, z = quat / np.linalg.norm(quat)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float32))


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


def extract_pose_data(output: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract pose components from SAM3D output."""
    rotation_quat = np.asarray(to_jsonable(output["rotation"]), dtype=np.float32).flatten()
    translation = np.asarray(to_jsonable(output["translation"]), dtype=np.float32).flatten()
    scale = np.asarray(to_jsonable(output["scale"]), dtype=np.float32).flatten()
    euler_xyz_deg = quaternion_to_euler_xyz_degrees(rotation_quat)

    return {
        "rotation_quat": rotation_quat,
        "translation": translation,
        "scale": scale,
        "euler_xyz_deg": euler_xyz_deg,
    }


def save_pose_json(
    output: Dict[str, Any],
    pose_data: Dict[str, np.ndarray],
    output_path: Path,
    focal_length_mm: float,
    camera_intrinsics_json: Path,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    """Save pose.json for one frame/object pair."""
    transform_data: Dict[str, Any] = {}
    for key in sorted(KEYS_TO_SAVE):
        if key in output:
            transform_data[key] = to_jsonable(output[key])

    transform_data["rotation_quaternion_wxyz"] = pose_data["rotation_quat"].tolist()
    transform_data["rotation_euler_xyz_degrees"] = pose_data["euler_xyz_deg"].tolist()
    transform_data["focal_length_mm_used_for_overlay"] = float(focal_length_mm)
    transform_data["camera_intrinsics_json"] = str(camera_intrinsics_json)

    if extra_fields:
        transform_data.update(extra_fields)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transform_data, f, indent=2)


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
            results.append((child.name, mask_path))

    if not results:
        raise FileNotFoundError(
            f"No first-frame masks found under {objects_root} for frame '{first_frame_stem}'"
        )
    return results
