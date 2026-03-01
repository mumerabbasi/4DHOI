"""
Generate 3D meshes from segmentation masks.

Reads frames and per-object masks from a Segment_Video output directory.
Uniformly samples --num_frames frames from the video and generates per-frame
object poses with SAM 3D Objects.

Run this script with the sam3d-objects environment.

Pipeline:
    1. Discover objects from Segment_Video/output/<video_xx>/objects/
    2. Discover frames from Segment_Video/output/<video_xx>/_frames/
    3. Uniformly sample --num_frames frames (default 4)
    4. Estimate camera intrinsics once from the first sampled frame
    5. Build one canonical mesh per object from the first sampled frame only
    6. For each sampled frame, for each object:
       - Load RGB frame and object mask
       - Use SAM 3D Objects to estimate per-frame transforms
       - Pose the first-frame canonical mesh with that frame's transforms
    7. Save outputs to:
       Generate_Object_Mesh/output/<video_xx>/
         - <frame_XXXX>/camera_intrinsics.json
         - <first_frame>/<object>/mesh.ply
         - <frame_XXXX>/<object>/pose.json
         - <frame_XXXX>/<object>/mesh_posed.ply
         - <frame_XXXX>/<object>/mesh_posed_overlay.png
         - <frame_XXXX>/all_objects_overlay.png

Coordinate System Notes:
    - SAM 3D Objects outputs PLY meshes in Z-up (PyTorch3D) coordinate system.
    - SAM 3D Objects transforms (rotation, translation, scale) are in Z-up.
    - All meshes are kept in Z-up throughout.

Rendering Notes:
    - PyTorch3D uses: +X is left, +Y is up
    - Screen space uses: +X is right, +Y is down
    - Convert coordinates with F_P3D_TO_CV before projection
    - Source: https://github.com/facebookresearch/sam-3d-objects/issues/56
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh

# Configuration
SAM3D_PATH = "/my_workspace/4DHHOI/sam-3d-objects"

KEYS_TO_SAVE = {
    "6drotation_normalized",
    "rotation",
    "translation",
    "scale",
    "translation_scale",
}

# P3D camera coords (+X left, +Y up, +Z forward) -> OpenCV (+X right, +Y down, +Z forward)
F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

SENSOR_WIDTH_MM = 36.0
SENSOR_HEIGHT_MM = 24.0


# =============================================================================
# Camera intrinsics
# =============================================================================


def estimate_camera_intrinsics(sam3d: Any, image_rgb: np.ndarray) -> Dict[str, Any]:
    """Estimate camera intrinsics from MoGe (via SAM3D depth model).

    Returns dict with pixel intrinsics and Blender-ready focal length in mm.
    """
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

    # MoGe intrinsics are normalized to image size.
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


# =============================================================================
# Tensor / numpy helpers
# =============================================================================


def _to_numpy(value: Any) -> Any:
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


def _to_jsonable(value: Any) -> Any:
    """Convert tensor/numpy types to JSON-serializable Python types."""
    if value is None:
        return None
    value = _to_numpy(value)
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0]) if value.size == 1 else value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


# =============================================================================
# Rendering helpers
# =============================================================================


def _project_mesh_vertices(
    mesh: trimesh.Trimesh,
    image_height: int,
    image_width: int,
    focal_length_mm: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Project visible Z-up mesh vertices to pixel coordinates."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        return None

    valid_mask = vertices[:, 2] > 0.1
    if not np.any(valid_mask):
        return None

    pts_cv = vertices[valid_mask] @ F_P3D_TO_CV.T
    z = pts_cv[:, 2]
    focal_px = focal_length_mm * image_width / SENSOR_WIDTH_MM
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


def _draw_projected_vertices(
    canvas_bgr: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    point_radius: int,
    vertex_colors: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    in_view: Optional[np.ndarray],
) -> None:
    """Draw projected points using mesh vertex colors only."""
    if vertex_colors is None or valid_mask is None or in_view is None:
        raise ValueError("Mesh is missing vertex colors; fallback coloring is disabled.")

    colors = np.asarray(vertex_colors)[valid_mask][in_view]
    for x, y, c in zip(u, v, colors):
        cv2.circle(
            canvas_bgr,
            (int(x), int(y)),
            point_radius,
            (int(c[2]), int(c[1]), int(c[0])),
            -1,
        )


def render_posed_mesh_overlay(
    image_rgb: np.ndarray,
    posed_mesh: trimesh.Trimesh,
    focal_length_mm: float,
    point_radius: int = 1,
) -> np.ndarray:
    """Render Z-up posed mesh vertices overlaid on an image (returns BGR)."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    proj = _project_mesh_vertices(posed_mesh, h, w, focal_length_mm)
    if proj is None:
        return result
    u, v, vm, iv = proj
    _draw_projected_vertices(result, u, v, point_radius, _get_vertex_colors(posed_mesh), vm, iv)
    return result


def render_all_posed_meshes_overlay(
    image_rgb: np.ndarray,
    posed_meshes: List[trimesh.Trimesh],
    focal_length_mm: float,
    point_radius: int = 1,
) -> np.ndarray:
    """Render multiple Z-up posed meshes overlaid on a single image (returns BGR)."""
    h, w = image_rgb.shape[:2]
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    for mesh in posed_meshes:
        proj = _project_mesh_vertices(mesh, h, w, focal_length_mm)
        if proj is None:
            continue
        u, v, vm, iv = proj
        _draw_projected_vertices(result, u, v, point_radius, _get_vertex_colors(mesh), vm, iv)

    return result


# =============================================================================
# Quaternion / transform utilities
# =============================================================================


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
    verts = np.array(canonical_mesh.vertices, dtype=np.float32)
    r_mat = quaternion_to_rotation_matrix(rotation_quat)
    verts = (verts * scale) @ r_mat + translation

    posed = canonical_mesh.copy()
    posed.vertices = verts
    return posed


# =============================================================================
# SAM 3D Objects interface
# =============================================================================


def load_sam3d():
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


def generate_mesh(inference, image: np.ndarray, mask: np.ndarray) -> dict:
    """Generate SAM 3D Objects output with mesh and pose fields."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if mask.dtype != bool:
        mask = mask > 0
    return inference(image, mask, seed=42)


def sam3d_mesh_to_trimesh(sam3d_mesh: Any) -> trimesh.Trimesh:
    """Convert SAM3D mesh output to trimesh (Z-up PLY)."""
    vertices = _to_numpy(sam3d_mesh.vertices)
    faces = _to_numpy(sam3d_mesh.faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    vertex_attrs = _to_numpy(getattr(sam3d_mesh, "vertex_attrs", None))
    if vertex_attrs is not None and getattr(vertex_attrs, "ndim", 0) == 2 and vertex_attrs.shape[1] >= 3:
        mesh.visual.vertex_colors = np.clip(vertex_attrs[:, :3] * 255.0, 0, 255).astype(np.uint8)
    return mesh


# =============================================================================
# Save helpers
# =============================================================================


def extract_pose_data(output: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract pose components from SAM3D output."""
    rotation_quat = np.asarray(_to_jsonable(output["rotation"]), dtype=np.float32).flatten()
    translation = np.asarray(_to_jsonable(output["translation"]), dtype=np.float32).flatten()
    scale = np.asarray(_to_jsonable(output["scale"]), dtype=np.float32).flatten()
    euler_xyz_deg = quaternion_to_euler_xyz_degrees(rotation_quat)

    return {
        "rotation_quat": rotation_quat,
        "translation": translation,
        "scale": scale,
        "euler_xyz_deg": euler_xyz_deg,
    }


def save_canonical_mesh(canonical_mesh: trimesh.Trimesh, canonical_mesh_path: Path) -> None:
    """Save canonical mesh for one frame/object pair."""
    canonical_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_mesh.export(str(canonical_mesh_path))
    print(f"    Saved canonical mesh: {canonical_mesh_path.name}")


def save_pose_json(
    output: Dict[str, Any],
    pose_data: Dict[str, np.ndarray],
    output_path: Path,
    focal_length_mm: float,
    camera_intrinsics_json: Path,
) -> None:
    """Save pose.json for one frame/object pair."""
    transform_data: Dict[str, Any] = {}
    for key in sorted(KEYS_TO_SAVE):
        if key in output:
            transform_data[key] = _to_jsonable(output[key])

    transform_data["rotation_quaternion_wxyz"] = pose_data["rotation_quat"].tolist()
    transform_data["rotation_euler_xyz_degrees"] = pose_data["euler_xyz_deg"].tolist()
    transform_data["focal_length_mm_used_for_overlay"] = float(focal_length_mm)
    transform_data["camera_intrinsics_json"] = str(camera_intrinsics_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(transform_data, f, indent=2)
    print(f"    Saved: {output_path.name}")


def save_posed_mesh_and_overlay(
    posed_mesh: trimesh.Trimesh,
    output_dir: Path,
    image_rgb: np.ndarray,
    focal_length_mm: float,
) -> None:
    """Save posed mesh and overlay image for one frame/object pair."""
    output_dir.mkdir(parents=True, exist_ok=True)

    posed_mesh_path = output_dir / "mesh_posed.ply"
    posed_mesh.export(str(posed_mesh_path))
    print(f"    Saved: {posed_mesh_path.name}")

    overlay = render_posed_mesh_overlay(image_rgb, posed_mesh, focal_length_mm)
    overlay_path = output_dir / "mesh_posed_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"    Saved: {overlay_path.name}")


# =============================================================================
# Discovery and sampling
# =============================================================================


def discover_objects(input_dir: Path) -> List[Tuple[str, Path]]:
    """Auto-discover objects under <input_dir>/objects/."""
    objects_root = input_dir / "objects"
    results: List[Tuple[str, Path]] = []
    for child in sorted(objects_root.iterdir()):
        if not child.is_dir():
            continue
        mask_dir = child / "object_segmentation" / "masks"
        if mask_dir.is_dir():
            results.append((child.name, mask_dir))
    if not results:
        raise FileNotFoundError(f"No valid object dirs found under {objects_root}")
    return results


def discover_frames(input_dir: Path) -> List[str]:
    """List frame stems from <input_dir>/_frames/."""
    frames_dir = input_dir / "_frames"
    stems = sorted(
        p.stem for p in frames_dir.iterdir() if p.is_file() and p.stem.startswith("frame_")
    )
    if not stems:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
    return stems


def sample_frames_uniformly(frame_stems: List[str], num_frames: int) -> List[str]:
    """Uniformly sample num_frames (first & last always included)."""
    total = len(frame_stems)
    if num_frames >= total:
        return list(frame_stems)
    indices = np.unique(np.linspace(0, total - 1, num_frames, dtype=int))
    return [frame_stems[i] for i in indices]


def find_frame_image_path(frames_dir: Path, frame_stem: str) -> Optional[Path]:
    """Resolve frame image path by trying supported extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = frames_dir / f"{frame_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# =============================================================================
# Main orchestrator
# =============================================================================


def process_video_directory(
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
    num_frames: int = 4,
    focal_length_mm: Optional[float] = None,
    f_scale: float = 0.8,
) -> None:
    """Generate meshes for uniformly sampled frames.

    Output structure::

        <mesh_output_root>/<video_xx>/<frame_XXXX>/
            camera_intrinsics.json
            <object>/
              mesh.ply  # only written under first sampled frame
              pose.json
              mesh_posed.ply
              mesh_posed_overlay.png
            all_objects_overlay.png
    """
    objects = discover_objects(input_dir)
    frame_stems = discover_frames(input_dir)
    sampled = sample_frames_uniformly(frame_stems, num_frames)
    if not sampled:
        raise RuntimeError("No sampled frames available.")

    video_name = input_dir.name
    output_root = (mesh_output_root / video_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames_dir = input_dir / "_frames"

    print(f"Video: {video_name}")
    print(f"Objects: {[n for n, _ in objects]}")
    print(f"Total frames: {len(frame_stems)}, sampled: {len(sampled)}")
    print(f"Sampled frames: {sampled}")
    print(f"Output: {output_root}\n")

    first_frame_stem = sampled[0]
    first_frame_path = find_frame_image_path(frames_dir, first_frame_stem)
    if first_frame_path is None:
        raise FileNotFoundError(f"Frame image not found for first sampled frame: {first_frame_stem}")

    first_frame_bgr = cv2.imread(str(first_frame_path))
    if first_frame_bgr is None:
        raise RuntimeError(f"Could not load first sampled frame: {first_frame_path}")
    first_frame_rgb = cv2.cvtColor(first_frame_bgr, cv2.COLOR_BGR2RGB)

    print(f"Estimating intrinsics from first sampled frame only: {first_frame_stem}")
    camera_info_shared = estimate_camera_intrinsics(sam3d, first_frame_rgb)

    if focal_length_mm is None:
        camera_info_shared = scale_camera_intrinsics(camera_info_shared, f_scale)
        focal_overlay_mm = float(camera_info_shared["blender_recommendation"]["lens_mm"])
        print(
            "Focal mode: auto+f_scale "
            f"(f_scale={f_scale}, focal_for_projection={focal_overlay_mm:.3f}mm)"
        )
    else:
        focal_overlay_mm = float(focal_length_mm)
        print(
            "Focal mode: explicit --focal_length "
            f"(focal_for_projection={focal_overlay_mm:.3f}mm, f_scale ignored for projection)"
        )

    canonical_meshes: Dict[str, trimesh.Trimesh] = {}
    first_frame_outputs: Dict[str, Dict[str, Any]] = {}

    print(f"\nPrecomputing canonical meshes from first sampled frame: {first_frame_stem}")
    first_frame_output_dir = output_root / first_frame_stem
    first_frame_output_dir.mkdir(parents=True, exist_ok=True)

    for obj_name, mask_dir in objects:
        print(f"\n{'=' * 50}")
        print(f"  Canonical object: {obj_name}  |  Frame: {first_frame_stem}")

        mask_path = mask_dir / f"{first_frame_stem}.png"
        if not mask_path.exists():
            print(f"    Warning: mask not found for canonical frame - {mask_path}; skipping object.")
            continue

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"    Warning: failed to read mask for canonical frame - {mask_path}; skipping object.")
            continue
        mask = (mask_gray > 127).astype(np.uint8)

        try:
            output = generate_mesh(sam3d, first_frame_rgb, mask)
            mesh_data = output["mesh"]
            canonical_mesh = sam3d_mesh_to_trimesh(mesh_data[0])
            if _get_vertex_colors(canonical_mesh) is None:
                raise ValueError("Canonical mesh is missing vertex colors.")

            canonical_meshes[obj_name] = canonical_mesh
            first_frame_outputs[obj_name] = output

            canonical_path = first_frame_output_dir / obj_name / "mesh.ply"
            save_canonical_mesh(canonical_mesh, canonical_path)
        except Exception as exc:
            print(f"    Canonical mesh generation failed: {exc}")

    if not canonical_meshes:
        raise RuntimeError(
            f"Failed to build canonical meshes from first sampled frame: {first_frame_stem}"
        )

    for frame_idx, frame_stem in enumerate(sampled):
        print(f"\n{'#' * 60}")
        print(f"Frame {frame_idx + 1}/{len(sampled)}: {frame_stem}")
        print(f"{'#' * 60}")

        frame_path = find_frame_image_path(frames_dir, frame_stem)
        if frame_path is None:
            print(f"  Error: frame image not found for {frame_stem}")
            continue

        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            print(f"  Error: could not load {frame_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        frame_output_dir = output_root / frame_stem
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        cam_json = frame_output_dir / "camera_intrinsics.json"
        with cam_json.open("w", encoding="utf-8") as f:
            json.dump(camera_info_shared, f, indent=2)
        print(f"  Saved frame intrinsics: {cam_json}")

        posed_meshes: List[trimesh.Trimesh] = []

        for obj_name, mask_dir in objects:
            print(f"\n{'=' * 50}")
            print(f"  Object: {obj_name}  |  Frame: {frame_stem}")

            canonical_mesh = canonical_meshes.get(obj_name)
            if canonical_mesh is None:
                print(
                    "    Warning: canonical mesh unavailable from first sampled frame; "
                    "skipping object."
                )
                continue

            try:
                if frame_stem == first_frame_stem:
                    output = first_frame_outputs.get(obj_name)
                    if output is None:
                        print("    Warning: missing cached first-frame output; skipping object.")
                        continue
                else:
                    mask_path = mask_dir / f"{frame_stem}.png"
                    if not mask_path.exists():
                        print(f"    Warning: mask not found - {mask_path}; skipping.")
                        continue

                    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask_gray is None:
                        print(f"    Warning: failed to read mask - {mask_path}; skipping.")
                        continue
                    mask = (mask_gray > 127).astype(np.uint8)
                    output = generate_mesh(sam3d, image_rgb, mask)

                pose_data = extract_pose_data(output)
                print(f"    Rotation (quat): {pose_data['rotation_quat']}")
                print(f"    Rotation (euler xyz deg): {pose_data['euler_xyz_deg']}")
                print(f"    Translation: {pose_data['translation']}")
                print(f"    Scale: {pose_data['scale']}")

                object_frame_dir = frame_output_dir / obj_name
                object_frame_dir.mkdir(parents=True, exist_ok=True)
                if frame_stem == first_frame_stem:
                    canonical_path = object_frame_dir / "mesh.ply"
                    if not canonical_path.exists():
                        save_canonical_mesh(canonical_mesh, canonical_path)

                save_pose_json(
                    output=output,
                    pose_data=pose_data,
                    output_path=object_frame_dir / "pose.json",
                    focal_length_mm=focal_overlay_mm,
                    camera_intrinsics_json=cam_json,
                )

                posed_mesh = create_posed_mesh(
                    canonical_mesh,
                    pose_data["rotation_quat"],
                    pose_data["translation"],
                    pose_data["scale"],
                )
                save_posed_mesh_and_overlay(
                    posed_mesh=posed_mesh,
                    output_dir=object_frame_dir,
                    image_rgb=image_rgb,
                    focal_length_mm=focal_overlay_mm,
                )
                posed_meshes.append(posed_mesh)

            except Exception as exc:
                print(f"    Mesh generation failed: {exc}")

        if posed_meshes:
            print(f"\n{'=' * 50}")
            print(f"Generating combined overlay for {frame_stem}...")
            try:
                overlay = render_all_posed_meshes_overlay(
                    image_rgb=image_rgb,
                    posed_meshes=posed_meshes,
                    focal_length_mm=focal_overlay_mm,
                )
                overlay_path = frame_output_dir / "all_objects_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                print(f"Saved: {overlay_path}")
            except Exception as exc:
                print(f"Failed to generate combined overlay: {exc}")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from Segment_Video output (sam3d-objects env).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../Segment_Video/output/video_01",
        help="Segment_Video output dir with _frames/ and objects/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Mesh output root (<output_dir>/<video_xx>/).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=4,
        help="Number of frames to uniformly sample (default: 4).",
    )
    parser.add_argument(
        "--focal_length",
        type=float,
        default=None,
        help="Focal length in mm for projection (default: auto from first sampled frame).",
    )
    parser.add_argument(
        "--f_scale",
        type=float,
        default=0.9,
        help="Scale factor for auto-estimated fx/fy/lens from first sampled frame (default: 0.8).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path(__file__).parent / input_dir
    input_dir = input_dir.resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent / output_dir
    output_dir = output_dir.resolve()

    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()
    print("SAM 3D Objects loaded successfully\n")

    focal_msg = f"{args.focal_length:.3f}mm" if args.focal_length is not None else "auto (MoGe first sampled frame)"
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Focal length: {focal_msg}")
    print(f"Focal scale (auto mode only): {args.f_scale}")
    print(f"Num frames: {args.num_frames}\n")

    process_video_directory(
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=output_dir,
        num_frames=args.num_frames,
        focal_length_mm=args.focal_length,
        f_scale=args.f_scale,
    )

    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()
