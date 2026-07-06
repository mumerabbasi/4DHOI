from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BLENDER_BIN = Path("/my_workspace/blender-4.2.17-linux-x64/blender")
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float32)
CONTACT_SEGMENT_BY_BODY_SEGMENT = {
    "left_hand": "left_hand_contact",
    "right_hand": "right_hand_contact",
    "left_arm": "left_arm_contact",
    "right_arm": "right_arm_contact",
    "left_leg": "left_leg_contact",
    "right_leg": "right_leg_contact",
    "left_foot": "left_foot_contact",
    "right_foot": "right_foot_contact",
    "head": "head_contact",
    "hips": "hips_contact",
    "back": "back_contact",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def build_blender_env(gpu_index: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_index is not None and str(gpu_index).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index).strip()
    return env


def build_default_paths(
    interaction_name: str,
    output_mode: str = "output",
) -> dict[str, Path]:
    return {
        "input_scene_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        "sig_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / interaction_name
        / "sig.json",
        "smpl_seg_json": PROJECT_DIR
        / "04_Estimate_Human_Pose"
        / "assets"
        / "smplx_vert_segmentation.json",
        "human_mesh_world": PROJECT_DIR
        / "05_Optimize_Static_Scene"
        / output_mode
        / interaction_name
        / "meshes"
        / "frame_0000_world.ply",
        "contact_spec_json": PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "contact_spec.json",
        "contact_render_image": PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
        / "target_scene_crop.png",
        "output_root": SCRIPT_DIR / output_mode / interaction_name / "semantics",
    }


def resolve_scannet_root(raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (PROJECT_DIR.parent / "Scannet++" / "data").resolve()


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = np.array(
        [
            [float(transforms_payload["fl_x"]), 0.0, float(transforms_payload["cx"])],
            [0.0, float(transforms_payload["fl_y"]), float(transforms_payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, width, height


def colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def load_colmap_pose(
    colmap_images_path: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for line in colmap_images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(
        f"Could not find camera '{camera_name}' in {colmap_images_path}"
    )


def resolve_scene_paths(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )

    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "image_path": scene_root / image_rel / camera_name,
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
    }


def load_scannet_camera(
    scene_paths: dict[str, Path],
    scene_context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    transforms_payload = load_json(scene_paths["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )
    return (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    )


def load_contact_spec_intrinsics(contact_spec_path: Path) -> np.ndarray:
    payload = load_json(contact_spec_path)
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"Contact spec missing camera object: {contact_spec_path}")
    intrinsics = np.asarray(camera.get("intrinsics_3x3"), dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            "Contact spec camera.intrinsics_3x3 must be a 3x3 matrix: "
            f"{contact_spec_path}"
        )
    return intrinsics


def load_image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to read image dimensions: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def normalize_label(text: str) -> str:
    return " ".join(
        str(text).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def normalize_scene_element(text: str, target_label: str) -> str:
    raw = str(text).strip().lower()
    normalized = normalize_label(text)
    target_norm = normalize_label(target_label)
    if (
        raw == "target_object"
        or raw.startswith("target_object_")
        or normalized in {"target object", "object", "target object 1", "target object 2", target_norm}
    ):
        return "target_object"
    return normalized


def resolve_sig_target_label(sig_payload: dict[str, Any]) -> str:
    target_objects = sig_payload.get("target_objects")
    if isinstance(target_objects, list) and target_objects:
        first_target = target_objects[0]
        if isinstance(first_target, dict):
            label = str(first_target.get("label", "")).strip()
            if label:
                return label
    target_object = sig_payload.get("target_object", {})
    if not isinstance(target_object, dict):
        return ""
    return str(target_object.get("label", "")).strip()


def iter_interaction_human_parts(sig_payload: dict[str, Any]) -> list[str]:
    target_label = resolve_sig_target_label(sig_payload)
    parts: list[str] = []
    seen: set[str] = set()
    interactions = sig_payload.get("interaction_edges", [])
    if not isinstance(interactions, list):
        return parts
    for interaction in interactions:
        if not isinstance(interaction, dict):
            continue
        scene_element = normalize_scene_element(
            str(interaction.get("scene_element", "")),
            target_label,
        )
        if scene_element not in {"target_object", "floor"}:
            continue
        human_part = normalize_label(str(interaction.get("human_part", "")))
        if human_part and human_part not in seen:
            parts.append(human_part)
            seen.add(human_part)
    return parts


def iter_interaction_human_parts_by_scene(
    sig_payload: dict[str, Any],
    scene_elements: set[str],
) -> list[str]:
    target_label = resolve_sig_target_label(sig_payload)
    parts: list[str] = []
    seen: set[str] = set()
    interactions = sig_payload.get("interaction_edges", [])
    if not isinstance(interactions, list):
        return parts
    for interaction in interactions:
        if not isinstance(interaction, dict):
            continue
        scene_element = normalize_scene_element(
            str(interaction.get("scene_element", "")),
            target_label,
        )
        if scene_element not in scene_elements:
            continue
        human_part = normalize_label(str(interaction.get("human_part", "")))
        if human_part and human_part not in seen:
            parts.append(human_part)
            seen.add(human_part)
    return parts


def load_smpl_body_segments(seg_path: Path) -> tuple[int, dict[str, np.ndarray], set[str]]:
    payload = load_json(seg_path)
    raw_segments = payload.get("segments")
    contact_segment_ids = payload.get("contact_segment_ids")
    if not isinstance(raw_segments, dict):
        raise KeyError(f"Expected 'segments' mapping in {seg_path}.")
    if not isinstance(contact_segment_ids, list):
        raise KeyError(f"Expected 'contact_segment_ids' list in {seg_path}.")

    vertex_count = int(payload["vertex_count"])
    segments: dict[str, np.ndarray] = {}
    for segment_id, indices in raw_segments.items():
        indices_array = np.unique(np.asarray(indices, dtype=np.int64))
        if indices_array.size == 0:
            continue
        if indices_array[0] < 0 or indices_array[-1] >= vertex_count:
            raise ValueError(f"Segment '{segment_id}' has out-of-range vertex ids.")
        segments[str(segment_id)] = indices_array

    contact_ids = {str(segment_id) for segment_id in contact_segment_ids}
    missing = sorted(contact_id for contact_id in contact_ids if contact_id not in segments)
    if missing:
        raise KeyError(f"Missing SMPL-X contact segments in {seg_path}: {missing}")
    return vertex_count, segments, contact_ids


def build_interaction_part_vertex_ids(
    sig_payload: dict[str, Any],
    smpl_segments: dict[str, np.ndarray],
    contact_segment_ids: set[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    part_names = iter_interaction_human_parts(sig_payload)
    part_vertex_ids: dict[str, np.ndarray] = {}
    part_records: list[dict[str, Any]] = []
    missing_parts: list[str] = []
    for part_name in part_names:
        body_segment_id = slugify_segment_name(part_name)
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(body_segment_id)
        if segment_id not in contact_segment_ids:
            missing_parts.append(part_name)
            continue
        vertex_ids = np.asarray(smpl_segments[segment_id], dtype=np.int64)
        part_vertex_ids[part_name] = vertex_ids
        part_records.append(
            {
                "part": part_name,
                "segment_id": segment_id,
                "vertex_count": int(vertex_ids.shape[0]),
            }
        )

    if not part_vertex_ids:
        return {}, {
            "available": False,
            "human_parts": part_names,
            "parts": [],
            "vertex_count": 0,
            "missing_human_parts": missing_parts,
        }

    return part_vertex_ids, {
        "available": True,
        "human_parts": part_names,
        "parts": part_records,
        "vertex_count": int(
            sum(record["vertex_count"] for record in part_records)
        ),
        "missing_human_parts": missing_parts,
    }


def build_interaction_part_vertices(
    sig_payload: dict[str, Any],
    smpl_segments: dict[str, np.ndarray],
    contact_segment_ids: set[str],
    human_vertices_world: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    part_vertex_ids, metadata = build_interaction_part_vertex_ids(
        sig_payload=sig_payload,
        smpl_segments=smpl_segments,
        contact_segment_ids=contact_segment_ids,
    )
    part_vertices: dict[str, np.ndarray] = {}
    for part_name, vertex_ids in part_vertex_ids.items():
        part_vertices[part_name] = human_vertices_world[vertex_ids].astype(np.float32)
    return part_vertices, metadata


def transform_world_to_camera(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return points_world @ rotation_world_to_camera.T + translation_world_to_camera[None]


def transform_camera_to_world(
    points_camera: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (points_camera - translation_world_to_camera[None]) @ rotation_world_to_camera


def filter_faces_to_camera_view(
    verts_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float | None = None,
    border_px: float = 96.0,
) -> np.ndarray:
    triangles = verts_camera[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1)
    if max_depth_m is not None:
        positive &= np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return faces[:0].copy()

    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * triangles[..., 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * triangles[..., 1] / z_safe + intrinsics[1, 2] - 0.5
    u_min = np.min(u, axis=1)
    u_max = np.max(u, axis=1)
    v_min = np.min(v, axis=1)
    v_max = np.max(v, axis=1)

    overlaps = (
        positive
        & (u_max >= -float(border_px))
        & (u_min <= float(width - 1) + float(border_px))
        & (v_max >= -float(border_px))
        & (v_min <= float(height - 1) + float(border_px))
    )
    return faces[overlaps].astype(np.int64)


def filter_face_indices_to_camera_view(
    verts_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float | None = None,
    border_px: float = 96.0,
) -> np.ndarray:
    triangles = verts_camera[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1)
    if max_depth_m is not None:
        positive &= np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return np.zeros((0,), dtype=np.int64)

    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * triangles[..., 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * triangles[..., 1] / z_safe + intrinsics[1, 2] - 0.5
    u_min = np.min(u, axis=1)
    u_max = np.max(u, axis=1)
    v_min = np.min(v, axis=1)
    v_max = np.max(v, axis=1)

    overlaps = (
        positive
        & (u_max >= -float(border_px))
        & (u_min <= float(width - 1) + float(border_px))
        & (v_max >= -float(border_px))
        & (v_min <= float(height - 1) + float(border_px))
    )
    return np.flatnonzero(overlaps).astype(np.int64)


def compact_colored_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.shape[0] == 0:
        raise RuntimeError("Cannot compact an empty mesh.")
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_verts = verts[unique_vids].astype(np.float32)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64)
    compact_colors = colors[unique_vids].astype(np.uint8)
    return compact_verts, compact_faces, compact_colors


def load_colored_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
    if colors.ndim != 2 or colors.shape[0] != verts.shape[0] or colors.shape[1] < 3:
        raise ValueError(f"Mesh has no RGB vertex colors: {path}")
    return verts, faces, colors[:, :3]


def write_colored_ascii_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors_uint8: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    if vertex_colors_uint8.shape != (vertices.shape[0], 3):
        raise ValueError(
            "vertex_colors_uint8 must have shape (V, 3); got "
            f"{vertex_colors_uint8.shape} for {vertices.shape[0]} vertices."
        )
    with path.open("w", encoding="utf-8") as file_obj:
        file_obj.write("ply\n")
        file_obj.write("format ascii 1.0\n")
        file_obj.write(f"element vertex {len(vertices)}\n")
        file_obj.write("property float x\n")
        file_obj.write("property float y\n")
        file_obj.write("property float z\n")
        file_obj.write("property uchar red\n")
        file_obj.write("property uchar green\n")
        file_obj.write("property uchar blue\n")
        file_obj.write(f"element face {len(faces)}\n")
        file_obj.write("property list uchar int vertex_indices\n")
        file_obj.write("end_header\n")
        for vertex, color in zip(vertices, vertex_colors_uint8.astype(np.uint8)):
            file_obj.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            file_obj.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def blender_camera_matrix_world(
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    rotation_camera_to_world = rotation_world_to_camera.T
    camera_center_world = -rotation_camera_to_world @ translation_world_to_camera
    opencv_to_blender = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rotation_blender_to_world = rotation_camera_to_world @ opencv_to_blender
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation_blender_to_world.astype(np.float32)
    matrix[:3, 3] = camera_center_world.astype(np.float32)
    return matrix


def camera_extrinsics_from_blender_matrix_world(
    camera_matrix_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(camera_matrix_world, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"camera_matrix_world must be 4x4; got {matrix.shape}.")
    opencv_to_blender = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rotation_blender_to_world = matrix[:3, :3]
    camera_center_world = matrix[:3, 3]
    rotation_camera_to_world = rotation_blender_to_world @ opencv_to_blender
    rotation_world_to_camera = rotation_camera_to_world.T.astype(np.float32)
    translation_world_to_camera = (
        -rotation_world_to_camera @ camera_center_world
    ).astype(np.float32)
    return rotation_world_to_camera, translation_world_to_camera


def load_output_camera_config(interaction_name: str) -> tuple[Path, dict[str, Any]] | None:
    config_path = (
        SCRIPT_DIR
        / "output"
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    if not config_path.exists():
        return None
    payload = load_json(config_path)
    views = payload.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"Reusable camera config has no views: {config_path}")
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"Reusable camera view {index} is malformed: {config_path}")
        matrix = np.asarray(view.get("camera_matrix_world"), dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Reusable camera view {index} has invalid camera_matrix_world "
                f"shape {matrix.shape}: {config_path}"
            )
        intrinsics = np.asarray(
            view.get("intrinsics", payload.get("intrinsics")),
            dtype=np.float32,
        )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Reusable camera view {index} has invalid intrinsics shape "
                f"{intrinsics.shape}: {config_path}"
            )
        if "width" not in view and "width" not in payload:
            raise ValueError(f"Reusable camera view {index} missing width: {config_path}")
        if "height" not in view and "height" not in payload:
            raise ValueError(f"Reusable camera view {index} missing height: {config_path}")
    return config_path, payload


def camera_center_from_extrinsics(
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (-rotation_world_to_camera.T @ translation_world_to_camera).astype(np.float32)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return (vector / norm).astype(np.float32)


def rotate_about_up(vector: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.deg2rad(float(degrees))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x, y, z = vector.astype(np.float32)
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float32)


def rotation_between_unit_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = normalize_vector(source)
    target = normalize_vector(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1e-7:
        return np.eye(3, dtype=np.float32)
    if dot < -1.0 + 1e-7:
        axis = np.cross(source, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(source, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        axis = normalize_vector(axis)
        x, y, z = axis
        return np.array(
            [
                [-1.0 + 2.0 * x * x, 2.0 * x * y, 2.0 * x * z],
                [2.0 * y * x, -1.0 + 2.0 * y * y, 2.0 * y * z],
                [2.0 * z * x, 2.0 * z * y, -1.0 + 2.0 * z * z],
            ],
            dtype=np.float32,
        )

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32)
        + skew
        + skew @ skew * (1.0 / (1.0 + dot))
    ).astype(np.float32)


def look_at_world_to_camera(
    camera_center: np.ndarray,
    focus: np.ndarray,
    world_up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    forward = normalize_vector(focus - camera_center)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalize_vector(right)
    down = normalize_vector(np.cross(forward, right))
    rotation_camera_to_world = np.stack([right, down, forward], axis=1).astype(np.float32)
    rotation_world_to_camera = rotation_camera_to_world.T.astype(np.float32)
    translation_world_to_camera = (
        -rotation_world_to_camera @ camera_center.astype(np.float32)
    ).astype(np.float32)
    return rotation_world_to_camera, translation_world_to_camera


def look_at_world_to_camera_centered(
    camera_center: np.ndarray,
    focus: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    image_height: int,
    world_up: np.ndarray = WORLD_UP,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_world_to_camera, _translation = look_at_world_to_camera(
        camera_center,
        focus,
        world_up=world_up,
    )
    rotation_camera_to_world = rotation_world_to_camera.T
    image_center_ray_camera = np.array(
        [
            (float(width) * 0.5 - float(intrinsics[0, 2])) / float(intrinsics[0, 0]),
            (float(image_height) * 0.5 - float(intrinsics[1, 2]))
            / float(intrinsics[1, 1]),
            1.0,
        ],
        dtype=np.float32,
    )
    camera_space_adjustment = rotation_between_unit_vectors(
        image_center_ray_camera,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    rotation_camera_to_world = (
        rotation_camera_to_world @ camera_space_adjustment
    ).astype(np.float32)
    rotation_world_to_camera = rotation_camera_to_world.T.astype(np.float32)
    translation_world_to_camera = (
        -rotation_world_to_camera @ camera_center.astype(np.float32)
    ).astype(np.float32)
    return rotation_world_to_camera, translation_world_to_camera


def load_mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    return np.asarray(mesh.vertices, dtype=np.float32)


def human_focus_point(human_vertices_world: np.ndarray) -> np.ndarray:
    vmin = human_vertices_world.min(axis=0)
    vmax = human_vertices_world.max(axis=0)
    center = (vmin + vmax) * 0.5
    height = float(vmax[2] - vmin[2])
    focus = center.copy()
    focus[2] = float(vmin[2] + 0.55 * height)
    return focus.astype(np.float32)


def interaction_focus_point(
    sig_payload: dict[str, Any],
    human_vertices_world: np.ndarray,
    contact_part_vertices_world: dict[str, np.ndarray],
) -> np.ndarray:
    body_focus = human_focus_point(human_vertices_world)
    if not contact_part_vertices_world:
        return body_focus

    preferred_parts = iter_interaction_human_parts_by_scene(
        sig_payload,
        scene_elements={"target_object"},
    )
    preferred_vertices = [
        contact_part_vertices_world[part_name]
        for part_name in preferred_parts
        if part_name in contact_part_vertices_world
    ]
    if not preferred_vertices:
        preferred_vertices = list(contact_part_vertices_world.values())
    contact_focus = np.concatenate(preferred_vertices, axis=0).mean(axis=0)
    focus = (0.58 * body_focus + 0.42 * contact_focus).astype(np.float32)
    vmin = human_vertices_world.min(axis=0)
    vmax = human_vertices_world.max(axis=0)
    focus[2] = float(np.clip(focus[2], vmin[2] + 0.22, vmax[2] - 0.12))
    return focus


def project_camera_points_to_image(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * points_camera[:, 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * points_camera[:, 1] / z_safe + intrinsics[1, 2] - 0.5
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def make_scene_depth_buffer(
    scene_depth_points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float,
    depth_width: int,
    depth_sample_spacing_m: float = 0.0,
    max_splat_radius_px: int = 0,
) -> tuple[np.ndarray, float]:
    scale = min(1.0, float(depth_width) / float(width))
    depth_h = max(1, int(round(float(height) * scale)))
    depth_w = max(1, int(round(float(width) * scale)))
    depth = np.full((depth_h, depth_w), np.inf, dtype=np.float32)

    points_camera = transform_world_to_camera(
        scene_depth_points_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    u, v, z = project_camera_points_to_image(points_camera, intrinsics)
    valid = (
        (z > 1e-5)
        & (z < float(max_depth_m))
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
        & np.isfinite(u)
        & np.isfinite(v)
    )
    if not np.any(valid):
        return depth, scale

    ui = np.clip(np.rint(u[valid] * scale).astype(np.int32), 0, depth_w - 1)
    vi = np.clip(np.rint(v[valid] * scale).astype(np.int32), 0, depth_h - 1)
    np.minimum.at(depth, (vi, ui), z[valid].astype(np.float32))

    finite = np.isfinite(depth)
    if np.any(finite) and depth_sample_spacing_m > 0.0 and max_splat_radius_px > 0:
        dense_depth = np.full_like(depth, np.inf)
        sentinel = np.float32(float(max_depth_m) + 1.0)
        focal = max(float(intrinsics[0, 0]), float(intrinsics[1, 1]))
        bin_edges = np.asarray(
            [
                0.01,
                0.35,
                0.55,
                0.85,
                1.25,
                1.75,
                2.50,
                3.50,
                5.00,
                7.50,
                10.00,
                float(max_depth_m),
            ],
            dtype=np.float32,
        )
        bin_edges = np.unique(
            np.clip(bin_edges, 0.01, float(max_depth_m)).astype(np.float32)
        )
        if bin_edges[-1] < float(max_depth_m):
            bin_edges = np.concatenate(
                [bin_edges, np.asarray([float(max_depth_m)], dtype=np.float32)]
            )
        for z_min, z_max in zip(bin_edges[:-1], bin_edges[1:]):
            bin_mask = finite & (depth >= float(z_min)) & (depth < float(z_max))
            if not np.any(bin_mask):
                continue
            radius = int(
                np.ceil(
                    float(depth_sample_spacing_m)
                    * focal
                    * scale
                    / max(float(z_min), 0.08)
                )
            )
            radius = int(np.clip(radius, 1, int(max_splat_radius_px)))
            kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
            depth_for_bin = np.where(bin_mask, depth, sentinel).astype(np.float32)
            eroded = cv2.erode(depth_for_bin, kernel)
            eroded[eroded >= sentinel] = np.inf
            dense_depth = np.minimum(dense_depth, eroded)
        if np.any(np.isfinite(dense_depth)):
            depth = dense_depth
            finite = np.isfinite(depth)

    if np.any(finite):
        filled = np.where(finite, depth, np.float32(max_depth_m))
        depth = cv2.erode(filled, np.ones((5, 5), dtype=np.uint8))
        depth[depth >= float(max_depth_m)] = np.inf

    return depth, scale


def point_visibility_fraction_from_depth(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    scene_depth: np.ndarray,
    depth_scale: float,
    depth_tolerance_m: float,
    require_scene_depth: bool,
    occluder_depth: np.ndarray | None = None,
    occluder_depth_tolerance_m: float = 0.0,
    require_occluder_depth: bool = False,
) -> float:
    if points_world.shape[0] == 0:
        return 1.0
    points_camera = transform_world_to_camera(
        points_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    u, v, z = project_camera_points_to_image(points_camera, intrinsics)
    valid_z = z > 1e-6
    in_frame = (
        valid_z
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )
    if not np.any(in_frame):
        return 0.0

    depth_h, depth_w = scene_depth.shape
    ui = np.clip(np.rint(u[in_frame] * depth_scale).astype(np.int32), 0, depth_w - 1)
    vi = np.clip(np.rint(v[in_frame] * depth_scale).astype(np.int32), 0, depth_h - 1)
    point_z = z[in_frame]
    scene_z = scene_depth[vi, ui]
    has_scene_depth = np.isfinite(scene_z)
    if require_scene_depth:
        visible = has_scene_depth & (point_z <= scene_z + float(depth_tolerance_m))
    else:
        visible = (~has_scene_depth) | (point_z <= scene_z + float(depth_tolerance_m))
    if occluder_depth is not None:
        occluder_z = occluder_depth[vi, ui]
        has_occluder_depth = np.isfinite(occluder_z)
        if require_occluder_depth:
            visible &= has_occluder_depth & (
                point_z <= occluder_z + float(occluder_depth_tolerance_m)
            )
        else:
            visible &= (~has_occluder_depth) | (
                point_z <= occluder_z + float(occluder_depth_tolerance_m)
            )
    return float(np.count_nonzero(visible) / max(points_world.shape[0], 1))


def projection_framing_metrics(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    if points_world.shape[0] == 0:
        return {
            "in_frame_fraction": 1.0,
            "center_offset": 0.0,
            "bbox_fill": 0.0,
        }

    points_camera = transform_world_to_camera(
        points_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    u, v, z = project_camera_points_to_image(points_camera, intrinsics)
    valid_z = z > 1e-6
    in_frame = (
        valid_z
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )
    in_frame_fraction = float(np.count_nonzero(in_frame) / points_world.shape[0])
    if not np.any(valid_z):
        return {
            "in_frame_fraction": in_frame_fraction,
            "center_offset": float("inf"),
            "bbox_fill": 0.0,
        }

    u_valid = u[valid_z]
    v_valid = v[valid_z]
    u_min = float(np.min(u_valid))
    u_max = float(np.max(u_valid))
    v_min = float(np.min(v_valid))
    v_max = float(np.max(v_valid))
    bbox_center_x = 0.5 * (u_min + u_max)
    bbox_center_y = 0.5 * (v_min + v_max)
    image_center_x = 0.5 * float(width - 1)
    image_center_y = 0.5 * float(height - 1)
    center_offset = float(
        np.sqrt(
            ((bbox_center_x - image_center_x) / float(width)) ** 2
            + ((bbox_center_y - image_center_y) / float(height)) ** 2
        )
    )
    bbox_fill = float(
        max(
            (u_max - u_min) / max(float(width), 1.0),
            (v_max - v_min) / max(float(height), 1.0),
        )
    )
    return {
        "in_frame_fraction": in_frame_fraction,
        "center_offset": center_offset,
        "bbox_fill": bbox_fill,
    }


def contact_parts_covered_by_view(
    view: dict[str, Any],
    min_contact_part_visible_fraction: float,
) -> list[str]:
    fractions = view.get("contact_part_visible_fractions", {})
    if not isinstance(fractions, dict):
        return []
    return sorted(
        str(part)
        for part, fraction in fractions.items()
        if float(fraction) >= float(min_contact_part_visible_fraction)
    )


def estimate_full_body_camera_radius(
    human_vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    fallback_radius_m: float,
) -> float:
    vmin = human_vertices_world.min(axis=0)
    vmax = human_vertices_world.max(axis=0)
    human_height = max(float(vmax[2] - vmin[2]), 0.25)
    human_xy_span = max(float(vmax[0] - vmin[0]), float(vmax[1] - vmin[1]), 0.25)
    fx = max(float(intrinsics[0, 0]), 1e-6)
    fy = max(float(intrinsics[1, 1]), 1e-6)
    horizontal_fov = 2.0 * float(np.arctan(float(width) / (2.0 * fx)))
    vertical_fov = 2.0 * float(np.arctan(float(height) / (2.0 * fy)))
    target_fill = 0.68
    radius_for_height = human_height / (
        2.0 * max(np.tan(vertical_fov * 0.5), 1e-6) * target_fill
    )
    radius_for_width = human_xy_span / (
        2.0 * max(np.tan(horizontal_fov * 0.5), 1e-6) * target_fill
    )
    return float(
        np.clip(
            max(float(fallback_radius_m), radius_for_height, radius_for_width),
            0.9,
            3.5,
        )
    )


def evaluate_render_view(
    label: str,
    yaw_deg: float,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    focus: np.ndarray,
    human_vertices_world: np.ndarray,
    contact_part_vertices_world: dict[str, np.ndarray],
    scene_depth_points_world: np.ndarray,
    scene_vertex_tree: cKDTree,
    intrinsics: np.ndarray,
    width: int,
    image_height: int,
    min_camera_scene_distance_m: float,
    min_human_visible_fraction: float,
    min_human_in_frame_fraction: float,
    max_human_center_offset: float,
    min_human_bbox_fill: float,
    max_human_bbox_fill: float,
    min_interaction_part_visible_fraction: float,
    min_contact_part_visible_fraction: float,
    visibility_depth_width: int,
    visibility_depth_tolerance_m: float,
    scene_depth_sample_spacing_m: float,
    scene_depth_max_splat_radius_px: int,
    human_depth_sample_spacing_m: float,
    human_depth_max_splat_radius_px: int,
    max_depth_m: float,
) -> dict[str, Any]:
    camera_center = camera_center_from_extrinsics(
        rotation_world_to_camera,
        translation_world_to_camera,
    )
    nearest_distance = float(scene_vertex_tree.query(camera_center)[0])
    scene_depth, depth_scale = make_scene_depth_buffer(
        scene_depth_points_world=scene_depth_points_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        intrinsics=intrinsics,
        width=width,
        height=image_height,
        max_depth_m=max_depth_m,
        depth_width=visibility_depth_width,
        depth_sample_spacing_m=scene_depth_sample_spacing_m,
        max_splat_radius_px=scene_depth_max_splat_radius_px,
    )
    scene_depth_finite_fraction = float(
        np.count_nonzero(np.isfinite(scene_depth)) / max(scene_depth.size, 1)
    )
    human_depth, _ = make_scene_depth_buffer(
        scene_depth_points_world=human_vertices_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        intrinsics=intrinsics,
        width=width,
        height=image_height,
        max_depth_m=max_depth_m,
        depth_width=visibility_depth_width,
        depth_sample_spacing_m=human_depth_sample_spacing_m,
        max_splat_radius_px=human_depth_max_splat_radius_px,
    )
    human_visible_fraction = point_visibility_fraction_from_depth(
        human_vertices_world,
        rotation_world_to_camera,
        translation_world_to_camera,
        intrinsics,
        width,
        image_height,
        scene_depth=scene_depth,
        depth_scale=depth_scale,
        depth_tolerance_m=visibility_depth_tolerance_m,
        require_scene_depth=False,
    )
    framing_metrics = projection_framing_metrics(
        human_vertices_world,
        rotation_world_to_camera,
        translation_world_to_camera,
        intrinsics,
        width,
        image_height,
    )
    contact_part_visible_fractions: dict[str, float] = {}
    contact_part_in_frame_fractions: dict[str, float] = {}
    contact_part_scene_visible_fractions: dict[str, float] = {}
    contact_part_self_visible_fractions: dict[str, float] = {}
    for part_name, vertices in contact_part_vertices_world.items():
        part_framing_metrics = projection_framing_metrics(
            vertices,
            rotation_world_to_camera,
            translation_world_to_camera,
            intrinsics,
            width,
            image_height,
        )
        part_in_frame_fraction = float(part_framing_metrics["in_frame_fraction"])
        scene_visible_fraction = point_visibility_fraction_from_depth(
            vertices,
            rotation_world_to_camera,
            translation_world_to_camera,
            intrinsics,
            width,
            image_height,
            scene_depth=scene_depth,
            depth_scale=depth_scale,
            depth_tolerance_m=visibility_depth_tolerance_m,
            require_scene_depth=False,
        )
        self_visible_fraction = point_visibility_fraction_from_depth(
            vertices,
            rotation_world_to_camera,
            translation_world_to_camera,
            intrinsics,
            width,
            image_height,
            scene_depth=scene_depth,
            depth_scale=depth_scale,
            depth_tolerance_m=visibility_depth_tolerance_m,
            require_scene_depth=False,
            occluder_depth=human_depth,
            occluder_depth_tolerance_m=visibility_depth_tolerance_m,
            require_occluder_depth=True,
        )
        contact_part_in_frame_fractions[part_name] = part_in_frame_fraction
        contact_part_scene_visible_fractions[part_name] = scene_visible_fraction
        contact_part_self_visible_fractions[part_name] = self_visible_fraction
        contact_part_visible_fractions[part_name] = min(
            part_in_frame_fraction,
            max(
                self_visible_fraction,
                scene_visible_fraction,
                0.35 * part_in_frame_fraction,
            ),
        )
    if contact_part_visible_fractions:
        interaction_part_visible_fraction = float(
            np.mean(list(contact_part_visible_fractions.values()))
        )
    else:
        interaction_part_visible_fraction = 1.0
    covered_contact_parts = sorted(
        part_name
        for part_name, fraction in contact_part_visible_fractions.items()
        if fraction >= float(min_contact_part_visible_fraction)
    )

    valid = (
        nearest_distance >= float(min_camera_scene_distance_m)
        and human_visible_fraction >= float(min_human_visible_fraction)
        and framing_metrics["in_frame_fraction"] >= float(min_human_in_frame_fraction)
        and framing_metrics["center_offset"] <= float(max_human_center_offset)
        and framing_metrics["bbox_fill"] >= float(min_human_bbox_fill)
        and framing_metrics["bbox_fill"] <= float(max_human_bbox_fill)
        and interaction_part_visible_fraction
        >= float(min_interaction_part_visible_fraction)
    )
    renderable = (
        human_visible_fraction >= 0.50
        and framing_metrics["in_frame_fraction"] >= 0.75
        and framing_metrics["bbox_fill"] >= 0.20
        and framing_metrics["center_offset"] <= 0.45
    )
    relaxed_valid = (
        nearest_distance >= max(0.05, float(min_camera_scene_distance_m) - 0.05)
        and human_visible_fraction >= max(0.55, float(min_human_visible_fraction) - 0.15)
        and framing_metrics["in_frame_fraction"]
        >= max(0.75, float(min_human_in_frame_fraction) - 0.12)
        and framing_metrics["center_offset"] <= float(max_human_center_offset) + 0.12
        and framing_metrics["bbox_fill"] >= max(0.20, float(min_human_bbox_fill) * 0.70)
        and framing_metrics["bbox_fill"] <= float(max_human_bbox_fill) + 0.25
        and interaction_part_visible_fraction
        >= max(0.12, float(min_interaction_part_visible_fraction) * 0.50)
        and renderable
    )
    center_quality = 1.0 - min(
        framing_metrics["center_offset"] / max(float(max_human_center_offset), 1e-6),
        1.0,
    )
    target_fill = 0.68
    fill_quality = 1.0 - min(
        abs(framing_metrics["bbox_fill"] - target_fill) / max(target_fill, 1e-6),
        1.0,
    )
    contact_quality = (
        float(
            np.mean(
                [
                    min(fraction / max(float(min_contact_part_visible_fraction), 1e-6), 1.0)
                    for fraction in contact_part_visible_fractions.values()
                ]
            )
        )
        if contact_part_visible_fractions
        else 1.0
    )
    quality = (
        0.28 * human_visible_fraction
        + 0.22 * framing_metrics["in_frame_fraction"]
        + 0.18 * center_quality
        + 0.12 * fill_quality
        + 0.20 * contact_quality
    )
    return {
        "label": label,
        "yaw_deg": float(yaw_deg),
        "camera_radius_m": float(np.linalg.norm(camera_center - focus)),
        "camera_height_offset_m": float(camera_center[2] - focus[2]),
        "camera_center_world": camera_center.astype(np.float32),
        "rotation_world_to_camera": rotation_world_to_camera.astype(np.float32),
        "translation_world_to_camera": translation_world_to_camera.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
        "width": int(width),
        "height": int(image_height),
        "nearest_scene_distance_m": nearest_distance,
        "human_visible_fraction": human_visible_fraction,
        "human_in_frame_fraction": float(framing_metrics["in_frame_fraction"]),
        "human_center_offset": float(framing_metrics["center_offset"]),
        "human_bbox_fill": float(framing_metrics["bbox_fill"]),
        "scene_depth_finite_fraction": scene_depth_finite_fraction,
        "interaction_part_visible_fraction": interaction_part_visible_fraction,
        "interaction_part_filter_available": bool(contact_part_visible_fractions),
        "contact_part_visible_fractions": contact_part_visible_fractions,
        "contact_part_in_frame_fractions": contact_part_in_frame_fractions,
        "contact_part_scene_visible_fractions": contact_part_scene_visible_fractions,
        "contact_part_self_visible_fractions": contact_part_self_visible_fractions,
        "covered_contact_parts": covered_contact_parts,
        "quality": float(quality),
        "renderable": bool(renderable),
        "valid": bool(valid),
        "relaxed_valid": bool(relaxed_valid),
    }


def select_synthetic_views_for_coverage(
    candidates: list[dict[str, Any]],
    initial_views: list[dict[str, Any]],
    focus: np.ndarray,
    min_num_views: int,
    max_num_views: int,
    contact_part_names: list[str],
    min_contact_part_visible_fraction: float,
    min_view_angular_separation_deg: float,
    allow_relaxed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_parts = set(contact_part_names)
    min_view_angle_rad = np.deg2rad(float(min_view_angular_separation_deg))
    min_count = max(0, int(min_num_views))
    max_count = max(min_count, int(max_num_views))

    def greedy(allowed_candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
        selected: list[dict[str, Any]] = []
        covered = set()
        for view in initial_views:
            covered.update(
                contact_parts_covered_by_view(view, min_contact_part_visible_fraction)
            )
        available = list(allowed_candidates)
        selected_dirs = [
            normalize_vector(view["camera_center_world"] - focus)
            for view in initial_views
            if float(np.linalg.norm(view["camera_center_world"] - focus)) > 1e-6
        ]
        while len(selected) < max_count and available:
            uncovered = target_parts - covered
            if len(selected) >= min_count and not uncovered:
                break

            def candidate_score(candidate: dict[str, Any]) -> float:
                candidate_covered = set(
                    contact_parts_covered_by_view(
                        candidate,
                        min_contact_part_visible_fraction,
                    )
                )
                new_parts = candidate_covered & uncovered
                fractions = candidate.get("contact_part_visible_fractions", {})
                uncovered_fraction_sum = sum(
                    float(fractions.get(part, 0.0)) for part in uncovered
                )
                direction = normalize_vector(candidate["camera_center_world"] - focus)
                if selected_dirs:
                    diversity = min(
                        float(
                            np.arccos(
                                np.clip(np.dot(direction, selected_dir), -1.0, 1.0)
                            )
                        )
                        for selected_dir in selected_dirs
                    )
                else:
                    diversity = float(np.pi)
                duplicate_penalty = (
                    50.0
                    if selected_dirs
                    and diversity < min_view_angle_rad
                    and not new_parts
                    else 0.0
                )
                return (
                    100.0 * len(new_parts)
                    + 28.0 * uncovered_fraction_sum
                    + 14.0 * float(candidate["interaction_part_visible_fraction"])
                    + 6.0 * float(candidate["quality"])
                    + 2.5 * diversity
                    - duplicate_penalty
                )

            best = max(available, key=candidate_score)
            selected.append(best)
            covered.update(
                contact_parts_covered_by_view(best, min_contact_part_visible_fraction)
            )
            selected_dirs.append(normalize_vector(best["camera_center_world"] - focus))
            available = [candidate for candidate in available if candidate is not best]
        return selected, covered

    strict_candidates = [
        candidate
        for candidate in candidates
        if candidate["valid"] and candidate["renderable"]
    ]
    selected, covered = greedy(strict_candidates)
    coverage_relaxed = False
    if (len(selected) < min_count or target_parts - covered) and allow_relaxed:
        relaxed_candidates = [
            candidate
            for candidate in candidates
            if candidate["relaxed_valid"] and candidate["renderable"]
        ]
        relaxed_selected, relaxed_covered = greedy(relaxed_candidates)
        if (
            len(relaxed_selected) >= len(selected)
            and len(target_parts - relaxed_covered) <= len(target_parts - covered)
        ):
            selected = relaxed_selected
            covered = relaxed_covered
            coverage_relaxed = True

    if len(selected) < min_count:
        selected_ids = {id(view) for view in selected}
        selected_dirs = [
            normalize_vector(view["camera_center_world"] - focus)
            for view in [*initial_views, *selected]
            if float(np.linalg.norm(view["camera_center_world"] - focus)) > 1e-6
        ]
        fallback_candidates = sorted(
            candidates,
            key=lambda candidate: (
                bool(candidate["renderable"]),
                bool(candidate["valid"]),
                bool(candidate["relaxed_valid"]),
                float(candidate["interaction_part_visible_fraction"]),
                float(candidate["quality"]),
            ),
            reverse=True,
        )
        for require_diverse in (True, False):
            for candidate in fallback_candidates:
                if len(selected) >= min_count:
                    break
                if id(candidate) in selected_ids or not bool(candidate["renderable"]):
                    continue
                direction = normalize_vector(candidate["camera_center_world"] - focus)
                if require_diverse and selected_dirs:
                    diversity = min(
                        float(
                            np.arccos(
                                np.clip(
                                    np.dot(direction, selected_dir),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                        for selected_dir in selected_dirs
                    )
                    if diversity < min_view_angle_rad:
                        continue
                selected.append(candidate)
                selected_ids.add(id(candidate))
                selected_dirs.append(direction)
                covered.update(
                    contact_parts_covered_by_view(
                        candidate,
                        min_contact_part_visible_fraction,
                    )
                )
            if len(selected) >= min_count:
                break

    selected = selected[:max_count]
    covered = set()
    for view in [*initial_views, *selected]:
        covered.update(contact_parts_covered_by_view(view, min_contact_part_visible_fraction))
    return selected, {
        "covered_contact_parts": sorted(covered & target_parts),
        "uncovered_contact_parts": sorted(target_parts - covered),
        "coverage_relaxed": bool(coverage_relaxed),
        "requested_min_synthetic_views": int(min_count),
        "requested_max_synthetic_views": int(max_count),
        "selected_synthetic_views": int(len(selected)),
    }


def build_candidate_views(
    original_rotation_world_to_camera: np.ndarray,
    original_translation_world_to_camera: np.ndarray,
    focus: np.ndarray,
    human_vertices_world: np.ndarray,
    contact_part_vertices_world: dict[str, np.ndarray],
    initial_views: list[dict[str, Any]],
    scene_depth_points_world: np.ndarray,
    scene_vertex_tree: cKDTree,
    intrinsics: np.ndarray,
    width: int,
    image_height: int,
    min_num_views: int,
    max_num_views: int,
    min_camera_scene_distance_m: float,
    min_human_visible_fraction: float,
    min_human_in_frame_fraction: float,
    max_human_center_offset: float,
    min_human_bbox_fill: float,
    max_human_bbox_fill: float,
    min_interaction_part_visible_fraction: float,
    min_contact_part_visible_fraction: float,
    camera_radius_m: float,
    visibility_depth_width: int,
    visibility_depth_tolerance_m: float,
    scene_depth_sample_spacing_m: float,
    scene_depth_max_splat_radius_px: int,
    human_depth_sample_spacing_m: float,
    human_depth_max_splat_radius_px: int,
    min_view_angular_separation_deg: float,
    max_depth_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original_center = camera_center_from_extrinsics(
        original_rotation_world_to_camera,
        original_translation_world_to_camera,
    )
    base_vector = original_center - focus
    base_vector_xy = base_vector.copy()
    base_vector_xy[2] = 0.0
    if float(np.linalg.norm(base_vector_xy)) < 1e-6:
        base_vector_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    base_radius = estimate_full_body_camera_radius(
        human_vertices_world=human_vertices_world,
        intrinsics=intrinsics,
        width=width,
        height=image_height,
        fallback_radius_m=float(camera_radius_m),
    )
    camera_height_offset = float(original_center[2] - focus[2])
    camera_height_offset = float(np.clip(camera_height_offset, 0.35, 1.15))
    base_dir = normalize_vector(base_vector_xy)

    candidates: list[dict[str, Any]] = []
    yaw_offsets = [-75.0, -45.0, -20.0, 20.0, 45.0, 75.0, 110.0, -110.0, 180.0]
    relaxed_yaw_offsets = [-140.0, -125.0, 0.0, 125.0, 140.0]
    radii = [base_radius * scale for scale in (0.95, 1.1, 1.25)]
    relaxed_radii = [base_radius * scale for scale in (0.85, 1.4)]
    height_offsets = [
        camera_height_offset - 0.20,
        camera_height_offset,
        camera_height_offset + 0.25,
    ]
    relaxed_height_offsets = [0.20, camera_height_offset + 0.45, 1.45]
    candidate_specs = [
        (yaw, radius, height_offset, False)
        for radius in radii
        for yaw in yaw_offsets
        for height_offset in height_offsets
    ] + [
        (yaw, radius, height_offset, True)
        for radius in relaxed_radii
        for yaw in relaxed_yaw_offsets
        for height_offset in relaxed_height_offsets
    ]
    seen_specs: set[tuple[int, int, int]] = set()
    for yaw, radius, height_offset, is_relaxed_camera in candidate_specs:
        key = (
            int(round(yaw * 10.0)),
            int(round(radius * 100.0)),
            int(round(height_offset * 100.0)),
        )
        if key in seen_specs:
            continue
        seen_specs.add(key)
        direction = normalize_vector(rotate_about_up(base_dir, float(yaw)))
        center = focus + direction * float(radius)
        center[2] = focus[2] + float(np.clip(height_offset, 0.15, 1.55))
        rotation, translation = look_at_world_to_camera_centered(
            center,
            focus,
            intrinsics=intrinsics,
            width=width,
            image_height=image_height,
        )
        view = evaluate_render_view(
            label=(
                "relaxed_yaw"
                if is_relaxed_camera
                else "view_axis_radius"
                if yaw == 0.0
                else f"yaw_{int(yaw):+d}"
            ),
            yaw_deg=float(yaw),
            rotation_world_to_camera=rotation,
            translation_world_to_camera=translation,
            focus=focus,
            human_vertices_world=human_vertices_world,
            contact_part_vertices_world=contact_part_vertices_world,
            scene_depth_points_world=scene_depth_points_world,
            scene_vertex_tree=scene_vertex_tree,
            intrinsics=intrinsics,
            width=width,
            image_height=image_height,
            min_camera_scene_distance_m=min_camera_scene_distance_m,
            min_human_visible_fraction=min_human_visible_fraction,
            min_human_in_frame_fraction=min_human_in_frame_fraction,
            max_human_center_offset=max_human_center_offset,
            min_human_bbox_fill=min_human_bbox_fill,
            max_human_bbox_fill=max_human_bbox_fill,
            min_interaction_part_visible_fraction=min_interaction_part_visible_fraction,
            min_contact_part_visible_fraction=min_contact_part_visible_fraction,
            visibility_depth_width=visibility_depth_width,
            visibility_depth_tolerance_m=visibility_depth_tolerance_m,
            scene_depth_sample_spacing_m=scene_depth_sample_spacing_m,
            scene_depth_max_splat_radius_px=scene_depth_max_splat_radius_px,
            human_depth_sample_spacing_m=human_depth_sample_spacing_m,
            human_depth_max_splat_radius_px=human_depth_max_splat_radius_px,
            max_depth_m=max_depth_m,
        )
        view["relaxed_camera_candidate"] = bool(is_relaxed_camera)
        candidates.append(view)

    return select_synthetic_views_for_coverage(
        candidates=candidates,
        initial_views=initial_views,
        focus=focus,
        min_num_views=int(min_num_views),
        max_num_views=int(max_num_views),
        contact_part_names=list(contact_part_vertices_world.keys()),
        min_contact_part_visible_fraction=min_contact_part_visible_fraction,
        min_view_angular_separation_deg=min_view_angular_separation_deg,
        allow_relaxed=True,
    )


def write_blender_driver(path: Path) -> None:
    path.write_text(
        r'''
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def import_ply(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    after = set(bpy.context.scene.objects)
    new_objects = list(after - before)
    if not new_objects:
        raise RuntimeError(f"Failed to import PLY: {path}")
    return new_objects[0]


def assign_vertex_color_material(obj):
    mesh = obj.data
    mat = bpy.data.materials.new(name=f"{obj.name}_vertex_color")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr = nodes.new(type="ShaderNodeAttribute")
    if getattr(mesh, "color_attributes", None) and len(mesh.color_attributes) > 0:
        attr.attribute_name = mesh.color_attributes[0].name
    else:
        attr.attribute_name = "Col"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def assign_human_material(obj):
    mat = bpy.data.materials.new(name="human_soft_blue")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.45, 0.62, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.55
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def configure_cycles_gpu(samples):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = int(samples)
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 3
    bpy.context.scene.cycles.glossy_bounces = 3
    bpy.context.scene.cycles.transparent_max_bounces = 4

    prefs = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [device for device in prefs.devices if device.type != "CPU"]
            if gpu_devices:
                selected_backend = backend
                break
        except Exception as exc:
            print(f"Cycles GPU backend {backend} unavailable: {exc}")

    if selected_backend is None:
        bpy.context.scene.cycles.device = "CPU"
        print("Cycles GPU device unavailable; falling back to CPU")
        return

    for device in prefs.devices:
        device.use = device.type != "CPU"
    enabled = [
        f"{device.name} ({device.type})"
        for device in prefs.devices
        if device.use
    ]
    print(f"Cycles GPU backend: {selected_backend}")
    print(f"Cycles GPU devices: {enabled}")


def object_bounds_center_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = Vector((0.0, 0.0, 0.0))
    for corner in corners:
        center += corner
    return center / max(len(corners), 1)


def aim_object_at(obj, target):
    direction = Vector(target) - obj.location
    if direction.length < 1e-6:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_shadowless_area_light(name, location, target, energy, size):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(energy)
    light_data.size = float(size)
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = False
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    aim_object_at(light_obj, target)
    return light_obj


def configure_soft_room_lighting(human_obj, camera_objects):
    focus = object_bounds_center_world(human_obj)
    focus.z += 0.65

    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
        background.inputs["Strength"].default_value = 0.25

    add_shadowless_area_light(
        "room_overhead_softbox",
        (focus.x, focus.y, focus.z + 2.4),
        focus,
        energy=100.0,
        size=4.0,
    )
    print("Lighting: low world fill + shadowless room area light")


argv = sys.argv
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene_obj = import_ply(config["scene_crop_ply"])
scene_obj.name = "colored_scannet_crop"
assign_vertex_color_material(scene_obj)

human_obj = import_ply(config["human_mesh_world"])
human_obj.name = "optimized_human"
assign_human_material(human_obj)

default_width = int(config["width"])
default_height = int(config["height"])
default_intrinsics = config["intrinsics"]
sensor_width = 36.0
camera_objects = []
for view in config["views"]:
    width = int(view.get("width", default_width))
    height = int(view.get("height", default_height))
    intrinsics = view.get("intrinsics", default_intrinsics)
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    camera_data = bpy.data.cameras.new(view["name"])
    camera_obj = bpy.data.objects.new(view["name"], camera_data)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.matrix_world = Matrix(view["camera_matrix_world"])
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = sensor_width
    camera_data.lens = fx * sensor_width / float(width)
    camera_data.shift_x = (float(width) * 0.5 - cx) / float(width)
    camera_data.shift_y = (cy - float(height) * 0.5) / float(width)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera_objects.append((camera_obj, view["render_path"], width, height))

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = bpy.data.worlds.new("world") if bpy.context.scene.world is None else bpy.context.scene.world
configure_soft_room_lighting(human_obj, camera_objects)
bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])
for camera_obj, render_path, width, height in camera_objects:
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = int(width)
    bpy.context.scene.render.resolution_y = int(height)
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
'''.lstrip(),
        encoding="utf-8",
    )


def render_interaction(
    interaction_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name, args.output_mode)
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    smpl_seg_json_path = resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    human_mesh_world_path = resolve_path(args.human_mesh_world, defaults["human_mesh_world"])
    contact_spec_json_path = resolve_path(
        args.contact_spec_json,
        defaults["contact_spec_json"],
    )
    contact_render_image_path = resolve_path(
        args.contact_render_image,
        defaults["contact_render_image"],
    )
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    scannet_root = resolve_scannet_root(args.scannet_root)

    if not human_mesh_world_path.exists():
        raise FileNotFoundError(f"Optimized human world mesh not found: {human_mesh_world_path}")
    if not contact_spec_json_path.exists():
        raise FileNotFoundError(f"Contact spec JSON not found: {contact_spec_json_path}")

    input_payload = load_json(input_scene_json_path)
    sig_payload = load_json(sig_json_path)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)
    (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    ) = load_scannet_camera(scene_paths, scene_context)
    contact_intrinsics = load_contact_spec_intrinsics(contact_spec_json_path)
    contact_width, contact_height = load_image_size(contact_render_image_path)

    print(f"Loading optimized human mesh from: {human_mesh_world_path}")
    human_vertices_world = load_mesh_vertices(human_mesh_world_path)
    smpl_vertex_count, smpl_segments, contact_segment_ids = load_smpl_body_segments(
        smpl_seg_json_path
    )
    if human_vertices_world.shape[0] != smpl_vertex_count:
        raise ValueError(
            "Optimized human mesh vertex count does not match SMPL-X segmentation: "
            f"mesh={human_vertices_world.shape[0]} segmentation={smpl_vertex_count}"
        )
    interaction_part_vertex_ids, interaction_part_metadata = (
        build_interaction_part_vertex_ids(
            sig_payload=sig_payload,
            smpl_segments=smpl_segments,
            contact_segment_ids=contact_segment_ids,
        )
    )
    contact_part_vertices_world = {
        part_name: human_vertices_world[vertex_ids].astype(np.float32)
        for part_name, vertex_ids in interaction_part_vertex_ids.items()
    }
    focus_world = interaction_focus_point(
        sig_payload=sig_payload,
        human_vertices_world=human_vertices_world,
        contact_part_vertices_world=contact_part_vertices_world,
    )
    print(
        "Interaction human-part visibility vertices: "
        f"{interaction_part_metadata.get('vertex_count', 0)}"
    )

    print(f"Loading colored ScanNet mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces, scene_colors = load_colored_mesh(
        scene_paths["mesh_path"]
    )
    scene_depth_points_world = np.concatenate(
        [
            scene_verts_world,
            scene_verts_world[scene_faces].mean(axis=1).astype(np.float32),
        ],
        axis=0,
    )
    if scene_depth_points_world.shape[0] > int(args.max_scene_depth_points):
        rng = np.random.default_rng(int(args.seed))
        choice = rng.choice(
            scene_depth_points_world.shape[0],
            size=int(args.max_scene_depth_points),
            replace=False,
        )
        scene_depth_points_world = scene_depth_points_world[choice]
    scene_vertex_tree = cKDTree(scene_verts_world)
    reusable_camera_config = (
        load_output_camera_config(interaction_name)
        if bool(args.reuse_output_cameras)
        else None
    )
    camera_source = "generated"
    camera_source_config_path: str | None = None

    if reusable_camera_config is not None:
        camera_config_path, camera_config_payload = reusable_camera_config
        print(f"Reusing semantic cameras from: {camera_config_path}")
        camera_source = "baseline_output"
        camera_source_config_path = str(camera_config_path.resolve())
        source_selected_views_path = camera_config_path.with_name("selected_views.json")
        source_selected_views = (
            load_json(source_selected_views_path)
            if source_selected_views_path.exists()
            else {}
        )
        source_view_metadata = {
            str(view.get("name")): view
            for view in source_selected_views.get("views", [])
            if isinstance(view, dict) and view.get("name") is not None
        }
        selected_views = []
        covered_contact_parts: set[str] = set()
        for index, render_view in enumerate(camera_config_payload["views"]):
            render_name = str(
                render_view.get("name")
                or Path(str(render_view.get("render_path", ""))).stem
                or f"view_{index:02d}"
            )
            camera_matrix_world_payload = render_view["camera_matrix_world"]
            camera_matrix_world = np.asarray(camera_matrix_world_payload, dtype=np.float32)
            view_intrinsics = np.asarray(
                render_view.get("intrinsics", camera_config_payload.get("intrinsics")),
                dtype=np.float32,
            )
            view_width = int(render_view.get("width", camera_config_payload.get("width")))
            view_height = int(
                render_view.get("height", camera_config_payload.get("height"))
            )
            view_rotation, view_translation = camera_extrinsics_from_blender_matrix_world(
                camera_matrix_world
            )
            source_metadata = source_view_metadata.get(render_name, {})
            view = evaluate_render_view(
                label=str(source_metadata.get("source_label", "output_camera")),
                yaw_deg=float(source_metadata.get("yaw_deg", 0.0)),
                rotation_world_to_camera=view_rotation,
                translation_world_to_camera=view_translation,
                focus=focus_world,
                human_vertices_world=human_vertices_world,
                contact_part_vertices_world=contact_part_vertices_world,
                scene_depth_points_world=scene_depth_points_world,
                scene_vertex_tree=scene_vertex_tree,
                intrinsics=view_intrinsics,
                width=view_width,
                image_height=view_height,
                min_camera_scene_distance_m=float(args.min_camera_scene_distance_m),
                min_human_visible_fraction=float(args.min_human_visible_fraction),
                min_human_in_frame_fraction=float(args.min_human_in_frame_fraction),
                max_human_center_offset=float(args.max_human_center_offset),
                min_human_bbox_fill=float(args.min_human_bbox_fill),
                max_human_bbox_fill=float(args.max_human_bbox_fill),
                min_interaction_part_visible_fraction=float(
                    args.min_interaction_part_visible_fraction
                ),
                min_contact_part_visible_fraction=float(
                    args.min_contact_part_visible_fraction
                ),
                visibility_depth_width=int(args.visibility_depth_width),
                visibility_depth_tolerance_m=float(args.visibility_depth_tolerance_m),
                scene_depth_sample_spacing_m=float(args.scene_depth_sample_spacing_m),
                scene_depth_max_splat_radius_px=int(args.scene_depth_max_splat_radius_px),
                human_depth_sample_spacing_m=float(args.human_depth_sample_spacing_m),
                human_depth_max_splat_radius_px=int(args.human_depth_max_splat_radius_px),
                max_depth_m=float(args.max_depth_m),
            )
            view["render_name"] = render_name
            view["camera_matrix_world"] = camera_matrix_world_payload
            view["reused_camera_config_path"] = camera_source_config_path
            selected_views.append(view)
            covered_contact_parts.update(
                contact_parts_covered_by_view(
                    view,
                    float(args.min_contact_part_visible_fraction),
                )
            )

        synthetic_views = selected_views[1:]
        min_total_views = len(selected_views)
        max_total_views = len(selected_views)
        target_parts = set(contact_part_vertices_world)
        coverage_metadata = {
            "covered_contact_parts": sorted(covered_contact_parts & target_parts),
            "uncovered_contact_parts": sorted(target_parts - covered_contact_parts),
            "coverage_relaxed": False,
            "requested_min_synthetic_views": int(max(0, len(selected_views) - 1)),
            "requested_max_synthetic_views": int(max(0, len(selected_views) - 1)),
            "selected_synthetic_views": int(max(0, len(selected_views) - 1)),
        }
    else:
        contact_view = evaluate_render_view(
            label="contact_spec_camera",
            yaw_deg=0.0,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
            focus=focus_world,
            human_vertices_world=human_vertices_world,
            contact_part_vertices_world=contact_part_vertices_world,
            scene_depth_points_world=scene_depth_points_world,
            scene_vertex_tree=scene_vertex_tree,
            intrinsics=contact_intrinsics,
            width=contact_width,
            image_height=contact_height,
            min_camera_scene_distance_m=float(args.min_camera_scene_distance_m),
            min_human_visible_fraction=float(args.min_human_visible_fraction),
            min_human_in_frame_fraction=float(args.min_human_in_frame_fraction),
            max_human_center_offset=float(args.max_human_center_offset),
            min_human_bbox_fill=float(args.min_human_bbox_fill),
            max_human_bbox_fill=float(args.max_human_bbox_fill),
            min_interaction_part_visible_fraction=float(
                args.min_interaction_part_visible_fraction
            ),
            min_contact_part_visible_fraction=float(args.min_contact_part_visible_fraction),
            visibility_depth_width=int(args.visibility_depth_width),
            visibility_depth_tolerance_m=float(args.visibility_depth_tolerance_m),
            scene_depth_sample_spacing_m=float(args.scene_depth_sample_spacing_m),
            scene_depth_max_splat_radius_px=int(args.scene_depth_max_splat_radius_px),
            human_depth_sample_spacing_m=float(args.human_depth_sample_spacing_m),
            human_depth_max_splat_radius_px=int(args.human_depth_max_splat_radius_px),
            max_depth_m=float(args.max_depth_m),
        )
        contact_view["render_name"] = "view_00"
        contact_view["contact_spec_path"] = str(contact_spec_json_path)
        contact_view["reference_image_path"] = str(contact_render_image_path)
        initial_views = [contact_view]
        if args.num_views is not None:
            max_total_views = 1 + max(0, int(args.num_views))
        else:
            max_total_views = max(1, int(args.max_views))
        min_total_views = max(1, min(int(args.min_views), max_total_views))
        min_synthetic_views = max(0, min_total_views - 1)
        max_synthetic_views = max(0, max_total_views - 1)

        synthetic_views, coverage_metadata = build_candidate_views(
            original_rotation_world_to_camera=rotation_world_to_camera,
            original_translation_world_to_camera=translation_world_to_camera,
            focus=focus_world,
            human_vertices_world=human_vertices_world,
            contact_part_vertices_world=contact_part_vertices_world,
            initial_views=initial_views,
            scene_depth_points_world=scene_depth_points_world,
            scene_vertex_tree=scene_vertex_tree,
            intrinsics=contact_intrinsics,
            width=contact_width,
            image_height=contact_height,
            min_num_views=min_synthetic_views,
            max_num_views=max_synthetic_views,
            min_camera_scene_distance_m=float(args.min_camera_scene_distance_m),
            min_human_visible_fraction=float(args.min_human_visible_fraction),
            min_human_in_frame_fraction=float(args.min_human_in_frame_fraction),
            max_human_center_offset=float(args.max_human_center_offset),
            min_human_bbox_fill=float(args.min_human_bbox_fill),
            max_human_bbox_fill=float(args.max_human_bbox_fill),
            min_interaction_part_visible_fraction=float(
                args.min_interaction_part_visible_fraction
            ),
            min_contact_part_visible_fraction=float(args.min_contact_part_visible_fraction),
            camera_radius_m=float(args.camera_radius_m),
            visibility_depth_width=int(args.visibility_depth_width),
            visibility_depth_tolerance_m=float(args.visibility_depth_tolerance_m),
            scene_depth_sample_spacing_m=float(args.scene_depth_sample_spacing_m),
            scene_depth_max_splat_radius_px=int(args.scene_depth_max_splat_radius_px),
            human_depth_sample_spacing_m=float(args.human_depth_sample_spacing_m),
            human_depth_max_splat_radius_px=int(args.human_depth_max_splat_radius_px),
            min_view_angular_separation_deg=float(args.min_view_angular_separation_deg),
            max_depth_m=float(args.max_depth_m),
        )
        for index, view in enumerate(synthetic_views, start=1):
            view["render_name"] = f"view_{index:02d}"
            view["intrinsics"] = contact_intrinsics.astype(np.float32)
            view["width"] = int(contact_width)
            view["height"] = int(contact_height)
        selected_views = [contact_view, *synthetic_views]
    view_metadata = [
        {
            "name": str(view["render_name"]),
            "source_label": str(view["label"]),
            "yaw_deg": float(view["yaw_deg"]),
            "width": int(view["width"]),
            "height": int(view["height"]),
            "camera_center_world": view["camera_center_world"].astype(float).tolist(),
            "nearest_scene_distance_m": float(view["nearest_scene_distance_m"]),
            "human_visible_fraction": float(view["human_visible_fraction"]),
            "human_in_frame_fraction": float(view["human_in_frame_fraction"]),
            "human_center_offset": float(view["human_center_offset"]),
            "human_bbox_fill": float(view["human_bbox_fill"]),
            "scene_depth_finite_fraction": float(
                view["scene_depth_finite_fraction"]
            ),
            "interaction_part_visible_fraction": float(
                view["interaction_part_visible_fraction"]
            ),
            "interaction_part_filter_available": bool(
                view["interaction_part_filter_available"]
            ),
            "contact_part_visible_fractions": {
                str(part): float(fraction)
                for part, fraction in view["contact_part_visible_fractions"].items()
            },
            "contact_part_in_frame_fractions": {
                str(part): float(fraction)
                for part, fraction in view["contact_part_in_frame_fractions"].items()
            },
            "contact_part_scene_visible_fractions": {
                str(part): float(fraction)
                for part, fraction in view[
                    "contact_part_scene_visible_fractions"
                ].items()
            },
            "contact_part_self_visible_fractions": {
                str(part): float(fraction)
                for part, fraction in view[
                    "contact_part_self_visible_fractions"
                ].items()
            },
            "covered_contact_parts": list(view["covered_contact_parts"]),
            "quality": float(view["quality"]),
            "renderable": bool(view["renderable"]),
            "valid": bool(view["valid"]),
            "relaxed_valid": bool(view["relaxed_valid"]),
            "relaxed_camera_candidate": bool(
                view.get("relaxed_camera_candidate", False)
            ),
        }
        for view in selected_views
    ]
    save_json(
        assets_dir / "selected_views.json",
        {
            "interaction_parts": interaction_part_metadata,
            "camera_source": camera_source,
            "camera_source_config_path": camera_source_config_path,
            "reference_view_used": True,
            "synthetic_view_count": int(len(synthetic_views)),
            "min_total_views": int(min_total_views),
            "max_total_views": int(max_total_views),
            "target_total_views": int(1 + len(synthetic_views)),
            "covered_contact_parts": coverage_metadata["covered_contact_parts"],
            "uncovered_contact_parts": coverage_metadata["uncovered_contact_parts"],
            "coverage_relaxed": bool(coverage_metadata["coverage_relaxed"]),
            "view_selection": coverage_metadata,
            "contact_spec_path": str(contact_spec_json_path),
            "contact_render_image_path": str(contact_render_image_path),
            "views": view_metadata,
        },
    )

    scene_crop_ply = assets_dir / "scene_semantics_view_crop.ply"
    if not scene_crop_ply.exists() or bool(args.overwrite_scene_crop):
        face_index_chunks = []
        for view in selected_views:
            scene_verts_camera = transform_world_to_camera(
                scene_verts_world,
                rotation_world_to_camera=view["rotation_world_to_camera"],
                translation_world_to_camera=view["translation_world_to_camera"],
            )
            face_index_chunks.append(
                filter_face_indices_to_camera_view(
                    verts_camera=scene_verts_camera,
                    faces=scene_faces,
                    intrinsics=view["intrinsics"],
                    width=int(view["width"]),
                    height=int(view["height"]),
                    max_depth_m=float(args.max_depth_m),
                    border_px=float(args.border_px),
                )
            )
        face_indices = (
            np.unique(np.concatenate(face_index_chunks))
            if face_index_chunks
            else np.zeros((0,), dtype=np.int64)
        )
        if face_indices.shape[0] == 0:
            raise RuntimeError("No ScanNet faces are visible from selected render views.")
        scene_faces_in_views = scene_faces[face_indices]
        compact_verts_world, compact_faces, compact_colors = compact_colored_mesh(
            scene_verts_world,
            scene_faces_in_views,
            scene_colors,
        )
        write_colored_ascii_ply(
            scene_crop_ply,
            compact_verts_world.astype(np.float32),
            compact_faces,
            compact_colors,
        )
        print(
            f"  wrote scene crop: vertices={compact_verts_world.shape[0]} "
            f"faces={compact_faces.shape[0]} views={len(selected_views)}"
        )

    blend_path = assets_dir / "render_scene.blend"
    blender_driver_path = assets_dir / "render_driver.py"
    config_path = assets_dir / "render_config.json"
    write_blender_driver(blender_driver_path)
    for stale_render in renders_dir.glob("view_*.png"):
        stale_render.unlink()
    render_views = [
        {
            "name": str(view["render_name"]),
            "render_path": str((renders_dir / f"{view['render_name']}.png").resolve()),
            "camera_matrix_world": np.asarray(
                view.get(
                    "camera_matrix_world",
                    blender_camera_matrix_world(
                        view["rotation_world_to_camera"],
                        view["translation_world_to_camera"],
                    ),
                ),
                dtype=float,
            ).tolist(),
            "intrinsics": view["intrinsics"].astype(float).tolist(),
            "width": int(view["width"]),
            "height": int(view["height"]),
        }
        for view in selected_views
    ]
    config = {
        "scene_crop_ply": str(scene_crop_ply.resolve()),
        "human_mesh_world": str(human_mesh_world_path.resolve()),
        "blend_path": str(blend_path.resolve()),
        "views": render_views,
        "camera_source": camera_source,
        "camera_source_config_path": camera_source_config_path,
        "intrinsics": contact_intrinsics.astype(float).tolist(),
        "width": int(contact_width),
        "height": int(contact_height),
        "resolution_percentage": int(args.resolution_percentage),
        "cycles_samples": int(args.cycles_samples),
    }
    save_json(config_path, config)

    blender_bin = resolve_path(args.blender_bin, BLENDER_BIN)
    if not blender_bin.exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")
    command = [
        str(blender_bin),
        "--background",
        "--python",
        str(blender_driver_path),
        "--",
        str(config_path),
    ]
    print(f"Rendering {interaction_name} with Blender")
    blender_env = build_blender_env(args.gpu_index)
    if "CUDA_VISIBLE_DEVICES" in blender_env:
        print(
            "Restricting Blender CUDA devices to: "
            f"{blender_env['CUDA_VISIBLE_DEVICES']}"
        )
    subprocess.run(command, check=True, env=blender_env)

    return {
        "interaction_name": interaction_name,
        "render_paths": [view["render_path"] for view in render_views],
        "scene_crop_ply": str(scene_crop_ply),
        "blend_path": str(blend_path),
        "selected_views_path": str(assets_dir / "selected_views.json"),
        "prompt_path": str(defaults["input_scene_json"]),
    }


def discover_interactions(output_mode: str) -> list[str]:
    output_root = PROJECT_DIR / "05_Optimize_Static_Scene" / output_mode
    names = [
        path.name
        for path in sorted(output_root.glob("interaction_*"))
        if (path / "meshes" / "frame_0000_world.ply").exists()
    ]
    if not names:
        raise RuntimeError(f"No optimized world meshes found under {output_root}.")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render optimized human-scene interactions in Blender using an "
            "original ScanNet camera plus constrained local synthetic views."
        )
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--output_mode",
        choices=("output", "output_round1"),
        default="output",
        help=(
            "Choose the matching optimization/evaluation output set. "
            "'output' uses 05_Optimize_Static_Scene/output and writes to "
            "06_Evaluate_Interaction/output by default; 'output_round1' "
            "uses/writes the output_round1 ablation folders."
        ),
    )
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--sig_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--human_mesh_world", type=str, default=None)
    parser.add_argument("--contact_spec_json", type=str, default=None)
    parser.add_argument("--contact_render_image", type=str, default=None)
    parser.add_argument("--scannet_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--blender_bin", type=str, default=None)
    parser.add_argument("--max_depth_m", type=float, default=20.0)
    parser.add_argument("--border_px", type=float, default=96.0)
    parser.add_argument("--resolution_percentage", type=int, default=75)
    parser.add_argument("--cycles_samples", type=int, default=64)
    parser.add_argument(
        "--gpu_index",
        type=str,
        default="0",
        help="CUDA device id(s) exposed to Blender, e.g. 1 or 0,1.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=None,
        help=(
            "Legacy override for the maximum number of synthetic views. "
            "By default the renderer chooses an adaptive total view count "
            "from --min_views to --max_views, including view_00."
        ),
    )
    parser.add_argument("--min_views", type=int, default=3)
    parser.add_argument("--max_views", type=int, default=6)
    parser.add_argument("--camera_radius_m", type=float, default=1.4)
    parser.add_argument("--min_camera_scene_distance_m", type=float, default=0.15)
    parser.add_argument("--min_human_visible_fraction", type=float, default=0.75)
    parser.add_argument("--min_interaction_part_visible_fraction", type=float, default=0.35)
    parser.add_argument("--min_contact_part_visible_fraction", type=float, default=0.20)
    parser.add_argument("--min_human_in_frame_fraction", type=float, default=0.92)
    parser.add_argument("--max_human_center_offset", type=float, default=0.18)
    parser.add_argument("--min_human_bbox_fill", type=float, default=0.35)
    parser.add_argument("--max_human_bbox_fill", type=float, default=0.92)
    parser.add_argument("--min_view_angular_separation_deg", type=float, default=20.0)
    parser.add_argument("--visibility_depth_width", type=int, default=384)
    parser.add_argument("--visibility_depth_tolerance_m", type=float, default=0.08)
    parser.add_argument("--scene_depth_sample_spacing_m", type=float, default=0.06)
    parser.add_argument("--scene_depth_max_splat_radius_px", type=int, default=28)
    parser.add_argument("--human_depth_sample_spacing_m", type=float, default=0.035)
    parser.add_argument("--human_depth_max_splat_radius_px", type=int, default=10)
    parser.add_argument("--max_scene_depth_points", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--overwrite_scene_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse_output_cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse baseline 06_Evaluate_Interaction/output cameras when "
            "render_config.json exists; use --no-reuse_output_cameras to "
            "force fresh view selection."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.all_interactions) or args.interaction_name == "all":
        if any(
            value is not None
            for value in (
                args.input_scene_json,
                args.sig_json,
                args.smpl_seg_json,
                args.human_mesh_world,
                args.contact_spec_json,
                args.contact_render_image,
                args.output_root,
            )
        ):
            raise ValueError(
                "--all_interactions cannot be combined with per-interaction "
                "input/output overrides."
            )
        interaction_names = discover_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    records = []
    for interaction_name in interaction_names:
        records.append(render_interaction(interaction_name, args))

    if len(records) > 1:
        save_json(SCRIPT_DIR / args.output_mode / "semantics_renders.json", records)


if __name__ == "__main__":
    main()
