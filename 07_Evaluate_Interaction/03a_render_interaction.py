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


def build_default_paths(interaction_name: str) -> dict[str, Path]:
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
        / "scene_interaction_graph.json",
        "smpl_seg_json": PROJECT_DIR
        / "05_Estimate_Human_Pose"
        / "assets"
        / "smplx_vert_segmentation.json",
        "human_mesh_world": PROJECT_DIR
        / "06_Optimize_Static_Scene"
        / "output"
        / interaction_name
        / "meshes"
        / "frame_0000_world.ply",
        "output_root": SCRIPT_DIR / "output" / interaction_name / "semantics",
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
    if raw == "target_object" or normalized in {"target object", "object", target_norm}:
        return "target_object"
    return normalized


def resolve_sig_target_label(sig_payload: dict[str, Any]) -> str:
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


def load_smpl_body_segments(seg_path: Path) -> tuple[int, dict[str, np.ndarray], set[str]]:
    payload = load_json(seg_path)
    raw_segments = payload.get("segments")
    body_segment_ids = payload.get("body_segment_ids")
    if not isinstance(raw_segments, dict):
        raise KeyError(f"Expected 'segments' mapping in {seg_path}.")
    if not isinstance(body_segment_ids, list):
        raise KeyError(f"Expected 'body_segment_ids' list in {seg_path}.")

    vertex_count = int(payload["vertex_count"])
    segments: dict[str, np.ndarray] = {}
    for segment_id, indices in raw_segments.items():
        indices_array = np.unique(np.asarray(indices, dtype=np.int64))
        if indices_array.size == 0:
            continue
        if indices_array[0] < 0 or indices_array[-1] >= vertex_count:
            raise ValueError(f"Segment '{segment_id}' has out-of-range vertex ids.")
        segments[str(segment_id)] = indices_array

    body_ids = {str(segment_id) for segment_id in body_segment_ids}
    missing = sorted(segment_id for segment_id in body_ids if segment_id not in segments)
    if missing:
        raise KeyError(f"Missing SMPL-X body segments in {seg_path}: {missing}")
    return vertex_count, segments, body_ids


def build_interaction_part_vertices(
    sig_payload: dict[str, Any],
    smpl_segments: dict[str, np.ndarray],
    body_segment_ids: set[str],
    human_vertices_world: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    part_names = iter_interaction_human_parts(sig_payload)
    segment_ids: list[str] = []
    missing_parts: list[str] = []
    for part_name in part_names:
        segment_id = slugify_segment_name(part_name)
        if segment_id not in body_segment_ids:
            missing_parts.append(part_name)
            continue
        segment_ids.append(segment_id)

    if not segment_ids:
        return np.zeros((0, 3), dtype=np.float32), {
            "available": False,
            "human_parts": part_names,
            "segment_ids": [],
            "vertex_count": 0,
            "missing_human_parts": missing_parts,
        }

    vertex_ids = np.unique(np.concatenate([smpl_segments[item] for item in segment_ids]))
    return human_vertices_world[vertex_ids].astype(np.float32), {
        "available": True,
        "human_parts": part_names,
        "segment_ids": segment_ids,
        "vertex_count": int(vertex_ids.shape[0]),
        "missing_human_parts": missing_parts,
    }


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


def build_candidate_views(
    original_rotation_world_to_camera: np.ndarray,
    original_translation_world_to_camera: np.ndarray,
    focus: np.ndarray,
    human_vertices_world: np.ndarray,
    interaction_part_vertices_world: np.ndarray,
    scene_depth_points_world: np.ndarray,
    scene_vertex_tree: cKDTree,
    intrinsics: np.ndarray,
    width: int,
    image_height: int,
    num_views: int,
    min_camera_scene_distance_m: float,
    min_human_visible_fraction: float,
    min_interaction_part_visible_fraction: float,
    camera_radius_m: float,
    visibility_depth_width: int,
    visibility_depth_tolerance_m: float,
    max_depth_m: float,
) -> list[dict[str, Any]]:
    original_center = camera_center_from_extrinsics(
        original_rotation_world_to_camera,
        original_translation_world_to_camera,
    )
    base_vector = original_center - focus
    base_vector_xy = base_vector.copy()
    base_vector_xy[2] = 0.0
    if float(np.linalg.norm(base_vector_xy)) < 1e-6:
        base_vector_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    base_radius = float(camera_radius_m)
    camera_height_offset = float(original_center[2] - focus[2])
    camera_height_offset = float(np.clip(camera_height_offset, 0.15, 1.35))
    base_dir = normalize_vector(base_vector_xy)
    interaction_filter_available = interaction_part_vertices_world.shape[0] > 0

    candidates: list[dict[str, Any]] = []
    yaw_offsets = [0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 180.0]
    radii = [base_radius * scale for scale in (0.75, 1.0, 1.2)]
    height_offsets = [
        camera_height_offset,
        camera_height_offset + 0.25,
    ]
    for radius in radii:
        for yaw in yaw_offsets:
            for height_offset in height_offsets:
                direction = normalize_vector(rotate_about_up(base_dir, float(yaw)))
                center = focus + direction * float(radius)
                center[2] = focus[2] + float(np.clip(height_offset, 0.05, 1.6))
                rotation, translation = look_at_world_to_camera(center, focus)
                nearest_distance = float(scene_vertex_tree.query(center)[0])
                scene_depth, depth_scale = make_scene_depth_buffer(
                    scene_depth_points_world=scene_depth_points_world,
                    rotation_world_to_camera=rotation,
                    translation_world_to_camera=translation,
                    intrinsics=intrinsics,
                    width=width,
                    height=image_height,
                    max_depth_m=max_depth_m,
                    depth_width=visibility_depth_width,
                )
                human_depth, _ = make_scene_depth_buffer(
                    scene_depth_points_world=human_vertices_world,
                    rotation_world_to_camera=rotation,
                    translation_world_to_camera=translation,
                    intrinsics=intrinsics,
                    width=width,
                    height=image_height,
                    max_depth_m=max_depth_m,
                    depth_width=visibility_depth_width,
                )
                human_visible_fraction = point_visibility_fraction_from_depth(
                    human_vertices_world,
                    rotation,
                    translation,
                    intrinsics,
                    width,
                    image_height,
                    scene_depth=scene_depth,
                    depth_scale=depth_scale,
                    depth_tolerance_m=visibility_depth_tolerance_m,
                    require_scene_depth=False,
                )
                if interaction_filter_available:
                    interaction_part_visible_fraction = point_visibility_fraction_from_depth(
                        interaction_part_vertices_world,
                        rotation,
                        translation,
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
                else:
                    interaction_part_visible_fraction = 1.0

                valid = (
                    nearest_distance >= float(min_camera_scene_distance_m)
                    and human_visible_fraction >= float(min_human_visible_fraction)
                    and interaction_part_visible_fraction
                    >= float(min_interaction_part_visible_fraction)
                )
                quality = (
                    0.45 * human_visible_fraction
                    + 0.55 * interaction_part_visible_fraction
                )
                candidates.append(
                    {
                        "label": "view_axis_radius" if yaw == 0.0 else f"yaw_{int(yaw):+d}",
                        "yaw_deg": float(yaw),
                        "camera_radius_m": float(radius),
                        "camera_height_offset_m": float(height_offset),
                        "camera_center_world": center.astype(np.float32),
                        "rotation_world_to_camera": rotation.astype(np.float32),
                        "translation_world_to_camera": translation.astype(np.float32),
                        "nearest_scene_distance_m": nearest_distance,
                        "human_visible_fraction": human_visible_fraction,
                        "interaction_part_visible_fraction": interaction_part_visible_fraction,
                        "interaction_part_filter_available": bool(interaction_filter_available),
                        "quality": float(quality),
                        "valid": bool(valid),
                    }
                )

    valid_candidates = sorted(
        [candidate for candidate in candidates if candidate["valid"]],
        key=lambda candidate: float(candidate["quality"]),
        reverse=True,
    )
    best_by_yaw: dict[float, dict[str, Any]] = {}
    for candidate in valid_candidates:
        yaw = float(candidate["yaw_deg"])
        if yaw not in best_by_yaw:
            best_by_yaw[yaw] = candidate
    diverse_candidates = sorted(
        best_by_yaw.values(),
        key=lambda candidate: float(candidate["quality"]),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < int(num_views) and diverse_candidates:
        if not selected:
            selected.append(diverse_candidates.pop(0))
            continue
        selected_dirs = [
            normalize_vector(item["camera_center_world"] - focus)
            for item in selected
        ]

        def view_score(candidate: dict[str, Any]) -> float:
            direction = normalize_vector(candidate["camera_center_world"] - focus)
            min_angle = min(
                float(np.arccos(np.clip(np.dot(direction, selected_dir), -1.0, 1.0)))
                for selected_dir in selected_dirs
            )
            return float(candidate["quality"]) + 0.25 * min_angle

        best = max(diverse_candidates, key=view_score)
        selected.append(best)
        diverse_candidates = [
            candidate for candidate in diverse_candidates if candidate is not best
        ]
    selected_ids = {id(candidate) for candidate in selected}
    for candidate in valid_candidates:
        if len(selected) >= int(num_views):
            break
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))

    fallback_candidates = sorted(
        candidates,
        key=lambda candidate: float(candidate["quality"]),
        reverse=True,
    )
    if not selected and fallback_candidates:
        selected.append(fallback_candidates[0])
    selected = selected[: int(num_views)]

    for index, item in enumerate(selected):
        item["render_name"] = f"view_{index:02d}"
    return selected


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
        background.inputs["Strength"].default_value = 0.12

    add_shadowless_area_light(
        "room_overhead_softbox",
        (focus.x, focus.y, focus.z + 2.4),
        focus,
        energy=55.0,
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

width = int(config["width"])
height = int(config["height"])
fx = float(config["intrinsics"][0][0])
fy = float(config["intrinsics"][1][1])
cx = float(config["intrinsics"][0][2])
cy = float(config["intrinsics"][1][2])
sensor_width = 36.0
camera_objects = []
for view in config["views"]:
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
    camera_objects.append((camera_obj, view["render_path"]))

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = bpy.data.worlds.new("world") if bpy.context.scene.world is None else bpy.context.scene.world
configure_soft_room_lighting(human_obj, camera_objects)
bpy.context.scene.render.resolution_x = width
bpy.context.scene.render.resolution_y = height
bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])
for camera_obj, render_path in camera_objects:
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)
'''.lstrip(),
        encoding="utf-8",
    )


def render_interaction(
    interaction_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name)
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    smpl_seg_json_path = resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    human_mesh_world_path = resolve_path(args.human_mesh_world, defaults["human_mesh_world"])
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    scannet_root = resolve_scannet_root(args.scannet_root)

    if not human_mesh_world_path.exists():
        raise FileNotFoundError(f"Optimized human world mesh not found: {human_mesh_world_path}")

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

    print(f"Loading optimized human mesh from: {human_mesh_world_path}")
    human_vertices_world = load_mesh_vertices(human_mesh_world_path)
    focus_world = human_focus_point(human_vertices_world)
    smpl_vertex_count, smpl_segments, body_segment_ids = load_smpl_body_segments(
        smpl_seg_json_path
    )
    if human_vertices_world.shape[0] != smpl_vertex_count:
        raise ValueError(
            "Optimized human mesh vertex count does not match SMPL-X segmentation: "
            f"mesh={human_vertices_world.shape[0]} segmentation={smpl_vertex_count}"
        )
    interaction_part_vertices_world, interaction_part_metadata = (
        build_interaction_part_vertices(
            sig_payload=sig_payload,
            smpl_segments=smpl_segments,
            body_segment_ids=body_segment_ids,
            human_vertices_world=human_vertices_world,
        )
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
    selected_views = build_candidate_views(
        original_rotation_world_to_camera=rotation_world_to_camera,
        original_translation_world_to_camera=translation_world_to_camera,
        focus=focus_world,
        human_vertices_world=human_vertices_world,
        interaction_part_vertices_world=interaction_part_vertices_world,
        scene_depth_points_world=scene_depth_points_world,
        scene_vertex_tree=scene_vertex_tree,
        intrinsics=intrinsics,
        width=width,
        image_height=height,
        num_views=int(args.num_views),
        min_camera_scene_distance_m=float(args.min_camera_scene_distance_m),
        min_human_visible_fraction=float(args.min_human_visible_fraction),
        min_interaction_part_visible_fraction=float(
            args.min_interaction_part_visible_fraction
        ),
        camera_radius_m=float(args.camera_radius_m),
        visibility_depth_width=int(args.visibility_depth_width),
        visibility_depth_tolerance_m=float(args.visibility_depth_tolerance_m),
        max_depth_m=float(args.max_depth_m),
    )
    view_metadata = [
        {
            "name": str(view["render_name"]),
            "source_label": str(view["label"]),
            "yaw_deg": float(view["yaw_deg"]),
            "camera_center_world": view["camera_center_world"].astype(float).tolist(),
            "nearest_scene_distance_m": float(view["nearest_scene_distance_m"]),
            "human_visible_fraction": float(view["human_visible_fraction"]),
            "interaction_part_visible_fraction": float(
                view["interaction_part_visible_fraction"]
            ),
            "interaction_part_filter_available": bool(
                view["interaction_part_filter_available"]
            ),
            "quality": float(view["quality"]),
            "valid": bool(view["valid"]),
        }
        for view in selected_views
    ]
    save_json(
        assets_dir / "selected_views.json",
        {
            "interaction_parts": interaction_part_metadata,
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
                    intrinsics=intrinsics,
                    width=width,
                    height=height,
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
            "camera_matrix_world": blender_camera_matrix_world(
                view["rotation_world_to_camera"],
                view["translation_world_to_camera"],
            ).tolist(),
        }
        for view in selected_views
    ]
    config = {
        "scene_crop_ply": str(scene_crop_ply.resolve()),
        "human_mesh_world": str(human_mesh_world_path.resolve()),
        "blend_path": str(blend_path.resolve()),
        "views": render_views,
        "intrinsics": intrinsics.astype(float).tolist(),
        "width": int(width),
        "height": int(height),
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


def discover_interactions() -> list[str]:
    output_root = PROJECT_DIR / "06_Optimize_Static_Scene" / "output"
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
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--sig_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--human_mesh_world", type=str, default=None)
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
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--camera_radius_m", type=float, default=1.4)
    parser.add_argument("--min_camera_scene_distance_m", type=float, default=0.15)
    parser.add_argument("--min_human_visible_fraction", type=float, default=0.75)
    parser.add_argument("--min_interaction_part_visible_fraction", type=float, default=0.35)
    parser.add_argument("--visibility_depth_width", type=int, default=384)
    parser.add_argument("--visibility_depth_tolerance_m", type=float, default=0.08)
    parser.add_argument("--max_scene_depth_points", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--overwrite_scene_crop",
        action=argparse.BooleanOptionalAction,
        default=True,
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
                args.output_root,
            )
        ):
            raise ValueError(
                "--all_interactions cannot be combined with per-interaction "
                "input/output overrides."
            )
        interaction_names = discover_interactions()
    else:
        interaction_names = [args.interaction_name]

    records = []
    for interaction_name in interaction_names:
        records.append(render_interaction(interaction_name, args))

    if len(records) > 1:
        save_json(SCRIPT_DIR / "output" / "semantics_renders.json", records)


if __name__ == "__main__":
    main()
