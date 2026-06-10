from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

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


def human_focus_and_probe_points(human_vertices_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vmin = human_vertices_world.min(axis=0)
    vmax = human_vertices_world.max(axis=0)
    center = (vmin + vmax) * 0.5
    height = float(vmax[2] - vmin[2])
    focus = center.copy()
    focus[2] = float(vmin[2] + 0.55 * height)
    shoulder_offset = max(float(vmax[0] - vmin[0]), float(vmax[1] - vmin[1])) * 0.22
    probe_points = np.asarray(
        [
            focus,
            [center[0], center[1], vmin[2] + 0.82 * height],
            [center[0], center[1], vmin[2] + 0.25 * height],
            [center[0] + shoulder_offset, center[1], vmin[2] + 0.62 * height],
            [center[0] - shoulder_offset, center[1], vmin[2] + 0.62 * height],
        ],
        dtype=np.float32,
    )
    return focus.astype(np.float32), probe_points


def project_points_fraction_in_frame(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> float:
    points_camera = transform_world_to_camera(
        points_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    z = points_camera[:, 2]
    valid_z = z > 1e-6
    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * points_camera[:, 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * points_camera[:, 1] / z_safe + intrinsics[1, 2] - 0.5
    in_frame = (
        valid_z
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )
    return float(np.count_nonzero(in_frame) / max(points_world.shape[0], 1))


def line_of_sight_fraction(
    scene_mesh: trimesh.Trimesh,
    camera_center: np.ndarray,
    probe_points: np.ndarray,
    clearance_m: float,
) -> float:
    origins = np.repeat(camera_center.reshape(1, 3), probe_points.shape[0], axis=0)
    vectors = probe_points - origins
    distances = np.linalg.norm(vectors, axis=1)
    valid = distances > 1e-6
    if not np.any(valid):
        return 0.0
    directions = np.zeros_like(vectors, dtype=np.float32)
    directions[valid] = vectors[valid] / distances[valid, None]
    locations, ray_ids, _tri_ids = scene_mesh.ray.intersects_location(
        ray_origins=origins[valid],
        ray_directions=directions[valid],
        multiple_hits=False,
    )
    blocked = np.zeros(int(np.count_nonzero(valid)), dtype=bool)
    if len(ray_ids) > 0:
        hit_distances = np.linalg.norm(locations - origins[valid][ray_ids], axis=1)
        target_distances = distances[valid][ray_ids]
        blocked[ray_ids] = hit_distances < (target_distances - float(clearance_m))
    return float(1.0 - np.count_nonzero(blocked) / max(blocked.shape[0], 1))


def build_candidate_views(
    original_rotation_world_to_camera: np.ndarray,
    original_translation_world_to_camera: np.ndarray,
    focus: np.ndarray,
    human_vertices_world: np.ndarray,
    probe_points: np.ndarray,
    scene_mesh: trimesh.Trimesh,
    scene_vertex_tree: cKDTree,
    intrinsics: np.ndarray,
    width: int,
    image_height: int,
    num_views: int,
    min_camera_scene_distance_m: float,
    min_line_of_sight_fraction: float,
    min_human_frame_fraction: float,
    camera_radius_m: float,
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
    radius = float(camera_radius_m)
    camera_height_offset = float(original_center[2] - focus[2])
    camera_height_offset = float(np.clip(camera_height_offset, 0.15, 1.35))
    base_dir = normalize_vector(base_vector_xy)

    candidates: list[dict[str, Any]] = []
    yaw_offsets = [0.0, 35.0, -35.0, 70.0, -70.0, 110.0, -110.0, 180.0]
    for yaw in yaw_offsets:
        direction = normalize_vector(rotate_about_up(base_dir, float(yaw)))
        center = focus + direction * radius
        center[2] = focus[2] + camera_height_offset
        rotation, translation = look_at_world_to_camera(center, focus)
        nearest_distance = float(scene_vertex_tree.query(center)[0])
        frame_fraction = project_points_fraction_in_frame(
            human_vertices_world,
            rotation,
            translation,
            intrinsics,
            width,
            image_height,
        )
        los_fraction = line_of_sight_fraction(
            scene_mesh,
            center,
            probe_points,
            clearance_m=0.08,
        )
        valid = (
            nearest_distance >= float(min_camera_scene_distance_m)
            and frame_fraction >= float(min_human_frame_fraction)
            and los_fraction >= float(min_line_of_sight_fraction)
        )
        candidates.append(
            {
                "label": "view_axis_radius" if yaw == 0.0 else f"yaw_{int(yaw):+d}",
                "yaw_deg": float(yaw),
                "camera_radius_m": radius,
                "camera_center_world": center.astype(np.float32),
                "rotation_world_to_camera": rotation.astype(np.float32),
                "translation_world_to_camera": translation.astype(np.float32),
                "nearest_scene_distance_m": nearest_distance,
                "human_frame_fraction": frame_fraction,
                "line_of_sight_fraction": los_fraction,
                "valid": bool(valid),
            }
        )

    selected = [candidate for candidate in candidates if candidate["valid"]]
    fallback_candidates = [candidate for candidate in candidates if candidate not in selected]
    fallback_candidates = sorted(
        fallback_candidates,
        key=lambda candidate: (
            float(candidate["nearest_scene_distance_m"]),
            float(candidate["line_of_sight_fraction"]),
            float(candidate["human_frame_fraction"]),
        ),
        reverse=True,
    )
    if not selected and fallback_candidates:
        selected.append(fallback_candidates.pop(0))
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
    human_mesh_world_path = resolve_path(args.human_mesh_world, defaults["human_mesh_world"])
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    scannet_root = resolve_scannet_root(args.scannet_root)

    if not human_mesh_world_path.exists():
        raise FileNotFoundError(f"Optimized human world mesh not found: {human_mesh_world_path}")

    input_payload = load_json(input_scene_json_path)
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
    focus_world, probe_points_world = human_focus_and_probe_points(human_vertices_world)

    print(f"Loading colored ScanNet mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces, scene_colors = load_colored_mesh(
        scene_paths["mesh_path"]
    )
    scene_mesh_for_rays = trimesh.Trimesh(
        vertices=scene_verts_world,
        faces=scene_faces,
        process=False,
    )
    scene_vertex_tree = cKDTree(scene_verts_world)
    selected_views = build_candidate_views(
        original_rotation_world_to_camera=rotation_world_to_camera,
        original_translation_world_to_camera=translation_world_to_camera,
        focus=focus_world,
        human_vertices_world=human_vertices_world,
        probe_points=probe_points_world,
        scene_mesh=scene_mesh_for_rays,
        scene_vertex_tree=scene_vertex_tree,
        intrinsics=intrinsics,
        width=width,
        image_height=height,
        num_views=int(args.num_views),
        min_camera_scene_distance_m=float(args.min_camera_scene_distance_m),
        min_line_of_sight_fraction=float(args.min_line_of_sight_fraction),
        min_human_frame_fraction=float(args.min_human_frame_fraction),
        camera_radius_m=float(args.camera_radius_m),
    )
    view_metadata = [
        {
            "name": str(view["render_name"]),
            "source_label": str(view["label"]),
            "yaw_deg": float(view["yaw_deg"]),
            "camera_center_world": view["camera_center_world"].astype(float).tolist(),
            "nearest_scene_distance_m": float(view["nearest_scene_distance_m"]),
            "human_frame_fraction": float(view["human_frame_fraction"]),
            "line_of_sight_fraction": float(view["line_of_sight_fraction"]),
            "valid": bool(view["valid"]),
        }
        for view in selected_views
    ]
    save_json(assets_dir / "selected_views.json", view_metadata)

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
    parser.add_argument("--min_camera_scene_distance_m", type=float, default=0.25)
    parser.add_argument("--min_line_of_sight_fraction", type=float, default=0.4)
    parser.add_argument("--min_human_frame_fraction", type=float, default=0.35)
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
            for value in (args.input_scene_json, args.human_mesh_world, args.output_root)
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
