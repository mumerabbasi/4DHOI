from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
BLENDER_BIN = Path("/my_workspace/blender-4.2.17-linux-x64/blender")
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}
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
EDGE_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
]


class SkipInteraction(RuntimeError):
    pass


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


def require_existing_paths(paths: dict[str, Path]) -> None:
    missing = [
        f"{label}: {path}"
        for label, path in paths.items()
        if not path.exists()
    ]
    if missing:
        raise SkipInteraction("missing required input(s): " + "; ".join(missing))


def normalize_label(text: str) -> str:
    return " ".join(
        str(text).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def slugify_file_name(text: str) -> str:
    return slugify_segment_name(text)


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
        / "sig.json",
        "smpl_seg_json": PROJECT_DIR
        / "04_Estimate_Human_Pose"
        / "assets"
        / "smplx_vert_segmentation.json",
        "init_human_mesh_world": PROJECT_DIR
        / "04_Estimate_Human_Pose"
        / "output"
        / interaction_name
        / "first_frame_smplx_world.ply",
        "optimized_human_mesh_world": PROJECT_DIR
        / "05_Optimize_Static_Scene"
        / "output"
        / interaction_name
        / "meshes"
        / "frame_0000_world.ply",
        "contact_spec_json": PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "contact_spec.json",
        "contact_masks_dir": PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "contact_masks",
        "contact_render_image": PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
        / "target_scene_crop.png",
        "module06_camera_config": PROJECT_DIR
        / "06_Evaluate_Interaction"
        / "output"
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json",
        "output_root": SCRIPT_DIR / "output" / interaction_name,
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
    raise ValueError(f"Could not find camera '{camera_name}' in {colmap_images_path}")


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
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def transform_world_to_camera(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return points_world @ rotation_world_to_camera.T + translation_world_to_camera[None]


def project_camera_points_to_image(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_camera[:, 2]
    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * points_camera[:, 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * points_camera[:, 1] / z_safe + intrinsics[1, 2] - 0.5
    return u.astype(np.float32), v.astype(np.float32), z.astype(np.float32)


def filter_face_indices_to_camera_view(
    verts_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float,
    border_px: float,
) -> np.ndarray:
    triangles = verts_camera[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1) & np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return np.zeros((0,), dtype=np.int64)

    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * triangles[..., 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * triangles[..., 1] / z_safe + intrinsics[1, 2] - 0.5
    overlaps = (
        positive
        & (np.max(u, axis=1) >= -float(border_px))
        & (np.min(u, axis=1) <= float(width - 1) + float(border_px))
        & (np.max(v, axis=1) >= -float(border_px))
        & (np.min(v, axis=1) <= float(height - 1) + float(border_px))
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


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    colors = None
    vertex_colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)
    if (
        vertex_colors.ndim == 2
        and vertex_colors.shape[0] == verts.shape[0]
        and vertex_colors.shape[1] >= 3
    ):
        colors = vertex_colors[:, :3].astype(np.uint8)
    return verts, faces, colors


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


def load_smpl_contact_segments(seg_path: Path) -> tuple[int, dict[str, np.ndarray]]:
    payload = load_json(seg_path)
    vertex_count = int(payload["vertex_count"])
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, dict):
        raise ValueError(f"Expected segments object in {seg_path}")
    segments = {
        str(segment_id): np.asarray(indices, dtype=np.int64)
        for segment_id, indices in raw_segments.items()
    }
    return vertex_count, segments


def iter_interaction_edges(sig_payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = sig_payload.get("interaction_edges", [])
    if not isinstance(edges, list):
        return []
    cleaned = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        human_part = normalize_label(str(edge.get("human_part", "")))
        scene_element = normalize_label(str(edge.get("scene_element", "")))
        if human_part and scene_element:
            cleaned.append(
                {
                    "human_part": human_part,
                    "scene_element": scene_element,
                    "notes": str(edge.get("notes", "")),
                }
            )
    return cleaned


def edge_palette(num_edges: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(int(num_edges)):
        colors.append(EDGE_COLORS[index % len(EDGE_COLORS)])
    return colors


def color_human_contacts(
    vertices: np.ndarray,
    faces: np.ndarray,
    edges: list[dict[str, Any]],
    segments: dict[str, np.ndarray],
    colors: list[tuple[int, int, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    vertex_colors = np.full((vertices.shape[0], 3), (150, 167, 190), dtype=np.uint8)
    records = []
    for edge_index, (edge, color) in enumerate(zip(edges, colors)):
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(
            slugify_segment_name(edge["human_part"])
        )
        if segment_id is None or segment_id not in segments:
            records.append({**edge, "edge_index": edge_index, "human_vertices": 0})
            continue
        vertex_ids = segments[segment_id]
        vertex_ids = vertex_ids[(vertex_ids >= 0) & (vertex_ids < vertices.shape[0])]
        vertex_colors[vertex_ids] = np.asarray(color, dtype=np.uint8)
        records.append(
            {
                **edge,
                "edge_index": edge_index,
                "segment_id": segment_id,
                "human_vertices": int(vertex_ids.shape[0]),
                "rgb": list(color),
            }
        )
    return vertex_colors, {"edges": records}


def load_mask(mask_path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = image.convert("L")
        if mask.width != width or mask.height != height:
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0


def color_scene_contacts(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    base_colors: np.ndarray,
    edges: list[dict[str, Any]],
    colors: list[tuple[int, int, int]],
    contact_masks_dir: Path,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertex_colors = np.clip(
        0.50 * base_colors.astype(np.float32) + 0.50 * np.asarray([190, 190, 190]),
        0,
        255,
    ).astype(np.uint8)
    centroids_world = vertices_world[faces].mean(axis=1).astype(np.float32)
    centroids_camera = transform_world_to_camera(
        centroids_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    u, v, z = project_camera_points_to_image(centroids_camera, intrinsics)
    valid = (
        (z > 1e-6)
        & (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
        & np.isfinite(u)
        & np.isfinite(v)
    )
    ui = np.clip(np.rint(u).astype(np.int32), 0, int(width) - 1)
    vi = np.clip(np.rint(v).astype(np.int32), 0, int(height) - 1)

    records = []
    for edge_index, (edge, color) in enumerate(zip(edges, colors)):
        mask_path = contact_masks_dir / f"{slugify_file_name(edge['human_part'])}.png"
        if not mask_path.exists():
            records.append({**edge, "edge_index": edge_index, "scene_faces": 0})
            continue
        mask = load_mask(mask_path, width=width, height=height)
        face_mask = valid & mask[vi, ui]
        face_ids = np.flatnonzero(face_mask).astype(np.int64)
        if face_ids.size > 0:
            vertex_ids = np.unique(faces[face_ids].reshape(-1))
            vertex_colors[vertex_ids] = np.asarray(color, dtype=np.uint8)
        records.append(
            {
                **edge,
                "edge_index": edge_index,
                "mask_path": str(mask_path),
                "scene_faces": int(face_ids.shape[0]),
                "rgb": list(color),
            }
        )
    return vertex_colors, {"edges": records}


def load_module06_render_views(camera_config_path: Path) -> list[dict[str, Any]]:
    payload = load_json(camera_config_path)
    raw_views = payload.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError(f"Module 06 camera config has no views: {camera_config_path}")

    views = []
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, dict):
            raise ValueError(
                f"Module 06 camera view {index} is malformed: {camera_config_path}"
            )
        name = str(
            raw_view.get("name")
            or Path(str(raw_view.get("render_path", ""))).stem
            or f"view_{index:02d}"
        )
        matrix = np.asarray(raw_view.get("camera_matrix_world"), dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Module 06 camera view {index} has invalid camera_matrix_world "
                f"shape {matrix.shape}: {camera_config_path}"
            )
        intrinsics = np.asarray(
            raw_view.get("intrinsics", payload.get("intrinsics")),
            dtype=np.float32,
        )
        if intrinsics.shape != (3, 3):
            raise ValueError(
                f"Module 06 camera view {index} has invalid intrinsics shape "
                f"{intrinsics.shape}: {camera_config_path}"
            )
        if "width" not in raw_view and "width" not in payload:
            raise ValueError(
                f"Module 06 camera view {index} missing width: {camera_config_path}"
            )
        if "height" not in raw_view and "height" not in payload:
            raise ValueError(
                f"Module 06 camera view {index} missing height: {camera_config_path}"
            )
        views.append(
            {
                "name": name,
                "camera_matrix_world": matrix.astype(float).tolist(),
                "intrinsics": intrinsics.astype(float).tolist(),
                "width": int(raw_view.get("width", payload.get("width"))),
                "height": int(raw_view.get("height", payload.get("height"))),
            }
        )
    return views


def crop_scene_to_views(
    scene_verts_world: np.ndarray,
    scene_faces: np.ndarray,
    scene_colors: np.ndarray,
    views: list[dict[str, Any]],
    max_depth_m: float,
    border_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = []
    for view in views:
        rotation, translation = camera_extrinsics_from_blender_matrix_world(
            np.asarray(view["camera_matrix_world"], dtype=np.float32)
        )
        verts_camera = transform_world_to_camera(
            scene_verts_world,
            rotation_world_to_camera=rotation,
            translation_world_to_camera=translation,
        )
        chunks.append(
            filter_face_indices_to_camera_view(
                verts_camera=verts_camera,
                faces=scene_faces,
                intrinsics=np.asarray(view["intrinsics"], dtype=np.float32),
                width=int(view["width"]),
                height=int(view["height"]),
                max_depth_m=max_depth_m,
                border_px=border_px,
            )
        )
    face_ids = (
        np.unique(np.concatenate(chunks))
        if chunks
        else np.zeros((0,), dtype=np.int64)
    )
    if face_ids.shape[0] == 0:
        raise RuntimeError("No scene faces are visible from selected render views.")
    return compact_colored_mesh(
        scene_verts_world,
        scene_faces[face_ids],
        scene_colors,
    )


def build_blender_env(gpu_index: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_index is not None and str(gpu_index).strip():
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index).strip()
    return env


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


def assign_vertex_color_material(obj, roughness=0.62):
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
    bsdf.inputs["Roughness"].default_value = float(roughness)
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


def configure_soft_lighting(human_objects):
    focus = object_bounds_center_world(human_objects[0])
    focus.z += 0.65
    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
        background.inputs["Strength"].default_value = 0.24
    add_shadowless_area_light(
        "thesis_overhead_softbox",
        (focus.x, focus.y, focus.z + 2.4),
        focus,
        energy=110.0,
        size=4.0,
    )


argv = sys.argv
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene_obj = import_ply(config["scene_contact_mesh"])
scene_obj.name = "scene_contact_regions"
assign_vertex_color_material(scene_obj, roughness=0.70)

human_objects = {}
for state in config["states"]:
    obj = import_ply(state["mesh_path"])
    obj.name = f"{state['name']}_human_contact_regions"
    assign_vertex_color_material(obj, roughness=0.55)
    obj.hide_render = True
    obj.hide_viewport = True
    human_objects[state["name"]] = obj

default_width = int(config["width"])
default_height = int(config["height"])
default_intrinsics = config["intrinsics"]
sensor_width = 36.0
camera_objects = {}
for view in config["views"]:
    width = int(view.get("width", default_width))
    height = int(view.get("height", default_height))
    intrinsics = view.get("intrinsics", default_intrinsics)
    fx = float(intrinsics[0][0])
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
    camera_objects[view["name"]] = (camera_obj, width, height)

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = (
    bpy.data.worlds.new("world")
    if bpy.context.scene.world is None
    else bpy.context.scene.world
)
configure_soft_lighting(list(human_objects.values()))
bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])

for job in config["render_jobs"]:
    for obj in human_objects.values():
        obj.hide_render = True
        obj.hide_viewport = True
    human_obj = human_objects[job["state"]]
    human_obj.hide_render = False
    human_obj.hide_viewport = False
    camera_obj, width, height = camera_objects[job["view"]]
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = int(width)
    bpy.context.scene.render.resolution_y = int(height)
    bpy.context.scene.render.filepath = job["render_path"]
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
    init_human_mesh_path = resolve_path(
        args.init_human_mesh_world,
        defaults["init_human_mesh_world"],
    )
    optimized_human_mesh_path = resolve_path(
        args.optimized_human_mesh_world,
        defaults["optimized_human_mesh_world"],
    )
    contact_spec_json_path = resolve_path(
        args.contact_spec_json,
        defaults["contact_spec_json"],
    )
    contact_masks_dir = resolve_path(args.contact_masks_dir, defaults["contact_masks_dir"])
    contact_render_image_path = resolve_path(
        args.contact_render_image,
        defaults["contact_render_image"],
    )
    module06_camera_config_path = resolve_path(
        args.module06_camera_config,
        defaults["module06_camera_config"],
    )
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    scannet_root = resolve_scannet_root(args.scannet_root)

    require_existing_paths(
        {
            "input_scene_json": input_scene_json_path,
            "sig_json": sig_json_path,
            "smpl_seg_json": smpl_seg_json_path,
            "init_human_mesh_world": init_human_mesh_path,
            "optimized_human_mesh_world": optimized_human_mesh_path,
            "contact_spec_json": contact_spec_json_path,
            "contact_masks_dir": contact_masks_dir,
            "contact_render_image": contact_render_image_path,
            "module06_camera_config": module06_camera_config_path,
        }
    )

    input_payload = load_json(input_scene_json_path)
    sig_payload = load_json(sig_json_path)
    edges = iter_interaction_edges(sig_payload)
    if not edges:
        raise SkipInteraction(f"No interaction_edges found in {sig_json_path}")
    edge_colors = edge_palette(len(edges))

    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)
    require_existing_paths(
        {
            "scannet_transforms": scene_paths["transforms_path"],
            "scannet_colmap_images": scene_paths["colmap_images_path"],
            "scannet_mesh": scene_paths["mesh_path"],
        }
    )
    (
        _intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        _width,
        _height,
    ) = load_scannet_camera(scene_paths, scene_context)
    contact_intrinsics = load_contact_spec_intrinsics(contact_spec_json_path)
    contact_width, contact_height = load_image_size(contact_render_image_path)
    views = load_module06_render_views(module06_camera_config_path)

    smpl_vertex_count, smpl_segments = load_smpl_contact_segments(smpl_seg_json_path)
    init_verts, init_faces, _init_colors = load_mesh(init_human_mesh_path)
    opt_verts, opt_faces, _opt_colors = load_mesh(optimized_human_mesh_path)
    if init_verts.shape[0] != smpl_vertex_count:
        raise ValueError(
            "Initial human mesh vertex count does not match segmentation: "
            f"mesh={init_verts.shape[0]} segmentation={smpl_vertex_count}"
        )
    if opt_verts.shape[0] != smpl_vertex_count:
        raise ValueError(
            "Optimized human mesh vertex count does not match segmentation: "
            f"mesh={opt_verts.shape[0]} segmentation={smpl_vertex_count}"
        )
    init_human_colors, init_human_meta = color_human_contacts(
        init_verts,
        init_faces,
        edges,
        smpl_segments,
        edge_colors,
    )
    opt_human_colors, opt_human_meta = color_human_contacts(
        opt_verts,
        opt_faces,
        edges,
        smpl_segments,
        edge_colors,
    )

    scene_verts, scene_faces, scene_original_colors = load_mesh(scene_paths["mesh_path"])
    if scene_original_colors is None:
        scene_original_colors = np.full(
            (scene_verts.shape[0], 3),
            (185, 185, 185),
            dtype=np.uint8,
        )
    scene_colors, scene_meta = color_scene_contacts(
        vertices_world=scene_verts,
        faces=scene_faces,
        base_colors=scene_original_colors,
        edges=edges,
        colors=edge_colors,
        contact_masks_dir=contact_masks_dir,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        intrinsics=contact_intrinsics,
        width=contact_width,
        height=contact_height,
    )
    scene_crop_verts, scene_crop_faces, scene_crop_colors = crop_scene_to_views(
        scene_verts,
        scene_faces,
        scene_colors,
        views=views,
        max_depth_m=float(args.max_depth_m),
        border_px=float(args.border_px),
    )

    scene_contact_mesh = assets_dir / "scene_contact_regions.ply"
    init_human_mesh = assets_dir / "human_init_contact_regions.ply"
    optimized_human_mesh = assets_dir / "human_optimized_contact_regions.ply"
    write_colored_ascii_ply(
        scene_contact_mesh,
        scene_crop_verts,
        scene_crop_faces,
        scene_crop_colors,
    )
    write_colored_ascii_ply(init_human_mesh, init_verts, init_faces, init_human_colors)
    write_colored_ascii_ply(
        optimized_human_mesh,
        opt_verts,
        opt_faces,
        opt_human_colors,
    )

    render_jobs = []
    for state_name in ("init", "optimized"):
        ensure_dir(renders_dir / state_name)
        for view in views:
            render_jobs.append(
                {
                    "state": state_name,
                    "view": view["name"],
                    "render_path": str(
                        (renders_dir / state_name / f"{view['name']}.png").resolve()
                    ),
                }
            )

    blend_path = assets_dir / "contact_comparison.blend"
    blender_driver_path = assets_dir / "render_driver.py"
    config_path = assets_dir / "render_config.json"
    write_blender_driver(blender_driver_path)
    config = {
        "interaction_name": interaction_name,
        "scene_contact_mesh": str(scene_contact_mesh.resolve()),
        "blend_path": str(blend_path.resolve()),
        "states": [
            {"name": "init", "mesh_path": str(init_human_mesh.resolve())},
            {"name": "optimized", "mesh_path": str(optimized_human_mesh.resolve())},
        ],
        "views": views,
        "render_jobs": render_jobs,
        "intrinsics": views[0]["intrinsics"],
        "width": int(views[0]["width"]),
        "height": int(views[0]["height"]),
        "resolution_percentage": int(args.resolution_percentage),
        "cycles_samples": int(args.cycles_samples),
        "camera_source": "module06_baseline_output",
        "camera_source_config_path": str(module06_camera_config_path),
        "contact_spec_path": str(contact_spec_json_path),
        "contact_render_image_path": str(contact_render_image_path),
        "contact_edges": [
            {
                "edge_index": index,
                "human_part": edge["human_part"],
                "scene_element": edge["scene_element"],
                "rgb": list(color),
            }
            for index, (edge, color) in enumerate(zip(edges, edge_colors))
        ],
        "contact_region_metadata": {
            "scene": scene_meta,
            "init_human": init_human_meta,
            "optimized_human": opt_human_meta,
        },
    }
    save_json(config_path, config)

    if bool(args.skip_blender):
        print(f"Wrote visualization assets for {interaction_name}: {assets_dir}")
    else:
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
        print(f"Rendering thesis contact comparison for {interaction_name}")
        subprocess.run(command, check=True, env=build_blender_env(args.gpu_index))

    return {
        "interaction_name": interaction_name,
        "config_path": str(config_path),
        "scene_contact_mesh": str(scene_contact_mesh),
        "init_human_mesh": str(init_human_mesh),
        "optimized_human_mesh": str(optimized_human_mesh),
        "renders": [job["render_path"] for job in render_jobs],
        "skipped_blender": bool(args.skip_blender),
    }


def discover_interactions() -> list[str]:
    output_root = PROJECT_DIR / "04_Estimate_Human_Pose" / "output"
    names = [
        path.name
        for path in sorted(output_root.glob("interaction_*"))
        if path.is_dir()
    ]
    if not names:
        raise RuntimeError(f"No interaction directories found under {output_root}.")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render thesis visualizations comparing GVHMR-init and optimized "
            "human meshes, with matched contact-edge colors on human and scene."
        )
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--sig_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--init_human_mesh_world", type=str, default=None)
    parser.add_argument("--optimized_human_mesh_world", type=str, default=None)
    parser.add_argument("--contact_spec_json", type=str, default=None)
    parser.add_argument("--contact_masks_dir", type=str, default=None)
    parser.add_argument("--contact_render_image", type=str, default=None)
    parser.add_argument(
        "--module06_camera_config",
        type=str,
        default=None,
        help=(
            "Module 06 baseline semantic render_config.json to use as the render "
            "camera source."
        ),
    )
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
        "--skip_blender",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write colored PLY/config assets without invoking Blender.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if any(
            value is not None
            for value in (
                args.input_scene_json,
                args.sig_json,
                args.smpl_seg_json,
                args.init_human_mesh_world,
                args.optimized_human_mesh_world,
                args.contact_spec_json,
                args.contact_masks_dir,
                args.contact_render_image,
                args.module06_camera_config,
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
    skipped = []
    for interaction_name in interaction_names:
        try:
            records.append(render_interaction(interaction_name, args))
        except SkipInteraction as exc:
            skipped.append(interaction_name)
            print(f"Skipping {interaction_name}: {exc}")

    if records:
        output_base = SCRIPT_DIR / "output"
        ensure_dir(output_base)
        save_json(output_base / "contact_comparison_renders.json", records)
    if skipped:
        print("Skipped interaction(s): " + ", ".join(skipped))
    if not records:
        print("No thesis contact comparison visualizations were produced.")


if __name__ == "__main__":
    main()
