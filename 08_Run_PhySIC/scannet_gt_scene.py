"""Build a calibrated visible ScanNet++ observation for the PhySIC adapter."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


IMAGE_SOURCE_TO_REL_PATHS = {
    "dslr_resized_undistorted": {
        "image": "dslr/resized_undistorted_images",
        "transforms": "dslr/nerfstudio/transforms_undistorted.json",
        "poses": "dslr/colmap/images.txt",
    }
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {path}, got {type(mesh)!r}.")
    return mesh


def _colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.asarray(
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


def _load_colmap_pose(path: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float32)
        tvec = np.asarray([float(value) for value in parts[5:8]], dtype=np.float32)
        return _colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(f"Camera {camera_name!r} was not found in {path}.")


def _target_size(path: Path, max_img_size: int) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if max(width, height) <= int(max_img_size):
        return width, height
    scale = float(max_img_size) / float(max(width, height))
    return int(width * scale), int(height * scale)


def _scale_intrinsics(
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    target_width, target_height = target_size
    scaled = np.asarray(intrinsics, dtype=np.float32).copy()
    scaled[0, :] *= float(target_width) / float(source_width)
    scaled[1, :] *= float(target_height) / float(source_height)
    scaled[2, :] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return scaled


def _filter_face_ids_to_camera_view(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float = 20.0,
    border_px: float = 96.0,
    chunk_size: int = 500_000,
) -> np.ndarray:
    z_vertices = vertices_camera[:, 2]
    z_safe = np.clip(z_vertices, 1e-6, None)
    u_vertices = (
        intrinsics[0, 0] * vertices_camera[:, 0] / z_safe
        + intrinsics[0, 2]
        - 0.5
    )
    v_vertices = (
        intrinsics[1, 1] * vertices_camera[:, 1] / z_safe
        + intrinsics[1, 2]
        - 0.5
    )
    kept: list[np.ndarray] = []
    for start in range(0, faces.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), faces.shape[0])
        face_chunk = faces[start:stop]
        z = z_vertices[face_chunk]
        positive = np.any(z > 1e-6, axis=1) & np.any(z < max_depth_m, axis=1)
        u = u_vertices[face_chunk]
        v = v_vertices[face_chunk]
        overlaps = (
            positive
            & (np.max(u, axis=1) >= -border_px)
            & (np.min(u, axis=1) <= float(width - 1) + border_px)
            & (np.max(v, axis=1) >= -border_px)
            & (np.min(v, axis=1) <= float(height - 1) + border_px)
        )
        if np.any(overlaps):
            kept.append(np.nonzero(overlaps)[0].astype(np.int64) + start)
    if not kept:
        return np.empty((0,), dtype=np.int64)
    return np.concatenate(kept)


def _compact_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    return (
        vertices[vertex_ids].astype(np.float32),
        inverse.reshape(-1, 3).astype(np.int64),
        colors[vertex_ids].astype(np.uint8),
        vertex_ids.astype(np.int64),
    )


def _rasterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    previous_default_dtype = torch.get_default_dtype()
    try:
        # Some loaded foundation models change PyTorch's process-wide default
        # dtype. Camera construction and GT rasterization must remain float32.
        torch.set_default_dtype(torch.float32)
        with torch.autocast(device_type="cuda", enabled=False):
            rotation = torch.eye(3, dtype=torch.float32, device=device)[None]
            translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
            camera_matrix = torch.from_numpy(intrinsics.astype(np.float32))[None].to(
                device=device, dtype=torch.float32
            )
            image_size = torch.tensor(
                [[height, width]], dtype=torch.float32, device=device
            )
            camera = cameras_from_opencv_projection(
                R=rotation,
                tvec=translation,
                camera_matrix=camera_matrix,
                image_size=image_size,
            )
            mesh = Meshes(
                verts=[
                    torch.from_numpy(vertices).to(
                        device=device, dtype=torch.float32
                    )
                ],
                faces=[torch.from_numpy(faces).to(device=device, dtype=torch.int64)],
            )
            rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=(height, width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    bin_size=None,
                    max_faces_per_bin=400_000,
                ),
            )
            with torch.no_grad():
                fragments = rasterizer(mesh)
    finally:
        torch.set_default_dtype(previous_default_dtype)
    face_ids = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy().astype(np.int64)
    depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    depth[face_ids < 0] = 0.0
    del fragments, rasterizer, mesh, camera
    torch.cuda.empty_cache()
    return depth, face_ids


def _unproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij",
    )
    points = np.full((height, width, 3), np.nan, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    # PyTorch3D's OpenCV camera conversion rasterizes at half-integer pixel
    # centers; this matches the -0.5 convention used by Module 05 filtering.
    points[valid, 0] = (x[valid] + 0.5 - intrinsics[0, 2]) * z / intrinsics[0, 0]
    points[valid, 1] = (y[valid] + 0.5 - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points[valid, 2] = z
    valid &= np.isfinite(points).all(axis=-1)
    points[~valid] = np.nan
    return points, valid


def _validate_depth_geometry(
    points: np.ndarray,
    valid: np.ndarray,
    face_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    tolerance_m: float,
    max_samples: int,
    seed: int,
) -> dict[str, Any]:
    rows, cols = np.nonzero(valid & (face_ids >= 0))
    if rows.size == 0:
        raise RuntimeError("GT rasterization produced no valid depth pixels.")
    count = min(int(max_samples), int(rows.size))
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(rows.size, size=count, replace=False)
    rows = rows[selected]
    cols = cols[selected]
    sampled_points = points[rows, cols]

    z = sampled_points[:, 2]
    reproj_u = intrinsics[0, 0] * sampled_points[:, 0] / z + intrinsics[0, 2]
    reproj_v = intrinsics[1, 1] * sampled_points[:, 1] / z + intrinsics[1, 2]
    reprojection = np.sqrt(
        (reproj_u - (cols + 0.5)) ** 2 + (reproj_v - (rows + 0.5)) ** 2
    )
    if not np.isfinite(reprojection).all() or float(reprojection.max()) > 1e-3:
        raise RuntimeError(
            "GT depth unprojection/reprojection failed: "
            f"max error={float(reprojection.max()):.6g}px."
        )

    triangles = vertices[faces[face_ids[rows, cols]]].astype(np.float64)
    closest = trimesh.triangles.closest_point(triangles, sampled_points.astype(np.float64))
    distances = np.linalg.norm(closest - sampled_points, axis=1)
    if not np.isfinite(distances).all():
        raise RuntimeError("GT point-to-source-triangle validation produced non-finite errors.")
    max_distance = float(distances.max())
    if max_distance > float(tolerance_m):
        raise RuntimeError(
            "GT rendered-depth geometry disagrees with its source mesh: "
            f"max sampled distance={max_distance:.6f}m, "
            f"tolerance={float(tolerance_m):.6f}m."
        )
    return {
        "sample_count": int(count),
        "tolerance_m": float(tolerance_m),
        "reprojection_max_px": float(reprojection.max()),
        "point_to_triangle_mean_m": float(distances.mean()),
        "point_to_triangle_median_m": float(np.median(distances)),
        "point_to_triangle_p95_m": float(np.percentile(distances, 95.0)),
        "point_to_triangle_max_m": max_distance,
    }


def _visible_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    visible_face_ids = np.unique(face_ids[face_ids >= 0])
    if visible_face_ids.size == 0:
        raise RuntimeError("No ScanNet++ mesh faces were visible in the GT render.")
    visible_faces = faces[visible_face_ids]
    compact_vertices, compact_faces, compact_colors, _ = _compact_mesh(
        vertices,
        visible_faces,
        colors,
    )
    return compact_vertices, compact_faces, compact_colors


def build_scannet_gt_observation(
    *,
    project_dir: Path,
    scannet_root: Path,
    interaction_name: str,
    human_image_path: Path,
    scene_image_path: Path,
    max_img_size: int,
    device: torch.device,
    validation_tolerance_m: float = 0.005,
    validation_samples: int = 4096,
    seed: int = 24017,
) -> dict[str, Any]:
    input_scene_path = (
        project_dir / "01_Generate_SIG" / "input_prompts" / interaction_name / "input_scene.json"
    )
    contact_spec_path = (
        project_dir / "03_Estimate_Contact_Agentic" / "output" / interaction_name / "contact_spec.json"
    )
    full_human_image_path = (
        project_dir
        / "02_Generate_Human_Frame"
        / "output"
        / interaction_name
        / "inpainted_frame_resized.png"
    )
    scene_context = _load_json(input_scene_path)["scene_context"]
    scene_id = str(scene_context["scene_id"])
    camera = scene_context["camera"]
    camera_source = str(camera["source"])
    camera_name = str(camera["name"])
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(f"Unsupported ScanNet++ camera source: {camera_source!r}.")

    rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    transforms_path = scene_root / rel["transforms"]
    poses_path = scene_root / rel["poses"]
    mesh_path = scene_root / "scans" / "mesh_aligned_0.05.ply"
    source_image_path = scene_root / rel["image"] / camera_name
    required = [
        input_scene_path,
        contact_spec_path,
        full_human_image_path,
        scene_image_path,
        transforms_path,
        poses_path,
        mesh_path,
        source_image_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing ScanNet++ GT input(s): " + "; ".join(missing))

    transforms = _load_json(transforms_path)
    full_intrinsics = np.asarray(
        [
            [float(transforms["fl_x"]), 0.0, float(transforms["cx"])],
            [0.0, float(transforms["fl_y"]), float(transforms["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    crop_intrinsics = np.asarray(
        _load_json(contact_spec_path)["camera"]["intrinsics_3x3"],
        dtype=np.float32,
    )
    if not np.allclose(crop_intrinsics[:2, :2], full_intrinsics[:2, :2], atol=1e-4):
        raise ValueError("Contact-crop focal lengths do not match ScanNet++ metadata.")

    full_size = (int(transforms["w"]), int(transforms["h"]))
    with Image.open(full_human_image_path) as full_human_image:
        if full_human_image.size != full_size:
            raise ValueError(
                "Full human frame does not match the ScanNet++ camera image size: "
                f"{full_human_image.size} vs {full_size}."
            )

    with Image.open(human_image_path) as human_image:
        source_size = human_image.size
    with Image.open(scene_image_path) as clean_image:
        if clean_image.size != source_size:
            raise ValueError(
                f"Human and clean scene crops differ: {source_size} vs {clean_image.size}."
            )
        target_size = _target_size(human_image_path, max_img_size)
        scene_image = clean_image.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
    intrinsics = _scale_intrinsics(crop_intrinsics, source_size, target_size)
    width, height = target_size
    crop_offset = np.asarray(
        [
            full_intrinsics[0, 2] - crop_intrinsics[0, 2],
            full_intrinsics[1, 2] - crop_intrinsics[1, 2],
        ],
        dtype=np.float32,
    )
    crop_offset_int = np.rint(crop_offset).astype(np.int64)
    if not np.allclose(crop_offset, crop_offset_int, atol=1e-4):
        raise ValueError(f"Non-integral contact crop offset: {crop_offset.tolist()}.")
    crop_x, crop_y = crop_offset_int.tolist()
    if (
        crop_x < 0
        or crop_y < 0
        or crop_x + source_size[0] > full_size[0]
        or crop_y + source_size[1] > full_size[1]
    ):
        raise ValueError(
            f"Contact crop lies outside the full human frame: offset={(crop_x, crop_y)}, "
            f"crop={source_size}, full={full_size}."
        )
    with Image.open(full_human_image_path) as full_human_image, Image.open(
        human_image_path
    ) as crop_human_image:
        expected_crop = full_human_image.convert("RGB").crop(
            (crop_x, crop_y, crop_x + source_size[0], crop_y + source_size[1])
        )
        if not np.array_equal(
            np.asarray(expected_crop), np.asarray(crop_human_image.convert("RGB"))
        ):
            raise ValueError(
                "The cropped human frame is not an exact crop of the full aligned frame."
            )

    rotation_w2c, translation_w2c = _load_colmap_pose(poses_path, camera_name)
    mesh = _load_mesh(mesh_path)
    vertices_world = np.asarray(mesh.vertices, dtype=np.float32)
    faces_world = np.asarray(mesh.faces, dtype=np.int64)
    visual_colors = getattr(mesh.visual, "vertex_colors", None)
    if visual_colors is None or len(visual_colors) != vertices_world.shape[0]:
        colors_world = np.tile(np.asarray([188, 188, 188], dtype=np.uint8), (vertices_world.shape[0], 1))
    else:
        colors_world = np.asarray(visual_colors, dtype=np.uint8)[:, :3]
    vertices_camera = vertices_world @ rotation_w2c.T + translation_w2c[None]

    frustum_face_ids = _filter_face_ids_to_camera_view(
        vertices_camera,
        faces_world,
        intrinsics,
        width,
        height,
    )
    if frustum_face_ids.size == 0:
        raise RuntimeError("No ScanNet++ faces remained after camera-view filtering.")
    frustum_vertices, frustum_faces, frustum_colors, _ = _compact_mesh(
        vertices_camera,
        faces_world[frustum_face_ids],
        colors_world,
    )
    depth, raster_face_ids = _rasterize(
        frustum_vertices,
        frustum_faces,
        intrinsics,
        width,
        height,
        device,
    )
    points, valid = _unproject_depth(depth, intrinsics)
    validation = _validate_depth_geometry(
        points,
        valid,
        raster_face_ids,
        frustum_vertices,
        frustum_faces,
        intrinsics,
        validation_tolerance_m,
        validation_samples,
        seed,
    )
    visible_vertices_camera, visible_faces, visible_colors = _visible_mesh(
        frustum_vertices,
        frustum_faces,
        frustum_colors,
        raster_face_ids,
    )
    visible_vertices_world = (
        visible_vertices_camera - translation_w2c[None]
    ) @ rotation_w2c

    return {
        "scene_image": np.asarray(scene_image, dtype=np.uint8),
        "camera_hmr_initialization": {
            "image_path": str(full_human_image_path),
            "intrinsics": full_intrinsics,
            "full_size": list(full_size),
            "crop_source_size": list(source_size),
            "crop_offset_xy": crop_offset.astype(np.float32),
        },
        "depth": depth,
        "K": intrinsics,
        "points": points,
        "valid_mask": valid,
        "raster_face_ids": raster_face_ids,
        "visible_vertices_camera": visible_vertices_camera,
        "visible_vertices_world": visible_vertices_world.astype(np.float32),
        "visible_faces": visible_faces,
        "visible_colors": visible_colors,
        "rotation_world_to_camera": rotation_w2c,
        "translation_world_to_camera": translation_w2c,
        "validation": validation,
        "alignment_seed": int(seed),
        "metadata": {
            "interaction_name": interaction_name,
            "scene_id": scene_id,
            "camera_name": camera_name,
            "camera_source": camera_source,
            "source_image_path": str(source_image_path),
            "full_human_image_path": str(full_human_image_path),
            "clean_crop_path": str(scene_image_path),
            "mesh_path": str(mesh_path),
            "transforms_path": str(transforms_path),
            "poses_path": str(poses_path),
            "source_size": list(source_size),
            "target_size": list(target_size),
            "crop_offset_xy": crop_offset.astype(float).tolist(),
            "frustum_face_count": int(frustum_faces.shape[0]),
            "visible_face_count": int(visible_faces.shape[0]),
            "raw_valid_point_count": int(valid.sum()),
        },
    }
