#!/usr/bin/env python3
"""Run the ScanNet++ GT-scene PhySIC baseline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from physic_eval_utils import write_evaluation_artifacts


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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhySIC against visible ScanNet++ GT geometry."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--output_root", type=Path, default=SCRIPT_DIR / "output")
    parser.add_argument("--scannet_root", type=Path, default=REPO_DIR / "Scannet++" / "data")
    parser.add_argument("--physic_root", type=Path, default=REPO_DIR / "Phy-SIC")
    parser.add_argument("--seed", type=int, default=24017)
    return parser.parse_args(argv)


def interaction_names(run_all: bool, requested: str) -> list[str]:
    if not run_all:
        return [requested]
    root = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"
    names = [path.name for path in root.glob("interaction_*") if path.is_dir()]
    if not names:
        raise FileNotFoundError(f"No interactions found under {root}.")
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def interaction_inputs(interaction_name: str) -> tuple[Path, Path]:
    assets = (
        PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
    )
    human_image = assets / "reference_inpainted_crop.png"
    scene_image = assets / "target_scene_crop.png"
    missing = [str(path) for path in (human_image, scene_image) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing interaction input(s): " + "; ".join(missing))
    return human_image, scene_image


def replace_path_prefix(value, old_prefix: str, new_prefix: str):
    if isinstance(value, dict):
        return {
            key: replace_path_prefix(item, old_prefix, new_prefix)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_path_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix):]
    return value


def publish(staging_root: Path, final_root: Path) -> None:
    manifest_path = staging_root / "metadata" / "artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = replace_path_prefix(
        manifest,
        str(staging_root.resolve()),
        str(final_root.resolve()),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    backup = final_root.with_name(f".{final_root.name}.backup-{uuid.uuid4().hex}")
    if final_root.exists():
        os.replace(final_root, backup)
    try:
        os.replace(staging_root, final_root)
    except Exception:
        if backup.exists() and not final_root.exists():
            os.replace(backup, final_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def gpu_metadata(torch) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_torch_device": "cuda:0",
        "name": properties.name,
        "memory_total_mb": int(properties.total_memory / 1024**2),
    }


def serialize_result(result, observation, torch, physic_root: Path, started: float):
    with torch.no_grad(), torch.amp.autocast(enabled=False, device_type="cuda"):
        final_joints, _, final_vertices, _, _ = result()
    if final_vertices.shape[0] != 1:
        raise ValueError(
            f"Expected one human, but PhySIC returned {final_vertices.shape[0]}."
        )

    diagnostics = result.scannet_gt_diagnostics
    return {
        "depth": result.depth.cpu().numpy(),
        "K": result.K.cpu().numpy(),
        "pts3d": result.pts3d.cpu().numpy(),
        "inlier_mask": result.inlier_mask.cpu().numpy(),
        "scale": result.scale.detach().cpu().numpy(),
        "normals": result.normals.cpu().numpy(),
        "plane_points": result.plane_points.cpu().numpy(),
        "plane_normal": result.plane_normal.cpu().numpy(),
        "body_params": {
            key: value.detach().cpu().numpy()
            for key, value in result.body_params.items()
        },
        "cam_trans": result.cam_trans.detach().cpu().numpy(),
        "human_vertices_camera": final_vertices[0].float().cpu().numpy(),
        "human_faces": result.body_model.faces.copy(),
        "human_root_joint_untranslated": (
            final_joints[0, 0] - result.cam_trans[0]
        ).float().detach().cpu().numpy(),
        "scannet_gt": {
            "gt_depth": observation["depth"],
            "gt_intrinsics": observation["K"],
            "raw_point_map": observation["points"],
            "raw_valid_mask": observation["valid_mask"],
            "raster_face_ids": observation["raster_face_ids"],
            "visible_vertices_camera": observation["visible_vertices_camera"],
            "visible_vertices_world": observation["visible_vertices_world"],
            "visible_faces": observation["visible_faces"],
            "visible_colors": observation["visible_colors"],
            "rotation_world_to_camera": observation["rotation_world_to_camera"],
            "translation_world_to_camera": observation["translation_world_to_camera"],
            "validation": observation["validation"],
            "metadata": observation["metadata"],
            "camera_hmr_initialization": result.camera_hmr_initialization,
            "raw_moge_depth": diagnostics["raw_moge_depth"],
            "aligned_moge_depth": diagnostics["aligned_moge_depth"],
            "moge_valid_mask": diagnostics["moge_valid_mask"],
            "human_mask_undilated": diagnostics["human_mask_undilated"],
            "per_human_masks_undilated": diagnostics["per_human_masks_undilated"],
            "alignment_fit_mask": diagnostics["alignment_fit_mask"],
            "aligned_valid_mask": diagnostics["aligned_valid_mask"],
            "alignment": diagnostics["alignment"],
            "filtered_scene_points": result.pts3d.detach().cpu().numpy(),
            "filtered_scene_normals": result.normals.detach().cpu().numpy(),
            "point_inlier_mask": result.inlier_mask_pts.detach().cpu().numpy(),
            "normal_inlier_mask": result.inlier_mask_normals.detach().cpu().numpy(),
            "human_target_points": result.pts_humans.detach().cpu().numpy(),
            "human_target_lengths": result.pts_human_lengths.detach().cpu().numpy(),
            "gpu": {
                **gpu_metadata(torch),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "python_executable": sys.executable,
                "physic_root": str(physic_root),
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
                "runtime_seconds_to_serialization": float(time.time() - started),
            },
        },
    }


def run_interaction(
    interaction_name: str,
    staging_root: Path,
    scannet_root: Path,
    physic_root: Path,
    cfg,
    torch,
    HumanScene,
    seed: int,
) -> None:
    started = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    human_image, scene_image = interaction_inputs(interaction_name)
    original_dir = staging_root / "original"
    original_dir.mkdir(parents=True)

    observation = build_scannet_gt_observation(
        project_dir=PROJECT_DIR,
        scannet_root=scannet_root,
        interaction_name=interaction_name,
        human_image_path=human_image,
        scene_image_path=scene_image,
        max_img_size=int(cfg.max_img_size),
        device=torch.device("cuda:0"),
        seed=seed,
    )
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        result = HumanScene(
            cfg,
            image_path=str(human_image),
            output_path=original_dir,
            scene_observation=observation,
        )
    if result.scale.requires_grad or float(result.scale.item()) != 1.0:
        raise RuntimeError("ScanNet++ scene scale must remain fixed at 1.0.")

    data = serialize_result(result, observation, torch, physic_root, started)
    with (original_dir / "scene_data_final.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    write_evaluation_artifacts(original_dir, staging_root, physic_root)

    manifest_path = staging_root / "metadata" / "artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"] = {
        **data["scannet_gt"]["gpu"],
        "seed": seed,
        "total_runtime_seconds": float(time.time() - started),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    scannet_root = args.scannet_root.resolve()
    physic_root = args.physic_root.resolve()
    names = interaction_names(args.all_interactions, args.interaction_name)
    for name in names:
        interaction_inputs(name)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir(physic_root)
    sys.path[:0] = [str(physic_root), str(physic_root / "external" / "CameraHMR")]

    import torch
    from omegaconf import OmegaConf
    from optimizer import (
        HumanScene,
        load_chmr,
        load_deco,
        load_gsam,
        load_moge,
        load_vitpose,
        load_wilor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("PhySIC requires CUDA, but CUDA is unavailable.")

    cfg = OmegaConf.load(physic_root / "cfg" / "v1.yaml")
    cfg.compute_floor_points = False
    for stage in ("opt_1", "opt_2", "opt_3"):
        cfg[stage].train_params = [
            name for name in cfg[stage].train_params if name != "scale"
        ]

    for loader in (load_gsam, load_vitpose, load_chmr, load_deco, load_wilor, load_moge):
        loader()
    torch.set_default_dtype(torch.float32)

    output_root.mkdir(parents=True, exist_ok=True)
    for interaction_name in names:
        final_root = output_root / interaction_name
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{interaction_name}.staging-", dir=output_root)
        )
        try:
            run_interaction(
                interaction_name,
                staging_root,
                scannet_root,
                physic_root,
                cfg,
                torch,
                HumanScene,
                args.seed,
            )
            publish(staging_root, final_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        print(f"{interaction_name}: {final_root}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
