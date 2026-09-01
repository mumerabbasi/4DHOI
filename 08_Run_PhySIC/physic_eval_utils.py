#!/usr/bin/env python3
"""Utilities shared by PhySIC runner and PhySIC-native evaluators."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_MODE = "output_scannet"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def physic_output_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / output_mode


def physic_interaction_root(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> Path:
    return physic_output_root(output_mode) / interaction_name


def physic_eval_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / "evaluation" / output_mode


def discover_physic_interactions(output_mode: str = DEFAULT_OUTPUT_MODE) -> list[str]:
    root = physic_output_root(output_mode)
    names = [
        path.name
        for path in sorted(root.glob("interaction_*"))
        if (path / "original" / "scene_data_final.pkl").exists()
    ]
    if not names:
        raise RuntimeError(f"No PhySIC interactions found under {root}.")
    return names


def configure_physic_imports(physic_root: Path | None = None) -> Path:
    root = (physic_root or REPO_DIR / "Phy-SIC").resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    camera_hmr = root / "external" / "CameraHMR"
    if str(camera_hmr) not in sys.path:
        sys.path.insert(0, str(camera_hmr))
    os.chdir(root)
    return root


def load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_scene_predictions(original_dir: Path) -> dict[str, Any]:
    with (original_dir / "scene_data_final.pkl").open("rb") as file_obj:
        return pickle.load(file_obj)


def export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray | None = None) -> None:
    ensure_dir(path.parent)
    kwargs: dict[str, Any] = {"process": False}
    if colors is not None:
        color_array = np.asarray(colors)
        if color_array.max(initial=0) <= 1.0:
            color_array = np.clip(color_array * 255.0, 0, 255)
        kwargs["vertex_colors"] = color_array.astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, **kwargs)
    mesh.export(path)


def export_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    ensure_dir(path.parent)
    color_array = None
    if colors is not None:
        color_array = np.asarray(colors)
        if color_array.max(initial=0) <= 1.0:
            color_array = np.clip(color_array * 255.0, 0, 255)
        color_array = color_array.astype(np.uint8)
    trimesh.points.PointCloud(
        vertices=np.asarray(points, dtype=np.float32),
        colors=color_array,
    ).export(path)


def _axis_angle_from_rot6d(rot6d: Any) -> np.ndarray:
    from pytorch3d.transforms import matrix_to_axis_angle
    from utils.geometry import rot6d_to_rotmat

    tensor = torch.as_tensor(rot6d, dtype=torch.float32)
    rotmat = rot6d_to_rotmat(tensor)
    axis_angle = matrix_to_axis_angle(rotmat)
    return axis_angle.detach().cpu().numpy()


def save_optimized_params(original_dir: Path, params_path: Path) -> dict[str, Any]:
    predictions = load_scene_predictions(original_dir)
    body_params = predictions["body_params"]
    global_orient = _axis_angle_from_rot6d(body_params["global_orient"]).reshape(-1)
    body_pose = _axis_angle_from_rot6d(body_params["body_pose"]).reshape(-1)
    betas = np.asarray(body_params["betas"], dtype=np.float32).reshape(-1)
    transl = np.asarray(predictions["cam_trans"], dtype=np.float32).reshape(-1)

    payload: dict[str, Any] = {
        "transl": transl.tolist(),
        "global_orient": global_orient.tolist(),
        "body_pose": body_pose.tolist(),
        "betas": betas.tolist(),
        "scale": 1.0,
        "physic_scene_scale": float(np.asarray(predictions["scale"]).reshape(())),
    }
    for key in ("left_hand_pose", "right_hand_pose"):
        if key in body_params:
            payload[key] = _axis_angle_from_rot6d(body_params[key]).reshape(-1).tolist()
    ensure_dir(params_path.parent)
    torch.save(payload, params_path)
    return payload


def save_world_optimized_params(
    original_dir: Path,
    camera_payload: dict[str, Any],
    root_joint_camera_untranslated: np.ndarray,
    params_path: Path,
) -> dict[str, Any]:
    predictions = load_scene_predictions(original_dir)
    scannet_gt = predictions.get("scannet_gt")
    if not isinstance(scannet_gt, dict):
        raise KeyError("World SMPL-X output requires ScanNet++ GT metadata.")
    rotation_w2c = np.asarray(
        scannet_gt["rotation_world_to_camera"], dtype=np.float32
    )
    translation_w2c = np.asarray(
        scannet_gt["translation_world_to_camera"], dtype=np.float32
    ).reshape(3)
    if rotation_w2c.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 world-to-camera rotation, got {rotation_w2c.shape}.")

    from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

    orient_camera = torch.as_tensor(
        camera_payload["global_orient"], dtype=torch.float32
    ).reshape(1, 3)
    rotation_camera = axis_angle_to_matrix(orient_camera)[0]
    rotation_camera_to_world = torch.from_numpy(rotation_w2c.T).float()
    rotation_world = rotation_camera_to_world @ rotation_camera
    orient_world = matrix_to_axis_angle(rotation_world[None])[0]

    transl_camera = np.asarray(camera_payload["transl"], dtype=np.float32).reshape(3)
    root_joint = np.asarray(root_joint_camera_untranslated, dtype=np.float32).reshape(3)
    # SMPL-X applies global orientation about its root joint, not the coordinate
    # origin. Include that pivot when changing the global orientation so that the
    # world-frame parameters reproduce the directly transformed camera mesh.
    transl_world = (
        (root_joint + transl_camera - translation_w2c) @ rotation_w2c
        - root_joint
    )
    payload = dict(camera_payload)
    payload["transl"] = transl_world.astype(np.float32).tolist()
    payload["global_orient"] = orient_world.detach().cpu().numpy().tolist()
    payload["coordinate_frame"] = "scannet_world"
    ensure_dir(params_path.parent)
    torch.save(payload, params_path)
    return payload


def build_scene_mesh_from_predictions(
    original_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = load_scene_predictions(original_dir)
    scannet_gt = predictions.get("scannet_gt")
    if isinstance(scannet_gt, dict):
        return (
            np.asarray(scannet_gt["visible_vertices_camera"], dtype=np.float32),
            np.asarray(scannet_gt["visible_faces"], dtype=np.int64),
            np.asarray(scannet_gt["visible_colors"], dtype=np.float32) / 255.0,
        )
    scene_image = np.asarray(Image.open(original_dir / "scene_image.png").convert("RGB"))
    height, width = scene_image.shape[:2]
    points = np.full((height, width, 3), np.nan, dtype=np.float32)
    inlier_mask = np.asarray(predictions["inlier_mask"], dtype=bool)
    scale = float(np.asarray(predictions["scale"]).reshape(()))
    points[inlier_mask] = np.asarray(predictions["pts3d"], dtype=np.float32) * scale

    valid = inlier_mask & np.isfinite(points).all(axis=-1)
    flat_valid = valid.reshape(-1)
    vertex_map = np.full((height * width,), -1, dtype=np.int64)
    vertex_map[flat_valid] = np.arange(int(flat_valid.sum()), dtype=np.int64)
    vertices = points.reshape(-1, 3)[flat_valid].astype(np.float32)
    colors = (scene_image.reshape(-1, 3)[flat_valid].astype(np.float32) / 255.0)

    grid = np.arange(height * width, dtype=np.int64).reshape(height, width)
    tl = grid[:-1, :-1].reshape(-1)
    tr = grid[:-1, 1:].reshape(-1)
    bl = grid[1:, :-1].reshape(-1)
    br = grid[1:, 1:].reshape(-1)
    tri_a_ok = flat_valid[tl] & flat_valid[bl] & flat_valid[tr]
    tri_b_ok = flat_valid[tr] & flat_valid[bl] & flat_valid[br]
    tri_a = np.stack(
        [vertex_map[tl[tri_a_ok]], vertex_map[bl[tri_a_ok]], vertex_map[tr[tri_a_ok]]],
        axis=1,
    )
    tri_b = np.stack(
        [vertex_map[tr[tri_b_ok]], vertex_map[bl[tri_b_ok]], vertex_map[br[tri_b_ok]]],
        axis=1,
    )
    faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int64)
    return vertices, faces, colors


def load_human_mesh_from_predictions(
    original_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = load_scene_predictions(original_dir)
    vertices = np.asarray(predictions["human_vertices_camera"], dtype=np.float32)
    faces = np.asarray(predictions["human_faces"], dtype=np.int64)
    root_joint_untranslated = np.asarray(
        predictions["human_root_joint_untranslated"], dtype=np.float32
    )
    colors = np.tile(
        np.asarray([188 / 255.0, 188 / 255.0, 188 / 255.0], dtype=np.float32),
        (vertices.shape[0], 1),
    )
    return vertices, faces, colors, root_joint_untranslated


def write_evaluation_artifacts(
    original_dir: Path,
    interaction_root: Path,
    physic_root: Path | None = None,
) -> dict[str, str]:
    configure_physic_imports(physic_root)

    meshes_dir = ensure_dir(interaction_root / "meshes")
    params_dir = ensure_dir(interaction_root / "debug" / "params")
    metadata_dir = ensure_dir(interaction_root / "metadata")

    scene_vertices, scene_faces, scene_colors = build_scene_mesh_from_predictions(original_dir)
    human_vertices, human_faces, human_colors, root_joint = (
        load_human_mesh_from_predictions(original_dir)
    )

    scene_camera_path = meshes_dir / "scene_camera.ply"
    human_camera_path = meshes_dir / "human_camera.ply"
    export_mesh(scene_camera_path, scene_vertices, scene_faces, scene_colors)
    export_mesh(human_camera_path, human_vertices, human_faces, human_colors)
    params_payload = save_optimized_params(
        original_dir,
        params_dir / "optimized_frame_0000.pt",
    )
    predictions = load_scene_predictions(original_dir)
    scannet_gt = predictions.get("scannet_gt")
    coordinate_frame = "physic_camera"
    world_params_path: Path | None = None
    scene_world_path: Path | None = None
    human_world_path: Path | None = None
    raw_points_camera_path: Path | None = None
    raw_points_world_path: Path | None = None
    filtered_points_camera_path: Path | None = None
    filtered_points_world_path: Path | None = None
    if isinstance(scannet_gt, dict):
        rotation_w2c = np.asarray(
            scannet_gt["rotation_world_to_camera"], dtype=np.float32
        )
        translation_w2c = np.asarray(
            scannet_gt["translation_world_to_camera"], dtype=np.float32
        ).reshape(3)
        scene_world = np.asarray(
            scannet_gt["visible_vertices_world"], dtype=np.float32
        )
        human_world = (human_vertices - translation_w2c[None]) @ rotation_w2c
        scene_world_path = meshes_dir / "scene_world.ply"
        human_world_path = meshes_dir / "human_world.ply"
        export_mesh(scene_world_path, scene_world, scene_faces, scene_colors)
        export_mesh(human_world_path, human_world, human_faces, human_colors)
        raw_valid = np.asarray(scannet_gt["raw_valid_mask"], dtype=bool)
        raw_points_camera = np.asarray(
            scannet_gt["raw_point_map"], dtype=np.float32
        )[raw_valid]
        filtered_points_camera = np.asarray(
            scannet_gt["filtered_scene_points"], dtype=np.float32
        )
        scene_image = np.asarray(
            Image.open(original_dir / "scene_image.png").convert("RGB")
        )
        raw_colors = scene_image[raw_valid]
        final_inliers = np.asarray(predictions["inlier_mask"], dtype=bool)
        filtered_colors = scene_image[final_inliers]
        if filtered_colors.shape[0] != filtered_points_camera.shape[0]:
            raise ValueError(
                "Filtered GT scene points and image-space inlier mask disagree: "
                f"{filtered_points_camera.shape[0]} vs {filtered_colors.shape[0]}."
            )
        raw_points_world = (
            raw_points_camera - translation_w2c[None]
        ) @ rotation_w2c
        filtered_points_world = (
            filtered_points_camera - translation_w2c[None]
        ) @ rotation_w2c
        raw_points_camera_path = meshes_dir / "scene_raw_points_camera.ply"
        raw_points_world_path = meshes_dir / "scene_raw_points_world.ply"
        filtered_points_camera_path = meshes_dir / "scene_filtered_points_camera.ply"
        filtered_points_world_path = meshes_dir / "scene_filtered_points_world.ply"
        export_point_cloud(raw_points_camera_path, raw_points_camera, raw_colors)
        export_point_cloud(raw_points_world_path, raw_points_world, raw_colors)
        export_point_cloud(
            filtered_points_camera_path,
            filtered_points_camera,
            filtered_colors,
        )
        export_point_cloud(
            filtered_points_world_path,
            filtered_points_world,
            filtered_colors,
        )
        world_params_path = params_dir / "optimized_frame_0000_world.pt"
        save_world_optimized_params(
            original_dir,
            params_payload,
            root_joint,
            world_params_path,
        )
        coordinate_frame = "scannet_camera"

    metadata = {
        "original_dir": str(original_dir),
        "coordinate_frame": coordinate_frame,
        "scene_mesh": str(scene_camera_path),
        "human_mesh": str(human_camera_path),
        "optimized_params": str(params_dir / "optimized_frame_0000.pt"),
        "scale_note": "Scene scale is fixed at exactly 1.0.",
        "physic_scene_scale": params_payload["physic_scene_scale"],
    }
    if isinstance(scannet_gt, dict):
        metadata.update(
            {
                "scene_world_mesh": str(scene_world_path),
                "human_world_mesh": str(human_world_path),
                "raw_scene_points_camera": str(raw_points_camera_path),
                "raw_scene_points_world": str(raw_points_world_path),
                "filtered_scene_points_camera": str(filtered_points_camera_path),
                "filtered_scene_points_world": str(filtered_points_world_path),
                "optimized_params_world": str(world_params_path),
                "scene_source": scannet_gt["metadata"],
                "gt_depth_validation": scannet_gt["validation"],
                "moge_depth_alignment": scannet_gt["alignment"],
                "raw_gt_point_count": int(
                    np.asarray(scannet_gt["raw_valid_mask"], dtype=bool).sum()
                ),
                "filtered_gt_point_count": int(
                    np.asarray(scannet_gt["filtered_scene_points"]).shape[0]
                ),
                "run": scannet_gt.get("gpu", {}),
            }
        )
    manifest_path = metadata_dir / "artifacts.json"
    if manifest_path.exists():
        previous_metadata = load_json(manifest_path)
        if "run" in previous_metadata:
            metadata["run"] = previous_metadata["run"]
    save_json(manifest_path, metadata)
    return {key: str(value) for key, value in metadata.items()}
