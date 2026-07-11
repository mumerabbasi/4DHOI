#!/usr/bin/env python3
"""Register PhySIC geometry to Module 06 and replay its exact cameras.

The registration has two stages:

1. Pair PhySIC per-pixel 3D points with ray hits on the Module 06 scene mesh
   using the exact Module 06 view_00 camera and resize-aware pixel centers.
2. Estimate a rigid scene transform and refine it with robust multiscale ICP.

The rigid transform is applied to both the PhySIC scene and human without
changing either shape. The registered PhySIC geometry is rendered with Module
06 cameras reused byte-for-byte. Outputs are isolated under ``evaluation_c``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    SCRIPT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_json,
    load_python_module,
    physic_interaction_root,
    save_json,
)


LEGACY_RENDERER = load_python_module(
    "module09_legacy_render_for_scene_registration",
    SCRIPT_DIR / "03a_render_interaction.py",
)
BASE_RENDERER = LEGACY_RENDERER.BASE_RENDERER
OPENCV_TO_BLENDER = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
DEFAULT_REGISTRATION_SAMPLES = 60000
MIN_PIXEL_RAY_HITS = 1000
ICP_VOXEL_M = 0.05
ICP_DISTANCE_SCHEDULE_M = (1.0, 0.6, 0.35, 0.2)


def evaluation_c_root(output_mode: str) -> Path:
    return SCRIPT_DIR / "evaluation_c" / output_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register PhySIC geometry into Module 06 world coordinates and "
            "render with the exact Module 06 cameras."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--blender_bin", type=str, default=None)
    parser.add_argument("--gpu_index", type=str, default="1")
    parser.add_argument(
        "--registration_samples",
        type=int,
        default=DEFAULT_REGISTRATION_SAMPLES,
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Register meshes and write configs without launching Blender.",
    )
    return parser.parse_args()


def load_triangle_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected triangle mesh at {path}, got {type(loaded)!r}")
    return loaded


def find_reference_view(source_config: dict[str, Any]) -> dict[str, Any]:
    view = next(
        (
            candidate
            for candidate in source_config.get("views", [])
            if str(candidate.get("name")) == "view_00"
        ),
        None,
    )
    if view is None:
        raise ValueError("Module 06 render configuration has no view_00.")
    return view


def load_physic_pixel_points(
    scene_data_path: Path,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, Any]]:
    with scene_data_path.open("rb") as file_obj:
        scene_data = pickle.load(file_obj)
    mask = np.asarray(scene_data["inlier_mask"], dtype=bool)
    points = np.asarray(scene_data["pts3d"], dtype=np.float64)
    scale = float(np.asarray(scene_data["scale"]).reshape(()))
    if mask.ndim != 2 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"Unexpected PhySIC point layout: mask={mask.shape}, points={points.shape}"
        )
    if int(mask.sum()) != len(points):
        raise ValueError(
            "PhySIC pts3d must follow row-major inlier-mask order; got "
            f"{len(points)} points for {int(mask.sum())} mask pixels."
        )
    points = points * scale
    pixel_rows_cols = np.argwhere(mask)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    pixel_rows_cols = pixel_rows_cols[finite]
    return points, pixel_rows_cols, tuple(mask.shape), {
        "physic_scene_scale": scale,
        "physic_intrinsics": np.asarray(scene_data["K"], dtype=np.float64).tolist(),
    }


def module06_world_rays_for_physic_pixels(
    pixel_rows_cols: np.ndarray,
    physic_shape: tuple[int, int],
    reference_view: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    physic_height, physic_width = physic_shape
    source_width = int(reference_view["width"])
    source_height = int(reference_view["height"])
    intrinsics = np.asarray(reference_view["intrinsics"], dtype=np.float64)
    camera_world = np.asarray(
        reference_view["camera_matrix_world"], dtype=np.float64
    )
    if intrinsics.shape != (3, 3) or camera_world.shape != (4, 4):
        raise ValueError("Module 06 view_00 camera or intrinsics are malformed.")

    # PIL resized the source scene image into the PhySIC processing resolution.
    # Invert that resize at pixel centers rather than treating integer corners
    # as corresponding locations.
    source_u = (
        (pixel_rows_cols[:, 1].astype(np.float64) + 0.5)
        * float(source_width)
        / float(physic_width)
        - 0.5
    )
    source_v = (
        (pixel_rows_cols[:, 0].astype(np.float64) + 0.5)
        * float(source_height)
        / float(physic_height)
        - 0.5
    )
    directions_camera = np.stack(
        [
            (source_u + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
            (source_v + 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(source_u),
        ],
        axis=1,
    )
    directions_camera /= np.linalg.norm(directions_camera, axis=1, keepdims=True)
    rotation_camera_to_world = camera_world[:3, :3] @ OPENCV_TO_BLENDER
    directions_world = directions_camera @ rotation_camera_to_world.T
    origins_world = np.repeat(
        camera_world[None, :3, 3], len(directions_world), axis=0
    )
    return origins_world, directions_world, {
        "physic_width": int(physic_width),
        "physic_height": int(physic_height),
        "module06_width": int(source_width),
        "module06_height": int(source_height),
        "pixel_scale_x": float(source_width) / float(physic_width),
        "pixel_scale_y": float(source_height) / float(physic_height),
    }


def raycast_module06_scene(
    scene_mesh_path: Path,
    origins_world: np.ndarray,
    directions_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    legacy_mesh = o3d.io.read_triangle_mesh(str(scene_mesh_path))
    if legacy_mesh.is_empty():
        raise ValueError(f"Could not load Module 06 scene mesh: {scene_mesh_path}")
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh))
    rays = np.concatenate([origins_world, directions_world], axis=1).astype(np.float32)
    hit_distance = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    valid = np.isfinite(hit_distance)
    hits = (
        origins_world[valid]
        + directions_world[valid] * hit_distance[valid, None].astype(np.float64)
    )
    return hits, valid


def robust_rigid_pixel_registration(
    physic_points: np.ndarray,
    module06_points: np.ndarray,
) -> dict[str, Any]:
    if len(physic_points) < MIN_PIXEL_RAY_HITS:
        raise ValueError(
            f"Only {len(physic_points)} pixel-ray correspondences; "
            f"need at least {MIN_PIXEL_RAY_HITS}."
        )
    inliers = np.ones(len(physic_points), dtype=bool)
    for _iteration in range(8):
        source = physic_points[inliers]
        target = module06_points[inliers]
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        u_matrix, _singular_values, vt_matrix = np.linalg.svd(
            (target - target_mean).T @ (source - source_mean)
        )
        signs = np.ones(3, dtype=np.float64)
        if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
            signs[-1] = -1.0
        rotation = u_matrix @ np.diag(signs) @ vt_matrix
        translation = target_mean - rotation @ source_mean
        predicted = physic_points @ rotation.T + translation[None]
        errors = np.linalg.norm(predicted - module06_points, axis=1)
        median = float(np.median(errors[inliers]))
        mad = float(np.median(np.abs(errors[inliers] - median)))
        threshold = max(0.10, median + 2.5 * max(mad, 1e-6))
        updated = errors <= threshold
        if int(updated.sum()) < MIN_PIXEL_RAY_HITS:
            break
        if np.array_equal(updated, inliers):
            inliers = updated
            break
        inliers = updated
    source = physic_points[inliers]
    target = module06_points[inliers]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(
        (target - target_mean).T @ (source - source_mean)
    )
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        signs[-1] = -1.0
    rotation = u_matrix @ np.diag(signs) @ vt_matrix
    translation = target_mean - rotation @ source_mean
    predicted = physic_points @ rotation.T + translation[None]
    errors = np.linalg.norm(predicted - module06_points, axis=1)
    return {
        "rotation": rotation,
        "translation": translation,
        "pixel_inlier_mask": inliers,
        "pixel_rmse_m": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "pixel_median_error_m": float(np.median(errors[inliers])),
    }


def refine_rigid_registration_with_icp(
    physic_scene_points: np.ndarray,
    module06_scene_path: Path,
    initial: dict[str, Any],
) -> dict[str, Any]:
    source_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(physic_scene_points)
    ).voxel_down_sample(ICP_VOXEL_M)
    target_mesh = o3d.io.read_triangle_mesh(str(module06_scene_path))
    target_cloud = o3d.geometry.PointCloud(target_mesh.vertices).voxel_down_sample(
        ICP_VOXEL_M
    )
    target_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.20, max_nn=40)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(initial["rotation"], dtype=np.float64)
    transform[:3, 3] = np.asarray(initial["translation"], dtype=np.float64)
    stages: list[dict[str, float]] = []
    for max_distance in ICP_DISTANCE_SCHEDULE_M:
        loss = o3d.pipelines.registration.TukeyLoss(
            k=max(0.12, 0.4 * max_distance)
        )
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            loss
        )
        result = o3d.pipelines.registration.registration_icp(
            source_cloud,
            target_cloud,
            max_distance,
            transform,
            estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
        )
        transform = np.asarray(result.transformation, dtype=np.float64)
        stages.append(
            {
                "max_correspondence_m": float(max_distance),
                "fitness": float(result.fitness),
                "inlier_rmse_m": float(result.inlier_rmse),
            }
        )
    rotation = transform[:3, :3]
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("Rigid ICP returned a non-rigid rotation matrix.")
    return {
        "rotation": rotation,
        "translation": transform[:3, 3],
        "icp_stages": stages,
        "fitness": stages[-1]["fitness"],
        "inlier_rmse_m": stages[-1]["inlier_rmse_m"],
        "source_point_count": int(len(source_cloud.points)),
        "target_point_count": int(len(target_cloud.points)),
    }


def estimate_scene_registration(
    scene_data_path: Path,
    module06_scene_path: Path,
    reference_view: dict[str, Any],
    max_samples: int,
) -> dict[str, Any]:
    physic_points, pixel_rows_cols, physic_shape, physic_metadata = (
        load_physic_pixel_points(scene_data_path)
    )
    sample_count = min(int(max_samples), len(physic_points))
    if sample_count < MIN_PIXEL_RAY_HITS:
        raise ValueError(f"registration_samples must be at least {MIN_PIXEL_RAY_HITS}.")
    rng = np.random.default_rng(20260711)
    sample_indices = rng.choice(len(physic_points), sample_count, replace=False)
    sampled_physic = physic_points[sample_indices]
    sampled_pixels = pixel_rows_cols[sample_indices]
    origins, directions, resize_metadata = module06_world_rays_for_physic_pixels(
        sampled_pixels, physic_shape, reference_view
    )
    module06_hits, hit_mask = raycast_module06_scene(
        module06_scene_path, origins, directions
    )
    paired_physic = sampled_physic[hit_mask]
    initial = robust_rigid_pixel_registration(
        physic_points=paired_physic, module06_points=module06_hits
    )
    registration = refine_rigid_registration_with_icp(
        physic_scene_points=physic_points,
        module06_scene_path=module06_scene_path,
        initial=initial,
    )
    return {
        **registration,
        "pixel_sample_count": int(sample_count),
        "pixel_ray_hit_count": int(len(module06_hits)),
        "pixel_ray_hit_fraction": float(len(module06_hits) / sample_count),
        "pixel_registration_inlier_count": int(
            np.count_nonzero(initial["pixel_inlier_mask"])
        ),
        "pixel_registration_rmse_m": float(initial["pixel_rmse_m"]),
        "pixel_registration_median_error_m": float(
            initial["pixel_median_error_m"]
        ),
        "resize_mapping": resize_metadata,
        **physic_metadata,
    }


def transform_and_export_mesh(
    source_path: Path,
    output_path: Path,
    registration: dict[str, Any],
) -> None:
    mesh = load_triangle_mesh(source_path).copy()
    rotation = np.asarray(registration["rotation"], dtype=np.float64)
    translation = np.asarray(registration["translation"], dtype=np.float64)
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T + translation[None]
    )
    ensure_dir(output_path.parent)
    mesh.export(output_path)


def registration_metadata(registration: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(registration)
    metadata["rotation_physic_to_module06_world"] = np.asarray(
        metadata.pop("rotation"), dtype=np.float64
    ).tolist()
    metadata["translation_physic_to_module06_world"] = np.asarray(
        metadata.pop("translation"), dtype=np.float64
    ).tolist()
    metadata["method"] = "robust_pixel_rigid_initialization_then_multiscale_rigid_icp"
    metadata["transform_type"] = "rigid_se3_no_scale_no_shear"
    return metadata


def build_registered_render_config(
    interaction_name: str,
    interaction_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, Path, Path]:
    source_config_path = (
        PROJECT_DIR
        / "06_Evaluate_Interaction"
        / "output"
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    source_config = load_json(source_config_path)
    source_views = source_config.get("views", [])
    if not source_views:
        raise ValueError(f"No Module 06 views in {source_config_path}")
    reference_view = find_reference_view(source_config)
    module06_scene_path = Path(source_config["scene_crop_ply"]).resolve()
    physic_scene_path = interaction_root / "meshes" / "scene_camera.ply"
    physic_human_path = interaction_root / "meshes" / "human_camera.ply"
    scene_data_path = interaction_root / "original" / "scene_data_final.pkl"
    required = (
        source_config_path,
        module06_scene_path,
        physic_scene_path,
        physic_human_path,
        scene_data_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing scene-registration input(s): " + "; ".join(missing))

    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    for stale_render in renders_dir.glob("view_*.png"):
        stale_render.unlink()
    registered_scene_path = assets_dir / "physic_scene_registered_module06_world.ply"
    registered_human_path = assets_dir / "physic_human_registered_module06_world.ply"
    registration = estimate_scene_registration(
        scene_data_path=scene_data_path,
        module06_scene_path=module06_scene_path,
        reference_view=reference_view,
        max_samples=int(args.registration_samples),
    )
    transform_and_export_mesh(physic_scene_path, registered_scene_path, registration)
    transform_and_export_mesh(physic_human_path, registered_human_path, registration)

    replay_views: list[dict[str, Any]] = []
    for index, source_view in enumerate(source_views):
        view = dict(source_view)
        view_name = str(
            source_view.get("name")
            or Path(str(source_view.get("render_path", ""))).stem
            or f"view_{index:02d}"
        )
        view["name"] = view_name
        view["render_path"] = str((renders_dir / f"{view_name}.png").resolve())
        view["camera_transfer"] = "exact_module06_camera_after_scene_registration"
        # camera_matrix_world, intrinsics, width, and height intentionally remain
        # unchanged from the source Module 06 configuration.
        replay_views.append(view)

    metadata = registration_metadata(registration)
    replay_config = dict(source_config)
    replay_config.update(
        {
            "scene_crop_ply": str(registered_scene_path.resolve()),
            "human_mesh_world": str(registered_human_path.resolve()),
            "blend_path": str((assets_dir / "render_scene.blend").resolve()),
            "views": replay_views,
            "camera_source": "exact_module06_cameras_after_physic_scene_registration",
            "camera_source_config_path": str(source_config_path.resolve()),
            "coordinate_frame": "module06_world",
            "scene_registration": metadata,
            "source_module06_scene_mesh": str(module06_scene_path),
            "source_physic_scene_mesh": str(physic_scene_path.resolve()),
            "source_physic_human_mesh": str(physic_human_path.resolve()),
            "registered_physic_scene_mesh": str(registered_scene_path.resolve()),
            "module06_scene_is_registration_target_only": True,
        }
    )
    return replay_config, source_config_path, registered_scene_path, registered_human_path


def render_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, Any]:
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else evaluation_c_root(args.output_mode) / interaction_name / "semantics"
    )
    assets_dir = ensure_dir(output_root / "assets")
    replay_config, source_config_path, registered_scene, registered_human = (
        build_registered_render_config(
            interaction_name=interaction_name,
            interaction_root=interaction_root,
            output_root=output_root,
            args=args,
        )
    )
    config_path = assets_dir / "render_config.json"
    driver_path = assets_dir / "render_driver.py"
    save_json(config_path, replay_config)
    BASE_RENDERER.write_blender_driver(driver_path)

    if not args.prepare_only:
        blender_bin = (
            Path(args.blender_bin).resolve()
            if args.blender_bin
            else Path(BASE_RENDERER.BLENDER_BIN).resolve()
        )
        if not blender_bin.exists():
            raise FileNotFoundError(f"Blender executable not found: {blender_bin}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        subprocess.run(
            [
                str(blender_bin),
                "--background",
                "--python",
                str(driver_path),
                "--",
                str(config_path),
            ],
            check=True,
            env=env,
        )

    selected_views_path = assets_dir / "selected_views.json"
    save_json(
        selected_views_path,
        {
            "interaction_name": interaction_name,
            "coordinate_frame": "module06_world",
            "source_render_config": str(source_config_path.resolve()),
            "registered_physic_scene_mesh": replay_config["scene_crop_ply"],
            "module06_scene_registration_target": replay_config[
                "source_module06_scene_mesh"
            ],
            "registered_human_mesh": str(registered_human),
            "camera_source": replay_config["camera_source"],
            "scene_registration": replay_config["scene_registration"],
            "views": replay_config["views"],
        },
    )
    action = "prepared" if args.prepare_only else "rendered"
    print(
        f"{interaction_name}: {action} {len(replay_config['views'])} exact-camera "
        f"views under {output_root}"
    )
    return {
        "interaction_name": interaction_name,
        "render_paths": [str(view["render_path"]) for view in replay_config["views"]],
        "scene_crop_ply": replay_config["scene_crop_ply"],
        "blend_path": str(assets_dir / "render_scene.blend"),
        "selected_views_path": str(selected_views_path),
        "prompt_path": str(
            PROJECT_DIR
            / "01_Generate_SIG"
            / "input_prompts"
            / interaction_name
            / "input_scene.json"
        ),
    }


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError("--all_interactions cannot be combined with --output_root.")
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]
    records = [render_interaction(name, args) for name in interaction_names]
    if len(records) > 1:
        save_json(
            evaluation_c_root(args.output_mode) / "semantics_renders.json",
            records,
        )


if __name__ == "__main__":
    main()
