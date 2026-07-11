#!/usr/bin/env python3
"""Render native PhySIC geometry from body-relative module 06 viewpoints."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_json,
    load_python_module,
    physic_eval_root,
    physic_interaction_root,
    save_json,
)


BASE_RENDERER = load_python_module(
    "module06_render_interaction",
    PROJECT_DIR / "06_Evaluate_Interaction" / "03a_render_interaction.py",
)
NATIVE_PHYSIC_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)
MIN_FRAME_MARGIN = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render native PhySIC geometry using module-06 body-relative views "
            "and the same Blender setup."
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
    return parser.parse_args()


def source_to_physic_rotation(
    reference_camera_matrix_world: list[list[float]],
) -> np.ndarray:
    reference_camera = np.asarray(reference_camera_matrix_world, dtype=np.float64)
    if reference_camera.shape != (4, 4):
        raise ValueError(
            "Module 06 reference camera matrix must be 4x4; got "
            f"{reference_camera.shape}."
        )
    return NATIVE_PHYSIC_CAMERA[:3, :3] @ reference_camera[:3, :3].T


def oriented_human_focus(
    vertices: np.ndarray,
    local_to_world_rotation: np.ndarray,
) -> tuple[np.ndarray, float]:
    vertices_local = np.asarray(vertices, dtype=np.float64) @ local_to_world_rotation
    lower = vertices_local.min(axis=0)
    upper = vertices_local.max(axis=0)
    focus_local = (lower + upper) * 0.5
    height = float(upper[2] - lower[2])
    focus_local[2] = lower[2] + 0.55 * height
    return focus_local @ local_to_world_rotation.T, height


def project_human_bbox(
    vertices: np.ndarray,
    camera_matrix_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    rotation, translation = BASE_RENDERER.camera_extrinsics_from_blender_matrix_world(
        camera_matrix_world
    )
    points_camera = (
        np.asarray(vertices, dtype=np.float64) @ rotation.astype(np.float64).T
        + translation.astype(np.float64)[None]
    )
    if np.any(points_camera[:, 2] <= 1e-5):
        raise ValueError("A body-centered camera places human vertices behind it.")
    u, v, _depth = BASE_RENDERER.project_camera_points_to_image(
        points_camera.astype(np.float32), intrinsics.astype(np.float32)
    )
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    return {
        "u_min": u_min,
        "u_max": u_max,
        "v_min": v_min,
        "v_max": v_max,
        "width_fill": (u_max - u_min) / float(width),
        "height_fill": (v_max - v_min) / float(height),
        "max_fill": max(
            (u_max - u_min) / float(width),
            (v_max - v_min) / float(height),
        ),
        "margin_left": u_min / float(width),
        "margin_right": (float(width - 1) - u_max) / float(width),
        "margin_top": v_min / float(height),
        "margin_bottom": (float(height - 1) - v_max) / float(height),
    }


def body_centered_camera(
    source_view: dict[str, Any],
    source_focus: np.ndarray,
    physic_focus: np.ndarray,
    frame_rotation: np.ndarray,
    body_scale: float,
    source_vertices: np.ndarray,
    physic_vertices: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    source_camera = np.asarray(source_view["camera_matrix_world"], dtype=np.float64)
    intrinsics = np.asarray(source_view["intrinsics"], dtype=np.float64)
    width = int(source_view["width"])
    height = int(source_view["height"])
    source_bbox = project_human_bbox(
        source_vertices, source_camera, intrinsics, width, height
    )
    target_fill = min(float(source_bbox["max_fill"]), 1.0 - 2.0 * MIN_FRAME_MARGIN)

    opencv_to_blender = NATIVE_PHYSIC_CAMERA[:3, :3]
    rotation_camera_to_source = source_camera[:3, :3] @ opencv_to_blender
    image_center_ray = np.array(
        [
            (float(width) * 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
            (float(height) * 0.5 - intrinsics[1, 2]) / intrinsics[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    image_center_ray /= np.linalg.norm(image_center_ray)
    image_center_ray_source = rotation_camera_to_source @ image_center_ray
    source_center = source_camera[:3, 3]
    target_depth = float(np.dot(source_focus - source_center, image_center_ray_source))
    source_target = source_center + image_center_ray_source * target_depth

    physic_target = physic_focus + body_scale * frame_rotation @ (
        source_target - source_focus
    )
    initial_center = physic_focus + body_scale * frame_rotation @ (
        source_center - source_focus
    )
    source_radius = float(np.linalg.norm(source_center - source_target))
    if source_radius < 1e-6:
        raise ValueError("Module 06 synthetic camera is at its optical target.")
    direction = initial_center - physic_target
    direction /= np.linalg.norm(direction)
    world_up = frame_rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def camera_at_radius(radius: float) -> tuple[np.ndarray, dict[str, float]]:
        center = physic_target + direction * float(radius)
        rotation, translation = BASE_RENDERER.look_at_world_to_camera_centered(
            camera_center=center.astype(np.float32),
            focus=physic_target.astype(np.float32),
            intrinsics=intrinsics.astype(np.float32),
            width=width,
            image_height=height,
            world_up=world_up.astype(np.float32),
        )
        matrix = BASE_RENDERER.blender_camera_matrix_world(rotation, translation).astype(
            np.float64
        )
        return matrix, project_human_bbox(
            physic_vertices, matrix, intrinsics, width, height
        )

    initial_radius = source_radius * body_scale
    lower = max(0.1, initial_radius * 0.25)
    upper = max(1.0, initial_radius * 2.0)
    _matrix, upper_bbox = camera_at_radius(upper)
    while upper_bbox["max_fill"] > target_fill:
        upper *= 1.5
        if upper > 20.0:
            raise ValueError("Could not frame the PhySIC human within 20 meters.")
        _matrix, upper_bbox = camera_at_radius(upper)
    for _iteration in range(40):
        radius = 0.5 * (lower + upper)
        _matrix, bbox = camera_at_radius(radius)
        if bbox["max_fill"] > target_fill:
            lower = radius
        else:
            upper = radius
    matrix, bbox = camera_at_radius(upper)
    minimum_margin = min(
        bbox["margin_left"],
        bbox["margin_right"],
        bbox["margin_top"],
        bbox["margin_bottom"],
    )
    return matrix, {
        "source_radius_m": source_radius,
        "physic_radius_m": float(upper),
        "source_body_fill": float(source_bbox["max_fill"]),
        "physic_body_fill": float(bbox["max_fill"]),
        "minimum_frame_margin": float(minimum_margin),
        "source_optical_target_world": source_target.tolist(),
        "physic_optical_target_camera": physic_target.tolist(),
    }


def load_native_camera_model(scene_data_path: Path) -> tuple[list[list[float]], int, int]:
    with scene_data_path.open("rb") as file_obj:
        scene_data = pickle.load(file_obj)
    intrinsics = np.asarray(scene_data["K"], dtype=np.float64)
    depth = np.asarray(scene_data["depth"])
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"PhySIC intrinsics must be 3x3; got {intrinsics.shape} in "
            f"{scene_data_path}."
        )
    if depth.ndim != 2:
        raise ValueError(
            f"PhySIC depth must be HxW; got {depth.shape} in {scene_data_path}."
        )
    height, width = depth.shape
    return intrinsics.tolist(), int(width), int(height)


def build_replay_config(
    interaction_name: str,
    scene_camera_path: Path,
    human_camera_path: Path,
    scene_data_path: Path,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    source_config_path = (
        PROJECT_DIR
        / "06_Evaluate_Interaction"
        / "output"
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    if not source_config_path.exists():
        raise FileNotFoundError(
            "Module 06 baseline render configuration is required for camera replay: "
            f"{source_config_path}"
        )
    source_config = load_json(source_config_path)
    source_views = source_config.get("views", [])
    reference_view = next(
        (view for view in source_views if str(view.get("name")) == "view_00"),
        None,
    )
    if reference_view is None:
        raise ValueError(f"Module 06 render config has no view_00: {source_config_path}")
    reference_camera = reference_view["camera_matrix_world"]
    native_intrinsics, native_width, native_height = load_native_camera_model(
        scene_data_path
    )
    source_human_path = Path(source_config["human_mesh_world"]).resolve()
    if not source_human_path.exists():
        raise FileNotFoundError(
            f"Module 06 human mesh is required for body-relative cameras: "
            f"{source_human_path}"
        )
    source_vertices = BASE_RENDERER.load_mesh_vertices(source_human_path).astype(
        np.float64
    )
    physic_vertices = BASE_RENDERER.load_mesh_vertices(human_camera_path).astype(
        np.float64
    )
    source_focus = BASE_RENDERER.human_focus_point(source_vertices).astype(np.float64)
    source_height = float(source_vertices[:, 2].max() - source_vertices[:, 2].min())
    frame_rotation = source_to_physic_rotation(reference_camera)
    physic_focus, physic_height = oriented_human_focus(
        physic_vertices, frame_rotation
    )
    if source_height <= 1e-6 or physic_height <= 1e-6:
        raise ValueError(
            f"Invalid human heights: module06={source_height}, physic={physic_height}."
        )
    body_scale = physic_height / source_height

    renders_dir = ensure_dir(output_root / "renders")
    assets_dir = ensure_dir(output_root / "assets")
    for stale_render in renders_dir.glob("view_*.png"):
        stale_render.unlink()

    replay_views = []
    for index, source_view in enumerate(source_views):
        view = dict(source_view)
        view_name = str(
            source_view.get("name")
            or Path(str(source_view.get("render_path", ""))).stem
            or f"view_{index:02d}"
        )
        view["name"] = view_name
        view["render_path"] = str((renders_dir / f"{view_name}.png").resolve())
        view["source_camera_matrix_world"] = source_view["camera_matrix_world"]
        if view_name == "view_00":
            view["camera_matrix_world"] = NATIVE_PHYSIC_CAMERA.tolist()
            view["intrinsics"] = native_intrinsics
            view["width"] = native_width
            view["height"] = native_height
            view["resolution_percentage"] = 100
            view["camera_transfer"] = "native_physic_source_camera"
        else:
            camera_physic, framing = body_centered_camera(
                source_view=source_view,
                source_focus=source_focus,
                physic_focus=physic_focus,
                frame_rotation=frame_rotation,
                body_scale=body_scale,
                source_vertices=source_vertices,
                physic_vertices=physic_vertices,
            )
            view["camera_matrix_world"] = camera_physic.tolist()
            view["camera_transfer"] = "module06_body_relative"
            view["framing"] = framing
        replay_views.append(view)
    if not replay_views:
        raise ValueError(f"No camera views found in {source_config_path}")

    replay_config = dict(source_config)
    replay_config.update(
        {
            "scene_crop_ply": str(scene_camera_path.resolve()),
            "human_mesh_world": str(human_camera_path.resolve()),
            "blend_path": str((assets_dir / "render_scene.blend").resolve()),
            "views": replay_views,
            "camera_source": "native_view00_and_module06_body_relative_views",
            "camera_source_config_path": str(source_config_path.resolve()),
            "coordinate_frame": "physic_camera",
            "native_camera_matrix_world": NATIVE_PHYSIC_CAMERA.tolist(),
            "native_camera_intrinsics": native_intrinsics,
            "native_camera_width": native_width,
            "native_camera_height": native_height,
            "native_camera_source": str(scene_data_path.resolve()),
            "source_human_mesh": str(source_human_path),
            "source_human_focus_world": source_focus.tolist(),
            "physic_human_focus_camera": physic_focus.tolist(),
            "source_human_height_m": source_height,
            "physic_human_height_m": physic_height,
            "body_height_scale": body_scale,
            "minimum_frame_margin": MIN_FRAME_MARGIN,
            "source_scene_crop_ply": str(Path(source_config["scene_crop_ply"]).resolve()),
        }
    )
    return replay_config, source_config_path


def render_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, Any]:
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    scene_camera_path = interaction_root / "meshes" / "scene_camera.ply"
    human_camera_path = interaction_root / "meshes" / "human_camera.ply"
    scene_data_path = interaction_root / "original" / "scene_data_final.pkl"
    required = [scene_camera_path, human_camera_path, scene_data_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Native PhySIC render artifacts are missing. Run 00_run_physic.py for "
            f"{interaction_name} in scannet mode. Missing: {'; '.join(missing)}"
        )

    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode) / interaction_name / "semantics"
    )
    assets_dir = ensure_dir(output_root / "assets")
    scene_crop_path = assets_dir / "scene_semantics_view_crop.ply"
    if scene_crop_path.exists():
        scene_crop_path.unlink()
    try:
        os.link(scene_camera_path, scene_crop_path)
    except OSError:
        shutil.copy2(scene_camera_path, scene_crop_path)
    replay_config, source_config_path = build_replay_config(
        interaction_name=interaction_name,
        scene_camera_path=scene_crop_path,
        human_camera_path=human_camera_path,
        scene_data_path=scene_data_path,
        output_root=output_root,
    )
    config_path = assets_dir / "render_config.json"
    driver_path = assets_dir / "render_driver.py"
    save_json(config_path, replay_config)
    BASE_RENDERER.write_blender_driver(driver_path)

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

    metadata = {
        "interaction_name": interaction_name,
        "coordinate_frame": "physic_camera",
        "scene_camera_mesh": str(scene_camera_path.resolve()),
        "human_camera_mesh": str(human_camera_path.resolve()),
        "source_render_config": str(source_config_path.resolve()),
        "camera_source": replay_config["camera_source"],
        "native_camera_matrix_world": replay_config["native_camera_matrix_world"],
        "native_camera_intrinsics": replay_config["native_camera_intrinsics"],
        "native_camera_width": replay_config["native_camera_width"],
        "native_camera_height": replay_config["native_camera_height"],
        "native_camera_source": replay_config["native_camera_source"],
        "source_human_mesh": replay_config["source_human_mesh"],
        "source_human_focus_world": replay_config["source_human_focus_world"],
        "physic_human_focus_camera": replay_config["physic_human_focus_camera"],
        "source_human_height_m": replay_config["source_human_height_m"],
        "physic_human_height_m": replay_config["physic_human_height_m"],
        "body_height_scale": replay_config["body_height_scale"],
        "minimum_frame_margin": replay_config["minimum_frame_margin"],
        "views": replay_config["views"],
    }
    selected_views_path = assets_dir / "selected_views.json"
    save_json(selected_views_path, metadata)
    print(
        f"{interaction_name}: wrote {len(replay_config['views'])} matched-view "
        f"native PhySIC renders to {output_root / 'renders'}"
    )
    return {
        "interaction_name": interaction_name,
        "render_paths": [str(view["render_path"]) for view in replay_config["views"]],
        "scene_crop_ply": str(scene_crop_path),
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
            physic_eval_root(args.output_mode) / "semantics_renders.json",
            records,
        )


if __name__ == "__main__":
    main()
