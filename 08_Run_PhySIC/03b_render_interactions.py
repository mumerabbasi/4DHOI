#!/usr/bin/env python3
"""Render PhySIC geometry from torso-aligned Module 06 viewpoints.

This experimental renderer writes to ``evaluation_b`` and leaves the legacy
``03a`` renderer and its outputs untouched.  Camera azimuth and elevation are
transferred through a robust SMPL-X torso similarity transform.  Framing may
only change through a small optical-axis dolly.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

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
    "module09_legacy_render_interaction",
    SCRIPT_DIR / "03a_render_interaction.py",
)
BASE_RENDERER = LEGACY_RENDERER.BASE_RENDERER
NATIVE_PHYSIC_CAMERA = LEGACY_RENDERER.NATIVE_PHYSIC_CAMERA
FULL_SMPLX_SEGMENTS_PATH = (
    PROJECT_DIR.parent
    / "GVHMR"
    / "hmr4d"
    / "utils"
    / "body_model"
    / "smplx_vert_segmentation.json"
)
TORSO_SEGMENTS = (
    "hips",
    "spine",
    "spine1",
    "spine2",
    "neck",
    "leftShoulder",
    "rightShoulder",
)
MIN_DOLLY_SCALE = 0.85
MAX_DOLLY_SCALE = 1.15
NUM_DOLLY_SAMPLES = 61


def evaluation_b_root(output_mode: str) -> Path:
    return SCRIPT_DIR / "evaluation_b" / output_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render native PhySIC geometry using robust torso-aligned Module 06 "
            "cameras. Outputs are isolated under evaluation_b."
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
        "--preserve_native_view_zero",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep view_00 at the native PhySIC camera. By default every view, "
            "including view_00, is torso-aligned for strict Module 06 matching."
        ),
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Write render configs without launching Blender.",
    )
    return parser.parse_args()


def load_torso_vertex_indices(vertex_count: int) -> np.ndarray:
    if not FULL_SMPLX_SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Full SMPL-X segmentation is required: {FULL_SMPLX_SEGMENTS_PATH}"
        )
    segments = load_json(FULL_SMPLX_SEGMENTS_PATH)
    missing = [name for name in TORSO_SEGMENTS if name not in segments]
    if missing:
        raise KeyError(
            "Missing torso segment(s) in SMPL-X segmentation: " + ", ".join(missing)
        )
    indices = np.unique(
        np.concatenate(
            [np.asarray(segments[name], dtype=np.int64) for name in TORSO_SEGMENTS]
        )
    )
    if indices.size < 100 or int(indices.max()) >= vertex_count:
        raise ValueError(
            f"Invalid torso selection of {indices.size} vertices for a "
            f"{vertex_count}-vertex mesh."
        )
    return indices


def _umeyama_similarity(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return rotation, scale, translation for y = scale * R @ x + t."""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Similarity points must be matching Nx3 arrays; got "
            f"{source.shape} and {target.shape}."
        )
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance < 1e-12:
        raise ValueError("Source torso vertices have near-zero variance.")

    covariance = target_centered.T @ source_centered / float(len(source))
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(u_matrix @ vt_matrix) < 0.0:
        sign[-1] = -1.0
    rotation = u_matrix @ np.diag(sign) @ vt_matrix
    scale = float(np.sum(singular_values * sign) / source_variance)
    translation = target_mean - scale * rotation @ source_mean
    return rotation, scale, translation


def estimate_torso_similarity(
    source_vertices: np.ndarray,
    physic_vertices: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source_vertices, dtype=np.float64)
    target = np.asarray(physic_vertices, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(
            "Torso alignment requires corresponding SMPL-X topology; got "
            f"{source.shape} and {target.shape}."
        )
    torso_indices = load_torso_vertex_indices(len(source))
    source_torso = source[torso_indices]
    target_torso = target[torso_indices]
    inliers = np.ones(len(torso_indices), dtype=bool)

    # Iteratively discard only strong correspondence outliers. This keeps the
    # fit insensitive to local pose/cloth deviations without changing topology.
    for _iteration in range(4):
        rotation, scale, translation = _umeyama_similarity(
            source_torso[inliers], target_torso[inliers]
        )
        predicted = source_torso @ rotation.T * scale + translation[None]
        errors = np.linalg.norm(predicted - target_torso, axis=1)
        median = float(np.median(errors[inliers]))
        mad = float(np.median(np.abs(errors[inliers] - median)))
        threshold = max(0.025, median + 3.5 * max(mad, 1e-6))
        updated = errors <= threshold
        if int(updated.sum()) < max(100, int(0.65 * len(torso_indices))):
            break
        if np.array_equal(updated, inliers):
            inliers = updated
            break
        inliers = updated

    rotation, scale, translation = _umeyama_similarity(
        source_torso[inliers], target_torso[inliers]
    )
    predicted = source_torso @ rotation.T * scale + translation[None]
    errors = np.linalg.norm(predicted - target_torso, axis=1)
    return {
        "rotation": rotation,
        "scale": scale,
        "translation": translation,
        "torso_vertex_indices": torso_indices,
        "inlier_mask": inliers,
        "rmse_m": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "median_error_m": float(np.median(errors[inliers])),
        "max_inlier_error_m": float(errors[inliers].max()),
    }


def transform_point(point: np.ndarray, alignment: dict[str, Any]) -> np.ndarray:
    return (
        float(alignment["scale"])
        * np.asarray(alignment["rotation"], dtype=np.float64)
        @ np.asarray(point, dtype=np.float64)
        + np.asarray(alignment["translation"], dtype=np.float64)
    )


def source_image_center_target(
    source_view: dict[str, Any],
    source_focus: np.ndarray,
) -> np.ndarray:
    camera = np.asarray(source_view["camera_matrix_world"], dtype=np.float64)
    intrinsics = np.asarray(source_view["intrinsics"], dtype=np.float64)
    width = int(source_view["width"])
    height = int(source_view["height"])
    image_center_ray = np.array(
        [
            (0.5 * width - intrinsics[0, 2]) / intrinsics[0, 0],
            (0.5 * height - intrinsics[1, 2]) / intrinsics[1, 1],
            1.0,
        ],
        dtype=np.float64,
    )
    image_center_ray /= np.linalg.norm(image_center_ray)
    camera_to_source = camera[:3, :3] @ NATIVE_PHYSIC_CAMERA[:3, :3]
    ray_source = camera_to_source @ image_center_ray
    center = camera[:3, 3]
    depth = float(np.dot(source_focus - center, ray_source))
    if depth <= 1e-6:
        raise ValueError("Module 06 human focus is behind the source camera.")
    return center + depth * ray_source


def torso_aligned_camera(
    source_view: dict[str, Any],
    source_focus: np.ndarray,
    source_vertices: np.ndarray,
    physic_vertices: np.ndarray,
    alignment: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_camera = np.asarray(source_view["camera_matrix_world"], dtype=np.float64)
    intrinsics = np.asarray(source_view["intrinsics"], dtype=np.float64)
    width = int(source_view["width"])
    height = int(source_view["height"])
    source_bbox = LEGACY_RENDERER.project_human_bbox(
        source_vertices, source_camera, intrinsics, width, height
    )

    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    target = transform_point(
        source_image_center_target(source_view, source_focus), alignment
    )
    initial_center = transform_point(source_camera[:3, 3], alignment)
    center_delta = initial_center - target
    initial_radius = float(np.linalg.norm(center_delta))
    if initial_radius < 1e-6:
        raise ValueError("Transferred camera is at its optical target.")

    transferred_orientation = rotation @ source_camera[:3, :3]
    candidates: list[tuple[float, float, np.ndarray, dict[str, float]]] = []
    for dolly_scale in np.linspace(
        MIN_DOLLY_SCALE, MAX_DOLLY_SCALE, NUM_DOLLY_SAMPLES
    ):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = transferred_orientation
        matrix[:3, 3] = target + float(dolly_scale) * center_delta
        try:
            bbox = LEGACY_RENDERER.project_human_bbox(
                physic_vertices, matrix, intrinsics, width, height
            )
        except ValueError:
            continue
        fill_error = abs(float(bbox["max_fill"]) - float(source_bbox["max_fill"]))
        candidates.append((fill_error, float(dolly_scale), matrix, bbox))
    if not candidates:
        raise ValueError("No valid constrained-dolly camera keeps the human in front.")
    _error, dolly_scale, matrix, bbox = min(candidates, key=lambda item: item[0])
    minimum_margin = min(
        bbox["margin_left"],
        bbox["margin_right"],
        bbox["margin_top"],
        bbox["margin_bottom"],
    )
    return matrix, {
        "alignment": "robust_smplx_torso_similarity",
        "dolly_scale": dolly_scale,
        "dolly_was_clamped": bool(
            np.isclose(dolly_scale, MIN_DOLLY_SCALE)
            or np.isclose(dolly_scale, MAX_DOLLY_SCALE)
        ),
        "source_radius_m": initial_radius / float(alignment["scale"]),
        "physic_radius_m": float(dolly_scale * initial_radius),
        "source_body_fill": float(source_bbox["max_fill"]),
        "physic_body_fill": float(bbox["max_fill"]),
        "body_fill_error": float(bbox["max_fill"] - source_bbox["max_fill"]),
        "minimum_frame_margin": float(minimum_margin),
        "physic_optical_target_camera": target.tolist(),
    }


def build_replay_config(
    interaction_name: str,
    scene_camera_path: Path,
    human_camera_path: Path,
    scene_data_path: Path,
    output_root: Path,
    preserve_native_view_zero: bool,
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
            f"Module 06 render configuration is required: {source_config_path}"
        )
    source_config = load_json(source_config_path)
    source_views = source_config.get("views", [])
    if not source_views:
        raise ValueError(f"No Module 06 camera views in {source_config_path}")

    source_human_path = Path(source_config["human_mesh_world"]).resolve()
    source_vertices = BASE_RENDERER.load_mesh_vertices(source_human_path).astype(
        np.float64
    )
    physic_vertices = BASE_RENDERER.load_mesh_vertices(human_camera_path).astype(
        np.float64
    )
    source_focus = BASE_RENDERER.human_focus_point(source_vertices).astype(np.float64)
    alignment = estimate_torso_similarity(source_vertices, physic_vertices)
    native_intrinsics, native_width, native_height = (
        LEGACY_RENDERER.load_native_camera_model(scene_data_path)
    )

    renders_dir = ensure_dir(output_root / "renders")
    assets_dir = ensure_dir(output_root / "assets")
    for stale_render in renders_dir.glob("view_*.png"):
        stale_render.unlink()

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
        view["source_camera_matrix_world"] = source_view["camera_matrix_world"]
        if view_name == "view_00" and preserve_native_view_zero:
            view["camera_matrix_world"] = NATIVE_PHYSIC_CAMERA.tolist()
            view["intrinsics"] = native_intrinsics
            view["width"] = native_width
            view["height"] = native_height
            view["resolution_percentage"] = 100
            view["camera_transfer"] = "native_physic_source_camera"
        else:
            camera_physic, framing = torso_aligned_camera(
                source_view=source_view,
                source_focus=source_focus,
                source_vertices=source_vertices,
                physic_vertices=physic_vertices,
                alignment=alignment,
            )
            view["camera_matrix_world"] = camera_physic.tolist()
            view["camera_transfer"] = "module06_robust_torso_similarity"
            view["framing"] = framing
        replay_views.append(view)

    alignment_metadata = {
        "method": "robust_umeyama_smplx_torso",
        "segments": list(TORSO_SEGMENTS),
        "vertex_count": int(len(alignment["torso_vertex_indices"])),
        "inlier_count": int(np.count_nonzero(alignment["inlier_mask"])),
        "rotation_source_to_physic": alignment["rotation"].tolist(),
        "scale_source_to_physic": float(alignment["scale"]),
        "translation_source_to_physic": alignment["translation"].tolist(),
        "rmse_m": float(alignment["rmse_m"]),
        "median_error_m": float(alignment["median_error_m"]),
        "max_inlier_error_m": float(alignment["max_inlier_error_m"]),
        "dolly_scale_bounds": [MIN_DOLLY_SCALE, MAX_DOLLY_SCALE],
    }
    replay_config = dict(source_config)
    replay_config.update(
        {
            "scene_crop_ply": str(scene_camera_path.resolve()),
            "human_mesh_world": str(human_camera_path.resolve()),
            "blend_path": str((assets_dir / "render_scene.blend").resolve()),
            "views": replay_views,
            "camera_source": "module06_robust_torso_similarity",
            "camera_source_config_path": str(source_config_path.resolve()),
            "coordinate_frame": "physic_camera",
            "preserve_native_view_zero": bool(preserve_native_view_zero),
            "native_camera_matrix_world": NATIVE_PHYSIC_CAMERA.tolist(),
            "native_camera_intrinsics": native_intrinsics,
            "native_camera_width": native_width,
            "native_camera_height": native_height,
            "native_camera_source": str(scene_data_path.resolve()),
            "source_human_mesh": str(source_human_path),
            "torso_alignment": alignment_metadata,
            "source_scene_crop_ply": str(Path(source_config["scene_crop_ply"]).resolve()),
        }
    )
    return replay_config, source_config_path


def render_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, Any]:
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    scene_camera_path = interaction_root / "meshes" / "scene_camera.ply"
    human_camera_path = interaction_root / "meshes" / "human_camera.ply"
    scene_data_path = interaction_root / "original" / "scene_data_final.pkl"
    missing = [
        str(path)
        for path in (scene_camera_path, human_camera_path, scene_data_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing native PhySIC artifacts: " + "; ".join(missing))

    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else evaluation_b_root(args.output_mode) / interaction_name / "semantics"
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
        preserve_native_view_zero=bool(args.preserve_native_view_zero),
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
    metadata = {
        "interaction_name": interaction_name,
        "coordinate_frame": "physic_camera",
        "source_render_config": str(source_config_path.resolve()),
        "scene_camera_mesh": str(scene_camera_path.resolve()),
        "human_camera_mesh": str(human_camera_path.resolve()),
        "camera_source": replay_config["camera_source"],
        "preserve_native_view_zero": replay_config["preserve_native_view_zero"],
        "torso_alignment": replay_config["torso_alignment"],
        "views": replay_config["views"],
    }
    save_json(selected_views_path, metadata)
    action = "prepared" if args.prepare_only else "rendered"
    print(
        f"{interaction_name}: {action} {len(replay_config['views'])} torso-aligned "
        f"views under {output_root}"
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
            evaluation_b_root(args.output_mode) / "semantics_renders.json",
            records,
        )


if __name__ == "__main__":
    main()
