#!/usr/bin/env python3
"""Render GT-scene PhySIC humans directly in the ScanNet++ world frame."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

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


BASE_RENDERER = load_python_module(
    "module08_gt_world_renderer",
    PROJECT_DIR / "06_Evaluate_Interaction" / "03a_render_interaction.py",
)


def evaluation_c_root(output_mode: str) -> Path:
    return SCRIPT_DIR / "evaluation_c" / output_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render ScanNet++ GT-scene PhySIC outputs with the exact Module 06 "
            "world-frame scene and cameras."
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
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def build_direct_render_config(
    interaction_name: str,
    interaction_root: Path,
    output_root: Path,
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
    artifacts_path = interaction_root / "metadata" / "artifacts.json"
    required = [source_config_path, artifacts_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing GT-world render input(s): " + "; ".join(missing))

    source_config = load_json(source_config_path)
    artifacts = load_json(artifacts_path)
    if artifacts.get("protocol") != "scannet_gt_visible_depth_aligned_v2":
        raise ValueError(
            f"Expected a ScanNet++ GT-scene artifact manifest: {artifacts_path}"
        )
    human_world_path = Path(artifacts["human_world_mesh"]).resolve()
    scene_world_path = Path(source_config["scene_crop_ply"]).resolve()
    if not human_world_path.exists() or not scene_world_path.exists():
        raise FileNotFoundError(
            f"Missing world mesh: human={human_world_path}, scene={scene_world_path}"
        )

    assets_dir = ensure_dir(output_root / "assets")
    renders_dir = ensure_dir(output_root / "renders")
    for stale_render in renders_dir.glob("view_*.png"):
        stale_render.unlink()

    replay_views: list[dict[str, Any]] = []
    for index, source_view in enumerate(source_config.get("views", [])):
        view = dict(source_view)
        view_name = str(
            source_view.get("name")
            or Path(str(source_view.get("render_path", ""))).stem
            or f"view_{index:02d}"
        )
        view["name"] = view_name
        view["render_path"] = str((renders_dir / f"{view_name}.png").resolve())
        view["camera_transfer"] = "exact_scannet_world_camera_no_registration"
        replay_views.append(view)
    if not replay_views:
        raise ValueError(f"No source evaluation views in {source_config_path}.")

    replay_config = dict(source_config)
    replay_config.update(
        {
            "scene_crop_ply": str(scene_world_path),
            "human_mesh_world": str(human_world_path),
            "blend_path": str((assets_dir / "render_scene.blend").resolve()),
            "views": replay_views,
            "camera_source": "exact_module06_scannet_world_cameras",
            "camera_source_config_path": str(source_config_path.resolve()),
            "coordinate_frame": "scannet_world",
            "scene_registration": None,
            "registration_applied": False,
            "source_physic_human_camera": artifacts["human_mesh"],
            "source_physic_human_world": str(human_world_path),
            "source_module06_scene_mesh": str(scene_world_path),
        }
    )
    return replay_config, source_config_path, scene_world_path, human_world_path


def render_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, Any]:
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else evaluation_c_root(args.output_mode) / interaction_name / "semantics"
    )
    assets_dir = ensure_dir(output_root / "assets")
    replay_config, source_config_path, scene_world, human_world = (
        build_direct_render_config(interaction_name, interaction_root, output_root)
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
            "coordinate_frame": "scannet_world",
            "source_render_config": str(source_config_path.resolve()),
            "scene_world_mesh": str(scene_world),
            "human_world_mesh": str(human_world),
            "registration_applied": False,
            "camera_source": replay_config["camera_source"],
            "views": replay_config["views"],
        },
    )
    action = "prepared" if args.prepare_only else "rendered"
    print(f"{interaction_name}: {action} {len(replay_config['views'])} GT-world views")
    return {
        "interaction_name": interaction_name,
        "render_paths": [str(view["render_path"]) for view in replay_config["views"]],
        "scene_crop_ply": str(scene_world),
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
    rows = [render_interaction(name, args) for name in interaction_names]
    if all_mode:
        save_json(
            evaluation_c_root(args.output_mode) / "semantics_renders.json",
            {row["interaction_name"]: row for row in rows},
        )


if __name__ == "__main__":
    main()
