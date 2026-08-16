#!/usr/bin/env python3
"""Render PhySIC humans by replacing only the human in Module 06's blend."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Any

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_json,
    physic_eval_root,
    physic_interaction_root,
    save_json,
)


DEFAULT_BLENDER_BIN = Path("/my_workspace/blender-4.2.17-linux-x64/blender")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a PhySIC human in Module 06's exact Blender scene, changing "
            "only the human mesh."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--blender_bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument("--gpu_index", default="0")
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def build_render_config(
    interaction_name: str,
    interaction_root: Path,
    output_root: Path,
) -> dict[str, Any]:
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
    source_config = load_json(source_config_path)
    artifacts = load_json(artifacts_path)

    source_blend = Path(source_config["blend_path"]).resolve()
    scene_world = Path(source_config["scene_crop_ply"]).resolve()
    source_human = Path(source_config["human_mesh_world"]).resolve()
    physic_human = Path(artifacts["human_world_mesh"]).resolve()
    missing = [
        str(path)
        for path in (
            source_config_path,
            artifacts_path,
            source_blend,
            scene_world,
            source_human,
            physic_human,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing render input(s): " + "; ".join(missing))

    renders_dir = ensure_dir(output_root / "renders")
    views = []
    for view in source_config["views"]:
        replay_view = dict(view)
        replay_view["render_path"] = str(
            (renders_dir / f"{view['name']}.png").resolve()
        )
        views.append(replay_view)

    config = dict(source_config)
    config.update(
        {
            "interaction_name": interaction_name,
            "source_render_config": str(source_config_path.resolve()),
            "source_blend_path": str(source_blend),
            "source_human_mesh_world": str(source_human),
            "human_mesh_world": str(physic_human),
            "blend_path": str((output_root / "assets" / "render_scene.blend").resolve()),
            "views": views,
            "coordinate_frame": "scannet_world",
            "human_replacement_only": True,
        }
    )
    return config


def write_blender_driver(path: Path) -> None:
    path.write_text(
        r'''
import json
import sys
from pathlib import Path

import bpy


def import_ply(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    imported = list(set(bpy.context.scene.objects) - before)
    if len(imported) != 1:
        raise RuntimeError(f"Expected one imported PLY object, got {len(imported)}")
    return imported[0]


def enable_cycles_gpu():
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            if any(device.type != "CPU" for device in preferences.devices):
                for device in preferences.devices:
                    device.use = device.type != "CPU"
                scene.cycles.device = "GPU"
                return
        except Exception:
            pass


config_path = Path(sys.argv[sys.argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

human = bpy.data.objects.get("optimized_human")
if human is None or human.type != "MESH":
    raise RuntimeError("Module 06 blend does not contain the optimized_human mesh")

old_mesh = human.data
materials = list(old_mesh.materials)
replacement = import_ply(config["human_mesh_world"])
human.data = replacement.data
bpy.data.objects.remove(replacement, do_unlink=True)
human.data.materials.clear()
for material in materials:
    human.data.materials.append(material)
if old_mesh.users == 0:
    bpy.data.meshes.remove(old_mesh)

enable_cycles_gpu()
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])

default_width = int(config["width"])
default_height = int(config["height"])
default_percentage = int(config["resolution_percentage"])
for view in config["views"]:
    camera = bpy.data.objects.get(view["name"])
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Missing Module 06 camera: {view['name']}")
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = int(view.get("width", default_width))
    bpy.context.scene.render.resolution_y = int(view.get("height", default_height))
    bpy.context.scene.render.resolution_percentage = int(
        view.get("resolution_percentage", default_percentage)
    )
    bpy.context.scene.render.filepath = view["render_path"]
    bpy.ops.render.render(write_still=True)
'''.lstrip(),
        encoding="utf-8",
    )


def render_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, Any]:
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    output_root = ensure_dir(
        args.output_root.resolve()
        if args.output_root is not None
        else physic_eval_root(args.output_mode) / interaction_name / "semantics"
    )
    assets_dir = ensure_dir(output_root / "assets")
    config = build_render_config(interaction_name, interaction_root, output_root)
    config_path = assets_dir / "render_config.json"
    driver_path = assets_dir / "render_driver.py"
    save_json(config_path, config)
    write_blender_driver(driver_path)

    if not args.prepare_only:
        blender_bin = args.blender_bin.resolve()
        if not blender_bin.exists():
            raise FileNotFoundError(f"Blender executable not found: {blender_bin}")
        for stale_render in (output_root / "renders").glob("view_*.png"):
            stale_render.unlink()
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        subprocess.run(
            [
                str(blender_bin),
                "--background",
                config["source_blend_path"],
                "--python",
                str(driver_path),
                "--",
                str(config_path),
            ],
            check=True,
            env=environment,
        )

    selected_views_path = assets_dir / "selected_views.json"
    save_json(
        selected_views_path,
        {
            "interaction_name": interaction_name,
            "source_render_config": config["source_render_config"],
            "source_blend_path": config["source_blend_path"],
            "scene_world_mesh": config["scene_crop_ply"],
            "source_human_mesh_world": config["source_human_mesh_world"],
            "physic_human_mesh_world": config["human_mesh_world"],
            "human_replacement_only": True,
            "views": config["views"],
        },
    )
    action = "prepared" if args.prepare_only else "rendered"
    print(f"{interaction_name}: {action} {len(config['views'])} views")
    return {
        "interaction_name": interaction_name,
        "render_paths": [view["render_path"] for view in config["views"]],
        "scene_crop_ply": config["scene_crop_ply"],
        "blend_path": config["blend_path"],
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
    names = (
        discover_physic_interactions(args.output_mode)
        if args.all_interactions or args.interaction_name == "all"
        else [args.interaction_name]
    )
    if len(names) > 1 and args.output_root is not None:
        raise ValueError("--output_root can only be used with one interaction")
    rows = [render_interaction(name, args) for name in names]
    if len(names) > 1:
        save_json(
            physic_eval_root(args.output_mode) / "semantics_renders.json",
            {row["interaction_name"]: row for row in rows},
        )


if __name__ == "__main__":
    main()
