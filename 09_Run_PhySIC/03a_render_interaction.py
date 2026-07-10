#!/usr/bin/env python3
"""Render PhySIC reconstructed interactions for CLIP/VLM evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    configure_physic_imports,
    discover_physic_interactions,
    ensure_dir,
    load_mesh,
    orbit_camera_poses,
    physic_eval_root,
    physic_interaction_root,
    physic_original_dir,
    save_json,
    write_evaluation_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render PhySIC outputs from PhySIC-native reconstructed meshes."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--physic_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    return parser.parse_args()


def render_mesh(
    mesh_path: Path,
    output_root: Path,
    num_views: int,
    width: int,
    height: int,
) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    import trimesh

    mesh = load_mesh(mesh_path)
    render_mesh_obj = mesh.copy()
    if render_mesh_obj.visual.kind is None:
        render_mesh_obj.visual.vertex_colors = np.tile(
            np.asarray([190, 190, 190, 255], dtype=np.uint8),
            (render_mesh_obj.vertices.shape[0], 1),
        )

    render_scene = pyrender.Scene(
        bg_color=np.asarray([255, 255, 255, 255], dtype=np.uint8),
        ambient_light=np.asarray([0.35, 0.35, 0.35, 1.0], dtype=np.float32),
    )
    render_scene.add(
        pyrender.Mesh.from_trimesh(render_mesh_obj, smooth=False),
        pose=np.eye(4, dtype=np.float32),
    )
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

    renders_dir = ensure_dir(output_root / "renders")
    metadata = []
    poses = orbit_camera_poses(mesh, num_views=num_views)
    try:
        for item in poses:
            pose = np.asarray(item["pose"], dtype=np.float32)
            camera_node = render_scene.add(camera, pose=pose)
            light_node = render_scene.add(light, pose=pose)
            color, _depth = renderer.render(render_scene)
            render_scene.remove_node(camera_node)
            render_scene.remove_node(light_node)
            path = renders_dir / f"view_{int(item['view_id']):03d}.png"
            Image.fromarray(color).save(path)
            metadata.append({k: v for k, v in item.items() if k != "pose"} | {"path": str(path)})
    finally:
        renderer.delete()
    save_json(output_root / "render_metadata.json", metadata)


def render_interaction(interaction_name: str, args: argparse.Namespace) -> None:
    physic_root = configure_physic_imports(
        Path(args.physic_root).resolve() if args.physic_root else None
    )
    interaction_root = physic_interaction_root(interaction_name, args.output_mode)
    original_dir = physic_original_dir(interaction_name, args.output_mode)
    if not (original_dir / "scene_data_final.pkl").exists():
        raise FileNotFoundError(f"Missing PhySIC original output: {original_dir}")
    write_evaluation_artifacts(
        original_dir=original_dir,
        interaction_root=interaction_root,
        physic_root=physic_root,
    )
    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode) / interaction_name / "semantics"
    )
    render_mesh(
        mesh_path=interaction_root / "meshes" / "humanscene_camera.ply",
        output_root=output_root,
        num_views=int(args.num_views),
        width=int(args.width),
        height=int(args.height),
    )
    print(f"{interaction_name}: wrote PhySIC renders to {output_root / 'renders'}")


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError("--all_interactions cannot be combined with --output_root.")
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]
    for interaction_name in interaction_names:
        render_interaction(interaction_name, args)


if __name__ == "__main__":
    main()
