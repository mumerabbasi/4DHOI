#!/usr/bin/env python3
"""Evaluate PROX in Module 06's shared scene crop and source camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from prox_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    METRIC_CSV_FIELDNAMES,
    aggregate_physical_rows,
    compute_mesh_penetration,
    contact_segment_id_for_part,
    discover_prox_interactions,
    ensure_dir,
    load_mesh,
    load_sig_edges,
    load_smplx_segments,
    mask_vertex_ids_for_part,
    nearest_distance_stats,
    physical_summary_row,
    prox_eval_root,
    prox_interaction_root,
    save_csv_rows,
    save_json,
    shared_scene_camera,
    world_to_camera,
    write_contact_debug_scene,
)


COMBINED_CSV_FIELDNAMES = [
    "interaction_name",
    "num_edges",
    "mean_min_contact_distance_m",
    "mean_max_contact_distance_m",
    "mean_contact_distance_m",
    "ncs",
    "mean_penetration_m",
    "max_penetration_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PROX contact and penetration in Module 06's shared "
            "4DHSI scene crop."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_02")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--non_collision_surface_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=24017)
    return parser.parse_args()


def evaluate_interaction(
    interaction_name: str,
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    prox_root = prox_interaction_root(interaction_name, args.output_mode)
    human_world_path = prox_root / "final_smplx_world.ply"
    if not human_world_path.is_file():
        raise FileNotFoundError(f"Missing optimized PROX world mesh: {human_world_path}")
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else prox_eval_root(args.output_mode)
        / interaction_name
        / "physical_plausibility"
    )

    # Scene geometry and camera are exclusively the shared Module 06 artifacts.
    context = shared_scene_camera(interaction_name)
    scene_world = context["scene_world"]
    human_world = load_mesh(human_world_path)
    scene_vertices_camera = world_to_camera(np.asarray(scene_world.vertices), context)
    human_vertices_camera = world_to_camera(np.asarray(human_world.vertices), context)
    scene_camera = trimesh.Trimesh(
        vertices=scene_vertices_camera,
        faces=np.asarray(scene_world.faces),
        vertex_colors=np.asarray(scene_world.visual.vertex_colors),
        process=False,
    )
    human_camera = trimesh.Trimesh(
        vertices=human_vertices_camera,
        faces=np.asarray(human_world.faces),
        process=False,
    )

    segment_catalog = load_smplx_segments()
    if human_camera.vertices.shape[0] != segment_catalog["vertex_count"]:
        raise ValueError(
            "PROX mesh vertex count does not match SMPL-X segmentation: "
            f"{human_camera.vertices.shape[0]} vs {segment_catalog['vertex_count']}"
        )

    part_to_vertex_ids: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | str]] = []
    for edge in load_sig_edges(interaction_name):
        human_part = edge["human_part"]
        scene_ids = mask_vertex_ids_for_part(
            interaction_name=interaction_name,
            human_part=human_part,
            scene_vertices=scene_vertices_camera,
            intrinsics=context["intrinsics"],
            image_hw=(context["height"], context["width"]),
        )
        if scene_ids.size == 0:
            raise RuntimeError(
                f"GT contact mask for '{human_part}' did not hit any shared-scene vertices."
            )
        part_to_vertex_ids[human_part] = scene_ids
        human_ids = segment_catalog["segments"][contact_segment_id_for_part(human_part)]
        stats = nearest_distance_stats(
            source_points=human_vertices_camera[human_ids],
            target_points=scene_vertices_camera[scene_ids],
        )
        metric_rows.append(
            {"node_a": edge["node_a"], "node_b": edge["node_b"], **stats}
        )

    collision = compute_mesh_penetration(
        scene_mesh=scene_camera,
        human_mesh=human_camera,
        num_samples=int(args.non_collision_surface_samples),
        seed=int(args.seed),
    )
    rows = [{**row, **collision} for row in metric_rows]
    ensure_dir(output_root)
    save_csv_rows(output_root / "metrics.csv", rows, METRIC_CSV_FIELDNAMES)
    save_json(
        output_root / "metrics.json",
        {
            "interaction_name": interaction_name,
            "method_human_mesh": str(human_world_path),
            "scene_source": "Module 06 shared 4DHSI scene crop",
            "scene_world_mesh": str(context["scene_path"]),
            "camera_source": str(context["render_config_path"]),
            "metrics": [
                {
                    "node_a": row["node_a"],
                    "node_b": row["node_b"],
                    "contact": {
                        "min_distance_m": row["min_distance_m"],
                        "max_distance_m": row["max_distance_m"],
                        "mean_distance_m": row["mean_distance_m"],
                    },
                    "collision": collision,
                }
                for row in metric_rows
            ],
        },
    )
    write_contact_debug_scene(
        interaction_name,
        scene_camera,
        part_to_vertex_ids,
        output_root,
    )
    summary = physical_summary_row(interaction_name, rows)
    print(
        f"{interaction_name}: mean_contact={summary['mean_contact_distance_m']:.6f} "
        f"ncs={summary['ncs']:.6f} max_pen={summary['max_penetration_m']:.6f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError("--all_interactions cannot be combined with --output_root.")
        names = discover_prox_interactions(args.output_mode)
    else:
        names = [args.interaction_name]
    summaries = [evaluate_interaction(name, args) for name in names]
    if all_mode:
        root = ensure_dir(prox_eval_root(args.output_mode))
        aggregate = aggregate_physical_rows(summaries)
        save_csv_rows(
            root / "physical_plausibility.csv",
            summaries + [aggregate],
            COMBINED_CSV_FIELDNAMES,
        )
        save_json(
            root / "physical_plausibility.json",
            {
                "interactions": summaries,
                "aggregate": {
                    "num_interactions": len(summaries),
                    "mean_ncs": aggregate["ncs"],
                    "mean_of_mean_min_contact_distance_m": aggregate[
                        "mean_min_contact_distance_m"
                    ],
                    "mean_of_mean_max_contact_distance_m": aggregate[
                        "mean_max_contact_distance_m"
                    ],
                    "mean_of_mean_contact_distance_m": aggregate[
                        "mean_contact_distance_m"
                    ],
                    "mean_penetration_m": aggregate["mean_penetration_m"],
                    "mean_max_penetration_m": aggregate["max_penetration_m"],
                },
            },
        )


if __name__ == "__main__":
    main()
