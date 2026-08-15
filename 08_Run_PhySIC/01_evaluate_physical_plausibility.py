#!/usr/bin/env python3
"""Evaluate PhySIC outputs against their camera-frame scene observation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    METRIC_CSV_FIELDNAMES,
    aggregate_physical_rows,
    compute_mesh_penetration,
    configure_physic_imports,
    contact_segment_id_for_part,
    discover_physic_interactions,
    ensure_dir,
    image_size_from_original,
    load_mesh,
    load_scene_predictions,
    load_sig_edges,
    load_smplx_segments,
    mask_vertex_ids_for_part,
    nearest_distance_stats,
    physical_summary_row,
    physic_eval_root,
    physic_interaction_root,
    save_csv_rows,
    save_json,
    write_contact_debug_scene,
    write_evaluation_artifacts,
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
            "Evaluate contact and penetration for PhySIC using its serialized "
            "scene observation (visible ScanNet++ GT geometry in scannet mode)."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--physic_output_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--physic_root", type=str, default=None)
    parser.add_argument("--non_collision_surface_samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=24017)
    return parser.parse_args()


def resolve_roots(
    interaction_name: str,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    if args.physic_output_root:
        raw_root = Path(args.physic_output_root).resolve()
        if raw_root.name == "original":
            original_dir = raw_root
            interaction_root = raw_root.parent
        elif (raw_root / "original" / "scene_data_final.pkl").exists():
            interaction_root = raw_root
            original_dir = raw_root / "original"
        else:
            interaction_root = raw_root / interaction_name
            original_dir = interaction_root / "original"
    else:
        interaction_root = physic_interaction_root(interaction_name, args.output_mode)
        original_dir = interaction_root / "original"
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode)
        / interaction_name
        / "physical_plausibility"
    )
    return interaction_root, original_dir, output_root


def evaluate_interaction(interaction_name: str, args: argparse.Namespace) -> dict[str, float | int | str]:
    physic_root = configure_physic_imports(
        Path(args.physic_root).resolve() if args.physic_root else None
    )
    interaction_root, original_dir, output_root = resolve_roots(interaction_name, args)
    if not (original_dir / "scene_data_final.pkl").exists():
        raise FileNotFoundError(f"Missing PhySIC original output: {original_dir}")

    write_evaluation_artifacts(
        original_dir=original_dir,
        interaction_root=interaction_root,
        physic_root=physic_root,
    )
    meshes_dir = interaction_root / "meshes"
    scene_mesh = load_mesh(meshes_dir / "scene_camera.ply")
    human_mesh = load_mesh(meshes_dir / "human_camera.ply")
    predictions = load_scene_predictions(original_dir)
    intrinsics = np.asarray(predictions["K"], dtype=np.float32)
    image_hw = image_size_from_original(original_dir)

    segment_catalog = load_smplx_segments()
    segments = segment_catalog["segments"]
    if human_mesh.vertices.shape[0] != int(segment_catalog["vertex_count"]):
        raise ValueError(
            "PhySIC human vertex count does not match SMPL-X segmentation: "
            f"{human_mesh.vertices.shape[0]} vs {segment_catalog['vertex_count']}"
        )

    part_to_vertex_ids: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | str]] = []
    for edge in load_sig_edges(interaction_name):
        human_part = edge["human_part"]
        scene_ids = mask_vertex_ids_for_part(
            interaction_name=interaction_name,
            human_part=human_part,
            scene_vertices=np.asarray(scene_mesh.vertices, dtype=np.float32),
            intrinsics=intrinsics,
            image_hw=image_hw,
        )
        if scene_ids.shape[0] == 0:
            raise RuntimeError(
                f"GT contact mask for '{human_part}' did not hit any PhySIC scene vertices."
            )
        part_to_vertex_ids[human_part] = scene_ids
        segment_id = contact_segment_id_for_part(human_part)
        human_ids = segments[segment_id]
        stats = nearest_distance_stats(
            source_points=np.asarray(human_mesh.vertices[human_ids], dtype=np.float32),
            target_points=np.asarray(scene_mesh.vertices[scene_ids], dtype=np.float32),
        )
        metric_rows.append(
            {
                "node_a": edge["node_a"],
                "node_b": edge["node_b"],
                **stats,
            }
        )

    collision = compute_mesh_penetration(
        scene_mesh=scene_mesh,
        human_mesh=human_mesh,
        num_samples=int(args.non_collision_surface_samples),
        seed=int(args.seed),
    )
    csv_rows = [{**row, **collision} for row in metric_rows]
    ensure_dir(output_root)
    save_csv_rows(output_root / "metrics.csv", csv_rows, METRIC_CSV_FIELDNAMES)
    save_json(
        output_root / "metrics.json",
        [
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
    )
    write_contact_debug_scene(
        interaction_name=interaction_name,
        scene_mesh=scene_mesh,
        part_to_vertex_ids=part_to_vertex_ids,
        output_dir=output_root,
    )
    summary = physical_summary_row(interaction_name, csv_rows)
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
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    summaries = [evaluate_interaction(name, args) for name in interaction_names]
    if all_mode:
        combined_root = ensure_dir(physic_eval_root(args.output_mode))
        aggregate_row = aggregate_physical_rows(summaries)
        combined_rows = summaries + [aggregate_row]
        save_csv_rows(
            combined_root / "physical_plausibility.csv",
            combined_rows,
            COMBINED_CSV_FIELDNAMES,
        )
        save_json(
            combined_root / "physical_plausibility.json",
            {
                "interactions": summaries,
                "aggregate": {
                    "num_interactions": len(summaries),
                    "mean_ncs": aggregate_row["ncs"],
                    "mean_of_mean_min_contact_distance_m": aggregate_row[
                        "mean_min_contact_distance_m"
                    ],
                    "mean_of_mean_max_contact_distance_m": aggregate_row[
                        "mean_max_contact_distance_m"
                    ],
                    "mean_of_mean_contact_distance_m": aggregate_row[
                        "mean_contact_distance_m"
                    ],
                    "mean_penetration_m": aggregate_row["mean_penetration_m"],
                    "mean_max_penetration_m": aggregate_row["max_penetration_m"],
                },
            },
        )


if __name__ == "__main__":
    main()
