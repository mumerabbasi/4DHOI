"""Apply the aligned human transform to an entire human mesh sequence.

This script reads the human transform from 07_Align_Meshes outputs and applies it to
all .ply meshes in the human motion directory (the same directory where
01_align_meshes.py takes the first frame mesh).

Default behavior:
- Input human meshes: 06_Estimate_Human_Motion/output/<interaction_name>/humans/<person_x>/human_plys
- Transform source: 07_Align_Meshes/output/<interaction_name>/meshes/transforms.json
- Output meshes: 07_Align_Meshes/output/<interaction_name>/human_motion_aligned/<person_x>

Required transform key in transforms.json:
- source_to_output_matrix_4x4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from utils_align_meshes import load_json, resolve_path


def _as_4x4(matrix_like, key_name: str) -> np.ndarray:
    matrix = np.asarray(matrix_like, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected {key_name} to be shape (4,4), got {matrix.shape}")
    return matrix


def _apply_affine(vertices: np.ndarray, matrix_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to row-vector vertices."""
    rot = matrix_4x4[:3, :3]
    trans = matrix_4x4[:3, 3]
    return (vertices @ rot.T + trans).astype(np.float32)


def _resolve_human_transform_map(transforms_json_path: Path) -> dict[str, np.ndarray]:
    """Return all human source->output transform matrices keyed by slug/name."""
    transforms_data = load_json(transforms_json_path)
    transforms = transforms_data.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError(f"Invalid transforms payload in {transforms_json_path}")

    human_transforms: dict[str, np.ndarray] = {}
    for entry in transforms:
        if entry.get("kind") != "human" and entry.get("slug") != "human":
            continue
        if "source_to_output_matrix_4x4" not in entry:
            raise ValueError(
                "Human transform missing required key: source_to_output_matrix_4x4"
            )
        slug = str(entry.get("slug") or entry.get("name") or "human")
        human_transforms[slug] = _as_4x4(
            entry["source_to_output_matrix_4x4"],
            "source_to_output_matrix_4x4",
        )

    if not human_transforms:
        raise ValueError(f"No human transform entry found in {transforms_json_path}")
    return human_transforms


def _discover_input_human_dirs(humans_root: Path) -> list[tuple[str, Path]]:
    """Discover per-human input PLY directories from the new layout only."""
    if not humans_root.exists() or not humans_root.is_dir():
        raise NotADirectoryError(f"Human root directory not found: {humans_root}")

    discovered = [
        (person_dir.name, (person_dir / "human_plys").resolve())
        for person_dir in sorted(humans_root.iterdir())
        if person_dir.is_dir()
        and person_dir.name.startswith("person_")
        and (person_dir / "human_plys").is_dir()
    ]
    if not discovered:
        raise FileNotFoundError(
            f"No per-person human_plys directories found under: {humans_root}"
        )
    return discovered


def _resolve_transform_for_person(
    person_slug: str,
    human_transform_map: dict[str, np.ndarray],
) -> np.ndarray:
    """Match one per-person input sequence to the same per-person transform."""
    if person_slug not in human_transform_map:
        raise ValueError(
            f"Could not match human input '{person_slug}' to a transform entry. "
            f"Available transform keys: {sorted(human_transform_map.keys())}"
        )
    return human_transform_map[person_slug]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply aligned human transform to all human .ply meshes."
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--align_video_dir",
        type=str,
        default=None,
        help="07_Align_Meshes output interaction dir. Default: ./output/<interaction_name>",
    )
    parser.add_argument(
        "--human_video_dir",
        type=str,
        default=None,
        help=(
            "06_Estimate_Human_Motion output interaction dir. "
            "Default: ../06_Estimate_Human_Motion/output/<interaction_name>"
        ),
    )
    parser.add_argument(
        "--input_human_dir",
        type=str,
        default=None,
        help=(
            "Optional explicit human root directory. "
            "Default: <human_video_dir>/humans"
        ),
    )
    parser.add_argument(
        "--transforms_json",
        type=str,
        default=None,
        help="Optional explicit transforms.json path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output root directory. Default: "
            "<align_video_dir>/human_motion_aligned/<person_x>"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    align_video_dir = resolve_path(args.align_video_dir, script_dir) or (
        script_dir / "output" / args.interaction_name
    ).resolve()
    human_video_dir = resolve_path(args.human_video_dir, script_dir) or (
        script_dir.parent / "06_Estimate_Human_Motion" / "output" / args.interaction_name
    ).resolve()
    humans_root = resolve_path(args.input_human_dir, script_dir) or (
        human_video_dir / "humans"
    ).resolve()
    transforms_json_path = resolve_path(args.transforms_json, script_dir) or (
        align_video_dir / "meshes" / "transforms.json"
    ).resolve()
    output_root = resolve_path(args.output_dir, script_dir) or (
        align_video_dir / "human_motion_aligned"
    ).resolve()

    if not transforms_json_path.exists():
        raise FileNotFoundError(f"Transforms JSON not found: {transforms_json_path}")

    discovered_inputs = _discover_input_human_dirs(humans_root)
    human_transform_map = _resolve_human_transform_map(
        transforms_json_path=transforms_json_path
    )

    print(f"Transforms: {transforms_json_path}")
    print("Transform key: source_to_output_matrix_4x4")
    print(f"Discovered human inputs: {[slug for slug, _ in discovered_inputs]}")

    total_written = 0
    for input_slug, input_dir in discovered_inputs:
        ply_paths = sorted(input_dir.glob("*.ply"))
        if not ply_paths:
            raise FileNotFoundError(f"No .ply files found in {input_dir}")

        matrix = _resolve_transform_for_person(input_slug, human_transform_map)
        output_dir = (output_root / input_slug).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Input human meshes [{input_slug}]: {input_dir}")
        print(f"Saving aligned sequence [{input_slug}] to: {output_dir}")

        for mesh_path in ply_paths:
            mesh = trimesh.load(str(mesh_path), force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                raise ValueError(f"Failed to load mesh as Trimesh: {mesh_path}")

            verts = np.asarray(mesh.vertices, dtype=np.float32)
            verts_aligned = _apply_affine(verts, matrix)

            out_mesh = mesh.copy()
            out_mesh.vertices = verts_aligned
            out_path = output_dir / mesh_path.name
            out_mesh.export(str(out_path))
            total_written += 1

    print(f"Done. Wrote {total_written} aligned human meshes to {output_root}")


if __name__ == "__main__":
    main()
