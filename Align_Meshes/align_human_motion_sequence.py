"""Apply the aligned human transform to an entire human mesh sequence.

This script reads the human transform from Align_Meshes outputs and applies it to
all .ply meshes in the human motion directory (the same directory where
align_meshes.py takes the first frame mesh).

Default behavior:
- Input human meshes: Estimate_Human_Motion/output/<video_name>/output_plys
- Transform source: Align_Meshes/output/<video_name>/meshes/transforms.json
- Output meshes: Align_Meshes/output/<video_name>/human_motion_aligned

Required transform key in transforms.json:
- source_to_aligned_cv_matrix_4x4
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


def _resolve_human_transform(transforms_json_path: Path) -> np.ndarray:
    """Return human source->aligned_cv transform matrix."""
    transforms_data = load_json(transforms_json_path)
    transforms = transforms_data.get("transforms", [])
    if not isinstance(transforms, list):
        raise ValueError(f"Invalid transforms payload in {transforms_json_path}")

    human_entry = None
    for entry in transforms:
        if entry.get("kind") == "human" or entry.get("slug") == "human":
            human_entry = entry
            break

    if human_entry is None:
        raise ValueError(f"No human transform entry found in {transforms_json_path}")

    if "source_to_aligned_cv_matrix_4x4" not in human_entry:
        raise ValueError(
            "Human transform missing required key: source_to_aligned_cv_matrix_4x4"
        )
    return _as_4x4(
        human_entry["source_to_aligned_cv_matrix_4x4"],
        "source_to_aligned_cv_matrix_4x4",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply aligned human transform to all human .ply meshes."
    )
    parser.add_argument("--video_name", type=str, default="video_01")
    parser.add_argument(
        "--align_video_dir",
        type=str,
        default=None,
        help="Align_Meshes output video dir. Default: ./output/<video_name>",
    )
    parser.add_argument(
        "--human_video_dir",
        type=str,
        default=None,
        help=(
            "Estimate_Human_Motion output video dir. "
            "Default: ../Estimate_Human_Motion/output/<video_name>"
        ),
    )
    parser.add_argument(
        "--input_human_dir",
        type=str,
        default=None,
        help="Optional explicit input human .ply directory. Default: <human_video_dir>/output_plys",
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
        help="Output directory for aligned human sequence. Default: <align_video_dir>/human_motion_aligned",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    align_video_dir = resolve_path(args.align_video_dir, script_dir) or (
        script_dir / "output" / args.video_name
    ).resolve()
    human_video_dir = resolve_path(args.human_video_dir, script_dir) or (
        script_dir.parent / "Estimate_Human_Motion" / "output" / args.video_name
    ).resolve()
    input_human_dir = resolve_path(args.input_human_dir, script_dir) or (
        human_video_dir / "output_plys"
    ).resolve()
    transforms_json_path = resolve_path(args.transforms_json, script_dir) or (
        align_video_dir / "meshes" / "transforms.json"
    ).resolve()
    output_dir = resolve_path(args.output_dir, script_dir) or (
        align_video_dir / "human_motion_aligned"
    ).resolve()

    if not input_human_dir.exists() or not input_human_dir.is_dir():
        raise NotADirectoryError(f"Input human mesh directory not found: {input_human_dir}")
    if not transforms_json_path.exists():
        raise FileNotFoundError(f"Transforms JSON not found: {transforms_json_path}")

    ply_paths = sorted(input_human_dir.glob("*.ply"))
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files found in {input_human_dir}")

    matrix = _resolve_human_transform(transforms_json_path=transforms_json_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input human meshes: {input_human_dir}")
    print(f"Transforms: {transforms_json_path}")
    print("Transform key: source_to_aligned_cv_matrix_4x4")
    print(f"Saving aligned sequence to: {output_dir}")

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

    print(f"Done. Wrote {len(ply_paths)} aligned meshes to {output_dir}")


if __name__ == "__main__":
    main()
