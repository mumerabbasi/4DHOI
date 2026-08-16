"""Prepare GenZI with a visibility-aware, uniformly fused TSDF.

This is an isolated experimental successor to ``00_prepare_genzi.py``.  It
reuses that script's ScanNet++, SAM 3, viewpoint-selection, rendering, config,
and debug machinery, but replaces only the SDF fusion rule and writes to a
different output tree by default.

Fusion semantics for a voxel x and selected view i are based on the projective
signed distance d_i = rendered_depth_i - camera_depth_i(x):

* d_i > 0: observed free space; contribute min(d_i, tau)
* -tau <= d_i <= 0: narrow behind-surface band; contribute d_i
* d_i < -tau: occluded by the first surface; do not contribute

Every contributing selected view has weight one.  This avoids allowing an
occluded view to contradict a view that directly observes free space while
retaining the small negative band needed by a scalar TSDF.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = MODULE_DIR.parent.parent
BASE_SCRIPT = MODULE_DIR / "00_prepare_genzi.py"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output_visibility_tsdf"
DEFAULT_THIN_TRUNCATION_M = 0.075
TSDF_METHOD = "selected_view_uniform_visibility_tsdf_v1"


def load_base_module() -> ModuleType:
    """Load the existing preparation implementation without modifying it."""
    module_name = "_genzi_prepare_base"
    spec = importlib.util.spec_from_file_location(module_name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load preparation code from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def save_visibility_arrays(
    sdf_dir: Path,
    scene_id: str,
    vote_count: Any,
    free_vote_count: Any,
    negative_vote_count: Any,
    occluded_vote_count: Any,
) -> dict[str, Path]:
    """Save lossless per-voxel fusion diagnostics next to the GenZI SDF."""
    import numpy as np

    paths = {
        "vote_count": sdf_dir / f"{scene_id}_vote_count.npy",
        "free_vote_count": sdf_dir / f"{scene_id}_free_vote_count.npy",
        "negative_vote_count": sdf_dir / f"{scene_id}_negative_vote_count.npy",
        "occluded_vote_count": sdf_dir / f"{scene_id}_occluded_vote_count.npy",
    }
    np.save(paths["vote_count"], vote_count)
    np.save(paths["free_vote_count"], free_vote_count)
    np.save(paths["negative_vote_count"], negative_vote_count)
    np.save(paths["occluded_vote_count"], occluded_vote_count)
    return paths


def build_uniform_visibility_tsdf(
    scene_id: str,
    scene_vertices_world: Any,
    renderer: Any,
    scene_depths: Any,
    sdf_dir: Path,
    dim: int,
    padding_m: float,
    trunc_m: float,
) -> tuple[Path, dict[str, Any]]:
    """Fuse every selected depth view with uniform, visibility-aware weights."""
    import numpy as np

    sdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = sdf_dir / f"{scene_id}.json"
    sdf_path = sdf_dir / f"{scene_id}_sdf.npy"
    observed_path = sdf_dir / f"{scene_id}_observed.npy"

    vertices = np.asarray(scene_vertices_world, dtype=np.float32)
    bbox_min = vertices.min(axis=0) - float(padding_m)
    bbox_max = vertices.max(axis=0) + float(padding_m)
    dim = int(dim)
    trunc_m = float(trunc_m)
    if dim < 2:
        raise ValueError(f"SDF dimension must be at least 2; got {dim}")
    if not np.isfinite(trunc_m) or trunc_m <= 0.0:
        raise ValueError(f"TSDF truncation must be positive and finite; got {trunc_m}")

    xs = np.linspace(bbox_min[0], bbox_max[0], dim, dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], dim, dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], dim, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    depths = np.asarray(scene_depths, dtype=np.float32)
    if depths.ndim != 3:
        raise ValueError(f"Expected scene depths shaped [views, height, width], got {depths.shape}")
    num_views, height, width = depths.shape
    if num_views == 0:
        raise ValueError("Cannot build a TSDF without selected views")
    if num_views > np.iinfo(np.uint16).max:
        raise ValueError(f"Too many views for uint16 diagnostic counts: {num_views}")
    if len(renderer.modelviews) != num_views:
        raise ValueError(
            f"Renderer has {len(renderer.modelviews)} cameras but rendered {num_views} depth maps"
        )

    camera_ids = list(range(num_views))
    znear = float(renderer.camera_args.get("znear", 0.1))
    zfar = float(renderer.camera_args.get("zfar", 20.0))
    total_voxels = grid.shape[0]
    distance_sum = np.zeros((total_voxels,), dtype=np.float32)
    vote_count = np.zeros((total_voxels,), dtype=np.uint16)
    free_vote_count = np.zeros((total_voxels,), dtype=np.uint16)
    negative_vote_count = np.zeros((total_voxels,), dtype=np.uint16)
    occluded_vote_count = np.zeros((total_voxels,), dtype=np.uint16)

    # Keeping chunks moderate bounds the [num_views, chunk, 2] projection array.
    chunk_size = 65536
    for start in range(0, total_voxels, chunk_size):
        end = min(start + chunk_size, total_voxels)
        points = grid[start:end]
        screen = renderer.project(points, camera_ids=camera_ids)
        hom = np.concatenate(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)), axis=1
        )

        chunk_sum = np.zeros((points.shape[0],), dtype=np.float32)
        chunk_votes = np.zeros((points.shape[0],), dtype=np.uint16)
        chunk_free = np.zeros((points.shape[0],), dtype=np.uint16)
        chunk_negative = np.zeros((points.shape[0],), dtype=np.uint16)
        chunk_occluded = np.zeros((points.shape[0],), dtype=np.uint16)

        for view_idx, modelview in enumerate(renderer.modelviews):
            camera_points = hom @ modelview.T
            point_depth = -camera_points[:, 2]
            uv = screen[view_idx]
            ui = np.rint(uv[:, 0]).astype(np.int64)
            vi = np.rint(uv[:, 1]).astype(np.int64)
            valid = (
                (point_depth > znear)
                & (point_depth < zfar)
                & (ui >= 0)
                & (ui < width)
                & (vi >= 0)
                & (vi < height)
            )
            if not valid.any():
                continue

            sampled_depth = np.zeros((points.shape[0],), dtype=np.float32)
            sampled_depth[valid] = depths[view_idx, vi[valid], ui[valid]]
            valid &= np.isfinite(sampled_depth) & (sampled_depth > 0.0)
            if not valid.any():
                continue

            signed_distance = sampled_depth - point_depth
            occluded = valid & (signed_distance < -trunc_m)
            contributes = valid & ~occluded
            if occluded.any():
                chunk_occluded[occluded] += 1
            if not contributes.any():
                continue

            contribution = np.clip(signed_distance, -trunc_m, trunc_m)
            chunk_sum[contributes] += contribution[contributes].astype(np.float32)
            chunk_votes[contributes] += 1
            chunk_free[contributes & (signed_distance > 0.0)] += 1
            chunk_negative[contributes & (signed_distance < 0.0)] += 1

        distance_sum[start:end] = chunk_sum
        vote_count[start:end] = chunk_votes
        free_vote_count[start:end] = chunk_free
        negative_vote_count[start:end] = chunk_negative
        occluded_vote_count[start:end] = chunk_occluded

    observed = vote_count > 0
    sdf = np.full((total_voxels,), trunc_m, dtype=np.float32)
    sdf[observed] = distance_sum[observed] / vote_count[observed].astype(np.float32)

    volume_shape = (dim, dim, dim)
    sdf = sdf.reshape(volume_shape)
    observed_volume = observed.reshape(volume_shape)
    vote_count = vote_count.reshape(volume_shape)
    free_vote_count = free_vote_count.reshape(volume_shape)
    negative_vote_count = negative_vote_count.reshape(volume_shape)
    occluded_vote_count = occluded_vote_count.reshape(volume_shape)
    np.save(sdf_path, sdf)
    np.save(observed_path, observed_volume)
    diagnostic_paths = save_visibility_arrays(
        sdf_dir=sdf_dir,
        scene_id=scene_id,
        vote_count=vote_count,
        free_vote_count=free_vote_count,
        negative_vote_count=negative_vote_count,
        occluded_vote_count=occluded_vote_count,
    )

    voxel_size = (bbox_max - bbox_min) / float(dim - 1)
    conflict = (free_vote_count > 0) & (negative_vote_count > 0)
    metadata = {
        "dim": dim,
        "min": bbox_min.astype(float).tolist(),
        "max": bbox_max.astype(float).tolist(),
        "mesh_path": "",
        "method": TSDF_METHOD,
        "fusion": "uniform arithmetic mean over all non-occluded selected-view observations",
        "view_weight": 1.0,
        "trunc_m": trunc_m,
        "trunc_voxels_per_axis": (trunc_m / voxel_size).astype(float).tolist(),
        "num_views": int(num_views),
        "uses_all_selected_views": True,
        "depth_image_size": [int(height), int(width)],
        "unknown_value": trunc_m,
        "unknown_semantics": "no contributing view; stored positive for GenZI compatibility",
        "observation_rule": {
            "free": "d = rendered_depth - voxel_camera_depth > 0; positive vote clipped at +trunc_m",
            "negative_band": "-trunc_m <= d < 0; negative vote",
            "occluded": "d < -trunc_m; no TSDF vote",
            "invalid": "outside image/frustum or no rendered depth; no vote",
        },
        "observed_mask_path": str(observed_path.resolve()),
        "vote_count_path": str(diagnostic_paths["vote_count"].resolve()),
        "free_vote_count_path": str(diagnostic_paths["free_vote_count"].resolve()),
        "negative_vote_count_path": str(diagnostic_paths["negative_vote_count"].resolve()),
        "occluded_vote_count_path": str(diagnostic_paths["occluded_vote_count"].resolve()),
        "observed_voxels": int(np.count_nonzero(observed_volume)),
        "total_voxels": int(observed_volume.size),
        "observed_fraction": float(observed_volume.mean()),
        "negative_voxels": int(np.count_nonzero(observed_volume & (sdf < 0.0))),
        "positive_observed_voxels": int(np.count_nonzero(observed_volume & (sdf > 0.0))),
        "neutral_observed_voxels": int(np.count_nonzero(observed_volume & (sdf == 0.0))),
        "unknown_voxels": int(np.count_nonzero(~observed_volume)),
        "conflicting_observed_voxels": int(np.count_nonzero(conflict)),
        "mean_contributing_views_per_observed_voxel": float(vote_count[observed_volume].mean()),
        "max_contributing_views_per_voxel": int(vote_count.max()),
        "total_free_votes": int(free_vote_count.sum(dtype=np.uint64)),
        "total_negative_votes": int(negative_vote_count.sum(dtype=np.uint64)),
        "total_occluded_no_votes": int(occluded_vote_count.sum(dtype=np.uint64)),
        "sign_convention": (
            "positive is directly observed free space in front of rendered depth; "
            "negative is only the narrow band behind an observed surface"
        ),
    }
    # Use the base serializer so paths and numpy values follow the existing format.
    BASE_MODULE.save_json(meta_path, metadata)
    return meta_path, metadata


def option_present(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def arguments_with_experiment_defaults(arguments: list[str]) -> list[str]:
    result = list(arguments)
    if not option_present(result, "--output-base"):
        result.extend(["--output-base", str(DEFAULT_OUTPUT_BASE)])
    if not option_present(result, "--depth-sdf-trunc-m"):
        result.extend(["--depth-sdf-trunc-m", str(DEFAULT_THIN_TRUNCATION_M)])
    return result


BASE_MODULE = load_base_module()


def main(argv: list[str] | None = None) -> None:
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = arguments_with_experiment_defaults(raw_arguments)

    # Re-enter this experimental wrapper—not the old script—in the GenZI env.
    parsed = BASE_MODULE.parse_args(arguments)
    requested_python = Path(parsed.genzi_python).resolve()
    if not parsed._runtime_child and Path(sys.executable).resolve() != requested_python:
        if not requested_python.exists():
            raise FileNotFoundError(f"GenZI Python does not exist: {requested_python}")
        command = [requested_python, Path(__file__).resolve(), *raw_arguments, "--_runtime-child"]
        completed = subprocess.run([str(value) for value in command], cwd=str(WORKSPACE_ROOT))
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    BASE_MODULE.TSDF_METHOD = TSDF_METHOD
    BASE_MODULE.build_depth_tsdf_sdf = build_uniform_visibility_tsdf
    BASE_MODULE.main(arguments)
    print(
        "[*] Visibility-TSDF experiment output: "
        f"{Path(parsed.output_base).resolve()}",
        flush=True,
    )
    print(
        "[*] Run GenZI with: python 01_run_genzi.py --output-base "
        f"{Path(parsed.output_base).resolve()} --interaction-name {parsed.interaction_name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
