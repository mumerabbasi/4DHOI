"""Generate SDF volumes and MeshLab-friendly SDF visualizations for object meshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from pysdf import SDF as PySDF
from skimage import measure


def find_mesh_paths(mesh_root: Path, interaction_name: str | None) -> list[Path]:
    """Find module-04 mesh files."""
    if interaction_name:
        search_root = mesh_root / interaction_name / "meshes"
        return sorted(search_root.glob("*.ply"))
    return sorted(mesh_root.glob("*/meshes/*.ply"))


def make_sdf_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    padding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build SDF grid with the same sign convention as module 11: negative inside."""
    sdf_func = PySDF(vertices.astype(np.float32), faces.astype(np.uint32))
    bbox_min = vertices.min(axis=0).astype(np.float32) - float(padding)
    bbox_max = vertices.max(axis=0).astype(np.float32) + float(padding)
    axes = [
        np.linspace(bbox_min[i], bbox_max[i], resolution, dtype=np.float32)
        for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    query_points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    sdf_values = -sdf_func(query_points).astype(np.float32)
    return sdf_values.reshape(resolution, resolution, resolution), bbox_min, bbox_max


def export_zero_level_mesh(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    output_path: Path,
) -> bool:
    """Export the SDF zero level set as a PLY mesh."""
    if not (float(sdf_grid.min()) <= 0.0 <= float(sdf_grid.max())):
        return False

    spacing = (bbox_max - bbox_min) / (np.asarray(sdf_grid.shape, dtype=np.float32) - 1.0)
    verts, faces, normals, _ = measure.marching_cubes(
        sdf_grid,
        level=0.0,
        spacing=tuple(float(x) for x in spacing),
    )
    verts = verts.astype(np.float32) + bbox_min.reshape(1, 3)
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces.astype(np.int64),
        vertex_normals=normals.astype(np.float32),
        process=False,
    )
    mesh.visual.vertex_colors = np.tile(
        np.array([[230, 240, 255, 255]], dtype=np.uint8),
        (len(mesh.vertices), 1),
    )
    mesh.export(str(output_path))
    return True


def sdf_colors(sdf_values: np.ndarray, max_abs: float) -> np.ndarray:
    """Map signed distance values to blue/white/red vertex colors."""
    colors = np.full((sdf_values.shape[0], 4), 255, dtype=np.uint8)
    denom = max(float(max_abs), 1e-8)

    negative = sdf_values < 0.0
    neg_t = np.clip(-sdf_values[negative] / denom, 0.0, 1.0)
    colors[negative, 0] = np.rint(255.0 * (1.0 - neg_t)).astype(np.uint8)
    colors[negative, 1] = np.rint(255.0 * (1.0 - neg_t)).astype(np.uint8)
    colors[negative, 2] = 255

    positive = ~negative
    pos_t = np.clip(sdf_values[positive] / denom, 0.0, 1.0)
    colors[positive, 0] = 255
    colors[positive, 1] = np.rint(255.0 * (1.0 - pos_t)).astype(np.uint8)
    colors[positive, 2] = np.rint(255.0 * (1.0 - pos_t)).astype(np.uint8)
    return colors


def export_sdf_point_cloud(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    output_path: Path,
    band: float | None,
    max_points: int,
    seed: int,
) -> int:
    """Export a colored near-surface SDF point cloud as PLY."""
    resolution = sdf_grid.shape[0]
    axes = [
        np.linspace(bbox_min[i], bbox_max[i], resolution, dtype=np.float32)
        for i in range(3)
    ]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    sdf_values = sdf_grid.ravel()

    voxel = float(np.linalg.norm((bbox_max - bbox_min) / max(resolution - 1, 1)))
    band_width = float(band) if band is not None else 2.0 * voxel
    keep = np.flatnonzero(np.abs(sdf_values) <= band_width)
    if keep.size == 0:
        keep = np.arange(sdf_values.shape[0])

    if keep.size > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(keep, size=max_points, replace=False)

    kept_sdf = sdf_values[keep]
    colors = sdf_colors(kept_sdf, max_abs=band_width)
    cloud = trimesh.points.PointCloud(points[keep], colors=colors)
    cloud.export(str(output_path))
    return int(keep.size)


def export_inside_point_cloud(
    sdf_grid: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    output_path: Path,
    max_points: int,
    seed: int,
) -> int:
    """Export grid points classified as inside the mesh as a blue PLY point cloud."""
    # resolution = sdf_grid.shape[0]
    inside = np.flatnonzero(sdf_grid.ravel() < 0.0)
    if inside.size == 0:
        cloud = trimesh.points.PointCloud(np.zeros((0, 3), dtype=np.float32))
        cloud.export(str(output_path))
        return 0

    if inside.size > max_points:
        rng = np.random.default_rng(seed)
        inside = rng.choice(inside, size=max_points, replace=False)

    ix, iy, iz = np.unravel_index(inside, sdf_grid.shape)
    spacing = (bbox_max - bbox_min) / (np.asarray(sdf_grid.shape, dtype=np.float32) - 1.0)
    points = bbox_min.reshape(1, 3) + np.stack([ix, iy, iz], axis=1).astype(np.float32) * spacing
    colors = np.tile(np.array([[40, 110, 255, 255]], dtype=np.uint8), (points.shape[0], 1))
    cloud = trimesh.points.PointCloud(points.astype(np.float32), colors=colors)
    cloud.export(str(output_path))
    return int(points.shape[0])


def process_mesh(
    mesh_path: Path,
    mesh_root: Path,
    resolution: int,
    padding: float,
    band: float | None,
    max_points: int,
    seed: int,
) -> dict[str, object]:
    """Generate SDF artifacts for one mesh."""
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh as Trimesh: {mesh_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    sdf_grid, bbox_min, bbox_max = make_sdf_grid(vertices, faces, resolution, padding)

    interaction_dir = mesh_path.parents[1]
    output_dir = interaction_dir / "sdf_visualizations" / mesh_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    sdf_npy_path = output_dir / "sdf_volume.npy"
    meta_path = output_dir / "sdf_meta.json"
    iso_path = output_dir / "sdf_zero_level.ply"
    points_path = output_dir / "sdf_band_points.ply"
    inside_points_path = output_dir / "sdf_inside_points.ply"

    np.save(sdf_npy_path, sdf_grid.astype(np.float32))
    iso_written = export_zero_level_mesh(sdf_grid, bbox_min, bbox_max, iso_path)
    point_count = export_sdf_point_cloud(
        sdf_grid=sdf_grid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        output_path=points_path,
        band=band,
        max_points=max_points,
        seed=seed,
    )
    inside_point_count = export_inside_point_cloud(
        sdf_grid=sdf_grid,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        output_path=inside_points_path,
        max_points=max_points,
        seed=seed + 104729,
    )

    edge_counts = np.bincount(mesh.edges_unique_inverse)
    metadata = {
        "mesh_path": str(mesh_path),
        "sdf_volume_npy": str(sdf_npy_path),
        "sdf_zero_level_ply": str(iso_path) if iso_written else None,
        "sdf_band_points_ply": str(points_path),
        "sdf_inside_points_ply": str(inside_points_path),
        "resolution": int(resolution),
        "padding": float(padding),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "sign_convention": "negative_inside_same_as_11_Track_Human_Object_Mesh",
        "mesh_is_watertight": bool(mesh.is_watertight),
        "mesh_is_winding_consistent": bool(mesh.is_winding_consistent),
        "mesh_is_volume": bool(mesh.is_volume),
        "boundary_edges": int((edge_counts == 1).sum()),
        "nonmanifold_edges": int((edge_counts > 2).sum()),
        "sdf_min": float(sdf_grid.min()),
        "sdf_max": float(sdf_grid.max()),
        "band_point_count": int(point_count),
        "inside_point_count": int(inside_point_count),
        "note": (
            "For non-watertight meshes, the SDF magnitude is still useful, but "
            "inside/outside sign may be unreliable."
        ),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    rel = mesh_path.relative_to(mesh_root)
    print(
        f"{rel}: volume={mesh.is_volume} watertight={mesh.is_watertight} "
        f"sdf=[{metadata['sdf_min']:.5g}, {metadata['sdf_max']:.5g}] "
        f"band_points={point_count} inside_points={inside_point_count} out={output_dir}"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SDF volumes and MeshLab PLY visualizations for module-04 meshes."
    )
    parser.add_argument(
        "--mesh_root",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Root containing <interaction>/meshes/*.ply.",
    )
    parser.add_argument(
        "--interaction_name",
        type=str,
        default=None,
        help="Only process one interaction, e.g. interaction_01.",
    )
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--padding", type=float, default=0.05)
    parser.add_argument(
        "--band",
        type=float,
        default=None,
        help="Distance band for colored point clouds. Defaults to two voxel diagonals.",
    )
    parser.add_argument("--max_points", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_root = args.mesh_root.resolve()
    mesh_paths = find_mesh_paths(mesh_root, args.interaction_name)
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes found under {mesh_root}")

    print(f"Found {len(mesh_paths)} mesh(es).")
    for mesh_path in mesh_paths:
        process_mesh(
            mesh_path=mesh_path,
            mesh_root=mesh_root,
            resolution=args.resolution,
            padding=args.padding,
            band=args.band,
            max_points=args.max_points,
            seed=args.seed,
        )
    print("Done.")


if __name__ == "__main__":
    main()
