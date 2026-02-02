"""
Convert GLB mesh to point cloud by sampling surface points.

This script samples points from mesh surfaces and saves them as:
- points.npy: (N, 3) array of XYZ coordinates
- points.ply: PLY file for visualization

Usage:
    python mesh_to_pointcloud.py --input objects/iron/mesh.glb --num_points 100000
"""

import argparse
import numpy as np
from pathlib import Path

import trimesh


def sample_mesh_to_pointcloud(
    mesh_path: str,
    num_points: int = 100000,
    include_colors: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Sample points uniformly from mesh surface.

    Args:
        mesh_path: Path to GLB/OBJ/PLY mesh file.
        num_points: Number of points to sample.
        include_colors: Whether to sample vertex colors.

    Returns:
        Tuple of (points, colors) where:
        - points: (N, 3) float array of XYZ coordinates
        - colors: (N, 3) uint8 array of RGB colors, or None
    """
    # Load mesh (handles GLB, OBJ, PLY, etc.)
    scene = trimesh.load(mesh_path, force='scene')

    # Combine all meshes in the scene
    if isinstance(scene, trimesh.Scene):
        # Get all meshes with their transforms applied
        meshes = []
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph[node_name]
            geometry = scene.geometry[geometry_name]
            if isinstance(geometry, trimesh.Trimesh):
                # Apply transform to get world coordinates
                mesh_copy = geometry.copy()
                mesh_copy.apply_transform(transform)
                meshes.append(mesh_copy)

        if not meshes:
            raise ValueError(f"No meshes found in {mesh_path}")

        # Concatenate all meshes
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene

    print(f"Combined mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")

    # Sample points uniformly from surface
    points, face_indices = combined.sample(num_points, return_index=True)

    # Get colors if available
    colors = None
    if include_colors and combined.visual is not None:
        try:
            # Try to get colors from visual
            if hasattr(combined.visual, 'face_colors'):
                face_colors = combined.visual.face_colors
                colors = face_colors[face_indices][:, :3]  # RGB only
            elif hasattr(combined.visual, 'vertex_colors'):
                # Interpolate vertex colors for sampled points
                vertex_colors = combined.visual.vertex_colors
                faces = combined.faces[face_indices]
                # Simple: use first vertex color of each face
                colors = vertex_colors[faces[:, 0]][:, :3]
        except Exception as e:
            print(f"Could not extract colors: {e}")
            colors = None

    return points.astype(np.float32), colors


def save_pointcloud_ply(
    filepath: str,
    points: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """Save point cloud as PLY file."""
    n_points = len(points)

    with open(filepath, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(n_points):
            x, y, z = points[i]
            if colors is not None:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_pointcloud_npy(filepath: str, points: np.ndarray) -> None:
    """Save point cloud as numpy array."""
    np.save(filepath, points)


def main():
    parser = argparse.ArgumentParser(
        description="Convert mesh to point cloud by surface sampling."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input mesh file (GLB, OBJ, PLY, etc.).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input mesh directory).",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=100000,
        help="Number of points to sample (default: 100000).",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Object name is the parent directory name (e.g., objects/iron/mesh.glb -> iron)
    object_dir = input_path.parent
    # mesh_name = object_dir.name

    # Output to same directory as mesh by default
    if args.output_dir is None:
        output_dir = object_dir
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Sampling {args.num_points} points...")

    # Sample point cloud
    points, colors = sample_mesh_to_pointcloud(
        str(input_path),
        num_points=args.num_points,
        include_colors=True,
    )

    print(f"Sampled {len(points)} points")
    print(f"  Bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
    print(f"  Center: {points.mean(axis=0)}")

    # Save as NPY (for projection script)
    npy_path = output_dir / "points.npy"
    save_pointcloud_npy(str(npy_path), points)
    print(f"Saved: {npy_path}")

    # Save as PLY (for visualization)
    ply_path = output_dir / "points.ply"
    save_pointcloud_ply(str(ply_path), points, colors)
    print(f"Saved: {ply_path}")

    # Also save colors if available
    if colors is not None:
        colors_path = output_dir / "points_colors.npy"
        np.save(str(colors_path), colors)
        print(f"Saved: {colors_path}")

    print("\nDone! Point cloud ready for projection.")


if __name__ == "__main__":
    main()
