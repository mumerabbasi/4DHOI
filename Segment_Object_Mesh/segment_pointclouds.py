"""
Segment a mesh into parts using multi-view part masks and face ID renders.

Pipeline:
1. Load mesh
2. For each view, map part mask pixels to triangles via face ID image
3. Aggregate triangle votes across views and assign final part labels
4. Sample points uniformly on mesh surface with part labels
5. (Optional) Smooth labels using spatial KNN voting
6. Export as labeled point cloud (combined and per-part)

Only triangles with actual votes receive labels - no propagation/filling.

Usage:
    python segment_pointclouds.py --mesh_dir objects/video_01/iron
    python segment_pointclouds.py --mesh_dir objects/video_01/iron --smooth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import OpenEXR
import Imath
from PIL import Image
from scipy.spatial import cKDTree


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Load mesh from GLB file."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(
        f"Loaded mesh: {len(mesh.vertices)} vertices, "
        f"{len(mesh.triangles)} triangles"
    )
    return mesh


def load_face_ids(exr_path: Path) -> np.ndarray:
    """Load face IDs from EXR file (stored in R channel as raw float)."""
    exr = OpenEXR.InputFile(str(exr_path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data = np.frombuffer(exr.channel("R", pt), dtype=np.float32)
    return np.rint(data.reshape(height, width)).astype(np.int64)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load binary mask from PNG file."""
    img = Image.open(mask_path)
    if img.mode == "RGBA":
        mask = np.array(img)[:, :, 3] > 128
    elif img.mode == "L":
        mask = np.array(img) > 128
    else:
        mask = np.array(img.convert("L")) > 128
    return mask


def discover_part_names(masks_dir: Path) -> set[str]:
    """Discover unique part names from mask filenames."""
    parts = set()
    for mask_file in masks_dir.glob("*.png"):
        name = mask_file.stem
        parts_of_name = name.rsplit("_", 1)
        if len(parts_of_name) == 2:
            parts.add(parts_of_name[1])
    return parts


def collect_triangle_votes(
    renders_dir: Path,
    masks_dir: Path,
    part_names: set[str],
    num_triangles: int,
) -> dict[str, np.ndarray]:
    """
    Collect votes for each part by mapping mask pixels to triangles.

    Returns:
        Dict mapping part name to vote array (one count per triangle).
    """
    votes = {part: np.zeros(num_triangles, dtype=np.int32) for part in part_names}

    face_id_dir = renders_dir / "face_id"
    face_id_files = sorted(face_id_dir.glob("*.exr"))

    for face_id_file in face_id_files:
        view_name = face_id_file.stem.replace("0001", "")
        face_ids = load_face_ids(face_id_file)

        for part_name in part_names:
            mask_path = masks_dir / f"{view_name}_{part_name}.png"
            if not mask_path.exists():
                continue

            mask = load_mask(mask_path)

            if mask.shape != face_ids.shape:
                mask = np.array(
                    Image.fromarray(mask).resize(
                        (face_ids.shape[1], face_ids.shape[0]),
                        Image.NEAREST
                    )
                )

            masked_face_ids = face_ids[mask]
            valid_ids = masked_face_ids[
                (masked_face_ids > 0) & (masked_face_ids < num_triangles)
            ]

            if len(valid_ids) > 0:
                counts = np.bincount(valid_ids, minlength=num_triangles)
                votes[part_name] += counts[:num_triangles]

        print(f"  Processed view: {view_name}")

    return votes


def assign_labels_by_votes(
    votes: dict[str, np.ndarray],
    min_votes: int = 1,
    min_margin: float = 0.6,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Assign part labels to triangles based on highest vote count.

    A triangle is only labeled if:
    1. It has at least min_votes total votes, AND
    2. The winning part has at least min_margin fraction of total votes

    Args:
        votes: Dict mapping part name to vote counts per triangle.
        min_votes: Minimum total votes required to label a triangle.
        min_margin: Minimum fraction of votes the winner must have (0-1).

    Returns:
        - labels: Array of part label indices (-1 for unlabeled)
        - label_map: Dict mapping part name to label index
    """
    part_names = sorted(votes.keys())
    label_map = {name: idx for idx, name in enumerate(part_names)}

    num_triangles = len(next(iter(votes.values())))
    labels = np.full(num_triangles, -1, dtype=np.int32)

    vote_matrix = np.stack([votes[name] for name in part_names], axis=0)
    total_votes = vote_matrix.sum(axis=0)
    max_votes = vote_matrix.max(axis=0)
    best_part = vote_matrix.argmax(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        margin = np.where(total_votes > 0, max_votes / total_votes, 0.0)

    valid_mask = (total_votes >= min_votes) & (margin >= min_margin)
    labels[valid_mask] = best_part[valid_mask]

    labeled = (labels >= 0).sum()
    ambiguous = ((total_votes >= min_votes) & (margin < min_margin)).sum()
    print(
        f"Labeled {labeled}/{num_triangles} triangles "
        f"({100 * labeled / num_triangles:.1f}%)"
    )
    print(f"Ambiguous (boundary) triangles skipped: {ambiguous}")
    for name, idx in label_map.items():
        count = (labels == idx).sum()
        print(f"  {name}: {count} triangles")

    return labels, label_map


def sample_points_on_mesh(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface.

    Returns:
        - points: (N, 3) array of sampled points
        - triangle_indices: (N,) array of source triangle index for each point
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    probs = areas / areas.sum()
    triangle_indices = np.random.choice(len(triangles), size=num_points, p=probs)

    r1 = np.random.rand(num_points)
    r2 = np.random.rand(num_points)

    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = r2 * sqrt_r1
    w = 1 - u - v

    sampled_tris = triangles[triangle_indices]
    p0 = vertices[sampled_tris[:, 0]]
    p1 = vertices[sampled_tris[:, 1]]
    p2 = vertices[sampled_tris[:, 2]]

    points = u[:, None] * p0 + v[:, None] * p1 + w[:, None] * p2

    return points, triangle_indices


def smooth_point_labels(
    points: np.ndarray,
    labels: np.ndarray,
    num_parts: int,
    k_neighbors: int = 50,
    min_agreement: float = 0.6,
) -> np.ndarray:
    """
    Smooth point labels using KNN voting to fix isolated mislabeled points.

    For each point, look at its k nearest neighbors and reassign the label
    based on majority vote. This fixes:
    - Unlabeled points surrounded by labeled points
    - Isolated mislabeled points inside a part region

    Args:
        points: (N, 3) array of point positions.
        labels: (N,) array of current labels (-1 for unlabeled).
        num_parts: Number of distinct part labels.
        k_neighbors: Number of neighbors to consider for voting.
        min_agreement: Minimum fraction of neighbors that must agree.

    Returns:
        Smoothed labels array.
    """
    print(f"Building KD-tree for {len(points):,} points...")
    tree = cKDTree(points)

    smoothed = labels.copy()
    changed_count = 0
    fixed_unlabeled = 0

    print(f"Smoothing with k={k_neighbors} neighbors...")

    # Query all neighbors at once for efficiency
    _, neighbor_indices = tree.query(points, k=k_neighbors + 1)
    # Exclude self (first neighbor)
    neighbor_indices = neighbor_indices[:, 1:]

    for i in range(len(points)):
        neighbor_labels = labels[neighbor_indices[i]]

        # Count votes for each label (excluding unlabeled)
        valid_neighbors = neighbor_labels[neighbor_labels >= 0]
        if len(valid_neighbors) == 0:
            continue

        votes = np.bincount(valid_neighbors, minlength=num_parts)
        best_label = votes.argmax()
        agreement = votes[best_label] / len(valid_neighbors)

        if agreement >= min_agreement:
            if labels[i] == -1:
                # Fix unlabeled point
                smoothed[i] = best_label
                fixed_unlabeled += 1
                changed_count += 1
            elif labels[i] != best_label:
                # Fix mislabeled point (isolated outlier)
                smoothed[i] = best_label
                changed_count += 1

    print(f"  Fixed {fixed_unlabeled} unlabeled points")
    print(f"  Corrected {changed_count - fixed_unlabeled} mislabeled points")
    print(f"  Total changes: {changed_count}")

    return smoothed


def create_labeled_pointcloud(
    points: np.ndarray,
    labels: np.ndarray,
    label_map: dict[str, int],
) -> o3d.geometry.PointCloud:
    """Create point cloud with colors based on part labels."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    num_parts = len(label_map)
    np.random.seed(42)
    part_colors = np.random.rand(num_parts, 3)
    part_colors = part_colors * 0.7 + 0.3

    colors = np.zeros((len(points), 3))
    for label_idx in range(num_parts):
        mask = labels == label_idx
        colors[mask] = part_colors[label_idx]

    colors[labels == -1] = [0.5, 0.5, 0.5]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def export_pointclouds(
    points: np.ndarray,
    labels: np.ndarray,
    label_map: dict[str, int],
    output_dir: Path,
    mesh_name: str,
) -> None:
    """Export combined and per-part point clouds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = create_labeled_pointcloud(points, labels, label_map)
    combined_path = output_dir / f"{mesh_name}_segmented.ply"
    o3d.io.write_point_cloud(str(combined_path), pcd)
    print(f"Saved combined point cloud: {combined_path}")

    for part_name, label_idx in label_map.items():
        mask = labels == label_idx
        if mask.sum() == 0:
            continue

        part_pcd = o3d.geometry.PointCloud()
        part_pcd.points = o3d.utility.Vector3dVector(points[mask])

        part_path = output_dir / f"{mesh_name}_{part_name}.ply"
        o3d.io.write_point_cloud(str(part_path), part_pcd)
        print(f"  Saved {part_name}: {mask.sum()} points")

    meta_path = output_dir / f"{mesh_name}_labels.json"
    with open(meta_path, "w") as f:
        json.dump({
            "label_map": label_map,
            "num_points": len(points),
            "parts": {
                name: int((labels == idx).sum())
                for name, idx in label_map.items()
            }
        }, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def save_triangle_labels(
    labels: np.ndarray,
    label_map: dict[str, int],
    output_path: Path,
) -> None:
    """Save per-triangle labels for later use or visualization."""
    data = {
        "label_map": label_map,
        "num_triangles": len(labels),
        "triangle_labels": labels.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved triangle labels: {output_path}")


def segment_mesh(
    mesh_dir: Path,
    num_points: int = 100_000,
    min_votes: int = 1,
    min_margin: float = 0.6,
    smooth: bool = False,
    k_neighbors: int = 50,
) -> None:
    """Main pipeline to segment a mesh into labeled point cloud."""
    mesh_path = mesh_dir / "mesh.glb"
    renders_dir = mesh_dir / "renders"
    masks_dir = mesh_dir / "masks"
    output_dir = mesh_dir / "pointclouds"

    mesh_name = mesh_dir.name
    num_steps = 6 if smooth else 5

    print(f"\n{'=' * 60}")
    print(f"Segmenting: {mesh_name}")
    print(f"{'=' * 60}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load mesh
    print(f"\n[1/{num_steps}] Loading mesh...")
    mesh = load_mesh(mesh_path)
    num_triangles = len(mesh.triangles)

    # Step 2: Collect votes
    print(f"\n[2/{num_steps}] Collecting triangle votes from masks...")
    part_names = discover_part_names(masks_dir)
    print(f"Found parts: {sorted(part_names)}")
    votes = collect_triangle_votes(
        renders_dir, masks_dir, part_names, num_triangles
    )

    # Step 3: Assign labels
    print(f"\n[3/{num_steps}] Assigning labels by vote count...")
    labels, label_map = assign_labels_by_votes(
        votes, min_votes=min_votes, min_margin=min_margin
    )
    unlabeled_count = (labels == -1).sum()
    print(f"Unlabeled triangles: {unlabeled_count}")

    # Step 4: Sample points
    print(f"\n[4/{num_steps}] Sampling {num_points:,} points on mesh surface...")
    points, triangle_indices = sample_points_on_mesh(mesh, num_points)
    point_labels = labels[triangle_indices]

    labeled_points = (point_labels >= 0).sum()
    unlabeled_points = (point_labels == -1).sum()
    print(
        f"Points with labels: {labeled_points} "
        f"({100 * labeled_points / num_points:.1f}%)"
    )
    print(f"Points without labels: {unlabeled_points}")

    # Step 5 (optional): Smooth labels
    if smooth:
        print(f"\n[5/{num_steps}] Smoothing point labels with KNN voting...")
        point_labels = smooth_point_labels(
            points, point_labels, len(label_map), k_neighbors=k_neighbors
        )

        labeled_after = (point_labels >= 0).sum()
        print(
            f"After smoothing: {labeled_after} labeled "
            f"({100 * labeled_after / num_points:.1f}%)"
        )

    # Save triangle labels
    print("\nSaving triangle labels...")
    tri_labels_path = output_dir / f"{mesh_name}_triangle_labels.json"
    save_triangle_labels(labels, label_map, tri_labels_path)

    # Final step: Export
    step_num = num_steps
    print(f"\n[{step_num}/{num_steps}] Exporting point clouds...")
    export_pointclouds(points, point_labels, label_map, output_dir, mesh_name)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Segment mesh into labeled point cloud using multi-view masks."
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default="./objects/video_01/ironing_board",
        help="Directory containing mesh.glb, renders/, and masks/",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=100_000,
        help="Number of points to sample (default: 100000)",
    )
    parser.add_argument(
        "--min_votes",
        type=int,
        default=1,
        help="Minimum votes required to assign a label (default: 1)",
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.6,
        help="Minimum vote fraction for winner (0-1, default: 0.6)",
    )
    parser.add_argument(
        "--not_smooth",
        action="store_true",
        help="Disable KNN-based label smoothing post-processing",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=100,
        help="Number of neighbors for smoothing (default: 100)",
    )

    args = parser.parse_args()
    segment_mesh(
        mesh_dir=Path(args.mesh_dir),
        num_points=args.num_points,
        min_votes=args.min_votes,
        min_margin=args.min_margin,
        smooth=not args.not_smooth,
        k_neighbors=args.k_neighbors,
    )


if __name__ == "__main__":
    main()
