"""
Segment aligned object meshes into semantic parts using multi-view masks + face IDs.

Pipeline (per object mesh):
1. Load aligned .ply mesh
2. Map 2D part masks to mesh triangles via rendered face-ID EXRs
3. Aggregate multi-view triangle votes and assign final part labels
4. (Optional) Smooth noisy triangle labels with KNN voting on triangle centroids
5. Export segmented part meshes + combined colored segmented mesh
6. Save triangle labels and part->color mapping JSON files

Usage:
    python segment_meshes.py --video_name video_01
    python segment_meshes.py --video_name video_02 --not_smooth
"""

import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
import OpenEXR  # type: ignore
import Imath  # type: ignore


MASK_EROSION_PIXELS = 3
UNLABELED_COLOR_RGB = [128, 128, 128]


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_default_dirs(
    args: argparse.Namespace,
    script_dir: Path,
) -> tuple[Path, Path, Path]:
    if args.mesh_dir is None:
        mesh_dir = (
            script_dir.parent / "Align_Meshes" / "output" / args.video_name / "meshes"
        ).resolve()
    else:
        mesh_dir = _resolve_path(args.mesh_dir, script_dir)

    if args.masks_video_dir is None:
        masks_video_dir = (script_dir / "output" / args.video_name).resolve()
    else:
        masks_video_dir = _resolve_path(args.masks_video_dir, script_dir)

    if args.output_video_dir is None:
        output_video_dir = masks_video_dir
    else:
        output_video_dir = _resolve_path(args.output_video_dir, script_dir)

    return mesh_dir, masks_video_dir, output_video_dir


def load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Load mesh from PLY file."""
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh (empty): {mesh_path}")

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print(
        f"Loaded mesh: {mesh_path.name} | "
        f"{len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles"
    )
    return mesh


def load_face_ids(exr_path: Path) -> np.ndarray:
    """Load face IDs from EXR file (stored in R channel as raw float)."""
    if OpenEXR is None:
        raise ModuleNotFoundError(
            "OpenEXR is not installed. Install OpenEXR/Imath to read face-id EXR files."
        )

    exr = OpenEXR.InputFile(str(exr_path))
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    if Imath is not None:
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
    elif hasattr(OpenEXR, "PixelType") and hasattr(OpenEXR, "FLOAT"):
        # Fallback for OpenEXR bindings that expose PixelType directly.
        pt = OpenEXR.PixelType(OpenEXR.FLOAT)
    else:
        raise ModuleNotFoundError(
            "Imath is not installed and OpenEXR PixelType fallback is unavailable."
        )

    data = np.frombuffer(exr.channel("R", pt), dtype=np.float32)
    return np.rint(data.reshape(height, width)).astype(np.int64)


def load_mask(mask_path: Path) -> np.ndarray:
    """Load binary mask from PNG file as bool array."""
    img = Image.open(mask_path)
    if img.mode == "RGBA":
        mask = np.array(img)[:, :, 3] > 128
    elif img.mode == "L":
        mask = np.array(img) > 128
    else:
        mask = np.array(img.convert("L")) > 128
    return mask


def erode_mask(mask: np.ndarray, radius_px: int = MASK_EROSION_PIXELS) -> np.ndarray:
    """Erode a binary mask by an approximately circular radius in pixels."""
    if radius_px <= 0 or not mask.any():
        return mask

    yy, xx = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    structure = (xx * xx + yy * yy) <= (radius_px * radius_px)
    return binary_erosion(mask, structure=structure)


def _view_name_from_face_id_file(face_id_file: Path) -> str:
    stem = face_id_file.stem
    return stem[:-4] if stem.endswith("0001") else stem


def collect_face_id_files(face_id_dir: Path) -> dict[str, Path]:
    """Collect face-ID EXR files keyed by view name."""
    view_to_file: dict[str, Path] = {}
    for face_id_file in sorted(face_id_dir.glob("*.exr")):
        view_to_file[_view_name_from_face_id_file(face_id_file)] = face_id_file
    return view_to_file


def discover_part_names(masks_dir: Path, view_names: set[str]) -> set[str]:
    """Discover part names from <view_name>_<part_name>.png masks."""
    parts: set[str] = set()
    for mask_file in masks_dir.glob("*.png"):
        stem = mask_file.stem
        for view_name in view_names:
            prefix = f"{view_name}_"
            if stem.startswith(prefix) and len(stem) > len(prefix):
                parts.add(stem[len(prefix):])
                break
    return parts


def collect_triangle_votes(
    face_id_files: dict[str, Path],
    masks_dir: Path,
    part_names: set[str],
    num_triangles: int,
    mask_erosion_px: int = MASK_EROSION_PIXELS,
) -> dict[str, np.ndarray]:
    """
    Collect per-triangle votes for each part by mapping masks to face IDs.

    Returns:
        Dict mapping part name -> vote counts per triangle.
    """
    votes = {part: np.zeros(num_triangles, dtype=np.int32) for part in part_names}

    for view_name, face_id_file in sorted(face_id_files.items()):
        face_ids = load_face_ids(face_id_file)

        for part_name in part_names:
            mask_path = masks_dir / f"{view_name}_{part_name}.png"
            if not mask_path.exists():
                continue

            mask = load_mask(mask_path)
            if mask.shape != face_ids.shape:
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8)).resize(
                        (face_ids.shape[1], face_ids.shape[0]),
                        Image.NEAREST,
                    )
                ) > 0
            mask = erode_mask(mask, radius_px=mask_erosion_px)

            masked_face_ids = face_ids[mask]
            valid_ids = masked_face_ids[
                (masked_face_ids >= 0) & (masked_face_ids < num_triangles)
            ]

            if valid_ids.size > 0:
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
    Assign part labels to triangles based on vote counts.

    A triangle is labeled only if:
    1. total votes >= min_votes
    2. winning part fraction >= min_margin
    """
    if not votes:
        raise RuntimeError("No votes collected; cannot assign labels")

    part_names = sorted(votes.keys())
    label_map = {name: idx for idx, name in enumerate(part_names)}

    num_triangles = len(next(iter(votes.values())))
    labels = np.full(num_triangles, -1, dtype=np.int32)

    vote_matrix = np.stack([votes[name] for name in part_names], axis=0)
    total_votes = vote_matrix.sum(axis=0)
    max_votes = vote_matrix.max(axis=0)
    best_part = vote_matrix.argmax(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(total_votes > 0, max_votes / total_votes, 0.0)

    valid_mask = (total_votes >= min_votes) & (margin >= min_margin)
    labels[valid_mask] = best_part[valid_mask]

    labeled = int((labels >= 0).sum())
    ambiguous = int(((total_votes >= min_votes) & (margin < min_margin)).sum())
    print(
        f"Labeled {labeled}/{num_triangles} triangles "
        f"({100 * labeled / max(1, num_triangles):.1f}%)"
    )
    print(f"Ambiguous (boundary) triangles skipped: {ambiguous}")
    for name, idx in label_map.items():
        count = int((labels == idx).sum())
        print(f"  {name}: {count} triangles")

    return labels, label_map


def compute_triangle_centroids(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    return vertices[triangles].mean(axis=1)


def smooth_triangle_labels(
    centroids: np.ndarray,
    labels: np.ndarray,
    num_parts: int,
    k_neighbors: int = 100,
    min_agreement: float = 0.6,
) -> np.ndarray:
    """Smooth triangle labels with KNN majority voting."""
    num_items = len(labels)
    if num_items < 2:
        return labels

    k_eff = max(1, min(k_neighbors, num_items - 1))
    print(f"Building KD-tree for {num_items:,} triangle centroids...")
    tree = cKDTree(centroids)

    print(f"Smoothing with k={k_eff} neighbors...")
    _, neighbor_indices = tree.query(centroids, k=k_eff + 1)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    # Exclude self (first neighbor)
    neighbor_indices = neighbor_indices[:, 1:]

    smoothed = labels.copy()
    changed_count = 0
    fixed_unlabeled = 0

    for idx in range(num_items):
        neighbor_labels = labels[neighbor_indices[idx]]
        valid_neighbors = neighbor_labels[neighbor_labels >= 0]
        if valid_neighbors.size == 0:
            continue

        votes = np.bincount(valid_neighbors, minlength=num_parts)
        best_label = int(votes.argmax())
        agreement = float(votes[best_label] / len(valid_neighbors))

        if agreement >= min_agreement:
            if labels[idx] == -1:
                smoothed[idx] = best_label
                fixed_unlabeled += 1
                changed_count += 1
            elif labels[idx] != best_label:
                smoothed[idx] = best_label
                changed_count += 1

    print(f"  Fixed {fixed_unlabeled} unlabeled triangles")
    print(f"  Corrected {changed_count - fixed_unlabeled} mislabeled triangles")
    print(f"  Total changes: {changed_count}")
    return smoothed


def extract_submesh(
    mesh: o3d.geometry.TriangleMesh,
    triangle_mask: np.ndarray,
) -> o3d.geometry.TriangleMesh | None:
    """Extract a submesh containing only selected triangles."""
    triangle_indices = np.flatnonzero(triangle_mask)
    if triangle_indices.size == 0:
        return None

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    vertices = np.asarray(mesh.vertices)

    selected_triangles = triangles[triangle_indices]
    unique_vertices, inverse = np.unique(
        selected_triangles.reshape(-1), return_inverse=True
    )

    sub_vertices = vertices[unique_vertices]
    sub_triangles = inverse.reshape(-1, 3).astype(np.int32)

    submesh = o3d.geometry.TriangleMesh()
    submesh.vertices = o3d.utility.Vector3dVector(sub_vertices)
    submesh.triangles = o3d.utility.Vector3iVector(sub_triangles)
    submesh.compute_vertex_normals()
    submesh.compute_triangle_normals()
    return submesh


def generate_part_colors(label_map: dict[str, int]) -> dict[str, list[int]]:
    """Generate deterministic, visually separated RGB colors (0-255) for any part count."""
    colors: dict[str, list[int]] = {}

    # Golden-ratio hue spacing avoids clustering even for many parts.
    golden_ratio = 0.618033988749895
    for color_idx, (part_name, _) in enumerate(sorted(label_map.items(), key=lambda kv: kv[1])):
        hue = (color_idx * golden_ratio) % 1.0
        # Alternate sat/value bands to keep colors distinct while avoiding neon extremes.
        saturation = 0.65 + 0.15 * (color_idx % 3)  # 0.65, 0.80, 0.95
        value = 0.80 + 0.15 * ((color_idx // 3) % 2)  # 0.80, 0.95
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[part_name] = [int(round(255 * r)), int(round(255 * g)), int(round(255 * b))]

    return colors


def save_triangle_labels(
    labels: np.ndarray,
    label_map: dict[str, int],
    output_path: Path,
) -> None:
    data = {
        "label_map": label_map,
        "num_triangles": int(len(labels)),
        "triangle_labels": labels.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved triangle labels: {output_path}")


def save_part_colors(
    label_map: dict[str, int],
    part_colors: dict[str, list[int]],
    output_path: Path,
) -> None:
    payload = {
        "label_map": label_map,
        "part_colors_rgb_255": part_colors,
        "unlabeled_color_rgb_255": UNLABELED_COLOR_RGB,
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved part-color mapping: {output_path}")


def export_segmented_meshes(
    mesh: o3d.geometry.TriangleMesh,
    triangle_labels: np.ndarray,
    label_map: dict[str, int],
    output_dir: Path,
    mesh_name: str,
    part_colors: dict[str, list[int]],
) -> None:
    """Export per-part meshes and one combined colored mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = o3d.geometry.TriangleMesh()

    for part_name, label_idx in sorted(label_map.items(), key=lambda kv: kv[1]):
        mask = triangle_labels == label_idx
        tri_count = int(mask.sum())
        if tri_count == 0:
            continue

        part_mesh = extract_submesh(mesh, mask)
        if part_mesh is None:
            continue

        rgb = np.array(part_colors[part_name], dtype=np.float64) / 255.0
        part_mesh.paint_uniform_color(rgb)

        part_path = output_dir / f"{mesh_name}_{part_name}.ply"
        o3d.io.write_triangle_mesh(str(part_path), part_mesh, write_ascii=False)
        print(f"  Saved part mesh: {part_name} ({tri_count} triangles) -> {part_path}")

        combined += part_mesh

    unlabeled_mask = triangle_labels == -1
    unlabeled_count = int(unlabeled_mask.sum())
    if unlabeled_count > 0:
        unlabeled_mesh = extract_submesh(mesh, unlabeled_mask)
        if unlabeled_mesh is not None:
            unlabeled_mesh.paint_uniform_color(
                np.array(UNLABELED_COLOR_RGB, dtype=np.float64) / 255.0
            )
            unlabeled_path = output_dir / f"{mesh_name}_unlabeled.ply"
            o3d.io.write_triangle_mesh(str(unlabeled_path), unlabeled_mesh, write_ascii=False)
            print(f"  Saved unlabeled mesh ({unlabeled_count} triangles) -> {unlabeled_path}")
            combined += unlabeled_mesh

    if combined.is_empty():
        raise RuntimeError("Combined segmented mesh is empty; no triangles were exported")

    combined.compute_vertex_normals()
    combined.compute_triangle_normals()
    combined_path = output_dir / f"{mesh_name}_segmented_combined.ply"
    o3d.io.write_triangle_mesh(str(combined_path), combined, write_ascii=False)
    print(f"Saved combined segmented mesh: {combined_path}")


def segment_single_object(
    mesh_path: Path,
    object_input_dir: Path,
    object_output_dir: Path,
    min_votes: int,
    min_margin: float,
    mask_erosion_px: int,
    smooth: bool,
    k_neighbors: int,
    min_neighbor_agreement: float,
) -> None:
    """Segment one object mesh using corresponding renders/masks directory."""
    mesh_name = mesh_path.stem
    renders_dir = object_input_dir / "renders"
    masks_dir = object_input_dir / "masks"
    face_id_dir = renders_dir / "face_id"

    if not renders_dir.exists():
        raise FileNotFoundError(f"Missing renders directory: {renders_dir}")
    if not face_id_dir.exists():
        raise FileNotFoundError(f"Missing face_id directory: {face_id_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

    print(f"\n{'=' * 70}")
    print(f"Segmenting object: {mesh_name}")
    print(f"Mesh:   {mesh_path}")
    print(f"Masks:  {masks_dir}")
    print(f"Renders:{renders_dir}")
    print(f"Output: {object_output_dir}")
    print(f"Mask erosion: {mask_erosion_px}px")
    print(f"{'=' * 70}")

    mesh = load_mesh(mesh_path)
    num_triangles = len(mesh.triangles)

    face_id_files = collect_face_id_files(face_id_dir)
    if not face_id_files:
        raise RuntimeError(f"No face-ID EXR files found in: {face_id_dir}")

    part_names = discover_part_names(masks_dir, set(face_id_files.keys()))
    if not part_names:
        raise RuntimeError(
            f"No part masks found with pattern <view>_<part>.png in: {masks_dir}"
        )

    print(f"Found {len(part_names)} parts: {sorted(part_names)}")

    print("\nCollecting triangle votes from masks...")
    votes = collect_triangle_votes(
        face_id_files=face_id_files,
        masks_dir=masks_dir,
        part_names=part_names,
        num_triangles=num_triangles,
        mask_erosion_px=mask_erosion_px,
    )

    print("\nAssigning triangle labels by vote count...")
    triangle_labels, label_map = assign_labels_by_votes(
        votes=votes,
        min_votes=min_votes,
        min_margin=min_margin,
    )

    if smooth:
        print("\nSmoothing triangle labels...")
        centroids = compute_triangle_centroids(mesh)
        triangle_labels = smooth_triangle_labels(
            centroids=centroids,
            labels=triangle_labels,
            num_parts=len(label_map),
            k_neighbors=k_neighbors,
            min_agreement=min_neighbor_agreement,
        )

    unlabeled = int((triangle_labels == -1).sum())
    print(f"Unlabeled triangles after post-processing: {unlabeled}")

    object_output_dir.mkdir(parents=True, exist_ok=True)
    part_colors = generate_part_colors(label_map)

    print("\nExporting segmented meshes...")
    export_segmented_meshes(
        mesh=mesh,
        triangle_labels=triangle_labels,
        label_map=label_map,
        output_dir=object_output_dir,
        mesh_name=mesh_name,
        part_colors=part_colors,
    )

    save_triangle_labels(
        labels=triangle_labels,
        label_map=label_map,
        output_path=object_output_dir / f"{mesh_name}_triangle_labels.json",
    )
    save_part_colors(
        label_map=label_map,
        part_colors=part_colors,
        output_path=object_output_dir / f"{mesh_name}_part_colors.json",
    )


def segment_video_meshes(
    mesh_dir: Path,
    masks_video_dir: Path,
    output_video_dir: Path,
    output_subdir: str,
    min_votes: int,
    min_margin: float,
    mask_erosion_px: int,
    smooth: bool,
    k_neighbors: int,
    min_neighbor_agreement: float,
) -> None:
    """Segment all .ply meshes for a video."""
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    if not masks_video_dir.exists():
        raise FileNotFoundError(f"Masks video directory not found: {masks_video_dir}")

    mesh_files = sorted(mesh_dir.glob("*.ply"))
    if not mesh_files:
        raise FileNotFoundError(f"No .ply meshes found in: {mesh_dir}")

    print(f"\nMesh directory:   {mesh_dir}")
    print(f"Masks directory:  {masks_video_dir}")
    print(f"Output directory: {output_video_dir}")
    print(f"Found {len(mesh_files)} mesh files")

    processed: list[str] = []
    skipped: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []

    for mesh_path in mesh_files:
        object_slug = mesh_path.stem
        object_input_dir = masks_video_dir / object_slug

        if not object_input_dir.exists():
            reason = f"No matching object directory in masks video dir: {object_input_dir}"
            print(f"\nSkipping {object_slug}: {reason}")
            skipped.append({"object": object_slug, "reason": reason})
            continue

        object_output_dir = output_video_dir / object_slug / output_subdir

        try:
            segment_single_object(
                mesh_path=mesh_path,
                object_input_dir=object_input_dir,
                object_output_dir=object_output_dir,
                min_votes=min_votes,
                min_margin=min_margin,
                mask_erosion_px=mask_erosion_px,
                smooth=smooth,
                k_neighbors=k_neighbors,
                min_neighbor_agreement=min_neighbor_agreement,
            )
            processed.append(object_slug)
        except Exception as exc:
            print(f"\nFailed {object_slug}: {exc}")
            failed.append({"object": object_slug, "error": str(exc)})

    print(f"\n{'=' * 70}")
    print("Segmentation complete")
    print(f"Processed: {len(processed)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Failed:    {len(failed)}")
    print(f"{'=' * 70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment aligned .ply meshes into part meshes using multi-view masks."
        )
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to resolve default mesh/mask directories.",
    )
    parser.add_argument(
        "--mesh_dir",
        type=str,
        default=None,
        help=(
            "Directory of .ply meshes to segment "
            "(default: ../Align_Meshes/output/<video_name>/meshes)."
        ),
    )
    parser.add_argument(
        "--masks_video_dir",
        type=str,
        default=None,
        help=(
            "Directory with per-object masks/renders for the video "
            "(default: ./output/<video_name>)."
        ),
    )
    parser.add_argument(
        "--output_video_dir",
        type=str,
        default=None,
        help=(
            "Directory to write segmented mesh outputs "
            "(default: same as --masks_video_dir)."
        ),
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="segmented_meshes",
        help="Per-object output subdirectory name (default: segmented_meshes).",
    )
    parser.add_argument(
        "--min_votes",
        type=int,
        default=1,
        help="Minimum total votes required to label a triangle (default: 1).",
    )
    parser.add_argument(
        "--min_margin",
        type=float,
        default=0.6,
        help="Minimum winning vote fraction in [0,1] (default: 0.6).",
    )
    parser.add_argument(
        "--mask_erosion_px",
        type=int,
        default=MASK_EROSION_PIXELS,
        help=(
            "Erode binary masks by this many pixels before vote aggregation "
            f"(default: {MASK_EROSION_PIXELS}; use 0 to disable)."
        ),
    )
    parser.add_argument(
        "--not_smooth",
        action="store_true",
        help="Disable triangle-label smoothing post-processing.",
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=200,
        help="K neighbors for smoothing (default: 200).",
    )
    parser.add_argument(
        "--min_neighbor_agreement",
        type=float,
        default=0.9,
        help="Neighbor vote agreement threshold for smoothing in [0,1] (default: 0.9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    mesh_dir, masks_video_dir, output_video_dir = resolve_default_dirs(args, script_dir)

    segment_video_meshes(
        mesh_dir=mesh_dir,
        masks_video_dir=masks_video_dir,
        output_video_dir=output_video_dir,
        output_subdir=args.output_subdir,
        min_votes=args.min_votes,
        min_margin=args.min_margin,
        mask_erosion_px=args.mask_erosion_px,
        smooth=not args.not_smooth,
        k_neighbors=args.k_neighbors,
        min_neighbor_agreement=args.min_neighbor_agreement,
    )


if __name__ == "__main__":
    main()
