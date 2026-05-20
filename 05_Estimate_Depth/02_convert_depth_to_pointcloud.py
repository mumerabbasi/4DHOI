from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve relative paths against base_dir."""
    path = Path(path_str)
    if not path.is_absolute():
        # Keep script-local defaults (./, ../) relative to this script,
        # but treat other relative inputs as workspace/cwd-relative.
        if path.parts and path.parts[0] in (".", ".."):
            path = base_dir / path
        else:
            path = Path.cwd() / path
    return path.resolve()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_3x3_intrinsics(raw: Any, source_name: str) -> np.ndarray:
    """Normalize nested intrinsics input into one (3,3) array."""
    k = np.asarray(raw, dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(f"{source_name} intrinsics must be (3,3), got: {k.shape}")
    if not np.isfinite(k).all():
        raise ValueError(f"{source_name} intrinsics contain non-finite values.")
    if abs(float(k[0, 0])) < 1e-8 or abs(float(k[1, 1])) < 1e-8:
        raise ValueError(f"{source_name} intrinsics have near-zero focal length.")
    return k


def load_camera_intrinsics(json_path: Path, source_name: str = "") -> np.ndarray:
    """Load intrinsics_pixels_3x3 from a camera_intrinsics.json file."""
    data = load_json(json_path)
    if "intrinsics_pixels_3x3" not in data:
        raise KeyError(
            f"Missing 'intrinsics_pixels_3x3' in {json_path}"
        )
    return ensure_3x3_intrinsics(data["intrinsics_pixels_3x3"], source_name or str(json_path))


def load_metric_depth(depth_path: Path) -> np.ndarray:
    """Load metric depth map as float32 HxW array."""
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth file not found: {depth_path}")
    depth = np.load(depth_path)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H,W), got: {depth.shape}")
    return depth


def find_metric_depth_path(
    depth_output_dir: Path,
    metric_depth_dirname: str,
    depth_file: str,
) -> Path:
    """Find metric depth file in new layout first, then legacy flat layout."""
    candidates = [
        depth_output_dir / metric_depth_dirname / depth_file,
        depth_output_dir / depth_file,  # Backward compatibility
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise FileNotFoundError(
            "Metric depth file not found. Tried: "
            + ", ".join(str(p) for p in candidates)
        )
    return found


def default_sam3d_output_dir(depth_output_dir: Path, script_dir: Path) -> Path:
    """Infer SAM3D output dir from depth_output_dir's video_xx folder."""
    video_dir_name = depth_output_dir.name
    return (script_dir.parent / "Generate_Object_Mesh" / "output" / video_dir_name).resolve()


def build_default_paths(video_name: str, script_dir: Path) -> tuple[Path, Path]:
    """Build default depth and SAM3D output directories for one video."""
    depth_output_dir = script_dir / "output" / video_name
    sam3d_output_dir = (script_dir.parent / "Generate_Object_Mesh" / "output" / video_name).resolve()
    return depth_output_dir, sam3d_output_dir


def load_rgb_image(image_path: Path, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Load RGB image for optional point coloring."""
    try:
        import cv2
    except ImportError:
        return None

    if not image_path.exists():
        return None
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        return None

    target_h, target_w = target_hw
    if image_bgr.shape[:2] != (target_h, target_w):
        image_bgr = cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    rgb_image: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Back-project depth into camera-space point cloud.

    Convention:
        X right, Y down, Z forward (OpenCV-like camera coordinates).
    """
    h, w = depth.shape
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])

    yy, xx = np.indices((h, w), dtype=np.float32)
    z = depth
    valid = np.isfinite(z) & (z > 0.0)

    x = ((xx - cx) / fx) * z
    y = ((yy - cy) / fy) * z

    points = np.stack((x[valid], y[valid], z[valid]), axis=-1).astype(np.float32)

    colors = None
    if rgb_image is not None:
        if rgb_image.shape[:2] != (h, w):
            raise ValueError(
                f"RGB image shape must match depth shape {(h, w)}, got {rgb_image.shape[:2]}"
            )
        colors = rgb_image.reshape(-1, 3)[valid.reshape(-1)].astype(np.uint8)

    return points, colors


def write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Write point cloud as ASCII PLY (optionally with RGB colors)."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got: {points.shape}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")

        if colors is not None:
            if colors.shape != (points.shape[0], 3):
                raise ValueError(
                    f"colors must be (N,3) matching points, got: {colors.shape}"
                )
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")

        f.write("end_header\n")

        if colors is None:
            np.savetxt(f, points, fmt=["%.6f", "%.6f", "%.6f"])
            return

        colors_u8 = np.clip(colors, 0, 255).astype(np.uint8)
        data = np.empty((points.shape[0], 6), dtype=np.float64)
        data[:, :3] = points
        data[:, 3:] = colors_u8
        np.savetxt(
            f,
            data,
            fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"],
        )


def pointcloud_stats(points: np.ndarray) -> dict[str, Any]:
    """Simple spatial stats for summary JSON."""
    if points.shape[0] == 0:
        return {
            "num_points": 0,
            "bounds_min_xyz": [0.0, 0.0, 0.0],
            "bounds_max_xyz": [0.0, 0.0, 0.0],
        }
    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    return {
        "num_points": int(points.shape[0]),
        "bounds_min_xyz": bounds_min.astype(float).tolist(),
        "bounds_max_xyz": bounds_max.astype(float).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert DA3 metric depth to point clouds using DA3 intrinsics and "
            "SAM3D intrinsics."
        )
    )
    parser.add_argument(
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--depth_output_dir",
        type=str,
        default=None,
        help=(
            "Depth output directory (Estimate_Depth/output/video_xx). "
            "Contains camera_intrinsics.json and depth subdirectories. "
            "Defaults to Estimate_Depth/output/<video_name>/."
        ),
    )
    parser.add_argument(
        "--metric_depth_dirname",
        type=str,
        default="metric_depth",
        help="Subdirectory under depth_output_dir containing metric depth artifacts.",
    )
    parser.add_argument(
        "--sam3d_output_dir",
        type=str,
        default=None,
        help=(
            "Directory containing camera_intrinsics.json from SAM3D object generation. "
            "Default: ../Generate_Object_Mesh/output/<video_xx inferred from depth_output_dir>."
        ),
    )
    parser.add_argument(
        "--depth_file",
        type=str,
        default="metric_depth.npy",
        help="Metric depth file name inside metric depth directory.",
    )
    parser.add_argument(
        "--da3_intrinsics_file",
        type=str,
        default="camera_intrinsics.json",
        help="DA3 camera intrinsics JSON file name inside depth_output_dir.",
    )
    parser.add_argument(
        "--sam3d_intrinsics_file",
        type=str,
        default="camera_intrinsics.json",
        help="SAM3D camera intrinsics JSON file name inside sam3d_output_dir.",
    )
    parser.add_argument(
        "--color_image_file",
        type=str,
        default="frame_00.png",
        help="Optional RGB image file in depth_output_dir for point colors.",
    )
    parser.add_argument(
        "--save_colors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store per-point RGB color from color_image_file when available.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="metric_depth_pointcloud",
        help="Prefix for output point cloud filenames.",
    )
    parser.add_argument(
        "--pointcloud_dirname",
        type=str,
        default="pointclouds",
        help="Subdirectory under depth_output_dir where point cloud outputs are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    default_depth_output_dir, default_sam3d_output_dir_path = build_default_paths(
        args.video_name, script_dir
    )
    depth_output_dir = (
        resolve_path(args.depth_output_dir, script_dir)
        if args.depth_output_dir
        else default_depth_output_dir
    )
    if args.sam3d_output_dir is None:
        sam3d_output_dir = default_sam3d_output_dir_path
    else:
        sam3d_output_dir = resolve_path(args.sam3d_output_dir, script_dir)

    if not depth_output_dir.is_dir():
        raise NotADirectoryError(f"Depth output directory not found: {depth_output_dir}")
    if not sam3d_output_dir.is_dir():
        raise NotADirectoryError(f"SAM3D output directory not found: {sam3d_output_dir}")

    depth_path = find_metric_depth_path(
        depth_output_dir=depth_output_dir,
        metric_depth_dirname=args.metric_depth_dirname,
        depth_file=args.depth_file,
    )
    pointcloud_output_dir = depth_output_dir / args.pointcloud_dirname
    pointcloud_output_dir.mkdir(parents=True, exist_ok=True)
    da3_intrinsics_path = depth_output_dir / args.da3_intrinsics_file
    sam3d_intrinsics_path = sam3d_output_dir / args.sam3d_intrinsics_file
    color_image_path = depth_output_dir / args.color_image_file

    depth = load_metric_depth(depth_path)
    da3_intrinsics = load_camera_intrinsics(da3_intrinsics_path, "DA3")
    sam3d_intrinsics = load_camera_intrinsics(sam3d_intrinsics_path, "SAM3D")

    rgb_image = None
    if args.save_colors:
        rgb_image = load_rgb_image(color_image_path, target_hw=depth.shape)
        if rgb_image is None:
            print(f"Color image unavailable, saving geometry-only point clouds: {color_image_path}")

    da3_points, da3_colors = depth_to_pointcloud(depth, da3_intrinsics, rgb_image)
    sam3d_points, sam3d_colors = depth_to_pointcloud(depth, sam3d_intrinsics, rgb_image)

    da3_ply_path = pointcloud_output_dir / f"{args.output_prefix}_da3_intrinsics.ply"
    sam3d_ply_path = pointcloud_output_dir / f"{args.output_prefix}_sam3d_intrinsics.ply"

    write_ply_ascii(da3_ply_path, da3_points, da3_colors)
    write_ply_ascii(sam3d_ply_path, sam3d_points, sam3d_colors)

    summary = {
        "pointcloud_output_dir": str(pointcloud_output_dir),
        "inputs": {
            "metric_depth_npy": str(depth_path),
            "da3_intrinsics_json": str(da3_intrinsics_path),
            "sam3d_intrinsics_json": str(sam3d_intrinsics_path),
            "color_image": str(color_image_path) if args.save_colors else None,
        },
        "depth_shape_hw": [int(depth.shape[0]), int(depth.shape[1])],
        "outputs": {
            "da3_intrinsics_pointcloud": {
                "ply": str(da3_ply_path),
                "has_color": bool(da3_colors is not None),
                **pointcloud_stats(da3_points),
            },
            "sam3d_intrinsics_pointcloud": {
                "ply": str(sam3d_ply_path),
                "has_color": bool(sam3d_colors is not None),
                **pointcloud_stats(sam3d_points),
            },
        },
    }

    summary_path = pointcloud_output_dir / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved point cloud (DA3 intrinsics): {da3_ply_path}")
    print(f"Saved point cloud (SAM3D intrinsics): {sam3d_ply_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
