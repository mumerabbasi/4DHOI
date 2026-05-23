from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve relative paths against base_dir."""
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def find_single_mp4(video_dir: Path) -> Path:
    """Find exactly one MP4 file in a directory."""
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if len(mp4_files) == 0:
        raise FileNotFoundError(f"No .mp4 found in interaction dir: {video_dir}")
    if len(mp4_files) > 1:
        raise RuntimeError(
            f"Expected exactly one .mp4 in {video_dir}, found {[p.name for p in mp4_files]}"
        )
    return mp4_files[0]


def extract_first_frame_bgr(video_path: Path) -> np.ndarray:
    """Extract the first frame from a video as BGR uint8."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from video: {video_path}")
    return frame_bgr


def save_rgb_png(path: Path, image_rgb: np.ndarray) -> None:
    """Save RGB uint8 image to PNG."""
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def depth_to_mask_u16(depth: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Convert depth map to normalized 16-bit PNG-friendly mask."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)

    mask = np.zeros(depth.shape, dtype=np.uint16)
    stats = {
        "p01": 0.0,
        "p99": 0.0,
    }

    if not np.any(valid):
        return mask, stats

    vals = depth[valid]
    p01 = float(np.percentile(vals, 1.0))
    p99 = float(np.percentile(vals, 99.0))

    if not np.isfinite(p01) or not np.isfinite(p99) or p99 <= p01:
        p01 = float(np.min(vals))
        p99 = float(np.max(vals))

    denom = max(p99 - p01, 1e-6)
    normalized = np.clip((depth - p01) / denom, 0.0, 1.0)
    mask[valid] = np.round(normalized[valid] * 65535.0).astype(np.uint16)

    stats["p01"] = p01
    stats["p99"] = p99
    return mask, stats


def save_depth_artifacts(
    out_dir: Path,
    prefix: str,
    depth: np.ndarray,
    visualize_depth_fn: Any,
    output_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Save depth map as raw array, mask PNG, and color visualization."""
    depth = np.asarray(depth, dtype=np.float32)
    model_output_hw = [int(depth.shape[0]), int(depth.shape[1])]

    if output_hw is not None:
        out_h, out_w = int(output_hw[0]), int(output_hw[1])
        if (depth.shape[0], depth.shape[1]) != (out_h, out_w):
            depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    npy_path = out_dir / f"{prefix}.npy"
    np.save(npy_path, depth)

    mask_u16, mask_stats = depth_to_mask_u16(depth)
    mask_path = out_dir / f"{prefix}_mask.png"
    if not cv2.imwrite(str(mask_path), mask_u16):
        raise RuntimeError(f"Failed to write depth mask: {mask_path}")

    vis_rgb = visualize_depth_fn(depth, cmap="Spectral")
    vis_path = out_dir / f"{prefix}_vis.png"
    save_rgb_png(vis_path, vis_rgb)

    return {
        "npy": str(npy_path),
        "mask_png": str(mask_path),
        "vis_png": str(vis_path),
        "mask_percentile_range": mask_stats,
        "model_output_hw": model_output_hw,
        "saved_output_hw": [int(depth.shape[0]), int(depth.shape[1])],
    }


def compute_focal_length_mm_from_intrinsics(
    intrinsics: np.ndarray,
    image_width_px: int,
    sensor_width_mm: float = 36.0,
) -> dict[str, Any]:
    """Compute focal length in mm from fx (pixels) and sensor width."""
    if image_width_px <= 0:
        raise ValueError(f"image_width_px must be > 0, got: {image_width_px}")
    if sensor_width_mm <= 0:
        raise ValueError(f"sensor_width_mm must be > 0, got: {sensor_width_mm}")

    k = np.asarray(intrinsics, dtype=np.float32)
    if k.ndim != 3 or k.shape[1:] != (3, 3):
        raise ValueError(f"Expected intrinsics shape (N,3,3), got: {k.shape}")

    fx_px = k[:, 0, 0].astype(np.float64)
    focal_mm = fx_px * (float(sensor_width_mm) / float(image_width_px))
    return {
        "sensor_width_mm_assumed": float(sensor_width_mm),
        "image_width_px_for_intrinsics": int(image_width_px),
        "fx_px_per_view": fx_px.tolist(),
        "focal_length_mm_per_view": focal_mm.tolist(),
        "focal_length_mm_mean": float(focal_mm.mean()),
    }


def parse_device(device: str) -> torch.device:
    """Validate and parse a torch device string."""
    device = device.strip()
    if not device:
        return torch.device("cpu")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested but torch.cuda.is_available() is False.")
        if device != "cuda":
            idx = int(device.split(":", maxsplit=1)[1])
            num_cuda = torch.cuda.device_count()
            if idx < 0 or idx >= num_cuda:
                raise ValueError(
                    f"Requested device {device}, but only {num_cuda} CUDA device(s) are visible."
                )

    return torch.device(device)


def build_default_paths(interaction_name: str, script_dir: Path) -> tuple[Path, Path]:
    """Build default input/output paths for a given interaction name."""
    project_dir = script_dir.parent
    video_dir = project_dir / "02_Generate_Video" / "output" / interaction_name
    output_root = script_dir / "output"
    return video_dir, output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract first frame and estimate depth + pose using Depth Anything 3."
    )
    parser.add_argument(
        "--interaction_name",
        type=str,
        default="interaction_01",
        help="Interaction name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help=(
            "Directory like */interaction_xx containing exactly one .mp4, "
            "or a direct path to an .mp4 file. "
            "Defaults to ../02_Generate_Video/output/<interaction_name>/."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output root directory. Final output is <output_root>/<interaction_xx>.",
    )
    parser.add_argument(
        "--da3_repo",
        type=str,
        default="../../Depth-Anything-3",
        help="Path to Depth Anything 3 repo.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--process_res",
        type=int,
        default=None,
        help=(
            "DA3 processing resolution. "
            "If omitted, uses the first-frame longest side (max(H, W))."
        ),
    )
    parser.add_argument(
        "--process_res_method",
        type=str,
        default="upper_bound_resize",
        choices=["upper_bound_resize", "lower_bound_resize"],
        help="DA3 preprocessing resize mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="torch device string for inference (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--use_ray_pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use DA3 ray-based pose estimation.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--estimate_relative_depth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Estimate and save relative depth from metric depth using DA3 scale factor. "
            "Disabled by default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    default_video_dir, default_output_root = build_default_paths(args.interaction_name, script_dir)

    video_input = (
        resolve_path(args.video_dir, script_dir) if args.video_dir else default_video_dir
    )
    if video_input.is_file():
        if video_input.suffix.lower() != ".mp4":
            raise ValueError(f"Expected an .mp4 file, got: {video_input}")
        video_dir = video_input.parent
        video_mp4 = video_input
    elif video_input.is_dir():
        video_dir = video_input
        video_mp4 = find_single_mp4(video_dir)
    else:
        raise FileNotFoundError(f"Video input does not exist: {video_input}")

    interaction_name = args.interaction_name

    output_root = (
        resolve_path(args.output_root, script_dir)
        if args.output_root
        else default_output_root
    )
    output_dir = output_root / interaction_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_bgr = extract_first_frame_bgr(video_mp4)
    frame_h, frame_w = frame_bgr.shape[:2]
    if args.process_res is not None and args.process_res <= 0:
        raise ValueError(f"--process_res must be > 0 when provided, got: {args.process_res}")
    process_res = args.process_res if args.process_res is not None else max(frame_h, frame_w)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_path = output_dir / "frame_00.png"
    save_rgb_png(frame_path, frame_rgb)

    da3_repo = resolve_path(args.da3_repo, script_dir)
    da3_src = da3_repo / "src"
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))

    try:
        from depth_anything_3.api import DepthAnything3
        from depth_anything_3.utils.visualize import visualize_depth
    except ImportError as exc:
        raise RuntimeError(
            "Could not import `depth_anything_3`. "
            "Install DA3 in your env first (e.g. `pip install -e <Depth-Anything-3>`)."
        ) from exc

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)

    device = parse_device(args.device)

    # Let HF Hub handle checkpoint caching automatically.
    model = DepthAnything3.from_pretrained(args.model_id)
    model = model.to(device=device)
    model.device = device
    model.eval()

    prediction = model.inference(
        image=[str(frame_path)],
        process_res=process_res,
        process_res_method=args.process_res_method,
        use_ray_pose=args.use_ray_pose,
    )

    if prediction.depth is None or prediction.depth.shape[0] != 1:
        raise RuntimeError("Unexpected DA3 output: expected one depth map for one input frame.")
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("Pose estimation outputs are missing extrinsics/intrinsics.")

    model_depth_w = int(prediction.depth.shape[2])
    focal_info = compute_focal_length_mm_from_intrinsics(
        intrinsics=prediction.intrinsics,
        image_width_px=model_depth_w,
        sensor_width_mm=36.0,
    )
    metric_depth = prediction.depth[0].astype(np.float32)

    scale_factor = prediction.scale_factor
    scale_factor_f = float(scale_factor) if scale_factor is not None else None
    metric_depth_dir = output_dir / "metric_depth"
    metric_depth_dir.mkdir(parents=True, exist_ok=True)

    metric_artifacts = save_depth_artifacts(
        out_dir=metric_depth_dir,
        prefix="metric_depth",
        depth=metric_depth,
        visualize_depth_fn=visualize_depth,
        output_hw=(frame_h, frame_w),
    )
    relative_artifacts: dict[str, Any] | None = None
    if args.estimate_relative_depth:
        relative_depth_dir = output_dir / "relative_depth"
        relative_depth_dir.mkdir(parents=True, exist_ok=True)
        if scale_factor_f is not None and np.isfinite(scale_factor_f) and abs(scale_factor_f) > 1e-8:
            relative_depth = metric_depth / scale_factor_f
        else:
            relative_depth = metric_depth.copy()
        relative_artifacts = save_depth_artifacts(
            out_dir=relative_depth_dir,
            prefix="relative_depth",
            depth=relative_depth,
            visualize_depth_fn=visualize_depth,
            output_hw=(frame_h, frame_w),
        )

    # Build pixel-space 3x3 intrinsics from the DA3 prediction.
    raw_intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    while raw_intrinsics.ndim > 2:
        raw_intrinsics = raw_intrinsics[0]
    intrinsics_pixels_3x3 = raw_intrinsics.tolist()

    sensor_width_mm = 36.0
    sensor_height_mm = 24.0
    lens_mm = float(focal_info["focal_length_mm_mean"])

    camera_intrinsics_path = output_dir / "camera_intrinsics.json"
    camera_intrinsics_json = {
        "source": "da3",
        "intrinsics_pixels_3x3": intrinsics_pixels_3x3,
        "blender_recommendation": {
            "sensor_fit": "HORIZONTAL",
            "lens_mm": lens_mm,
            "sensor_width_mm": sensor_width_mm,
            "sensor_height_mm": sensor_height_mm,
            "note": "Lens uses fx with full-frame horizontal fit.",
        },
    }
    with open(camera_intrinsics_path, "w", encoding="utf-8") as f:
        json.dump(camera_intrinsics_json, f, indent=2)

    run_summary = {
        "output_dir": str(output_dir),
        "inputs": {
            "video_dir": str(video_dir),
            "video_mp4": str(video_mp4),
            "model_id": args.model_id,
            "device": str(device),
            "process_res_used": int(process_res),
        },
        "outputs": {
            "frame_00": str(frame_path),
            "relative_depth": (
                None
                if relative_artifacts is None
                else {"dir": str(output_dir / "relative_depth"), **relative_artifacts}
            ),
            "metric_depth": {
                "dir": str(metric_depth_dir),
                **metric_artifacts,
            },
            "camera_intrinsics": {
                "json": str(camera_intrinsics_path),
            },
        },
    }
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"Saved depth outputs to: {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
