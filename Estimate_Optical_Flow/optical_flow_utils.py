"""Shared utilities for optical flow estimation scripts."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np


def start_ffmpeg_writer(out_path: Path, fps: float, size_hw: Tuple[int, int]) -> subprocess.Popen:
    """Start system ffmpeg to write H.264 MP4 from raw BGR frames."""
    h, w = size_hw
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def close_ffmpeg(writer: Optional[subprocess.Popen]) -> None:
    """Close ffmpeg writer and raise on errors."""
    if writer is None:
        return
    if writer.stdin is not None:
        writer.stdin.close()
    stderr = writer.stderr.read() if writer.stderr is not None else b""
    ret = writer.wait()
    if ret != 0:
        msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed with code {ret}. stderr:\n{msg}")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve path against base_dir when relative."""
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def find_single_mp4(video_dir: Path) -> Path:
    """Find exactly one MP4 in video_dir."""
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 found in video dir: {video_dir}")
    if len(mp4_files) > 1:
        names = [p.name for p in mp4_files]
        raise RuntimeError(f"Expected exactly one .mp4 in {video_dir}, found: {names}")
    return mp4_files[0]


def find_single_summary(object_mesh_dir: Path) -> Path:
    """Find exactly one segmentation summary file in object_mesh_dir."""
    files = sorted(object_mesh_dir.glob("*_segmentation_summary.json"))
    if not files:
        raise FileNotFoundError(
            "No segmentation summary JSON found in object mesh dir: "
            f"{object_mesh_dir}"
        )
    if len(files) > 1:
        names = [p.name for p in files]
        raise RuntimeError(
            "Expected exactly one segmentation summary JSON in "
            f"{object_mesh_dir}, found: {names}"
        )
    return files[0]


def make_track_colors(points_xy: np.ndarray) -> np.ndarray:
    """Create deterministic BGR colors from point XY coordinates."""
    if points_xy.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    xy = points_xy.astype(np.float32)
    pmin = xy.min(axis=0)
    pmax = xy.max(axis=0)
    denom = np.maximum(pmax - pmin, 1e-8)
    xy_norm = (xy - pmin) / denom

    x = xy_norm[:, 0]
    y = xy_norm[:, 1]

    # B/G from spatial coordinates, R from their average for smoother gradients.
    bgr = np.stack([255.0 * x, 255.0 * y, 255.0 * (0.5 * (x + y))], axis=1)
    return bgr.clip(0.0, 255.0).astype(np.uint8)


def sample_visualization_indices(
    num_points: int,
    percent: float,
    seed: int,
) -> np.ndarray:
    """Uniformly sample point indices for visualization only."""
    if num_points <= 0:
        return np.zeros((0,), dtype=np.int32)

    if percent <= 0 or percent > 100:
        raise ValueError(f"--vis_point_percent must be in (0, 100], got {percent}")

    keep = int(np.ceil(num_points * (percent / 100.0)))
    keep = max(1, min(num_points, keep))

    rng = np.random.default_rng(seed)
    idx = rng.choice(num_points, size=keep, replace=False)
    return np.sort(idx.astype(np.int32))


def render_trails_video(
    frames_bgr: Sequence[np.ndarray],
    tracks: np.ndarray,
    visibility: np.ndarray,
    out_path: Path,
    fps: float,
    trail_length: int,
    point_indices: Optional[np.ndarray] = None,
) -> None:
    """Render colored points and short trails from 2D tracks."""
    if len(frames_bgr) == 0:
        raise RuntimeError("No frames available for trail rendering")

    if point_indices is not None:
        tracks = tracks[:, point_indices]
        visibility = visibility[:, point_indices]

    h, w = frames_bgr[0].shape[:2]
    num_frames, num_points = tracks.shape[:2]
    colors = make_track_colors(tracks[0] if num_frames > 0 else np.zeros((0, 2), dtype=np.float32))
    writer = start_ffmpeg_writer(out_path, fps, (h, w))
    trail_len = max(1, int(trail_length))

    def clamp_xy(xy: np.ndarray) -> Tuple[int, int]:
        x = int(np.clip(np.round(xy[0]), 0, w - 1))
        y = int(np.clip(np.round(xy[1]), 0, h - 1))
        return x, y

    for t in range(min(num_frames, len(frames_bgr))):
        canvas = frames_bgr[t].copy()
        t0 = max(0, t - trail_len + 1)

        for p_idx in range(num_points):
            color = tuple(int(v) for v in colors[p_idx])
            prev = None
            for k in range(t0, t + 1):
                if not bool(visibility[k, p_idx]):
                    prev = None
                    continue
                curr = clamp_xy(tracks[k, p_idx])
                if prev is not None:
                    alpha = (k - t0 + 1) / max(1, (t - t0 + 1))
                    line_color = tuple(int(alpha * ch) for ch in color)
                    cv2.line(canvas, prev, curr, line_color, 2)
                prev = curr

            if bool(visibility[t, p_idx]):
                curr = clamp_xy(tracks[t, p_idx])
                cv2.circle(canvas, curr, 1, color, -1)

        if writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        writer.stdin.write(np.ascontiguousarray(canvas).tobytes())

    close_ffmpeg(writer)


def erode_mask(mask: np.ndarray, erode_px: int) -> np.ndarray:
    """Erode binary mask by erode_px pixels."""
    if erode_px <= 0:
        return mask
    kernel_size = int(2 * erode_px + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def sample_points_from_mask(mask: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Randomly sample up to max_points (x, y) from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(xs))
    rng.shuffle(idx)
    idx = idx[: min(len(idx), int(max_points))]

    points_xy = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return points_xy


def compute_target_track_points(
    mask_area_px: int,
    density_per_1kpx: float,
) -> int:
    """Compute target tracking points from mask area and density."""
    if density_per_1kpx <= 0:
        raise ValueError(f"--track_point_density must be > 0, got {density_per_1kpx}")

    target = int(np.ceil((float(mask_area_px) * density_per_1kpx) / 1000.0))
    return max(1, target)


def normalize_slug(obj_info: dict[str, Any]) -> str:
    """Resolve stable object slug from summary object entry."""
    if "output_dir" in obj_info and obj_info["output_dir"]:
        return Path(str(obj_info["output_dir"])).name
    return str(obj_info.get("object", "object")).replace(" ", "_")


def resolve_mask_path(object_mesh_dir: Path, obj_info: dict[str, Any]) -> Path:
    """Resolve frame-0 mask path for one object."""
    slug = normalize_slug(obj_info)

    if "output_dir" in obj_info and obj_info["output_dir"]:
        obj_dir = Path(str(obj_info["output_dir"]))
    else:
        obj_dir = object_mesh_dir / slug

    mask_rel = str(obj_info.get("mask_file", "mask/frame_00.png"))
    mask_path = obj_dir / mask_rel
    if mask_path.exists():
        return mask_path

    return object_mesh_dir / slug / "mask" / "frame_00.png"
