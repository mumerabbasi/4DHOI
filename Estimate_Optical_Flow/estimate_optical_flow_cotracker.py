#!/usr/bin/env python3
"""Estimate per-object CoTracker3 tracks from frame-0 masks.

Run with a CoTracker-capable environment.

Input:
- --video_dir: directory like */videos/video_xx containing exactly one .mp4
- --object_mesh_dir: directory like ../Generate_Object_Mesh/output/video_xx
  containing frame_00_segmentation_summary.json and per-object masks.

Output (under THIS script's directory by default):
./output_cotracker/video_xx/
  |_ <object_name>/
      |_ seed_points_frame0.npy
      |_ tracks.npy            # [T, N, 2]
      |_ visibility.npy        # [T, N] bool
      |_ trails.mp4            # colored points + trails
      |_ metadata.json
  |_ run_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


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


def make_track_colors(num_points: int) -> np.ndarray:
    """Create distinct BGR colors for each tracked point."""
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.uint8)

    hsv = np.zeros((num_points, 1, 3), dtype=np.uint8)
    for i in range(num_points):
        hue = int((179 * i) / max(1, num_points))
        hsv[i, 0] = (hue, 255, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, :]


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
    """Render CoTracker-style colored tracks with short trails."""
    if len(frames_bgr) == 0:
        raise RuntimeError("No frames available for trail rendering")

    if point_indices is not None:
        tracks = tracks[:, point_indices]
        visibility = visibility[:, point_indices]

    h, w = frames_bgr[0].shape[:2]
    num_frames, num_points = tracks.shape[:2]
    colors = make_track_colors(num_points)
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
                cv2.circle(canvas, curr, 2, color, -1)

        if writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        writer.stdin.write(np.ascontiguousarray(canvas).tobytes())

    close_ffmpeg(writer)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run CoTracker3 offline tracking per object for a video_xx directory."
    )
    parser.add_argument(
        "--video_dir",
        default="../Generate_Video/videos/video_01",
        type=str,
        help="Path to directory like */videos/video_xx containing one .mp4.",
    )
    parser.add_argument(
        "--object_mesh_dir",
        default=None,
        type=str,
        help=(
            "Path to Generate_Object_Mesh output for this video. "
            "Default: ../Generate_Object_Mesh/output/<video_xx>"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=(
            "Output directory for this video. "
            "Default: <script_dir>/output_cotracker/<video_xx>"
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="torch device string, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--track_point_density",
        default=200.0,
        type=float,
        help="Tracking seed density: points per 1000 mask pixels.",
    )
    parser.add_argument(
        "--mask_threshold",
        default=127,
        type=int,
        help="Threshold applied to mask grayscale image.",
    )
    parser.add_argument(
        "--mask_erode_px",
        default=2,
        type=int,
        help="Erode mask by this many pixels before point sampling.",
    )
    parser.add_argument(
        "--trail_length",
        default=10,
        type=int,
        help="Number of recent frames kept in track trails for visualization.",
    )
    parser.add_argument(
        "--vis_fps",
        default=6,
        type=float,
        help="Visualization FPS override. Default: source video FPS.",
    )
    parser.add_argument(
        "--vis_point_percent",
        default=2.5,
        type=float,
        help=(
            "Percentage of tracked points to visualize in trails.mp4. "
            "Tracking outputs still keep all points. Range: (0, 100]."
        ),
    )
    parser.add_argument(
        "--vis_seed",
        default=1234,
        type=int,
        help="Random seed used for visualization point subsampling.",
    )
    parser.add_argument(
        "--hub_repo",
        default="facebookresearch/co-tracker",
        type=str,
        help="torch.hub repository for CoTracker3.",
    )
    parser.add_argument(
        "--hub_model",
        default="cotracker3_offline",
        type=str,
        help="torch.hub model name.",
    )
    return parser.parse_args()


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


def load_video_tensor(
    video_mp4: Path,
    device: torch.device,
) -> tuple[torch.Tensor, list[np.ndarray], float]:
    """Load MP4 into tensor [1, T, 3, H, W] float32 and return BGR frames + FPS."""
    cap = cv2.VideoCapture(str(video_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 24.0

    frames_bgr = []
    frames_rgb = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames_bgr.append(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames_rgb.append(frame_rgb)

    cap.release()

    if len(frames_rgb) < 2:
        raise RuntimeError(f"Need at least 2 frames, found {len(frames_rgb)}")

    video_np = np.stack(frames_rgb, axis=0).astype(np.float32)  # [T, H, W, 3]
    video_t = torch.from_numpy(video_np).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, 3, H, W]
    return video_t.to(device=device), frames_bgr, float(fps)


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

    fallback = object_mesh_dir / slug / "mask" / "frame_00.png"
    return fallback


def run_cotracker_queries(
    cotracker: torch.nn.Module,
    video: torch.Tensor,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CoTracker with query points at frame 0.

    Returns:
        tracks: [T, N, 2] float32
        visibility: [T, N] bool
    """
    num_points = int(points_xy.shape[0])
    if num_points == 0:
        return np.zeros((0, 0, 2), dtype=np.float32), np.zeros((0, 0), dtype=bool)

    device = video.device
    queries = np.zeros((1, num_points, 3), dtype=np.float32)
    # CoTracker query format: [t, x, y]
    queries[0, :, 1:] = points_xy
    queries_t = torch.from_numpy(queries).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video, queries=queries_t)

    tracks = pred_tracks[0].detach().cpu().numpy().astype(np.float32)  # [T, N, 2]
    vis = pred_visibility[0].detach().cpu().numpy()

    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]

    visibility = vis > 0.5
    return tracks, visibility


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    video_dir = resolve_path(args.video_dir, script_dir)
    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Video dir not found: {video_dir}")

    video_name = video_dir.name
    video_mp4 = find_single_mp4(video_dir)

    if args.object_mesh_dir is None:
        object_mesh_dir = (script_dir.parent / "Generate_Object_Mesh" / "output" / video_name).resolve()
    else:
        object_mesh_dir = resolve_path(args.object_mesh_dir, script_dir)

    if not object_mesh_dir.exists() or not object_mesh_dir.is_dir():
        raise NotADirectoryError(f"Object mesh dir not found: {object_mesh_dir}")

    summary_path = find_single_summary(object_mesh_dir)

    if args.output_dir is None:
        output_video_dir = (script_dir / "output_cotracker" / video_name).resolve()
    else:
        output_video_dir = resolve_path(args.output_dir, script_dir)
    output_video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] video_dir: {video_dir}")
    print(f"[INFO] video_mp4: {video_mp4.name}")
    print(f"[INFO] object_mesh_dir: {object_mesh_dir}")
    print(f"[INFO] summary: {summary_path.name}")
    print(f"[INFO] output_dir: {output_video_dir}")
    print(f"[INFO] device: {device}")

    print("[INFO] Loading video tensor...")
    video, frames_bgr, video_fps = load_video_tensor(video_mp4, device=device)
    _, num_frames, _, height, width = video.shape
    print(f"[OK] video tensor shape: {tuple(video.shape)}")
    vis_fps = float(args.vis_fps) if args.vis_fps is not None else float(video_fps)

    print("[INFO] Loading CoTracker (torch.hub)...")
    cotracker = torch.hub.load(args.hub_repo, args.hub_model).to(device)
    cotracker.eval()
    print("[OK] CoTracker loaded")

    with summary_path.open("r") as f:
        summary = json.load(f)

    objects = [obj for obj in summary.get("objects", []) if obj.get("success", False)]
    if not objects:
        raise RuntimeError(f"No successful objects found in summary: {summary_path}")

    run_summary = {
        "video_dir": str(video_dir),
        "video_file": str(video_mp4),
        "summary_path": str(summary_path),
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "track_point_density": float(args.track_point_density),
        "vis_point_percent": float(args.vis_point_percent),
        "objects": [],
    }

    for obj_idx, obj_info in enumerate(objects):
        obj_name = str(obj_info.get("object", f"object_{obj_idx}"))
        slug = normalize_slug(obj_info)
        mask_path = resolve_mask_path(object_mesh_dir, obj_info)

        print(f"\n[OBJECT {obj_idx + 1}/{len(objects)}] {obj_name} ({slug})")

        if not mask_path.exists():
            print(f"[WARN] Mask not found, skipping: {mask_path}")
            continue

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"[WARN] Failed to read mask, skipping: {mask_path}")
            continue

        if mask_gray.shape[:2] != (height, width):
            mask_gray = cv2.resize(mask_gray, (width, height), interpolation=cv2.INTER_NEAREST)

        mask_bin = (mask_gray > int(args.mask_threshold)).astype(np.uint8)
        mask_bin = erode_mask(mask_bin, int(args.mask_erode_px))
        mask_area_px = int(mask_bin.sum())
        if mask_area_px <= 0:
            print("[WARN] Empty mask area after threshold/erosion, skipping")
            continue

        target_track_points = compute_target_track_points(
            mask_area_px=mask_area_px,
            density_per_1kpx=float(args.track_point_density),
        )

        seed_xy = sample_points_from_mask(
            mask=mask_bin,
            max_points=target_track_points,
            seed=42 + obj_idx,
        )
        if len(seed_xy) == 0:
            print("[WARN] No valid seed points after mask filtering, skipping")
            continue

        tracks, visibility = run_cotracker_queries(cotracker, video, seed_xy)

        obj_out = output_video_dir / slug
        obj_out.mkdir(parents=True, exist_ok=True)

        np.save(str(obj_out / "seed_points_frame0.npy"), seed_xy.astype(np.float32))
        np.save(str(obj_out / "tracks.npy"), tracks.astype(np.float32))
        np.save(str(obj_out / "visibility.npy"), visibility.astype(np.bool_))
        vis_indices = sample_visualization_indices(
            num_points=tracks.shape[1],
            percent=float(args.vis_point_percent),
            seed=int(args.vis_seed) + obj_idx,
        )
        render_trails_video(
            frames_bgr=frames_bgr,
            tracks=tracks,
            visibility=visibility,
            out_path=obj_out / "trails.mp4",
            fps=vis_fps,
            trail_length=int(args.trail_length),
            point_indices=vis_indices,
        )

        metadata = {
            "object": obj_name,
            "slug": slug,
            "mask_path": str(mask_path),
            "mask_area_px": mask_area_px,
            "track_point_density": float(args.track_point_density),
            "track_target_points": int(target_track_points),
            "seed_points": int(len(seed_xy)),
            "num_frames": int(tracks.shape[0]),
            "tracks_file": "tracks.npy",
            "visibility_file": "visibility.npy",
            "trails_video": "trails.mp4",
            "vis_point_percent": float(args.vis_point_percent),
            "vis_points_count": int(len(vis_indices)),
        }
        with (obj_out / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

        run_summary["objects"].append(metadata)
        print(f"[OK] Saved: {obj_out}")

    with (output_video_dir / "run_summary.json").open("w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
