"""Estimate per-object CoTracker3 tracks from 03_Segment_Video frame-0 masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from optical_flow_utils import (
    compute_target_track_points,
    erode_mask,
    find_single_mp4,
    render_trails_video,
    resolve_path,
    sample_points_from_mask,
    sample_visualization_indices,
)


def build_default_paths(interaction_name: str, script_dir: Path) -> tuple[Path, Path, Path]:
    """Build default video, segmentation, and output directories for one video."""
    project_dir = script_dir.parent
    video_dir = project_dir / "02_Generate_Video" / "output" / interaction_name
    segment_video_dir = project_dir / "03_Segment_Video" / "output" / interaction_name
    output_dir = script_dir / "output_cotracker" / interaction_name
    return video_dir, segment_video_dir, output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run CoTracker3 offline tracking per object for an interaction_xx directory."
    )
    parser.add_argument(
        "--interaction_name",
        default="interaction_01",
        type=str,
        help="Interaction name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--video_dir",
        default=None,
        type=str,
        help="Path to directory like */output/interaction_xx containing one .mp4. "
        "Defaults to ../02_Generate_Video/output/<interaction_name>/.",
    )
    parser.add_argument(
        "--segment_video_dir",
        default=None,
        type=str,
        help=(
            "Path to 03_Segment_Video output for this video. "
            "Default: ../03_Segment_Video/output/<interaction_xx>"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=(
            "Output directory for this video. "
            "Default: <script_dir>/output_cotracker/<interaction_xx>"
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
        default=1000.0,
        type=float,
        help="Tracking seed density: points per 1000 mask pixels.",
    )
    parser.add_argument(
        "--track_point_max",
        default=5000,
        type=int,
        help="Maximum number of tracking seed points per object.",
    )
    parser.add_argument(
        "--mask_threshold",
        default=127,
        type=int,
        help="Threshold applied to mask grayscale image.",
    )
    parser.add_argument(
        "--mask_erode_px",
        default=3,
        type=int,
        help="Erode mask by this many pixels before point sampling.",
    )
    parser.add_argument(
        "--trail_length",
        default=0,
        type=int,
        help="Number of recent frames kept in track trails for visualization.",
    )
    parser.add_argument(
        "--vis_fps",
        default=6.0,
        type=float,
        help="Visualization FPS.",
    )
    parser.add_argument(
        "--vis_point_percent",
        default=5.0,
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


def _find_first_frame_mask(mask_dir: Path) -> Path | None:
    """Return frame-0 mask path, with fallback to first frame_*.png."""
    preferred = sorted(mask_dir.glob("frame_0000.*"))
    if preferred:
        return preferred[0]

    fallback = sorted(mask_dir.glob("frame_*.png"))
    if fallback:
        return fallback[0]

    fallback_any = sorted(mask_dir.glob("frame_*.*"))
    if fallback_any:
        return fallback_any[0]
    return None


def discover_segment_video_object_masks(segment_video_dir: Path) -> list[tuple[str, Path]]:
    """Discover object slugs and first-frame masks from 03_Segment_Video output."""
    objects_root = segment_video_dir / "objects"
    if not objects_root.exists() or not objects_root.is_dir():
        raise NotADirectoryError(f"03_Segment_Video objects dir not found: {objects_root}")

    discovered: list[tuple[str, Path]] = []
    for object_dir in sorted(p for p in objects_root.iterdir() if p.is_dir()):
        mask_dir = object_dir / "object_segmentation" / "masks"
        if not mask_dir.exists() or not mask_dir.is_dir():
            print(f"[WARN] Missing mask dir, skipping object '{object_dir.name}': {mask_dir}")
            continue

        frame0_mask = _find_first_frame_mask(mask_dir)
        if frame0_mask is None:
            print(f"[WARN] No frame masks found, skipping object '{object_dir.name}': {mask_dir}")
            continue

        discovered.append((object_dir.name, frame0_mask))

    if not discovered:
        raise RuntimeError(f"No object masks found in 03_Segment_Video dir: {segment_video_dir}")
    return discovered


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if int(args.track_point_max) <= 0:
        raise ValueError(f"--track_point_max must be > 0, got {args.track_point_max}")

    default_video_dir, default_segmentation_dir, default_output_video_dir = (
        build_default_paths(args.interaction_name, script_dir)
    )

    video_dir = (
        resolve_path(args.video_dir, script_dir) if args.video_dir else default_video_dir
    )
    if not video_dir.exists() or not video_dir.is_dir():
        raise NotADirectoryError(f"Interaction dir not found: {video_dir}")

    video_mp4 = find_single_mp4(video_dir)

    if args.segment_video_dir is not None:
        segmentation_dir = resolve_path(args.segment_video_dir, script_dir)
    else:
        segmentation_dir = default_segmentation_dir

    if not segmentation_dir.exists() or not segmentation_dir.is_dir():
        raise NotADirectoryError(f"03_Segment_Video dir not found: {segmentation_dir}")

    if args.output_dir is None:
        output_video_dir = default_output_video_dir
    else:
        output_video_dir = resolve_path(args.output_dir, script_dir)
    output_video_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] video_dir: {video_dir}")
    print(f"[INFO] video_mp4: {video_mp4.name}")
    print(f"[INFO] segment_video_dir: {segmentation_dir}")
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

    objects = discover_segment_video_object_masks(segmentation_dir)
    print(f"[INFO] discovered {len(objects)} object masks from 03_Segment_Video")

    run_summary = {
        "video_dir": str(video_dir),
        "video_file": str(video_mp4),
        "segment_video_dir": str(segmentation_dir),
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "track_point_density": float(args.track_point_density),
        "track_point_max": int(args.track_point_max),
        "vis_point_percent": float(args.vis_point_percent),
        "objects": [],
    }

    for obj_idx, (slug, mask_path) in enumerate(objects):
        obj_name = slug

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

        target_track_points_uncapped = compute_target_track_points(
            mask_area_px=mask_area_px,
            density_per_1kpx=float(args.track_point_density),
        )
        target_track_points = min(target_track_points_uncapped, int(args.track_point_max))
        if target_track_points < target_track_points_uncapped:
            print(
                "[INFO] Capped target points "
                f"from {target_track_points_uncapped} to {target_track_points}"
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
            "track_point_max": int(args.track_point_max),
            "track_target_points_uncapped": int(target_track_points_uncapped),
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
