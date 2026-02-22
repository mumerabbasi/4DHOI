"""Estimate per-object CoTracker3 tracks from frame-0 masks and render trail videos."""

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
    find_single_summary,
    normalize_slug,
    render_trails_video,
    resolve_mask_path,
    resolve_path,
    sample_points_from_mask,
    sample_visualization_indices,
)


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
        default=10,
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


def clear_dir(path: Path) -> None:
    """Delete directory contents if it exists, then recreate it."""
    if path.exists():
        for child in path.glob("*"):
            if child.is_file():
                child.unlink()
            else:
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def save_frames(frames_bgr: list[np.ndarray], frames_dir: Path) -> None:
    """Save decoded video frames as BGR PNGs to frames_dir."""
    clear_dir(frames_dir)
    for idx, frame_bgr in enumerate(frames_bgr):
        out_path = frames_dir / f"frame_{idx:04d}.png"
        ok = cv2.imwrite(str(out_path), frame_bgr)
        if not ok:
            raise RuntimeError(f"Failed to write frame: {out_path}")


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
    frames_dir = output_video_dir / "_frames"

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
    save_frames(frames_bgr, frames_dir)
    print(f"[OK] Saved {len(frames_bgr)} frames -> {frames_dir}")
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
