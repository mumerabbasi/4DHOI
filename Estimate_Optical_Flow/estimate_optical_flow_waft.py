#!/usr/bin/env python3
"""
Estimate optical flow on a video using WAFT (Spring checkpoint).
Run with conda environment: waft

Input:
- --input_dir: directory like */videos/video_xx containing exactly one .mp4

Output (created under THIS script's directory):
./output_waft/video_xx/
  |_ _frames/                   extracted frames (BGR PNGs)
  |_ _frames_visualization/     flow visualization PNGs
  |_ _frames_arrows/            arrow overlay PNGs on source frames
  |_ optical_flow/              raw flow .npy files
  |_ visualization.mp4          flow visualization video (MP4)
  |_ arrows.mp4                 arrow overlay video (MP4)
  |_ <object_name>/             per-object visualization from masks
      |_ trails.mp4             colored points + trails over video
      |_ seed_points_frame0.npy
      |_ tracks.npy             [T, N, 2]
      |_ visibility.npy         [T, N] bool
      |_ metadata.json
  |_ run_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from optical_flow_utils import (
    close_ffmpeg,
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
    start_ffmpeg_writer,
)


@dataclass(frozen=True)
class Paths:
    """Derived input/output paths."""
    script_dir: Path
    waft_dir: Path
    input_dir: Path
    input_mp4: Path
    out_video_dir: Path
    frames_dir: Path
    frames_vis_dir: Path
    arrows_vis_dir: Path
    flow_dir: Path
    vis_mp4_path: Path
    arrows_mp4_path: Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run WAFT optical flow on a video directory (video_xx)."
    )
    parser.add_argument(
        "--input_dir",
        default="../Generate_Video/videos/video_01",
        type=str,
        help="Path to a directory like */videos/video_xx containing an .mp4.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help=(
            "Output directory for this video. "
            "Default: <script_dir>/output_waft/<video_xx>"
        ),
    )
    parser.add_argument(
        "--waft_dir",
        default="/my_workspace/4DHHOI/WAFT",
        type=str,
        help="Path to WAFT repo root. Default: <script_dir>/WAFT",
    )
    parser.add_argument(
        "--cfg",
        default=None,
        type=str,
        help=(
            "WAFT config JSON. Default: "
            "<waft_dir>/config/a2/dinov3/tar-c-t-spring-540p.json"
        ),
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="Checkpoint .pth. Default: <waft_dir>/ckpts/a2/dinov3/spring.pth",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="torch device string, e.g. 'cuda' or 'cuda:0' or 'cpu'.",
    )
    parser.add_argument(
        "--scale",
        default=0.0,
        type=float,
        help="WAFT InferenceWrapper scale factor (0.0 keeps native).",
    )
    parser.add_argument(
        "--arrow_scale",
        default=1.0,
        type=float,
        help="Scale factor for arrow length in visualization (default 1.0).",
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
        help="Number of recent frames kept in trail visualization.",
    )
    parser.add_argument(
        "--trails_fps",
        default=6.0,
        type=float,
        help="Per-object trails FPS. If <= 0, falls back to source video FPS.",
    )
    parser.add_argument(
        "--vis_point_percent",
        default=2.5,
        type=float,
        help="Percentage of tracked points to visualize in trails.mp4. Range: (0, 100].",
    )
    parser.add_argument(
        "--vis_seed",
        default=1234,
        type=int,
        help="Random seed used for visualization point subsampling.",
    )
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> Paths:
    """Build Paths object from CLI args."""
    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir is not a directory: {input_dir}")

    input_mp4 = find_single_mp4(input_dir)
    video_name = input_dir.name  # expected "video_xx"

    waft_dir = Path(args.waft_dir).resolve() if args.waft_dir else (script_dir / "WAFT")
    if not waft_dir.exists():
        raise FileNotFoundError(
            f"WAFT repo not found at: {waft_dir}. "
            "Pass --waft_dir to point to your WAFT clone."
        )

    if args.output_dir:
        out_video_dir = Path(args.output_dir)
        if not out_video_dir.is_absolute():
            out_video_dir = script_dir / out_video_dir
        out_video_dir = out_video_dir.resolve()
    else:
        out_video_dir = (script_dir / "output_waft" / video_name).resolve()

    frames_dir = out_video_dir / "_frames"
    frames_vis_dir = out_video_dir / "_frames_visualization"
    arrows_vis_dir = out_video_dir / "_frames_arrows"
    flow_dir = out_video_dir / "optical_flow"
    vis_mp4_path = out_video_dir / "visualization.mp4"
    arrows_mp4_path = out_video_dir / "visualization_arrows.mp4"

    return Paths(
        script_dir=script_dir,
        waft_dir=waft_dir,
        input_dir=input_dir,
        input_mp4=input_mp4,
        out_video_dir=out_video_dir,
        frames_dir=frames_dir,
        frames_vis_dir=frames_vis_dir,
        arrows_vis_dir=arrows_vis_dir,
        flow_dir=flow_dir,
        vis_mp4_path=vis_mp4_path,
        arrows_mp4_path=arrows_mp4_path,
    )


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


def extract_frames(mp4_path: Path, frames_dir: Path) -> Tuple[int, float]:
    """
    Extract all frames from mp4 into frames_dir as PNG.

    Returns:
        (num_frames, fps)
    """
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 24.0

    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        out_path = frames_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(out_path), frame_bgr)
        idx += 1

    cap.release()
    return idx, float(fps)


def add_waft_to_syspath(waft_dir: Path) -> None:
    """Ensure WAFT modules are importable."""
    waft_dir_str = str(waft_dir)
    if waft_dir_str not in sys.path:
        sys.path.insert(0, waft_dir_str)


def _patch_waft_dinov3_weights() -> None:
    """Patch WAFT's dinov3 module to use direct download URLs."""
    # Direct download URLs for DINOv3 weights (licensed from Meta)
    # These are signed URLs that bypass the public endpoint restrictions
    dinov3_urls = {
        "vits": (
            "https://dinov3.llamameta.net/dinov3_vits16/"
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth?"
            "Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZHhub3htb2VzcXpsM3V"
            "zYnd3aDRpc2RtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0Y"
            "S5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFR"
            "pbWUiOjE3NzA1Mjk4Mzh9fX1dfQ__&Signature=pS2lHy5xzRbTqr6bQqCKqzzEGNy"
            "oRGSp2DaKHowhG-gALWgF4HAGsvvyfCfy%7ExqRFPRcBwQFOmWQXG4vBweC%7ET3CH"
            "8bixblE%7EM5n%7EwbA80a3HWHWZ33mJ%7ExTXS1WQML6B3dJWfSuK4R3xaIYd5Ev6"
            "1MwOZsdGtma0%7EByzlWQSUZCU9YLB2pHEYEkJSBvaeFn2TM0whIKo7m0MH02N9%7E"
            "7ett-BBmnGqJ-XT-1XCula0YZzTUmF4HcXG3u83SNrq7n4zjJRlIXK-0r7m9NoWjwv"
            "IFLVvM74Ru3XHUZxSG-cump-VX2c%7EFDtjibFArHzX4nKeZE2VmZYlQLJ-WJDUMRt"
            "A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1457298125810808"
        ),
        "vitb": (
            "https://dinov3.llamameta.net/dinov3_vitb16/"
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth?"
            "Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZHhub3htb2VzcXpsM3V"
            "zYnd3aDRpc2RtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0Y"
            "S5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFR"
            "pbWUiOjE3NzA1Mjk4Mzh9fX1dfQ__&Signature=pS2lHy5xzRbTqr6bQqCKqzzEGNy"
            "oRGSp2DaKHowhG-gALWgF4HAGsvvyfCfy%7ExqRFPRcBwQFOmWQXG4vBweC%7ET3CH"
            "8bixblE%7EM5n%7EwbA80a3HWHWZ33mJ%7ExTXS1WQML6B3dJWfSuK4R3xaIYd5Ev6"
            "1MwOZsdGtma0%7EByzlWQSUZCU9YLB2pHEYEkJSBvaeFn2TM0whIKo7m0MH02N9%7E"
            "7ett-BBmnGqJ-XT-1XCula0YZzTUmF4HcXG3u83SNrq7n4zjJRlIXK-0r7m9NoWjwv"
            "IFLVvM74Ru3XHUZxSG-cump-VX2c%7EFDtjibFArHzX4nKeZE2VmZYlQLJ-WJDUMRt"
            "A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1457298125810808"
        ),
        "vitl": (
            "https://dinov3.llamameta.net/dinov3_vitl16/"
            "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth?"
            "Policy=eyJTdGF0ZW1lbnQiOlt7InVuaXF1ZV9oYXNoIjoiZHhub3htb2VzcXpsM3V"
            "zYnd3aDRpc2RtIiwiUmVzb3VyY2UiOiJodHRwczpcL1wvZGlub3YzLmxsYW1hbWV0Y"
            "S5uZXRcLyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFR"
            "pbWUiOjE3NzA1Mjk4Mzh9fX1dfQ__&Signature=pS2lHy5xzRbTqr6bQqCKqzzEGNy"
            "oRGSp2DaKHowhG-gALWgF4HAGsvvyfCfy%7ExqRFPRcBwQFOmWQXG4vBweC%7ET3CH"
            "8bixblE%7EM5n%7EwbA80a3HWHWZ33mJ%7ExTXS1WQML6B3dJWfSuK4R3xaIYd5Ev6"
            "1MwOZsdGtma0%7EByzlWQSUZCU9YLB2pHEYEkJSBvaeFn2TM0whIKo7m0MH02N9%7E"
            "7ett-BBmnGqJ-XT-1XCula0YZzTUmF4HcXG3u83SNrq7n4zjJRlIXK-0r7m9NoWjwv"
            "IFLVvM74Ru3XHUZxSG-cump-VX2c%7EFDtjibFArHzX4nKeZE2VmZYlQLJ-WJDUMRt"
            "A__&Key-Pair-Id=K15QRJLYKIFSLZ&Download-Request-ID=1457298125810808"
        ),
    }

    # Patch the WEIGHTS_URLS in WAFT's dinov3 backbone module
    import model.backbone.dinov3 as waft_dinov3  # type: ignore

    waft_dinov3.WEIGHTS_URLS = dinov3_urls


def load_waft_model(
    cfg_path: Path,
    ckpt_path: Path,
    device: torch.device,
    scale: float,
    waft_dir: Path,
):
    """
    Load WAFT model and wrap it with InferenceWrapper.

    Imports happen after sys.path modification.
    """
    # WAFT uses relative paths for thirdparty modules
    original_cwd = os.getcwd()
    os.chdir(waft_dir)

    try:
        from config.parser import json_to_args  # type: ignore
        from inference_tools import InferenceWrapper  # type: ignore
        from model import fetch_model  # type: ignore
        from utils.utils import load_ckpt  # type: ignore

        # Patch dinov3 weights before model creation
        _patch_waft_dinov3_weights()

        # Load config from JSON and set required attributes
        waft_args = json_to_args(str(cfg_path))
        waft_args.cfg = str(cfg_path)
        waft_args.ckpt = str(ckpt_path)
        waft_args.scale = scale

        model = fetch_model(waft_args)
        load_ckpt(model, str(ckpt_path))
        model = model.to(device).eval()

        wrapped = InferenceWrapper(
            model,
            scale=scale,
            train_size=waft_args.image_size,
            pad_to_train_size=False,
            tiling=False,
        )
        return wrapped
    finally:
        os.chdir(original_cwd)


def read_frame_rgb(frames_dir: Path, idx: int) -> np.ndarray:
    """Read frame as RGB uint8 (H, W, 3)."""
    frame_path = frames_dir / f"frame_{idx:04d}.png"
    bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _sample_grid_points(
    h: int, w: int, step: int = 20,
) -> np.ndarray:
    """Return uniformly sampled (y, x) grid points as (N, 2) int array."""
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)
    grid = np.stack(np.meshgrid(ys, xs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def draw_flow_arrows(
    frame_bgr: np.ndarray,
    flow: np.ndarray,
    points: np.ndarray,
    arrow_scale: float = 1.0,
    arrow_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    tip_length: float = 0.3,
) -> np.ndarray:
    """Draw flow arrows on a copy of *frame_bgr* at the given sample *points*."""
    vis = frame_bgr.copy()
    for y, x in points:
        dx, dy = flow[y, x]
        x2 = int(round(x + arrow_scale * dx))
        y2 = int(round(y + arrow_scale * dy))
        cv2.arrowedLine(
            vis, (int(x), int(y)), (x2, y2),
            color=arrow_color, thickness=thickness, tipLength=tip_length,
        )
    return vis


def load_frames_bgr(frames_dir: Path, num_frames: int) -> list[np.ndarray]:
    """Load extracted frames as BGR arrays."""
    frames_bgr = []
    for idx in range(num_frames):
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read frame: {frame_path}")
        frames_bgr.append(frame)
    return frames_bgr


def propagate_tracks_from_flow(
    seed_xy: np.ndarray,
    flow_dir: Path,
    num_frames: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate seed points using dense frame-to-frame optical flow."""
    num_points = int(seed_xy.shape[0])
    tracks = np.zeros((num_frames, num_points, 2), dtype=np.float32)
    visibility = np.zeros((num_frames, num_points), dtype=np.bool_)
    if num_points == 0:
        return tracks, visibility

    u = seed_xy.astype(np.float32).copy()
    alive = np.ones(num_points, dtype=bool)
    tracks[0] = u
    visibility[0] = alive

    for t in range(num_frames - 1):
        flow_path = flow_dir / f"frame_{t:04d}.npy"
        if not flow_path.exists():
            raise FileNotFoundError(f"Missing flow file: {flow_path}")

        flow = np.load(str(flow_path)).astype(np.float32)
        if flow.shape[:2] != (height, width) or flow.shape[2] != 2:
            raise ValueError(f"Bad flow shape {flow.shape} at {flow_path}")

        idx_alive = np.where(alive)[0]
        if len(idx_alive) > 0:
            x_idx = np.round(u[idx_alive, 0]).astype(np.int32)
            y_idx = np.round(u[idx_alive, 1]).astype(np.int32)

            in_bounds_before = (
                (x_idx >= 0)
                & (x_idx <= (width - 1))
                & (y_idx >= 0)
                & (y_idx <= (height - 1))
            )
            alive[idx_alive[~in_bounds_before]] = False
            idx_alive = idx_alive[in_bounds_before]
            if len(idx_alive) > 0:
                x_safe = np.clip(x_idx[in_bounds_before], 0, width - 1)
                y_safe = np.clip(y_idx[in_bounds_before], 0, height - 1)
                d = flow[y_safe, x_safe]
                finite = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])
                alive[idx_alive[~finite]] = False

                idx_valid = idx_alive[finite]
                if len(idx_valid) > 0:
                    u[idx_valid] = u[idx_valid] + d[finite]

        in_bounds_after = (
            (u[:, 0] >= 0.0)
            & (u[:, 0] <= (width - 1))
            & (u[:, 1] >= 0.0)
            & (u[:, 1] <= (height - 1))
        )
        alive &= in_bounds_after
        tracks[t + 1] = u
        visibility[t + 1] = alive

    return tracks, visibility


def generate_object_trails(
    paths: Paths,
    args: argparse.Namespace,
    num_frames: int,
    video_fps: float,
) -> None:
    """Generate per-object colored trails.mp4 using frame-0 masks."""
    video_name = paths.input_dir.name
    if args.object_mesh_dir is None:
        object_mesh_dir = (
            paths.script_dir.parent / "Generate_Object_Mesh" / "output" / video_name
        ).resolve()
    else:
        object_mesh_dir = resolve_path(args.object_mesh_dir, paths.script_dir)

    if not object_mesh_dir.exists() or not object_mesh_dir.is_dir():
        raise NotADirectoryError(f"Object mesh dir not found: {object_mesh_dir}")

    summary_path = find_single_summary(object_mesh_dir)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    objects = [obj for obj in summary.get("objects", []) if obj.get("success", False)]
    if not objects:
        print(f"[WARN] No successful objects found in summary: {summary_path}")
        return

    frames_bgr = load_frames_bgr(paths.frames_dir, num_frames)
    height, width = frames_bgr[0].shape[:2]
    trails_fps = float(video_fps) if float(args.trails_fps) <= 0 else float(args.trails_fps)

    run_summary = {
        "video_dir": str(paths.input_dir),
        "video_file": str(paths.input_mp4),
        "summary_path": str(summary_path),
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "track_point_density": float(args.track_point_density),
        "vis_point_percent": float(args.vis_point_percent),
        "objects": [],
    }

    print(f"[INFO] object_mesh_dir: {object_mesh_dir}")
    print(f"[INFO] summary: {summary_path.name}")

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

        tracks, visibility = propagate_tracks_from_flow(
            seed_xy=seed_xy,
            flow_dir=paths.flow_dir,
            num_frames=num_frames,
            width=width,
            height=height,
        )

        vis_indices = sample_visualization_indices(
            num_points=tracks.shape[1],
            percent=float(args.vis_point_percent),
            seed=int(args.vis_seed) + obj_idx,
        )

        obj_out = paths.out_video_dir / slug
        clear_dir(obj_out)
        np.save(str(obj_out / "seed_points_frame0.npy"), seed_xy.astype(np.float32))
        np.save(str(obj_out / "tracks.npy"), tracks.astype(np.float32))
        np.save(str(obj_out / "visibility.npy"), visibility.astype(np.bool_))

        render_trails_video(
            frames_bgr=frames_bgr,
            tracks=tracks,
            visibility=visibility,
            out_path=obj_out / "trails.mp4",
            fps=trails_fps,
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
        with (obj_out / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        run_summary["objects"].append(metadata)
        print(f"[OK] Saved: {obj_out}")

    with (paths.out_video_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)


def run_flow(
    wrapped_model,
    frames_dir: Path,
    frames_vis_dir: Path,
    arrows_vis_dir: Path,
    flow_dir: Path,
    vis_mp4_path: Path,
    arrows_mp4_path: Path,
    num_frames: int,
    device: torch.device,
    fps: float,
    arrow_scale: float = 1.0,
) -> None:
    """Compute optical flow for consecutive pairs and save outputs."""
    from utils.flow_viz import flow_to_image  # type: ignore

    if num_frames < 2:
        raise RuntimeError(f"Need at least 2 frames, found {num_frames}")

    first_rgb = read_frame_rgb(frames_dir, 0)
    h, w = first_rgb.shape[:2]
    sample_pts = _sample_grid_points(h, w, step=20)

    vis_writer = start_ffmpeg_writer(vis_mp4_path, fps, (h, w))
    arrow_writer = start_ffmpeg_writer(arrows_mp4_path, fps, (h, w))

    for i in range(num_frames - 1):
        rgb1 = read_frame_rgb(frames_dir, i)
        rgb2 = read_frame_rgb(frames_dir, i + 1)

        im1 = torch.from_numpy(rgb1).float().permute(2, 0, 1)[None].to(device)
        im2 = torch.from_numpy(rgb2).float().permute(2, 0, 1)[None].to(device)

        with torch.no_grad():
            out = wrapped_model.calc_flow(im1, im2)

        flow = out["flow"][-1][0].permute(1, 2, 0).detach().cpu().numpy()
        np.save(str(flow_dir / f"frame_{i:04d}.npy"), flow)

        # Color flow visualization
        vis_bgr = flow_to_image(flow, convert_to_bgr=True)
        cv2.imwrite(str(frames_vis_dir / f"frame_{i:04d}.png"), vis_bgr)
        if vis_writer.stdin is not None:
            vis_writer.stdin.write(vis_bgr.tobytes())

        # Arrow visualization on source frame
        frame_bgr = cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
        arrow_bgr = draw_flow_arrows(
            frame_bgr, flow, sample_pts, arrow_scale=arrow_scale
        )
        cv2.imwrite(str(arrows_vis_dir / f"frame_{i:04d}.png"), arrow_bgr)
        if arrow_writer.stdin is not None:
            arrow_writer.stdin.write(arrow_bgr.tobytes())

    close_ffmpeg(vis_writer)
    close_ffmpeg(arrow_writer)


def main() -> None:
    """Entry point."""
    args = parse_args()
    paths = build_paths(args)

    cfg_default = (
        paths.waft_dir
        / "config"
        / "a2"
        / "dinov3"
        / "tar-c-t-spring-540p.json"
    )
    ckpt_default = paths.waft_dir / "ckpts" / "a2" / "dinov3" / "spring.pth"

    cfg_path = Path(args.cfg).resolve() if args.cfg else cfg_default
    ckpt_path = Path(args.ckpt).resolve() if args.ckpt else ckpt_default

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Overwrite outputs by default
    clear_dir(paths.frames_dir)
    clear_dir(paths.frames_vis_dir)
    clear_dir(paths.arrows_vis_dir)
    clear_dir(paths.flow_dir)
    for mp4 in (paths.vis_mp4_path, paths.arrows_mp4_path):
        if mp4.exists():
            mp4.unlink()

    num_frames, fps = extract_frames(paths.input_mp4, paths.frames_dir)
    print(f"[OK] Extracted {num_frames} frames -> {paths.frames_dir}")

    add_waft_to_syspath(paths.waft_dir)
    device = torch.device(args.device)

    wrapped_model = load_waft_model(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        device=device,
        scale=float(args.scale),
        waft_dir=paths.waft_dir,
    )
    print(f"[OK] Loaded WAFT checkpoint: {ckpt_path.name}")

    run_flow(
        wrapped_model=wrapped_model,
        frames_dir=paths.frames_dir,
        frames_vis_dir=paths.frames_vis_dir,
        arrows_vis_dir=paths.arrows_vis_dir,
        flow_dir=paths.flow_dir,
        vis_mp4_path=paths.vis_mp4_path,
        arrows_mp4_path=paths.arrows_mp4_path,
        num_frames=num_frames,
        device=device,
        fps=fps,
        arrow_scale=float(args.arrow_scale),
    )
    print("[OK] Saved dense optical flow and global visualizations")

    generate_object_trails(
        paths=paths,
        args=args,
        num_frames=num_frames,
        video_fps=fps,
    )

    print(f"[OK] Saved frames -> {paths.frames_dir}")
    print(f"[OK] Saved flow PNGs -> {paths.frames_vis_dir}")
    print(f"[OK] Saved arrow PNGs -> {paths.arrows_vis_dir}")
    print(f"[OK] Saved flows -> {paths.flow_dir}")
    print(f"[OK] Saved video -> {paths.vis_mp4_path}")
    print(f"[OK] Saved arrows video -> {paths.arrows_mp4_path}")
    print(f"[OK] Saved object trails summary -> {paths.out_video_dir / 'run_summary.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    main()
