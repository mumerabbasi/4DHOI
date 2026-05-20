"""Estimate per-object WAFT tracks from 03_Segment_Video frame-0 masks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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
    output_dir = script_dir / "output_waft" / interaction_name
    return video_dir, segment_video_dir, output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run WAFT-based tracking per object for an interaction_xx directory."
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
            "Default: <script_dir>/output_waft/<interaction_xx>"
        ),
    )
    parser.add_argument(
        "--waft_dir",
        default="/my_workspace/4DHHOI/WAFT",
        type=str,
        help="Path to WAFT repo root.",
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
        help="WAFT checkpoint .pth. Default: <waft_dir>/ckpts/a2/dinov3/spring.pth",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="torch device string, e.g. 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--scale",
        default=0.0,
        type=float,
        help="WAFT InferenceWrapper scale factor (0.0 keeps native).",
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
    return parser.parse_args()


def load_video_frames(video_mp4: Path) -> tuple[list[np.ndarray], float]:
    """Load MP4 as BGR frames and return video FPS."""
    cap = cv2.VideoCapture(str(video_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_mp4}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 24.0

    frames_bgr: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames_bgr.append(frame_bgr)

    cap.release()

    if len(frames_bgr) < 2:
        raise RuntimeError(f"Need at least 2 frames, found {len(frames_bgr)}")
    return frames_bgr, float(fps)


def add_waft_to_syspath(waft_dir: Path) -> None:
    """Ensure WAFT modules are importable."""
    waft_dir_str = str(waft_dir)
    if waft_dir_str not in sys.path:
        sys.path.insert(0, waft_dir_str)


def _patch_waft_dinov3_weights() -> None:
    """Patch WAFT's dinov3 module to use direct download URLs."""
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

    import model.backbone.dinov3 as waft_dinov3  # type: ignore

    waft_dinov3.WEIGHTS_URLS = dinov3_urls


def load_waft_model(
    cfg_path: Path,
    ckpt_path: Path,
    device: torch.device,
    scale: float,
    waft_dir: Path,
):
    """Load WAFT model and wrap it with InferenceWrapper."""
    original_cwd = os.getcwd()
    os.chdir(waft_dir)

    try:
        from config.parser import json_to_args  # type: ignore
        from inference_tools import InferenceWrapper  # type: ignore
        from model import fetch_model  # type: ignore
        from utils.utils import load_ckpt  # type: ignore

        _patch_waft_dinov3_weights()

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


def compute_dense_flows(
    wrapped_model,
    frames_bgr: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    """Compute WAFT dense flow for all consecutive frame pairs."""
    num_frames = len(frames_bgr)
    if num_frames < 2:
        raise RuntimeError(f"Need at least 2 frames, found {num_frames}")

    flows: list[np.ndarray] = []
    for t in range(num_frames - 1):
        rgb1 = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frames_bgr[t + 1], cv2.COLOR_BGR2RGB)

        im1 = torch.from_numpy(rgb1).float().permute(2, 0, 1)[None].to(device)
        im2 = torch.from_numpy(rgb2).float().permute(2, 0, 1)[None].to(device)

        with torch.no_grad():
            out = wrapped_model.calc_flow(im1, im2)

        flow = out["flow"][-1][0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        flows.append(flow)

    return flows


def propagate_tracks_from_flows(
    seed_xy: np.ndarray,
    flows: list[np.ndarray],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate seed points using dense frame-to-frame optical flow."""
    num_frames = len(flows) + 1
    num_points = int(seed_xy.shape[0])
    tracks = np.zeros((num_frames, num_points, 2), dtype=np.float32)
    visibility = np.zeros((num_frames, num_points), dtype=np.bool_)
    if num_points == 0:
        return tracks, visibility

    u = seed_xy.astype(np.float32).copy()
    alive = np.ones(num_points, dtype=bool)
    tracks[0] = u
    visibility[0] = alive

    for t, flow in enumerate(flows):
        if flow.shape[:2] != (height, width) or flow.shape[2] != 2:
            raise ValueError(f"Bad flow shape {flow.shape} at frame {t}")

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


def cleanup_waft_extras(output_video_dir: Path) -> None:
    """Remove legacy WAFT-only artifacts so output mirrors CoTracker layout."""
    legacy_paths = [
        output_video_dir / "optical_flow",
        output_video_dir / "visualization.mp4",
        output_video_dir / "visualization_arrows.mp4",
    ]
    for path in legacy_paths:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


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
    cleanup_waft_extras(output_video_dir)

    waft_dir = resolve_path(args.waft_dir, script_dir)
    if not waft_dir.exists() or not waft_dir.is_dir():
        raise NotADirectoryError(f"WAFT repo not found: {waft_dir}")

    cfg_default = waft_dir / "config" / "a2" / "dinov3" / "tar-c-t-spring-540p.json"
    ckpt_default = waft_dir / "ckpts" / "a2" / "dinov3" / "spring.pth"
    cfg_path = resolve_path(args.cfg, script_dir) if args.cfg else cfg_default
    ckpt_path = resolve_path(args.ckpt, script_dir) if args.ckpt else ckpt_default

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device(
        "cuda" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    print(f"[INFO] video_dir: {video_dir}")
    print(f"[INFO] video_mp4: {video_mp4.name}")
    print(f"[INFO] segment_video_dir: {segmentation_dir}")
    print(f"[INFO] output_dir: {output_video_dir}")
    print(f"[INFO] waft_dir: {waft_dir}")
    print(f"[INFO] cfg: {cfg_path}")
    print(f"[INFO] ckpt: {ckpt_path}")
    print(f"[INFO] device: {device}")

    print("[INFO] Loading video frames...")
    frames_bgr, video_fps = load_video_frames(video_mp4)
    num_frames = len(frames_bgr)
    height, width = frames_bgr[0].shape[:2]
    print(f"[OK] loaded {num_frames} frames")
    vis_fps = float(args.vis_fps) if args.vis_fps is not None else float(video_fps)

    print("[INFO] Loading WAFT model...")
    add_waft_to_syspath(waft_dir)
    wrapped_model = load_waft_model(
        cfg_path=cfg_path,
        ckpt_path=ckpt_path,
        device=device,
        scale=float(args.scale),
        waft_dir=waft_dir,
    )
    print("[OK] WAFT loaded")

    print("[INFO] Computing dense WAFT flow...")
    flows = compute_dense_flows(wrapped_model=wrapped_model, frames_bgr=frames_bgr, device=device)
    print(f"[OK] computed {len(flows)} flow fields")

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

        tracks, visibility = propagate_tracks_from_flows(
            seed_xy=seed_xy,
            flows=flows,
            width=width,
            height=height,
        )

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
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    main()
