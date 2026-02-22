"""Rigid 4D tracking (MVP + Improvement 1) using CoTracker tracks + visibility.

This script mirrors track_object_mesh.py outputs but consumes CoTracker outputs
produced by estimate_optical_flow_cotracker.py.

Input layout (per video):
  Estimate_Optical_Flow/output_cotracker/video_xx/
    |_ _frames/frame_XXXX.png
    |_ <object>/
        |_ seed_points_frame0.npy
        |_ tracks.npy         # [T, N, 2]
        |_ visibility.npy     # [T, N]

Output layout:
  ./output_cotracker/video_xx/<object>/
    |_ meshes/frame_XXXX.ply
    |_ overlays/overlay_XXXX.png
    |_ overlay.mp4
    |_ poses.npy
    |_ poses.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import cv2
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tracking_utils import (
    apply_pose,
    close_ffmpeg,
    discover_object_dirs,
    draw_overlay,
    ensure_dir,
    estimate_pose_pnp_ransac,
    list_images,
    load_intrinsics_pixels_3x3,
    load_mesh,
    make_rasterizer,
    pixel_to_points,
    project_points_via_f,
    rasterize_gbuffer,
    resolve_device,
    resolve_path,
    save_pose_outputs,
    start_ffmpeg_writer,
    y_up_to_z_up,
)


def list_object_slugs(mesh_source: str, mesh_video_dir: Path) -> List[str]:
    if mesh_source == "generate":
        return discover_object_dirs(mesh_video_dir, None)
    meshes_dir = mesh_video_dir / "meshes"
    if not meshes_dir.exists() or not meshes_dir.is_dir():
        raise NotADirectoryError(f"Aligned meshes dir not found: {meshes_dir}")
    return sorted([p.stem for p in meshes_dir.glob("*.ply") if p.stem != "human"])


def resolve_intrinsics_path(script_dir: Path, video_name: str, mesh_video_dir: Path) -> Path:
    candidates = [
        mesh_video_dir / "camera_intrinsics.json",
        script_dir.parent / "Generate_Object_Mesh" / "output" / video_name / "camera_intrinsics.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "camera_intrinsics.json not found. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _rotation_matrix_to_quaternion_wxyz(r: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to normalized quaternion [w, x, y, z]."""
    m00, m01, m02 = float(r[0, 0]), float(r[0, 1]), float(r[0, 2])
    m10, m11, m12 = float(r[1, 0]), float(r[1, 1]), float(r[1, 2])
    m20, m21, m22 = float(r[2, 0]), float(r[2, 1]), float(r[2, 2])
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m00 - m11 - m22))
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m11 - m00 - m22))
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m22 - m00 - m11))
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to rotation matrix."""
    q = q.astype(np.float64, copy=False)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = q / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return r.astype(np.float32)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions."""
    q0 = q0.astype(np.float64, copy=False)
    q1 = q1.astype(np.float64, copy=False)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + float(t) * (q1 - q0)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return q0
        return q / n

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = float(t) * theta_0
    sin_theta = float(np.sin(theta))

    s0 = float(np.sin(theta_0 - theta) / max(sin_theta_0, 1e-12))
    s1 = float(sin_theta / max(sin_theta_0, 1e-12))
    q = s0 * q0 + s1 * q1
    n = np.linalg.norm(q)
    if n < 1e-12:
        return q0
    return q / n


def smooth_pose_ema_slerp(
    r_prev: np.ndarray,
    t_prev: np.ndarray,
    r_curr: np.ndarray,
    t_curr: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """EMA smoothing for translation + SLERP-EMA smoothing for rotation."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return r_prev.copy(), t_prev.copy()
    if a >= 1.0:
        return r_curr.copy(), t_curr.copy()

    t_sm = (1.0 - a) * t_prev + a * t_curr
    q_prev = _rotation_matrix_to_quaternion_wxyz(r_prev)
    q_curr = _rotation_matrix_to_quaternion_wxyz(r_curr)
    q_sm = _slerp_wxyz(q_prev, q_curr, a)
    r_sm = _quaternion_wxyz_to_rotation_matrix(q_sm)
    return r_sm.astype(np.float32), t_sm.astype(np.float32)


def rotation_delta_deg(r_prev: np.ndarray, r_curr: np.ndarray) -> float:
    """Angular difference between two rotation matrices in degrees."""
    r_rel = r_prev.T @ r_curr
    cos_theta = 0.5 * (np.trace(r_rel) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def compute_correspondence_debug(
    x_obj: np.ndarray,
    u: np.ndarray,
    alive: np.ndarray,
    track_ids: np.ndarray,
    vis_conf: np.ndarray,
    frame_idx: int,
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    k: np.ndarray,
    reproj_px: float,
) -> dict[str, np.ndarray | float | int]:
    """Compute residual-based debug stats and correspondence partitions."""
    idx_alive = np.where(alive)[0]
    num_alive = int(len(idx_alive))
    if num_alive == 0:
        return {
            "idx_alive": idx_alive,
            "u_alive": np.zeros((0, 2), dtype=np.float32),
            "u_proj": np.zeros((0, 2), dtype=np.float32),
            "errors": np.zeros((0,), dtype=np.float32),
            "inlier_mask": np.zeros((0,), dtype=bool),
            "num_alive": 0,
            "num_reproj_inliers": 0,
            "inlier_ratio": 0.0,
            "reproj_mean": np.nan,
            "reproj_median": np.nan,
            "reproj_p90": np.nan,
            "visibility_mean": np.nan,
        }

    x_alive = x_obj[idx_alive]
    u_alive = u[idx_alive]
    u_proj = project_points_via_f(apply_pose(x_alive, r_mat, t_vec), k).astype(np.float32)
    residual = u_alive - u_proj
    errors = np.linalg.norm(residual, axis=1).astype(np.float32)
    inlier_mask = errors <= float(reproj_px)
    num_reproj_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_reproj_inliers / max(1, num_alive))

    vis_alive = vis_conf[int(frame_idx), track_ids[idx_alive]].astype(np.float32)
    visibility_mean = float(np.mean(vis_alive)) if len(vis_alive) > 0 else np.nan

    return {
        "idx_alive": idx_alive,
        "u_alive": u_alive,
        "u_proj": u_proj,
        "errors": errors,
        "inlier_mask": inlier_mask,
        "num_alive": num_alive,
        "num_reproj_inliers": num_reproj_inliers,
        "inlier_ratio": inlier_ratio,
        "reproj_mean": float(np.mean(errors)),
        "reproj_median": float(np.median(errors)),
        "reproj_p90": float(np.percentile(errors, 90.0)),
        "visibility_mean": visibility_mean,
    }


def save_debug_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Save per-frame debug metrics to CSV."""
    if len(rows) == 0:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_debug_plots(
    out_dir: Path,
    rows: list[dict[str, float | int]],
    reproj_px: float,
) -> None:
    """Save basic time-series plots for tracking diagnostics."""
    if len(rows) == 0:
        return

    frames = np.array([int(r["frame_idx"]) for r in rows], dtype=np.int32)
    alive = np.array([float(r["num_alive"]) for r in rows], dtype=np.float32)
    ransac_in = np.array([float(r["ransac_inliers"]) for r in rows], dtype=np.float32)
    reproj_in = np.array([float(r["reproj_inliers"]) for r in rows], dtype=np.float32)
    reproj_mean = np.array([float(r["reproj_mean"]) for r in rows], dtype=np.float32)
    reproj_med = np.array([float(r["reproj_median"]) for r in rows], dtype=np.float32)
    reproj_p90 = np.array([float(r["reproj_p90"]) for r in rows], dtype=np.float32)
    delta_t = np.array([float(r["delta_t"]) for r in rows], dtype=np.float32)
    delta_rot = np.array([float(r["delta_rot_deg"]) for r in rows], dtype=np.float32)
    vis_mean = np.array([float(r["visibility_mean"]) for r in rows], dtype=np.float32)

    def _finalize(fig: plt.Figure, path: Path) -> None:
        fig.tight_layout()
        fig.savefig(str(path), dpi=150)
        plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(frames, alive, label="alive")
    ax.plot(frames, ransac_in, label="ransac_inliers")
    ax.plot(frames, reproj_in, label="reproj_inliers")
    ax.set_title("Tracks and Inliers")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize(fig, out_dir / "tracks_inliers.png")

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(frames, reproj_mean, label="mean")
    ax.plot(frames, reproj_med, label="median")
    ax.plot(frames, reproj_p90, label="p90")
    ax.axhline(float(reproj_px), color="k", linestyle="--", linewidth=1.0, label="ransac_reproj_px")
    ax.set_title("Reprojection Error (px)")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Pixels")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize(fig, out_dir / "reprojection_error.png")

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(111)
    ax1.plot(frames, delta_t, color="tab:blue", label="delta_t")
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Translation Step")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(frames, delta_rot, color="tab:orange", label="delta_rot_deg")
    ax2.set_ylabel("Rotation Step (deg)")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Pose Step Size")
    _finalize(fig, out_dir / "pose_steps.png")

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(frames, vis_mean, color="tab:green")
    ax.set_title("Mean Track Visibility")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Visibility")
    ax.grid(True, alpha=0.3)
    _finalize(fig, out_dir / "visibility_mean.png")


def draw_debug_correspondences(
    frame_bgr: np.ndarray,
    u_obs: np.ndarray,
    u_proj: np.ndarray,
    inlier_mask: np.ndarray,
    max_points: int,
    text_lines: list[str],
) -> np.ndarray:
    """Draw observed/projected correspondences and residual arrows."""
    h, w = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()
    max_pts = max(1, int(max_points))

    n = int(u_obs.shape[0])
    if n > 0:
        if n <= max_pts:
            idx = np.arange(n, dtype=np.int32)
        else:
            idx = np.linspace(0, n - 1, max_pts).astype(np.int32)

        for i in idx:
            ox = int(np.clip(np.round(u_obs[i, 0]), 0, w - 1))
            oy = int(np.clip(np.round(u_obs[i, 1]), 0, h - 1))
            px = int(np.clip(np.round(u_proj[i, 0]), 0, w - 1))
            py = int(np.clip(np.round(u_proj[i, 1]), 0, h - 1))
            obs_color = (0, 255, 0) if bool(inlier_mask[i]) else (0, 0, 255)
            cv2.circle(canvas, (ox, oy), 2, obs_color, -1)
            cv2.circle(canvas, (px, py), 2, (255, 128, 0), -1)
            cv2.arrowedLine(canvas, (px, py), (ox, oy), (0, 255, 255), 1, tipLength=0.3)

    y = 24
    for line in text_lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track object meshes with CoTracker tracks and visibility."
    )

    parser.add_argument(
        "--cotracker_video_dir",
        type=str,
        default="../Estimate_Optical_Flow/output_cotracker/video_01",
        help="Directory containing _frames and per-object CoTracker outputs.",
    )
    parser.add_argument(
        "--mesh_source",
        type=str,
        choices=["generate", "align"],
        default="align",
        help="Mesh source: Generate_Object_Mesh output or Align_Meshes output.",
    )
    parser.add_argument(
        "--mesh_video_dir",
        type=str,
        default=None,
        help=(
            "Directory containing source meshes for this video. "
            "Default depends on --mesh_source and <video_name>."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_cotracker",
        help="Root directory for tracking outputs.",
    )
    parser.add_argument(
        "--output_coord",
        type=str,
        choices=["opencv", "pytorch3d"],
        default="opencv",
        help="Coordinate system used when saving mesh .ply files.",
    )

    parser.add_argument(
        "--overlay_fps",
        type=float,
        default=6.0,
        help="FPS for output overlay video.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string: cpu, cuda, cuda:0, cuda:1, etc.",
    )
    parser.add_argument("--bin_size", type=int, default=0, help="0 => naive rasterization")

    parser.add_argument("--ransac_reproj_px", type=float, default=6.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=30)
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=0.5,
        help="Drop tracks with visibility below this threshold.",
    )
    parser.add_argument(
        "--pose_smooth_alpha",
        type=float,
        default=0.9,
        help="EMA smoothing factor for pose. 0 keeps previous pose, 1 keeps raw PnP pose.",
    )
    parser.add_argument(
        "--debug_max_points",
        type=int,
        default=800,
        help="Max correspondences drawn per frame in debug overlay.",
    )
    parser.add_argument(
        "--debug_failure_p90_mult",
        type=float,
        default=2.0,
        help="Failure snapshot threshold multiplier: reproj_p90 > multiplier * ransac_reproj_px.",
    )

    parser.add_argument("--overlay_max_verts", type=int, default=20000)
    parser.add_argument("--overlay_point_radius", type=int, default=1)
    parser.add_argument("--min_seed_points", type=int, default=50)

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def track_single_object(
    object_slug: str,
    frames_dir: Path,
    tracks_path: Path,
    visibility_path: Path,
    seed_points_path: Path,
    mesh_path: Path,
    out_dir: Path,
    k: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Track one object mesh over a sequence using CoTracker tracks."""
    ensure_dir(out_dir)
    meshes_dir = out_dir / "meshes"
    overlays_dir = out_dir / "overlays"
    debug_dir = out_dir / "debug"
    failure_dir = debug_dir / "failures"
    ensure_dir(meshes_dir)
    ensure_dir(overlays_dir)
    ensure_dir(debug_dir)
    ensure_dir(failure_dir)

    frame_paths = list_images(frames_dir)

    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frames_dir}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.npy not found: {tracks_path}")
    if not visibility_path.exists():
        raise FileNotFoundError(f"visibility.npy not found: {visibility_path}")
    if not seed_points_path.exists():
        raise FileNotFoundError(f"seed_points_frame0.npy not found: {seed_points_path}")

    if int(args.start_frame) != 0:
        raise ValueError("CoTracker tracking currently supports --start_frame=0 only.")

    tracks_all = np.load(str(tracks_path)).astype(np.float32)
    vis_raw = np.load(str(visibility_path))

    if tracks_all.ndim != 3 or tracks_all.shape[2] != 2:
        raise ValueError(f"Bad tracks shape in {tracks_path}: {tracks_all.shape}")

    if vis_raw.ndim == 3 and vis_raw.shape[-1] == 1:
        vis_raw = vis_raw[..., 0]
    if vis_raw.ndim != 2:
        raise ValueError(f"Bad visibility shape in {visibility_path}: {vis_raw.shape}")

    if vis_raw.shape[0] != tracks_all.shape[0] or vis_raw.shape[1] != tracks_all.shape[1]:
        raise ValueError(
            "tracks/visibility shape mismatch: "
            f"tracks={tracks_all.shape}, visibility={vis_raw.shape}"
        )

    max_frames = min(len(frame_paths), tracks_all.shape[0])
    start = int(args.start_frame)
    end = max_frames - 1 if args.end_frame < 0 else min(int(args.end_frame), max_frames - 1)
    if start < 0 or start > end:
        raise ValueError(f"Invalid range start={start}, end={end}, max_frames={max_frames}")

    frame0 = cv2.imread(str(frame_paths[start]))
    if frame0 is None:
        raise FileNotFoundError(f"Failed to read: {frame_paths[start]}")
    h, w = frame0.shape[:2]

    device = resolve_device(args.device)

    mesh_template = load_mesh(mesh_path)
    verts_in = np.array(mesh_template.vertices, dtype=np.float32)
    faces = np.array(mesh_template.faces, dtype=np.int64)
    if args.mesh_source == "generate":
        v_obj = y_up_to_z_up(verts_in)
    else:
        # Align_Meshes outputs are OpenCV camera coordinates -> convert to PyTorch3D.
        v_obj = verts_in.copy()
        v_obj[:, 0] *= -1.0
        v_obj[:, 1] *= -1.0

    rasterizer = make_rasterizer(device=device, k=k, width=w, height=h, bin_size=int(args.bin_size))

    seed_px_all = np.load(str(seed_points_path)).astype(np.float32)
    if seed_px_all.ndim != 2 or seed_px_all.shape[1] != 2:
        raise ValueError(f"Bad seed points shape: {seed_px_all.shape}")
    if len(seed_px_all) != tracks_all.shape[1]:
        raise ValueError(
            "seed_points and tracks count mismatch: "
            f"seed_points={len(seed_px_all)}, tracks_N={tracks_all.shape[1]}"
        )

    pix0, bary0, _ = rasterize_gbuffer(rasterizer, v_obj, faces, device)
    x_obj_all, valid_seed = pixel_to_points(seed_px_all, pix0, bary0, v_obj, faces)

    track_ids_all = np.arange(len(seed_px_all), dtype=np.int32)
    x_obj = x_obj_all[valid_seed]
    track_ids = track_ids_all[valid_seed]

    if len(x_obj) < int(args.min_seed_points):
        raise RuntimeError(
            f"Too few valid seeded tracks after mesh mapping: {len(x_obj)} "
            f"(min required: {int(args.min_seed_points)})."
        )

    vis_conf = vis_raw.astype(np.float32)
    u = tracks_all[start, track_ids].astype(np.float32)
    # Track validity mask used by Improvement 1.
    alive = vis_conf[start, track_ids] >= float(args.visibility_threshold)
    alive &= np.isfinite(u[:, 0]) & np.isfinite(u[:, 1])

    in_bounds_start = (
        (u[:, 0] >= 0.0)
        & (u[:, 0] <= (w - 1))
        & (u[:, 1] >= 0.0)
        & (u[:, 1] <= (h - 1))
    )
    alive &= in_bounds_start

    # Pose at frame 0 is identity (object coordinates are frame-0 posed mesh)
    r_prev = np.eye(3, dtype=np.float32)
    t_prev = np.zeros(3, dtype=np.float32)

    video_path = out_dir / "overlay.mp4"
    ffmpeg_writer = start_ffmpeg_writer(video_path, float(args.overlay_fps), (h, w))
    debug_video_path = debug_dir / "correspondences_debug.mp4"
    debug_writer = start_ffmpeg_writer(debug_video_path, float(args.overlay_fps), (h, w))

    def write_video_frame(frame_bgr: np.ndarray) -> None:
        if ffmpeg_writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        ffmpeg_writer.stdin.write(frame_bgr.tobytes())

    def write_debug_frame(frame_bgr: np.ndarray) -> None:
        if debug_writer.stdin is None:
            raise RuntimeError("debug ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        debug_writer.stdin.write(frame_bgr.tobytes())

    def save_mesh(frame_idx: int, r_mat: np.ndarray, t_vec: np.ndarray) -> None:
        v_cam = apply_pose(v_obj, r_mat, t_vec)
        if args.output_coord == "opencv":
            v_save = v_cam.copy()
            v_save[:, 0] *= -1.0
            v_save[:, 1] *= -1.0
        else:
            v_save = v_cam
        mesh = mesh_template.copy()
        mesh.vertices = v_save.astype(np.float32)
        mesh.export(str(meshes_dir / f"frame_{frame_idx:04d}.ply"))

    r_list: List[np.ndarray] = [r_prev.copy()]
    t_list: List[np.ndarray] = [t_prev.copy()]
    metrics_rows: list[dict[str, float | int]] = []

    # Keep raw pose stream for diagnostics (separate from smoothed output stream).
    r_raw_prev = r_prev.copy()
    t_raw_prev = t_prev.copy()

    save_mesh(start, r_prev, t_prev)
    overlay0 = draw_overlay(
        frame0,
        apply_pose(v_obj, r_prev, t_prev),
        k,
        int(args.overlay_max_verts),
        int(args.overlay_point_radius),
    )
    cv2.imwrite(str(overlays_dir / f"overlay_{start:04d}.png"), overlay0)
    write_video_frame(overlay0)

    debug0 = compute_correspondence_debug(
        x_obj=x_obj,
        u=u,
        alive=alive,
        track_ids=track_ids,
        vis_conf=vis_conf,
        frame_idx=start,
        r_mat=r_prev,
        t_vec=t_prev,
        k=k,
        reproj_px=float(args.ransac_reproj_px),
    )
    debug0_lines = [
        f"frame={start:04d} (init)",
        f"alive={debug0['num_alive']} reproj_inliers={debug0['num_reproj_inliers']}",
        f"reproj_mean={float(debug0['reproj_mean']):.2f}px p90={float(debug0['reproj_p90']):.2f}px",
        f"visibility_mean={float(debug0['visibility_mean']):.3f}",
    ]
    debug_frame0 = draw_debug_correspondences(
        frame_bgr=frame0,
        u_obs=debug0["u_alive"],  # type: ignore[arg-type]
        u_proj=debug0["u_proj"],  # type: ignore[arg-type]
        inlier_mask=debug0["inlier_mask"],  # type: ignore[arg-type]
        max_points=int(args.debug_max_points),
        text_lines=debug0_lines,
    )
    write_debug_frame(debug_frame0)
    metrics_rows.append(
        {
            "frame_idx": int(start),
            "pnp_ok": 1,
            "num_alive": int(debug0["num_alive"]),
            "ransac_inliers": int(debug0["num_reproj_inliers"]),
            "reproj_inliers": int(debug0["num_reproj_inliers"]),
            "reproj_inlier_ratio": float(debug0["inlier_ratio"]),
            "reproj_mean": float(debug0["reproj_mean"]),
            "reproj_median": float(debug0["reproj_median"]),
            "reproj_p90": float(debug0["reproj_p90"]),
            "visibility_mean": float(debug0["visibility_mean"]),
            "delta_t": 0.0,
            "delta_rot_deg": 0.0,
            "delta_t_raw": 0.0,
            "delta_rot_raw_deg": 0.0,
        }
    )

    for frame_idx in range(start + 1, end + 1):
        # ---------------------------------------------------------------------
        # Improvement 1: minimal validity filtering for correspondence propagation
        #   - Kill tracks with low visibility / NaN / Inf at next frame
        #   - Kill tracks that go out of bounds after coordinate update
        # ---------------------------------------------------------------------
        idx_alive = np.where(alive)[0]
        if len(idx_alive) > 0:
            track_cols = track_ids[idx_alive]
            uv_next = tracks_all[frame_idx, track_cols].astype(np.float32)
            vis_next = vis_conf[frame_idx, track_cols] >= float(args.visibility_threshold)
            finite_next = np.isfinite(uv_next[:, 0]) & np.isfinite(uv_next[:, 1])
            keep = vis_next & finite_next

            alive[idx_alive[~keep]] = False
            idx_keep = idx_alive[keep]
            if len(idx_keep) > 0:
                u[idx_keep] = uv_next[keep]

        in_bounds_after = (
            (u[:, 0] >= 0.0)
            & (u[:, 0] <= (w - 1))
            & (u[:, 1] >= 0.0)
            & (u[:, 1] <= (h - 1))
        )
        alive &= in_bounds_after

        idx_alive = np.where(alive)[0]
        ok_pose, r_curr_raw, t_curr_raw, inliers = estimate_pose_pnp_ransac(
            x_obj_p3d=x_obj[idx_alive],
            u_px=u[idx_alive],
            k=k,
            reproj_px=float(args.ransac_reproj_px),
            iters=int(args.ransac_iters),
            min_inliers=int(args.min_inliers),
        )
        if not ok_pose:
            r_curr_raw, t_curr_raw = r_prev.copy(), t_prev.copy()

        prev_r_smooth = r_prev.copy()
        prev_t_smooth = t_prev.copy()
        r_curr, t_curr = smooth_pose_ema_slerp(
            r_prev=prev_r_smooth,
            t_prev=prev_t_smooth,
            r_curr=r_curr_raw,
            t_curr=t_curr_raw,
            alpha=float(args.pose_smooth_alpha),
        )

        delta_t = float(np.linalg.norm(t_curr - prev_t_smooth))
        delta_rot = rotation_delta_deg(prev_r_smooth, r_curr)
        delta_t_raw = float(np.linalg.norm(t_curr_raw - t_raw_prev))
        delta_rot_raw = rotation_delta_deg(r_raw_prev, r_curr_raw)

        r_raw_prev, t_raw_prev = r_curr_raw.copy(), t_curr_raw.copy()
        r_prev, t_prev = r_curr.copy(), t_curr.copy()
        r_list.append(r_prev.copy())
        t_list.append(t_prev.copy())

        save_mesh(frame_idx, r_prev, t_prev)

        frame = cv2.imread(str(frame_paths[frame_idx]))
        if frame is None:
            raise FileNotFoundError(f"Failed to read: {frame_paths[frame_idx]}")

        overlay = draw_overlay(
            frame,
            apply_pose(v_obj, r_prev, t_prev),
            k,
            int(args.overlay_max_verts),
            int(args.overlay_point_radius),
        )
        cv2.imwrite(str(overlays_dir / f"overlay_{frame_idx:04d}.png"), overlay)
        write_video_frame(overlay)

        dbg = compute_correspondence_debug(
            x_obj=x_obj,
            u=u,
            alive=alive,
            track_ids=track_ids,
            vis_conf=vis_conf,
            frame_idx=frame_idx,
            r_mat=r_prev,
            t_vec=t_prev,
            k=k,
            reproj_px=float(args.ransac_reproj_px),
        )
        debug_lines = [
            f"frame={frame_idx:04d} pnp_ok={int(ok_pose)}",
            f"alive={dbg['num_alive']} ransac_inliers={int(inliers)} reproj_inliers={dbg['num_reproj_inliers']}",
            f"reproj_mean={float(dbg['reproj_mean']):.2f}px p90={float(dbg['reproj_p90']):.2f}px vis_mean={float(dbg['visibility_mean']):.3f}",
            f"delta_t={delta_t:.5f} delta_rot={delta_rot:.3f}deg alpha={float(args.pose_smooth_alpha):.3f}",
        ]
        debug_canvas = draw_debug_correspondences(
            frame_bgr=frame,
            u_obs=dbg["u_alive"],  # type: ignore[arg-type]
            u_proj=dbg["u_proj"],  # type: ignore[arg-type]
            inlier_mask=dbg["inlier_mask"],  # type: ignore[arg-type]
            max_points=int(args.debug_max_points),
            text_lines=debug_lines,
        )
        write_debug_frame(debug_canvas)

        failure_by_inliers = int(inliers) < int(args.min_inliers)
        failure_by_reproj = (
            np.isfinite(float(dbg["reproj_p90"]))
            and float(dbg["reproj_p90"]) > float(args.debug_failure_p90_mult) * float(args.ransac_reproj_px)
        )
        if failure_by_inliers or failure_by_reproj:
            cv2.imwrite(str(failure_dir / f"frame_{frame_idx:04d}.png"), debug_canvas)

        metrics_rows.append(
            {
                "frame_idx": int(frame_idx),
                "pnp_ok": int(ok_pose),
                "num_alive": int(dbg["num_alive"]),
                "ransac_inliers": int(inliers),
                "reproj_inliers": int(dbg["num_reproj_inliers"]),
                "reproj_inlier_ratio": float(dbg["inlier_ratio"]),
                "reproj_mean": float(dbg["reproj_mean"]),
                "reproj_median": float(dbg["reproj_median"]),
                "reproj_p90": float(dbg["reproj_p90"]),
                "visibility_mean": float(dbg["visibility_mean"]),
                "delta_t": float(delta_t),
                "delta_rot_deg": float(delta_rot),
                "delta_t_raw": float(delta_t_raw),
                "delta_rot_raw_deg": float(delta_rot_raw),
            }
        )

        if args.verbose:
            print(
                f"[{object_slug}] frame={frame_idx:04d} "
                f"ok_pose={ok_pose} inliers={inliers} alive={int(alive.sum())} "
                f"reproj_p90={float(dbg['reproj_p90']):.2f}px"
            )

    close_ffmpeg(ffmpeg_writer)
    close_ffmpeg(debug_writer)
    save_debug_metrics_csv(debug_dir / "metrics.csv", metrics_rows)
    save_debug_plots(debug_dir, metrics_rows, reproj_px=float(args.ransac_reproj_px))
    save_pose_outputs(out_dir, r_list, t_list)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    cotracker_video_dir = resolve_path(args.cotracker_video_dir, script_dir)
    video_name = cotracker_video_dir.name
    if args.mesh_video_dir is None:
        if args.mesh_source == "generate":
            mesh_video_dir = (
                script_dir.parent / "Generate_Object_Mesh" / "output" / video_name
            ).resolve()
        else:
            mesh_video_dir = (
                script_dir.parent / "Align_Meshes" / "output" / video_name
            ).resolve()
    else:
        mesh_video_dir = resolve_path(args.mesh_video_dir, script_dir)
    output_root = resolve_path(args.output_root, script_dir)

    if not cotracker_video_dir.exists() or not cotracker_video_dir.is_dir():
        raise NotADirectoryError(f"CoTracker video dir not found: {cotracker_video_dir}")
    if not mesh_video_dir.exists() or not mesh_video_dir.is_dir():
        raise NotADirectoryError(f"Mesh video dir not found: {mesh_video_dir}")
    intrinsics_path = resolve_intrinsics_path(script_dir, video_name, mesh_video_dir)
    k = load_intrinsics_pixels_3x3(intrinsics_path)

    frames_dir = cotracker_video_dir / "_frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    object_slugs = list_object_slugs(args.mesh_source, mesh_video_dir)
    if not object_slugs:
        raise RuntimeError(f"No object dirs found in: {mesh_video_dir}")

    print(f"[INFO] Tracking video: {video_name}")
    print(f"[INFO] Objects: {object_slugs}")

    for object_slug in object_slugs:
        tracks_path = cotracker_video_dir / object_slug / "tracks.npy"
        visibility_path = cotracker_video_dir / object_slug / "visibility.npy"
        seed_points_path = cotracker_video_dir / object_slug / "seed_points_frame0.npy"
        if args.mesh_source == "generate":
            mesh_path = mesh_video_dir / object_slug / "mesh_posed.glb"
        else:
            mesh_path = mesh_video_dir / "meshes" / f"{object_slug}.ply"
        out_dir = output_root / video_name / object_slug

        if not tracks_path.exists():
            print(f"[WARN] Missing tracks.npy, skipping {object_slug}: {tracks_path}")
            continue
        if not visibility_path.exists():
            print(f"[WARN] Missing visibility.npy, skipping {object_slug}: {visibility_path}")
            continue
        if not seed_points_path.exists():
            print(f"[WARN] Missing seed points, skipping {object_slug}: {seed_points_path}")
            continue
        if not mesh_path.exists():
            print(f"[WARN] Missing mesh file, skipping {object_slug}: {mesh_path}")
            continue

        print(f"\n[OBJECT] {object_slug}")
        track_single_object(
            object_slug=object_slug,
            frames_dir=frames_dir,
            tracks_path=tracks_path,
            visibility_path=visibility_path,
            seed_points_path=seed_points_path,
            mesh_path=mesh_path,
            out_dir=out_dir,
            k=k,
            args=args,
        )
        print(f"[OK] Saved tracking outputs: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
