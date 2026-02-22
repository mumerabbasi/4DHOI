"""Rigid 4D tracking (MVP + Improvement 1 + Improvement 2) using CoTracker tracks + visibility.

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
    |_ meshes/frame_XXXX.glb
    |_ overlays/overlay_XXXX.png
    |_ overlay.mp4
    |_ poses.npy
    |_ poses.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from tracking_utils import (
    apply_pose,
    build_intrinsics,
    close_ffmpeg,
    dilate,
    discover_object_dirs,
    draw_overlay,
    ensure_dir,
    estimate_pose_pnp_ransac,
    list_images,
    load_mesh_glb_y_up,
    make_rasterizer,
    pixel_to_points,
    rasterize_gbuffer,
    resolve_device,
    resolve_path,
    save_pose_outputs,
    start_ffmpeg_writer,
    y_up_to_z_up,
    z_up_to_y_up,
)


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
        "--mesh_video_dir",
        type=str,
        default=None,
        help=(
            "Directory containing per-object mesh_posed.glb files. "
            "Default: ../Generate_Object_Mesh/output/<video_name> "
            "derived from --cotracker_video_dir."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output_cotracker",
        help="Root directory for tracking outputs.",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Optional single object slug to track (e.g., iron).",
    )

    parser.add_argument(
        "--focal_length_mm",
        type=float,
        default=23.0,
        help="Focal length in mm (sensor width fixed to 36mm).",
    )
    parser.add_argument(
        "--overlay_fps",
        type=float,
        default=24.0,
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

    # Improvement 2
    parser.add_argument(
        "--silhouette_dilate_px",
        type=int,
        default=3,
        help="Dilate rendered silhouette by this many pixels before gating.",
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
    args: argparse.Namespace,
) -> None:
    """Track one object mesh over a sequence using CoTracker tracks."""
    ensure_dir(out_dir)
    meshes_dir = out_dir / "meshes"
    overlays_dir = out_dir / "overlays"
    ensure_dir(meshes_dir)
    ensure_dir(overlays_dir)

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

    k = build_intrinsics(w, h, float(args.focal_length_mm))

    device = resolve_device(args.device)

    mesh0_y = load_mesh_glb_y_up(mesh_path)
    v0_y = np.array(mesh0_y.vertices, dtype=np.float32)
    faces = np.array(mesh0_y.faces, dtype=np.int64)
    v_obj = y_up_to_z_up(v0_y)

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
    # Track validity mask (used by Improvement 1 and Improvement 2)
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

    def write_video_frame(frame_bgr: np.ndarray) -> None:
        if ffmpeg_writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        ffmpeg_writer.stdin.write(frame_bgr.tobytes())

    def save_mesh(frame_idx: int, r_mat: np.ndarray, t_vec: np.ndarray) -> None:
        v_cam = apply_pose(v_obj, r_mat, t_vec)
        v_y = z_up_to_y_up(v_cam)
        mesh = mesh0_y.copy()
        mesh.vertices = v_y.astype(np.float32)
        mesh.export(str(meshes_dir / f"frame_{frame_idx:04d}.glb"))

    r_list: List[np.ndarray] = [r_prev.copy()]
    t_list: List[np.ndarray] = [t_prev.copy()]

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
        ok_pose, r_curr, t_curr, inliers = estimate_pose_pnp_ransac(
            x_obj_p3d=x_obj[idx_alive],
            u_px=u[idx_alive],
            k=k,
            reproj_px=float(args.ransac_reproj_px),
            iters=int(args.ransac_iters),
            min_inliers=int(args.min_inliers),
        )
        if not ok_pose:
            r_curr, t_curr = r_prev.copy(), t_prev.copy()

        # ---------------------------------------------------------------------
        # Improvement 2: rendered-silhouette gating (after first pose estimate)
        #   - Render silhouette of mesh at predicted pose
        #   - Keep only tracks whose u lies inside (optionally dilated)
        #   - Re-run PnP on gated subset
        # ---------------------------------------------------------------------
        v_cam_pred = apply_pose(v_obj, r_curr, t_curr)
        _, _, sil_pred = rasterize_gbuffer(rasterizer, v_cam_pred, faces, device)
        sil_pred = dilate(sil_pred, int(args.silhouette_dilate_px))

        idx_alive = np.where(alive)[0]
        if len(idx_alive) > 0:
            xs = np.round(u[idx_alive, 0]).astype(np.int32)
            ys = np.round(u[idx_alive, 1]).astype(np.int32)
            xs = np.clip(xs, 0, w - 1)
            ys = np.clip(ys, 0, h - 1)
            keep = sil_pred[ys, xs] > 0
            alive[idx_alive[~keep]] = False

        idx_alive = np.where(alive)[0]
        ok_pose2, r_ref, t_ref, inliers2 = estimate_pose_pnp_ransac(
            x_obj_p3d=x_obj[idx_alive],
            u_px=u[idx_alive],
            k=k,
            reproj_px=float(args.ransac_reproj_px),
            iters=int(args.ransac_iters),
            min_inliers=int(args.min_inliers),
        )
        if ok_pose2:
            r_curr, t_curr = r_ref, t_ref
            ok_pose = True
            inliers = inliers2

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

        if args.verbose:
            print(
                f"[{object_slug}] frame={frame_idx:04d} "
                f"ok_pose={ok_pose} inliers={inliers} alive={int(alive.sum())}"
            )

    close_ffmpeg(ffmpeg_writer)
    save_pose_outputs(out_dir, r_list, t_list)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    cotracker_video_dir = resolve_path(args.cotracker_video_dir, script_dir)
    video_name = cotracker_video_dir.name
    if args.mesh_video_dir is None:
        mesh_video_dir = (
            script_dir.parent / "Generate_Object_Mesh" / "output" / video_name
        ).resolve()
    else:
        mesh_video_dir = resolve_path(args.mesh_video_dir, script_dir)
    output_root = resolve_path(args.output_root, script_dir)

    if not cotracker_video_dir.exists() or not cotracker_video_dir.is_dir():
        raise NotADirectoryError(f"CoTracker video dir not found: {cotracker_video_dir}")
    if not mesh_video_dir.exists() or not mesh_video_dir.is_dir():
        raise NotADirectoryError(f"Mesh video dir not found: {mesh_video_dir}")

    frames_dir = cotracker_video_dir / "_frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")

    object_slugs = discover_object_dirs(cotracker_video_dir, args.object)
    if not object_slugs:
        raise RuntimeError(f"No object dirs found in: {cotracker_video_dir}")

    print(f"[INFO] Tracking video: {video_name}")
    print(f"[INFO] Objects: {object_slugs}")

    for object_slug in object_slugs:
        tracks_path = cotracker_video_dir / object_slug / "tracks.npy"
        visibility_path = cotracker_video_dir / object_slug / "visibility.npy"
        seed_points_path = cotracker_video_dir / object_slug / "seed_points_frame0.npy"
        mesh_path = mesh_video_dir / object_slug / "mesh_posed.glb"
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
            print(f"[WARN] Missing mesh_posed.glb, skipping {object_slug}: {mesh_path}")
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
            args=args,
        )
        print(f"[OK] Saved tracking outputs: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
