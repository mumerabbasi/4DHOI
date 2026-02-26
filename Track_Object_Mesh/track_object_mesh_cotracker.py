"""Rigid 4D tracking (MVP + Improvement 1) using CoTracker tracks + visibility."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from tracking_utils import (
    apply_pose,
    close_ffmpeg,
    compute_correspondence_debug,
    draw_overlay,
    ensure_dir,
    estimate_pose_pnp_ransac,
    list_images,
    list_object_slugs,
    load_intrinsics_pixels_3x3,
    load_mesh,
    make_rasterizer,
    pixel_to_points,
    rasterize_gbuffer,
    render_overlay_sequence,
    resolve_device,
    resolve_intrinsics_path,
    resolve_path,
    rotation_delta_deg,
    save_debug_metrics_csv,
    save_pose_outputs,
    smooth_pose_sequence_post,
    start_ffmpeg_writer,
    y_up_to_z_up,
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
        "--mesh_source",
        type=str,
        choices=["generate", "align"],
        default="align",
        help=(
            "Mesh source: Generate_Object_Mesh output or Align_Meshes output."
        ),
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
    parser.add_argument("--bin_size", type=int, default=0,
                        help="0 => naive rasterization")

    parser.add_argument("--ransac_reproj_px", type=float, default=6.0)
    parser.add_argument("--ransac_iters", type=int, default=10000)
    parser.add_argument("--min_inliers", type=int, default=30)
    parser.add_argument(
        "--visibility_threshold",
        type=float,
        default=0.99,
        help="Drop tracks with visibility below this threshold.",
    )
    parser.add_argument(
        "--post_smooth_sigma",
        type=float,
        default=0.5,
        help="Gaussian smoothing sigma in frames. 0 disables smoothing.",
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
    meshes_raw_dir = out_dir / "meshes_raw"
    meshes_dir = out_dir / "meshes"
    overlays_dir = out_dir / "overlays"
    overlays_smoothed_dir = out_dir / "overlays_smoothed"
    debug_dir = out_dir / "debug"
    ensure_dir(meshes_raw_dir)
    ensure_dir(meshes_dir)
    ensure_dir(overlays_dir)
    ensure_dir(overlays_smoothed_dir)
    ensure_dir(debug_dir)

    frame_paths = list_images(frames_dir)

    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frames_dir}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"tracks.npy not found: {tracks_path}")
    if not visibility_path.exists():
        raise FileNotFoundError(f"visibility.npy not found: {visibility_path}")
    if not seed_points_path.exists():
        raise FileNotFoundError(
            f"seed_points_frame0.npy not found: {seed_points_path}")

    if int(args.start_frame) != 0:
        raise ValueError(
            "CoTracker tracking currently supports --start_frame=0 only.")

    tracks_all = np.load(str(tracks_path)).astype(np.float32)
    vis_raw = np.load(str(visibility_path))

    if tracks_all.ndim != 3 or tracks_all.shape[2] != 2:
        raise ValueError(
            f"Bad tracks shape in {tracks_path}: {tracks_all.shape}")

    if vis_raw.ndim == 3 and vis_raw.shape[-1] == 1:
        vis_raw = vis_raw[..., 0]
    if vis_raw.ndim != 2:
        raise ValueError(
            f"Bad visibility shape in {visibility_path}: {vis_raw.shape}")

    if (
        vis_raw.shape[0] != tracks_all.shape[0]
        or vis_raw.shape[1] != tracks_all.shape[1]
    ):
        raise ValueError(
            "tracks/visibility shape mismatch: "
            f"tracks={tracks_all.shape}, visibility={vis_raw.shape}"
        )

    max_frames = min(len(frame_paths), tracks_all.shape[0])
    start = int(args.start_frame)
    end = max_frames - \
        1 if args.end_frame < 0 else min(int(args.end_frame), max_frames - 1)
    if start < 0 or start > end:
        raise ValueError(
            f"Invalid range start={start}, end={end}, max_frames={max_frames}")

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
        # Align_Meshes outputs are OpenCV camera coordinates.
        # Convert them to PyTorch3D coordinates.
        v_obj = verts_in.copy()
        v_obj[:, 0] *= -1.0
        v_obj[:, 1] *= -1.0

    rasterizer = make_rasterizer(
        device=device, k=k, width=w, height=h, bin_size=int(args.bin_size))

    seed_px_all = np.load(str(seed_points_path)).astype(np.float32)
    if seed_px_all.ndim != 2 or seed_px_all.shape[1] != 2:
        raise ValueError(f"Bad seed points shape: {seed_px_all.shape}")
    if len(seed_px_all) != tracks_all.shape[1]:
        raise ValueError(
            "seed_points and tracks count mismatch: "
            f"seed_points={len(seed_px_all)}, tracks_N={tracks_all.shape[1]}"
        )

    pix0, bary0, _ = rasterize_gbuffer(rasterizer, v_obj, faces, device)
    x_obj_all, valid_seed = pixel_to_points(
        seed_px_all, pix0, bary0, v_obj, faces)

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
    ffmpeg_writer = start_ffmpeg_writer(
        video_path, float(args.overlay_fps), (h, w))

    def write_video_frame(frame_bgr: np.ndarray) -> None:
        if ffmpeg_writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        ffmpeg_writer.stdin.write(frame_bgr.tobytes())

    def save_mesh(
        frame_idx: int,
        r_mat: np.ndarray,
        t_vec: np.ndarray,
        meshes_out_dir: Path,
    ) -> None:
        v_cam = apply_pose(v_obj, r_mat, t_vec)
        if args.output_coord == "opencv":
            v_save = v_cam.copy()
            v_save[:, 0] *= -1.0
            v_save[:, 1] *= -1.0
        else:
            v_save = v_cam
        mesh = mesh_template.copy()
        mesh.vertices = v_save.astype(np.float32)
        mesh.export(str(meshes_out_dir / f"frame_{frame_idx:04d}.ply"))

    r_list: List[np.ndarray] = [r_prev.copy()]
    t_list: List[np.ndarray] = [t_prev.copy()]
    metrics_rows: list[dict[str, float | int]] = []

    save_mesh(start, r_prev, t_prev, meshes_raw_dir)
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
        r_mat=r_prev,
        t_vec=t_prev,
        k=k,
        reproj_px=float(args.ransac_reproj_px),
        visibility_values=vis_conf[start, track_ids],
    )
    metrics_rows.append(
        {
            "frame_idx": int(start),
            "pnp_ok": 1,
            "num_alive": int(debug0["num_alive"]),
            "ransac_inliers": int(debug0["num_reproj_inliers"]),
            "reproj_mean": float(debug0["reproj_mean"]),
            "reproj_p90": float(debug0["reproj_p90"]),
            "visibility_mean": float(debug0["visibility_mean"]),
            "delta_t": 0.0,
            "delta_rot_deg": 0.0,
        }
    )

    for frame_idx in range(start + 1, end + 1):
        # ---------------------------------------------------------------------
        # Improvement 1:
        # minimal validity filtering for correspondence propagation
        #   - Kill tracks with low visibility / NaN / Inf at next frame
        #   - Kill tracks that go out of bounds after coordinate update
        # ---------------------------------------------------------------------
        idx_alive = np.where(alive)[0]
        if len(idx_alive) > 0:
            track_cols = track_ids[idx_alive]
            uv_next = tracks_all[frame_idx, track_cols].astype(np.float32)
            vis_next = vis_conf[frame_idx, track_cols] >= float(
                args.visibility_threshold)
            finite_next = np.isfinite(
                uv_next[:, 0]) & np.isfinite(uv_next[:, 1])
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

        delta_t = float(np.linalg.norm(t_curr - t_prev))
        delta_rot = rotation_delta_deg(r_prev, r_curr)

        r_prev, t_prev = r_curr.copy(), t_curr.copy()
        r_list.append(r_prev.copy())
        t_list.append(t_prev.copy())

        save_mesh(frame_idx, r_prev, t_prev, meshes_raw_dir)

        frame = cv2.imread(str(frame_paths[frame_idx]))
        if frame is None:
            raise FileNotFoundError(
                f"Failed to read: {frame_paths[frame_idx]}")

        overlay = draw_overlay(
            frame,
            apply_pose(v_obj, r_prev, t_prev),
            k,
            int(args.overlay_max_verts),
            int(args.overlay_point_radius),
        )
        cv2.imwrite(
            str(overlays_dir / f"overlay_{frame_idx:04d}.png"), overlay)
        write_video_frame(overlay)

        dbg = compute_correspondence_debug(
            x_obj=x_obj,
            u=u,
            alive=alive,
            r_mat=r_prev,
            t_vec=t_prev,
            k=k,
            reproj_px=float(args.ransac_reproj_px),
            visibility_values=vis_conf[frame_idx, track_ids],
        )

        metrics_rows.append(
            {
                "frame_idx": int(frame_idx),
                "pnp_ok": int(ok_pose),
                "num_alive": int(dbg["num_alive"]),
                "ransac_inliers": int(inliers),
                "reproj_mean": float(dbg["reproj_mean"]),
                "reproj_p90": float(dbg["reproj_p90"]),
                "visibility_mean": float(dbg["visibility_mean"]),
                "delta_t": float(delta_t),
                "delta_rot_deg": float(delta_rot),
            }
        )

        if args.verbose:
            print(
                f"[{object_slug}] frame={frame_idx:04d} "
                f"ok_pose={ok_pose} inliers={inliers} "
                f"alive={int(alive.sum())} "
                f"reproj_p90={float(dbg['reproj_p90']):.2f}px"
            )

    close_ffmpeg(ffmpeg_writer)
    save_debug_metrics_csv(debug_dir / "metrics.csv", metrics_rows)
    r_sm, t_sm = smooth_pose_sequence_post(
        r_seq=r_list,
        t_seq=t_list,
        sigma=float(args.post_smooth_sigma),
    )
    for local_idx, (r_mat, t_vec) in enumerate(zip(r_sm, t_sm)):
        save_mesh(start + local_idx, r_mat, t_vec, meshes_dir)

    render_overlay_sequence(
        frame_paths=frame_paths,
        start=start,
        r_seq=r_sm,
        t_seq=t_sm,
        verts_obj=v_obj,
        k=k,
        overlay_max_verts=int(args.overlay_max_verts),
        overlay_point_radius=int(args.overlay_point_radius),
        overlays_out_dir=overlays_smoothed_dir,
        video_out_path=out_dir / "overlay_smoothed.mp4",
        fps=float(args.overlay_fps),
    )

    save_pose_outputs(
        out_dir=out_dir,
        r_raw=r_list,
        t_raw=t_list,
        r_smoothed=r_sm,
        t_smoothed=t_sm,
    )


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    cotracker_video_dir = resolve_path(args.cotracker_video_dir, script_dir)
    video_name = cotracker_video_dir.name
    if args.mesh_video_dir is None:
        if args.mesh_source == "generate":
            mesh_video_dir = (
                script_dir.parent
                / "Generate_Object_Mesh"
                / "output"
                / video_name
            ).resolve()
        else:
            mesh_video_dir = (
                script_dir.parent / "Align_Meshes" / "output" / video_name
            ).resolve()
    else:
        mesh_video_dir = resolve_path(args.mesh_video_dir, script_dir)
    output_root = resolve_path(args.output_root, script_dir)

    if not cotracker_video_dir.exists() or not cotracker_video_dir.is_dir():
        raise NotADirectoryError(
            f"CoTracker video dir not found: {cotracker_video_dir}")
    if not mesh_video_dir.exists() or not mesh_video_dir.is_dir():
        raise NotADirectoryError(f"Mesh video dir not found: {mesh_video_dir}")
    intrinsics_path = resolve_intrinsics_path(
        script_dir, video_name, mesh_video_dir)
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
        seed_points_path = (
            cotracker_video_dir
            / object_slug
            / "seed_points_frame0.npy"
        )
        if args.mesh_source == "generate":
            mesh_path = mesh_video_dir / object_slug / "mesh_posed.glb"
        else:
            mesh_path = mesh_video_dir / "meshes" / f"{object_slug}.ply"
        out_dir = output_root / video_name / object_slug

        if not tracks_path.exists():
            print(
                f"[WARN] Missing tracks.npy, skipping "
                f"{object_slug}: {tracks_path}"
            )
            continue
        if not visibility_path.exists():
            print(
                f"[WARN] Missing visibility.npy, skipping "
                f"{object_slug}: {visibility_path}"
            )
            continue
        if not seed_points_path.exists():
            print(
                f"[WARN] Missing seed points, skipping "
                f"{object_slug}: {seed_points_path}"
            )
            continue
        if not mesh_path.exists():
            print(
                f"[WARN] Missing mesh file, skipping "
                f"{object_slug}: {mesh_path}"
            )
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
