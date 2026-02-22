"""
Rigid 4D tracking (MVP + Improvement 1).

Key conventions (consistent everywhere):
  - Internal camera/object coordinates follow PyTorch3D-style convention:
      +X is left, +Y is up, +Z forward.
  - We "stick with F" everywhere:
      F = diag([-1, -1, 1])
    - For PnP: convert 3D points to OpenCV coords with F before solvePnP.
    - For projection: convert camera points to OpenCV coords with F, then project.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
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
    rasterize_gbuffer,
    resolve_device,
    resolve_path,
    save_pose_outputs,
    start_ffmpeg_writer,
    y_up_to_z_up,
)


def _extract_index(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 10**18


def list_flows(flow_dir: Path) -> List[Path]:
    files = [p for p in flow_dir.iterdir() if p.suffix.lower() == ".npy"]
    return sorted(files, key=_extract_index)


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


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def stratified_sample(
    mask: np.ndarray,
    num_samples: int,
    grid: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simple stratified sampling; no spacing constraints (MVP)."""
    h, w = mask.shape[:2]
    samples: List[Tuple[int, int]] = []

    gh = max(1, int(grid))
    gw = max(1, int(grid))
    step_y = h / gh
    step_x = w / gw

    for gy in range(gh):
        for gx in range(gw):
            if len(samples) >= num_samples:
                break
            y0 = int(gy * step_y)
            y1 = int(min(h, (gy + 1) * step_y))
            x0 = int(gx * step_x)
            x1 = int(min(w, (gx + 1) * step_x))

            cell = mask[y0:y1, x0:x1]
            cy, cx = np.where(cell > 0)
            if len(cx) == 0:
                continue
            idx = int(rng.integers(0, len(cx)))
            samples.append((int(cx[idx] + x0), int(cy[idx] + y0)))

    if len(samples) < num_samples:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            perm = rng.permutation(len(xs))
            for idx in perm:
                if len(samples) >= num_samples:
                    break
                samples.append((int(xs[idx]), int(ys[idx])))

    return np.array(samples, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rigid tracking + Improvement 1."
    )

    parser.add_argument(
        "--flow_video_dir",
        type=str,
        default="../Estimate_Optical_Flow/output_waft/video_01",
        help="Directory containing _frames and optical_flow.",
    )
    parser.add_argument(
        "--mesh_source",
        type=str,
        choices=["generate", "align"],
        default="align",
        help="Mesh source: Generate_Object_Mesh(SAM3D-Objects) output or Align_Meshes output.",
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
        default="./output_waft",
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

    parser.add_argument("--num_tracks", type=int, default=2000)
    parser.add_argument("--seed_grid", type=int, default=30)
    parser.add_argument("--erode_px", type=int, default=8)

    parser.add_argument("--ransac_reproj_px", type=float, default=6.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=30)

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
    flow_dir: Path,
    mesh_path: Path,
    out_dir: Path,
    k: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """Track one object mesh over a sequence using frame-to-frame optical flow."""
    ensure_dir(out_dir)
    meshes_dir = out_dir / "meshes"
    overlays_dir = out_dir / "overlays"
    ensure_dir(meshes_dir)
    ensure_dir(overlays_dir)

    frame_paths = list_images(frames_dir)
    flow_paths = list_flows(flow_dir)

    if not frame_paths:
        raise FileNotFoundError(f"No frames found: {frames_dir}")
    if not flow_paths:
        raise FileNotFoundError(f"No flows found: {flow_dir}")

    max_frames = min(len(frame_paths), len(flow_paths) + 1)
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

    rasterizer = make_rasterizer(
        device=device,
        k=k,
        width=w,
        height=h,
        bin_size=int(args.bin_size),
    )
    rng = np.random.default_rng(42)

    # Pose at frame 0 is identity (because object coords are frame0 posed mesh)
    r_prev = np.eye(3, dtype=np.float32)
    t_prev = np.zeros(3, dtype=np.float32)

    # Seed tracks from frame-0 rasterization
    pix0, bary0, sil0 = rasterize_gbuffer(rasterizer, v_obj, faces, device)
    sil0 = erode(sil0, int(args.erode_px))
    seed_px = stratified_sample(sil0, int(args.num_tracks), int(args.seed_grid), rng)
    x_seed, ok = pixel_to_points(seed_px, pix0, bary0, v_obj, faces)
    seed_px = seed_px[ok]
    x_seed = x_seed[ok]

    if len(x_seed) < int(args.min_seed_points):
        raise RuntimeError(
            f"Too few seeded tracks: {len(x_seed)}. "
            f"(min required: {int(args.min_seed_points)}). "
            "Check rasterization / intrinsics / pose alignment."
        )

    x_obj = x_seed.astype(np.float32)
    u = seed_px.astype(np.float32)

    # Track validity mask used by Improvement 1.
    alive = np.ones(len(u), dtype=bool)

    video_path = out_dir / "overlay.mp4"
    ffmpeg_writer = start_ffmpeg_writer(video_path, float(args.overlay_fps), (h, w))

    def write_video_frame(frame_bgr: np.ndarray) -> None:
        if ffmpeg_writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        ffmpeg_writer.stdin.write(frame_bgr.tobytes())

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
        flow = np.load(str(flow_paths[frame_idx - 1])).astype(np.float32)
        if flow.shape[:2] != (h, w) or flow.shape[2] != 2:
            raise ValueError(f"Bad flow shape {flow.shape} at {flow_paths[frame_idx - 1]}")

        # ---------------------------------------------------------------------
        # Improvement 1: minimal validity filtering for flow propagation
        #   - Kill tracks already out of bounds before sampling flow
        #   - Kill tracks with NaN/Inf flow
        #   - Kill tracks that go out of bounds after u = u + d
        # ---------------------------------------------------------------------
        in_bounds_before = (
            (u[:, 0] >= 0.0)
            & (u[:, 0] <= (w - 1))
            & (u[:, 1] >= 0.0)
            & (u[:, 1] <= (h - 1))
        )
        alive &= in_bounds_before

        idx_alive = np.where(alive)[0]
        if len(idx_alive) > 0:
            x_idx = np.round(u[idx_alive, 0]).astype(np.int32)
            y_idx = np.round(u[idx_alive, 1]).astype(np.int32)

            # indices are safe because idx_alive are in-bounds already;
            # clipping here is only defensive.
            x_idx = np.clip(x_idx, 0, w - 1)
            y_idx = np.clip(y_idx, 0, h - 1)

            d = flow[y_idx, x_idx]

            finite = np.isfinite(d[:, 0]) & np.isfinite(d[:, 1])
            alive[idx_alive[~finite]] = False

            idx_valid = idx_alive[finite]
            u[idx_valid] = u[idx_valid] + d[finite]

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

    flow_video_dir = resolve_path(args.flow_video_dir, script_dir)
    video_name = flow_video_dir.name
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

    if not flow_video_dir.exists() or not flow_video_dir.is_dir():
        raise NotADirectoryError(f"Flow video dir not found: {flow_video_dir}")
    if not mesh_video_dir.exists() or not mesh_video_dir.is_dir():
        raise NotADirectoryError(f"Mesh video dir not found: {mesh_video_dir}")
    intrinsics_path = resolve_intrinsics_path(script_dir, video_name, mesh_video_dir)
    k = load_intrinsics_pixels_3x3(intrinsics_path)

    frames_dir = flow_video_dir / "_frames"
    flow_dir = flow_video_dir / "optical_flow"
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not flow_dir.exists():
        raise FileNotFoundError(f"Optical flow dir not found: {flow_dir}")

    object_slugs = list_object_slugs(args.mesh_source, mesh_video_dir)
    if not object_slugs:
        raise RuntimeError(f"No object dirs found in: {mesh_video_dir}")

    print(f"[INFO] Tracking video: {video_name}")
    print(f"[INFO] Objects: {object_slugs}")

    for object_slug in object_slugs:
        if args.mesh_source == "generate":
            mesh_path = mesh_video_dir / object_slug / "mesh_posed.glb"
        else:
            mesh_path = mesh_video_dir / "meshes" / f"{object_slug}.ply"
        out_dir = output_root / video_name / object_slug

        if not mesh_path.exists():
            print(f"[WARN] Missing mesh file, skipping {object_slug}: {mesh_path}")
            continue

        print(f"\n[OBJECT] {object_slug}")
        track_single_object(
            object_slug=object_slug,
            frames_dir=frames_dir,
            flow_dir=flow_dir,
            mesh_path=mesh_path,
            out_dir=out_dir,
            k=k,
            args=args,
        )
        print(f"[OK] Saved tracking outputs: {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
