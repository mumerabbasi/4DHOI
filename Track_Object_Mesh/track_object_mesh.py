#!/usr/bin/env python3
"""
Bare-minimum rigid 4D tracking (no flow gating / no track pruning).

Key conventions (consistent everywhere):
  - Internal camera/object coordinates follow your PyTorch3D-style convention:
      +X is left, +Y is up, +Z forward.
  - Pixel projection uses the equivalent sign flip:
      u = (-X * fx)/Z + cx
      v = (-Y * fy)/Z + cy
    which is identical to converting to OpenCV camera coords via:
      X_cv = F * X_p3d, where F = diag([-1, -1, 1]),
    and then using standard OpenCV projection.

We will "stick with F" conceptually and use it explicitly in both places:
  - For PnP: convert object points to OpenCV coords with F before calling OpenCV.
  - For projection: convert camera points to OpenCV coords with F, then project.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes


SENSOR_WIDTH_MM = 36.0

R_Y_UP_TO_Z_UP = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ],
    dtype=np.float32,
)

R_Z_UP_TO_Y_UP = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ],
    dtype=np.float32,
)

# P3D camera coords (+X left, +Y up, +Z forward) -> OpenCV (+X right, +Y down, +Z forward)
F_P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def start_ffmpeg_writer(out_path: Path, fps: float, size_hw: Tuple[int, int]) -> subprocess.Popen:
    """Start system ffmpeg to write H.264 MP4 from raw BGR frames."""
    h, w = size_hw
    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def close_ffmpeg(writer: Optional[subprocess.Popen]) -> None:
    """Close ffmpeg writer; raise if ffmpeg fails."""
    if writer is None:
        return
    if writer.stdin is not None:
        writer.stdin.close()
    stderr = writer.stderr.read() if writer.stderr is not None else b""
    ret = writer.wait()
    if ret != 0:
        msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed with code {ret}. stderr:\n{msg}")


def _extract_index(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 10**18


def list_images(frames_dir: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(files, key=_extract_index)


def list_flows(flow_dir: Path) -> List[Path]:
    files = [p for p in flow_dir.iterdir() if p.suffix.lower() == ".npy"]
    return sorted(files, key=_extract_index)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_intrinsics(width: int, height: int, focal_length_mm: float) -> np.ndarray:
    """Construct K from focal length (mm), sensor width (mm), and image size."""
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    fx = focal_length_mm * width / SENSOR_WIDTH_MM
    sensor_h_mm = SENSOR_WIDTH_MM * (height / width)
    fy = focal_length_mm * height / sensor_h_mm

    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def load_mesh_glb_y_up(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load GLB as trimesh.Trimesh: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh


def y_up_to_z_up(v_y: np.ndarray) -> np.ndarray:
    return (v_y.astype(np.float32) @ R_Y_UP_TO_Z_UP.T).astype(np.float32)


def z_up_to_y_up(v_z: np.ndarray) -> np.ndarray:
    return (v_z.astype(np.float32) @ R_Z_UP_TO_Y_UP.T).astype(np.float32)


def make_rasterizer(
    device: torch.device,
    k: np.ndarray,
    width: int,
    height: int,
    bin_size: int,
) -> MeshRasterizer:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[height, width]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )

    rast_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),  # 0 => naive rasterization
        max_faces_per_bin=300000,
    )

    return MeshRasterizer(cameras=cameras, raster_settings=rast_settings)


def rasterize_gbuffer(
    rasterizer: MeshRasterizer,
    verts_cam_z: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = torch.from_numpy(verts_cam_z).to(device=device, dtype=torch.float32)
    f = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    meshes = Meshes(verts=[v], faces=[f])
    fragments = rasterizer(meshes)

    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    bary = fragments.bary_coords[0, ..., 0, :].detach().cpu().numpy()
    sil = (pix_to_face >= 0).astype(np.uint8)
    return pix_to_face, bary.astype(np.float32), sil


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


def stratified_sample(mask: np.ndarray, num_samples: int, grid: int, rng: np.random.Generator) -> np.ndarray:
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


def pixel_to_points(
    pixels_xy: np.ndarray,
    pix_to_face: np.ndarray,
    bary: np.ndarray,
    verts_obj: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = pix_to_face.shape[:2]
    x = np.clip(np.round(pixels_xy[:, 0]).astype(np.int32), 0, w - 1)
    y = np.clip(np.round(pixels_xy[:, 1]).astype(np.int32), 0, h - 1)

    f_id = pix_to_face[y, x]
    valid = f_id >= 0
    x_obj = np.zeros((len(pixels_xy), 3), dtype=np.float32)

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return x_obj, valid

    tri = faces[f_id[idx].astype(np.int64)]
    b = bary[y[idx], x[idx], :]

    v0 = verts_obj[tri[:, 0]]
    v1 = verts_obj[tri[:, 1]]
    v2 = verts_obj[tri[:, 2]]
    x_obj[idx] = (b[:, [0]] * v0 + b[:, [1]] * v1 + b[:, [2]] * v2).astype(np.float32)
    return x_obj, valid


def apply_pose(verts: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """X_cam = R * X_obj + t."""
    return (r @ verts.T).T + t.reshape(1, 3)


def project_points_via_f(pts_cam_p3d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Project using explicit conversion to OpenCV coords via F:
      X_cv = F * X_p3d
    then standard pinhole projection (no negation in the formula).
    """
    pts_cv = (F_P3D_TO_CV @ pts_cam_p3d.T).T.astype(np.float32)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])

    z = np.maximum(pts_cv[:, 2], 1e-4)
    u = np.empty((len(pts_cv), 2), dtype=np.float32)
    u[:, 0] = (pts_cv[:, 0] * fx) / z + cx
    u[:, 1] = (pts_cv[:, 1] * fy) / z + cy
    return u


def estimate_pose_pnp_ransac(
    x_obj_p3d: np.ndarray,
    u_px: np.ndarray,
    k: np.ndarray,
    reproj_px: float,
    iters: int,
    min_inliers: int,
) -> Tuple[bool, np.ndarray, np.ndarray, int]:
    """
    Estimate object pose (R,t) in P3D camera coords, using OpenCV PnP by
    converting 3D points to OpenCV coords with F.
    """
    if len(x_obj_p3d) < 4:
        return False, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0

    # Convert object points from P3D coords to OpenCV coords
    x_obj_cv = (F_P3D_TO_CV @ x_obj_p3d.T).T.astype(np.float32)
    obj_pts = x_obj_cv.reshape(-1, 1, 3)
    img_pts = u_px.reshape(-1, 1, 2).astype(np.float32)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=k.astype(np.float32),
        distCoeffs=None,
        iterationsCount=int(iters),
        reprojectionError=float(reproj_px),
        confidence=0.99,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < int(min_inliers):
        return False, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0

    in_idx = inliers.reshape(-1).astype(np.int32)

    # Refine
    rvec, tvec = cv2.solvePnPRefineLM(
        objectPoints=obj_pts[in_idx],
        imagePoints=img_pts[in_idx],
        cameraMatrix=k.astype(np.float32),
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
    )

    r_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec.reshape(3).astype(np.float32)

    # Convert back to P3D coords: R_p3d = F * R_cv * F, t_p3d = F * t_cv
    r_p3d = (F_P3D_TO_CV @ r_cv @ F_P3D_TO_CV).astype(np.float32)
    t_p3d = (F_P3D_TO_CV @ t_cv.reshape(3, 1)).reshape(3).astype(np.float32)

    return True, r_p3d, t_p3d, int(len(in_idx))


def draw_overlay(
    frame_bgr: np.ndarray,
    verts_cam_p3d: np.ndarray,
    k: np.ndarray,
    max_verts: int,
    radius: int,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    v = verts_cam_p3d
    if len(v) > max_verts:
        idx = np.linspace(0, len(v) - 1, max_verts).astype(np.int64)
        v = v[idx]

    u = project_points_via_f(v, k)
    x = u[:, 0].astype(np.int32)
    y = u[:, 1].astype(np.int32)

    in_view = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[in_view]
    y = y[in_view]

    out = frame_bgr.copy()
    for xi, yi in zip(x, y):
        cv2.circle(out, (int(xi), int(yi)), int(radius), (0, 165, 255), -1)
    return out


def save_pose_outputs(out_dir: Path, r_list: List[np.ndarray], t_list: List[np.ndarray]) -> None:
    r = np.stack(r_list, axis=0).astype(np.float32)
    t = np.stack(t_list, axis=0).astype(np.float32)
    np.save(str(out_dir / "poses.npy"), {"R": r, "t": t})

    poses_json = [{"frame": i, "R": r_list[i].tolist(), "t": t_list[i].tolist()}
                  for i in range(len(r_list))]
    with open(out_dir / "poses.json", "w") as f:
        json.dump(poses_json, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bare-minimum rigid tracking + ffmpeg (no flow gating). Uses F=diag([-1,-1,1]) everywhere."
    )

    parser.add_argument(
        "--frames_dir",
        type=str,
        default="../Estimate_Optical_Flow/videos/video_01/_frames",
        help="Directory containing video frames (images).",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        default="../Estimate_Optical_Flow/videos/video_01/optical_flow",
        help="Directory containing forward flow .npy files (sorted).",
    )
    parser.add_argument(
        "--posed_mesh_glb",
        type=str,
        default="../Generate_Object_Mesh/objects/video_01/iron/mesh_posed.glb",
        help="Path to posed mesh GLB for frame 0 (Y-up).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./track_out/video_01",
        help="Output directory.",
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

    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--bin_size", type=int, default=0, help="0 => naive rasterization")

    parser.add_argument("--num_tracks", type=int, default=2000)
    parser.add_argument("--seed_grid", type=int, default=30)
    parser.add_argument("--erode_px", type=int, default=8)

    parser.add_argument("--ransac_reproj_px", type=float, default=6.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=30)

    parser.add_argument("--overlay_max_verts", type=int, default=20000)
    parser.add_argument("--overlay_point_radius", type=int, default=1)

    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames_dir = Path(args.frames_dir)
    flow_dir = Path(args.flow_dir)
    mesh_path = Path(args.posed_mesh_glb)
    out_dir = Path(args.out_dir)

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

    k = build_intrinsics(w, h, float(args.focal_length_mm))

    device = torch.device(
        "cuda" if args.device.lower() == "cuda" and torch.cuda.is_available() else "cpu"
    )

    mesh0_y = load_mesh_glb_y_up(mesh_path)
    v0_y = np.array(mesh0_y.vertices, dtype=np.float32)
    faces = np.array(mesh0_y.faces, dtype=np.int64)
    v_obj = y_up_to_z_up(v0_y)  # object coords in Z-up, P3D camera convention

    rasterizer = make_rasterizer(device=device, k=k, width=w, height=h, bin_size=int(args.bin_size))
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

    if len(x_seed) < 50:
        raise RuntimeError(
            f"Too few seeded tracks: {len(x_seed)}. "
            "Check rasterization / K / focal length / pose alignment."
        )

    x_obj = x_seed.astype(np.float32)
    u = seed_px.astype(np.float32)

    # ffmpeg writer
    video_path = out_dir / "overlay.mp4"
    ffmpeg_writer = start_ffmpeg_writer(video_path, float(args.overlay_fps), (h, w))

    def write_video_frame(frame_bgr: np.ndarray) -> None:
        if ffmpeg_writer.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame_bgr = np.ascontiguousarray(frame_bgr.astype(np.uint8))
        ffmpeg_writer.stdin.write(frame_bgr.tobytes())

    def save_mesh(frame_idx: int, r: np.ndarray, t: np.ndarray) -> None:
        v_cam = apply_pose(v_obj, r, t)     # Z-up (P3D convention)
        v_y = z_up_to_y_up(v_cam)           # back to Y-up for GLB export
        m = mesh0_y.copy()
        m.vertices = v_y.astype(np.float32)
        m.export(str(meshes_dir / f"frame_{frame_idx:04d}.glb"))

    r_list: List[np.ndarray] = [r_prev.copy()]
    t_list: List[np.ndarray] = [t_prev.copy()]

    # Frame 0 outputs
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

    # Main loop
    for t in range(start + 1, end + 1):
        flow = np.load(str(flow_paths[t - 1])).astype(np.float32)
        if flow.shape[:2] != (h, w) or flow.shape[2] != 2:
            raise ValueError(f"Bad flow shape {flow.shape} at {flow_paths[t - 1]}")

        # Raw flow propagation (no gating). Indices clipped to avoid crash.
        x_idx = np.clip(np.round(u[:, 0]).astype(np.int32), 0, w - 1)
        y_idx = np.clip(np.round(u[:, 1]).astype(np.int32), 0, h - 1)
        d = flow[y_idx, x_idx]
        u = u + d

        ok_pose, r_curr, t_curr, inliers = estimate_pose_pnp_ransac(
            x_obj_p3d=x_obj,
            u_px=u,
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

        save_mesh(t, r_prev, t_prev)

        frame = cv2.imread(str(frame_paths[t]))
        if frame is None:
            raise FileNotFoundError(f"Failed to read: {frame_paths[t]}")

        overlay = draw_overlay(
            frame,
            apply_pose(v_obj, r_prev, t_prev),
            k,
            int(args.overlay_max_verts),
            int(args.overlay_point_radius),
        )
        cv2.imwrite(str(overlays_dir / f"overlay_{t:04d}.png"), overlay)
        write_video_frame(overlay)

        if args.verbose:
            mean_flow = float(np.linalg.norm(d, axis=1).mean())
            print(
                f"[frame {t:04d}] ok_pose={ok_pose} inliers={inliers} "
                f"mean|flow|={mean_flow:.3f} "
                f"t=({t_prev[0]:+.3f},{t_prev[1]:+.3f},{t_prev[2]:+.3f})"
            )

    close_ffmpeg(ffmpeg_writer)
    save_pose_outputs(out_dir, r_list, t_list)

    print(f"Done. Outputs in: {out_dir}")
    print(f"Overlay video: {out_dir / 'overlay.mp4'}")
    print(f"Meshes: {out_dir / 'meshes'}")
    print(f"Overlays: {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
