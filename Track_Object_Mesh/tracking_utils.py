"""Shared utilities for object mesh tracking scripts."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes

plt.switch_backend("Agg")

# TODO 1: Or should we use K given by depth-anything3, instead of full-frame sensor assumption?

# TODO 2: Implement temporal smoothing constraints in estimate_pose_pnp_ransac, e.g. by using previous
# pose as a prior and/or by enforcing velocity smoothness.
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
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def list_images(frames_dir: Path) -> List[Path]:
    """List image files sorted by frame index in filename."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [path for path in frames_dir.iterdir() if path.suffix.lower() in exts]
    return sorted(files, key=_extract_index)


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
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


def load_intrinsics_pixels_3x3(camera_intrinsics_path: Path) -> np.ndarray:
    """Load SAM3D intrinsics matrix from camera_intrinsics.json."""
    if not camera_intrinsics_path.exists():
        raise FileNotFoundError(f"camera_intrinsics.json not found: {camera_intrinsics_path}")

    with camera_intrinsics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if "intrinsics_pixels_3x3" not in payload:
        raise KeyError(
            "Missing 'intrinsics_pixels_3x3' in camera intrinsics file: "
            f"{camera_intrinsics_path}"
        )

    k = np.array(payload["intrinsics_pixels_3x3"], dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(
            "Expected intrinsics_pixels_3x3 to resolve to shape (3, 3), "
            f"got {k.shape} in {camera_intrinsics_path}"
        )
    return k.astype(np.float32)


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load a mesh and validate topology."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as trimesh.Trimesh: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh


def load_mesh_glb_y_up(mesh_path: Path) -> trimesh.Trimesh:
    """Backward-compatible alias for existing callers."""
    return load_mesh(mesh_path)


def y_up_to_z_up(verts_y: np.ndarray) -> np.ndarray:
    """Convert vertices from Y-up to Z-up."""
    return (verts_y.astype(np.float32) @ R_Y_UP_TO_Z_UP.T).astype(np.float32)


def z_up_to_y_up(verts_z: np.ndarray) -> np.ndarray:
    """Convert vertices from Z-up to Y-up."""
    return (verts_z.astype(np.float32) @ R_Z_UP_TO_Y_UP.T).astype(np.float32)


def make_rasterizer(
    device: torch.device,
    k: np.ndarray,
    width: int,
    height: int,
    bin_size: int,
) -> MeshRasterizer:
    """Build a PyTorch3D rasterizer using camera intrinsics K."""
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

    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),  # 0 => naive rasterization
        max_faces_per_bin=300000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def rasterize_gbuffer(
    rasterizer: MeshRasterizer,
    verts_cam_z: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize mesh and return face ids, barycentric coords, and silhouette."""
    verts = torch.from_numpy(verts_cam_z).to(device=device, dtype=torch.float32)
    tri_faces = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    mesh = Meshes(verts=[verts], faces=[tri_faces])
    fragments = rasterizer(mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0].detach().cpu().numpy()
    bary = fragments.bary_coords[0, ..., 0, :].detach().cpu().numpy()
    sil = (pix_to_face >= 0).astype(np.uint8)
    return pix_to_face, bary.astype(np.float32), sil


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary dilate by px pixels."""
    if px <= 0:
        return mask
    k = 2 * px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)


def pixel_to_points(
    pixels_xy: np.ndarray,
    pix_to_face: np.ndarray,
    bary: np.ndarray,
    verts_obj: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map image pixels to 3D object points using rasterized barycentrics."""
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


def apply_pose(verts: np.ndarray, r_mat: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
    """Apply rigid pose: X_cam = R * X_obj + t."""
    return (r_mat @ verts.T).T + t_vec.reshape(1, 3)


def project_points_via_f(pts_cam_p3d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project by converting to OpenCV coords with F, then pinhole project."""
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
) -> Tuple[np.ndarray, np.ndarray]:
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


def smooth_pose_sequence_post(
    r_seq: List[np.ndarray],
    t_seq: List[np.ndarray],
    alpha: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Post-process a full pose trajectory with simple EMA+SLERP smoothing."""
    if len(r_seq) == 0:
        return [], []
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return [r.copy() for r in r_seq], [t.copy() for t in t_seq]

    r_out = [r_seq[0].copy()]
    t_out = [t_seq[0].copy()]
    for idx in range(1, len(r_seq)):
        r_sm, t_sm = smooth_pose_ema_slerp(
            r_prev=r_out[-1],
            t_prev=t_out[-1],
            r_curr=r_seq[idx],
            t_curr=t_seq[idx],
            alpha=a,
        )
        r_out.append(r_sm.copy())
        t_out.append(t_sm.copy())
    return r_out, t_out


def list_object_slugs(mesh_source: str, mesh_video_dir: Path) -> List[str]:
    """List object slugs from the selected mesh source."""
    if mesh_source == "generate":
        return discover_object_dirs(mesh_video_dir, None)
    meshes_dir = mesh_video_dir / "meshes"
    if not meshes_dir.exists() or not meshes_dir.is_dir():
        raise NotADirectoryError(f"Aligned meshes dir not found: {meshes_dir}")
    return sorted([p.stem for p in meshes_dir.glob("*.ply") if p.stem != "human"])


def resolve_intrinsics_path(
    script_dir: Path,
    video_name: str,
    mesh_video_dir: Path,
) -> Path:
    """Find camera_intrinsics.json for the current video."""
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
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    k: np.ndarray,
    reproj_px: float,
    visibility_values: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | float | int]:
    """Compute residual-based correspondence diagnostics."""
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
            "visibility_mean": np.nan if visibility_values is not None else 0.0,
        }

    x_alive = x_obj[idx_alive]
    u_alive = u[idx_alive]
    u_proj = project_points_via_f(apply_pose(x_alive, r_mat, t_vec), k).astype(np.float32)
    residual = u_alive - u_proj
    errors = np.linalg.norm(residual, axis=1).astype(np.float32)
    inlier_mask = errors <= float(reproj_px)
    num_reproj_inliers = int(inlier_mask.sum())
    inlier_ratio = float(num_reproj_inliers / max(1, num_alive))

    if visibility_values is None:
        visibility_mean = float(num_alive / max(1, int(len(alive))))
    else:
        vis_alive = visibility_values[idx_alive].astype(np.float32)
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


def save_debug_metrics_csv(
    path: Path,
    rows: List[dict[str, float | int]],
) -> None:
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
    rows: List[dict[str, float | int]],
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
    text_lines: List[str],
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


def render_overlay_sequence(
    frame_paths: List[Path],
    start: int,
    r_seq: List[np.ndarray],
    t_seq: List[np.ndarray],
    verts_obj: np.ndarray,
    k: np.ndarray,
    overlay_max_verts: int,
    overlay_point_radius: int,
    overlays_out_dir: Path,
    video_out_path: Path,
    fps: float,
) -> None:
    """Render pose sequence into overlay PNGs + MP4."""
    ensure_dir(overlays_out_dir)
    first = cv2.imread(str(frame_paths[start]))
    if first is None:
        raise FileNotFoundError(f"Failed to read: {frame_paths[start]}")
    h, w = first.shape[:2]

    writer = start_ffmpeg_writer(video_out_path, float(fps), (h, w))
    try:
        for local_idx, (r_mat, t_vec) in enumerate(zip(r_seq, t_seq)):
            frame_idx = start + local_idx
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                raise FileNotFoundError(f"Failed to read: {frame_paths[frame_idx]}")

            overlay_img = draw_overlay(
                frame,
                apply_pose(verts_obj, r_mat, t_vec),
                k,
                int(overlay_max_verts),
                int(overlay_point_radius),
            )
            cv2.imwrite(
                str(overlays_out_dir / f"overlay_{frame_idx:04d}.png"),
                overlay_img,
            )
            if writer.stdin is None:
                raise RuntimeError("overlay ffmpeg stdin is closed")
            overlay_img = np.ascontiguousarray(overlay_img.astype(np.uint8))
            writer.stdin.write(overlay_img.tobytes())
    finally:
        close_ffmpeg(writer)


def estimate_pose_pnp_ransac(
    x_obj_p3d: np.ndarray,
    u_px: np.ndarray,
    k: np.ndarray,
    reproj_px: float,
    iters: int,
    min_inliers: int,
) -> Tuple[bool, np.ndarray, np.ndarray, int]:
    """Estimate pose in P3D coords; internally uses OpenCV PnP with F conversion."""
    if len(x_obj_p3d) < 4:
        return False, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 0

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
    """Render projected mesh vertices as points on a frame."""
    h, w = frame_bgr.shape[:2]

    verts = verts_cam_p3d
    if len(verts) > max_verts:
        idx = np.linspace(0, len(verts) - 1, max_verts).astype(np.int64)
        verts = verts[idx]

    u = project_points_via_f(verts, k)
    x = u[:, 0].astype(np.int32)
    y = u[:, 1].astype(np.int32)
    in_view = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    x = x[in_view]
    y = y[in_view]

    out = frame_bgr.copy()
    for x_i, y_i in zip(x, y):
        cv2.circle(out, (int(x_i), int(y_i)), int(radius), (0, 165, 255), -1)
    return out


def save_pose_outputs(out_dir: Path, r_list: List[np.ndarray], t_list: List[np.ndarray]) -> None:
    """Save pose outputs as .npy and .json."""
    r = np.stack(r_list, axis=0).astype(np.float32)
    t = np.stack(t_list, axis=0).astype(np.float32)
    np.save(str(out_dir / "poses.npy"), {"R": r, "t": t})

    poses_json = [
        {"frame": i, "R": r_list[i].tolist(), "t": t_list[i].tolist()}
        for i in range(len(r_list))
    ]
    with open(out_dir / "poses.json", "w", encoding="utf-8") as f:
        json.dump(poses_json, f, indent=2)


def resolve_path(path_str: str, script_dir: Path) -> Path:
    """Resolve path against script_dir when relative."""
    path = Path(path_str)
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()


def sanitize_object_name(name: str) -> str:
    """Convert object label to directory-friendly slug."""
    return name.replace(" ", "_")


def discover_object_dirs(base_dir: Path, one_object: Optional[str]) -> List[str]:
    """Discover object subdirectories under a video directory."""
    if one_object is not None:
        return [sanitize_object_name(one_object)]

    names = []
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        names.append(child.name)
    return sorted(names)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string and validate cuda index when specified."""
    try:
        device = torch.device(device_str)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {device_str}") from exc

    if device.type != "cuda":
        return device

    if not torch.cuda.is_available():
        return torch.device("cpu")

    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device_str}, but only {torch.cuda.device_count()} CUDA device(s) available."
        )
    return device
