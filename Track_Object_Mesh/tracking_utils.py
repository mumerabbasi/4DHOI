"""Shared utilities for object mesh tracking scripts."""

from __future__ import annotations

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


def load_mesh_glb_y_up(mesh_path: Path) -> trimesh.Trimesh:
    """Load a GLB mesh and validate topology."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load GLB as trimesh.Trimesh: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")
    return mesh


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
