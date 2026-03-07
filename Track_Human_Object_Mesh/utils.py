"""Local utilities for joint human-object mesh refinement."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def start_ffmpeg_writer(
    out_path: Path,
    fps: float,
    size_hw: tuple[int, int],
) -> subprocess.Popen:
    """Start ffmpeg to write an H.264 MP4 from raw BGR frames."""
    h, w = size_hw
    ffmpeg = (
        "/usr/bin/ffmpeg"
        if Path("/usr/bin/ffmpeg").exists()
        else "ffmpeg"
    )

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


def close_ffmpeg(writer: subprocess.Popen | None) -> None:
    """Close ffmpeg writer and raise on ffmpeg failure."""
    if writer is None:
        return
    if writer.stdin is not None:
        writer.stdin.close()
    stderr = writer.stderr.read() if writer.stderr is not None else b""
    ret = writer.wait()
    if ret != 0:
        msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed with code {ret}. stderr:\n{msg}")


def ensure_dir(path: Path) -> None:
    """Create directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str, script_dir: Path) -> Path:
    """Resolve a possibly relative path against the script directory."""
    path = Path(path_str)
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()


def _extract_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def list_images(frames_dir: Path) -> list[Path]:
    """List image files sorted by frame index."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [
        path for path in frames_dir.iterdir() if path.suffix.lower() in exts
    ]
    return sorted(files, key=_extract_index)


def _load_intrinsics_from_alignment_summary(
    aligned_mesh_video_dir: Path,
) -> tuple[np.ndarray, Path]:
    summary_path = (
        aligned_mesh_video_dir / "alignment_summary.json"
    ).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(
            f"alignment_summary.json not found: {summary_path}"
        )

    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise KeyError(
            f"Missing 'camera' dictionary in alignment summary: {summary_path}"
        )

    k_raw = camera.get("intrinsics_3x3")
    if k_raw is None:
        raise KeyError(
            "Missing 'camera.intrinsics_3x3' in alignment summary: "
            f"{summary_path}"
        )

    k = np.array(k_raw, dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(
            "Expected camera.intrinsics_3x3 to resolve to shape (3, 3), "
            f"got {k.shape} in {summary_path}"
        )
    return k.astype(np.float32), summary_path


def _to_device(args_device: str) -> torch.device:
    try:
        dev = torch.device(args_device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {args_device}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None:
        if dev.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {args_device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) available."
            )
    return dev


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _project_points_cv(
    points_cv: np.ndarray,
    k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_cv, dtype=np.float32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    uv = np.zeros((points.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        pts = points[valid]
        z_valid = pts[:, 2]
        uv_valid = np.empty((pts.shape[0], 2), dtype=np.float32)
        uv_valid[:, 0] = (
            (pts[:, 0] * float(k[0, 0])) / z_valid + float(k[0, 2])
        )
        uv_valid[:, 1] = (
            (pts[:, 1] * float(k[1, 1])) / z_valid + float(k[1, 2])
        )
        uv[valid] = uv_valid
    return uv, valid


def _rasterize_mask_from_projected_triangles(
    uv: np.ndarray,
    valid_vertices: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if uv.shape[0] == 0 or faces.shape[0] == 0:
        return mask

    if faces.dtype == np.int32:
        faces_i32 = np.ascontiguousarray(faces)
    else:
        faces_i32 = np.asarray(faces, dtype=np.int32)
    face_valid = valid_vertices[faces_i32].all(axis=1)
    if not np.any(face_valid):
        return mask

    tri = uv[faces_i32[face_valid]]
    tri_min = np.min(tri, axis=1)
    tri_max = np.max(tri, axis=1)
    in_frame = (
        (tri_max[:, 0] >= 0.0)
        & (tri_min[:, 0] <= float(width - 1))
        & (tri_max[:, 1] >= 0.0)
        & (tri_min[:, 1] <= float(height - 1))
    )
    tri = tri[in_frame]
    if tri.shape[0] == 0:
        return mask

    tri_i32 = np.round(tri).astype(np.int32)
    try:
        cv2.fillPoly(mask, tri_i32, 255, lineType=cv2.LINE_8)
    except cv2.error:
        cv2.fillPoly(
            mask,
            [poly for poly in tri_i32],
            255,
            lineType=cv2.LINE_8,
        )
    return mask


def _draw_mask_outline_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    fill_alpha: float = 0.35,
    contour_thickness: int = 2,
) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return image_bgr.copy()

    out = image_bgr.astype(np.float32).copy()
    color_arr = np.array(color_bgr, dtype=np.float32)
    alpha = float(np.clip(fill_alpha, 0.0, 1.0))
    out[mask_bool] = (1.0 - alpha) * out[mask_bool] + alpha * color_arr
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)

    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours_info = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = (
        contours_info[0] if len(contours_info) == 2 else contours_info[1]
    )
    if int(contour_thickness) > 0:
        outline = tuple(int(np.clip(c + 48, 0, 255)) for c in color_bgr)
        cv2.drawContours(
            out_u8,
            contours,
            contourIdx=-1,
            color=outline,
            thickness=int(contour_thickness),
            lineType=cv2.LINE_AA,
        )
    return out_u8


def draw_overlay(
    frame_bgr: np.ndarray,
    verts_cv: np.ndarray,
    faces: np.ndarray,
    k: np.ndarray,
    fill_alpha: float = 0.35,
    contour_thickness: int = 2,
    color_bgr: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """Render an overlay by rasterizing projected triangles into a mask."""
    h, w = frame_bgr.shape[:2]
    if verts_cv.size == 0 or faces.size == 0:
        return frame_bgr.copy()

    uv, valid = _project_points_cv(verts_cv, k)
    mask = _rasterize_mask_from_projected_triangles(
        uv=uv,
        valid_vertices=valid,
        faces=faces,
        width=w,
        height=h,
    )
    return _draw_mask_outline_overlay(
        image_bgr=frame_bgr,
        mask=mask,
        color_bgr=color_bgr,
        fill_alpha=fill_alpha,
        contour_thickness=contour_thickness,
    )


__all__ = [
    "_load_intrinsics_from_alignment_summary",
    "_save_csv",
    "_to_device",
    "close_ffmpeg",
    "draw_overlay",
    "ensure_dir",
    "list_images",
    "resolve_path",
    "start_ffmpeg_writer",
]
