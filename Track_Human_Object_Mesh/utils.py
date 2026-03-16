"""Shared low-level helpers for joint human-object mesh refinement."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def list_images(frames_dir: Path) -> list[Path]:
    """List image files in filename order."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in exts
    )


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of rows to CSV, preserving the first row's column order."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_dirs(
    args: argparse.Namespace,
    script_dir: Path,
) -> dict[str, Path]:
    """Resolve the standard project directories for one video."""
    vname = args.video_name
    parent = script_dir.parent
    aligned = (
        Path(args.aligned_mesh_dir)
        if args.aligned_mesh_dir
        else parent / "Align_Meshes" / "output" / vname
    )
    tracked = (
        Path(args.tracked_object_dir)
        if args.tracked_object_dir
        else parent / "Track_Object_Mesh" / "output" / vname
    )
    seg_obj = (
        Path(args.segment_object_dir)
        if args.segment_object_dir
        else parent / "Segment_Object_Mesh" / "output" / vname
    )
    seg_vid = (
        Path(args.segment_video_dir)
        if args.segment_video_dir
        else parent / "Segment_Video" / "output" / vname
    )
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = script_dir / output_root
    output = output_root / vname

    return {
        "aligned": aligned,
        "tracked": tracked,
        "seg_obj": seg_obj,
        "seg_vid": seg_vid,
        "output": output,
    }


def resolve_pag_path(args: argparse.Namespace, script_dir: Path) -> Path:
    """Return the PAG JSON path for the current video."""
    if args.pag_file is not None:
        return Path(args.pag_file)
    return (
        script_dir.parent
        / "Generate_PAG"
        / "output"
        / args.video_name
        / "output_pag_deepseek_r1_32b.json"
    )


def resolve_smpl_seg(args: argparse.Namespace, script_dir: Path) -> Path:
    """Return the SMPL vertex segmentation path."""
    if args.smpl_seg_json is not None:
        return Path(args.smpl_seg_json)
    return (
        script_dir.parent.parent
        / "GVHMR"
        / "hmr4d"
        / "utils"
        / "body_model"
        / "smpl_vert_segmentation.json"
    )


def resolve_frames_dir(dirs: dict[str, Path]) -> Path:
    """Return the standard frame directory used in this repo."""
    return dirs["seg_vid"] / "_frames"


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
    "close_ffmpeg",
    "draw_overlay",
    "ensure_dir",
    "list_images",
    "resolve_dirs",
    "resolve_frames_dir",
    "resolve_pag_path",
    "resolve_smpl_seg",
    "save_csv",
    "start_ffmpeg_writer",
]
