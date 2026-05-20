"""Utilities used by CoTracker-based SE(3) object mesh tracking."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


def start_ffmpeg_writer(out_path: Path, fps: float, size_hw: tuple[int, int]) -> subprocess.Popen:
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


def close_ffmpeg(writer: subprocess.Popen | None) -> None:
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


def ensure_dir(path: Path) -> None:
    """Create directory if missing."""
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str, script_dir: Path) -> Path:
    """Resolve path against script_dir when relative."""
    path = Path(path_str)
    if not path.is_absolute():
        path = script_dir / path
    return path.resolve()


def _extract_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def list_images(frames_dir: Path) -> list[Path]:
    """List image files sorted by frame index in filename."""
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [path for path in frames_dir.iterdir() if path.suffix.lower() in exts]
    return sorted(files, key=_extract_index)


def _sanitize_object_name(name: str) -> str:
    """Segment_Video object folder name convention."""
    return name.strip().replace(" ", "_").replace("-", "_")


def _resolve_default_dirs(
    args: argparse.Namespace,
    script_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    video_name = args.video_name

    if args.cotracker_video_dir is None:
        cotracker_video_dir = (
            script_dir.parent / "Estimate_Optical_Flow" / "output_cotracker" / video_name
        ).resolve()
    else:
        cotracker_video_dir = resolve_path(args.cotracker_video_dir, script_dir)

    if args.aligned_mesh_video_dir is None:
        aligned_mesh_video_dir = (
            script_dir.parent / "Align_Meshes" / "output" / video_name
        ).resolve()
    else:
        aligned_mesh_video_dir = resolve_path(args.aligned_mesh_video_dir, script_dir)

    if args.segment_video_dir is None:
        segment_video_dir = (
            script_dir.parent / "Segment_Video" / "output" / video_name
        ).resolve()
    else:
        segment_video_dir = resolve_path(args.segment_video_dir, script_dir)

    output_root = resolve_path(args.output_root, script_dir)
    return cotracker_video_dir, aligned_mesh_video_dir, segment_video_dir, output_root


def _resolve_pag_path(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.pag_file is not None:
        pag_path = resolve_path(args.pag_file, script_dir)
        if not pag_path.exists():
            raise FileNotFoundError(f"PAG file not found: {pag_path}")
        return pag_path

    pag_dir = (script_dir.parent / "Generate_PAG" / "output" / args.video_name).resolve()
    if not pag_dir.exists():
        raise FileNotFoundError(f"PAG directory not found: {pag_dir}")
    pag_candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not pag_candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return pag_candidates[0]


def _load_intrinsics_from_alignment_summary(
    aligned_mesh_video_dir: Path,
) -> tuple[np.ndarray, Path]:
    summary_path = (aligned_mesh_video_dir / "alignment_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"alignment_summary.json not found: {summary_path}")

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
            f"Missing 'camera.intrinsics_3x3' in alignment summary: {summary_path}"
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


def _load_pag_objects_from_states_only(pag_path: Path) -> list[tuple[str, str]]:
    with pag_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    object_states = payload.get("object states")
    if not isinstance(object_states, list):
        raise RuntimeError(
            "PAG must contain a list in 'object states'. "
            f"Got: {type(object_states).__name__}"
        )

    objects: list[tuple[str, str]] = []
    seen = set()
    for item in object_states:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        objects.append((name, _sanitize_object_name(name)))

    if not objects:
        raise RuntimeError(
            "No valid object names found in PAG 'object states'. "
            f"File: {pag_path}"
        )
    return objects


def _resolve_frames_dir(cotracker_video_dir: Path, segment_video_dir: Path) -> Path | None:
    candidates = [
        cotracker_video_dir / "_frames",
        segment_video_dir / "_frames",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    return None


def _resolve_object_mask_dir(segment_video_dir: Path, object_slug: str) -> Path:
    return (
        segment_video_dir
        / "objects"
        / object_slug
        / "object_segmentation"
        / "masks"
    ).resolve()


def _list_mask_files(mask_dir: Path) -> list[Path]:
    mask_paths = sorted(mask_dir.glob("frame_*.png"), key=_extract_index)
    if not mask_paths:
        mask_paths = sorted(mask_dir.glob("*.png"), key=_extract_index)
    return mask_paths


def _load_mask_stack(mask_paths: list[Path], mask_threshold: int) -> tuple[np.ndarray, int, int]:
    if not mask_paths:
        raise RuntimeError("mask_paths is empty")
    masks = []
    h_ref, w_ref = -1, -1
    for idx, path in enumerate(mask_paths):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {path}")
        if idx == 0:
            h_ref, w_ref = mask.shape[:2]
        if mask.shape[:2] != (h_ref, w_ref):
            mask = cv2.resize(mask, (w_ref, h_ref), interpolation=cv2.INTER_NEAREST)
        masks.append((mask > int(mask_threshold)).astype(np.float32))
    return np.stack(masks, axis=0).astype(np.float32), h_ref, w_ref


def _normalize_tracks_vis_with_mask_length(
    tracks_raw: np.ndarray,
    vis_raw: np.ndarray,
    expected_t: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize to tracks [N, T, 2], vis [N, T] based on expected_t."""
    tracks = np.asarray(tracks_raw)
    vis = np.asarray(vis_raw)

    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError(f"Expected tracks shape [*, *, 2], got {tracks.shape}")

    if vis.ndim == 3 and vis.shape[-1] == 1:
        vis = vis[..., 0]
    if vis.ndim != 2:
        raise ValueError(f"Expected visibility shape [*, *], got {vis.shape}")

    candidates: list[tuple[np.ndarray, np.ndarray]] = []

    if tracks.shape[0] == expected_t:
        t, n = tracks.shape[0], tracks.shape[1]
        if vis.shape == (t, n):
            candidates.append((tracks.transpose(1, 0, 2), vis.transpose(1, 0)))
        elif vis.shape == (n, t):
            candidates.append((tracks.transpose(1, 0, 2), vis))

    if tracks.shape[1] == expected_t:
        n, t = tracks.shape[0], tracks.shape[1]
        if vis.shape == (n, t):
            candidates.append((tracks, vis))
        elif vis.shape == (t, n):
            candidates.append((tracks, vis.transpose(1, 0)))

    if not candidates:
        if vis.shape == tracks.shape[:2]:
            if tracks.shape[0] < tracks.shape[1]:
                candidates.append((tracks.transpose(1, 0, 2), vis.transpose(1, 0)))
            else:
                candidates.append((tracks, vis))
        elif vis.shape == (tracks.shape[1], tracks.shape[0]):
            if tracks.shape[0] < tracks.shape[1]:
                candidates.append((tracks.transpose(1, 0, 2), vis))
            else:
                candidates.append((tracks, vis.transpose(1, 0)))

    if not candidates:
        raise ValueError(
            "Could not infer tracks/visibility orientation. "
            f"tracks={tracks.shape}, vis={vis.shape}, expected_t={expected_t}"
        )

    tracks_nt2, vis_nt = candidates[0]
    return tracks_nt2.astype(np.float32), vis_nt.astype(np.float32)


def _to_device(args_device: str) -> torch.device:
    try:
        dev = torch.device(args_device)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {args_device}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {args_device}, but only {torch.cuda.device_count()} CUDA device(s) available."
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


def _save_loss_plots(debug_dir: Path, iter_rows: list[dict[str, Any]]) -> dict[str, str]:
    """Save separate loss-term plots."""
    if not iter_rows:
        return {}

    it = np.array([int(r["iter"]) for r in iter_rows], dtype=np.int32)
    total = np.array([float(r["total"]) for r in iter_rows], dtype=np.float32)
    e_img = np.array([float(r["e_img"]) for r in iter_rows], dtype=np.float32)
    e_smooth = np.array([float(r["e_smooth"]) for r in iter_rows], dtype=np.float32)
    e_vel = np.array([float(r["e_vel"]) for r in iter_rows], dtype=np.float32)

    debug_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    def _finite_minmax(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        if abs(mx - mn) < 1e-12:
            pad = max(abs(mx) * 0.05, 1e-3)
            mn -= pad
            mx += pad
        return mn, mx

    def _plot_with_cv2(
        series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
        title: str,
        out_path: Path,
    ) -> None:
        h, w = 720, 1280
        l, r, t, b = 95, 36, 70, 95
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        x0, x1 = l, w - r
        y0, y1 = t, h - b

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (210, 210, 210), 1)

        y_all = np.concatenate([s[1] for s in series], axis=0)
        y_min, y_max = _finite_minmax(y_all)
        x_min = float(np.min(it))
        x_max = float(np.max(it))
        if abs(x_max - x_min) < 1e-12:
            x_max = x_min + 1.0

        for k in range(6):
            frac = float(k) / 5.0
            yy = int(round(y1 - frac * (y1 - y0)))
            xx = int(round(x0 + frac * (x1 - x0)))
            cv2.line(canvas, (x0, yy), (x1, yy), (238, 238, 238), 1, cv2.LINE_AA)
            cv2.line(canvas, (xx, y0), (xx, y1), (238, 238, 238), 1, cv2.LINE_AA)
            y_tick = y_min + frac * (y_max - y_min)
            x_tick = x_min + frac * (x_max - x_min)
            cv2.putText(
                canvas,
                f"{y_tick:.3g}",
                (8, yy + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{int(round(x_tick))}",
                (xx - 18, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )

        def _to_canvas_xy(xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
            xn = (xv - x_min) / (x_max - x_min)
            yn = (yv - y_min) / (y_max - y_min)
            px = x0 + xn * (x1 - x0)
            py = y1 - yn * (y1 - y0)
            pts = np.stack([px, py], axis=1)
            pts = np.round(np.clip(pts, [x0, y0], [x1, y1])).astype(np.int32)
            return pts

        legend_y = y0 + 10
        for label, ys, color in series:
            pts = _to_canvas_xy(it.astype(np.float32), ys.astype(np.float32))
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x1 - 220, legend_y), (x1 - 180, legend_y), color, 3, cv2.LINE_AA)
            cv2.putText(
                canvas,
                label,
                (x1 - 172, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            legend_y += 22

        cv2.putText(canvas, title, (x0, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Iteration",
            (w // 2 - 40, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Loss",
            (16, y0 - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (35, 35, 35),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(out_path), canvas)

    def _save_single(
        y: np.ndarray,
        label: str,
        title: str,
        out_path: Path,
        lw: float = 2.0,
    ) -> None:
        if _HAS_MPL:
            fig = plt.figure(figsize=(9, 5))
            ax = fig.add_subplot(111)
            ax.plot(it, y, label=label, linewidth=lw)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(str(out_path), dpi=160)
            plt.close(fig)
        else:
            _plot_with_cv2([(label, y, (32, 119, 180))], title, out_path)

    total_path = debug_dir / "loss_total.png"
    e_img_path = debug_dir / "loss_e_img.png"
    e_smooth_path = debug_dir / "loss_e_smooth.png"
    e_vel_path = debug_dir / "loss_e_vel.png"
    legacy_combined = debug_dir / "loss_curves.png"

    _save_single(total, "E_total", "Total Loss", total_path, lw=2.0)
    _save_single(e_img, "E_img", "Image Reprojection Loss", e_img_path, lw=1.8)
    _save_single(e_smooth, "E_smooth", "Acceleration Smoothness Loss", e_smooth_path, lw=1.8)
    _save_single(e_vel, "E_vel", "Velocity Regularization Loss", e_vel_path, lw=1.8)
    if legacy_combined.exists():
        legacy_combined.unlink()

    saved["total"] = str(total_path)
    saved["e_img"] = str(e_img_path)
    saved["e_smooth"] = str(e_smooth_path)
    saved["e_vel"] = str(e_vel_path)
    return saved


def _draw_frame0_correspondence(
    frame_bgr: np.ndarray,
    obs_uv_m2: np.ndarray,
    pred_uv_m2: np.ndarray,
    max_points: int,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    n = int(obs_uv_m2.shape[0])
    if n <= 0:
        return out
    max_points = max(1, int(max_points))
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int32)
    else:
        idx = np.arange(n, dtype=np.int32)

    for i in idx:
        ox = int(np.clip(np.round(obs_uv_m2[i, 0]), 0, w - 1))
        oy = int(np.clip(np.round(obs_uv_m2[i, 1]), 0, h - 1))
        px = int(np.clip(np.round(pred_uv_m2[i, 0]), 0, w - 1))
        py = int(np.clip(np.round(pred_uv_m2[i, 1]), 0, h - 1))
        cv2.circle(out, (ox, oy), 2, (0, 255, 0), -1)
        cv2.circle(out, (px, py), 2, (0, 165, 255), -1)
        cv2.arrowedLine(out, (px, py), (ox, oy), (0, 255, 255), 1, tipLength=0.25)
    return out


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
        uv_valid[:, 0] = (pts[:, 0] * float(k[0, 0])) / z_valid + float(k[0, 2])
        uv_valid[:, 1] = (pts[:, 1] * float(k[1, 1])) / z_valid + float(k[1, 2])
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

    tri = uv[faces_i32[face_valid]]  # [Fv,3,2]
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

    # Batch fill in one OpenCV call to avoid per-triangle Python overhead.
    tri_i32 = np.round(tri).astype(np.int32)
    try:
        cv2.fillPoly(mask, tri_i32, 255, lineType=cv2.LINE_8)
    except cv2.error:
        cv2.fillPoly(mask, [poly for poly in tri_i32], 255, lineType=cv2.LINE_8)
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

    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    contours_info = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
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
    """Fast overlay via projected triangle fill + alpha + contour."""
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
    "_draw_frame0_correspondence",
    "_list_mask_files",
    "_load_intrinsics_from_alignment_summary",
    "_load_mask_stack",
    "_load_pag_objects_from_states_only",
    "_normalize_tracks_vis_with_mask_length",
    "_resolve_default_dirs",
    "_resolve_frames_dir",
    "_resolve_object_mask_dir",
    "_resolve_pag_path",
    "_sanitize_object_name",
    "_save_csv",
    "_save_loss_plots",
    "_to_device",
    "close_ffmpeg",
    "draw_overlay",
    "ensure_dir",
    "list_images",
    "resolve_path",
    "start_ffmpeg_writer",
]
