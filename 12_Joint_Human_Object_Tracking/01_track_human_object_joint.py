"""Unified object-trajectory and human-object contact optimization.

This module intentionally does not depend on module-10 outputs.  It reads the
same object point tracks that module 10 used, builds the same frame-0
mesh-surface correspondences, and then runs two stages over one object state:

1. robust object trajectory tracking from 2D tracks;
2. contact-aware refinement with the module-11 human/object losses.

Humans are frozen GVHMR SMPL-X sequences transformed by module-09's alignment
matrix.  Only object SE(3) trajectories and bounded object scales are optimized.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import roma
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


OBJECT_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),
    (255, 128, 0),
    (0, 255, 0),
    (255, 0, 255),
    (128, 255, 128),
    (0, 128, 255),
]

BODY_PART_TO_SEG_KEYS: dict[str, list[str]] = {
    "left hand": ["leftHand", "leftHandIndex1"],
    "right hand": ["rightHand", "rightHandIndex1"],
    "left foot": ["leftFoot", "leftToeBase"],
    "right foot": ["rightFoot", "rightToeBase"],
    "left shoulder": ["leftShoulder"],
    "right shoulder": ["rightShoulder"],
    "left arm": ["leftArm", "leftForeArm"],
    "right arm": ["rightArm", "rightForeArm"],
    "left leg": ["leftUpLeg", "leftLeg"],
    "right leg": ["rightUpLeg", "rightLeg"],
    "left hip": ["leftUpLeg"],
    "right hip": ["rightUpLeg"],
    "hips": ["hips"],
    "head": ["head"],
    "neck": ["neck"],
    "spine": ["spine", "spine1", "spine2"],
}

PROJECT_BODY_PART_TO_SEGMENT_ID: dict[str, str] = {
    "left hand": "left_hand",
    "right hand": "right_hand",
    "left arm": "left_arm",
    "right arm": "right_arm",
    "left shoulder": "left_shoulder",
    "right shoulder": "right_shoulder",
    "left leg": "left_leg",
    "right leg": "right_leg",
    "left hip": "left_leg",
    "right hip": "right_leg",
    "left foot": "left_foot",
    "right foot": "right_foot",
    "head": "head",
    "hips": "hips",
}

PROJECT_BODY_PART_TO_CONTACT_SEGMENT_ID: dict[str, str] = {
    "left hand": "left_hand_inner",
    "right hand": "right_hand_inner",
    "left foot": "left_foot_bottom",
    "right foot": "right_foot_bottom",
    "hips": "hips_contact",
}

SURFACE_SAMPLE_SEED = 7


LOSS_KEYS_STAGE2 = (
    "track_reproj",
    "object_cd2d",
    "object_part_cd2d",
    "object_smooth_trans",
    "object_smooth_rot",
    "object_scale",
    "intersect",
    "nocontact",
    "contact_drift",
)

LOSS_TERM_KEYS = (
    "tracking",
    "object_cd2d",
    "object_part_cd2d",
    "object_smooth_trans",
    "object_smooth_rot",
    "object_scale",
    "intersect",
    "nocontact",
    "contact_drift",
)
FRAME_DIAGNOSTIC_TERM_KEYS = tuple(key for key in LOSS_TERM_KEYS if key != "object_scale")


@dataclass
class PAGObjectState:
    name: str
    slug: str
    is_translational: bool
    is_rotational: bool


@dataclass
class PAGEdge:
    node_a: str
    node_b: str
    is_continuous: bool
    is_rel_static: bool


@dataclass
class PAG:
    object_states: list[PAGObjectState]
    body_part_nodes: list[str]
    object_part_nodes: list[str]
    edges: list[PAGEdge]


@dataclass
class PackedPointCloud2D:
    points: torch.Tensor
    lengths: torch.Tensor


@dataclass
class ObjectPartSegments:
    vert_ids: dict[str, np.ndarray]
    face_ids: dict[str, np.ndarray]


@dataclass
class SDFGrid:
    sdf_volume: torch.Tensor
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor


@dataclass
class HumanData:
    name: str
    slug: str
    base_verts: torch.Tensor
    faces: np.ndarray
    faces_torch: torch.Tensor
    part_points: dict[str, torch.Tensor]
    part_vert_ids: dict[str, np.ndarray]
    contact_part_points: dict[str, torch.Tensor]
    contact_part_vert_ids: dict[str, np.ndarray]


@dataclass
class ObjectData:
    name: str
    slug: str
    state: PAGObjectState
    template_verts: torch.Tensor
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    faces_torch: torch.Tensor
    tracked_poses: np.ndarray
    tracked_poses_torch: torch.Tensor
    tracked_rotvecs: torch.Tensor
    tracked_trans: torch.Tensor
    part_vert_ids: dict[str, np.ndarray]
    part_face_ids: dict[str, np.ndarray]
    sampled_points: torch.Tensor
    part_sampled_points: dict[str, torch.Tensor]
    mask_points_2d: PackedPointCloud2D | None
    part_mask_points_2d: dict[str, PackedPointCloud2D]
    sdf_grid: SDFGrid | None
    color_bgr: tuple[int, int, int]


@dataclass
class InteractionNode:
    raw_node: str
    entity_name: str
    part_name: str
    is_human: bool
    human_slug: str | None
    object_slug: str | None
    resolved_part_name: str | None
    vert_ids: np.ndarray


@dataclass
class InteractionEdge:
    node_a: InteractionNode
    node_b: InteractionNode
    is_continuous: bool
    is_rel_static: bool


@dataclass
class LossResult:
    total: torch.Tensor
    tracking: torch.Tensor
    object_cd2d: torch.Tensor
    object_part_cd2d: torch.Tensor
    object_smooth_trans: torch.Tensor
    object_smooth_rot: torch.Tensor
    object_scale: torch.Tensor
    intersect: torch.Tensor
    nocontact: torch.Tensor
    contact_drift: torch.Tensor
    weights: dict[str, float]


@dataclass
class DiagnosticLossResult:
    sequence: LossResult
    per_frame_raw: dict[str, torch.Tensor]
    global_raw: dict[str, torch.Tensor]


@dataclass
class ProblemContext:
    dirs: dict[str, Path]
    out_dir: Path
    pag_path: Path
    smpl_seg_path: Path
    intr_path: Path
    device: torch.device
    k: np.ndarray
    k_torch: torch.Tensor
    width: int
    height: int
    num_frames: int
    pag: PAG
    humans: dict[str, HumanData]
    human_keys: list[str]
    objects: dict[str, ObjectData]
    obj_keys: list[str]
    interaction_edges: list[InteractionEdge]


@dataclass
class ObjectTrackState:
    name: str
    slug: str
    is_translational: bool
    is_rotational: bool
    mesh: trimesh.Trimesh
    verts: np.ndarray
    faces: np.ndarray
    vertex_colors: np.ndarray | None
    x0: torch.Tensor
    obs_uv: torch.Tensor
    vis: torch.Tensor
    masks: torch.Tensor
    rotvecs: torch.nn.Parameter
    trans: torch.nn.Parameter
    raw_scale_delta: torch.nn.Parameter
    num_input_tracks: int
    num_valid_seed_tracks: int
    num_dropped_invalid_face: int
    num_dropped_outside_mask0: int
    num_dropped_nonfinite_seed: int
    pnp_info: list[dict[str, Any]]


@dataclass
class TrackingLossBundle:
    total: torch.Tensor
    track_reproj: torch.Tensor
    smooth_trans: torch.Tensor
    smooth_rot: torch.Tensor
    scale_prior: torch.Tensor
    per_object: dict[str, dict[str, torch.Tensor]]


@dataclass
class Stage2LossBundle:
    total: torch.Tensor
    track_reproj: torch.Tensor
    contact_result: Any


def _resolve_path(path_str: str | None, base: Path) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def _extract_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**18


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images(frames_dir: Path) -> list[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [path for path in frames_dir.iterdir() if path.suffix.lower() in exts]
    return sorted(files, key=_extract_index)


def start_ffmpeg_writer(out_path: Path, fps: float, size_hw: tuple[int, int]) -> subprocess.Popen:
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
    if writer is None:
        return
    if writer.stdin is not None:
        writer.stdin.close()
    stderr = writer.stderr.read() if writer.stderr is not None else b""
    ret = writer.wait()
    if ret != 0:
        msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed with code {ret}. stderr:\n{msg}")


def _project_points_cv(points_cv: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    faces_i32 = np.ascontiguousarray(faces.astype(np.int32, copy=False))
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
    mask_u8 = mask_bool.astype(np.uint8) * 255
    contours_info = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    h, w = frame_bgr.shape[:2]
    if verts_cv.size == 0 or faces.size == 0:
        return frame_bgr.copy()
    uv, valid = _project_points_cv(verts_cv, k)
    mask = _rasterize_mask_from_projected_triangles(uv, valid, faces, w, h)
    return _draw_mask_outline_overlay(
        frame_bgr,
        mask,
        color_bgr=color_bgr,
        fill_alpha=fill_alpha,
        contour_thickness=contour_thickness,
    )


def _save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_pag_path(args: argparse.Namespace) -> Path:
    if args.pag_file is not None:
        pag_path = _resolve_path(args.pag_file, SCRIPT_DIR)
        if pag_path is None or not pag_path.exists():
            raise FileNotFoundError(f"PAG file not found: {pag_path}")
        return pag_path

    pag_dir = PROJECT_DIR / "01_Generate_PAG" / "output" / args.interaction_name
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in {pag_dir}")
    return candidates[0].resolve()


def _resolve_smpl_seg_path(args: argparse.Namespace) -> Path:
    if args.smpl_seg_json is not None:
        seg_path = _resolve_path(args.smpl_seg_json, SCRIPT_DIR)
        if seg_path is None or not seg_path.exists():
            raise FileNotFoundError(f"SMPL-X segmentation JSON not found: {seg_path}")
        return seg_path
    return (
        PROJECT_DIR
        / "06_Estimate_Human_Motion"
        / "assets"
        / "smplx_vert_segmentation.json"
    ).resolve()


def _resolve_dirs(args: argparse.Namespace) -> dict[str, Path]:
    interaction = args.interaction_name
    output_root = _resolve_path(args.output_root, SCRIPT_DIR)
    assert output_root is not None
    return {
        "object_tracks": (
            _resolve_path(args.object_point_tracks_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "07_Track_Object_Points" / "output" / interaction)
        ).resolve(),
        "aligned": (
            _resolve_path(args.aligned_mesh_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "09_Align_Meshes" / "output" / interaction)
        ).resolve(),
        "human_motion": (
            _resolve_path(args.human_motion_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "06_Estimate_Human_Motion" / "output" / interaction)
        ).resolve(),
        "seg_vid": (
            _resolve_path(args.segment_video_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "03_Segment_Video" / "output" / interaction)
        ).resolve(),
        "seg_obj": (
            _resolve_path(args.segment_object_dir, SCRIPT_DIR)
            or (PROJECT_DIR / "08_Segment_Object_Mesh" / "output" / interaction)
        ).resolve(),
        "output": (output_root / interaction).resolve(),
    }


def _to_device(device_name: str) -> torch.device:
    try:
        dev = torch.device(device_name)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid --device value: {device_name}") from exc
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if dev.type == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {device_name}, but only {torch.cuda.device_count()} CUDA device(s) available."
        )
    return dev


def _resolve_frames_dir(object_point_tracks_dir: Path, segment_video_dir: Path) -> Path | None:
    for cand in (object_point_tracks_dir / "_frames", segment_video_dir / "_frames"):
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


def _load_intrinsics_from_alignment_summary(aligned_mesh_video_dir: Path) -> tuple[np.ndarray, Path]:
    summary_path = (aligned_mesh_video_dir / "alignment_summary.json").resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"alignment_summary.json not found: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise KeyError(f"Missing 'camera' dictionary in alignment summary: {summary_path}")
    k_raw = camera.get("intrinsics_3x3")
    if k_raw is None:
        raise KeyError(f"Missing 'camera.intrinsics_3x3' in alignment summary: {summary_path}")
    k = np.array(k_raw, dtype=np.float32)
    while k.ndim > 2:
        k = k[0]
    if k.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3, 3), got {k.shape} in {summary_path}")
    return k.astype(np.float32), summary_path


def _human_entity_to_slug(entity_name: str) -> str:
    return _sanitize(entity_name).lower()


def _object_part_to_slug(part_name: str) -> str:
    return _sanitize(part_name).lower()


def _parse_pag(pag_path: Path) -> PAG:
    with pag_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    obj_states: list[PAGObjectState] = []
    for item in data.get("object states", []):
        name = item["name"].strip()
        obj_states.append(
            PAGObjectState(
                name=name,
                slug=_sanitize(name),
                is_translational=bool(item.get("is_translational", True)),
                is_rotational=bool(item.get("is_rotational", True)),
            )
        )

    edges: list[PAGEdge] = []
    for edge in data.get("interaction edges", []):
        nodes = edge["nodes"]
        edges.append(
            PAGEdge(
                node_a=nodes[0].strip(),
                node_b=nodes[1].strip(),
                is_continuous=bool(edge.get("is_continuous", True)),
                is_rel_static=bool(edge.get("is_rel_static", False)),
            )
        )

    return PAG(
        object_states=obj_states,
        body_part_nodes=[s.strip() for s in data.get("body part nodes", [])],
        object_part_nodes=[s.strip() for s in data.get("object part nodes", [])],
        edges=edges,
    )


def _parse_node(node_str: str) -> tuple[str, str]:
    parts = node_str.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse PAG node: '{node_str}'")
    return parts[0].strip(), parts[1].strip()


def _is_human_node(node_str: str) -> bool:
    return node_str.lower().startswith("person")


def _load_smpl_body_and_contact_seg(
    seg_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with seg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if "segments" in raw:
        segments = raw["segments"]
        body_result: dict[str, np.ndarray] = {}
        for pag_name, segment_id in PROJECT_BODY_PART_TO_SEGMENT_ID.items():
            if segment_id in segments:
                body_result[pag_name] = np.array(segments[segment_id], dtype=np.int64)

        contact_result: dict[str, np.ndarray] = {}
        for pag_name, segment_id in PROJECT_BODY_PART_TO_CONTACT_SEGMENT_ID.items():
            if segment_id in segments:
                contact_result[pag_name] = np.array(segments[segment_id], dtype=np.int64)
        return body_result, contact_result

    body_result: dict[str, np.ndarray] = {}
    for pag_name, seg_keys in BODY_PART_TO_SEG_KEYS.items():
        indices: list[int] = []
        for seg_key in seg_keys:
            indices.extend(raw.get(seg_key, []))
        if indices:
            body_result[pag_name] = np.unique(np.array(indices, dtype=np.int64))
    return body_result, {}


def _load_object_part_segments(
    seg_obj_dir: Path,
    obj_slug: str,
    mesh_faces: np.ndarray,
) -> ObjectPartSegments:
    labels_path = (
        seg_obj_dir
        / obj_slug
        / "segmented_meshes"
        / f"{obj_slug}_triangle_labels.json"
    )
    if not labels_path.exists():
        raise FileNotFoundError(f"Triangle labels not found: {labels_path}")
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    label_map: dict[str, int] = data["label_map"]
    tri_labels = np.array(data["triangle_labels"], dtype=np.int32)
    part_vert_ids: dict[str, np.ndarray] = {}
    part_face_ids: dict[str, np.ndarray] = {}
    for part_name, label_id in label_map.items():
        tri_mask = tri_labels == label_id
        face_subset = mesh_faces[tri_mask]
        vert_ids = np.unique(face_subset.ravel())
        if vert_ids.size > 0:
            part_vert_ids[part_name] = vert_ids
            part_face_ids[part_name] = np.flatnonzero(tri_mask).astype(np.int64)
    return ObjectPartSegments(vert_ids=part_vert_ids, face_ids=part_face_ids)


def _subsample_indices(n: int, max_pts: int) -> np.ndarray:
    if n <= max_pts:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_pts).astype(np.int64)


def _sample_surface_points(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count <= 0 or faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri = vertices[faces]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * 0.5
    positive = areas > 1e-12
    if not np.any(positive):
        return vertices[_subsample_indices(vertices.shape[0], count)]
    valid_tri = tri[positive]
    weights = areas[positive]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(valid_tri.shape[0], size=count, replace=True, p=weights)
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack([1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2], axis=1)
    sampled_tri = valid_tri[face_ids]
    return np.sum(sampled_tri * bary[:, :, None], axis=1).astype(np.float32)


def _mask_to_point_cloud(mask: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs.astype(np.float32) + 0.5, ys.astype(np.float32) + 0.5], axis=1)
    return pts[_subsample_indices(len(pts), max_points)].astype(np.float32)


def _pack_2d_point_clouds(point_arrays: list[np.ndarray], device: torch.device) -> PackedPointCloud2D:
    max_len = max((arr.shape[0] for arr in point_arrays), default=0)
    max_len = max(max_len, 1)
    packed = np.zeros((len(point_arrays), max_len, 2), dtype=np.float32)
    lengths = np.zeros((len(point_arrays),), dtype=np.int64)
    for i, arr in enumerate(point_arrays):
        lengths[i] = arr.shape[0]
        if arr.shape[0] > 0:
            packed[i, : arr.shape[0]] = arr
    return PackedPointCloud2D(
        points=torch.from_numpy(packed).to(device),
        lengths=torch.from_numpy(lengths).to(device),
    )


def _load_mask_point_clouds(
    masks_dir: Path,
    num_frames: int,
    width: int,
    height: int,
    max_points: int,
    device: torch.device,
) -> PackedPointCloud2D | None:
    if not masks_dir.exists():
        return None
    arrays: list[np.ndarray] = []
    for frame_idx in range(num_frames):
        mask_path = masks_dir / f"frame_{frame_idx:04d}.png"
        if not mask_path.exists():
            arrays.append(np.zeros((0, 2), dtype=np.float32))
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            arrays.append(np.zeros((0, 2), dtype=np.float32))
            continue
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        arrays.append(_mask_to_point_cloud(mask, max_points))
    return _pack_2d_point_clouds(arrays, device)


def _infer_image_size(dirs: dict[str, Path]) -> tuple[int, int]:
    frames_dir = _resolve_frames_dir(dirs["object_tracks"], dirs["seg_vid"])
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
        if frame_paths:
            frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if frame is not None:
                height, width = frame.shape[:2]
                return width, height
    sample_candidates = sorted(dirs["seg_vid"].glob("humans/*/masks/frame_0000.png"))
    sample_candidates.extend(
        sorted(dirs["seg_vid"].glob("objects/*/object_segmentation/masks/frame_0000.png"))
    )
    for sample_path in sample_candidates:
        mask = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            height, width = mask.shape[:2]
            return width, height
    raise FileNotFoundError("Could not infer image size from frames or masks.")


def _load_human_data(
    human_name: str,
    human_slug: str,
    human_verts_np: np.ndarray,
    human_faces: np.ndarray,
    body_seg: dict[str, np.ndarray],
    contact_seg: dict[str, np.ndarray],
    device: torch.device,
) -> HumanData:
    vertex_count = human_verts_np.shape[1]
    all_segment_ids = list(body_seg.values()) + list(contact_seg.values())
    max_segment_id = max((int(ids.max()) for ids in all_segment_ids if ids.size > 0), default=-1)
    if max_segment_id >= vertex_count:
        raise ValueError(
            f"Human mesh '{human_slug}' has {vertex_count} vertices, but the "
            f"SMPL segmentation references vertex {max_segment_id}."
        )
    base_verts = torch.from_numpy(human_verts_np).float().to(device)
    faces_torch = torch.from_numpy(human_faces.astype(np.int64)).to(device)
    return HumanData(
        name=human_name,
        slug=human_slug,
        base_verts=base_verts,
        faces=human_faces,
        faces_torch=faces_torch,
        part_points={part_name: base_verts[:, vert_ids, :] for part_name, vert_ids in body_seg.items()},
        part_vert_ids=body_seg,
        contact_part_points={
            part_name: base_verts[:, vert_ids, :] for part_name, vert_ids in contact_seg.items()
        },
        contact_part_vert_ids=contact_seg,
    )


def _load_object_mask_targets(
    seg_vid_dir: Path,
    slug: str,
    part_names: list[str],
    num_frames: int,
    width: int,
    height: int,
    device: torch.device,
    max_points: int,
) -> tuple[PackedPointCloud2D | None, dict[str, PackedPointCloud2D]]:
    object_mask_dir = seg_vid_dir / "objects" / slug / "object_segmentation" / "masks"
    object_mask_points = _load_mask_point_clouds(
        object_mask_dir,
        num_frames,
        width,
        height,
        max_points,
        device,
    )
    part_mask_points: dict[str, PackedPointCloud2D] = {}
    for part_name in part_names:
        part_mask_dir = seg_vid_dir / "objects" / slug / "parts_segmentation" / "masks" / part_name
        packed = _load_mask_point_clouds(part_mask_dir, num_frames, width, height, max_points, device)
        if packed is not None:
            part_mask_points[part_name] = packed
    return object_mask_points, part_mask_points


def _build_sdf_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    device: torch.device,
    padding: float = 0.05,
) -> SDFGrid:
    from pysdf import SDF as PySDF

    sdf_func = PySDF(vertices.astype(np.float32), faces.astype(np.uint32))
    vmin = vertices.min(axis=0) - padding
    vmax = vertices.max(axis=0) + padding
    lin = [np.linspace(vmin[i], vmax[i], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    query_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
    sdf_vals = -sdf_func(query_pts)
    sdf_vol = sdf_vals.reshape(1, 1, resolution, resolution, resolution)
    return SDFGrid(
        sdf_volume=torch.from_numpy(sdf_vol.astype(np.float32)).to(device),
        bbox_min=torch.tensor(vmin.reshape(1, 1, 3), dtype=torch.float32, device=device),
        bbox_max=torch.tensor(vmax.reshape(1, 1, 3), dtype=torch.float32, device=device),
    )


def _extract_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray | None:
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None
    raw_colors = getattr(visual, "vertex_colors", None)
    if raw_colors is None:
        return None
    colors = np.asarray(raw_colors)
    if colors.shape[0] != len(mesh.vertices):
        return None
    return colors.copy()


def _resolve_interaction_node(
    node_str: str,
    humans: dict[str, HumanData],
    objects: dict[str, ObjectData],
    body_seg: dict[str, np.ndarray],
) -> InteractionNode:
    entity_name, part_name = _parse_node(node_str)
    part_name_norm = part_name.lower().strip()
    if _is_human_node(node_str):
        human_slug = _human_entity_to_slug(entity_name)
        if human_slug not in humans:
            raise KeyError(f"Human '{human_slug}' not loaded")
        if part_name_norm not in body_seg:
            raise KeyError(f"Body part '{part_name_norm}' not in segmentation")
        return InteractionNode(
            raw_node=node_str,
            entity_name=entity_name,
            part_name=part_name_norm,
            is_human=True,
            human_slug=human_slug,
            object_slug=None,
            resolved_part_name=part_name_norm,
            vert_ids=body_seg[part_name_norm],
        )

    obj_slug = _sanitize(entity_name)
    if obj_slug not in objects:
        raise KeyError(f"Object '{obj_slug}' not loaded")

    resolved_part_name = None
    vert_ids = np.arange(objects[obj_slug].template_verts.shape[0], dtype=np.int64)
    part_slug = _object_part_to_slug(part_name)
    for candidate_name, candidate_vids in objects[obj_slug].part_vert_ids.items():
        if _object_part_to_slug(candidate_name) == part_slug:
            resolved_part_name = candidate_name
            vert_ids = candidate_vids
            break
    if resolved_part_name is None:
        print(f"  [WARN] Part '{part_name}' not found in {obj_slug}, using whole mesh.")
    return InteractionNode(
        raw_node=node_str,
        entity_name=entity_name,
        part_name=part_name_norm,
        is_human=False,
        human_slug=None,
        object_slug=obj_slug,
        resolved_part_name=resolved_part_name,
        vert_ids=vert_ids,
    )


def _load_alignment_transforms(aligned_dir: Path) -> dict[str, dict[str, Any]]:
    path = aligned_dir / "meshes" / "transforms.json"
    payload = _load_json(path)
    rows = payload["transforms"]
    return {str(row["slug"]): row for row in rows}


def _human_result_path(
    human_slug: str,
    row: dict[str, Any],
    human_motion_dir: Path,
) -> Path:
    input_path = row.get("input_mesh_path")
    if isinstance(input_path, str) and input_path:
        path = Path(input_path)
        if path.exists():
            return path.resolve()
    return (
        human_motion_dir / "humans" / human_slug / "hmr4d_results.pt"
    ).resolve()


def _apply_transform_np(verts: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    rot_scale = matrix[:3, :3].astype(np.float32)
    trans = matrix[:3, 3].astype(np.float32)
    return (verts.astype(np.float32) @ rot_scale.T) + trans[None, None, :]


def _load_smplx_human_sequence(
    result_path: Path,
    smplx_layer: Any,
    alignment_matrix: np.ndarray,
    num_frames: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = torch.load(str(result_path), map_location="cpu")
    params = data["smpl_params_incam"]
    body_pose = params["body_pose"].detach().clone().float()[:num_frames]
    global_orient = params["global_orient"].detach().clone().float()[:num_frames]
    transl = params["transl"].detach().clone().float()[:num_frames]
    betas_raw = params["betas"].detach().clone().float()
    beta0 = betas_raw[:1] if betas_raw.ndim > 1 else betas_raw.view(1, -1)

    verts_batches: list[np.ndarray] = []
    for start in range(0, body_pose.shape[0], batch_size):
        end = min(start + batch_size, body_pose.shape[0])
        count = end - start
        zeros_3 = torch.zeros(count, 3, dtype=torch.float32)
        zeros_hand = torch.zeros(
            count,
            int(getattr(smplx_layer, "num_pca_comps", 12)),
            dtype=torch.float32,
        )
        zeros_expr = torch.zeros(
            count,
            int(getattr(smplx_layer, "num_expression_coeffs", 10)),
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = smplx_layer(
                betas=beta0.repeat(count, 1),
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                transl=transl[start:end],
                left_hand_pose=zeros_hand,
                right_hand_pose=zeros_hand,
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
                expression=zeros_expr,
            )
        verts_batches.append(output.vertices.detach().cpu().numpy().astype(np.float32))

    verts = np.concatenate(verts_batches, axis=0)
    verts = _apply_transform_np(verts, alignment_matrix)
    faces = np.asarray(smplx_layer.faces, dtype=np.int32)
    return verts.astype(np.float32), faces


def _load_frozen_humans(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    transforms_by_slug: dict[str, dict[str, Any]],
    body_seg: dict[str, np.ndarray],
    contact_seg: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, HumanData], list[str], dict[str, dict[str, Any]], int]:
    import smplx

    smpl_folder = _resolve_path(args.smpl_folder, SCRIPT_DIR)
    if smpl_folder is None or not smpl_folder.exists():
        raise FileNotFoundError(f"SMPL-X body model folder not found: {smpl_folder}")

    print(f"Loading frozen SMPL-X humans from GVHMR using: {smpl_folder}")
    smplx_layer = smplx.create(
        str(smpl_folder),
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
    )
    smplx_layer.eval()

    human_rows = [
        row for row in transforms_by_slug.values() if row.get("kind") == "human"
    ]
    if not human_rows:
        raise RuntimeError("No human rows found in module-09 transforms.json.")

    humans: dict[str, HumanData] = {}
    human_keys: list[str] = []
    human_metadata: dict[str, dict[str, Any]] = {}
    num_frames: int | None = None

    for row in sorted(human_rows, key=lambda item: str(item["slug"])):
        slug = str(row["slug"])
        result_path = _human_result_path(slug, row, dirs["human_motion"])
        if not result_path.exists():
            raise FileNotFoundError(f"GVHMR result file not found: {result_path}")

        data = torch.load(str(result_path), map_location="cpu")
        params = data["smpl_params_incam"]
        available_frames = int(params["body_pose"].shape[0])
        if num_frames is None:
            num_frames = available_frames
        else:
            num_frames = min(num_frames, available_frames)

    if args.end_frame >= 0:
        num_frames = min(int(num_frames), int(args.end_frame) + 1)
    assert num_frames is not None and num_frames > 0

    for row in sorted(human_rows, key=lambda item: str(item["slug"])):
        slug = str(row["slug"])
        name = str(row.get("name", slug)).replace("_", " ")
        result_path = _human_result_path(slug, row, dirs["human_motion"])
        matrix = np.asarray(
            row["source_to_output_matrix_4x4"],
            dtype=np.float32,
        )
        verts_np, faces_np = _load_smplx_human_sequence(
            result_path=result_path,
            smplx_layer=smplx_layer,
            alignment_matrix=matrix,
            num_frames=num_frames,
            batch_size=max(1, int(args.smplx_batch_size)),
        )
        humans[slug] = _load_human_data(
            human_name=name,
            human_slug=slug,
            human_verts_np=verts_np,
            human_faces=faces_np,
            body_seg=body_seg,
            contact_seg=contact_seg,
            device=device,
        )
        human_keys.append(slug)
        human_metadata[slug] = {
            "source": "module_06_gvhmr_smpl_params_incam",
            "hmr4d_results": str(result_path),
            "module09_transform": matrix.tolist(),
            "num_frames": int(verts_np.shape[0]),
            "num_verts": int(verts_np.shape[1]),
            "num_faces": int(faces_np.shape[0]),
            "future_human_optimization_hooks": {
                "root_translation_delta": False,
                "root_rotation_delta": False,
                "body_pose_delta": False,
                "gvhmr_root_orient_prior": False,
                "gvhmr_body_pose_prior": False,
                "human_temporal_smoothness": False,
            },
        }
        print(
            f"  Human {slug}: {verts_np.shape[0]} frames, "
            f"{verts_np.shape[1]} verts, {faces_np.shape[0]} faces"
        )

    return humans, human_keys, human_metadata, int(num_frames)


def _cv_to_p3d_torch(pts: torch.Tensor) -> torch.Tensor:
    out = pts.clone()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _build_rasterizer(
    device: torch.device,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    h: int,
    w: int,
    bin_size: int,
) -> MeshRasterizer:
    cameras = PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[h, w]], device=device, dtype=torch.float32),
        in_ndc=False,
        device=device,
    )
    raster_settings = RasterizationSettings(
        image_size=(h, w),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=int(bin_size),
        max_faces_per_bin=300_000,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def _sample_mask_bilinear_single(mask_hw: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if uv.numel() == 0:
        return torch.zeros(0, dtype=torch.float32, device=mask_hw.device)
    h, w = mask_hw.shape
    grid_x = (2.0 * uv[:, 0] / max(w - 1, 1)) - 1.0
    grid_y = (2.0 * uv[:, 1] / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        mask_hw.view(1, 1, h, w),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


def _sample_masks_bilinear_seq(masks: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    if uv.numel() == 0:
        return torch.zeros(masks.shape[0], 0, dtype=torch.float32, device=masks.device)
    _, h, w = masks.shape
    grid_x = (2.0 * uv[..., 0] / max(w - 1, 1)) - 1.0
    grid_y = (2.0 * uv[..., 1] / max(h - 1, 1)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)
    sampled = F.grid_sample(
        masks.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)


@dataclass
class SeedMappingResult:
    points_cv: torch.Tensor
    valid_seed_mask: np.ndarray
    invalid_face_count: int
    outside_mask0_count: int
    nonfinite_seed_count: int


def _map_seed_points_to_mesh(
    seed_uv: np.ndarray,
    verts_cv: np.ndarray,
    faces: np.ndarray,
    mask0: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    h: int,
    w: int,
    device: torch.device,
    bin_size: int,
    mask_gate_threshold: float,
) -> SeedMappingResult:
    rasterizer = _build_rasterizer(device, fx, fy, cx, cy, h, w, bin_size)
    verts_cv_t = torch.from_numpy(verts_cv).to(device=device, dtype=torch.float32)
    verts_p3d = _cv_to_p3d_torch(verts_cv_t)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device)
    mesh = Meshes(verts=[verts_p3d], faces=[faces_t])
    with torch.no_grad():
        fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0, :]

    seed_t = torch.from_numpy(seed_uv).to(device=device, dtype=torch.float32)
    finite = torch.isfinite(seed_t).all(dim=1)
    x_idx = torch.clamp(torch.round(seed_t[:, 0]).long(), 0, w - 1)
    y_idx = torch.clamp(torch.round(seed_t[:, 1]).long(), 0, h - 1)
    face_id = pix_to_face[y_idx, x_idx]
    bary_seed = bary_coords[y_idx, x_idx]
    mask0_t = torch.from_numpy(mask0.astype(np.float32)).to(device)
    mask_vals = _sample_mask_bilinear_single(mask0_t, seed_t)
    valid = finite & (face_id >= 0) & (mask_vals >= mask_gate_threshold)

    points_all = torch.zeros(seed_t.shape[0], 3, dtype=torch.float32, device=device)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    if idx.numel() > 0:
        tri_verts = verts_cv_t[faces_t[face_id[idx].long()]]
        points_all[idx] = (bary_seed[idx].unsqueeze(-1) * tri_verts).sum(dim=1)

    return SeedMappingResult(
        points_cv=points_all[idx],
        valid_seed_mask=valid.cpu().numpy().astype(bool),
        invalid_face_count=int(((face_id < 0) & finite).sum().item()),
        outside_mask0_count=int(
            ((mask_vals < mask_gate_threshold) & (face_id >= 0) & finite).sum().item()
        ),
        nonfinite_seed_count=int((~finite).sum().item()),
    )


def _build_T_matrices(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    t_frames: int,
    device: torch.device,
) -> torch.Tensor:
    eye = torch.eye(4, device=device, dtype=rotvecs.dtype if rotvecs.numel() else torch.float32).unsqueeze(0)
    if t_frames <= 1:
        return eye
    R = axis_angle_to_matrix(rotvecs)
    T = torch.zeros(t_frames - 1, 4, 4, device=device, dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    T[:, 3, 3] = 1.0
    return torch.cat([eye.to(dtype=T.dtype), T], dim=0)


def _huber_on_squared(s: torch.Tensor, delta: float) -> torch.Tensor:
    d2 = delta * delta
    sqrt_s = torch.sqrt(s.clamp(min=1e-12))
    return torch.where(s <= d2, s, 2.0 * delta * sqrt_s - d2)


def _pnp_sequential_init(
    x0_cv: np.ndarray,
    obs_uv: np.ndarray,
    vis: np.ndarray,
    k: np.ndarray,
    ransac_thresh: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    t_total, _ = obs_uv.shape[:2]
    rotvecs_out = np.zeros((t_total - 1, 3), dtype=np.float32)
    trans_out = np.zeros((t_total - 1, 3), dtype=np.float32)
    info: list[dict[str, Any]] = []
    prev_rvec = np.zeros((3, 1), dtype=np.float64)
    prev_tvec = np.zeros((3, 1), dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)
    k64 = k.astype(np.float64)

    for frame_idx in range(1, t_total):
        valid = vis[frame_idx] > 0.5
        n_valid = int(valid.sum())
        if n_valid < 6:
            rotvecs_out[frame_idx - 1] = prev_rvec.ravel().astype(np.float32)
            trans_out[frame_idx - 1] = prev_tvec.ravel().astype(np.float32)
            info.append({"frame": frame_idx, "n_valid": n_valid, "n_inliers": 0, "pnp_ok": False})
            continue

        pts3d = x0_cv[valid].astype(np.float64)
        pts2d = obs_uv[frame_idx, valid].astype(np.float64)
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            k64,
            dist_coeffs,
            rvec=prev_rvec.copy(),
            tvec=prev_tvec.copy(),
            useExtrinsicGuess=(frame_idx > 1),
            iterationsCount=200,
            reprojectionError=ransac_thresh,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        n_inliers = len(inliers) if (success and inliers is not None) else 0
        if success and n_inliers >= 6:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    pts3d[inliers.ravel()],
                    pts2d[inliers.ravel()],
                    k64,
                    dist_coeffs,
                    rvec,
                    tvec,
                )
            except cv2.error:
                pass
        if success:
            rotvecs_out[frame_idx - 1] = rvec.ravel().astype(np.float32)
            trans_out[frame_idx - 1] = tvec.ravel().astype(np.float32)
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()
        else:
            rotvecs_out[frame_idx - 1] = prev_rvec.ravel().astype(np.float32)
            trans_out[frame_idx - 1] = prev_tvec.ravel().astype(np.float32)
        info.append(
            {"frame": frame_idx, "n_valid": n_valid, "n_inliers": n_inliers, "pnp_ok": bool(success)}
        )
    return rotvecs_out, trans_out, info


def _identify_outlier_tracks(
    x0: torch.Tensor,
    obs_uv: torch.Tensor,
    vis: torch.Tensor,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    threshold_px: float,
    max_fraction: float,
) -> torch.Tensor:
    with torch.no_grad():
        num_frames = obs_uv.shape[0]
        num_tracks = x0.shape[0]
        T_mats = _build_T_matrices(rotvecs, trans, num_frames, x0.device)
        R_all = T_mats[:, :3, :3]
        t_all = T_mats[:, :3, 3]
        xt = torch.einsum("tij,mj->tmi", R_all, x0) + t_all.unsqueeze(1)
        z = xt[..., 2].clamp(min=1e-6)
        pred_uv = torch.stack([fx * xt[..., 0] / z + cx, fy * xt[..., 1] / z + cy], dim=-1)
        err = torch.sqrt(((obs_uv - pred_uv) ** 2).sum(dim=-1).clamp(min=1e-12))
        vis_binary = (vis > 0.5).float()
        mean_err = (err * vis_binary).sum(dim=0) / vis_binary.sum(dim=0).clamp(min=1.0)
        outlier = mean_err > threshold_px
        n_outlier = int(outlier.sum().item())
        max_reject = int(num_tracks * max_fraction)
        if n_outlier > max_reject and max_reject > 0:
            _, sorted_idx = mean_err.sort(descending=True)
            outlier = torch.zeros(num_tracks, dtype=torch.bool, device=x0.device)
            outlier[sorted_idx[:max_reject]] = True
    return outlier


def _tracks_total_frames(
    tracks_path: Path,
    vis_path: Path,
    mask_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    tracks_raw = np.load(str(tracks_path))
    vis_raw = np.load(str(vis_path))
    tracks_nt2, vis_nt = _normalize_tracks_vis_with_mask_length(
        tracks_raw,
        vis_raw,
        mask_count,
    )
    total = min(int(mask_count), int(tracks_nt2.shape[1]), int(vis_nt.shape[1]))
    return tracks_nt2, vis_nt, total


def _discover_num_frames(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag: Any,
    human_num_frames: int,
) -> int:
    num_frames = human_num_frames
    for state in pag.object_states:
        slug = state.slug
        mask_dir = _resolve_object_mask_dir(dirs["seg_vid"], slug)
        mask_paths = _list_mask_files(mask_dir)
        tracks_path = dirs["object_tracks"] / slug / "tracks.npy"
        vis_path = dirs["object_tracks"] / slug / "visibility.npy"
        if not tracks_path.exists() or not vis_path.exists() or not mask_paths:
            continue
        _, _, total = _tracks_total_frames(tracks_path, vis_path, len(mask_paths))
        num_frames = min(num_frames, total)

    if args.end_frame >= 0:
        num_frames = min(num_frames, int(args.end_frame) + 1)
    if num_frames <= 0:
        raise RuntimeError("No frames available after clipping inputs.")
    return int(num_frames)


def _initial_object_params(
    x0: torch.Tensor,
    tracks_valid: np.ndarray,
    vis_valid: np.ndarray,
    k: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    n_params = max(tracks_valid.shape[1] - 1, 0)
    if args.disable_pnp_init or n_params == 0:
        return (
            torch.zeros(n_params, 3, device=device),
            torch.zeros(n_params, 3, device=device),
            [],
        )

    obs_uv_tm = tracks_valid.transpose(1, 0, 2)
    vis_tm = vis_valid.transpose(1, 0)
    rv_init, tr_init, pnp_info = _pnp_sequential_init(
        x0.cpu().numpy(),
        obs_uv_tm,
        vis_tm,
        k,
        float(args.pnp_ransac_thresh),
    )
    return (
        torch.from_numpy(rv_init).to(device, torch.float32),
        torch.from_numpy(tr_init).to(device, torch.float32),
        pnp_info,
    )


def _load_track_object_state(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    state: Any,
    k: np.ndarray,
    num_frames: int,
    device: torch.device,
) -> ObjectTrackState | None:
    slug = state.slug
    mesh_path = dirs["aligned"] / "meshes" / f"{slug}.ply"
    tracks_path = dirs["object_tracks"] / slug / "tracks.npy"
    vis_path = dirs["object_tracks"] / slug / "visibility.npy"
    mask_dir = _resolve_object_mask_dir(dirs["seg_vid"], slug)

    missing = [
        str(path)
        for path in (mesh_path, tracks_path, vis_path, mask_dir)
        if not path.exists()
    ]
    if missing:
        print(f"  [SKIP] {slug}: missing {', '.join(missing)}")
        return None

    mesh = trimesh.load(str(mesh_path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Cannot load mesh: {mesh_path}")
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertex_colors = _extract_vertex_colors(mesh)

    mask_paths = _list_mask_files(mask_dir)
    masks_np, h_mask, w_mask = _load_mask_stack(
        mask_paths[:num_frames],
        int(args.mask_threshold),
    )
    tracks_nt2, vis_nt, _ = _tracks_total_frames(
        tracks_path,
        vis_path,
        len(mask_paths),
    )
    tracks_nt2 = tracks_nt2[:, :num_frames].astype(np.float32)
    vis_nt = vis_nt[:, :num_frames].astype(np.float32)

    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    mapping = _map_seed_points_to_mesh(
        tracks_nt2[:, 0, :],
        verts,
        faces,
        masks_np[0],
        fx,
        fy,
        cx,
        cy,
        h_mask,
        w_mask,
        device,
        int(args.bin_size),
        float(args.mask_gate_threshold),
    )
    valid = mapping.valid_seed_mask
    tracks_valid = tracks_nt2[valid]
    vis_valid = vis_nt[valid]
    if tracks_valid.shape[0] < int(args.min_valid_tracks):
        raise RuntimeError(
            f"{slug}: too few valid tracks {tracks_valid.shape[0]} "
            f"(need {args.min_valid_tracks})"
        )

    obs_uv = torch.from_numpy(tracks_valid).to(device, torch.float32).permute(1, 0, 2)
    vis_tm = torch.from_numpy(vis_valid).to(device, torch.float32).permute(1, 0)
    masks_t = torch.from_numpy(masks_np).to(device, torch.float32)

    rv_init, tr_init, pnp_info = _initial_object_params(
        mapping.points_cv,
        tracks_valid,
        vis_valid,
        k,
        args,
        device,
    )
    if pnp_info:
        pnp_ok = sum(1 for item in pnp_info if item["pnp_ok"])
        print(f"  {slug}: PnP init {pnp_ok}/{len(pnp_info)} frames OK")

    if float(args.outlier_reproj_thresh_px) > 0.0 and rv_init.numel() > 0:
        outlier_mask = _identify_outlier_tracks(
            mapping.points_cv,
            obs_uv,
            vis_tm,
            rv_init,
            tr_init,
            fx,
            fy,
            cx,
            cy,
            float(args.outlier_reproj_thresh_px),
            float(args.outlier_max_fraction),
        )
        n_outlier = int(outlier_mask.sum().item())
        if n_outlier > 0:
            vis_tm[:, outlier_mask] = 0.0
            print(f"  {slug}: removed {n_outlier} global track outliers")

    return ObjectTrackState(
        name=state.name,
        slug=slug,
        is_translational=bool(state.is_translational),
        is_rotational=bool(state.is_rotational),
        mesh=mesh,
        verts=verts,
        faces=faces,
        vertex_colors=vertex_colors,
        x0=mapping.points_cv,
        obs_uv=obs_uv,
        vis=vis_tm,
        masks=masks_t,
        rotvecs=torch.nn.Parameter(rv_init.clone()),
        trans=torch.nn.Parameter(tr_init.clone()),
        raw_scale_delta=torch.nn.Parameter(torch.zeros(1, device=device)),
        num_input_tracks=int(tracks_nt2.shape[0]),
        num_valid_seed_tracks=int(tracks_valid.shape[0]),
        num_dropped_invalid_face=int(mapping.invalid_face_count),
        num_dropped_outside_mask0=int(mapping.outside_mask0_count),
        num_dropped_nonfinite_seed=int(mapping.nonfinite_seed_count),
        pnp_info=pnp_info,
    )


def bounded_log_scale_delta(raw_value: torch.Tensor, max_log_scale_delta: float) -> torch.Tensor:
    return math.fabs(max_log_scale_delta) * torch.tanh(raw_value.squeeze())


def _scale_from_raw(raw_scale_delta: torch.Tensor, max_log_scale_delta: float) -> torch.Tensor:
    return torch.exp(bounded_log_scale_delta(raw_scale_delta, max_log_scale_delta))


def _l2_loss(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ((x - y) ** 2).sum(dim).mean()


def simple_static_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] < 2:
        return x.new_tensor(0.0)
    return _l2_loss(x[..., 1:, :], x[..., :-1, :])


def simple_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] < 3:
        return x.new_tensor(0.0)
    return _l2_loss(x[..., 1:-1, :], 0.5 * (x[..., :-2, :] + x[..., 2:, :]))


def rotation_static_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    return simple_static_loss(roma.rotmat_to_rotvec(x))


def rotation_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 3:
        return x.new_tensor(0.0)
    interp = roma.rotmat_slerp(
        x[:-2],
        x[2:],
        torch.tensor(0.5, device=x.device, dtype=x.dtype),
    )
    diff = roma.rotmat_geodesic_distance(interp, x[1:-1])
    return (diff**2).mean()


def _shared_object_motion_losses(
    trans: torch.Tensor,
    rot_mats: torch.Tensor,
    scale: torch.Tensor,
    is_translational: bool,
    is_rotational: bool,
) -> dict[str, torch.Tensor]:
    if is_translational:
        smooth_trans = simple_smoothness_loss(trans)
    else:
        smooth_trans = simple_static_loss(trans) * 10.0

    if is_rotational:
        smooth_rot = rotation_smoothness_loss(rot_mats)
    else:
        smooth_rot = rotation_static_loss(rot_mats) * 10.0

    scale_prior = F.relu(torch.abs(scale - 1.0) - 0.1).reshape(())
    return {
        "smooth_trans": smooth_trans,
        "smooth_rot": smooth_rot,
        "scale_prior": scale_prior,
    }


def _full_T(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    return _build_T_matrices(rotvecs, trans, num_frames, device)


def _tracking_object_loss(
    obj: ObjectTrackState,
    args: argparse.Namespace,
    k: np.ndarray,
    huber_delta: float,
) -> dict[str, torch.Tensor]:
    device = obj.x0.device
    num_frames = obj.obs_uv.shape[0]
    T_mats = _full_T(obj.rotvecs, obj.trans, num_frames, device)
    scale = _scale_from_raw(obj.raw_scale_delta, float(args.max_log_scale_delta))
    points = obj.x0 * scale
    rot = T_mats[:, :3, :3]
    trans = T_mats[:, :3, 3]
    xt = torch.einsum("tij,mj->tmi", rot, points) + trans.unsqueeze(1)

    z = xt[..., 2]
    z_valid = z > 1e-6
    z_safe = torch.where(z_valid, z, torch.ones_like(z))
    pred_u = float(k[0, 0]) * xt[..., 0] / z_safe + float(k[0, 2])
    pred_v = float(k[1, 1]) * xt[..., 1] / z_safe + float(k[1, 2])
    pred_uv = torch.stack([pred_u, pred_v], dim=-1)

    finite_obs = torch.isfinite(obj.obs_uv).all(dim=-1)
    mask_vals = _sample_masks_bilinear_seq(obj.masks, obj.obs_uv)
    mask_gate = mask_vals >= float(args.mask_gate_threshold)
    vis_w = (
        obj.vis
        if float(args.visibility_threshold) <= 0.0
        else torch.where(
            obj.vis >= float(args.visibility_threshold),
            obj.vis,
            torch.zeros_like(obj.vis),
        )
    )
    weights = vis_w * mask_gate.float() * finite_obs.float() * z_valid.float()
    r2 = ((obj.obs_uv - pred_uv) ** 2).sum(dim=-1)
    robust = _huber_on_squared(r2, float(huber_delta))
    reproj = (weights * robust).sum() / weights.sum().clamp(min=1.0)

    shared = _shared_object_motion_losses(
        trans,
        rot,
        scale,
        obj.is_translational,
        obj.is_rotational,
    )

    return {
        "reproj": reproj,
        "smooth_trans": shared["smooth_trans"],
        "smooth_rot": shared["smooth_rot"],
        "scale_prior": shared["scale_prior"],
        "r2": r2,
        "weights": weights,
        "pred_uv": pred_uv,
        "T_mats": T_mats,
        "scale": scale.reshape(()),
    }


def _compute_stage1_loss(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
    huber_delta: float,
) -> TrackingLossBundle:
    per_object = {
        slug: _tracking_object_loss(obj, args, k, huber_delta)
        for slug, obj in objects.items()
    }
    reproj = torch.stack([item["reproj"] for item in per_object.values()]).mean()
    smooth_trans = torch.stack(
        [item["smooth_trans"] for item in per_object.values()]
    ).mean()
    smooth_rot = torch.stack(
        [item["smooth_rot"] for item in per_object.values()]
    ).mean()
    scale_prior = torch.stack(
        [item["scale_prior"] for item in per_object.values()]
    ).mean()
    total = (
        float(args.lambda_track_reproj) * reproj
        + float(args.lambda_smooth_trans) * smooth_trans
        + float(args.lambda_smooth_rot) * smooth_rot
        + float(args.lambda_scale) * scale_prior
    )
    return TrackingLossBundle(
        total=total,
        track_reproj=reproj,
        smooth_trans=smooth_trans,
        smooth_rot=smooth_rot,
        scale_prior=scale_prior,
        per_object=per_object,
    )


def _stage1_row(iteration: int, bundle: TrackingLossBundle, args: argparse.Namespace) -> dict[str, Any]:
    active_pairs = sum(
        int(item["weights"].detach().gt(0.0).sum().item())
        for item in bundle.per_object.values()
    )
    return {
        "iter": int(iteration),
        "total": float(bundle.total.detach().item()),
        "track_reproj_raw": float(bundle.track_reproj.detach().item()),
        "track_reproj_scaled": float(
            float(args.lambda_track_reproj) * bundle.track_reproj.detach().item()
        ),
        "smooth_trans_raw": float(bundle.smooth_trans.detach().item()),
        "smooth_trans_scaled": float(
            float(args.lambda_smooth_trans) * bundle.smooth_trans.detach().item()
        ),
        "smooth_rot_raw": float(bundle.smooth_rot.detach().item()),
        "smooth_rot_scaled": float(
            float(args.lambda_smooth_rot) * bundle.smooth_rot.detach().item()
        ),
        "scale_prior_raw": float(bundle.scale_prior.detach().item()),
        "scale_prior_scaled": float(
            float(args.lambda_scale) * bundle.scale_prior.detach().item()
        ),
        "active_pairs": active_pairs,
    }


def _retrim_tracks(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
) -> int:
    trimmed_total = 0
    with torch.no_grad():
        for obj in objects.values():
            item = _tracking_object_loss(
                obj,
                args,
                k,
                huber_delta=float(args.huber_delta_px),
            )
            err = torch.sqrt(item["r2"].clamp(min=1e-12))
            for frame_idx in range(obj.vis.shape[0]):
                active = obj.vis[frame_idx] > 0.0
                if int(active.sum().item()) < 20:
                    continue
                threshold = torch.quantile(
                    err[frame_idx, active],
                    float(args.retrim_percentile) / 100.0,
                )
                to_zero = active & (err[frame_idx] > threshold)
                n_zero = int(to_zero.sum().item())
                if n_zero > 0:
                    obj.vis[frame_idx, to_zero] = 0.0
                    trimmed_total += n_zero
    return trimmed_total


def _run_stage1(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
) -> list[dict[str, Any]]:
    params: list[torch.nn.Parameter] = []
    for obj in objects.values():
        params.extend([obj.rotvecs, obj.trans, obj.raw_scale_delta])
    if not params:
        raise RuntimeError("No object parameters to optimize.")

    iter_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    opt = torch.optim.Adam(params, lr=float(args.stage1_lr))
    scheduler = None
    if args.stage1_lr_schedule == "cosine" and args.stage1_iters > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(args.stage1_iters),
            eta_min=float(args.stage1_lr) / 20.0,
        )

    huber_target = float(args.huber_delta_px)
    huber_start = huber_target * 3.0 if args.graduated_huber else huber_target

    print(f"\n[Stage 1] object tracking init: {args.stage1_iters} Adam iterations")
    if args.stage1_iters <= 0:
        with torch.no_grad():
            bundle = _compute_stage1_loss(objects, args, k, huber_target)
        iter_rows.append(_stage1_row(0, bundle, args))
        return iter_rows

    for iteration in range(1, int(args.stage1_iters) + 1):
        if args.graduated_huber:
            frac = min(iteration / max(args.stage1_iters * 0.5, 1), 1.0)
            huber = huber_start + (huber_target - huber_start) * frac
        else:
            huber = huber_target

        opt.zero_grad(set_to_none=True)
        bundle = _compute_stage1_loss(objects, args, k, huber)
        if not torch.isfinite(bundle.total):
            raise RuntimeError(f"Non-finite Stage-1 loss at iter {iteration}")
        bundle.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
        opt.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            eval_bundle = _compute_stage1_loss(objects, args, k, huber_target)
            cur = float(eval_bundle.total.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {
                    slug: {
                        "rot": obj.rotvecs.detach().clone(),
                        "trans": obj.trans.detach().clone(),
                        "scale": obj.raw_scale_delta.detach().clone(),
                    }
                    for slug, obj in objects.items()
                }

            if (
                args.retrim_interval > 0
                and iteration % int(args.retrim_interval) == 0
                and iteration < int(args.stage1_iters)
            ):
                n_trimmed = _retrim_tracks(objects, args, k)
                if n_trimmed > 0:
                    print(
                        f"  Stage1 retrim@{iteration}: zeroed {n_trimmed} "
                        "track-frame pairs"
                    )

            if (
                args.debug_save_interval <= 1
                or iteration % int(args.debug_save_interval) == 0
                or iteration == int(args.stage1_iters)
            ):
                iter_rows.append(_stage1_row(iteration, eval_bundle, args))

            if (
                args.log_every > 0
                and (iteration % int(args.log_every) == 0 or iteration == args.stage1_iters)
            ):
                print(
                    f"  stage1 {iteration:05d} total={eval_bundle.total.item():.6f} "
                    f"track={eval_bundle.track_reproj.item():.6f} "
                    f"smooth_t={eval_bundle.smooth_trans.item():.6f} "
                    f"smooth_r={eval_bundle.smooth_rot.item():.6f} "
                    f"scale={eval_bundle.scale_prior.item():.6f}"
                )

    if best_state:
        with torch.no_grad():
            for slug, state in best_state.items():
                objects[slug].rotvecs.copy_(state["rot"])
                objects[slug].trans.copy_(state["trans"])
                objects[slug].raw_scale_delta.copy_(state["scale"])
    return iter_rows


def _full_delta_vars(
    objects: dict[str, ObjectTrackState],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    full_rot: dict[str, torch.Tensor] = {}
    full_trans: dict[str, torch.Tensor] = {}
    raw_scales: dict[str, torch.Tensor] = {}
    for slug, obj in objects.items():
        zero_rot = torch.zeros(1, 3, device=obj.rotvecs.device, dtype=obj.rotvecs.dtype)
        zero_trans = torch.zeros(1, 3, device=obj.trans.device, dtype=obj.trans.dtype)
        full_rot[slug] = torch.cat([zero_rot, obj.rotvecs], dim=0)
        full_trans[slug] = torch.cat([zero_trans, obj.trans], dim=0)
        raw_scales[slug] = obj.raw_scale_delta
    return full_rot, full_trans, raw_scales


def _linear_weight(start: float, end: float, step_id: int, n_steps: int) -> float:
    if n_steps <= 0:
        return float(start)
    return float(start) + (float(end) - float(start)) * float(step_id) / float(n_steps)


def _stage2_weights(args: argparse.Namespace, iteration: int) -> dict[str, float]:
    total_iters = max(int(args.stage2_iters), 1)
    return {
        "tracking": 0.0,
        "object_cd2d": float(args.object_cd2d_weight),
        "object_part_cd2d": float(args.object_part_cd2d_weight),
        "object_smooth_trans": float(args.object_smooth_trans_weight),
        "object_smooth_rot": float(args.object_smooth_rot_weight),
        "object_scale": float(args.object_scale_weight),
        "intersect": _linear_weight(
            float(args.intersect_weight_start),
            float(args.intersect_weight_end),
            iteration,
            total_iters,
        ),
        "nocontact": float(args.nocontact_weight),
        "contact_drift": _linear_weight(
            float(args.contact_drift_weight_start),
            float(args.contact_drift_weight_end),
            iteration,
            total_iters,
        ),
    }


def get_scaled_loss_terms(result: LossResult) -> dict[str, torch.Tensor]:
    return {
        key: getattr(result, key) * float(result.weights[key])
        for key in LOSS_TERM_KEYS
    }


def compose_T_sequence(rotvecs: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    R = axis_angle_to_matrix(rotvecs)
    T = torch.zeros((rotvecs.shape[0], 4, 4), dtype=rotvecs.dtype, device=rotvecs.device)
    T[:, :3, :3] = R
    T[:, :3, 3] = trans
    T[:, 3, 3] = 1.0
    return T


def apply_similarity_sequence(points: torch.Tensor, T_seq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = points * scale
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(scaled.unsqueeze(0), R.transpose(1, 2)) + t[:, None, :]


def apply_inverse_similarity_sequence(
    points_seq: torch.Tensor,
    T_seq: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(points_seq - t[:, None, :], R) / scale


def project_points_with_intrinsics(points: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    px = points[..., 0] * fx / points[..., 2] + cx
    py = points[..., 1] * fy / points[..., 2] + cy
    return torch.stack([px, py, points[..., 2]], dim=-1)


def query_sdf(sdf_grid: SDFGrid, points: torch.Tensor) -> torch.Tensor:
    shape = points.shape[:-1]
    pts = points.reshape(1, -1, 3)
    normalised = (
        (pts - sdf_grid.bbox_min)
        / (sdf_grid.bbox_max - sdf_grid.bbox_min)
        * 2.0
        - 1.0
    )
    grid = normalised[:, :, [2, 1, 0]].view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        sdf_grid.sdf_volume,
        grid,
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(*shape)


def pcd_distance(
    p1: torch.Tensor | None,
    p2: torch.Tensor | None,
    reduction: str = "min",
    error_func=None,
) -> torch.Tensor | None:
    if p1 is None or p2 is None:
        return None
    nnres = knn_points(p1=p1, p2=p2, norm=2, K=1)
    nndists = nnres.dists[..., 0]
    if error_func is not None:
        nndists = error_func(nndists)
    if reduction == "min":
        return torch.min(nndists, dim=1)[0]
    if reduction == "mean":
        return torch.mean(nndists, dim=1)
    raise RuntimeError(f"Unknown reduction: {reduction}")


def _get_human_device_and_num_frames(humans: dict[str, HumanData]) -> tuple[torch.device, int]:
    if not humans:
        raise RuntimeError("No humans loaded for optimisation.")
    first_human = humans[next(iter(humans))]
    return first_human.base_verts.device, first_human.base_verts.shape[0]


def _concat_all_human_points(humans: dict[str, HumanData]) -> torch.Tensor:
    return torch.cat([humans[slug].base_verts for slug in sorted(humans)], dim=1)


def _build_effective_object_state(
    delta_rotvecs: dict[str, torch.Tensor],
    delta_trans: dict[str, torch.Tensor],
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    obj_keys: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    eff_T: dict[str, torch.Tensor] = {}
    eff_rot_mats: dict[str, torch.Tensor] = {}
    eff_trans: dict[str, torch.Tensor] = {}
    eff_scales: dict[str, torch.Tensor] = {}
    for slug in obj_keys:
        delta_T = compose_T_sequence(delta_rotvecs[slug], delta_trans[slug])
        T_eff = torch.matmul(objects[slug].tracked_poses_torch, delta_T)
        eff_T[slug] = T_eff
        eff_rot_mats[slug] = T_eff[:, :3, :3]
        eff_trans[slug] = T_eff[:, :3, 3]
        eff_scales[slug] = _scale_from_raw(raw_scale_deltas[slug], float(args.max_log_scale_delta))
    return eff_T, eff_rot_mats, eff_trans, eff_scales


def _get_reduction(nodes: tuple) -> str:
    for node in nodes:
        if node.is_human and node.part_name.split(" ")[-1] in ("hand", "foot", "hips"):
            return "mean"
    return "min"


def _has_interaction(
    interaction_edges: list[InteractionEdge],
    human_name: str,
    human_part: str,
    object_name: str,
    object_part: str,
) -> bool:
    for edge in interaction_edges:
        nodes = (edge.node_a, edge.node_b)
        has_human = any(
            node.is_human and node.entity_name == human_name and node.part_name == human_part
            for node in nodes
        )
        has_object = any(
            (not node.is_human)
            and node.entity_name == object_name
            and node.part_name == object_part
            for node in nodes
        )
        if has_human and has_object:
            return True
    return False


def _build_human_part_getter(
    humans: dict[str, HumanData],
    device: torch.device,
    prefer_contact_regions: bool = True,
):
    human_part_cache: dict[tuple[str, ...], torch.Tensor | None] = {}

    def get_human_part_points(human_slug: str | None, part_name: str | list[str]) -> torch.Tensor | None:
        if human_slug is None or human_slug not in humans:
            return None
        human_data = humans[human_slug]
        if isinstance(part_name, str):
            key = (human_slug, part_name)
            if key not in human_part_cache:
                if prefer_contact_regions and part_name in human_data.contact_part_points:
                    human_part_cache[key] = human_data.contact_part_points[part_name]
                else:
                    human_part_cache[key] = human_data.part_points.get(part_name)
            return human_part_cache[key]

        key = tuple([human_slug] + sorted(part_name))
        if key not in human_part_cache:
            part_ids = []
            for name in part_name:
                if prefer_contact_regions and name in human_data.contact_part_vert_ids:
                    part_ids.append(human_data.contact_part_vert_ids[name])
                elif name in human_data.part_vert_ids:
                    part_ids.append(human_data.part_vert_ids[name])
            if not part_ids:
                human_part_cache[key] = None
            else:
                merged = np.unique(np.concatenate(part_ids, axis=0))
                index = torch.from_numpy(merged.astype(np.int64)).to(device)
                human_part_cache[key] = human_data.base_verts.index_select(1, index)
        return human_part_cache[key]

    return get_human_part_points


def _one_way_2d_chamfer_diagnostic(
    observed_points,
    model_points_world: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    num_frames = model_points_world.shape[0]
    per_frame = torch.zeros(num_frames, device=model_points_world.device)
    if observed_points is None:
        return None, per_frame
    projected = project_points_with_intrinsics(model_points_world, k)
    frame_losses: list[torch.Tensor] = []
    for frame_idx in range(num_frames):
        obs_len = int(observed_points.lengths[frame_idx].item())
        if obs_len == 0:
            continue
        obs_pts = observed_points.points[frame_idx, :obs_len, :].unsqueeze(0)
        model_pts = projected[frame_idx, :, :2].unsqueeze(0)
        cdist = pcd_distance(obs_pts, model_pts, reduction="mean")
        if cdist is None:
            continue
        frame_loss = cdist.squeeze(0)
        per_frame[frame_idx] = frame_loss
        frame_losses.append(frame_loss)
    if not frame_losses:
        return None, per_frame
    return torch.stack(frame_losses, dim=0).mean(), per_frame


def _compute_intersect_diagnostic(
    world_points: torch.Tensor,
    obj_data: ObjectData,
    obj_T_seq: torch.Tensor,
    obj_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = world_points.shape[0]
    per_frame = torch.zeros(num_frames, device=world_points.device)
    if obj_data.sdf_grid is None:
        return world_points.new_tensor(0.0), per_frame
    pts_canon = apply_inverse_similarity_sequence(world_points, obj_T_seq, obj_scale)
    sdf_vals = query_sdf(obj_data.sdf_grid, pts_canon)
    intersects = F.relu(-sdf_vals)
    icount = (intersects > 0).sum()
    if icount.item() == 0:
        return world_points.new_tensor(0.0), per_frame
    flat = intersects.reshape(num_frames, -1)
    frame_counts = (flat > 0).sum(dim=1)
    valid = frame_counts > 0
    per_frame[valid] = flat[valid].sum(dim=1) / frame_counts[valid].to(flat.dtype)
    return intersects.sum() / icount, per_frame


def _compute_contact_losses(
    context: ProblemContext,
    args: argparse.Namespace,
    iteration: int,
    shared_losses: TrackingLossBundle,
) -> LossResult:
    device, _ = _get_human_device_and_num_frames(context.humans)
    weights = _stage2_weights(args, iteration)
    eff_T = {
        slug: shared_losses.per_object[slug]["T_mats"]
        for slug in context.obj_keys
    }
    eff_scales = {
        slug: shared_losses.per_object[slug]["scale"]
        for slug in context.obj_keys
    }

    object_points_cache: dict[tuple[str, str | None], torch.Tensor] = {}

    def get_object_points(slug: str, part_name: str | None = None) -> torch.Tensor:
        key = (slug, part_name)
        if key not in object_points_cache:
            od = context.objects[slug]
            base_points = od.part_sampled_points.get(part_name, od.sampled_points) if part_name else od.sampled_points
            object_points_cache[key] = apply_similarity_sequence(base_points, eff_T[slug], eff_scales[slug])
        return object_points_cache[key]

    object_cd2d_values: list[torch.Tensor] = []
    for slug in context.obj_keys:
        scalar, _ = _one_way_2d_chamfer_diagnostic(
            context.objects[slug].mask_points_2d,
            get_object_points(slug),
            context.k_torch,
        )
        if scalar is not None:
            object_cd2d_values.append(scalar)
    loss_object_cd2d = (
        torch.stack(object_cd2d_values, dim=0).mean()
        if object_cd2d_values
        else torch.tensor(0.0, device=device)
    )

    object_part_cd2d_values: list[torch.Tensor] = []
    for slug in context.obj_keys:
        object_frame_values: list[torch.Tensor] = []
        for part_name, packed_points in context.objects[slug].part_mask_points_2d.items():
            scalar, _ = _one_way_2d_chamfer_diagnostic(
                packed_points,
                get_object_points(slug, part_name),
                context.k_torch,
            )
            if scalar is not None:
                object_frame_values.append(scalar)
        if object_frame_values:
            object_part_cd2d_values.append(torch.stack(object_frame_values, dim=0).mean())
    loss_object_part_cd2d = (
        torch.stack(object_part_cd2d_values, dim=0).mean()
        if object_part_cd2d_values
        else torch.tensor(0.0, device=device)
    )

    all_human_points = _concat_all_human_points(context.humans)
    intersect_values: list[torch.Tensor] = []
    for slug in context.obj_keys:
        scalar, _ = _compute_intersect_diagnostic(
            all_human_points,
            context.objects[slug],
            eff_T[slug],
            eff_scales[slug],
        )
        if scalar.item() > 0.0:
            intersect_values.append(scalar)
    loss_intersect = (
        torch.stack(intersect_values, dim=0).mean()
        if intersect_values
        else torch.tensor(0.0, device=device)
    )

    get_human_part_points = _build_human_part_getter(context.humans, device)

    def to_canonical(slug: str, points_world: torch.Tensor) -> torch.Tensor:
        return apply_inverse_similarity_sequence(points_world, eff_T[slug], eff_scales[slug])

    nocontact_values: list[torch.Tensor] = []
    contact_drift_values: list[torch.Tensor] = []
    visited: set[tuple[str, str, str, str]] = set()
    for edge in context.interaction_edges:
        nodes = [edge.node_a, edge.node_b]
        has_hpart = nodes[0].is_human or nodes[1].is_human
        if has_hpart and not nodes[0].is_human:
            nodes = [nodes[1], nodes[0]]
        reduction = _get_reduction((nodes[0], nodes[1]))
        dedup_key = (nodes[0].entity_name, nodes[0].part_name, nodes[1].entity_name, nodes[1].part_name)
        if dedup_key in visited:
            continue

        pdists = None
        pcano = None
        if has_hpart:
            human_node = nodes[0]
            object_node = nodes[1]
            hname = human_node.entity_name
            hpart = human_node.part_name.split(" ")[-1]
            oname = object_node.entity_name
            opart = object_node.part_name
            object_points = get_object_points(object_node.object_slug, object_node.resolved_part_name)
            if hpart in ("head", "hips"):
                visited.add((hname, hpart, oname, opart))
                visited.add((oname, opart, hname, hpart))
                human_points = get_human_part_points(human_node.human_slug, hpart)
                pdists = pcd_distance(human_points, object_points, reduction=reduction)
                if human_points is not None:
                    pcano = to_canonical(object_node.object_slug, human_points)
            else:
                visited.add((hname, f"left {hpart}", oname, opart))
                visited.add((hname, f"right {hpart}", oname, opart))
                visited.add((oname, opart, hname, f"left {hpart}"))
                visited.add((oname, opart, hname, f"right {hpart}"))
                has_left = _has_interaction(context.interaction_edges, hname, f"left {hpart}", oname, opart)
                has_right = _has_interaction(context.interaction_edges, hname, f"right {hpart}", oname, opart)
                if has_left and has_right:
                    human_points = get_human_part_points(human_node.human_slug, [f"left {hpart}", f"right {hpart}"])
                    pdists = pcd_distance(human_points, object_points, reduction=reduction)
                    if human_points is not None:
                        pcano = to_canonical(object_node.object_slug, human_points)
                else:
                    human_points_left = get_human_part_points(human_node.human_slug, f"left {hpart}")
                    human_points_right = get_human_part_points(human_node.human_slug, f"right {hpart}")
                    pdists_left = pcd_distance(human_points_left, object_points, reduction=reduction)
                    pdists_right = pcd_distance(human_points_right, object_points, reduction=reduction)
                    if pdists_left is None or pdists_right is None:
                        continue
                    if edge.is_continuous:
                        sel_left = pdists_left.mean().item() < pdists_right.mean().item()
                    else:
                        sel_left = pdists_left.min().item() < pdists_right.min().item()
                    if sel_left:
                        pdists = pdists_left
                        pcano = to_canonical(object_node.object_slug, human_points_left)
                    else:
                        pdists = pdists_right
                        pcano = to_canonical(object_node.object_slug, human_points_right)
        else:
            visited.add((nodes[0].entity_name, nodes[0].part_name, nodes[1].entity_name, nodes[1].part_name))
            visited.add((nodes[1].entity_name, nodes[1].part_name, nodes[0].entity_name, nodes[0].part_name))
            part_pcds = [
                get_object_points(nodes[0].object_slug, nodes[0].resolved_part_name),
                get_object_points(nodes[1].object_slug, nodes[1].resolved_part_name),
            ]
            part_diags = [
                torch.linalg.norm(ppcd[0, :, :].max(dim=0)[0] - ppcd[0, :, :].min(dim=0)[0]).item()
                for ppcd in part_pcds
            ]
            if part_diags[0] < part_diags[1]:
                pdists = pcd_distance(part_pcds[0], part_pcds[1], reduction=reduction)
            else:
                pdists = pcd_distance(part_pcds[1], part_pcds[0], reduction=reduction)
            pcano = to_canonical(nodes[0].object_slug, part_pcds[1])

        if pdists is None or pcano is None:
            continue
        nocontact_values.append(pdists.mean() if edge.is_continuous else pdists.min())
        pcano_seq = pcano.permute(1, 0, 2).contiguous()
        contact_drift_values.append(
            simple_static_loss(pcano_seq) if edge.is_rel_static else simple_smoothness_loss(pcano_seq)
        )

    loss_nocontact = (
        torch.stack(nocontact_values, dim=0).mean()
        if nocontact_values
        else torch.tensor(0.0, device=device)
    )
    loss_contact_drift = (
        torch.stack(contact_drift_values, dim=0).mean()
        if contact_drift_values
        else torch.tensor(0.0, device=device)
    )
    zero = torch.tensor(0.0, device=device)
    total = (
        loss_object_cd2d * weights["object_cd2d"]
        + loss_object_part_cd2d * weights["object_part_cd2d"]
        + shared_losses.smooth_trans * weights["object_smooth_trans"]
        + shared_losses.smooth_rot * weights["object_smooth_rot"]
        + shared_losses.scale_prior * weights["object_scale"]
        + loss_intersect * weights["intersect"]
        + loss_nocontact * weights["nocontact"]
        + loss_contact_drift * weights["contact_drift"]
    )
    return LossResult(
        total=total,
        tracking=zero,
        object_cd2d=loss_object_cd2d,
        object_part_cd2d=loss_object_part_cd2d,
        object_smooth_trans=shared_losses.smooth_trans,
        object_smooth_rot=shared_losses.smooth_rot,
        object_scale=shared_losses.scale_prior,
        intersect=loss_intersect,
        nocontact=loss_nocontact,
        contact_drift=loss_contact_drift,
        weights=weights,
    )


def _compute_stage2_loss(
    objects: dict[str, ObjectTrackState],
    context: ProblemContext,
    args: argparse.Namespace,
    iteration: int,
) -> Stage2LossBundle:
    track_bundle = _compute_stage1_loss(
        objects,
        args,
        context.k,
        float(args.huber_delta_px),
    )
    contact_result = _compute_contact_losses(context, args, iteration, track_bundle)
    total = float(args.lambda_track_reproj) * track_bundle.track_reproj
    total = total + contact_result.total
    return Stage2LossBundle(
        total=total,
        track_reproj=track_bundle.track_reproj,
        contact_result=contact_result,
    )


def _stage2_row(iteration: int, bundle: Stage2LossBundle, args: argparse.Namespace) -> dict[str, Any]:
    scaled_terms = get_scaled_loss_terms(bundle.contact_result)
    row: dict[str, Any] = {
        "iter": int(iteration),
        "total": float(bundle.total.detach().item()),
        "track_reproj_weight": float(args.lambda_track_reproj),
        "track_reproj_raw": float(bundle.track_reproj.detach().item()),
        "track_reproj_scaled": float(
            float(args.lambda_track_reproj) * bundle.track_reproj.detach().item()
        ),
    }
    for key in (
        "object_cd2d",
        "object_part_cd2d",
        "object_smooth_trans",
        "object_smooth_rot",
        "object_scale",
        "intersect",
        "nocontact",
        "contact_drift",
    ):
        row[f"{key}_weight"] = float(bundle.contact_result.weights[key])
        row[f"{key}_raw"] = float(getattr(bundle.contact_result, key).detach().item())
        row[f"{key}_scaled"] = float(scaled_terms[key].detach().item())
    return row


def compute_final_loss_diagnostics(
    delta_rotvecs: dict[str, torch.Tensor],
    delta_trans: dict[str, torch.Tensor],
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    humans: dict[str, HumanData],
    interaction_edges: list[InteractionEdge],
    obj_keys: list[str],
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
    k: torch.Tensor,
    width: int,
    height: int,
) -> DiagnosticLossResult:
    del total_iters, width, height
    device, num_frames = _get_human_device_and_num_frames(humans)
    per_object: dict[str, dict[str, torch.Tensor]] = {}
    smooth_trans_values: list[torch.Tensor] = []
    smooth_rot_values: list[torch.Tensor] = []
    scale_values: list[torch.Tensor] = []
    for slug in obj_keys:
        T_mats = compose_T_sequence(delta_rotvecs[slug], delta_trans[slug])
        scale = _scale_from_raw(raw_scale_deltas[slug], float(args.max_log_scale_delta)).reshape(())
        shared = _shared_object_motion_losses(
            T_mats[:, :3, 3],
            T_mats[:, :3, :3],
            scale,
            objects[slug].state.is_translational,
            objects[slug].state.is_rotational,
        )
        per_object[slug] = {"T_mats": T_mats, "scale": scale, **shared}
        smooth_trans_values.append(shared["smooth_trans"])
        smooth_rot_values.append(shared["smooth_rot"])
        scale_values.append(shared["scale_prior"])

    zero = torch.tensor(0.0, device=device)
    shared_bundle = TrackingLossBundle(
        total=zero,
        track_reproj=zero,
        smooth_trans=torch.stack(smooth_trans_values).mean() if smooth_trans_values else zero,
        smooth_rot=torch.stack(smooth_rot_values).mean() if smooth_rot_values else zero,
        scale_prior=torch.stack(scale_values).mean() if scale_values else zero,
        per_object=per_object,
    )
    context = ProblemContext(
        dirs={},
        out_dir=Path("."),
        pag_path=Path("."),
        smpl_seg_path=Path("."),
        intr_path=Path("."),
        device=device,
        k=np.eye(3, dtype=np.float32),
        k_torch=k,
        width=0,
        height=0,
        num_frames=num_frames,
        pag=PAG([], [], [], []),
        humans=humans,
        human_keys=list(humans),
        objects=objects,
        obj_keys=obj_keys,
        interaction_edges=interaction_edges,
    )
    sequence = _compute_contact_losses(context, args, iteration, shared_bundle)
    per_frame_raw = {
        key: torch.zeros(num_frames, device=device)
        for key in FRAME_DIAGNOSTIC_TERM_KEYS
    }
    return DiagnosticLossResult(
        sequence=sequence,
        per_frame_raw=per_frame_raw,
        global_raw={"object_scale": sequence.object_scale.detach().clone()},
    )


def _run_stage2(
    objects: dict[str, ObjectTrackState],
    context: ProblemContext,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    iter_rows: list[dict[str, Any]] = []
    if int(args.stage2_iters) <= 0:
        print("\n[Stage 2] skipped")
        return iter_rows

    params: list[torch.nn.Parameter] = []
    for obj in objects.values():
        params.extend([obj.rotvecs, obj.trans, obj.raw_scale_delta])

    opt = torch.optim.Adam(params, lr=float(args.stage2_lr))
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}

    print(f"\n[Stage 2] contact refinement: {args.stage2_iters} Adam iterations")
    for iteration in range(1, int(args.stage2_iters) + 1):
        opt.zero_grad(set_to_none=True)
        bundle = _compute_stage2_loss(objects, context, args, iteration)
        if not torch.isfinite(bundle.total):
            raise RuntimeError(f"Non-finite Stage-2 loss at iter {iteration}")
        bundle.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(args.grad_clip_norm))
        opt.step()

        with torch.no_grad():
            eval_bundle = _compute_stage2_loss(objects, context, args, iteration)
            cur = float(eval_bundle.total.item())
            if cur < best_loss:
                best_loss = cur
                best_state = {
                    slug: {
                        "rot": obj.rotvecs.detach().clone(),
                        "trans": obj.trans.detach().clone(),
                        "scale": obj.raw_scale_delta.detach().clone(),
                    }
                    for slug, obj in objects.items()
                }
            if (
                args.debug_save_interval <= 1
                or iteration % int(args.debug_save_interval) == 0
                or iteration == int(args.stage2_iters)
            ):
                iter_rows.append(_stage2_row(iteration, eval_bundle, args))
            if (
                args.log_every > 0
                and (iteration % int(args.log_every) == 0 or iteration == args.stage2_iters)
            ):
                print(
                    f"  stage2 {iteration:05d} total={eval_bundle.total.item():.6f} "
                    f"track={eval_bundle.track_reproj.item():.6f} "
                    f"intersect={eval_bundle.contact_result.intersect.item():.6f} "
                    f"nocontact={eval_bundle.contact_result.nocontact.item():.6f} "
                    f"drift={eval_bundle.contact_result.contact_drift.item():.6f}"
                )

    if best_state:
        with torch.no_grad():
            for slug, state in best_state.items():
                objects[slug].rotvecs.copy_(state["rot"])
                objects[slug].trans.copy_(state["trans"])
                objects[slug].raw_scale_delta.copy_(state["scale"])
    return iter_rows


def _identity_poses(num_frames: int) -> np.ndarray:
    poses = np.tile(np.eye(4, dtype=np.float32)[None, :, :], (num_frames, 1, 1))
    return poses


def _build_contact_context(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag_path: Path,
    smpl_seg_path: Path,
    k: np.ndarray,
    humans: dict[str, HumanData],
    human_keys: list[str],
    objects: dict[str, ObjectTrackState],
    num_frames: int,
    device: torch.device,
) -> ProblemContext:
    pag = _parse_pag(pag_path)
    body_seg, _ = _load_smpl_body_and_contact_seg(smpl_seg_path)
    width, height = _infer_image_size(dirs)
    k_torch = torch.from_numpy(k.astype(np.float32)).to(device)

    object_data: dict[str, ObjectData] = {}
    obj_keys: list[str] = []
    for idx, state in enumerate(pag.object_states):
        slug = state.slug
        if slug not in objects:
            continue
        track_obj = objects[slug]
        faces_i32 = track_obj.faces.astype(np.int32)

        try:
            part_segments = _load_object_part_segments(dirs["seg_obj"], slug, faces_i32)
        except FileNotFoundError:
            print(f"  [WARN] {slug}: no part segmentation, using whole mesh")
            part_segments = ObjectPartSegments(vert_ids={}, face_ids={})

        sampled_points = torch.from_numpy(
            _sample_surface_points(
                track_obj.verts,
                faces_i32,
                int(args.num_object_surface_points),
                SURFACE_SAMPLE_SEED + idx,
            )
        ).float().to(device)

        part_sampled_points: dict[str, torch.Tensor] = {}
        for part_idx, (part_name, face_ids) in enumerate(
            sorted(part_segments.face_ids.items())
        ):
            part_faces = faces_i32[face_ids]
            part_points = _sample_surface_points(
                track_obj.verts,
                part_faces,
                int(args.num_part_surface_points),
                SURFACE_SAMPLE_SEED + 97 * idx + part_idx + 1,
            )
            part_sampled_points[part_name] = torch.from_numpy(part_points).float().to(device)

        object_mask_points, part_mask_points = _load_object_mask_targets(
            dirs["seg_vid"],
            slug,
            list(part_segments.face_ids.keys()),
            num_frames,
            width,
            height,
            device,
            int(args.num_mask_points_2d),
        )

        print(f"  Building SDF for {slug} (res={args.sdf_resolution})")
        sdf_grid = _build_sdf_grid(
            track_obj.verts,
            faces_i32,
            int(args.sdf_resolution),
            device,
        )

        tracked_poses = _identity_poses(num_frames)
        zeros_rot = torch.zeros(num_frames, 3, device=device)
        zeros_trans = torch.zeros(num_frames, 3, device=device)
        object_data[slug] = ObjectData(
            name=state.name,
            slug=slug,
            state=state,
            template_verts=torch.from_numpy(track_obj.verts).float().to(device),
            faces=faces_i32,
            vertex_colors=track_obj.vertex_colors,
            faces_torch=torch.from_numpy(faces_i32.astype(np.int64)).to(device),
            tracked_poses=tracked_poses,
            tracked_poses_torch=torch.from_numpy(tracked_poses).float().to(device),
            tracked_rotvecs=zeros_rot,
            tracked_trans=zeros_trans,
            part_vert_ids=part_segments.vert_ids,
            part_face_ids=part_segments.face_ids,
            sampled_points=sampled_points,
            part_sampled_points=part_sampled_points,
            mask_points_2d=object_mask_points,
            part_mask_points_2d=part_mask_points,
            sdf_grid=sdf_grid,
            color_bgr=OBJECT_COLORS_BGR[idx % len(OBJECT_COLORS_BGR)],
        )
        obj_keys.append(slug)

    interaction_edges: list[InteractionEdge] = []
    print("\nResolving PAG interaction edges for Stage 2...")
    for edge in pag.edges:
        try:
            node_a = _resolve_interaction_node(edge.node_a, humans, object_data, body_seg)
            node_b = _resolve_interaction_node(edge.node_b, humans, object_data, body_seg)
        except (KeyError, ValueError) as exc:
            print(f"  [WARN] Skipping edge: {exc}")
            continue
        interaction_edges.append(
            InteractionEdge(
                node_a=node_a,
                node_b=node_b,
                is_continuous=edge.is_continuous,
                is_rel_static=edge.is_rel_static,
            )
        )
        print(
            f"  Edge: {node_a.raw_node} <-> {node_b.raw_node} "
            f"(continuous={edge.is_continuous}, static={edge.is_rel_static})"
        )

    return ProblemContext(
        dirs=dirs,
        out_dir=dirs["output"],
        pag_path=pag_path,
        smpl_seg_path=smpl_seg_path,
        intr_path=dirs["aligned"] / "alignment_summary.json",
        device=device,
        k=k,
        k_torch=k_torch,
        width=width,
        height=height,
        num_frames=num_frames,
        pag=pag,
        humans=humans,
        human_keys=human_keys,
        objects=object_data,
        obj_keys=obj_keys,
        interaction_edges=interaction_edges,
    )


def _final_T_mats(
    objects: dict[str, ObjectTrackState],
    num_frames: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    return {
        slug: _full_T(obj.rotvecs, obj.trans, num_frames, device)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        for slug, obj in objects.items()
    }


def _final_scales(
    objects: dict[str, ObjectTrackState],
    max_log_scale_delta: float,
) -> dict[str, float]:
    return {
        slug: float(
            _scale_from_raw(obj.raw_scale_delta.detach(), max_log_scale_delta).item()
        )
        for slug, obj in objects.items()
    }


def _save_pose_json(path: Path, T_mats: np.ndarray) -> None:
    rows = [
        {"frame": int(idx), "T_4x4": T_mats[idx].tolist()}
        for idx in range(T_mats.shape[0])
    ]
    _save_json(path, rows)


def _save_transform_refined(path: Path, T_mats: np.ndarray, scale: float) -> None:
    rows = [
        {"frame": int(idx), "T_4x4": T_mats[idx].tolist()}
        for idx in range(T_mats.shape[0])
    ]
    _save_json(path, {"global_scale": float(scale), "frames": rows})


def _save_object_meshes(
    out_dir: Path,
    obj: ObjectTrackState,
    T_mats: np.ndarray,
    scale: float,
) -> None:
    ensure_dir(out_dir)
    for frame_idx in range(T_mats.shape[0]):
        rot = T_mats[frame_idx, :3, :3]
        trans = T_mats[frame_idx, :3, 3]
        verts = (obj.verts * scale) @ rot.T + trans[None, :]
        mesh = obj.mesh.copy()
        mesh.vertices = verts.astype(np.float32)
        mesh.export(str(out_dir / f"frame_{frame_idx:04d}.ply"))


def _save_human_meshes(
    out_dir: Path,
    human: HumanData,
) -> None:
    ensure_dir(out_dir)
    verts = human.base_verts.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts[0], faces=human.faces, process=False)
    for frame_idx in range(verts.shape[0]):
        mesh_t = mesh.copy()
        mesh_t.vertices = verts[frame_idx].astype(np.float32)
        mesh_t.export(str(out_dir / f"frame_{frame_idx:04d}.ply"))


def _build_delta_stats(
    obj: ObjectTrackState,
    scale: float,
) -> dict[str, Any]:
    with torch.no_grad():
        rot_norm = obj.rotvecs.norm(dim=-1)
        trans_norm = obj.trans.norm(dim=-1)
    return {
        "slug": obj.slug,
        "state_type": "absolute_object_trajectory_from_module12",
        "frame0_fixed_identity": True,
        "max_rot_deg": float(rot_norm.max().item() * 180.0 / math.pi)
        if rot_norm.numel() else 0.0,
        "mean_rot_deg": float(rot_norm.mean().item() * 180.0 / math.pi)
        if rot_norm.numel() else 0.0,
        "max_trans_m": float(trans_norm.max().item()) if trans_norm.numel() else 0.0,
        "mean_trans_m": float(trans_norm.mean().item()) if trans_norm.numel() else 0.0,
        "global_scale": float(scale),
        "num_input_tracks": obj.num_input_tracks,
        "num_valid_seed_tracks": obj.num_valid_seed_tracks,
        "num_dropped_invalid_face": obj.num_dropped_invalid_face,
        "num_dropped_outside_mask0": obj.num_dropped_outside_mask0,
        "num_dropped_nonfinite_seed": obj.num_dropped_nonfinite_seed,
        "active_track_frame_pairs": int(obj.vis.gt(0.0).sum().item()),
    }


def _frame_metrics(
    objects: dict[str, ObjectTrackState],
    args: argparse.Namespace,
    k: np.ndarray,
    num_frames: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        per_object = {
            slug: _tracking_object_loss(obj, args, k, float(args.huber_delta_px))
            for slug, obj in objects.items()
        }
        for frame_idx in range(num_frames):
            row: dict[str, Any] = {"frame_idx": int(frame_idx)}
            reproj_vals = []
            active_total = 0
            for slug, item in per_object.items():
                active = item["weights"][frame_idx] > 0.0
                active_count = int(active.sum().item())
                active_total += active_count
                if active_count > 0:
                    reproj = torch.sqrt(item["r2"][frame_idx, active].clamp(min=1e-12))
                    row[f"{slug}_reproj_mean_px"] = float(reproj.mean().item())
                    reproj_vals.append(reproj.mean())
                else:
                    row[f"{slug}_reproj_mean_px"] = float("nan")
            row["active_pairs"] = active_total
            if reproj_vals:
                row["mean_reproj_px"] = float(torch.stack(reproj_vals).mean().item())
            else:
                row["mean_reproj_px"] = float("nan")
            rows.append(row)
    return rows


def _plot_loss_csv(csv_path: Path, out_path: Path, title: str) -> None:
    if not csv_path.exists():
        return
    try:
        import pandas as pd
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = pd.read_csv(csv_path)
    if "iter" not in data or "total" not in data:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(data["iter"], data["total"], label="total", linewidth=1.8)
    for col in data.columns:
        if col.endswith("_scaled") and col != "total_scaled":
            ax.plot(data["iter"], data[col], linewidth=1.0, alpha=0.75, label=col)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), dpi=140)
    plt.close(fig)


def _render_overlay(
    out_dir: Path,
    frame_paths: list[Path],
    humans: dict[str, HumanData],
    human_keys: list[str],
    objects: dict[str, ObjectTrackState],
    T_mats_by_slug: dict[str, np.ndarray],
    scales_by_slug: dict[str, float],
    k: np.ndarray,
    fps: float,
    save_pngs: bool,
    num_frames: int,
) -> None:
    if not frame_paths:
        print("  [WARN] No frames available for overlay")
        return
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        print("  [WARN] Cannot read first frame for overlay")
        return
    h, w = first.shape[:2]
    writer = start_ffmpeg_writer(out_dir / "overlay.mp4", float(fps), (h, w))
    overlays_dir = out_dir / "overlays"
    if save_pngs:
        ensure_dir(overlays_dir)
    human_colors = [(255, 200, 100), (100, 220, 255), (180, 255, 140)]

    try:
        for frame_idx in range(min(num_frames, len(frame_paths))):
            frame = cv2.imread(str(frame_paths[frame_idx]))
            if frame is None:
                continue
            overlay = frame
            for hidx, slug in enumerate(human_keys):
                verts = humans[slug].base_verts[frame_idx].detach().cpu().numpy()
                overlay = draw_overlay(
                    overlay,
                    verts,
                    humans[slug].faces,
                    k,
                    fill_alpha=0.32,
                    contour_thickness=0,
                    color_bgr=human_colors[hidx % len(human_colors)],
                )
            for oidx, (slug, obj) in enumerate(objects.items()):
                T = T_mats_by_slug[slug][frame_idx]
                scale = scales_by_slug[slug]
                verts = (obj.verts * scale) @ T[:3, :3].T + T[:3, 3][None, :]
                overlay = draw_overlay(
                    overlay,
                    verts.astype(np.float32),
                    obj.faces,
                    k,
                    fill_alpha=0.55,
                    contour_thickness=0,
                    color_bgr=OBJECT_COLORS_BGR[oidx % len(OBJECT_COLORS_BGR)],
                )
            if save_pngs:
                cv2.imwrite(str(overlays_dir / f"overlay_{frame_idx:04d}.png"), overlay)
            if writer.stdin is not None:
                writer.stdin.write(np.ascontiguousarray(overlay).tobytes())
    finally:
        close_ffmpeg(writer)


def _save_outputs(
    args: argparse.Namespace,
    dirs: dict[str, Path],
    pag_path: Path,
    smpl_seg_path: Path,
    k: np.ndarray,
    intr_path: Path,
    objects: dict[str, ObjectTrackState],
    humans: dict[str, HumanData],
    human_keys: list[str],
    human_metadata: dict[str, dict[str, Any]],
    stage1_rows: list[dict[str, Any]],
    stage2_rows: list[dict[str, Any]],
    context: ProblemContext | None,
    start_time: float,
) -> None:
    out_dir = dirs["output"]
    ensure_dir(out_dir)
    num_frames = next(iter(humans.values())).base_verts.shape[0]
    device = next(iter(objects.values())).x0.device
    T_mats = _final_T_mats(objects, num_frames, device)
    scales = _final_scales(objects, float(args.max_log_scale_delta))

    print("\nSaving module-12 outputs...")
    object_summaries: dict[str, Any] = {}
    for slug, obj in objects.items():
        obj_dir = out_dir / slug
        ensure_dir(obj_dir)
        _save_pose_json(obj_dir / "poses.json", T_mats[slug])
        _save_transform_refined(obj_dir / "transform_refined.json", T_mats[slug], scales[slug])
        _save_object_meshes(obj_dir / "meshes", obj, T_mats[slug], scales[slug])
        delta_stats = _build_delta_stats(obj, scales[slug])
        _save_json(obj_dir / "delta_stats.json", delta_stats)
        object_summaries[slug] = {
            "name": obj.name,
            "num_verts": int(obj.verts.shape[0]),
            "num_faces": int(obj.faces.shape[0]),
            "final_scale": float(scales[slug]),
            "delta_stats": delta_stats,
        }
        print(f"  {slug}: scale={scales[slug]:.4f}, meshes/ poses.json")

    human_summaries: dict[str, Any] = {}
    for slug in human_keys:
        human_dir = out_dir / slug
        _save_human_meshes(human_dir / "meshes", humans[slug])
        stats = {
            "status": "fixed_smplx_sequence",
            "name": humans[slug].name,
            "slug": slug,
            "num_frames": int(humans[slug].base_verts.shape[0]),
            "num_verts": int(humans[slug].base_verts.shape[1]),
            "num_faces": int(humans[slug].faces.shape[0]),
            "source_metadata": human_metadata[slug],
        }
        _save_json(human_dir / "fixed_input_stats.json", stats)
        human_summaries[slug] = stats

    debug_csv_dir = out_dir / "debug" / "csv"
    debug_plot_dir = out_dir / "debug" / "plots"
    _save_csv(debug_csv_dir / "stage1_iter_metrics.csv", stage1_rows)
    _save_csv(debug_csv_dir / "stage2_iter_metrics.csv", stage2_rows)
    frame_rows = _frame_metrics(objects, args, k, num_frames)
    _save_csv(debug_csv_dir / "frame_loss_metrics.csv", frame_rows)

    if context is not None:
        full_rot, full_trans, raw_scales = _full_delta_vars(objects)
        with torch.no_grad():
            diagnostic = compute_final_loss_diagnostics(
                full_rot,
                full_trans,
                raw_scales,
                context.objects,
                context.humans,
                context.interaction_edges,
                context.obj_keys,
                args,
                iteration=max(int(args.stage2_iters), 1),
                total_iters=max(int(args.stage2_iters), 1),
                k=context.k_torch,
                width=context.width,
                height=context.height,
            )
        final_contact = _diagnostic_summary(diagnostic)
        _save_json(out_dir / "debug" / "final_contact_diagnostic.json", final_contact)
    else:
        final_contact = None

    _plot_loss_csv(
        debug_csv_dir / "stage1_iter_metrics.csv",
        debug_plot_dir / "stage1_loss.png",
        "Stage 1 Loss",
    )
    _plot_loss_csv(
        debug_csv_dir / "stage2_iter_metrics.csv",
        debug_plot_dir / "stage2_loss.png",
        "Stage 2 Loss",
    )

    frames_dir = _resolve_frames_dir(dirs["object_tracks"], dirs["seg_vid"])
    frame_paths = list_images(frames_dir) if frames_dir and frames_dir.exists() else []
    if frame_paths:
        print("  Rendering joint overlay...")
        _render_overlay(
            out_dir,
            frame_paths,
            humans,
            human_keys,
            objects,
            T_mats,
            scales,
            k,
            float(args.overlay_fps),
            bool(args.overlay_save_pngs),
            num_frames,
        )

    summary = {
        "interaction_name": args.interaction_name,
        "status": "completed",
        "script": "01_track_human_object_joint.py",
        "num_frames": int(num_frames),
        "num_objects": len(objects),
        "num_humans": len(human_keys),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "inputs": {
            "object_point_tracks_dir": str(dirs["object_tracks"]),
            "aligned_mesh_dir": str(dirs["aligned"]),
            "human_motion_dir": str(dirs["human_motion"]),
            "segment_video_dir": str(dirs["seg_vid"]),
            "segment_object_dir": str(dirs["seg_obj"]),
            "pag_file": str(pag_path),
            "smpl_seg_json": str(smpl_seg_path),
            "intrinsics_source": str(intr_path),
        },
        "settings": {
            "stage1_iters": int(args.stage1_iters),
            "stage1_lr": float(args.stage1_lr),
            "stage2_iters": int(args.stage2_iters),
            "stage2_lr": float(args.stage2_lr),
            "huber_delta_px": float(args.huber_delta_px),
            "lambda_track_reproj": float(args.lambda_track_reproj),
            "lambda_smooth_trans": float(args.lambda_smooth_trans),
            "lambda_smooth_rot": float(args.lambda_smooth_rot),
            "lambda_scale": float(args.lambda_scale),
            "max_log_scale_delta": float(args.max_log_scale_delta),
            "retrim_interval": int(args.retrim_interval),
            "retrim_percentile": float(args.retrim_percentile),
            "stage2_contact_weights": {
                "object_cd2d_weight": float(args.object_cd2d_weight),
                "object_part_cd2d_weight": float(args.object_part_cd2d_weight),
                "object_smooth_trans_weight": float(args.object_smooth_trans_weight),
                "object_smooth_rot_weight": float(args.object_smooth_rot_weight),
                "object_scale_weight": float(args.object_scale_weight),
                "intersect_weight_start": float(args.intersect_weight_start),
                "intersect_weight_end": float(args.intersect_weight_end),
                "nocontact_weight": float(args.nocontact_weight),
                "contact_drift_weight_start": float(args.contact_drift_weight_start),
                "contact_drift_weight_end": float(args.contact_drift_weight_end),
            },
        },
        "objects": object_summaries,
        "humans": human_summaries,
        "final_contact_diagnostic": final_contact,
        "conventions": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "object_T_4x4": "standard column-vector [[R,t],[0,1]], p'=R@p+t",
            "object_scale": "bounded global scale applied before object T_4x4",
            "frame0_object_pose": "identity relative to module-09 aligned mesh",
            "human": "frozen GVHMR SMPL-X sequence transformed by module-09 similarity",
        },
    }
    _save_json(out_dir / "run_summary.json", summary)
    print(f"Done. Output: {out_dir}")


def _diagnostic_summary(diagnostic: DiagnosticLossResult) -> dict[str, Any]:
    sequence = diagnostic.sequence
    scaled = get_scaled_loss_terms(sequence)
    out: dict[str, Any] = {
        "total": float(sequence.total.item()),
        "terms": {},
    }
    for key, value in scaled.items():
        out["terms"][key] = {
            "weight": float(sequence.weights[key]),
            "raw": float(getattr(sequence, key).item()),
            "scaled": float(value.item()),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified human-object trajectory optimization.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--object_point_tracks_dir", default=None)
    parser.add_argument("--aligned_mesh_dir", default=None)
    parser.add_argument("--human_motion_dir", default=None)
    parser.add_argument("--segment_video_dir", default=None)
    parser.add_argument("--segment_object_dir", default=None)
    parser.add_argument("--pag_file", default=None)
    parser.add_argument("--smpl_seg_json", default=None)
    parser.add_argument(
        "--smpl_folder",
        default="../../GVHMR/inputs/checkpoints/body_models/",
    )
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--stage1_iters", type=int, default=4000)
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument(
        "--stage1_lr_schedule",
        choices=["none", "cosine"],
        default="cosine",
    )
    parser.add_argument("--stage2_iters", type=int, default=1000)
    parser.add_argument("--stage2_lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)

    parser.add_argument("--huber_delta_px", type=float, default=3.0)
    parser.add_argument("--lambda_track_reproj", type=float, default=1.0)
    parser.add_argument("--lambda_smooth_trans", type=float, default=300.0)
    parser.add_argument("--lambda_smooth_rot", type=float, default=80.0)
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--max_log_scale_delta", type=float, default=0.22)

    parser.add_argument("--object_cd2d_weight", type=float, default=1e-4)
    parser.add_argument("--object_part_cd2d_weight", type=float, default=1e-4)
    parser.add_argument("--object_smooth_trans_weight", type=float, default=1e3)
    parser.add_argument("--object_smooth_rot_weight", type=float, default=1e3)
    parser.add_argument("--object_scale_weight", type=float, default=1.0)
    parser.add_argument("--intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--intersect_weight_end", type=float, default=10.0)
    parser.add_argument("--nocontact_weight", type=float, default=1e3)
    parser.add_argument("--contact_drift_weight_start", type=float, default=10.0)
    parser.add_argument("--contact_drift_weight_end", type=float, default=1e3)

    parser.add_argument("--bin_size", type=int, default=0)
    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--mask_gate_threshold", type=float, default=0.5)
    parser.add_argument("--visibility_threshold", type=float, default=0.0)
    parser.add_argument("--min_valid_tracks", type=int, default=50)
    parser.add_argument("--disable_pnp_init", action="store_true")
    parser.add_argument("--pnp_ransac_thresh", type=float, default=8.0)
    parser.add_argument("--outlier_reproj_thresh_px", type=float, default=20.0)
    parser.add_argument("--outlier_max_fraction", type=float, default=0.4)
    parser.add_argument(
        "--graduated_huber",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--retrim_interval", type=int, default=1000)
    parser.add_argument("--retrim_percentile", type=float, default=90.0)

    parser.add_argument("--num_object_surface_points", type=int, default=3000)
    parser.add_argument("--num_part_surface_points", type=int, default=1000)
    parser.add_argument("--num_mask_points_2d", type=int, default=2500)
    parser.add_argument("--sdf_resolution", type=int, default=48)
    parser.add_argument("--smplx_batch_size", type=int, default=32)

    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--debug_save_interval", type=int, default=20)
    parser.add_argument("--overlay_fps", type=float, default=6.0)
    parser.add_argument(
        "--overlay_save_pngs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    dirs = _resolve_dirs(args)
    pag_path = _resolve_pag_path(args)
    smpl_seg_path = _resolve_smpl_seg_path(args)
    device = _to_device(args.device)
    ensure_dir(dirs["output"])

    for label, path in (
        ("Object point tracks", dirs["object_tracks"]),
        ("Aligned meshes", dirs["aligned"]),
        ("Human motion", dirs["human_motion"]),
        ("Video segmentation", dirs["seg_vid"]),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} dir not found: {path}")

    k, intr_path = _load_intrinsics_from_alignment_summary(dirs["aligned"])
    transforms_by_slug = _load_alignment_transforms(dirs["aligned"])
    body_seg, contact_seg = _load_smpl_body_and_contact_seg(smpl_seg_path)
    pag = _parse_pag(pag_path)
    if contact_seg:
        print(
            "Loaded SMPL-X contact subsets from "
            f"{smpl_seg_path.name}: {', '.join(sorted(contact_seg))}"
        )

    print("=" * 68)
    print("Module 12: Unified Human-Object Trajectory Optimization")
    print(f"  interaction: {args.interaction_name}")
    print(f"  device:      {device}")
    print(f"  K:           fx={k[0,0]:.1f} fy={k[1,1]:.1f} cx={k[0,2]:.1f} cy={k[1,2]:.1f}")
    print(f"  PAG:         {pag_path.name}")
    print("=" * 68)

    humans, human_keys, human_metadata, human_num_frames = _load_frozen_humans(
        args,
        dirs,
        transforms_by_slug,
        body_seg,
        contact_seg,
        device,
    )
    num_frames = _discover_num_frames(args, dirs, pag, human_num_frames)
    if num_frames < human_num_frames:
        for human in humans.values():
            human.base_verts = human.base_verts[:num_frames]
            for key in list(human.part_points):
                human.part_points[key] = human.part_points[key][:num_frames]
            for key in list(human.contact_part_points):
                human.contact_part_points[key] = human.contact_part_points[key][:num_frames]
        for metadata in human_metadata.values():
            metadata["num_frames"] = int(num_frames)
    print(f"Using {num_frames} frames")

    objects: dict[str, ObjectTrackState] = {}
    print("\nLoading object tracks and frame-0 surface anchors...")
    for state in pag.object_states:
        obj = _load_track_object_state(args, dirs, state, k, num_frames, device)
        if obj is not None:
            objects[obj.slug] = obj
            print(
                f"  {obj.slug}: {obj.num_valid_seed_tracks}/{obj.num_input_tracks} "
                "valid seed tracks"
            )
    if not objects:
        raise RuntimeError("No objects loaded for module-12 optimization.")

    stage1_rows = _run_stage1(objects, args, k)

    context: ProblemContext | None = None
    stage2_rows: list[dict[str, Any]] = []
    if int(args.stage2_iters) > 0:
        context = _build_contact_context(
            args=args,
            dirs=dirs,
            pag_path=pag_path,
            smpl_seg_path=smpl_seg_path,
            k=k,
            humans=humans,
            human_keys=human_keys,
            objects=objects,
            num_frames=num_frames,
            device=device,
        )
        stage2_rows = _run_stage2(objects, context, args)
    else:
        print("\n[Stage 2] disabled by --stage2_iters 0")

    _save_outputs(
        args=args,
        dirs=dirs,
        pag_path=pag_path,
        smpl_seg_path=smpl_seg_path,
        k=k,
        intr_path=intr_path,
        objects=objects,
        humans=humans,
        human_keys=human_keys,
        human_metadata=human_metadata,
        stage1_rows=stage1_rows,
        stage2_rows=stage2_rows,
        context=context,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
