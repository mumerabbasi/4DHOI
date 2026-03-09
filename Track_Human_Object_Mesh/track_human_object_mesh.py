"""Joint human-object mesh refinement with PAG contact constraints.

Given aligned human meshes, tracked object SE(3) trajectories, segmentation
masks, and a PAG, this script refines:

- optional per-frame global human SE(3) corrections,
- per-frame object SE(3) deltas on top of tracked poses,
- one global uniform scale per object.

The objective combines trajectory priors, PAG contact and dynamics, SDF-based
penetration penalties, temporal smoothness, and 2D mask chamfer losses for the
human, full objects, and object parts.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.ops import knn_points
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from utils import (
    _load_intrinsics_from_alignment_summary,
    _save_csv,
    _to_device,
    close_ffmpeg,
    draw_overlay,
    ensure_dir,
    list_images,
    resolve_path,
    start_ffmpeg_writer,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERLAY_FILL_ALPHA = 0.55
OVERLAY_CONTOUR_THICKNESS = 0
# Per-object colours (cycle if more objects).
OBJECT_COLORS_BGR: list[tuple[int, int, int]] = [
    (0, 255, 255),   # yellow-cyan
    (255, 128, 0),   # orange-blue
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (128, 255, 128),  # light green
    (0, 128, 255),   # orange
]
HUMAN_COLOR_BGR: tuple[int, int, int] = (255, 200, 100)  # light blue

# SMPL body-part → vertex-segmentation keys produced by
# export_segmented_human_motion.
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

# Maximum points to sample per part for contact / SDF queries
MAX_PART_POINTS = 2048
MASK_SAMPLE_SEED = 2026
SURFACE_SAMPLE_SEED = 7


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint human–object mesh refinement with PAG constraints."
    )
    p.add_argument("--video_name", type=str, default="video_01")
    p.add_argument(
        "--aligned_mesh_dir",
        type=str,
        default=None,
        help="Align_Meshes/output/<video> (auto-resolved).",
    )
    p.add_argument(
        "--tracked_object_dir",
        type=str,
        default=None,
        help="Track_Object_Mesh/output_cotracker/<video> (auto-resolved).",
    )
    p.add_argument(
        "--segment_object_dir",
        type=str,
        default=None,
        help="Segment_Object_Mesh/output/<video> (auto-resolved).",
    )
    p.add_argument(
        "--segment_video_dir",
        type=str,
        default=None,
        help="Segment_Video/output/<video> for frame images (auto-resolved).",
    )
    p.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="PAG JSON. Auto-resolved from Generate_PAG/output/<video>.",
    )
    p.add_argument(
        "--smpl_seg_json",
        type=str,
        default=None,
        help="SMPL vert segmentation JSON (auto-resolved from GVHMR).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root output directory.",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    # SDF
    p.add_argument(
        "--sdf_resolution",
        type=int,
        default=128,
        help="Voxel resolution for SDF grids.",
    )
    # Optimisation
    p.add_argument("--adam_iters", type=int, default=4000)
    p.add_argument("--adam_lr", type=float, default=1e-3)
    p.add_argument(
        "--early_stop_start",
        type=int,
        default=400,
        help="Iteration at which to begin checking early stopping.",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Consecutive no-improvement iterations before stopping. Set 0 to disable.",
    )
    p.add_argument(
        "--early_stop_rel_improve",
        type=float,
        default=1e-4,
        help="Minimum relative best-loss improvement required to reset patience.",
    )
    p.add_argument(
        "--optimize_human",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimise per-frame global human SE(3) corrections.",
    )
    p.add_argument(
        "--optimize_object_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimise one global uniform scale per object.",
    )
    p.add_argument(
        "--max_log_scale_delta",
        type=float,
        default=0.22,
        help="Maximum absolute log-scale correction per object.",
    )
    # Loss weights
    p.add_argument(
        "--lambda_prior",
        type=float,
        default=20.0,
        help="Motion-prior weight (stay close to tracked poses).",
    )
    p.add_argument(
        "--lambda_contact",
        type=float,
        default=200.0,
        help="Contact consistency weight.",
    )
    p.add_argument(
        "--lambda_dynamics",
        type=float,
        default=150.0,
        help="Contact dynamics weight.",
    )
    p.add_argument(
        "--lambda_penetration",
        type=float,
        default=20.0,
        help="Max penetration weight (annealed from 0).",
    )
    p.add_argument(
        "--lambda_smooth",
        type=float,
        default=12.0,
        help="Temporal smoothness weight.",
    )
    p.add_argument(
        "--lambda_human_prior",
        type=float,
        default=100.0,
        help="Human correction prior weight.",
    )
    p.add_argument(
        "--lambda_human_smooth",
        type=float,
        default=30.0,
        help="Human correction smoothness weight.",
    )
    p.add_argument(
        "--lambda_human_mask_2d",
        type=float,
        default=40.0,
        help="2D human silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_mask_2d",
        type=float,
        default=60.0,
        help="2D object silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_part_mask_2d",
        type=float,
        default=120.0,
        help="2D object part silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_scale",
        type=float,
        default=30.0,
        help="Object global scale regularisation weight.",
    )
    p.add_argument(
        "--num_mask_points_2d",
        type=int,
        default=2048,
        help="Maximum sampled 2D mask points per frame.",
    )
    p.add_argument(
        "--num_object_surface_points",
        type=int,
        default=4096,
        help="Whole-object sampled surface points.",
    )
    p.add_argument(
        "--num_part_surface_points",
        type=int,
        default=2048,
        help="Per-part sampled surface points.",
    )
    p.add_argument(
        "--num_human_surface_points",
        type=int,
        default=4096,
        help="Sampled human surface points per frame.",
    )
    # Rendering
    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--save_overlay_pngs", action="store_true")
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------
def _resolve_dirs(
    args: argparse.Namespace,
    script_dir: Path,
) -> dict[str, Path]:
    """Resolve all input/output directories."""
    vname = args.video_name
    parent = script_dir.parent  # 4DHOI root

    def _r(val: str | None, default: Path) -> Path:
        if val is not None:
            return resolve_path(val, script_dir)
        return default.resolve()

    aligned = _r(args.aligned_mesh_dir,
                 parent / "Align_Meshes" / "output" / vname)
    tracked = _r(args.tracked_object_dir,
                 parent / "Track_Object_Mesh" / "output_cotracker" / vname)
    seg_obj = _r(args.segment_object_dir,
                 parent / "Segment_Object_Mesh" / "output" / vname)
    seg_vid = _r(args.segment_video_dir,
                 parent / "Segment_Video" / "output" / vname)
    output = resolve_path(args.output_dir, script_dir) / vname

    return dict(
        aligned=aligned,
        tracked=tracked,
        seg_obj=seg_obj,
        seg_vid=seg_vid,
        output=output,
    )


def _resolve_pag_path(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.pag_file is not None:
        p = resolve_path(args.pag_file, script_dir)
        if not p.exists():
            raise FileNotFoundError(f"PAG file not found: {p}")
        return p
    pag_dir = (
        script_dir.parent / "Generate_PAG" / "output" / args.video_name
    ).resolve()
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No PAG JSON in {pag_dir}")
    return candidates[0]


def _resolve_smpl_seg(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.smpl_seg_json is not None:
        return resolve_path(args.smpl_seg_json, script_dir)
    # Try standard location
    candidates = [
        script_dir.parent.parent
        / "GVHMR"
        / "hmr4d"
        / "utils"
        / "body_model"
        / "smpl_vert_segmentation.json",
    ]
    for c in candidates:
        if c.resolve().exists():
            return c.resolve()
    raise FileNotFoundError(
        "Cannot find smpl_vert_segmentation.json. Pass --smpl_seg_json."
    )


def _resolve_frames_dir(dirs: dict[str, Path]) -> Path | None:
    """Find original video frames for overlay rendering."""
    candidates = [
        dirs["seg_vid"] / "frames",
        dirs["seg_vid"] / "_frames",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


# ---------------------------------------------------------------------------
# PAG parsing
# ---------------------------------------------------------------------------
@dataclass
class PAGObjectState:
    name: str
    slug: str
    is_translational: bool
    is_rotational: bool


@dataclass
class PAGEdge:
    node_a: str        # e.g. "iron, handle" or "person 1, right hand"
    node_b: str
    is_continuous: bool
    is_rel_static: bool


@dataclass
class PAG:
    object_states: list[PAGObjectState]
    body_part_nodes: list[str]
    object_part_nodes: list[str]
    edges: list[PAGEdge]


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_")


def _parse_pag(pag_path: Path) -> PAG:
    with pag_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Object states
    obj_states: list[PAGObjectState] = []
    for item in data.get("object states", []):
        name = item["name"].strip()
        obj_states.append(PAGObjectState(
            name=name, slug=_sanitize(name),
            is_translational=bool(item.get("is_translational", True)),
            is_rotational=bool(item.get("is_rotational", True)),
        ))

    body_nodes = [s.strip() for s in data.get("body part nodes", [])]
    obj_nodes = [s.strip() for s in data.get("object part nodes", [])]

    edges: list[PAGEdge] = []
    for e in data.get("interaction edges", []):
        nodes = e["nodes"]
        edges.append(PAGEdge(
            node_a=nodes[0].strip(), node_b=nodes[1].strip(),
            is_continuous=bool(e.get("is_continuous", True)),
            is_rel_static=bool(e.get("is_rel_static", False)),
        ))

    return PAG(object_states=obj_states, body_part_nodes=body_nodes,
               object_part_nodes=obj_nodes, edges=edges)


# ---------------------------------------------------------------------------
# Node parsing helpers
# ---------------------------------------------------------------------------
def _parse_node(node_str: str) -> tuple[str, str]:
    """Parse 'object_or_person, part' → (entity, part)."""
    parts = node_str.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse PAG node: '{node_str}'")
    return parts[0].strip(), parts[1].strip()


def _is_human_node(node_str: str) -> bool:
    return node_str.lower().startswith("person")


# ---------------------------------------------------------------------------
# SMPL body segmentation
# ---------------------------------------------------------------------------
def _load_smpl_body_seg(seg_path: Path) -> dict[str, np.ndarray]:
    """Load SMPL vert segmentation → {body_part_name: vertex_indices}."""
    with seg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # raw: {"leftHand": [idx, ...], "rightHand": [...], ...}
    # Build mapping from PAG body-part names to merged vertex indices
    result: dict[str, np.ndarray] = {}
    for pag_name, seg_keys in BODY_PART_TO_SEG_KEYS.items():
        indices: list[int] = []
        for sk in seg_keys:
            indices.extend(raw.get(sk, []))
        if indices:
            result[pag_name] = np.unique(np.array(indices, dtype=np.int64))
    return result


# ---------------------------------------------------------------------------
# Object part segmentation
# ---------------------------------------------------------------------------
@dataclass
class PackedPointCloud2D:
    points: torch.Tensor   # [T, P_max, 2]
    lengths: torch.Tensor  # [T]


@dataclass
class ObjectPartSegments:
    vert_ids: dict[str, np.ndarray]
    face_ids: dict[str, np.ndarray]


def _load_object_part_segments(
    seg_obj_dir: Path,
    obj_slug: str,
    mesh_faces: np.ndarray,
) -> ObjectPartSegments:
    """Load triangle labels → per-part vertex and face indices."""
    labels_path = (
        seg_obj_dir / obj_slug / "segmented_meshes"
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
            part_face_ids[part_name] = np.flatnonzero(
                tri_mask
            ).astype(np.int64)
    return ObjectPartSegments(
        vert_ids=part_vert_ids,
        face_ids=part_face_ids,
    )


def _sample_surface_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Deterministically sample points on a triangle mesh surface."""
    if count <= 0 or faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    tri = vertices[faces]  # [F, 3, 3]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * 0.5
    positive = areas > 1e-12
    if not np.any(positive):
        return vertices[_subsample_indices(vertices.shape[0], count)]

    valid_faces = faces[positive]
    valid_tri = tri[positive]
    weights = areas[positive]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    face_ids = rng.choice(
        valid_faces.shape[0],
        size=count,
        replace=True,
        p=weights,
    )
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    ).astype(np.float32)
    sampled_tri = valid_tri[face_ids]
    return np.sum(sampled_tri * bary[:, :, None], axis=1).astype(np.float32)


def _sample_face_barycentrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically sample face ids and barycentrics."""
    if count <= 0 or faces.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float32),
        )

    tri = vertices[faces]
    edge_1 = tri[:, 1] - tri[:, 0]
    edge_2 = tri[:, 2] - tri[:, 0]
    areas = np.linalg.norm(np.cross(edge_1, edge_2), axis=1) * 0.5
    positive = areas > 1e-12
    if not np.any(positive):
        face_ids = _subsample_indices(faces.shape[0], count)
        bary = np.tile(
            np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=np.float32),
            (len(face_ids), 1),
        )
        return face_ids, bary

    valid_face_ids = np.flatnonzero(positive)
    weights = areas[positive]
    weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_face_ids, size=count, replace=True, p=weights)
    r1 = rng.random(count, dtype=np.float32)
    r2 = rng.random(count, dtype=np.float32)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    ).astype(np.float32)
    return chosen.astype(np.int64), bary


def _sample_sequence_points_from_barycentrics(
    verts_seq: torch.Tensor,        # [T, V, 3]
    faces_torch: torch.Tensor,      # [F, 3]
    face_ids: torch.Tensor,         # [P]
    bary: torch.Tensor,             # [P, 3]
) -> torch.Tensor:
    if face_ids.numel() == 0:
        return torch.zeros(
            (verts_seq.shape[0], 0, 3),
            dtype=verts_seq.dtype,
            device=verts_seq.device,
        )
    face_vids = faces_torch[face_ids]  # [P, 3]
    tri = verts_seq[:, face_vids, :]   # [T, P, 3, 3]
    bary_exp = bary.view(1, -1, 3, 1)
    return (tri * bary_exp).sum(dim=2)


def _mask_to_point_cloud(
    mask: np.ndarray,
    width: int,
    height: int,
    max_points: int,
) -> np.ndarray:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack(
        [
            (xs.astype(np.float32) + 0.5) / float(width),
            (ys.astype(np.float32) + 0.5) / float(height),
        ],
        axis=1,
    )
    idx = _subsample_indices(len(pts), max_points)
    return pts[idx].astype(np.float32)


def _pack_2d_point_clouds(
    point_arrays: list[np.ndarray],
    device: torch.device,
) -> PackedPointCloud2D:
    max_len = max((arr.shape[0] for arr in point_arrays), default=0)
    max_len = max(max_len, 1)
    packed = np.zeros((len(point_arrays), max_len, 2), dtype=np.float32)
    lengths = np.zeros((len(point_arrays),), dtype=np.int64)
    for i, arr in enumerate(point_arrays):
        lengths[i] = arr.shape[0]
        if arr.shape[0] > 0:
            packed[i, :arr.shape[0]] = arr
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
            mask = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        arrays.append(_mask_to_point_cloud(mask, width, height, max_points))
    return _pack_2d_point_clouds(arrays, device)


def _infer_image_size(
    dirs: dict[str, Path],
) -> tuple[int, int]:
    frames_dir = _resolve_frames_dir(dirs)
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
        if frame_paths:
            frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
            if frame is not None:
                height, width = frame.shape[:2]
                return width, height

    sample_candidates = [
        dirs["seg_vid"] / "humans" / "person_1" / "masks" / "frame_0000.png",
    ]
    sample_candidates.extend(
        sorted(
            dirs["seg_vid"].glob(
                "objects/*/object_segmentation/masks/frame_0000.png"
            )
        )
    )
    for sample_path in sample_candidates:
        if sample_path.exists():
            mask = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                height, width = mask.shape[:2]
                return width, height
    raise FileNotFoundError("Could not infer image size from frames or masks.")


def _resolve_human_mask_dir(seg_vid_dir: Path) -> Path | None:
    humans_dir = seg_vid_dir / "humans"
    if not humans_dir.exists():
        return None
    candidates = sorted(
        d for d in humans_dir.iterdir()
        if d.is_dir() and (d / "masks").exists()
    )
    if not candidates:
        return None
    return candidates[0] / "masks"


def _load_human_data(
    human_verts_np: np.ndarray,
    human_faces: np.ndarray,
    body_seg: dict[str, np.ndarray],
    dirs: dict[str, Path],
    width: int,
    height: int,
    device: torch.device,
    args: argparse.Namespace,
) -> HumanData:
    base_verts = torch.from_numpy(human_verts_np).float().to(device)
    faces_torch = torch.from_numpy(human_faces.astype(np.int64)).to(device)
    part_points_base: dict[str, torch.Tensor] = {}
    for part_name, vert_ids in body_seg.items():
        sub_ids = _subsample_indices(
            len(vert_ids),
            args.num_part_surface_points,
        )
        selected = vert_ids[sub_ids]
        part_points_base[part_name] = base_verts[:, selected, :]

    face_ids_np, bary_np = _sample_face_barycentrics(
        human_verts_np[0],
        human_faces,
        args.num_human_surface_points,
        SURFACE_SAMPLE_SEED,
    )
    sampled_points_base = _sample_sequence_points_from_barycentrics(
        base_verts,
        faces_torch,
        torch.from_numpy(face_ids_np).to(device),
        torch.from_numpy(bary_np).to(device),
    )
    if "hips" in body_seg and body_seg["hips"].size > 0:
        centers = base_verts[:, body_seg["hips"], :].mean(dim=1)
    else:
        centers = base_verts.mean(dim=1)

    human_mask_dir = _resolve_human_mask_dir(dirs["seg_vid"])
    human_mask_points = None
    if human_mask_dir is not None:
        human_mask_points = _load_mask_point_clouds(
            human_mask_dir,
            human_verts_np.shape[0],
            width,
            height,
            args.num_mask_points_2d,
            device,
        )
    return HumanData(
        base_verts=base_verts,
        faces=human_faces,
        faces_torch=faces_torch,
        part_points_base=part_points_base,
        sampled_points_base=sampled_points_base,
        centers=centers,
        mask_points_2d=human_mask_points,
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
    object_mask_dir = (
        seg_vid_dir
        / "objects"
        / slug
        / "object_segmentation"
        / "masks"
    )
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
        part_mask_dir = (
            seg_vid_dir
            / "objects"
            / slug
            / "parts_segmentation"
            / "masks"
            / part_name
        )
        packed = _load_mask_point_clouds(
            part_mask_dir,
            num_frames,
            width,
            height,
            max_points,
            device,
        )
        if packed is not None:
            part_mask_points[part_name] = packed
    return object_mask_points, part_mask_points


# ---------------------------------------------------------------------------
# SDF precomputation and querying
# ---------------------------------------------------------------------------
@dataclass
class SDFGrid:
    """Pre-computed SDF volume for an object in canonical frame-0 pose."""
    sdf_volume: torch.Tensor   # [1, 1, D, D, D]
    bbox_min: torch.Tensor     # [1, 1, 3]
    bbox_max: torch.Tensor     # [1, 1, 3]


def _build_sdf_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int,
    device: torch.device,
    padding: float = 0.05,
) -> SDFGrid:
    """Build a voxel SDF grid for a mesh using pysdf."""
    from pysdf import SDF as PySDF

    sdf_func = PySDF(vertices.astype(np.float32), faces.astype(np.uint32))

    # Compute bounding box with padding
    vmin = vertices.min(axis=0) - padding
    vmax = vertices.max(axis=0) + padding

    # Create query grid
    lin = [np.linspace(vmin[i], vmax[i], resolution) for i in range(3)]
    gx, gy, gz = np.meshgrid(lin[0], lin[1], lin[2], indexing="ij")
    query_pts = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel()],
        axis=1,
    ).astype(np.float32)

    # Query SDF (pysdf: positive=inside, negative=outside)
    sdf_vals = sdf_func(query_pts)
    # Convert to convention: negative=inside for the optimization losses.
    sdf_vals = -sdf_vals
    sdf_vol = sdf_vals.reshape(1, 1, resolution, resolution, resolution)

    return SDFGrid(
        sdf_volume=torch.from_numpy(sdf_vol.astype(np.float32)).to(device),
        bbox_min=torch.tensor(
            vmin.reshape(1, 1, 3),
            dtype=torch.float32,
            device=device,
        ),
        bbox_max=torch.tensor(
            vmax.reshape(1, 1, 3),
            dtype=torch.float32,
            device=device,
        ),
    )


def _query_sdf(
    sdf_grid: SDFGrid,
    points: torch.Tensor,  # [N, 3]
) -> torch.Tensor:
    """Query SDF values for points.  Returns [N] values (negative = inside)."""
    pts = points.unsqueeze(0)  # [1, N, 3]
    # Normalise to [-1, 1] for grid_sample
    normalised = (
        (pts - sdf_grid.bbox_min)
        / (sdf_grid.bbox_max - sdf_grid.bbox_min)
        * 2.0
        - 1.0
    )
    # grid_sample expects [B, C, D, H, W] input and [B, N, 1, 1, 3] grid
    # in z,y,x order.
    grid = normalised[:, :, [2, 1, 0]].view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        sdf_grid.sdf_volume,
        grid,
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1)  # [N]


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------
def _load_tracked_poses(poses_path: Path) -> np.ndarray:
    """Load poses.json → [T, 4, 4] numpy."""
    with poses_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = sorted(data, key=lambda x: x["frame"])
    T_mats = np.array([f["T_4x4"] for f in frames], dtype=np.float32)
    return T_mats


def _decompose_T(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[4,4] → (rotvec [3], trans [3])."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_torch = torch.from_numpy(R).unsqueeze(0).float()
    rotvec = matrix_to_axis_angle(R_torch).squeeze(0).numpy()
    return rotvec.astype(np.float32), t.astype(np.float32)


def _compose_T(rotvec: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """rotvec [3], trans [3] → T [4,4]. No in-place ops for autograd safety."""
    R = axis_angle_to_matrix(rotvec.unsqueeze(0)).squeeze(0)  # [3,3]
    Rt = torch.cat([R, trans.unsqueeze(1)], dim=1)  # [3, 4]
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=rotvec.device,
    )
    return torch.cat([Rt, bottom], dim=0)  # [4, 4]


def _apply_T_batch(verts: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """verts [V,3], T [4,4] → transformed [V,3].  p' = R @ p + t."""
    R = T[:3, :3]
    t = T[:3, 3]
    return verts @ R.t() + t.unsqueeze(0)


def _apply_similarity_batch(
    points: torch.Tensor,    # [N, 3]
    T: torch.Tensor,         # [4, 4]
    scale: torch.Tensor,     # scalar
) -> torch.Tensor:
    return _apply_T_batch(points * scale, T)


def _apply_inverse_similarity_batch(
    points: torch.Tensor,    # [N, 3]
    T: torch.Tensor,         # [4, 4]
    scale: torch.Tensor,     # scalar
) -> torch.Tensor:
    R = T[:3, :3]
    t = T[:3, 3]
    return ((points - t.unsqueeze(0)) @ R) / scale


def _apply_similarity_sequence(
    points: torch.Tensor,    # [P, 3]
    T_seq: torch.Tensor,     # [T, 4, 4]
    scale: torch.Tensor,     # scalar
) -> torch.Tensor:
    scaled = points * scale
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(scaled.unsqueeze(0), R.transpose(1, 2)) + t[:, None, :]


def _apply_inverse_similarity_sequence(
    points_seq: torch.Tensor,    # [T, P, 3]
    T_seq: torch.Tensor,         # [T, 4, 4]
    scale: torch.Tensor,         # scalar
) -> torch.Tensor:
    R = T_seq[:, :3, :3]
    t = T_seq[:, :3, 3]
    return torch.matmul(points_seq - t[:, None, :], R) / scale


def _apply_local_se3_sequence(
    points_seq: torch.Tensor,    # [T, P, 3]
    rotvecs: torch.Tensor,       # [T, 3]
    trans: torch.Tensor,         # [T, 3]
    centers: torch.Tensor,       # [T, 3]
) -> torch.Tensor:
    if points_seq.numel() == 0:
        return points_seq
    R = axis_angle_to_matrix(rotvecs)  # [T, 3, 3]
    centered = points_seq - centers[:, None, :]
    rotated = torch.matmul(centered, R.transpose(1, 2))
    return rotated + centers[:, None, :] + trans[:, None, :]


def _project_points_normalized_torch(
    points: torch.Tensor,   # [T, P, 3]
    k: torch.Tensor,        # [3, 3]
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = points[..., 2]
    valid = torch.isfinite(points).all(dim=-1) & (z > 1e-6)
    uv = torch.zeros(
        (*points.shape[:2], 2),
        dtype=points.dtype,
        device=points.device,
    )
    z_safe = z.clamp(min=1e-6)
    uv[..., 0] = (
        points[..., 0] * k[0, 0] / z_safe + k[0, 2]
    ) / float(width)
    uv[..., 1] = (
        points[..., 1] * k[1, 1] / z_safe + k[1, 2]
    ) / float(height)
    return uv, valid


def _pack_projected_points(
    points_2d: torch.Tensor,    # [T, P, 2]
    valid: torch.Tensor,        # [T, P]
) -> tuple[torch.Tensor, torch.Tensor]:
    order = torch.argsort(valid.to(torch.int64), dim=1, descending=True)
    packed = torch.gather(
        points_2d,
        dim=1,
        index=order.unsqueeze(-1).expand(-1, -1, 2),
    )
    lengths = valid.sum(dim=1)
    return packed, lengths


def _masked_mean_from_lengths(
    values: torch.Tensor,      # [B, P]
    lengths: torch.Tensor,     # [B]
) -> torch.Tensor:
    idx = torch.arange(values.shape[1], device=values.device)[None, :]
    mask = idx < lengths[:, None]
    denom = mask.sum().clamp(min=1)
    return (values * mask).sum() / denom


def _inv_se3(T: torch.Tensor) -> torch.Tensor:
    """Analytical SE(3) inverse.  T_inv = [[R^T, -R^T @ t], [0,0,0,1]].
    Numerically stable — no general matrix inversion."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.t()
    t_inv = -(R_inv @ t)
    Rt = torch.cat([R_inv, t_inv.unsqueeze(1)], dim=1)  # [3, 4]
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]],
                          dtype=torch.float32, device=T.device)
    return torch.cat([Rt, bottom], dim=0)


def _geodesic_distance_sq(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Squared geodesic distance between two rotation matrices.
    Uses 1 - cos(angle) instead of acos for gradient stability."""
    R_rel = R1.t() @ R2
    cos_angle = (R_rel.trace() - 1.0) / 2.0
    cos_angle = cos_angle.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    # 1 - cos(angle) is proportional to angle^2/2 for small angles
    return 1.0 - cos_angle


# ---------------------------------------------------------------------------
# Data structures for optimisation
# ---------------------------------------------------------------------------
@dataclass
class HumanData:
    base_verts: torch.Tensor            # [T, V, 3]
    faces: np.ndarray                   # [F, 3]
    faces_torch: torch.Tensor           # [F, 3]
    part_points_base: dict[str, torch.Tensor]  # part -> [T, P, 3]
    sampled_points_base: torch.Tensor   # [T, P, 3]
    centers: torch.Tensor               # [T, 3]
    mask_points_2d: PackedPointCloud2D | None


@dataclass
class ObjectData:
    """All data for a single object."""
    name: str
    slug: str
    state: PAGObjectState
    template_verts: torch.Tensor       # [V, 3] canonical (frame-0 aligned)
    faces: np.ndarray                  # [F, 3] int
    faces_torch: torch.Tensor          # [F, 3] long
    tracked_poses: np.ndarray          # [T, 4, 4]
    tracked_rotvecs: torch.Tensor      # [T, 3]
    tracked_trans: torch.Tensor        # [T, 3]
    part_vert_ids: dict[str, np.ndarray]  # part_name → vertex indices
    part_face_ids: dict[str, np.ndarray]  # part_name → face indices
    sampled_points: torch.Tensor       # [P, 3] canonical whole-object samples
    part_sampled_points: dict[str, torch.Tensor]  # part_name -> [P, 3]
    mask_points_2d: PackedPointCloud2D | None
    part_mask_points_2d: dict[str, PackedPointCloud2D]
    sdf_grid: SDFGrid | None           # for penetration queries
    color_bgr: tuple[int, int, int]


@dataclass
class ResolvedEdge:
    """A PAG edge resolved to actual vertex index sets."""
    # Side A
    a_is_human: bool
    a_object_idx: int       # -1 if human
    a_vert_ids: np.ndarray  # vertex indices into the respective mesh
    a_part_name: str

    # Side B
    b_is_human: bool
    b_object_idx: int
    b_vert_ids: np.ndarray
    b_part_name: str

    # PAG attributes
    is_continuous: bool
    is_rel_static: bool
    contact_reduction: str   # "mean" or "min" for point-to-part contact
    contact_source_is_a: bool

    # For canonical-space dynamics: use the less-mobile object as reference.
    canonical_obj_idx: int  # index of reference object (-1 if no object)


def _mean_translation_step(tracked_poses: np.ndarray) -> float:
    if tracked_poses.shape[0] < 2:
        return 0.0
    diffs = np.diff(tracked_poses[:, :3, 3], axis=0)
    return float(np.linalg.norm(diffs, axis=1).mean())


def _mean_rotation_step(tracked_poses: np.ndarray) -> float:
    if tracked_poses.shape[0] < 2:
        return 0.0
    angles: list[float] = []
    for t in range(tracked_poses.shape[0] - 1):
        R1 = tracked_poses[t, :3, :3]
        R2 = tracked_poses[t + 1, :3, :3]
        R_rel = R1.T @ R2
        cos_angle = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.arccos(cos_angle)))
    return float(np.mean(angles))


def _reference_priority(od: ObjectData) -> tuple[int, int, float, float]:
    """Lower priority tuple means a better canonical reference object."""
    return (
        int(od.state.is_translational),
        int(od.state.is_rotational),
        _mean_translation_step(od.tracked_poses),
        _mean_rotation_step(od.tracked_poses),
    )


def _select_canonical_reference_obj(
    a_is_human: bool,
    a_obj_idx: int,
    b_is_human: bool,
    b_obj_idx: int,
    objects: dict[str, ObjectData],
    obj_keys: list[str],
) -> int:
    if a_is_human and b_is_human:
        return -1
    if a_is_human:
        return b_obj_idx
    if b_is_human:
        return a_obj_idx

    od_a = objects[obj_keys[a_obj_idx]]
    od_b = objects[obj_keys[b_obj_idx]]
    if _reference_priority(od_a) <= _reference_priority(od_b):
        return a_obj_idx
    return b_obj_idx


def _uses_mean_contact_reduction(is_human: bool, part_name: str) -> bool:
    if not is_human:
        return False
    part = part_name.lower().strip()
    return part.endswith("hand") or part.endswith("foot")


def _select_contact_reduction(
    a_is_human: bool,
    a_part_name: str,
    b_is_human: bool,
    b_part_name: str,
) -> str:
    if _uses_mean_contact_reduction(a_is_human, a_part_name):
        return "mean"
    if _uses_mean_contact_reduction(b_is_human, b_part_name):
        return "mean"
    return "min"


def _select_contact_source_is_a(
    a_is_human: bool,
    a_vert_ids: np.ndarray,
    b_is_human: bool,
    b_vert_ids: np.ndarray,
) -> bool:
    # Match original code: human part is the primary set for human-object
    # edges.
    if a_is_human != b_is_human:
        return a_is_human
    # For object-object edges, use the smaller part as the primary contact set.
    return len(a_vert_ids) <= len(b_vert_ids)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
@dataclass
class LossResult:
    total: torch.Tensor
    prior: torch.Tensor
    contact: torch.Tensor
    dynamics: torch.Tensor
    penetration: torch.Tensor
    smooth: torch.Tensor
    human_prior: torch.Tensor
    human_smooth: torch.Tensor
    human_mask_2d: torch.Tensor
    object_mask_2d: torch.Tensor
    object_part_mask_2d: torch.Tensor
    object_scale_reg: torch.Tensor
    timings: dict[str, float]


LOSS_WEIGHT_ATTRS = {
    "prior": "lambda_prior",
    "contact": "lambda_contact",
    "dynamics": "lambda_dynamics",
    "penetration": "lambda_penetration",
    "smooth": "lambda_smooth",
    "human_prior": "lambda_human_prior",
    "human_smooth": "lambda_human_smooth",
    "human_mask_2d": "lambda_human_mask_2d",
    "object_mask_2d": "lambda_object_mask_2d",
    "object_part_mask_2d": "lambda_object_part_mask_2d",
    "object_scale_reg": "lambda_object_scale",
}
LOSS_TERM_KEYS = tuple(LOSS_WEIGHT_ATTRS.keys())


def _get_scaled_loss_terms(
    result: LossResult,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    scaled_terms: dict[str, torch.Tensor] = {}
    for key, attr in LOSS_WEIGHT_ATTRS.items():
        scaled_terms[key] = getattr(args, attr) * getattr(result, key)
    return scaled_terms


def _build_loss_row(
    iteration: int,
    result: LossResult,
    args: argparse.Namespace,
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    scaled_terms = _get_scaled_loss_terms(result, args)
    row: dict[str, Any] = {
        "iter": iteration,
        "total": float(result.total.item()),
    }

    for key in LOSS_TERM_KEYS:
        row[f"{key}_raw"] = float(getattr(result, key).item())
        row[f"{key}_scaled"] = float(scaled_terms[key].item())

    row.update(result.timings)
    if extra_metrics is not None:
        row.update(extra_metrics)
    return row


def _format_loss_log(
    iteration: int,
    total_iterations: int,
    result: LossResult,
    args: argparse.Namespace,
    elapsed_s: float,
) -> list[str]:
    scaled_terms = _get_scaled_loss_terms(result, args)
    obj_scaled = "  ".join(
        [
            f"prior={scaled_terms['prior'].item():.5f}",
            f"contact={scaled_terms['contact'].item():.5f}",
            f"dyn={scaled_terms['dynamics'].item():.5f}",
            f"pen={scaled_terms['penetration'].item():.5f}",
            f"smooth={scaled_terms['smooth'].item():.5f}",
            f"scale={scaled_terms['object_scale_reg'].item():.5f}",
        ]
    )
    aux_scaled = "  ".join(
        [
            f"hprior={scaled_terms['human_prior'].item():.5f}",
            f"hsmooth={scaled_terms['human_smooth'].item():.5f}",
            f"h2d={scaled_terms['human_mask_2d'].item():.5f}",
            f"obj2d={scaled_terms['object_mask_2d'].item():.5f}",
            f"part2d={scaled_terms['object_part_mask_2d'].item():.5f}",
        ]
    )
    obj_raw = "  ".join(
        [
            f"prior={result.prior.item():.5f}",
            f"contact={result.contact.item():.5f}",
            f"dyn={result.dynamics.item():.5f}",
            f"pen={result.penetration.item():.5f}",
            f"smooth={result.smooth.item():.5f}",
            f"scale={result.object_scale_reg.item():.5f}",
        ]
    )
    aux_raw = "  ".join(
        [
            f"hprior={result.human_prior.item():.5f}",
            f"hsmooth={result.human_smooth.item():.5f}",
            f"h2d={result.human_mask_2d.item():.5f}",
            f"obj2d={result.object_mask_2d.item():.5f}",
            f"part2d={result.object_part_mask_2d.item():.5f}",
        ]
    )
    return [
        f"  [{iteration:4d}/{total_iterations}] "
        f"total={result.total.item():.5f}  ({elapsed_s:.0f}s)",
        f"      scaled(obj): {obj_scaled}",
        f"      scaled(aux): {aux_scaled}",
        f"      raw(obj):    {obj_raw}",
        f"      raw(aux):    {aux_raw}",
    ]


def _subsample_indices(n: int, max_pts: int) -> np.ndarray:
    """Deterministic subsample of indices."""
    if n <= max_pts:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, max_pts).astype(np.int64)


def _compute_contact_loss(
    pts_src: torch.Tensor,   # [T, Ps, 3]
    pts_dst: torch.Tensor,   # [T, Pd, 3]
    is_continuous: bool,
    reduction: str,
) -> torch.Tensor:
    """Contact consistency loss between two part point clouds over time.

    Mirrors the original code's semantics:
    - human hand/foot contacts use mean NN distance over the source part
    - most other contacts use min NN distance over the source part
    - continuous edges average over time
    - non-continuous edges use the best frame

    knn_points returns SQUARED distances.
    """
    d_sq = knn_points(pts_src, pts_dst, K=1).dists[..., 0].clamp(min=0.0)
    if reduction == "mean":
        per_frame = d_sq.mean(dim=1)
    elif reduction == "min":
        per_frame = d_sq.min(dim=1).values
    else:
        raise ValueError(f"Unsupported contact reduction: {reduction}")

    if is_continuous:
        return per_frame.mean()
    else:
        return per_frame.min()


def _compute_dynamics_loss(
    pts_contact: torch.Tensor,   # [T, P, 3] — contact part in WORLD space
    obj_T_mats: torch.Tensor,        # [T, 4, 4]
    obj_scale: torch.Tensor,         # scalar
    is_rel_static: bool,
) -> torch.Tensor:
    """Contact dynamics loss — transform contact points into object canonical
    space and penalise motion (static) or acceleration (sliding).

    pts_contact: the body-part (or other-object-part) points in world space.
    obj_T_mats: object's per-frame SE(3) transforms (world ← canonical).
    """
    num_frames = pts_contact.shape[0]
    if num_frames < 2:
        return torch.tensor(0.0, device=pts_contact.device)

    canonical = _apply_inverse_similarity_sequence(
        pts_contact,
        obj_T_mats,
        obj_scale,
    )

    if is_rel_static:
        diff = canonical[1:] - canonical[:-1]  # [T-1, P, 3]
        return (diff ** 2).mean()
    else:
        if num_frames < 3:
            diff = canonical[1:] - canonical[:-1]
            return (diff ** 2).mean() * 0.1
        mid = canonical[1:-1]
        avg = 0.5 * (canonical[:-2] + canonical[2:])
        accel = mid - avg
        return (accel ** 2).mean()


def _compute_penetration_loss(
    human_points_t: torch.Tensor,  # [P, 3]
    obj_data: ObjectData,
    obj_T: torch.Tensor,           # [4, 4]
    obj_scale: torch.Tensor,       # scalar
) -> torch.Tensor:
    """SDF-based penetration loss for human points inside an object."""
    if obj_data.sdf_grid is None:
        return torch.tensor(0.0, device=human_points_t.device)

    pts_canon = _apply_inverse_similarity_batch(
        human_points_t,
        obj_T,
        obj_scale,
    )

    sdf_vals = _query_sdf(obj_data.sdf_grid, pts_canon)
    penetration = F.relu(-sdf_vals)
    n_inside = (penetration > 0).sum().clamp(min=1)
    return penetration.sum() / n_inside.float()


def _compute_obj_obj_penetration_loss(
    obj_a_world_points: torch.Tensor,  # [P, 3]
    obj_b: ObjectData,
    obj_b_T: torch.Tensor,            # [4, 4]
    obj_b_scale: torch.Tensor,        # scalar
) -> torch.Tensor:
    """Approximate object-object penetration using B's SDF at A's points."""
    if obj_b.sdf_grid is None:
        return torch.tensor(0.0, device=obj_a_world_points.device)

    pts_in_b_canon = _apply_inverse_similarity_batch(
        obj_a_world_points,
        obj_b_T,
        obj_b_scale,
    )

    sdf_vals = _query_sdf(obj_b.sdf_grid, pts_in_b_canon)
    penetration = F.relu(-sdf_vals)
    n_inside = (penetration > 0).sum().clamp(min=1)
    return penetration.sum() / n_inside.float()


def _compute_smoothness_loss(
    rotvecs: torch.Tensor,   # [T, 3]
    trans: torch.Tensor,     # [T, 3]
    is_translational: bool,
    is_rotational: bool,
) -> torch.Tensor:
    """Temporal smoothness conditioned on object motion state."""
    device = rotvecs.device
    T = rotvecs.shape[0]
    loss = torch.tensor(0.0, device=device)

    if T < 2:
        return loss

    # Rotation smoothness
    R_mats = axis_angle_to_matrix(rotvecs)  # [T, 3, 3]
    if is_rotational:
        # Allow rotation but penalise acceleration (non-smooth changes)
        if T >= 3:
            geo_dists = []
            for t in range(1, T - 1):
                # Interpolated rotation between t-1 and t+1
                # Simple: penalise second-order differences of axis-angle
                mid = 0.5 * (rotvecs[t - 1] + rotvecs[t + 1])
                diff = rotvecs[t] - mid
                geo_dists.append((diff ** 2).sum())
            loss = loss + torch.stack(geo_dists).mean()
        else:
            diff = rotvecs[1:] - rotvecs[:-1]
            loss = loss + (diff ** 2).mean()
    else:
        # Object shouldn't rotate — penalise any rotation change
        for t in range(T - 1):
            gd_sq = _geodesic_distance_sq(R_mats[t], R_mats[t + 1])
            loss = loss + 10.0 * gd_sq
        loss = loss / max(T - 1, 1)

    # Translation smoothness
    if is_translational:
        # Allow translation but penalise acceleration
        if T >= 3:
            accel = trans[2:] + trans[:-2] - 2.0 * trans[1:-1]
            loss = loss + (accel ** 2).mean()
        else:
            diff = trans[1:] - trans[:-1]
            loss = loss + (diff ** 2).mean()
    else:
        # Object shouldn't translate — penalise any translation change
        diff = trans[1:] - trans[:-1]
        loss = loss + 10.0 * (diff ** 2).mean()

    return loss


def _compute_bidirectional_2d_chamfer(
    observed_points: PackedPointCloud2D | None,
    model_points_world: torch.Tensor,   # [T, P, 3]
    k: torch.Tensor,                    # [3, 3]
    width: int,
    height: int,
) -> torch.Tensor:
    if observed_points is None:
        return torch.tensor(0.0, device=model_points_world.device)

    projected, valid = _project_points_normalized_torch(
        model_points_world,
        k,
        width,
        height,
    )
    model_packed, model_lengths = _pack_projected_points(projected, valid)
    valid_frames = (observed_points.lengths > 0) & (model_lengths > 0)
    if not torch.any(valid_frames):
        return torch.tensor(0.0, device=model_points_world.device)

    obs_pts = observed_points.points[valid_frames]
    obs_lengths = observed_points.lengths[valid_frames]
    model_pts = model_packed[valid_frames]
    model_lengths = model_lengths[valid_frames]

    obs_to_model = knn_points(
        obs_pts,
        model_pts,
        lengths1=obs_lengths,
        lengths2=model_lengths,
        K=1,
    ).dists[..., 0].clamp(min=0.0)
    model_to_obs = knn_points(
        model_pts,
        obs_pts,
        lengths1=model_lengths,
        lengths2=obs_lengths,
        K=1,
    ).dists[..., 0].clamp(min=0.0)
    loss_fwd = _masked_mean_from_lengths(obs_to_model, obs_lengths)
    loss_bwd = _masked_mean_from_lengths(model_to_obs, model_lengths)
    return 0.5 * (loss_fwd + loss_bwd)


def _bounded_log_scale_delta(
    raw_value: torch.Tensor,
    max_log_scale_delta: float,
) -> torch.Tensor:
    return math.fabs(max_log_scale_delta) * torch.tanh(raw_value.squeeze())


# ---------------------------------------------------------------------------
# Full loss aggregation
# ---------------------------------------------------------------------------
def _compute_all_losses(
    delta_rotvecs: dict[str, torch.Tensor],   # obj_slug → [T, 3]
    delta_trans: dict[str, torch.Tensor],      # obj_slug → [T, 3]
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    human_data: HumanData,
    human_delta_rotvecs: torch.Tensor,
    human_delta_trans: torch.Tensor,
    resolved_edges: list[ResolvedEdge],
    obj_keys: list[str],                       # ordered object slugs
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
    k: torch.Tensor,
    width: int,
    height: int,
) -> LossResult:
    device = human_data.base_verts.device
    num_frames = human_data.base_verts.shape[0]
    timings = {
        "time_contact_s": 0.0,
        "time_dynamics_s": 0.0,
        "time_penetration_s": 0.0,
        "time_mask2d_s": 0.0,
    }

    eff_T: dict[str, torch.Tensor] = {}
    eff_rotvecs: dict[str, torch.Tensor] = {}
    eff_trans: dict[str, torch.Tensor] = {}
    eff_scales: dict[str, torch.Tensor] = {}
    for slug in obj_keys:
        od = objects[slug]
        T_list = []
        rvs = []
        trs = []
        for t in range(num_frames):
            base = torch.from_numpy(od.tracked_poses[t]).float().to(device)
            delta = _compose_T(delta_rotvecs[slug][t], delta_trans[slug][t])
            T_eff = base @ delta
            T_list.append(T_eff)
            R_eff = T_eff[:3, :3]
            rv = matrix_to_axis_angle(R_eff.unsqueeze(0)).squeeze(0)
            rvs.append(rv)
            trs.append(T_eff[:3, 3])
        eff_T[slug] = torch.stack(T_list, dim=0)
        eff_rotvecs[slug] = torch.stack(rvs)
        eff_trans[slug] = torch.stack(trs)
        if args.optimize_object_scale:
            eff_scales[slug] = torch.exp(
                _bounded_log_scale_delta(
                    raw_scale_deltas[slug],
                    args.max_log_scale_delta,
                )
            )
        else:
            eff_scales[slug] = torch.tensor(1.0, device=device)

    if args.optimize_human:
        human_points_whole = _apply_local_se3_sequence(
            human_data.sampled_points_base,
            human_delta_rotvecs,
            human_delta_trans,
            human_data.centers,
        )
    else:
        human_points_whole = human_data.sampled_points_base

    human_part_cache: dict[str, torch.Tensor] = {}

    def _get_human_part_points(part_name: str) -> torch.Tensor:
        if part_name not in human_part_cache:
            part_points = human_data.part_points_base[part_name]
            if args.optimize_human:
                human_part_cache[part_name] = _apply_local_se3_sequence(
                    part_points,
                    human_delta_rotvecs,
                    human_delta_trans,
                    human_data.centers,
                )
            else:
                human_part_cache[part_name] = part_points
        return human_part_cache[part_name]

    object_points_cache: dict[tuple[str, str], torch.Tensor] = {}

    def _get_object_points(
        slug: str,
        part_name: str | None = None,
    ) -> torch.Tensor:
        key = (slug, part_name or "__whole__")
        if key not in object_points_cache:
            od = objects[slug]
            if part_name and part_name in od.part_sampled_points:
                base_points = od.part_sampled_points[part_name]
            else:
                base_points = od.sampled_points
            object_points_cache[key] = _apply_similarity_sequence(
                base_points,
                eff_T[slug],
                eff_scales[slug],
            )
        return object_points_cache[key]

    # Object priors and regularisation.
    loss_prior = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        loss_prior = loss_prior + (delta_rotvecs[slug] ** 2).sum()
        loss_prior = loss_prior + (delta_trans[slug] ** 2).sum()
    n_params = sum(
        delta_rotvecs[s].numel() + delta_trans[s].numel()
        for s in obj_keys
    )
    loss_prior = loss_prior / max(n_params, 1)

    loss_object_scale_reg = torch.tensor(0.0, device=device)
    if args.optimize_object_scale and obj_keys:
        for slug in obj_keys:
            delta_log_scale = _bounded_log_scale_delta(
                raw_scale_deltas[slug],
                args.max_log_scale_delta,
            )
            loss_object_scale_reg = (
                loss_object_scale_reg + delta_log_scale.pow(2)
            )
        loss_object_scale_reg = loss_object_scale_reg / len(obj_keys)

    # Human priors.
    if args.optimize_human:
        loss_human_prior = (
            human_delta_rotvecs.pow(2).sum() + human_delta_trans.pow(2).sum()
        ) / float(human_delta_rotvecs.numel() + human_delta_trans.numel())
        loss_human_smooth = _compute_smoothness_loss(
            human_delta_rotvecs,
            human_delta_trans,
            is_translational=True,
            is_rotational=True,
        )
    else:
        loss_human_prior = torch.tensor(0.0, device=device)
        loss_human_smooth = torch.tensor(0.0, device=device)

    # Contact consistency loss.
    t_contact = time.perf_counter()
    loss_contact = torch.tensor(0.0, device=device)
    n_edges_contact = 0
    for edge in resolved_edges:
        if edge.a_is_human:
            pts_a = _get_human_part_points(edge.a_part_name)
        else:
            pts_a = _get_object_points(
                obj_keys[edge.a_object_idx],
                edge.a_part_name,
            )
        if edge.b_is_human:
            pts_b = _get_human_part_points(edge.b_part_name)
        else:
            pts_b = _get_object_points(
                obj_keys[edge.b_object_idx],
                edge.b_part_name,
            )
        pts_src = pts_a if edge.contact_source_is_a else pts_b
        pts_dst = pts_b if edge.contact_source_is_a else pts_a

        loss_contact = loss_contact + _compute_contact_loss(
            pts_src, pts_dst, edge.is_continuous, edge.contact_reduction
        )
        n_edges_contact += 1
    if n_edges_contact > 0:
        loss_contact = loss_contact / n_edges_contact
    timings["time_contact_s"] = time.perf_counter() - t_contact

    # Contact dynamics loss.
    t_dynamics = time.perf_counter()
    loss_dynamics = torch.tensor(0.0, device=device)
    n_edges_dyn = 0
    for edge in resolved_edges:
        if edge.canonical_obj_idx < 0:
            continue

        ref_slug = obj_keys[edge.canonical_obj_idx]
        ref_T = eff_T[ref_slug]
        ref_scale = eff_scales[ref_slug]

        if edge.canonical_obj_idx == edge.a_object_idx:
            if edge.b_is_human:
                pts_contact = _get_human_part_points(edge.b_part_name)
            else:
                other_slug = obj_keys[edge.b_object_idx]
                pts_contact = _get_object_points(other_slug, edge.b_part_name)
        else:
            if edge.a_is_human:
                pts_contact = _get_human_part_points(edge.a_part_name)
            else:
                other_slug = obj_keys[edge.a_object_idx]
                pts_contact = _get_object_points(other_slug, edge.a_part_name)

        loss_dynamics = loss_dynamics + _compute_dynamics_loss(
            pts_contact,
            ref_T,
            ref_scale,
            edge.is_rel_static,
        )
        n_edges_dyn += 1
    if n_edges_dyn > 0:
        loss_dynamics = loss_dynamics / n_edges_dyn
    timings["time_dynamics_s"] = time.perf_counter() - t_dynamics

    # Penetration loss (annealed).
    pen_progress = min(iteration / max(total_iters * 0.5, 1.0), 1.0)
    pen_weight_schedule = pen_progress

    t_penetration = time.perf_counter()
    loss_pen = torch.tensor(0.0, device=device)
    n_pen = 0
    for t in range(num_frames):
        human_sub = human_points_whole[t]
        for slug in obj_keys:
            od = objects[slug]
            loss_pen = loss_pen + _compute_penetration_loss(
                human_sub,
                od,
                eff_T[slug][t],
                eff_scales[slug],
            )
            n_pen += 1
    for i in range(len(obj_keys)):
        for j in range(i + 1, len(obj_keys)):
            obj_a_points = _get_object_points(obj_keys[i])
            obj_b_points = _get_object_points(obj_keys[j])
            for t in range(num_frames):
                loss_pen = loss_pen + _compute_obj_obj_penetration_loss(
                    obj_a_points[t],
                    objects[obj_keys[j]],
                    eff_T[obj_keys[j]][t],
                    eff_scales[obj_keys[j]],
                )
                n_pen += 1
                loss_pen = loss_pen + _compute_obj_obj_penetration_loss(
                    obj_b_points[t],
                    objects[obj_keys[i]],
                    eff_T[obj_keys[i]][t],
                    eff_scales[obj_keys[i]],
                )
                n_pen += 1
    if n_pen > 0:
        loss_pen = loss_pen / n_pen
    loss_pen = loss_pen * pen_weight_schedule
    timings["time_penetration_s"] = time.perf_counter() - t_penetration

    # Object smoothness loss.
    loss_smooth = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        od = objects[slug]
        loss_smooth = loss_smooth + _compute_smoothness_loss(
            eff_rotvecs[slug],
            eff_trans[slug],
            od.state.is_translational,
            od.state.is_rotational,
        )
    if len(obj_keys) > 0:
        loss_smooth = loss_smooth / len(obj_keys)

    # 2D mask chamfer losses.
    t_mask2d = time.perf_counter()
    if args.optimize_human:
        loss_human_mask_2d = _compute_bidirectional_2d_chamfer(
            human_data.mask_points_2d,
            human_points_whole,
            k,
            width,
            height,
        )
    else:
        loss_human_mask_2d = torch.tensor(0.0, device=device)

    loss_object_mask_2d = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        loss_object_mask_2d = (
            loss_object_mask_2d
            + _compute_bidirectional_2d_chamfer(
                objects[slug].mask_points_2d,
                _get_object_points(slug),
                k,
                width,
                height,
            )
        )
    if obj_keys:
        loss_object_mask_2d = loss_object_mask_2d / len(obj_keys)

    loss_object_part_mask_2d = torch.tensor(0.0, device=device)
    num_part_terms = 0
    for slug in obj_keys:
        for part_name, packed_points in (
            objects[slug].part_mask_points_2d.items()
        ):
            loss_object_part_mask_2d = (
                loss_object_part_mask_2d
                + _compute_bidirectional_2d_chamfer(
                    packed_points,
                    _get_object_points(slug, part_name),
                    k,
                    width,
                    height,
                )
            )
            num_part_terms += 1
    if num_part_terms > 0:
        loss_object_part_mask_2d = loss_object_part_mask_2d / num_part_terms
    timings["time_mask2d_s"] = time.perf_counter() - t_mask2d

    total = (
        args.lambda_prior * loss_prior
        + args.lambda_contact * loss_contact
        + args.lambda_dynamics * loss_dynamics
        + args.lambda_penetration * loss_pen
        + args.lambda_smooth * loss_smooth
        + args.lambda_human_prior * loss_human_prior
        + args.lambda_human_smooth * loss_human_smooth
        + args.lambda_human_mask_2d * loss_human_mask_2d
        + args.lambda_object_mask_2d * loss_object_mask_2d
        + args.lambda_object_part_mask_2d * loss_object_part_mask_2d
        + args.lambda_object_scale * loss_object_scale_reg
    )

    return LossResult(
        total=total,
        prior=loss_prior,
        contact=loss_contact,
        dynamics=loss_dynamics,
        penetration=loss_pen,
        smooth=loss_smooth,
        human_prior=loss_human_prior,
        human_smooth=loss_human_smooth,
        human_mask_2d=loss_human_mask_2d,
        object_mask_2d=loss_object_mask_2d,
        object_part_mask_2d=loss_object_part_mask_2d,
        object_scale_reg=loss_object_scale_reg,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# Pose saving (same format as track_object_mesh.py)
# ---------------------------------------------------------------------------
def _save_pose_json(
    path: Path,
    T_mats: np.ndarray,
    frame_offset: int = 0,
) -> None:
    rows = []
    for i in range(T_mats.shape[0]):
        rows.append(
            {
                "frame": int(frame_offset + i),
                "T_4x4": T_mats[i].tolist(),
            }
        )
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def _save_transform_json(
    path: Path,
    global_scale: float,
    T_mats: np.ndarray,
    frame_offset: int = 0,
) -> None:
    rows = []
    for i in range(T_mats.shape[0]):
        rows.append(
            {
                "frame": int(frame_offset + i),
                "T_4x4": T_mats[i].tolist(),
            }
        )
    payload = {
        "global_scale": float(global_scale),
        "frames": rows,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_mesh_sequence(
    verts_template: np.ndarray,
    faces: np.ndarray,
    T_mats: np.ndarray,
    meshes_dir: Path,
    global_scale: float = 1.0,
    frame_offset: int = 0,
) -> None:
    ensure_dir(meshes_dir)
    mesh_tmpl = trimesh.Trimesh(
        vertices=verts_template,
        faces=faces,
        process=False,
    )
    for i in range(T_mats.shape[0]):
        R = T_mats[i, :3, :3]
        t = T_mats[i, :3, 3]
        verts_t = ((verts_template * global_scale) @ R.T) + t[None, :]
        mesh = mesh_tmpl.copy()
        mesh.vertices = verts_t
        mesh.export(str(meshes_dir / f"frame_{frame_offset + i:04d}.ply"))


def _save_human_mesh_sequence(
    verts_seq: np.ndarray,
    faces: np.ndarray,
    meshes_dir: Path,
    frame_offset: int = 0,
) -> None:
    ensure_dir(meshes_dir)
    mesh_tmpl = trimesh.Trimesh(
        vertices=verts_seq[0],
        faces=faces,
        process=False,
    )
    for i in range(verts_seq.shape[0]):
        mesh = mesh_tmpl.copy()
        mesh.vertices = verts_seq[i]
        mesh.export(str(meshes_dir / f"frame_{frame_offset + i:04d}.ply"))


# ---------------------------------------------------------------------------
# Loss plot saving
# ---------------------------------------------------------------------------
def _save_joint_loss_plots(
    debug_dir: Path,
    iter_rows: list[dict[str, Any]],
) -> None:
    """Save loss-term plots for the joint optimisation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return  # skip if matplotlib unavailable

    if not iter_rows:
        return

    ensure_dir(debug_dir)
    iterations = [r["iter"] for r in iter_rows]
    raw_keys = [f"{key}_raw" for key in LOSS_TERM_KEYS]
    scaled_keys = [f"{key}_scaled" for key in LOSS_TERM_KEYS]
    plot_groups = [
        ("loss_total", ["total"], "Total Loss"),
        ("loss_all_raw_terms", raw_keys, "Raw Loss Terms"),
        ("loss_all_scaled_terms", scaled_keys, "Scaled Loss Terms"),
    ]

    for prefix, keys, title in plot_groups:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key in keys:
            vals = [float(r.get(key, 0.0)) for r in iter_rows]
            ax.plot(iterations, vals, linewidth=1.2, label=key)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        if len(keys) > 1:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(debug_dir / f"{prefix}.png"), dpi=140)
        plt.close(fig)

    for key in raw_keys + scaled_keys:
        vals = [float(r.get(key, 0.0)) for r in iter_rows]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(iterations, vals, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(key)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(debug_dir / f"{key}.png"), dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------
def _render_joint_overlay(
    frame_paths: list[Path],
    human_verts_np: np.ndarray,     # [T, V_h, 3]
    human_faces: np.ndarray,        # [F_h, 3]
    objects: dict[str, ObjectData],
    obj_keys: list[str],
    final_T_mats: dict[str, np.ndarray],  # slug → [T, 4, 4]
    final_scales: dict[str, float],
    k: np.ndarray,
    out_dir: Path,
    fps: float,
    save_pngs: bool = False,
) -> None:
    """Render human + all objects overlaid on video frames."""
    T = human_verts_np.shape[0]
    if not frame_paths or len(frame_paths) < T:
        print("[WARN] Not enough frames for overlay rendering.")
        return

    overlays_dir = out_dir / "overlays"
    if save_pngs:
        ensure_dir(overlays_dir)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print("[WARN] Cannot read first frame for overlay.")
        return
    h, w = first_frame.shape[:2]

    writer = start_ffmpeg_writer(out_dir / "overlay.mp4", fps, (h, w))
    try:
        for t in range(T):
            if t >= len(frame_paths):
                break
            frame = cv2.imread(str(frame_paths[t]))
            if frame is None:
                continue

            # Draw human
            overlay = draw_overlay(
                frame_bgr=frame,
                verts_cv=human_verts_np[t],
                faces=human_faces,
                k=k,
                fill_alpha=OVERLAY_FILL_ALPHA * 0.6,
                contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                color_bgr=HUMAN_COLOR_BGR,
            )

            # Draw each object
            for slug in obj_keys:
                od = objects[slug]
                verts_t = (
                    (od.template_verts.cpu().numpy() * final_scales[slug])
                    @ final_T_mats[slug][t, :3, :3].T
                    + final_T_mats[slug][t, :3, 3][None, :]
                )
                overlay = draw_overlay(
                    frame_bgr=overlay,
                    verts_cv=verts_t.astype(np.float32),
                    faces=od.faces,
                    k=k,
                    fill_alpha=OVERLAY_FILL_ALPHA,
                    contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                    color_bgr=od.color_bgr,
                )

            if save_pngs:
                cv2.imwrite(
                    str(overlays_dir / f"overlay_{t:04d}.png"),
                    overlay,
                )
            if writer.stdin is not None:
                writer.stdin.write(np.ascontiguousarray(overlay).tobytes())
    finally:
        close_ffmpeg(writer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    dirs = _resolve_dirs(args, script_dir)
    pag_path = _resolve_pag_path(args, script_dir)
    smpl_seg_path = _resolve_smpl_seg(args, script_dir)
    device = _to_device(args.device)

    out_dir = dirs["output"]
    ensure_dir(out_dir)

    # ── Camera intrinsics ──
    k, intr_path = _load_intrinsics_from_alignment_summary(dirs["aligned"])

    # ── PAG ──
    pag = _parse_pag(pag_path)

    # ── SMPL body segmentation ──
    body_seg = _load_smpl_body_seg(smpl_seg_path)
    width, height = _infer_image_size(dirs)
    k_torch = torch.from_numpy(k).float().to(device)

    # ── Human mesh sequence ──
    human_aligned_dir = dirs["aligned"] / "human_motion_aligned"
    if not human_aligned_dir.exists():
        raise FileNotFoundError(
            f"Human motion aligned dir missing: {human_aligned_dir}"
        )

    human_ply_paths = sorted(
        human_aligned_dir.glob("frame_*.ply"),
        key=lambda p: int(p.stem.split("_")[-1])
    )
    if not human_ply_paths:
        raise FileNotFoundError(f"No frame_*.ply in {human_aligned_dir}")

    print(f"Loading {len(human_ply_paths)} human mesh frames...")
    human_meshes = [
        trimesh.load(str(p), process=False) for p in human_ply_paths
    ]
    human_verts_np = np.stack(
        [np.asarray(m.vertices, dtype=np.float32) for m in human_meshes]
    )  # [T, V_h, 3]
    human_faces = np.asarray(human_meshes[0].faces, dtype=np.int32)
    num_frames = human_verts_np.shape[0]
    human_data = _load_human_data(
        human_verts_np=human_verts_np,
        human_faces=human_faces,
        body_seg=body_seg,
        dirs=dirs,
        width=width,
        height=height,
        device=device,
        args=args,
    )

    print(f"  Human: {num_frames} frames, {human_verts_np.shape[1]} verts, "
          f"{human_faces.shape[0]} faces")

    # ── Objects ──
    # obj_slug_to_state = {s.slug: s for s in pag.object_states}
    objects: dict[str, ObjectData] = {}
    obj_keys: list[str] = []

    for idx, state in enumerate(pag.object_states):
        slug = state.slug
        mesh_path = dirs["aligned"] / "meshes" / f"{slug}.ply"
        poses_path = dirs["tracked"] / slug / "poses.json"

        if not mesh_path.exists():
            print(f"  [SKIP] {slug}: aligned mesh not found at {mesh_path}")
            continue
        if not poses_path.exists():
            print(f"  [SKIP] {slug}: tracked poses not found at {poses_path}")
            continue

        mesh = trimesh.load(str(mesh_path), process=False)
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        tracked_poses = _load_tracked_poses(poses_path)

        # Truncate/pad to match human frames
        if tracked_poses.shape[0] > num_frames:
            tracked_poses = tracked_poses[:num_frames]
        elif tracked_poses.shape[0] < num_frames:
            pad = np.tile(
                tracked_poses[-1:],
                (num_frames - tracked_poses.shape[0], 1, 1),
            )
            tracked_poses = np.concatenate([tracked_poses, pad], axis=0)

        # Decompose tracked poses to rotvec + trans for initialisation
        rvs = []
        trs = []
        for t in range(num_frames):
            rv, tr = _decompose_T(tracked_poses[t])
            rvs.append(rv)
            trs.append(tr)

        # Load part segmentation
        try:
            part_segments = _load_object_part_segments(
                dirs["seg_obj"],
                slug,
                faces,
            )
        except FileNotFoundError:
            print(
                f"  [WARN] {slug}: part segmentation not found, "
                "using whole mesh."
            )
            part_segments = ObjectPartSegments(vert_ids={}, face_ids={})

        sampled_points = torch.from_numpy(
            _sample_surface_points(
                verts,
                faces,
                args.num_object_surface_points,
                SURFACE_SAMPLE_SEED + idx,
            )
        ).float().to(device)
        part_sampled_points: dict[str, torch.Tensor] = {}
        for part_idx, (part_name, face_ids) in enumerate(
            sorted(part_segments.face_ids.items())
        ):
            part_faces = faces[face_ids]
            part_sampled = _sample_surface_points(
                verts,
                part_faces,
                args.num_part_surface_points,
                SURFACE_SAMPLE_SEED + 97 * idx + part_idx + 1,
            )
            part_sampled_points[part_name] = (
                torch.from_numpy(part_sampled).float().to(device)
            )

        object_mask_points, part_mask_points = _load_object_mask_targets(
            dirs["seg_vid"],
            slug,
            list(part_segments.face_ids.keys()),
            num_frames,
            width,
            height,
            device,
            args.num_mask_points_2d,
        )

        # Build SDF
        print(
            f"  Building SDF for {slug} ({verts.shape[0]} verts, "
            f"res={args.sdf_resolution})..."
        )
        t0 = time.time()
        sdf_grid = _build_sdf_grid(verts, faces, args.sdf_resolution, device)
        print(f"    SDF done in {time.time() - t0:.1f}s")

        color = OBJECT_COLORS_BGR[idx % len(OBJECT_COLORS_BGR)]

        objects[slug] = ObjectData(
            name=state.name,
            slug=slug,
            state=state,
            template_verts=torch.from_numpy(verts).float().to(device),
            faces=faces,
            faces_torch=torch.from_numpy(faces.astype(np.int64)).to(device),
            tracked_poses=tracked_poses,
            tracked_rotvecs=torch.from_numpy(np.stack(rvs)).float().to(device),
            tracked_trans=torch.from_numpy(np.stack(trs)).float().to(device),
            part_vert_ids=part_segments.vert_ids,
            part_face_ids=part_segments.face_ids,
            sampled_points=sampled_points,
            part_sampled_points=part_sampled_points,
            mask_points_2d=object_mask_points,
            part_mask_points_2d=part_mask_points,
            sdf_grid=sdf_grid,
            color_bgr=color,
        )
        obj_keys.append(slug)
        part_names = ", ".join(part_segments.vert_ids.keys())
        print(
            f"  Loaded {slug}: {verts.shape[0]} verts, "
            f"{faces.shape[0]} faces, {len(part_segments.vert_ids)} parts "
            f"({part_names})"
        )

    if not obj_keys:
        raise RuntimeError("No objects loaded — nothing to optimise.")

    # ── Resolve PAG edges ──
    print("\nResolving PAG edges...")
    obj_slug_to_idx = {s: i for i, s in enumerate(obj_keys)}
    resolved_edges: list[ResolvedEdge] = []

    for edge in pag.edges:
        try:
            a_entity, a_part = _parse_node(edge.node_a)
            b_entity, b_part = _parse_node(edge.node_b)
        except ValueError as exc:
            print(f"  [WARN] Skipping edge: {exc}")
            continue

        a_is_human = _is_human_node(edge.node_a)
        b_is_human = _is_human_node(edge.node_b)

        # Resolve side A vertex IDs
        if a_is_human:
            a_obj_idx = -1
            a_part_norm = a_part.lower().strip()
            if a_part_norm not in body_seg:
                print(
                    f"  [WARN] Body part '{a_part_norm}' not in segmentation, "
                    "skipping edge."
                )
                continue
            a_vids = body_seg[a_part_norm]
        else:
            a_slug = _sanitize(a_entity)
            if a_slug not in obj_slug_to_idx:
                print(f"  [WARN] Object '{a_slug}' not loaded, skipping edge.")
                continue
            a_obj_idx = obj_slug_to_idx[a_slug]
            a_part_norm = a_part.lower().strip()
            part_verts = objects[a_slug].part_vert_ids
            # Try exact match, then case-insensitive
            matched = None
            for pname, pvids in part_verts.items():
                if pname.lower().strip() == a_part_norm:
                    matched = pvids
                    break
            if matched is None:
                # Fall back to whole mesh
                print(
                    f"  [WARN] Part '{a_part}' not found in {a_slug}, "
                    "using whole mesh."
                )
                matched = np.arange(objects[a_slug].template_verts.shape[0])
            a_vids = matched

        # Resolve side B vertex IDs
        if b_is_human:
            b_obj_idx = -1
            b_part_norm = b_part.lower().strip()
            if b_part_norm not in body_seg:
                print(
                    f"  [WARN] Body part '{b_part_norm}' not in segmentation, "
                    "skipping edge."
                )
                continue
            b_vids = body_seg[b_part_norm]
        else:
            b_slug = _sanitize(b_entity)
            if b_slug not in obj_slug_to_idx:
                print(f"  [WARN] Object '{b_slug}' not loaded, skipping edge.")
                continue
            b_obj_idx = obj_slug_to_idx[b_slug]
            b_part_norm = b_part.lower().strip()
            part_verts = objects[b_slug].part_vert_ids
            matched = None
            for pname, pvids in part_verts.items():
                if pname.lower().strip() == b_part_norm:
                    matched = pvids
                    break
            if matched is None:
                print(
                    f"  [WARN] Part '{b_part}' not found in {b_slug}, "
                    "using whole mesh."
                )
                matched = np.arange(objects[b_slug].template_verts.shape[0])
            b_vids = matched

        contact_reduction = _select_contact_reduction(
            a_is_human, a_part_norm, b_is_human, b_part_norm
        )
        contact_source_is_a = _select_contact_source_is_a(
            a_is_human, a_vids, b_is_human, b_vids
        )
        canonical_obj_idx = _select_canonical_reference_obj(
            a_is_human, a_obj_idx, b_is_human, b_obj_idx, objects, obj_keys
        )

        resolved_edges.append(ResolvedEdge(
            a_is_human=a_is_human,
            a_object_idx=a_obj_idx,
            a_vert_ids=a_vids,
            a_part_name=a_part_norm,
            b_is_human=b_is_human,
            b_object_idx=b_obj_idx,
            b_vert_ids=b_vids,
            b_part_name=b_part_norm,
            is_continuous=edge.is_continuous,
            is_rel_static=edge.is_rel_static,
            contact_reduction=contact_reduction,
            contact_source_is_a=contact_source_is_a,
            canonical_obj_idx=canonical_obj_idx,
        ))
        a_label = (
            f"human:{a_part}"
            if a_is_human
            else f"{obj_keys[a_obj_idx]}:{a_part}"
        )
        b_label = (
            f"human:{b_part}"
            if b_is_human
            else f"{obj_keys[b_obj_idx]}:{b_part}"
        )
        ref_label = (
            "none" if canonical_obj_idx < 0 else obj_keys[canonical_obj_idx]
        )
        print(f"  Edge: {a_label} ↔ {b_label}  "
              f"(continuous={edge.is_continuous}, "
              f"static={edge.is_rel_static}, "
              f"contact={contact_reduction}, ref={ref_label})")

    print(f"  → {len(resolved_edges)} edges resolved.\n")

    # ── Initialise optimisation parameters ──
    delta_rotvecs: dict[str, torch.Tensor] = {}
    delta_trans: dict[str, torch.Tensor] = {}
    raw_scale_deltas: dict[str, torch.Tensor] = {}
    params: list[torch.Tensor] = []

    for slug in obj_keys:
        dr = torch.zeros(num_frames, 3, device=device, requires_grad=True)
        dt = torch.zeros(num_frames, 3, device=device, requires_grad=True)
        ds = torch.zeros(
            1,
            device=device,
            requires_grad=args.optimize_object_scale,
        )
        delta_rotvecs[slug] = dr
        delta_trans[slug] = dt
        raw_scale_deltas[slug] = ds
        params.extend([dr, dt])
        if args.optimize_object_scale:
            params.append(ds)

    human_delta_rotvecs = torch.zeros(
        num_frames,
        3,
        device=device,
        requires_grad=args.optimize_human,
    )
    human_delta_trans = torch.zeros(
        num_frames,
        3,
        device=device,
        requires_grad=args.optimize_human,
    )
    if args.optimize_human:
        params.extend([human_delta_rotvecs, human_delta_trans])

    # ── Print summary ──
    print("=" * 60)
    print("Joint Human–Object Mesh Refinement")
    print(f"  video:    {args.video_name}")
    print(f"  device:   {device}")
    print(f"  frames:   {num_frames}")
    print(f"  objects:  {', '.join(obj_keys)}")
    print(f"  edges:    {len(resolved_edges)}")
    print(
        f"  human:    {'optimised' if args.optimize_human else 'fixed'}"
        f"  object_scale: {'on' if args.optimize_object_scale else 'off'}"
    )
    print(
        f"  K: fx={k[0, 0]:.1f}  fy={k[1, 1]:.1f}  "
        f"cx={k[0, 2]:.1f}  cy={k[1, 2]:.1f}"
    )
    print(
        f"  λ_prior={args.lambda_prior}  λ_contact={args.lambda_contact}  "
        f"λ_dyn={args.lambda_dynamics}  λ_pen={args.lambda_penetration}  "
        f"λ_smooth={args.lambda_smooth}"
    )
    print(
        f"  λ_hprior={args.lambda_human_prior}  "
        f"λ_hsmooth={args.lambda_human_smooth}  "
        f"λ_h2d={args.lambda_human_mask_2d}"
    )
    print(
        f"  λ_obj2d={args.lambda_object_mask_2d}  "
        f"λ_part2d={args.lambda_object_part_mask_2d}  "
        f"λ_scale={args.lambda_object_scale}"
    )
    print("=" * 60)

    # ── Adam optimisation ──
    total_iters = args.adam_iters
    optimizer = torch.optim.Adam(params, lr=args.adam_lr)
    iter_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    best_human_state: dict[str, torch.Tensor] = {}
    no_improve_iters = 0
    early_stop_triggered = False
    early_stop_enabled = args.early_stop_patience > 0

    print(f"\n[Adam] {args.adam_iters} iterations, lr={args.adam_lr}")
    t_start = time.time()

    for it in range(args.adam_iters):
        iter_t0 = time.perf_counter()
        optimizer.zero_grad()
        result = _compute_all_losses(
            delta_rotvecs,
            delta_trans,
            raw_scale_deltas,
            objects,
            human_data,
            human_delta_rotvecs,
            human_delta_trans,
            resolved_edges,
            obj_keys,
            args,
            iteration=it,
            total_iters=total_iters,
            k=k_torch,
            width=width,
            height=height,
        )
        backward_t0 = time.perf_counter()
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        backward_time = time.perf_counter() - backward_t0
        iter_time = time.perf_counter() - iter_t0

        loss_val = result.total.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            if best_state:
                for slug in obj_keys:
                    delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
                    delta_trans[slug].data.copy_(best_state[slug]["dt"])
                    raw_scale_deltas[slug].data.copy_(best_state[slug]["ds"])
                if best_human_state:
                    human_delta_rotvecs.data.copy_(best_human_state["dr"])
                    human_delta_trans.data.copy_(best_human_state["dt"])
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                print(
                    f"  [{it:4d}] NaN detected; restored best state and "
                    "halved lr."
                )
            continue
        rel_improve = (
            (best_loss - loss_val) / max(abs(best_loss), 1e-8)
            if math.isfinite(best_loss)
            else float("inf")
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                slug: {
                    "dr": delta_rotvecs[slug].detach().clone(),
                    "dt": delta_trans[slug].detach().clone(),
                    "ds": raw_scale_deltas[slug].detach().clone(),
                }
                for slug in obj_keys
            }
            best_human_state = {
                "dr": human_delta_rotvecs.detach().clone(),
                "dt": human_delta_trans.detach().clone(),
            }
            if rel_improve > args.early_stop_rel_improve:
                no_improve_iters = 0
            elif early_stop_enabled and it >= args.early_stop_start:
                no_improve_iters += 1
        elif early_stop_enabled and it >= args.early_stop_start:
            no_improve_iters += 1

        if it % args.log_interval == 0 or it == args.adam_iters - 1:
            row = _build_loss_row(
                it,
                result,
                args,
                extra_metrics={
                    "time_backward_s": backward_time,
                    "time_total_iter_s": iter_time,
                },
            )
            iter_rows.append(row)
            if (
                args.verbose
                or it % (args.log_interval * 4) == 0
                or it == args.adam_iters - 1
            ):
                elapsed = time.time() - t_start
                for line in _format_loss_log(
                    it, args.adam_iters, result, args, elapsed
                ):
                    print(line)

        if early_stop_enabled and no_improve_iters >= args.early_stop_patience:
            early_stop_triggered = True
            print(
                f"  Early stop at iter {it}: no relative improvement "
                f"greater than {args.early_stop_rel_improve:.1e} for "
                f"{args.early_stop_patience} iterations."
            )
            break

    # ── Restore best parameters ──
    if best_state:
        for slug in obj_keys:
            delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
            delta_trans[slug].data.copy_(best_state[slug]["dt"])
            raw_scale_deltas[slug].data.copy_(best_state[slug]["ds"])
    if best_human_state:
        human_delta_rotvecs.data.copy_(best_human_state["dr"])
        human_delta_trans.data.copy_(best_human_state["dt"])

    total_time = time.time() - t_start
    print(
        f"\nOptimisation complete in {total_time:.1f}s. "
        f"Best loss: {best_loss:.6f}"
    )

    # ── Extract final poses ──
    final_T_mats: dict[str, np.ndarray] = {}
    final_scales: dict[str, float] = {}

    for slug in obj_keys:
        od = objects[slug]
        T_out = np.zeros((num_frames, 4, 4), dtype=np.float32)
        for t in range(num_frames):
            base = torch.from_numpy(od.tracked_poses[t]).float().to(device)
            delta = _compose_T(delta_rotvecs[slug][t].detach(),
                               delta_trans[slug][t].detach())
            T_eff = (base @ delta).cpu().numpy()
            T_out[t] = T_eff
        final_T_mats[slug] = T_out
        final_scales[slug] = float(
            torch.exp(
                _bounded_log_scale_delta(
                    raw_scale_deltas[slug].detach(),
                    args.max_log_scale_delta,
                )
            ).item()
        )

    if args.optimize_human:
        final_human_verts_np = (
            _apply_local_se3_sequence(
                human_data.base_verts,
                human_delta_rotvecs.detach(),
                human_delta_trans.detach(),
                human_data.centers,
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        final_human_verts_np = human_verts_np.copy()

    # ── Save outputs ──
    print("\nSaving outputs...")

    # Per-object outputs
    for slug in obj_keys:
        od = objects[slug]
        obj_dir = out_dir / slug
        ensure_dir(obj_dir)

        # Refined poses
        _save_pose_json(obj_dir / "poses_refined.json", final_T_mats[slug])
        _save_transform_json(
            obj_dir / "transform_refined.json",
            final_scales[slug],
            final_T_mats[slug],
        )

        # Also save the original tracked poses for comparison
        _save_pose_json(obj_dir / "poses_original.json", od.tracked_poses)

        # Per-frame meshes
        _save_mesh_sequence(
            od.template_verts.cpu().numpy(),
            od.faces,
            final_T_mats[slug],
            obj_dir / "meshes",
            global_scale=final_scales[slug],
        )

        # Delta summary
        delta_log_scale = float(
            _bounded_log_scale_delta(
                raw_scale_deltas[slug].detach(),
                args.max_log_scale_delta,
            ).item()
        )
        delta_stats = {
            "slug": slug,
            "max_delta_rot_deg": float(
                delta_rotvecs[slug].detach().norm(dim=-1).max().item()
                * 180.0 / math.pi
            ),
            "mean_delta_rot_deg": float(
                delta_rotvecs[slug].detach().norm(dim=-1).mean().item()
                * 180.0 / math.pi
            ),
            "max_delta_trans_m": float(
                delta_trans[slug].detach().norm(dim=-1).max().item()
            ),
            "mean_delta_trans_m": float(
                delta_trans[slug].detach().norm(dim=-1).mean().item()
            ),
            "delta_log_scale": delta_log_scale,
            "global_scale": final_scales[slug],
        }
        with (obj_dir / "delta_stats.json").open("w", encoding="utf-8") as f:
            json.dump(delta_stats, f, indent=2)

        print(
            f"  {slug}: max Δrot={delta_stats['max_delta_rot_deg']:.2f}°, "
            f"max Δtrans={delta_stats['max_delta_trans_m']:.4f}m, "
            f"scale={delta_stats['global_scale']:.4f}"
        )

    # Human meshes
    human_out_dir = out_dir / "human" / "meshes"
    human_orig_dir = out_dir / "human" / "meshes_original"
    ensure_dir(human_out_dir)
    ensure_dir(human_orig_dir)
    _save_human_mesh_sequence(final_human_verts_np, human_faces, human_out_dir)
    for i, ply_path in enumerate(human_ply_paths[:num_frames]):
        dst = human_orig_dir / f"frame_{i:04d}.ply"
        shutil.copy2(str(ply_path), str(dst))
    human_delta_stats = {
        "max_delta_rot_deg": float(
            human_delta_rotvecs.detach().norm(dim=-1).max().item()
            * 180.0
            / math.pi
        ),
        "mean_delta_rot_deg": float(
            human_delta_rotvecs.detach().norm(dim=-1).mean().item()
            * 180.0
            / math.pi
        ),
        "max_delta_trans_m": float(
            human_delta_trans.detach().norm(dim=-1).max().item()
        ),
        "mean_delta_trans_m": float(
            human_delta_trans.detach().norm(dim=-1).mean().item()
        ),
        "optimize_human": bool(args.optimize_human),
    }
    with (out_dir / "human" / "delta_stats.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(human_delta_stats, f, indent=2)

    # Debug outputs
    debug_dir = out_dir / "debug"
    ensure_dir(debug_dir)
    _save_csv(debug_dir / "iter_metrics.csv", iter_rows)
    _save_joint_loss_plots(debug_dir, iter_rows)

    # Overlay video
    frames_dir = _resolve_frames_dir(dirs)
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
        if frame_paths:
            print("  Rendering overlay video...")
            _render_joint_overlay(
                frame_paths,
                final_human_verts_np,
                human_faces,
                objects,
                obj_keys,
                final_T_mats,
                final_scales,
                k,
                out_dir,
                args.fps,
                args.save_overlay_pngs,
            )
            print(f"  → {out_dir / 'overlay.mp4'}")
    else:
        print("  [WARN] No frames directory found — skipping overlay.")

    # Run summary
    summary = {
        "video_name": args.video_name,
        "status": "completed",
        "script": "track_human_object_mesh.py",
        "num_frames": num_frames,
        "num_objects": len(obj_keys),
        "num_edges": len(resolved_edges),
        "best_total_loss": best_loss,
        "optimisation_time_s": total_time,
        "inputs": {
            "aligned_mesh_dir": str(dirs["aligned"]),
            "tracked_object_dir": str(dirs["tracked"]),
            "segment_object_dir": str(dirs["seg_obj"]),
            "pag_file": str(pag_path),
            "smpl_seg_json": str(smpl_seg_path),
            "intrinsics_source": str(intr_path),
        },
        "weights": {
            "lambda_prior": args.lambda_prior,
            "lambda_contact": args.lambda_contact,
            "lambda_dynamics": args.lambda_dynamics,
            "lambda_penetration": args.lambda_penetration,
            "lambda_smooth": args.lambda_smooth,
            "lambda_human_prior": args.lambda_human_prior,
            "lambda_human_smooth": args.lambda_human_smooth,
            "lambda_human_mask_2d": args.lambda_human_mask_2d,
            "lambda_object_mask_2d": args.lambda_object_mask_2d,
            "lambda_object_part_mask_2d": args.lambda_object_part_mask_2d,
            "lambda_object_scale": args.lambda_object_scale,
        },
        "optimisation": {
            "adam_iters": args.adam_iters,
            "adam_lr": args.adam_lr,
            "sdf_resolution": args.sdf_resolution,
            "optimize_human": bool(args.optimize_human),
            "optimize_object_scale": bool(args.optimize_object_scale),
            "max_log_scale_delta": args.max_log_scale_delta,
            "early_stop_start": args.early_stop_start,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_rel_improve": args.early_stop_rel_improve,
            "early_stop_triggered": early_stop_triggered,
        },
        "debug_columns": list(iter_rows[0].keys()) if iter_rows else [],
        "objects": {
            slug: {
                "name": objects[slug].name,
                "num_verts": int(objects[slug].template_verts.shape[0]),
                "num_faces": int(objects[slug].faces.shape[0]),
                "num_parts": len(objects[slug].part_vert_ids),
                "parts": list(objects[slug].part_vert_ids.keys()),
                "is_translational": objects[slug].state.is_translational,
                "is_rotational": objects[slug].state.is_rotational,
                "final_scale": final_scales[slug],
            }
            for slug in obj_keys
        },
        "human": {
            "num_verts": int(human_data.base_verts.shape[1]),
            "num_faces": int(human_faces.shape[0]),
            "optimize_human": bool(args.optimize_human),
            "delta_stats": human_delta_stats,
        },
        "edges": [
            {
                "node_a": pag.edges[i].node_a if i < len(pag.edges) else "?",
                "node_b": pag.edges[i].node_b if i < len(pag.edges) else "?",
                "is_continuous": e.is_continuous,
                "is_rel_static": e.is_rel_static,
            }
            for i, e in enumerate(resolved_edges)
        ],
        "conventions": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "T_4x4": (
                "rigid component only; true object transform also uses "
                "global_scale"
            ),
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Output: {out_dir}")
    print(f"  Summary:  {out_dir / 'run_summary.json'}")
    print(f"  Overlay:  {out_dir / 'overlay.mp4'}")
    for slug in obj_keys:
        print(
            f"  {slug}:  poses_refined.json, "
            "transform_refined.json, meshes/"
        )
    print("  human:   meshes/, meshes_original/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
