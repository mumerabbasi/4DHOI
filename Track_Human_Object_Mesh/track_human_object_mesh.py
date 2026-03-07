"""Joint human–object mesh refinement with PAG contact constraints.

Given independently tracked human motion (frozen SMPL meshes) and object
SE(3) trajectories, this script jointly refines per-frame object poses so
that Part-level Affordance Graph (PAG) contact/dynamics constraints are
satisfied, human–object interpenetration is minimised, the original tracked
motion is preserved where possible, and temporal smoothness is maintained.

Human motion is the **fixed anchor** – only object SE(3) deltas are
optimised on top of the already-tracked poses.
"""

from __future__ import annotations

import argparse
import json
import math
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
        default=256,
        help="Voxel resolution for SDF grids.",
    )
    # Optimisation
    p.add_argument("--adam_iters", type=int, default=1200)
    p.add_argument("--adam_lr", type=float, default=1e-3)
    p.add_argument("--lbfgs_iters", type=int, default=100)
    p.add_argument("--lbfgs_lr", type=float, default=5e-4)
    p.add_argument("--disable_lbfgs", action="store_true")
    # Loss weights
    p.add_argument(
        "--lambda_prior",
        type=float,
        default=25.0,
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
        default=20.0,
        help="Temporal smoothness weight.",
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
def _load_object_part_verts(
    seg_obj_dir: Path,
    obj_slug: str,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load triangle labels → per-part vertex indices."""
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

    result: dict[str, np.ndarray] = {}
    for part_name, label_id in label_map.items():
        tri_mask = tri_labels == label_id
        face_subset = mesh_faces[tri_mask]
        vert_ids = np.unique(face_subset.ravel())
        if vert_ids.size > 0:
            result[part_name] = vert_ids
    return result


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


LOSS_TERM_KEYS = ("prior", "contact", "dynamics", "penetration", "smooth")


def _get_scaled_loss_terms(
    result: LossResult,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    return {
        "prior": args.lambda_prior * result.prior,
        "contact": args.lambda_contact * result.contact,
        "dynamics": args.lambda_dynamics * result.dynamics,
        "penetration": args.lambda_penetration * result.penetration,
        "smooth": args.lambda_smooth * result.smooth,
    }


def _build_loss_row(
    iteration: int,
    stage: str,
    result: LossResult,
    args: argparse.Namespace,
) -> dict[str, Any]:
    scaled_terms = _get_scaled_loss_terms(result, args)
    row: dict[str, Any] = {
        "iter": iteration,
        "stage": stage,
        "total": float(result.total.item()),
    }

    for key in LOSS_TERM_KEYS:
        row[f"{key}_raw"] = float(getattr(result, key).item())
        row[f"{key}_scaled"] = float(scaled_terms[key].item())

    return row


def _format_loss_log(
    iteration: int,
    total_iterations: int,
    result: LossResult,
    args: argparse.Namespace,
    elapsed_s: float,
) -> list[str]:
    scaled_terms = _get_scaled_loss_terms(result, args)
    scaled_str = "  ".join(
        f"{key}={scaled_terms[key].item():.5f}" for key in LOSS_TERM_KEYS
    )
    raw_str = "  ".join(
        f"{key}={getattr(result, key).item():.5f}" for key in LOSS_TERM_KEYS
    )
    return [
        f"  [{iteration:4d}/{total_iterations}] "
        f"total={result.total.item():.5f}  ({elapsed_s:.0f}s)",
        f"      scaled: {scaled_str}",
        f"      raw:    {raw_str}",
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
    T = pts_src.shape[0]
    per_frame_dists = []
    for t in range(T):
        d_sq = knn_points(
            pts_src[t:t + 1], pts_dst[t:t + 1], K=1
        ).dists[0, :, 0].clamp(min=0.0)
        if reduction == "mean":
            frame_dist = d_sq.mean()
        elif reduction == "min":
            frame_dist = d_sq.min()
        else:
            raise ValueError(f"Unsupported contact reduction: {reduction}")
        per_frame_dists.append(frame_dist)

    per_frame = torch.stack(per_frame_dists)  # [T]

    if is_continuous:
        return per_frame.mean()
    else:
        return per_frame.min()


def _compute_dynamics_loss(
    pts_contact: torch.Tensor,   # [T, P, 3] — contact part in WORLD space
    obj_T_mats: list[torch.Tensor],  # T elements of [4, 4]
    is_rel_static: bool,
) -> torch.Tensor:
    """Contact dynamics loss — transform contact points into object canonical
    space and penalise motion (static) or acceleration (sliding).

    pts_contact: the body-part (or other-object-part) points in world space.
    obj_T_mats: object's per-frame SE(3) transforms (world ← canonical).
    """
    T = pts_contact.shape[0]
    if T < 2:
        return torch.tensor(0.0, device=pts_contact.device)

    # Transform each frame's contact points into object canonical space
    canonical_pts = []
    for t in range(T):
        T_inv = _inv_se3(obj_T_mats[t])  # canonical ← world
        pts_canon = _apply_T_batch(pts_contact[t], T_inv)
        canonical_pts.append(pts_canon)
    canonical = torch.stack(canonical_pts)  # [T, P, 3]

    if is_rel_static:
        # Penalise ANY motion in canonical space
        diff = canonical[1:] - canonical[:-1]  # [T-1, P, 3]
        return (diff ** 2).mean()
    else:
        # Penalise acceleration (allow smooth sliding)
        if T < 3:
            diff = canonical[1:] - canonical[:-1]
            return (diff ** 2).mean() * 0.1  # light smoothness
        mid = canonical[1:-1]
        avg = 0.5 * (canonical[:-2] + canonical[2:])
        accel = mid - avg
        return (accel ** 2).mean()


def _compute_penetration_loss(
    human_verts_t: torch.Tensor,   # [V_h, 3]
    obj_data: ObjectData,
    obj_T: torch.Tensor,           # [4, 4] current frame's object pose
) -> torch.Tensor:
    """SDF-based penetration loss for human vertices inside object."""
    if obj_data.sdf_grid is None:
        return torch.tensor(0.0, device=human_verts_t.device)

    # Transform human verts to object canonical space
    T_inv = _inv_se3(obj_T)
    pts_canon = _apply_T_batch(human_verts_t, T_inv)

    sdf_vals = _query_sdf(obj_data.sdf_grid, pts_canon)
    # sdf_vals: negative = inside object
    penetration = F.relu(-sdf_vals)  # positive where inside
    n_inside = (penetration > 0).sum().clamp(min=1)
    return penetration.sum() / n_inside.float()


def _compute_obj_obj_penetration_loss(
    obj_a: ObjectData,
    obj_a_T: torch.Tensor,  # [4,4]
    obj_b: ObjectData,
    obj_b_T: torch.Tensor,  # [4,4]
    max_pts: int = 4096,
) -> torch.Tensor:
    """Approximate object-object penetration using B's SDF at A's verts."""
    if obj_b.sdf_grid is None:
        return torch.tensor(0.0, device=obj_a.template_verts.device)

    # Transform A's verts to world, then to B's canonical space
    verts_a_world = _apply_T_batch(obj_a.template_verts, obj_a_T)
    # Subsample if too many
    n = verts_a_world.shape[0]
    if n > max_pts:
        idx = torch.linspace(
            0,
            n - 1,
            max_pts,
            dtype=torch.long,
            device=verts_a_world.device,
        )
        verts_a_world = verts_a_world[idx]

    T_b_inv = _inv_se3(obj_b_T)
    pts_in_b_canon = _apply_T_batch(verts_a_world, T_b_inv)

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


# ---------------------------------------------------------------------------
# Full loss aggregation
# ---------------------------------------------------------------------------
def _compute_all_losses(
    delta_rotvecs: dict[str, torch.Tensor],   # obj_slug → [T, 3]
    delta_trans: dict[str, torch.Tensor],      # obj_slug → [T, 3]
    objects: dict[str, ObjectData],
    human_verts: torch.Tensor,                 # [T, V_h, 3]
    body_seg: dict[str, np.ndarray],
    resolved_edges: list[ResolvedEdge],
    obj_keys: list[str],                       # ordered object slugs
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> LossResult:
    device = human_verts.device
    T = human_verts.shape[0]

    # 1) Build effective poses and posed vertices for each object
    eff_T: dict[str, list[torch.Tensor]] = {}       # slug → [T x [4,4]]
    eff_rotvecs: dict[str, torch.Tensor] = {}       # slug → [T, 3]
    eff_trans: dict[str, torch.Tensor] = {}         # slug → [T, 3]
    obj_verts: dict[str, torch.Tensor] = {}         # slug → [T, V, 3]

    for slug in obj_keys:
        od = objects[slug]
        T_list = []
        rvs = []
        trs = []
        for t in range(T):
            base = torch.from_numpy(od.tracked_poses[t]).float().to(device)
            delta = _compose_T(delta_rotvecs[slug][t], delta_trans[slug][t])
            T_eff = base @ delta
            T_list.append(T_eff)
            # Decompose for smoothness
            R_eff = T_eff[:3, :3]
            rv = matrix_to_axis_angle(R_eff.unsqueeze(0)).squeeze(0)
            rvs.append(rv)
            trs.append(T_eff[:3, 3])

        eff_T[slug] = T_list
        eff_rotvecs[slug] = torch.stack(rvs)
        eff_trans[slug] = torch.stack(trs)

        # Posed vertices
        vt = torch.stack([
            _apply_T_batch(od.template_verts, T_list[t]) for t in range(T)
        ])
        obj_verts[slug] = vt  # [T, V, 3]

    # 2) Prior loss: penalise delta from zero
    loss_prior = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        loss_prior = loss_prior + (delta_rotvecs[slug] ** 2).sum()
        loss_prior = loss_prior + (delta_trans[slug] ** 2).sum()
    n_params = sum(
        delta_rotvecs[s].numel() + delta_trans[s].numel()
        for s in obj_keys
    )
    loss_prior = loss_prior / max(n_params, 1)

    # 3) Contact consistency loss
    loss_contact = torch.tensor(0.0, device=device)
    n_edges_contact = 0
    for edge in resolved_edges:
        # Get point clouds for both sides [T, P, 3]
        def _get_pts(
            is_human: bool,
            obj_idx: int,
            vert_ids: np.ndarray,
        ) -> torch.Tensor:
            sub_ids = _subsample_indices(len(vert_ids), MAX_PART_POINTS)
            vids = vert_ids[sub_ids]
            if is_human:
                return human_verts[:, vids, :]  # [T, P, 3]
            else:
                slug = obj_keys[obj_idx]
                return obj_verts[slug][:, vids, :]

        pts_a = _get_pts(edge.a_is_human, edge.a_object_idx, edge.a_vert_ids)
        pts_b = _get_pts(edge.b_is_human, edge.b_object_idx, edge.b_vert_ids)
        pts_src = pts_a if edge.contact_source_is_a else pts_b
        pts_dst = pts_b if edge.contact_source_is_a else pts_a

        loss_contact = loss_contact + _compute_contact_loss(
            pts_src, pts_dst, edge.is_continuous, edge.contact_reduction
        )
        n_edges_contact += 1
    if n_edges_contact > 0:
        loss_contact = loss_contact / n_edges_contact

    # 4) Contact dynamics loss
    loss_dynamics = torch.tensor(0.0, device=device)
    n_edges_dyn = 0
    for edge in resolved_edges:
        # Determine the moving contact side and the reference object.
        if edge.canonical_obj_idx < 0:
            continue  # no object in this edge — skip dynamics

        ref_slug = obj_keys[edge.canonical_obj_idx]
        ref_T_list = eff_T[ref_slug]

        # The contact points are on the OTHER side
        if edge.canonical_obj_idx == edge.a_object_idx:
            # Reference is side A. Contact points are side B.
            sub_ids = _subsample_indices(len(edge.b_vert_ids), MAX_PART_POINTS)
            if edge.b_is_human:
                pts_contact = human_verts[:, edge.b_vert_ids[sub_ids], :]
            else:
                other_slug = obj_keys[edge.b_object_idx]
                pts_contact = obj_verts[
                    other_slug
                ][:, edge.b_vert_ids[sub_ids], :]
        else:
            # Reference is side B. Contact points are side A.
            sub_ids = _subsample_indices(len(edge.a_vert_ids), MAX_PART_POINTS)
            if edge.a_is_human:
                pts_contact = human_verts[:, edge.a_vert_ids[sub_ids], :]
            else:
                other_slug = obj_keys[edge.a_object_idx]
                pts_contact = obj_verts[
                    other_slug
                ][:, edge.a_vert_ids[sub_ids], :]

        loss_dynamics = loss_dynamics + _compute_dynamics_loss(
            pts_contact, ref_T_list, edge.is_rel_static
        )
        n_edges_dyn += 1
    if n_edges_dyn > 0:
        loss_dynamics = loss_dynamics / n_edges_dyn

    # 5) Penetration loss (annealed)
    pen_progress = min(iteration / max(total_iters * 0.5, 1.0), 1.0)
    pen_weight_schedule = pen_progress  # ramp from 0 → 1 over first half

    loss_pen = torch.tensor(0.0, device=device)
    n_pen = 0
    # Subsample human verts for efficiency
    h_sub_ids = _subsample_indices(human_verts.shape[1], 4096)
    for t in range(T):
        human_sub = human_verts[t, h_sub_ids, :]
        for slug in obj_keys:
            od = objects[slug]
            loss_pen = loss_pen + _compute_penetration_loss(
                human_sub, od, eff_T[slug][t]
            )
            n_pen += 1
    # Object-object penetration (check all pairs)
    for i in range(len(obj_keys)):
        for j in range(i + 1, len(obj_keys)):
            for t in range(T):
                loss_pen = loss_pen + _compute_obj_obj_penetration_loss(
                    objects[obj_keys[i]], eff_T[obj_keys[i]][t],
                    objects[obj_keys[j]], eff_T[obj_keys[j]][t],
                    max_pts=2048,
                )
                n_pen += 1
    if n_pen > 0:
        loss_pen = loss_pen / n_pen
    loss_pen = loss_pen * pen_weight_schedule

    # 6) Smoothness loss
    loss_smooth = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        od = objects[slug]
        loss_smooth = loss_smooth + _compute_smoothness_loss(
            eff_rotvecs[slug], eff_trans[slug],
            od.state.is_translational, od.state.is_rotational,
        )
    if len(obj_keys) > 0:
        loss_smooth = loss_smooth / len(obj_keys)

    # Weighted total
    total = (
        args.lambda_prior * loss_prior
        + args.lambda_contact * loss_contact
        + args.lambda_dynamics * loss_dynamics
        + args.lambda_penetration * loss_pen
        + args.lambda_smooth * loss_smooth
    )

    return LossResult(
        total=total,
        prior=loss_prior,
        contact=loss_contact,
        dynamics=loss_dynamics,
        penetration=loss_pen,
        smooth=loss_smooth,
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


def _save_mesh_sequence(
    verts_template: np.ndarray,
    faces: np.ndarray,
    T_mats: np.ndarray,
    meshes_dir: Path,
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
        verts_t = (verts_template @ R.T) + t[None, :]
        mesh = mesh_tmpl.copy()
        mesh.vertices = verts_t
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
                verts_t = (od.template_verts.cpu().numpy()
                           @ final_T_mats[slug][t, :3, :3].T
                           + final_T_mats[slug][t, :3, 3][None, :])
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

    # ── Human mesh sequence (frozen) ──
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
    human_verts = torch.from_numpy(human_verts_np).float().to(device)

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
            part_verts = _load_object_part_verts(
                dirs["seg_obj"], slug, verts, faces
            )
        except FileNotFoundError:
            print(
                f"  [WARN] {slug}: part segmentation not found, "
                "using whole mesh."
            )
            part_verts = {}

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
            part_vert_ids=part_verts,
            sdf_grid=sdf_grid,
            color_bgr=color,
        )
        obj_keys.append(slug)
        part_names = ", ".join(part_verts.keys())
        print(
            f"  Loaded {slug}: {verts.shape[0]} verts, "
            f"{faces.shape[0]} faces, {len(part_verts)} parts "
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
    params: list[torch.Tensor] = []

    for slug in obj_keys:
        dr = torch.zeros(num_frames, 3, device=device, requires_grad=True)
        dt = torch.zeros(num_frames, 3, device=device, requires_grad=True)
        delta_rotvecs[slug] = dr
        delta_trans[slug] = dt
        params.extend([dr, dt])

    # ── Print summary ──
    print("=" * 60)
    print("Joint Human–Object Mesh Refinement")
    print(f"  video:    {args.video_name}")
    print(f"  device:   {device}")
    print(f"  frames:   {num_frames}")
    print(f"  objects:  {', '.join(obj_keys)}")
    print(f"  edges:    {len(resolved_edges)}")
    print(
        f"  K: fx={k[0, 0]:.1f}  fy={k[1, 1]:.1f}  "
        f"cx={k[0, 2]:.1f}  cy={k[1, 2]:.1f}"
    )
    print(f"  λ_prior={args.lambda_prior}  λ_contact={args.lambda_contact}")
    print(
        f"  λ_dynamics={args.lambda_dynamics}  "
        f"λ_pen={args.lambda_penetration}"
    )
    print(f"  λ_smooth={args.lambda_smooth}")
    print("=" * 60)

    # ── Adam optimisation ──
    total_iters = args.adam_iters + (
        0 if args.disable_lbfgs else args.lbfgs_iters
    )
    optimizer = torch.optim.Adam(params, lr=args.adam_lr)
    iter_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, dict[str, torch.Tensor]] = {}

    print(f"\n[Adam] {args.adam_iters} iterations, lr={args.adam_lr}")
    t_start = time.time()

    for it in range(args.adam_iters):
        optimizer.zero_grad()
        result = _compute_all_losses(
            delta_rotvecs, delta_trans, objects, human_verts,
            body_seg, resolved_edges, obj_keys, args,
            iteration=it, total_iters=total_iters,
        )
        result.total.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        loss_val = result.total.item()
        # NaN recovery: restore best state and reduce lr
        if math.isnan(loss_val) or math.isinf(loss_val):
            if best_state:
                for slug in obj_keys:
                    delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
                    delta_trans[slug].data.copy_(best_state[slug]["dt"])
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.5
                print(
                    f"  [{it:4d}] NaN detected; restored best state and "
                    "halved lr."
                )
            continue
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {
                slug: {
                    "dr": delta_rotvecs[slug].detach().clone(),
                    "dt": delta_trans[slug].detach().clone(),
                }
                for slug in obj_keys
            }

        if it % args.log_interval == 0 or it == args.adam_iters - 1:
            row = _build_loss_row(it, "adam", result, args)
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

    # ── Optional L-BFGS refinement ──
    if not args.disable_lbfgs and args.lbfgs_iters > 0:
        print(f"\n[L-BFGS] {args.lbfgs_iters} iterations, lr={args.lbfgs_lr}")
        lbfgs = torch.optim.LBFGS(
            params, lr=args.lbfgs_lr,
            max_iter=1, line_search_fn="strong_wolfe"
        )
        lbfgs_result: list[LossResult | None] = [None]

        for it_l in range(args.lbfgs_iters):
            global_it = args.adam_iters + it_l

            def closure():
                lbfgs.zero_grad()
                res = _compute_all_losses(
                    delta_rotvecs, delta_trans, objects, human_verts,
                    body_seg, resolved_edges, obj_keys, args,
                    iteration=global_it, total_iters=total_iters,
                )
                res.total.backward()
                lbfgs_result[0] = res
                return res.total

            lbfgs.step(closure)

            if (
                lbfgs_result[0] is not None
                and (
                    it_l % args.log_interval == 0
                    or it_l == args.lbfgs_iters - 1
                )
            ):
                row = _build_loss_row(
                    global_it, "lbfgs", lbfgs_result[0], args
                )
                iter_rows.append(row)

            if (
                lbfgs_result[0] is not None
                and (
                    args.verbose
                    or it_l % (args.log_interval * 4) == 0
                    or it_l == args.lbfgs_iters - 1
                )
            ):
                elapsed = time.time() - t_start
                for line in _format_loss_log(
                    global_it,
                    total_iters,
                    lbfgs_result[0],
                    args,
                    elapsed,
                ):
                    print(line)

            current_loss = (
                lbfgs_result[0].total.item()
                if lbfgs_result[0] is not None
                else float("inf")
            )
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {
                    slug: {
                        "dr": delta_rotvecs[slug].detach().clone(),
                        "dt": delta_trans[slug].detach().clone(),
                    }
                    for slug in obj_keys
                }

        print(f"  L-BFGS done. Best total loss: {best_loss:.6f}")

    # ── Restore best parameters ──
    if best_state:
        for slug in obj_keys:
            delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
            delta_trans[slug].data.copy_(best_state[slug]["dt"])

    total_time = time.time() - t_start
    print(
        f"\nOptimisation complete in {total_time:.1f}s. "
        f"Best loss: {best_loss:.6f}"
    )

    # ── Extract final poses ──
    final_T_mats: dict[str, np.ndarray] = {}

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

    # ── Save outputs ──
    print("\nSaving outputs...")

    # Per-object outputs
    for slug in obj_keys:
        od = objects[slug]
        obj_dir = out_dir / slug
        ensure_dir(obj_dir)

        # Refined poses
        _save_pose_json(obj_dir / "poses_refined.json", final_T_mats[slug])

        # Also save the original tracked poses for comparison
        _save_pose_json(obj_dir / "poses_original.json", od.tracked_poses)

        # Per-frame meshes
        _save_mesh_sequence(
            od.template_verts.cpu().numpy(), od.faces,
            final_T_mats[slug], obj_dir / "meshes",
        )

        # Delta summary
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
        }
        with (obj_dir / "delta_stats.json").open("w", encoding="utf-8") as f:
            json.dump(delta_stats, f, indent=2)

        print(f"  {slug}: max Δrot={delta_stats['max_delta_rot_deg']:.2f}°, "
              f"max Δtrans={delta_stats['max_delta_trans_m']:.4f}m")

    # Human meshes (just symlink/copy aligned PLYs for convenience)
    human_out_dir = out_dir / "human" / "meshes"
    ensure_dir(human_out_dir)
    for i, ply_path in enumerate(human_ply_paths[:num_frames]):
        dst = human_out_dir / f"frame_{i:04d}.ply"
        if not dst.exists():
            import shutil
            shutil.copy2(str(ply_path), str(dst))

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
                frame_paths, human_verts_np, human_faces,
                objects, obj_keys, final_T_mats, k,
                out_dir, args.fps, args.save_overlay_pngs,
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
        },
        "optimisation": {
            "adam_iters": args.adam_iters,
            "adam_lr": args.adam_lr,
            "lbfgs_enabled": not args.disable_lbfgs,
            "lbfgs_iters": args.lbfgs_iters,
            "sdf_resolution": args.sdf_resolution,
        },
        "objects": {
            slug: {
                "name": objects[slug].name,
                "num_verts": int(objects[slug].template_verts.shape[0]),
                "num_faces": int(objects[slug].faces.shape[0]),
                "num_parts": len(objects[slug].part_vert_ids),
                "parts": list(objects[slug].part_vert_ids.keys()),
                "is_translational": objects[slug].state.is_translational,
                "is_rotational": objects[slug].state.is_rotational,
            }
            for slug in obj_keys
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
            "T_4x4": "standard column-vector [[R,t],[0,1]], p' = R @ p + t",
        },
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Output: {out_dir}")
    print(f"  Summary:  {out_dir / 'run_summary.json'}")
    print(f"  Overlay:  {out_dir / 'overlay.mp4'}")
    for slug in obj_keys:
        print(f"  {slug}:  poses_refined.json, meshes/")
    print("  human:   meshes/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
