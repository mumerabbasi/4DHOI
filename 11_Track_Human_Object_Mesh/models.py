"""Shared data models for joint human-object mesh refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    """Pre-computed SDF volume for an object in its canonical frame."""

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
    smplx_layer: Any
    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    betas: torch.Tensor
    alignment_matrix: torch.Tensor


@dataclass
class ObjectData:
    """All data for a single object."""

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
    human_pose: torch.Tensor
    human_pose_smooth: torch.Tensor
    object_scale: torch.Tensor
    object_intersect: torch.Tensor
    intersect: torch.Tensor
    nocontact: torch.Tensor
    contact_drift: torch.Tensor
    weights: dict[str, float]


@dataclass
class DiagnosticLossResult:
    sequence: LossResult
    per_frame_raw: dict[str, torch.Tensor]
    global_raw: dict[str, torch.Tensor]


FIXED_LOSS_WEIGHT_ATTRS = {
    "tracking": "tracking_weight",
}

SCHEDULED_LOSS_WEIGHT_ATTRS = {
    "object_cd2d": (
        "object_cd2d_weight_start",
        "object_cd2d_weight_end",
    ),
    "object_part_cd2d": (
        "object_part_cd2d_weight_start",
        "object_part_cd2d_weight_end",
    ),
    "object_smooth_trans": (
        "object_smooth_trans_weight_start",
        "object_smooth_trans_weight_end",
    ),
    "object_smooth_rot": (
        "object_smooth_rot_weight_start",
        "object_smooth_rot_weight_end",
    ),
    "human_pose": (
        "human_pose_weight_start",
        "human_pose_weight_end",
    ),
    "human_pose_smooth": (
        "human_pose_smooth_weight_start",
        "human_pose_smooth_weight_end",
    ),
    "object_scale": (
        "object_scale_weight_start",
        "object_scale_weight_end",
    ),
    "object_intersect": (
        "object_intersect_weight_start",
        "object_intersect_weight_end",
    ),
    "intersect": (
        "intersect_weight_start",
        "intersect_weight_end",
    ),
    "nocontact": (
        "nocontact_weight_start",
        "nocontact_weight_end",
    ),
    "contact_drift": (
        "contact_drift_weight_start",
        "contact_drift_weight_end",
    ),
}

LOSS_TERM_KEYS = tuple(
    list(FIXED_LOSS_WEIGHT_ATTRS.keys())
    + list(SCHEDULED_LOSS_WEIGHT_ATTRS.keys())
)
FRAME_DIAGNOSTIC_TERM_KEYS = tuple(
    key for key in LOSS_TERM_KEYS if key in FIXED_LOSS_WEIGHT_ATTRS
    or key not in ("human_pose", "human_pose_smooth", "object_scale")
)


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
class OptimizationResult:
    best_loss: float
    best_iter: int
    optimisation_time_s: float
    early_stop_triggered: bool
    iter_rows: list[dict[str, Any]]
    frame_rows: list[dict[str, Any]]
    final_loss_summary_row: dict[str, Any]
    final_diagnostic: DiagnosticLossResult
    final_T_mats: dict[str, np.ndarray]
    final_scales: dict[str, float]
    final_human_verts_np_by_slug: dict[str, np.ndarray]
    object_delta_stats: dict[str, dict[str, Any]]
    human_delta_stats: dict[str, dict[str, Any]]
    active_human_pose_chains: dict[str, list[str]]
