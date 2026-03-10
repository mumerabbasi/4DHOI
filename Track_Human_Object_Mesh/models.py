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
    """Pre-computed SDF volume for an object in canonical frame-0 pose."""

    sdf_volume: torch.Tensor
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor


@dataclass
class HumanData:
    base_verts: torch.Tensor
    faces: np.ndarray
    faces_torch: torch.Tensor
    part_points_base: dict[str, torch.Tensor]
    sampled_points_base: torch.Tensor
    centers: torch.Tensor
    mask_points_2d: PackedPointCloud2D | None


@dataclass
class ObjectData:
    """All data for a single object."""

    name: str
    slug: str
    state: PAGObjectState
    template_verts: torch.Tensor
    faces: np.ndarray
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
class ResolvedEdge:
    """A PAG edge resolved to actual vertex index sets."""

    a_is_human: bool
    a_object_idx: int
    a_vert_ids: np.ndarray
    a_part_name: str
    b_is_human: bool
    b_object_idx: int
    b_vert_ids: np.ndarray
    b_part_name: str
    is_continuous: bool
    is_rel_static: bool
    contact_reduction: str
    contact_source_is_a: bool
    canonical_obj_idx: int


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


@dataclass
class DiagnosticLossResult:
    sequence: LossResult
    per_frame_raw: dict[str, torch.Tensor]
    global_raw: dict[str, torch.Tensor]


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
FRAME_DIAGNOSTIC_TERM_KEYS = tuple(
    key for key in LOSS_TERM_KEYS if key != "object_scale_reg"
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
    human_verts_np: np.ndarray
    human_faces: np.ndarray
    human_data: HumanData
    objects: dict[str, ObjectData]
    obj_keys: list[str]
    resolved_edges: list[ResolvedEdge]


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
    final_human_verts_np: np.ndarray
    human_delta_stats: dict[str, Any]
    object_delta_stats: dict[str, dict[str, Any]]
