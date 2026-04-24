from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import smplx
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

import align_human_to_scene as base

try:
    from mesh_intersection.bvh_search_tree import BVH
    from mesh_intersection.loss import DistanceFieldPenetrationLoss
except ImportError:
    BVH = None
    DistanceFieldPenetrationLoss = None


LOSS_TERM_KEYS = (
    "mask",
    # "front",
    # "root_trans_gvhmr",
    # "root_orient_gvhmr",
    # "pose_gvhmr",
    # "betas_gvhmr",
    # "scale_prior",
    "intersect",
    # "floor_intersect",
    "nocontact",
    "floor_nocontact",
    # "angle",
    # "self_intersect",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass
class DynamicContactEdge:
    node_a: base.InteractionNode
    node_b: base.InteractionNode
    moving_node: base.InteractionNode
    fixed_node: base.InteractionNode
    moving_part_name: str
    moving_segment_name: str
    moving_vertex_ids: np.ndarray
    fixed_points: np.ndarray
    is_continuous: bool
    reduction: str


class SMPLXAnglePrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip_idxs_signs = torch.as_tensor(
            [
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, -1),
                (4, 0, -1),
                (5, 0, -1),
                (6, 0, -1),
                (7, 0, -1),
                (8, 0, -1),
                (9, 0, -1),
                (12, 0, -1),
                (13, 1, 1),
                (14, 1, -1),
                (16, 1, 1),
                (17, 1, -1),
                (18, 1, 1),
                (19, 1, -1),
            ]
        ).int()
        zero_idxs = torch.as_tensor(
            [
                (10, 0),
                (10, 1),
                (10, 2),
                (11, 0),
                (11, 1),
                (11, 2),
                (15, 0),
                (15, 1),
                (15, 2),
                (20, 1),
                (21, 1),
            ]
        ).int()
        self.register_buffer("clip_idxs_signs", clip_idxs_signs)
        self.register_buffer("zero_idxs", zero_idxs)

    def forward(self, pose: torch.Tensor, with_pelvis: bool = False) -> torch.Tensor:
        assert pose.ndim == 2
        cdata = self.clip_idxs_signs
        zdata = self.zero_idxs
        if not with_pelvis:
            assert pose.shape[1] == 21 * 3
            cdata = torch.clone(cdata)
            cdata[:, 0] -= 1
            zdata = torch.clone(zdata)
            zdata[:, 0] -= 1
        else:
            assert pose.shape[1] == 22 * 3

        cidxs = cdata[:, 0] * 3 + cdata[:, 1]
        csigns = cdata[:, 2]
        cres = F.relu(pose[:, cidxs] * torch.unsqueeze(csigns, 0))

        zidxs = zdata[:, 0] * 3 + zdata[:, 1]
        zres = torch.abs(pose[:, zidxs])
        return torch.mean(torch.cat((cres, zres), dim=1))


class FullBodySMPLXParams(nn.Module):
    def __init__(
        self,
        transl_init: torch.Tensor,
        global_orient_init: torch.Tensor,
        body_pose_init: torch.Tensor,
        betas_init: torch.Tensor,
        max_log_scale_delta: float,
    ) -> None:
        super().__init__()
        orient_matrix = axis_angle_to_matrix(global_orient_init.view(1, 3))[0]
        orient_6d = matrix_to_rotation_6d(orient_matrix.view(1, 3, 3))[0]
        self.transl = nn.Parameter(transl_init.clone())
        self.global_orient_6d = nn.Parameter(orient_6d.clone())
        self.body_pose = nn.Parameter(body_pose_init.clone())
        self.betas = nn.Parameter(betas_init.clone())
        self.log_scale = nn.Parameter(
            torch.zeros((), dtype=transl_init.dtype, device=transl_init.device)
        )
        self.max_log_scale_delta = float(max_log_scale_delta)

    def forward(self, smplx_layer: Any) -> dict[str, torch.Tensor]:
        orient_matrix = rotation_6d_to_matrix(self.global_orient_6d.view(1, 6))[0]
        global_orient = matrix_to_axis_angle(orient_matrix.view(1, 3, 3))[0]
        log_scale = torch.clamp(
            self.log_scale,
            min=-float(self.max_log_scale_delta),
            max=float(self.max_log_scale_delta),
        )
        scale = torch.exp(log_scale)
        smplx_out = smplx_layer(
            transl=self.transl.view(1, 3),
            global_orient=global_orient.view(1, 3),
            body_pose=self.body_pose.view(1, -1),
            betas=self.betas.view(1, -1),
        )
        verts_unscaled = smplx_out.vertices[0]
        joints_unscaled = smplx_out.joints[0]
        verts = self.transl[None] + scale * (verts_unscaled - self.transl[None])
        joints = self.transl[None] + scale * (joints_unscaled - self.transl[None])
        return {
            "verts": verts,
            "verts_unscaled": verts_unscaled,
            "joints": joints,
            "joints_unscaled": joints_unscaled,
            "transl": self.transl,
            "global_orient_matrix": orient_matrix,
            "global_orient": global_orient,
            "body_pose": self.body_pose,
            "betas": self.betas,
            "log_scale": log_scale,
            "scale": scale,
        }


class SelfIntersectionHelper:
    def __init__(
        self,
        init_vertices: torch.Tensor,
        sample_vertex_count: int,
        local_distance_thresh_m: float,
        collision_margin_m: float,
        seed: int,
    ) -> None:
        self.use_exact = BVH is not None and DistanceFieldPenetrationLoss is not None
        if self.use_exact:
            self.bvh = BVH(max_collisions=8)
            self.dfp_loss = DistanceFieldPenetrationLoss(
                sigma=0.001,
                point2plane=False,
                vectorized=True,
                penalize_outside=True,
            )
            self.sample_vertex_ids = None
            self.pair_mask = None
            self.collision_margin_m = float(collision_margin_m)
            return

        num_vertices = int(init_vertices.shape[0])
        sample_vertex_count = max(2, min(int(sample_vertex_count), num_vertices))
        rng = np.random.default_rng(int(seed))
        sample_vertex_ids_np = np.sort(
            rng.choice(num_vertices, size=sample_vertex_count, replace=False)
        ).astype(np.int64)
        sample_vertex_ids = torch.from_numpy(sample_vertex_ids_np).to(
            device=init_vertices.device
        )
        sampled_init = init_vertices[sample_vertex_ids]
        init_dists = torch.cdist(sampled_init.unsqueeze(0), sampled_init.unsqueeze(0))[0]
        pair_mask = torch.triu(
            init_dists > float(local_distance_thresh_m),
            diagonal=1,
        )
        self.sample_vertex_ids = sample_vertex_ids
        self.pair_mask = pair_mask
        self.collision_margin_m = float(collision_margin_m)

    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        if self.use_exact:
            triangles = vertices[faces]
            triangles = triangles.unsqueeze(0)
            with torch.no_grad():
                collision_idxs = self.bvh(triangles)
            if collision_idxs.ge(0).sum().item() == 0:
                return vertices.new_tensor(0.0)
            return torch.mean(self.dfp_loss(triangles, collision_idxs))

        assert self.sample_vertex_ids is not None
        assert self.pair_mask is not None
        sampled_vertices = vertices[self.sample_vertex_ids]
        pair_dists = torch.cdist(
            sampled_vertices.unsqueeze(0), sampled_vertices.unsqueeze(0)
        )[0]
        close_distances = pair_dists[self.pair_mask]
        if close_distances.numel() == 0:
            return vertices.new_tensor(0.0)
        penetration = F.relu(float(self.collision_margin_m) - close_distances)
        if penetration.numel() == 0:
            return vertices.new_tensor(0.0)
        return penetration.pow(2).mean()


def build_default_paths(video_name: str) -> dict[str, Path]:
    defaults = base.build_default_paths(video_name)
    defaults["output_root"] = SCRIPT_DIR / "output_full_body" / video_name
    defaults["smpl_folder"] = (
        SCRIPT_DIR.parent.parent / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    )
    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize first-frame full-body SMPL-X human grounding against a "
            "metric ScanNet scene using PAG contact semantics."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_01")
    parser.add_argument("--generated_root", type=str, default=None)
    parser.add_argument("--selection_json", type=str, default=None)
    parser.add_argument("--input_pag_json", type=str, default=None)
    parser.add_argument("--segment_root", type=str, default=None)
    parser.add_argument("--human_motion_root", type=str, default=None)
    parser.add_argument("--pag_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--scannet_root", type=str, default=None)
    parser.add_argument("--smpl_folder", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--smpl_param_key", type=str, default="smpl_params_incam")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--surface_samples", type=int, default=6000)
    parser.add_argument("--target_surface_samples", type=int, default=6000)
    parser.add_argument("--mask_points", type=int, default=2500)
    parser.add_argument("--mask_vertex_samples", type=int, default=3000)
    parser.add_argument("--adam_iters", type=int, default=2000)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--front_margin_m", type=float, default=0.03)
    parser.add_argument("--mask_weight", type=float, default=500.0)
    parser.add_argument("--front_weight", type=float, default=1500.0)
    parser.add_argument("--root_trans_gvhmr_weight", type=float, default=20.0)
    parser.add_argument("--root_orient_gvhmr_weight", type=float, default=20.0)
    parser.add_argument("--pose_gvhmr_weight", type=float, default=10.0)
    parser.add_argument("--betas_gvhmr_weight", type=float, default=10.0)
    parser.add_argument("--scale_prior_weight", type=float, default=25.0)
    parser.add_argument("--intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--intersect_weight_end", type=float, default=15.0)
    parser.add_argument("--floor_intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--floor_intersect_weight_end", type=float, default=20.0)
    parser.add_argument("--nocontact_weight_start", type=float, default=500.0)
    parser.add_argument("--nocontact_weight_end", type=float, default=500.0)
    parser.add_argument("--floor_nocontact_weight_start", type=float, default=200.0)
    parser.add_argument("--floor_nocontact_weight_end", type=float, default=200.0)
    parser.add_argument("--angle_weight_start", type=float, default=0.0)
    parser.add_argument("--angle_weight_end", type=float, default=1.0)
    parser.add_argument("--self_intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--self_intersect_weight_end", type=float, default=0)  # original was 1e-5
    parser.add_argument("--self_intersect_sample_vertices", type=int, default=768)
    parser.add_argument("--self_intersect_local_dist_thresh_m", type=float, default=0.04)
    parser.add_argument("--self_intersect_margin_m", type=float, default=0.01)
    parser.add_argument("--visible_tol_m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sdf_resolution", type=int, default=128)
    parser.add_argument("--max_log_scale_delta", type=float, default=0.22)
    return parser.parse_args()


def get_loss_weights(
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> dict[str, float]:
    return {
        "mask": float(args.mask_weight),
        "front": float(args.front_weight),
        "root_trans_gvhmr": float(args.root_trans_gvhmr_weight),
        "root_orient_gvhmr": float(args.root_orient_gvhmr_weight),
        "pose_gvhmr": float(args.pose_gvhmr_weight),
        "betas_gvhmr": float(args.betas_gvhmr_weight),
        "scale_prior": float(args.scale_prior_weight),
        "intersect": base.linear_weight(
            args.intersect_weight_start,
            args.intersect_weight_end,
            iteration,
            total_iters,
        ),
        "floor_intersect": base.linear_weight(
            args.floor_intersect_weight_start,
            args.floor_intersect_weight_end,
            iteration,
            total_iters,
        ),
        "nocontact": base.linear_weight(
            args.nocontact_weight_start,
            args.nocontact_weight_end,
            iteration,
            total_iters,
        ),
        "floor_nocontact": base.linear_weight(
            args.floor_nocontact_weight_start,
            args.floor_nocontact_weight_end,
            iteration,
            total_iters,
        ),
        "angle": base.linear_weight(
            args.angle_weight_start,
            args.angle_weight_end,
            iteration,
            total_iters,
        ),
        "self_intersect": base.linear_weight(
            args.self_intersect_weight_start,
            args.self_intersect_weight_end,
            iteration,
            total_iters,
        ),
    }


def build_loss_row(
    iteration: int,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "iter": int(iteration),
        "total": float(losses["total"].detach().cpu().item()),
    }
    for key in LOSS_TERM_KEYS:
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = float(weights[key])
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = float(weights[key]) * raw_value
    return row


def format_loss_log(
    iteration: int,
    total_iterations: int,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> list[str]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    weights_fmt = "  ".join(f"{key}={weights[key]:.4g}" for key in LOSS_TERM_KEYS)
    raw_fmt = "  ".join(
        f"{key}={float(losses[key].detach().cpu().item()):.5f}" for key in LOSS_TERM_KEYS
    )
    scaled_fmt = "  ".join(
        f"{key}={weights[key] * float(losses[key].detach().cpu().item()):.5f}"
        for key in LOSS_TERM_KEYS
    )
    return [
        f"  [iter {iteration:4d}/{total_iterations}] "
        f"total={float(losses['total'].detach().cpu().item()):.5f}",
        f"      weights: {weights_fmt}",
        f"      raw:     {raw_fmt}",
        f"      scaled:  {scaled_fmt}",
    ]


def build_final_loss_summary_row(
    final_iter: int,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "final_iter": int(final_iter),
        "final_total_loss": float(losses["total"].detach().cpu().item()),
        "total_scaled": float(losses["total"].detach().cpu().item()),
    }
    for key in LOSS_TERM_KEYS:
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = float(weights[key])
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = float(weights[key]) * raw_value
    return row


def resolve_track_pag_name(track_name: str) -> str:
    return base.normalize_label(track_name)


def build_dynamic_contact_edges(
    pag_payload: dict[str, Any],
    track_name: str,
    target_object_name: str,
    body_segments: dict[str, np.ndarray],
    contact_segments: dict[str, np.ndarray],
    target_points_visible: np.ndarray,
) -> list[DynamicContactEdge]:
    pag_edges = base.parse_pag_interaction_edges(pag_payload)
    target_object_norm = base.normalize_label(target_object_name)
    track_name_norm = resolve_track_pag_name(track_name)
    contact_edges: list[DynamicContactEdge] = []
    seen: set[tuple[str, str]] = set()

    for edge in pag_edges:
        nodes = [edge.node_a, edge.node_b]
        if sum(node.is_human for node in nodes) != 1:
            continue

        moving_node = nodes[0] if nodes[0].is_human else nodes[1]
        fixed_node = nodes[1] if nodes[0].is_human else nodes[0]
        if base.normalize_label(moving_node.entity_name) != track_name_norm:
            continue
        if base.normalize_label(fixed_node.entity_name) != target_object_norm:
            continue

        moving_part_name = base.normalize_label(moving_node.part_name)
        part_vert_ids = contact_segments.get(
            moving_part_name,
            body_segments.get(moving_part_name),
        )
        moving_segment_name = (
            f"{moving_part_name} contact"
            if moving_part_name in contact_segments
            else moving_part_name
        )
        if part_vert_ids is None:
            raise KeyError(
                "Unsupported human contact part "
                f"'{moving_node.part_name}' for {track_name}. Missing "
                "SMPL-X segmentation mapping."
            )

        dedup_key = (
            base.normalize_label(edge.node_a.raw_node),
            base.normalize_label(edge.node_b.raw_node),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        contact_edges.append(
            DynamicContactEdge(
                node_a=edge.node_a,
                node_b=edge.node_b,
                moving_node=moving_node,
                fixed_node=fixed_node,
                moving_part_name=moving_part_name,
                moving_segment_name=moving_segment_name,
                moving_vertex_ids=np.unique(np.asarray(part_vert_ids, dtype=np.int64)),
                fixed_points=target_points_visible.astype(np.float32),
                is_continuous=bool(edge.is_continuous),
                reduction=base.resolve_contact_reduction(moving_part_name),
            )
        )

    if not contact_edges:
        raise RuntimeError(
            f"No PAG human-object contact edges found for {track_name} and "
            f"target object '{target_object_name}'."
        )
    return contact_edges


def build_dynamic_floor_edges(
    track_name: str,
    body_segments: dict[str, np.ndarray],
    floor_points_visible: np.ndarray,
) -> list[DynamicContactEdge]:
    if floor_points_visible.shape[0] == 0:
        return []
    floor_node = base.InteractionNode(
        raw_node="floor",
        entity_name="floor",
        part_name="floor",
        is_human=False,
    )
    contact_edges: list[DynamicContactEdge] = []
    for part_name in ("left foot", "right foot"):
        part_vert_ids = body_segments.get(part_name)
        if part_vert_ids is None:
            continue
        moving_node = base.InteractionNode(
            raw_node=f"{track_name}.{part_name.replace(' ', '_')}",
            entity_name=track_name,
            part_name=part_name,
            is_human=True,
        )
        contact_edges.append(
            DynamicContactEdge(
                node_a=moving_node,
                node_b=floor_node,
                moving_node=moving_node,
                fixed_node=floor_node,
                moving_part_name=part_name,
                moving_segment_name=part_name,
                moving_vertex_ids=np.unique(np.asarray(part_vert_ids, dtype=np.int64)),
                fixed_points=floor_points_visible.astype(np.float32),
                is_continuous=False,
                reduction="min",
            )
        )
    return contact_edges


def load_first_frame_smplx_params(
    result_dir: Path,
    param_key: str,
) -> dict[str, torch.Tensor]:
    result_path = result_dir / "hmr4d_results.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Could not find hmr4d_results.pt in: {result_dir}")
    payload = torch.load(result_path, weights_only=True)
    if param_key not in payload:
        raise KeyError(
            f"Could not find '{param_key}' in {result_path}. "
            f"Available keys: {sorted(payload.keys())}"
        )
    params = payload[param_key]
    return {
        "transl": params["transl"][0].detach().clone().float(),
        "global_orient": params["global_orient"][0].detach().clone().float(),
        "body_pose": params["body_pose"][0].detach().clone().float(),
        "betas": params["betas"][0].detach().clone().float(),
    }


def build_smplx_layer(smpl_folder: Path, device: torch.device) -> Any:
    layer = smplx.create(
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
    layer = layer.to(device)
    layer.requires_grad_(False)
    return layer


def compute_root_orient_loss(
    current_orient_matrix: torch.Tensor,
    init_orient_matrix: torch.Tensor,
) -> torch.Tensor:
    relative = init_orient_matrix.transpose(0, 1) @ current_orient_matrix
    relative_aa = matrix_to_axis_angle(relative.view(1, 3, 3))[0]
    return torch.mean(relative_aa.pow(2))


def compute_contact_distance_loss(
    current_vertices: torch.Tensor,
    edges: list[DynamicContactEdge],
) -> torch.Tensor:
    if not edges:
        return current_vertices.new_tensor(0.0)
    values: list[torch.Tensor] = []
    for edge in edges:
        moving_points_seq = current_vertices[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points = torch.from_numpy(edge.fixed_points).to(
            device=current_vertices.device,
            dtype=current_vertices.dtype,
        )
        if fixed_points.shape[0] == 0:
            continue
        fixed_points_seq = fixed_points.unsqueeze(0)
        pdists = base.pcd_distance(
            moving_points_seq,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        if pdists is None:
            continue
        values.append(pdists.mean() if edge.is_continuous else pdists.min())
    if not values:
        return current_vertices.new_tensor(0.0)
    return torch.stack(values, dim=0).mean()


def compute_segment_penetration_loss(
    current_vertices: torch.Tensor,
    edges: list[DynamicContactEdge],
    sdf_grid: base.SDFGrid | None,
) -> torch.Tensor:
    if sdf_grid is None or not edges:
        return current_vertices.new_tensor(0.0)
    values: list[torch.Tensor] = []
    for edge in edges:
        moving_points = current_vertices[edge.moving_vertex_ids]
        intersect = base.compute_penetration_loss(sdf_grid, moving_points)
        if intersect.item() > 0.0:
            values.append(intersect)
    if not values:
        return current_vertices.new_tensor(0.0)
    return torch.stack(values, dim=0).mean()


def compute_loss_dict(
    params_module: FullBodySMPLXParams,
    smplx_layer: Any,
    faces_t: torch.Tensor,
    mask_vertex_ids_t: torch.Tensor,
    obs_mask_pixels_norm_t: torch.Tensor,
    human_mask_t: torch.Tensor,
    scene_depth_t: torch.Tensor,
    contact_edges: list[DynamicContactEdge],
    floor_edges: list[DynamicContactEdge],
    target_sdf_grid: base.SDFGrid | None,
    floor_sdf_grid: base.SDFGrid | None,
    intrinsics_t: torch.Tensor,
    width: int,
    height: int,
    init_params: dict[str, torch.Tensor],
    angle_prior: SMPLXAnglePrior,
    self_intersection_helper: SelfIntersectionHelper | None,
    front_margin_m: float,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    current = params_module(smplx_layer)
    verts_camera = current["verts"]
    sampled_vertices = verts_camera[mask_vertex_ids_t]
    uv_pixels = base.project_points_torch(sampled_vertices, intrinsics_t)
    model_pixels_norm = torch.stack(
        [
            uv_pixels[:, 0] / max(float(width - 1), 1.0),
            uv_pixels[:, 1] / max(float(height - 1), 1.0),
        ],
        dim=1,
    )

    zero = torch.zeros((), device=verts_camera.device, dtype=verts_camera.dtype)

    if obs_mask_pixels_norm_t.shape[0] > 0 and model_pixels_norm.shape[0] > 0:
        d2_obs_to_model = base.min_distances_chunked(
            obs_mask_pixels_norm_t, model_pixels_norm
        )
        d2_model_to_obs = base.min_distances_chunked(
            model_pixels_norm, obs_mask_pixels_norm_t
        )
        mask_loss = 0.5 * (d2_obs_to_model.mean() + d2_model_to_obs.mean())
    else:
        mask_loss = zero

    depth_at_uv = base.sample_image_bilinear(
        scene_depth_t, uv_pixels, width=width, height=height
    )
    mask_at_uv = base.sample_image_bilinear(
        human_mask_t, uv_pixels, width=width, height=height
    )
    in_frame = (
        (uv_pixels[:, 0] >= 0.0)
        & (uv_pixels[:, 0] <= float(width - 1))
        & (uv_pixels[:, 1] >= 0.0)
        & (uv_pixels[:, 1] <= float(height - 1))
        & (sampled_vertices[:, 2] > 1e-6)
        & (depth_at_uv > 0.0)
        & (mask_at_uv > 0.1)
    )
    if torch.any(in_frame):
        penetration = sampled_vertices[in_frame, 2] - (
            depth_at_uv[in_frame] - float(front_margin_m)
        )
        front_loss = torch.relu(penetration).pow(2).mean()
    else:
        front_loss = zero

    root_trans_gvhmr = torch.mean((current["transl"] - init_params["transl"]) ** 2)
    root_orient_gvhmr = compute_root_orient_loss(
        current["global_orient_matrix"],
        init_params["global_orient_matrix"],
    )
    pose_gvhmr = torch.mean((current["body_pose"] - init_params["body_pose"]) ** 2)
    betas_gvhmr = torch.mean((current["betas"] - init_params["betas"]) ** 2)
    scale_prior = current["log_scale"].pow(2)

    intersect = compute_segment_penetration_loss(
        current_vertices=verts_camera,
        edges=contact_edges,
        sdf_grid=target_sdf_grid,
    )
    floor_intersect = compute_segment_penetration_loss(
        current_vertices=verts_camera,
        edges=floor_edges,
        sdf_grid=floor_sdf_grid,
    )
    nocontact = compute_contact_distance_loss(
        current_vertices=verts_camera,
        edges=contact_edges,
    )
    floor_nocontact = compute_contact_distance_loss(
        current_vertices=verts_camera,
        edges=floor_edges,
    )

    angle = angle_prior(current["body_pose"].view(1, -1), with_pelvis=False)
    if self_intersection_helper is not None and float(weights["self_intersect"]) > 0.0:
        self_intersect = self_intersection_helper(verts_camera, faces_t)
    else:
        self_intersect = zero

    total = (
        mask_loss * float(weights["mask"])
        + front_loss * float(weights["front"])
        + root_trans_gvhmr * float(weights["root_trans_gvhmr"])
        + root_orient_gvhmr * float(weights["root_orient_gvhmr"])
        + pose_gvhmr * float(weights["pose_gvhmr"])
        + betas_gvhmr * float(weights["betas_gvhmr"])
        + scale_prior * float(weights["scale_prior"])
        + intersect * float(weights["intersect"])
        + floor_intersect * float(weights["floor_intersect"])
        + nocontact * float(weights["nocontact"])
        + floor_nocontact * float(weights["floor_nocontact"])
        + angle * float(weights["angle"])
        + self_intersect * float(weights["self_intersect"])
    )
    return {
        "total": total,
        "mask": mask_loss,
        "front": front_loss,
        "root_trans_gvhmr": root_trans_gvhmr,
        "root_orient_gvhmr": root_orient_gvhmr,
        "pose_gvhmr": pose_gvhmr,
        "betas_gvhmr": betas_gvhmr,
        "scale_prior": scale_prior,
        "intersect": intersect,
        "floor_intersect": floor_intersect,
        "nocontact": nocontact,
        "floor_nocontact": floor_nocontact,
        "angle": angle,
        "self_intersect": self_intersect,
        "weights": weights,
        "current": current,
    }


def compute_contact_metrics(
    current_vertices: np.ndarray,
    edges: list[DynamicContactEdge],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    device = torch.device("cpu")
    current_vertices_t = torch.from_numpy(current_vertices.astype(np.float32)).to(device)
    for edge in edges:
        fixed_points_t = torch.from_numpy(edge.fixed_points.astype(np.float32)).to(device)
        if fixed_points_t.shape[0] == 0:
            continue
        moving_points_t = current_vertices_t[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points_seq = fixed_points_t.unsqueeze(0)
        pdists = base.pcd_distance(
            moving_points_t,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        if pdists is None:
            continue
        nocontact_raw = (
            pdists.mean() if edge.is_continuous else pdists.min()
        ).detach().cpu().item()
        metrics.append(
            {
                "node_a": edge.node_a.raw_node,
                "node_b": edge.node_b.raw_node,
                "moving_entity_name": edge.moving_node.entity_name,
                "moving_part_name": edge.moving_node.part_name,
                "moving_segment_name": edge.moving_segment_name,
                "fixed_entity_name": edge.fixed_node.entity_name,
                "fixed_part_name": edge.fixed_node.part_name,
                "reduction": edge.reduction,
                "is_continuous": bool(edge.is_continuous),
                "nocontact_raw": float(nocontact_raw),
                "nocontact_distance_m": float(math.sqrt(max(nocontact_raw, 0.0))),
            }
        )
    return metrics


def compute_visible_behind_fraction(
    verts_camera: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    scene_depth: np.ndarray,
    human_mask: np.ndarray,
    mask_vertex_ids: np.ndarray,
    front_margin_m: float,
) -> dict[str, float]:
    sampled = verts_camera[mask_vertex_ids]
    uv = base.project_points_np(sampled, intrinsics)
    z = sampled[:, 2]
    ui = np.round(uv[:, 0]).astype(np.int64)
    vi = np.round(uv[:, 1]).astype(np.int64)
    valid = (
        (ui >= 0)
        & (ui < width)
        & (vi >= 0)
        & (vi < height)
        & (z > 1e-6)
        & (scene_depth[vi, ui] > 0.0)
        & human_mask[vi, ui]
    )
    behind = np.zeros(sampled.shape[0], dtype=bool)
    if np.any(valid):
        idx = np.nonzero(valid)[0]
        behind[idx] = z[idx] > (scene_depth[vi[idx], ui[idx]] - float(front_margin_m))
    num_valid = int(np.count_nonzero(valid))
    num_behind = int(np.count_nonzero(behind))
    return {
        "num_visible_points": num_valid,
        "num_behind_points": num_behind,
        "behind_fraction": float(num_behind / num_valid) if num_valid > 0 else 0.0,
    }


def optimize_track(
    smplx_layer: Any,
    faces_t: torch.Tensor,
    init_params_np: dict[str, np.ndarray],
    obs_mask_pixels: np.ndarray,
    human_mask: np.ndarray,
    scene_depth: np.ndarray,
    contact_edges: list[DynamicContactEdge],
    floor_edges: list[DynamicContactEdge],
    target_sdf_grid: base.SDFGrid | None,
    floor_sdf_grid: base.SDFGrid | None,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    mask_vertex_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    init_params_t = {
        "transl": torch.from_numpy(init_params_np["transl"]).to(device=device, dtype=torch.float32),
        "global_orient": torch.from_numpy(init_params_np["global_orient"]).to(device=device, dtype=torch.float32),
        "body_pose": torch.from_numpy(init_params_np["body_pose"]).to(device=device, dtype=torch.float32),
        "betas": torch.from_numpy(init_params_np["betas"]).to(device=device, dtype=torch.float32),
    }
    init_params_t["global_orient_matrix"] = axis_angle_to_matrix(
        init_params_t["global_orient"].view(1, 3)
    )[0]

    params_module = FullBodySMPLXParams(
        transl_init=init_params_t["transl"],
        global_orient_init=init_params_t["global_orient"],
        body_pose_init=init_params_t["body_pose"],
        betas_init=init_params_t["betas"],
        max_log_scale_delta=float(args.max_log_scale_delta),
    ).to(device)
    angle_prior = SMPLXAnglePrior().to(device)
    with torch.no_grad():
        init_out = smplx_layer(
            transl=init_params_t["transl"].view(1, 3),
            global_orient=init_params_t["global_orient"].view(1, 3),
            body_pose=init_params_t["body_pose"].view(1, -1),
            betas=init_params_t["betas"].view(1, -1),
        )
    self_intersection_helper = SelfIntersectionHelper(
        init_vertices=init_out.vertices[0].detach(),
        sample_vertex_count=int(args.self_intersect_sample_vertices),
        local_distance_thresh_m=float(args.self_intersect_local_dist_thresh_m),
        collision_margin_m=float(args.self_intersect_margin_m),
        seed=int(args.seed),
    )

    obs_mask_pixels_norm_t = torch.from_numpy(obs_mask_pixels.astype(np.float32)).to(device)
    if obs_mask_pixels_norm_t.numel() > 0:
        obs_mask_pixels_norm_t[:, 0] /= max(float(width - 1), 1.0)
        obs_mask_pixels_norm_t[:, 1] /= max(float(height - 1), 1.0)
    human_mask_t = torch.from_numpy(human_mask.astype(np.float32)).to(device)
    scene_depth_t = torch.from_numpy(scene_depth.astype(np.float32)).to(device)
    intrinsics_t = torch.from_numpy(intrinsics.astype(np.float32)).to(device)
    mask_vertex_ids_t = torch.from_numpy(mask_vertex_ids.astype(np.int64)).to(device)

    optimizer = torch.optim.Adam(params_module.parameters(), lr=float(args.adam_lr))
    iter_rows: list[dict[str, Any]] = []

    if int(args.adam_iters) <= 0:
        raise RuntimeError("adam_iters must be > 0.")

    for iter_idx in range(1, int(args.adam_iters) + 1):
        weights = get_loss_weights(args, iter_idx - 1, int(args.adam_iters) - 1)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_loss_dict(
            params_module=params_module,
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            mask_vertex_ids_t=mask_vertex_ids_t,
            obs_mask_pixels_norm_t=obs_mask_pixels_norm_t,
            human_mask_t=human_mask_t,
            scene_depth_t=scene_depth_t,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            target_sdf_grid=target_sdf_grid,
            floor_sdf_grid=floor_sdf_grid,
            intrinsics_t=intrinsics_t,
            width=width,
            height=height,
            init_params=init_params_t,
            angle_prior=angle_prior,
            self_intersection_helper=self_intersection_helper,
            front_margin_m=float(args.front_margin_m),
            weights=weights,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params_module.parameters(), max_norm=5.0)
        optimizer.step()
        iter_rows.append(build_loss_row(iter_idx, losses))
        if (
            iter_idx == 1
            or iter_idx % max(int(args.log_every), 1) == 0
            or iter_idx == int(args.adam_iters)
        ):
            for line in format_loss_log(iter_idx, int(args.adam_iters), losses):
                print(line)

    final_weights = get_loss_weights(
        args,
        int(args.adam_iters) - 1,
        int(args.adam_iters) - 1,
    )
    with torch.no_grad():
        final_losses = compute_loss_dict(
            params_module=params_module,
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            mask_vertex_ids_t=mask_vertex_ids_t,
            obs_mask_pixels_norm_t=obs_mask_pixels_norm_t,
            human_mask_t=human_mask_t,
            scene_depth_t=scene_depth_t,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            target_sdf_grid=target_sdf_grid,
            floor_sdf_grid=floor_sdf_grid,
            intrinsics_t=intrinsics_t,
            width=width,
            height=height,
            init_params=init_params_t,
            angle_prior=angle_prior,
            self_intersection_helper=self_intersection_helper,
            front_margin_m=float(args.front_margin_m),
            weights=final_weights,
        )
        current = final_losses["current"]
    return {
        "iter_rows": iter_rows,
        "final_iter": int(args.adam_iters),
        "final_total_loss": float(final_losses["total"].detach().cpu().item()),
        "final_losses": final_losses,
        "verts_camera": current["verts"].detach().cpu().numpy().astype(np.float32),
        "joints_camera": current["joints"].detach().cpu().numpy().astype(np.float32),
        "transl": current["transl"].detach().cpu().numpy().astype(np.float32),
        "global_orient": current["global_orient"].detach().cpu().numpy().astype(np.float32),
        "body_pose": current["body_pose"].detach().cpu().numpy().astype(np.float32),
        "betas": current["betas"].detach().cpu().numpy().astype(np.float32),
        "scale": float(current["scale"].detach().cpu().item()),
        "log_scale": float(current["log_scale"].detach().cpu().item()),
    }


def main() -> None:
    args = parse_args()
    defaults = build_default_paths(args.video_name)
    generated_root = base.resolve_path(args.generated_root, defaults["generated_root"])
    selection_json_path = base.resolve_path(args.selection_json, defaults["selection_json"])
    input_pag_json_path = base.resolve_path(args.input_pag_json, defaults["input_pag_json"])
    segment_root = base.resolve_path(args.segment_root, defaults["segment_root"])
    human_motion_root = base.resolve_path(args.human_motion_root, defaults["human_motion_root"])
    pag_json_path = (
        Path(args.pag_json).resolve()
        if args.pag_json is not None
        else base.find_pag_json(defaults["pag_root"])
    )
    smpl_seg_json_path = base.resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    smpl_folder = base.resolve_path(args.smpl_folder, defaults["smpl_folder"])
    output_root = base.ensure_dir(base.resolve_path(args.output_root, defaults["output_root"]))
    scene_root = base.ensure_dir(output_root / "scene")
    scene_depth_dir = base.ensure_dir(scene_root / "depth")
    scene_target_dir = base.ensure_dir(scene_root / "target")
    debug_root = base.ensure_dir(output_root / "debug")
    summary_json_path = output_root / "alignment_summary.json"
    scannet_root = base.resolve_scannet_root(SCRIPT_DIR, args.scannet_root)
    device = base.parse_device(args.device)

    (
        camera_payload,
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    ) = base.load_camera_payload(generated_root / "resized_camera.json")
    first_frame_path = generated_root / "first_frames_resized" / "frame_00.png"
    if not first_frame_path.exists():
        raise FileNotFoundError(f"Generated first frame not found: {first_frame_path}")
    camera_ctx = base.build_identity_camera(
        intrinsics=intrinsics,
        width=width,
        height=height,
        device=device,
    )

    input_payload = base.load_json(input_pag_json_path)
    selection_payload = base.load_json(selection_json_path)
    pag_payload = base.load_json(pag_json_path)
    body_segments, contact_segments = base.load_smpl_segments(smpl_seg_json_path)
    scene_id = input_payload["scene_context"]["scene_id"]
    scene_paths = base.resolve_scene_paths(scannet_root, scene_id)

    print(f"Loading ScanNet scene mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces = base.load_mesh(scene_paths["mesh_path"])
    scene_verts_camera = base.transform_world_to_camera(
        scene_verts_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    scene_faces_in_view = base.filter_faces_to_camera_view(
        verts_camera=scene_verts_camera,
        faces=scene_faces,
        intrinsics=intrinsics,
        width=width,
        height=height,
        max_depth_m=20.0,
        border_px=96.0,
    )
    scene_verts_camera_render, scene_faces_render = base.compact_mesh(
        scene_verts_camera, scene_faces_in_view
    )
    if scene_faces_render.shape[0] == 0:
        raise RuntimeError("No scene faces remained after view-frustum filtering.")

    scene_depth, _, _ = base.rasterize_depth_and_mask(
        scene_verts_camera_render,
        scene_faces_render,
        camera_ctx=camera_ctx,
        device=device,
    )
    np.save(scene_depth_dir / "scene_depth.npy", scene_depth.astype(np.float32))
    base.save_depth_visualization(scene_depth_dir / "scene_depth_vis.png", scene_depth)

    segments_payload = base.load_json(scene_paths["segments_path"])
    anno_payload = base.load_json(scene_paths["segments_anno_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    candidate_faces, face_instance_ids, instance_meta = base.build_candidate_instances(
        mesh_faces=scene_faces,
        seg_indices=seg_indices,
        seg_groups=anno_payload["segGroups"],
    )

    target_instance_id = int(selection_payload["target_selection"]["instance_id"])
    target_meta = instance_meta.get(target_instance_id)
    if target_meta is None:
        raise KeyError(
            "Target instance_id "
            f"{target_instance_id} is not present in the scene annotations."
        )
    target_faces = candidate_faces[face_instance_ids == target_instance_id]
    if target_faces.shape[0] == 0:
        raise RuntimeError(f"No faces found for target instance {target_instance_id}.")
    target_verts_camera, target_faces_compact = base.compact_mesh(
        scene_verts_camera, target_faces
    )
    target_sdf_grid = base.build_sdf_grid(
        target_verts_camera,
        target_faces_compact,
        resolution=int(args.sdf_resolution),
        device=device,
    )
    target_surface_samples = base.sample_mesh_surface_points(
        verts=target_verts_camera,
        faces=target_faces_compact,
        num_samples=int(args.target_surface_samples),
        seed=int(args.seed),
    )
    target_points_visible, target_visible_stats = base.build_visible_subset(
        sampled_points=target_surface_samples,
        mesh_verts=target_verts_camera,
        mesh_faces=target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
        visible_tol_m=float(args.visible_tol_m),
    )
    np.save(scene_target_dir / "target_surface_visible_samples.npy", target_points_visible)
    base.write_point_cloud_ply(
        scene_target_dir / "target_surface_visible_samples.ply",
        target_points_visible.astype(np.float32),
    )

    floor_faces = base.build_faces_for_labels(
        mesh_faces=scene_faces,
        seg_indices=seg_indices,
        seg_groups=anno_payload["segGroups"],
        labels={"floor"},
    )
    floor_sdf_grid: base.SDFGrid | None = None
    floor_points_visible = np.zeros((0, 3), dtype=np.float32)
    if floor_faces.shape[0] > 0:
        floor_verts_camera, floor_faces_compact = base.compact_mesh(scene_verts_camera, floor_faces)
        if floor_faces_compact.shape[0] > 0:
            floor_sdf_grid = base.build_sdf_grid(
                floor_verts_camera,
                floor_faces_compact,
                resolution=int(args.sdf_resolution),
                device=device,
            )
            floor_surface_samples = base.sample_mesh_surface_points(
                verts=floor_verts_camera,
                faces=floor_faces_compact,
                num_samples=int(args.target_surface_samples),
                seed=int(args.seed),
            )
            floor_points_visible, _ = base.build_visible_subset(
                sampled_points=floor_surface_samples,
                mesh_verts=floor_verts_camera,
                mesh_faces=floor_faces_compact,
                camera_ctx=camera_ctx,
                device=device,
                visible_tol_m=float(args.visible_tol_m),
            )

    target_depth, target_mask_rendered, _ = base.rasterize_depth_and_mask(
        target_verts_camera,
        target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
    )
    stored_target_mask = base.build_observed_projected_target_mask(
        selection_json_path=selection_json_path,
        width=width,
        height=height,
    )
    target_mask_overlap = base.compute_binary_overlap(
        stored_target_mask, target_mask_rendered
    )
    target_overlay = base.render_target_overlay(
        background_bgr=base.read_bgr(first_frame_path),
        mask=target_mask_rendered,
        lines=[
            f"Target instance {target_instance_id}: {target_meta['label']}",
            f"Rendered vs stored IoU: {target_mask_overlap['iou']:.3f}",
            f"Rendered vs stored Dice: {target_mask_overlap['dice']:.3f}",
        ],
    )
    cv2.imwrite(str(scene_target_dir / "target_projection_overlay.png"), target_overlay)

    tracks = base.discover_human_tracks(
        segment_root=segment_root,
        human_motion_root=human_motion_root,
    )
    smplx_layer = build_smplx_layer(smpl_folder, device)
    faces_np = np.asarray(smplx_layer.faces, dtype=np.int64)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device)

    summary_tracks: list[dict[str, Any]] = []

    for track in tracks:
        print(f"\nProcessing human track: {track.name}")
        track_output_root = base.ensure_dir(output_root / track.name)
        meshes_root = base.ensure_dir(track_output_root / "meshes")
        debug_track_root = base.ensure_dir(debug_root / track.name)
        overlay_dir = base.ensure_dir(debug_track_root / "overlays")
        csv_dir = base.ensure_dir(debug_track_root / "csv")
        plot_dir = base.ensure_dir(debug_track_root / "plots" / "iter")
        params_dir = base.ensure_dir(debug_track_root / "params")

        human_mask = base.load_mask(track.mask_dir / "frame_0000.png")
        if human_mask.shape != (height, width):
            raise ValueError(
                f"Human mask shape mismatch for {track.name}: "
                f"got {human_mask.shape[::-1]}, expected {(width, height)}"
            )
        obs_mask_pixels = base.sample_mask_pixels(
            mask=human_mask,
            max_points=int(args.mask_points),
            seed=int(args.seed),
        )
        init_params_torch = load_first_frame_smplx_params(track.result_dir, args.smpl_param_key)
        with torch.no_grad():
            init_out = smplx_layer(
                transl=init_params_torch["transl"].view(1, 3).to(device),
                global_orient=init_params_torch["global_orient"].view(1, 3).to(device),
                body_pose=init_params_torch["body_pose"].view(1, -1).to(device),
                betas=init_params_torch["betas"].view(1, -1).to(device),
            )
            init_verts_camera = init_out.vertices[0].detach().cpu().numpy().astype(np.float32)

        rng = np.random.default_rng(int(args.seed))
        num_vertices = init_verts_camera.shape[0]
        mask_vertex_count = min(int(args.mask_vertex_samples), num_vertices)
        mask_vertex_ids = rng.choice(num_vertices, size=mask_vertex_count, replace=False)
        contact_edges = build_dynamic_contact_edges(
            pag_payload=pag_payload,
            track_name=track.name,
            target_object_name=selection_payload["target_selection"]["object"],
            body_segments=body_segments,
            contact_segments=contact_segments,
            target_points_visible=target_points_visible,
        )
        floor_edges = build_dynamic_floor_edges(
            track_name=track.name,
            body_segments=body_segments,
            floor_points_visible=floor_points_visible,
        )
        init_params_np = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in init_params_torch.items()
        }

        init_depth, init_mask_rendered, _ = base.rasterize_depth_and_mask(
            init_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        init_mask_overlap = base.compute_binary_overlap(human_mask, init_mask_rendered)
        init_overlay = base.render_mask_overlay(
            background_bgr=base.read_bgr(first_frame_path),
            observed_mask=human_mask,
            rendered_mask=init_mask_rendered,
            title_lines=[
                f"{track.name}: init full-body",
                f"Human IoU: {init_mask_overlap['iou']:.3f}",
                f"Human Dice: {init_mask_overlap['dice']:.3f}",
            ],
        )
        cv2.imwrite(str(overlay_dir / "frame_0000_init_mask_overlay.png"), init_overlay)
        base.save_depth_visualization(overlay_dir / "frame_0000_init_depth_vis.png", init_depth)
        init_contact_metrics = compute_contact_metrics(init_verts_camera, contact_edges + floor_edges)
        init_behind = compute_visible_behind_fraction(
            verts_camera=init_verts_camera,
            intrinsics=intrinsics,
            width=width,
            height=height,
            scene_depth=scene_depth,
            human_mask=human_mask,
            mask_vertex_ids=mask_vertex_ids,
            front_margin_m=float(args.front_margin_m),
        )

        optimization = optimize_track(
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            init_params_np=init_params_np,
            obs_mask_pixels=obs_mask_pixels,
            human_mask=human_mask,
            scene_depth=scene_depth,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            target_sdf_grid=target_sdf_grid,
            floor_sdf_grid=floor_sdf_grid,
            intrinsics=intrinsics,
            width=width,
            height=height,
            mask_vertex_ids=mask_vertex_ids,
            args=args,
            device=device,
        )

        final_verts_camera = optimization["verts_camera"]
        final_depth, final_mask_rendered, _ = base.rasterize_depth_and_mask(
            final_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        final_mask_overlap = base.compute_binary_overlap(human_mask, final_mask_rendered)
        final_overlay = base.render_mask_overlay(
            background_bgr=base.read_bgr(first_frame_path),
            observed_mask=human_mask,
            rendered_mask=final_mask_rendered,
            title_lines=[
                f"{track.name}: final full-body",
                f"Human IoU: {final_mask_overlap['iou']:.3f}",
                f"Human Dice: {final_mask_overlap['dice']:.3f}",
            ],
        )
        cv2.imwrite(str(overlay_dir / "frame_0000_final_mask_overlay.png"), final_overlay)
        base.save_depth_visualization(overlay_dir / "frame_0000_final_depth_vis.png", final_depth)

        final_verts_world = base.transform_camera_to_world(
            final_verts_camera,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        base.write_ascii_ply(meshes_root / "frame_0000_camera.ply", final_verts_camera, faces_np)
        base.write_ascii_ply(meshes_root / "frame_0000_world.ply", final_verts_world, faces_np)

        optimized_params_payload = {
            "transl": optimization["transl"].tolist(),
            "global_orient": optimization["global_orient"].tolist(),
            "body_pose": optimization["body_pose"].tolist(),
            "betas": optimization["betas"].tolist(),
            "scale": float(optimization["scale"]),
            "log_scale": float(optimization["log_scale"]),
        }
        torch.save(optimized_params_payload, params_dir / "optimized_frame_0000.pt")

        iter_metrics_csv = csv_dir / "iter_metrics.csv"
        final_loss_summary_csv = csv_dir / "final_loss_summary.csv"
        base.save_csv_rows(iter_metrics_csv, optimization["iter_rows"])
        base.save_csv_rows(
            final_loss_summary_csv,
            [build_final_loss_summary_row(optimization["final_iter"], optimization["final_losses"])],
        )
        base.save_loss_plot_tree(
            plot_dir,
            optimization["iter_rows"],
            x_key="iter",
            total_key="total",
            term_keys=LOSS_TERM_KEYS,
            x_label="Iteration",
            title_prefix=f"{track.name} Iter",
        )

        final_contact_metrics = compute_contact_metrics(final_verts_camera, contact_edges + floor_edges)
        final_behind = compute_visible_behind_fraction(
            verts_camera=final_verts_camera,
            intrinsics=intrinsics,
            width=width,
            height=height,
            scene_depth=scene_depth,
            human_mask=human_mask,
            mask_vertex_ids=mask_vertex_ids,
            front_margin_m=float(args.front_margin_m),
        )

        summary_tracks.append(
            {
                "name": track.name,
                "optimization": {
                    "final_iter": int(optimization["final_iter"]),
                    "final_total_loss": float(optimization["final_total_loss"]),
                },
                "init_frame_0": {
                    "mask_overlap": init_mask_overlap,
                    "behind_fraction": init_behind,
                    "contact_edges": init_contact_metrics,
                },
                "final_frame_0": {
                    "mask_overlap": final_mask_overlap,
                    "behind_fraction": final_behind,
                    "contact_edges": final_contact_metrics,
                },
                "artifacts": {
                    "camera_mesh": str(meshes_root / "frame_0000_camera.ply"),
                    "world_mesh": str(meshes_root / "frame_0000_world.ply"),
                    "init_overlay": str(overlay_dir / "frame_0000_init_mask_overlay.png"),
                    "final_overlay": str(overlay_dir / "frame_0000_final_mask_overlay.png"),
                    "optimized_params": str(params_dir / "optimized_frame_0000.pt"),
                    "csv": {
                        "iter_metrics": str(iter_metrics_csv),
                        "final_loss_summary": str(final_loss_summary_csv),
                    },
                },
            }
        )

    base.save_json(
        summary_json_path,
        {
            "video_name": args.video_name,
            "scene_id": scene_id,
            "target_selection": selection_payload["target_selection"],
            "optimizer": {
                "adam_iters": int(args.adam_iters),
                "adam_lr": float(args.adam_lr),
                "loss_weights": {
                    "mask": float(args.mask_weight),
                    "front": float(args.front_weight),
                    "root_trans_gvhmr": float(args.root_trans_gvhmr_weight),
                    "root_orient_gvhmr": float(args.root_orient_gvhmr_weight),
                    "pose_gvhmr": float(args.pose_gvhmr_weight),
                    "betas_gvhmr": float(args.betas_gvhmr_weight),
                    "scale_prior": float(args.scale_prior_weight),
                    "intersect": {
                        "start": float(args.intersect_weight_start),
                        "end": float(args.intersect_weight_end),
                    },
                    "floor_intersect": {
                        "start": float(args.floor_intersect_weight_start),
                        "end": float(args.floor_intersect_weight_end),
                    },
                    "nocontact": {
                        "start": float(args.nocontact_weight_start),
                        "end": float(args.nocontact_weight_end),
                    },
                    "floor_nocontact": {
                        "start": float(args.floor_nocontact_weight_start),
                        "end": float(args.floor_nocontact_weight_end),
                    },
                    "angle": {
                        "start": float(args.angle_weight_start),
                        "end": float(args.angle_weight_end),
                    },
                    "self_intersect": {
                        "start": float(args.self_intersect_weight_start),
                        "end": float(args.self_intersect_weight_end),
                    },
                    "self_intersect_surrogate": {
                        "sample_vertices": int(args.self_intersect_sample_vertices),
                        "local_distance_thresh_m": float(args.self_intersect_local_dist_thresh_m),
                        "margin_m": float(args.self_intersect_margin_m),
                    },
                },
            },
            "scene": {
                "target_instance_id": target_instance_id,
                "target_label": target_meta["label"],
                "target_mask_overlap": target_mask_overlap,
                "target_visible_surface": target_visible_stats,
                "num_floor_visible_points": int(floor_points_visible.shape[0]),
            },
            "tracks": summary_tracks,
        },
    )
    print(f"\nDone. Full-body outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
