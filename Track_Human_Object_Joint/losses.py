"""Loss computation for human-object mesh refinement."""

from __future__ import annotations

import argparse

import numpy as np
import roma
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from geometry import (
    apply_inverse_similarity_sequence,
    apply_similarity_sequence,
    bounded_log_scale_delta,
    compose_T_sequence,
    project_points_with_intrinsics,
    query_sdf,
)
from models import (
    DiagnosticLossResult,
    FIXED_LOSS_WEIGHT_ATTRS,
    FRAME_DIAGNOSTIC_TERM_KEYS,
    HumanData,
    InteractionEdge,
    LossResult,
    LOSS_TERM_KEYS,
    ObjectData,
    SCHEDULED_LOSS_WEIGHT_ATTRS,
)


def l2_loss(x: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return ((x - y) ** 2).sum(dim).mean()


def simple_static_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] < 2:
        return x.new_tensor(0.0)
    return l2_loss(x[..., 1:, :], x[..., :-1, :])


def simple_smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2] < 3:
        return x.new_tensor(0.0)
    return l2_loss(x[..., 1:-1, :], 0.5 * (x[..., :-2, :] + x[..., 2:, :]))


def rotation_static_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] < 2:
        return x.new_tensor(0.0)
    x = roma.rotmat_to_rotvec(x)
    return simple_static_loss(x)


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


def geman_mcclure_func(residual: torch.Tensor, rho: float = 0.2) -> torch.Tensor:
    squared_res = residual**2
    dist = torch.div(squared_res, squared_res + rho**2)
    return rho**2 * dist


def pcd_distance(
    p1: torch.Tensor | None,
    p2: torch.Tensor | None,
    reduction: str = "min",
    error_func=None,
) -> torch.Tensor | None:
    if p1 is None or p2 is None:
        return None
    assert p1.ndim == p2.ndim == 3
    nnres = knn_points(p1=p1, p2=p2, norm=2, K=1)
    nndists = nnres.dists[..., 0]
    if error_func is not None:
        nndists = error_func(nndists)
    if reduction == "min":
        return torch.min(nndists, dim=1)[0]
    if reduction == "mean":
        return torch.mean(nndists, dim=1)
    raise RuntimeError(f"Unknown reduction: {reduction}")


def _sample_masks_bilinear_seq(
    masks: torch.Tensor,
    uv: torch.Tensor,
) -> torch.Tensor:
    if uv.numel() == 0:
        return torch.zeros(masks.shape[0], 0, dtype=torch.float32, device=masks.device)
    t, h, w = masks.shape
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


def _huber_on_squared(s: torch.Tensor, delta: float) -> torch.Tensor:
    d2 = delta * delta
    sqrt_s = torch.sqrt(s.clamp(min=1e-12))
    return torch.where(s <= d2, s, 2.0 * delta * sqrt_s - d2)


def linear_weight(start: float, end: float, step_id: int, n_steps: int) -> float:
    if n_steps <= 0:
        return float(start)
    return float(start) + (float(end) - float(start)) * float(step_id) / float(
        n_steps
    )


def get_loss_weights(
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for key, attr in FIXED_LOSS_WEIGHT_ATTRS.items():
        weights[key] = float(getattr(args, attr))
    for key, (start_attr, end_attr) in SCHEDULED_LOSS_WEIGHT_ATTRS.items():
        weights[key] = linear_weight(
            getattr(args, start_attr),
            getattr(args, end_attr),
            iteration,
            total_iters,
        )
    return weights


def get_scaled_loss_terms(result: LossResult) -> dict[str, torch.Tensor]:
    return {
        key: getattr(result, key) * float(result.weights[key])
        for key in LOSS_TERM_KEYS
    }


def _mean_over_non_time_dims(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 1:
        return values
    reduce_dims = tuple(range(values.ndim - 1))
    return values.mean(dim=reduce_dims)


def _sequence_static_diagnostic(
    seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = seq.shape[-2]
    per_frame = torch.zeros(num_frames, device=seq.device, dtype=seq.dtype)
    if num_frames < 2:
        return seq.new_tensor(0.0), per_frame

    step_vals = (seq[..., 1:, :] - seq[..., :-1, :]).pow(2).sum(dim=-1)
    step_vals = _mean_over_non_time_dims(step_vals)
    per_frame[1:] = step_vals
    return step_vals.mean(), per_frame


def _sequence_smoothness_diagnostic(
    seq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = seq.shape[-2]
    per_frame = torch.zeros(num_frames, device=seq.device, dtype=seq.dtype)
    if num_frames < 3:
        return seq.new_tensor(0.0), per_frame

    midpoint_diff = seq[..., 1:-1, :] - 0.5 * (
        seq[..., :-2, :] + seq[..., 2:, :]
    )
    midpoint_vals = midpoint_diff.pow(2).sum(dim=-1)
    midpoint_vals = _mean_over_non_time_dims(midpoint_vals)
    per_frame[1:-1] = midpoint_vals
    return midpoint_vals.mean(), per_frame


def _rotation_static_diagnostic(
    rot_mats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rot_mats.shape[0] < 2:
        per_frame = torch.zeros(rot_mats.shape[0], device=rot_mats.device)
        return rot_mats.new_tensor(0.0), per_frame
    rotvecs = roma.rotmat_to_rotvec(rot_mats)
    return _sequence_static_diagnostic(rotvecs)


def _rotation_smoothness_diagnostic(
    rot_mats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = rot_mats.shape[0]
    per_frame = torch.zeros(
        num_frames,
        device=rot_mats.device,
        dtype=rot_mats.dtype,
    )
    if num_frames < 3:
        return rot_mats.new_tensor(0.0), per_frame

    interp = roma.rotmat_slerp(
        rot_mats[:-2],
        rot_mats[2:],
        torch.tensor(0.5, device=rot_mats.device, dtype=rot_mats.dtype),
    )
    diff = roma.rotmat_geodesic_distance(interp, rot_mats[1:-1])
    vals = diff.pow(2)
    per_frame[1:-1] = vals
    return vals.mean(), per_frame


def _compute_object_scale_loss(scales: list[torch.Tensor]) -> torch.Tensor:
    if not scales:
        return torch.tensor(0.0)
    values = [F.relu(torch.abs(scale - 1.0) - 0.1).reshape(1) for scale in scales]
    return torch.stack(values, dim=0).mean()


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

    pts_canon = apply_inverse_similarity_sequence(
        world_points,
        obj_T_seq,
        obj_scale,
    )
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


def _build_effective_object_state(
    delta_rotvecs: dict[str, torch.Tensor],
    delta_trans: dict[str, torch.Tensor],
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    obj_keys: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    eff_T: dict[str, torch.Tensor] = {}
    eff_rot_mats: dict[str, torch.Tensor] = {}
    eff_trans: dict[str, torch.Tensor] = {}
    eff_scales: dict[str, torch.Tensor] = {}

    for slug in obj_keys:
        od = objects[slug]
        delta_T = compose_T_sequence(delta_rotvecs[slug], delta_trans[slug])
        T_eff = torch.matmul(od.tracked_poses_torch, delta_T)
        eff_T[slug] = T_eff
        eff_rot_mats[slug] = T_eff[:, :3, :3]
        eff_trans[slug] = T_eff[:, :3, 3]
        if args.optimize_object_scale:
            eff_scales[slug] = torch.exp(
                bounded_log_scale_delta(
                    raw_scale_deltas[slug],
                    args.max_log_scale_delta,
                )
            )
        else:
            eff_scales[slug] = torch.tensor(1.0, device=device)

    return eff_T, eff_rot_mats, eff_trans, eff_scales


def _compute_tracking_diagnostic(
    obj_data: ObjectData,
    obj_T_seq: torch.Tensor,
    args: argparse.Namespace,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = obj_T_seq.device
    num_frames = obj_T_seq.shape[0]
    per_frame = torch.zeros(num_frames, device=device, dtype=obj_T_seq.dtype)

    if obj_data.track_points0_cv.numel() == 0:
        return obj_T_seq.new_tensor(0.0), per_frame

    R_all = obj_T_seq[:, :3, :3]
    t_all = obj_T_seq[:, :3, 3]
    x0 = obj_data.track_points0_cv
    obs_uv = obj_data.track_obs_uv
    vis = obj_data.track_vis_tm
    masks = obj_data.track_masks

    xt = torch.einsum("tij,mj->tmi", R_all, x0) + t_all.unsqueeze(1)
    z = xt[..., 2]
    z_valid = z > 1e-6
    z_safe = torch.where(z_valid, z, torch.ones_like(z))
    pred_uv = torch.stack(
        [
            k[0, 0] * xt[..., 0] / z_safe + k[0, 2],
            k[1, 1] * xt[..., 1] / z_safe + k[1, 2],
        ],
        dim=-1,
    )

    finite_obs = torch.isfinite(obs_uv).all(dim=-1)
    mask_vals = _sample_masks_bilinear_seq(masks, obs_uv)
    mask_gate = mask_vals >= float(args.mask_gate_threshold)
    if float(args.visibility_threshold) > 0.0:
        vis_w = torch.where(
            vis >= float(args.visibility_threshold),
            vis,
            torch.zeros_like(vis),
        )
    else:
        vis_w = vis
    weights = vis_w * mask_gate.float() * finite_obs.float() * z_valid.float()
    if weights.sum().item() <= 0:
        return obj_T_seq.new_tensor(0.0), per_frame

    r2 = ((obs_uv - pred_uv) ** 2).sum(dim=-1)
    robust = _huber_on_squared(r2, float(args.huber_delta_px))
    loss = (weights * robust).sum() / weights.sum().clamp(min=1.0)

    frame_denom = weights.sum(dim=1)
    valid = frame_denom > 0
    per_frame[valid] = (
        (weights[valid] * robust[valid]).sum(dim=1)
        / frame_denom[valid].clamp(min=1.0)
    )
    return loss, per_frame


def _get_reduction(nodes: tuple) -> str:
    for node in nodes:
        if node.is_human and node.part_name.split(" ")[-1] in ("hand", "foot"):
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
            node.is_human
            and node.entity_name == human_name
            and node.part_name == human_part
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


def _get_human_device_and_num_frames(
    humans: dict[str, HumanData],
) -> tuple[torch.device, int]:
    if not humans:
        raise RuntimeError("No humans loaded for optimisation.")
    first_human = humans[next(iter(humans))]
    return first_human.base_verts.device, first_human.base_verts.shape[0]


def _concat_all_human_points(
    humans: dict[str, HumanData],
) -> torch.Tensor:
    return torch.cat(
        [humans[slug].base_verts for slug in sorted(humans)],
        dim=1,
    )


def _build_human_part_getter(
    humans: dict[str, HumanData],
    device: torch.device,
):
    human_part_cache: dict[tuple[str, ...], torch.Tensor | None] = {}

    def get_human_part_points(
        human_slug: str | None,
        part_name: str | list[str],
    ) -> torch.Tensor | None:
        if human_slug is None or human_slug not in humans:
            return None
        human_data = humans[human_slug]
        if isinstance(part_name, str):
            key = (human_slug, part_name)
            if key not in human_part_cache:
                human_part_cache[key] = human_data.part_points.get(part_name)
            return human_part_cache[key]

        key = tuple([human_slug] + sorted(part_name))
        if key not in human_part_cache:
            part_ids = [
                human_data.part_vert_ids[name]
                for name in part_name
                if name in human_data.part_vert_ids
            ]
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
    for j in range(num_frames):
        obs_len = int(observed_points.lengths[j].item())
        if obs_len == 0:
            continue
        obs_pts = observed_points.points[j, :obs_len, :].unsqueeze(0)
        model_pts = projected[j, :, :2].unsqueeze(0)
        cdist = pcd_distance(
            obs_pts,
            model_pts,
            reduction="mean",
            error_func=geman_mcclure_func,
        )
        if cdist is None:
            continue
        frame_loss = cdist.squeeze(0)
        per_frame[j] = frame_loss
        frame_losses.append(frame_loss)
    if not frame_losses:
        return None, per_frame
    return torch.stack(frame_losses, dim=0).mean(), per_frame


def compute_all_losses(
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
) -> LossResult:
    device, _ = _get_human_device_and_num_frames(humans)
    weights = get_loss_weights(args, iteration, total_iters)
    eff_T, eff_rot_mats, eff_trans, eff_scales = _build_effective_object_state(
        delta_rotvecs,
        delta_trans,
        raw_scale_deltas,
        objects,
        obj_keys,
        args,
        device,
    )

    object_points_cache: dict[tuple[str, str | None], torch.Tensor] = {}

    def get_object_points(slug: str, part_name: str | None = None) -> torch.Tensor:
        key = (slug, part_name)
        if key not in object_points_cache:
            od = objects[slug]
            if part_name and part_name in od.part_sampled_points:
                base_points = od.part_sampled_points[part_name]
            else:
                base_points = od.sampled_points
            object_points_cache[key] = apply_similarity_sequence(
                base_points,
                eff_T[slug],
                eff_scales[slug],
            )
        return object_points_cache[key]

    get_human_part_points = _build_human_part_getter(humans, device)

    def to_canonical(slug: str, points_world: torch.Tensor) -> torch.Tensor:
        return apply_inverse_similarity_sequence(
            points_world,
            eff_T[slug],
            eff_scales[slug],
        )

    tracking_values: list[torch.Tensor] = []
    for slug in obj_keys:
        scalar, _ = _compute_tracking_diagnostic(
            objects[slug],
            eff_T[slug],
            args,
            k,
        )
        tracking_values.append(scalar)
    if tracking_values:
        loss_tracking = torch.stack(tracking_values, dim=0).mean()
    else:
        loss_tracking = torch.tensor(0.0, device=device)

    object_cd2d_values: list[torch.Tensor] = []
    for slug in obj_keys:
        scalar, _ = _one_way_2d_chamfer_diagnostic(
            objects[slug].mask_points_2d,
            get_object_points(slug),
            k,
        )
        if scalar is not None:
            object_cd2d_values.append(scalar)
    if object_cd2d_values:
        loss_object_cd2d = torch.stack(object_cd2d_values, dim=0).mean()
    else:
        loss_object_cd2d = torch.tensor(0.0, device=device)

    object_part_cd2d_values: list[torch.Tensor] = []
    for slug in obj_keys:
        object_frame_values: list[torch.Tensor] = []
        for part_name, packed_points in objects[slug].part_mask_points_2d.items():
            projected = get_object_points(slug, part_name)
            scalar, _ = _one_way_2d_chamfer_diagnostic(
                packed_points,
                projected,
                k,
            )
            if scalar is not None:
                object_frame_values.append(scalar)
        if object_frame_values:
            object_part_cd2d_values.append(
                torch.stack(object_frame_values, dim=0).mean()
            )
    if object_part_cd2d_values:
        loss_object_part_cd2d = torch.stack(
            object_part_cd2d_values,
            dim=0,
        ).mean()
    else:
        loss_object_part_cd2d = torch.tensor(0.0, device=device)

    object_smooth_trans_values: list[torch.Tensor] = []
    object_smooth_rot_values: list[torch.Tensor] = []
    for slug in obj_keys:
        od = objects[slug]
        if od.state.is_translational:
            object_smooth_trans_values.append(simple_smoothness_loss(eff_trans[slug]))
        else:
            object_smooth_trans_values.append(simple_static_loss(eff_trans[slug]) * 10.0)

        if od.state.is_rotational:
            object_smooth_rot_values.append(
                rotation_smoothness_loss(eff_rot_mats[slug])
            )
        else:
            object_smooth_rot_values.append(rotation_static_loss(eff_rot_mats[slug]) * 10.0)

    if object_smooth_trans_values:
        loss_object_smooth_trans = torch.stack(
            object_smooth_trans_values,
            dim=0,
        ).mean()
        loss_object_smooth_rot = torch.stack(
            object_smooth_rot_values,
            dim=0,
        ).mean()
    else:
        loss_object_smooth_trans = torch.tensor(0.0, device=device)
        loss_object_smooth_rot = torch.tensor(0.0, device=device)

    scale_values = [eff_scales[slug] for slug in obj_keys] if args.optimize_object_scale else []
    if scale_values:
        loss_object_scale = _compute_object_scale_loss(scale_values).to(device)
    else:
        loss_object_scale = torch.tensor(0.0, device=device)

    all_human_points = _concat_all_human_points(humans)
    intersect_values: list[torch.Tensor] = []
    for slug in obj_keys:
        scalar, _ = _compute_intersect_diagnostic(
            all_human_points,
            objects[slug],
            eff_T[slug],
            eff_scales[slug],
        )
        if scalar.item() > 0.0:
            intersect_values.append(scalar)
    if intersect_values:
        loss_intersect = torch.stack(intersect_values, dim=0).mean()
    else:
        loss_intersect = torch.tensor(0.0, device=device)

    nocontact_values: list[torch.Tensor] = []
    contact_drift_values: list[torch.Tensor] = []
    visited: set[tuple[str, str, str, str]] = set()

    for edge in interaction_edges:
        nodes = [edge.node_a, edge.node_b]
        has_hpart = nodes[0].is_human or nodes[1].is_human
        if has_hpart and not nodes[0].is_human:
            nodes = [nodes[1], nodes[0]]
        reduction = _get_reduction((nodes[0], nodes[1]))
        dedup_key = (
            nodes[0].entity_name,
            nodes[0].part_name,
            nodes[1].entity_name,
            nodes[1].part_name,
        )
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

            object_points = get_object_points(
                object_node.object_slug,
                object_node.resolved_part_name,
            )
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

                has_left = _has_interaction(
                    interaction_edges,
                    hname,
                    f"left {hpart}",
                    oname,
                    opart,
                )
                has_right = _has_interaction(
                    interaction_edges,
                    hname,
                    f"right {hpart}",
                    oname,
                    opart,
                )
                if has_left and has_right:
                    human_points = get_human_part_points(
                        human_node.human_slug,
                        [f"left {hpart}", f"right {hpart}"]
                    )
                    pdists = pcd_distance(
                        human_points,
                        object_points,
                        reduction=reduction,
                    )
                    if human_points is not None:
                        pcano = to_canonical(object_node.object_slug, human_points)
                else:
                    human_points_left = get_human_part_points(
                        human_node.human_slug,
                        f"left {hpart}",
                    )
                    human_points_right = get_human_part_points(
                        human_node.human_slug,
                        f"right {hpart}",
                    )
                    pdists_left = pcd_distance(
                        human_points_left,
                        object_points,
                        reduction=reduction,
                    )
                    pdists_right = pcd_distance(
                        human_points_right,
                        object_points,
                        reduction=reduction,
                    )
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
            visited.add(
                (
                    nodes[0].entity_name,
                    nodes[0].part_name,
                    nodes[1].entity_name,
                    nodes[1].part_name,
                )
            )
            visited.add(
                (
                    nodes[1].entity_name,
                    nodes[1].part_name,
                    nodes[0].entity_name,
                    nodes[0].part_name,
                )
            )
            part_pcds = [
                get_object_points(nodes[0].object_slug, nodes[0].resolved_part_name),
                get_object_points(nodes[1].object_slug, nodes[1].resolved_part_name),
            ]
            part_diags = [
                torch.linalg.norm(
                    ppcd[0, :, :].max(dim=0)[0] - ppcd[0, :, :].min(dim=0)[0]
                ).item()
                for ppcd in part_pcds
            ]
            if part_diags[0] < part_diags[1]:
                pdists = pcd_distance(part_pcds[0], part_pcds[1], reduction=reduction)
            else:
                pdists = pcd_distance(part_pcds[1], part_pcds[0], reduction=reduction)
            pcano = to_canonical(nodes[0].object_slug, part_pcds[1])

        if pdists is None or pcano is None:
            continue

        if edge.is_continuous:
            nocontact_values.append(pdists.mean())
        else:
            nocontact_values.append(pdists.min())

        pcano_seq = pcano.permute(1, 0, 2).contiguous()
        if edge.is_rel_static:
            contact_drift_values.append(simple_static_loss(pcano_seq))
        else:
            contact_drift_values.append(simple_smoothness_loss(pcano_seq))

    if nocontact_values:
        loss_nocontact = torch.stack(nocontact_values, dim=0).mean()
    else:
        loss_nocontact = torch.tensor(0.0, device=device)
    if contact_drift_values:
        loss_contact_drift = torch.stack(contact_drift_values, dim=0).mean()
    else:
        loss_contact_drift = torch.tensor(0.0, device=device)

    total = loss_tracking * weights["tracking"]
    total = total + loss_object_cd2d * weights["object_cd2d"]
    total = total + loss_object_part_cd2d * weights["object_part_cd2d"]
    total = total + loss_object_smooth_trans * weights["object_smooth_trans"]
    total = total + loss_object_smooth_rot * weights["object_smooth_rot"]
    total = total + loss_object_scale * weights["object_scale"]
    total = total + loss_intersect * weights["intersect"]
    total = total + loss_nocontact * weights["nocontact"]
    total = total + loss_contact_drift * weights["contact_drift"]

    return LossResult(
        total=total,
        tracking=loss_tracking,
        object_cd2d=loss_object_cd2d,
        object_part_cd2d=loss_object_part_cd2d,
        object_smooth_trans=loss_object_smooth_trans,
        object_smooth_rot=loss_object_smooth_rot,
        object_scale=loss_object_scale,
        intersect=loss_intersect,
        nocontact=loss_nocontact,
        contact_drift=loss_contact_drift,
        weights=weights,
    )


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
    device, num_frames = _get_human_device_and_num_frames(humans)
    sequence = compute_all_losses(
        delta_rotvecs,
        delta_trans,
        raw_scale_deltas,
        objects,
        humans,
        interaction_edges,
        obj_keys,
        args,
        iteration,
        total_iters,
        k,
        width,
        height,
    )

    eff_T, eff_rot_mats, eff_trans, eff_scales = _build_effective_object_state(
        delta_rotvecs,
        delta_trans,
        raw_scale_deltas,
        objects,
        obj_keys,
        args,
        device,
    )

    object_points_cache: dict[tuple[str, str | None], torch.Tensor] = {}

    def get_object_points(slug: str, part_name: str | None = None) -> torch.Tensor:
        key = (slug, part_name)
        if key not in object_points_cache:
            od = objects[slug]
            if part_name and part_name in od.part_sampled_points:
                base_points = od.part_sampled_points[part_name]
            else:
                base_points = od.sampled_points
            object_points_cache[key] = apply_similarity_sequence(
                base_points,
                eff_T[slug],
                eff_scales[slug],
            )
        return object_points_cache[key]

    get_human_part_points = _build_human_part_getter(humans, device)

    def to_canonical(slug: str, points_world: torch.Tensor) -> torch.Tensor:
        return apply_inverse_similarity_sequence(
            points_world,
            eff_T[slug],
            eff_scales[slug],
        )

    per_frame_raw = {
        key: torch.zeros(num_frames, device=device)
        for key in FRAME_DIAGNOSTIC_TERM_KEYS
    }
    global_raw = {
        "object_scale": sequence.object_scale.detach().clone(),
    }
    all_human_points = _concat_all_human_points(humans)

    if obj_keys:
        for slug in obj_keys:
            _, frame_vals = _compute_tracking_diagnostic(
                objects[slug],
                eff_T[slug],
                args,
                k,
            )
            per_frame_raw["tracking"] = per_frame_raw["tracking"] + frame_vals
        per_frame_raw["tracking"] = per_frame_raw["tracking"] / float(len(obj_keys))

    if obj_keys:
        object_count = 0
        for slug in obj_keys:
            _, frame_vals = _one_way_2d_chamfer_diagnostic(
                objects[slug].mask_points_2d,
                get_object_points(slug),
                k,
            )
            per_frame_raw["object_cd2d"] = per_frame_raw["object_cd2d"] + frame_vals
            object_count += 1
        per_frame_raw["object_cd2d"] = per_frame_raw["object_cd2d"] / max(object_count, 1)

    part_object_count = 0
    for slug in obj_keys:
        object_accum = torch.zeros(num_frames, device=device)
        part_count = 0
        for part_name, packed_points in objects[slug].part_mask_points_2d.items():
            _, frame_vals = _one_way_2d_chamfer_diagnostic(
                packed_points,
                get_object_points(slug, part_name),
                k,
            )
            object_accum = object_accum + frame_vals
            part_count += 1
        if part_count > 0:
            per_frame_raw["object_part_cd2d"] = (
                per_frame_raw["object_part_cd2d"] + object_accum / part_count
            )
            part_object_count += 1
    if part_object_count > 0:
        per_frame_raw["object_part_cd2d"] = (
            per_frame_raw["object_part_cd2d"] / part_object_count
        )

    if obj_keys:
        for slug in obj_keys:
            od = objects[slug]
            if od.state.is_translational:
                _, trans_frames = _sequence_smoothness_diagnostic(eff_trans[slug])
            else:
                _, trans_frames = _sequence_static_diagnostic(eff_trans[slug])
                trans_frames = trans_frames * 10.0
            per_frame_raw["object_smooth_trans"] = (
                per_frame_raw["object_smooth_trans"] + trans_frames
            )

            if od.state.is_rotational:
                _, rot_frames = _rotation_smoothness_diagnostic(eff_rot_mats[slug])
            else:
                _, rot_frames = _rotation_static_diagnostic(eff_rot_mats[slug])
                rot_frames = rot_frames * 10.0
            per_frame_raw["object_smooth_rot"] = (
                per_frame_raw["object_smooth_rot"] + rot_frames
            )

            _, intersect_frames = _compute_intersect_diagnostic(
                all_human_points,
                objects[slug],
                eff_T[slug],
                eff_scales[slug],
            )
            per_frame_raw["intersect"] = (
                per_frame_raw["intersect"] + intersect_frames
            )

        denom = float(len(obj_keys))
        per_frame_raw["object_smooth_trans"] = (
            per_frame_raw["object_smooth_trans"] / denom
        )
        per_frame_raw["object_smooth_rot"] = (
            per_frame_raw["object_smooth_rot"] / denom
        )
        per_frame_raw["intersect"] = per_frame_raw["intersect"] / denom

    nocontact_edges = 0
    drift_edges = 0
    visited: set[tuple[str, str, str, str]] = set()
    for edge in interaction_edges:
        nodes = [edge.node_a, edge.node_b]
        has_hpart = nodes[0].is_human or nodes[1].is_human
        if has_hpart and not nodes[0].is_human:
            nodes = [nodes[1], nodes[0]]
        reduction = _get_reduction((nodes[0], nodes[1]))
        dedup_key = (
            nodes[0].entity_name,
            nodes[0].part_name,
            nodes[1].entity_name,
            nodes[1].part_name,
        )
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
            object_points = get_object_points(
                object_node.object_slug,
                object_node.resolved_part_name,
            )
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
                has_left = _has_interaction(
                    interaction_edges,
                    hname,
                    f"left {hpart}",
                    oname,
                    opart,
                )
                has_right = _has_interaction(
                    interaction_edges,
                    hname,
                    f"right {hpart}",
                    oname,
                    opart,
                )
                if has_left and has_right:
                    human_points = get_human_part_points(
                        human_node.human_slug,
                        [f"left {hpart}", f"right {hpart}"]
                    )
                    pdists = pcd_distance(
                        human_points,
                        object_points,
                        reduction=reduction,
                    )
                    if human_points is not None:
                        pcano = to_canonical(object_node.object_slug, human_points)
                else:
                    human_points_left = get_human_part_points(
                        human_node.human_slug,
                        f"left {hpart}",
                    )
                    human_points_right = get_human_part_points(
                        human_node.human_slug,
                        f"right {hpart}",
                    )
                    pdists_left = pcd_distance(
                        human_points_left,
                        object_points,
                        reduction=reduction,
                    )
                    pdists_right = pcd_distance(
                        human_points_right,
                        object_points,
                        reduction=reduction,
                    )
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
            visited.add(
                (
                    nodes[0].entity_name,
                    nodes[0].part_name,
                    nodes[1].entity_name,
                    nodes[1].part_name,
                )
            )
            visited.add(
                (
                    nodes[1].entity_name,
                    nodes[1].part_name,
                    nodes[0].entity_name,
                    nodes[0].part_name,
                )
            )
            part_pcds = [
                get_object_points(nodes[0].object_slug, nodes[0].resolved_part_name),
                get_object_points(nodes[1].object_slug, nodes[1].resolved_part_name),
            ]
            part_diags = [
                torch.linalg.norm(
                    ppcd[0, :, :].max(dim=0)[0] - ppcd[0, :, :].min(dim=0)[0]
                ).item()
                for ppcd in part_pcds
            ]
            if part_diags[0] < part_diags[1]:
                pdists = pcd_distance(part_pcds[0], part_pcds[1], reduction=reduction)
            else:
                pdists = pcd_distance(part_pcds[1], part_pcds[0], reduction=reduction)
            pcano = to_canonical(nodes[0].object_slug, part_pcds[1])

        if pdists is None or pcano is None:
            continue

        per_frame_raw["nocontact"] = per_frame_raw["nocontact"] + pdists
        nocontact_edges += 1

        pcano_seq = pcano.permute(1, 0, 2).contiguous()
        if edge.is_rel_static:
            _, drift_frames = _sequence_static_diagnostic(pcano_seq)
        else:
            _, drift_frames = _sequence_smoothness_diagnostic(pcano_seq)
        per_frame_raw["contact_drift"] = (
            per_frame_raw["contact_drift"] + drift_frames
        )
        drift_edges += 1

    if nocontact_edges > 0:
        per_frame_raw["nocontact"] = (
            per_frame_raw["nocontact"] / float(nocontact_edges)
        )
    if drift_edges > 0:
        per_frame_raw["contact_drift"] = (
            per_frame_raw["contact_drift"] / float(drift_edges)
        )

    return DiagnosticLossResult(
        sequence=sequence,
        per_frame_raw=per_frame_raw,
        global_raw=global_raw,
    )
