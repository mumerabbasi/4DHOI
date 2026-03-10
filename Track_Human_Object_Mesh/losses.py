"""Loss computation for joint human-object mesh refinement."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle

from geometry import (
    apply_inverse_similarity_batch,
    apply_inverse_similarity_sequence,
    apply_local_se3_sequence,
    apply_similarity_sequence,
    bounded_log_scale_delta,
    compose_T,
    geodesic_distance_sq,
    masked_mean_from_lengths,
    masked_mean_per_lengths,
    pack_projected_points,
    project_points_normalized_torch,
    query_sdf,
)
from models import (
    DiagnosticLossResult,
    FRAME_DIAGNOSTIC_TERM_KEYS,
    HumanData,
    LossResult,
    LOSS_WEIGHT_ATTRS,
    ObjectData,
    ResolvedEdge,
)


def get_scaled_loss_terms(
    result: LossResult,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    scaled_terms: dict[str, torch.Tensor] = {}
    for key, attr in LOSS_WEIGHT_ATTRS.items():
        scaled_terms[key] = getattr(args, attr) * getattr(result, key)
    return scaled_terms


def _compute_contact_per_frame(
    pts_src: torch.Tensor,
    pts_dst: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    d_sq = knn_points(pts_src, pts_dst, K=1).dists[..., 0].clamp(min=0.0)
    if reduction == "mean":
        return d_sq.mean(dim=1)
    if reduction == "min":
        return d_sq.min(dim=1).values
    raise ValueError(f"Unsupported contact reduction: {reduction}")


def _compute_contact_loss(
    pts_src: torch.Tensor,
    pts_dst: torch.Tensor,
    is_continuous: bool,
    reduction: str,
) -> torch.Tensor:
    per_frame = _compute_contact_per_frame(pts_src, pts_dst, reduction)
    if is_continuous:
        return per_frame.mean()
    return per_frame.min()


def _compute_dynamics_diagnostic(
    pts_contact: torch.Tensor,
    obj_T_mats: torch.Tensor,
    obj_scale: torch.Tensor,
    is_rel_static: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_frames = pts_contact.shape[0]
    per_frame = torch.zeros(num_frames, device=pts_contact.device)
    if num_frames < 2:
        return torch.tensor(0.0, device=pts_contact.device), per_frame

    canonical = apply_inverse_similarity_sequence(
        pts_contact,
        obj_T_mats,
        obj_scale,
    )

    if is_rel_static:
        diff = canonical[1:] - canonical[:-1]
        step_vals = (diff ** 2).mean(dim=(1, 2))
        per_frame[1:] = step_vals
        return step_vals.mean(), per_frame

    if num_frames < 3:
        diff = canonical[1:] - canonical[:-1]
        step_vals = (diff ** 2).mean(dim=(1, 2)) * 0.1
        per_frame[1:] = step_vals
        return step_vals.mean(), per_frame

    mid = canonical[1:-1]
    avg = 0.5 * (canonical[:-2] + canonical[2:])
    accel = mid - avg
    accel_vals = (accel ** 2).mean(dim=(1, 2))
    per_frame[1:-1] = accel_vals
    return accel_vals.mean(), per_frame


def _compute_dynamics_loss(
    pts_contact: torch.Tensor,
    obj_T_mats: torch.Tensor,
    obj_scale: torch.Tensor,
    is_rel_static: bool,
) -> torch.Tensor:
    scalar, _ = _compute_dynamics_diagnostic(
        pts_contact,
        obj_T_mats,
        obj_scale,
        is_rel_static,
    )
    return scalar


def _compute_penetration_loss(
    human_points_t: torch.Tensor,
    obj_data: ObjectData,
    obj_T: torch.Tensor,
    obj_scale: torch.Tensor,
) -> torch.Tensor:
    if obj_data.sdf_grid is None:
        return torch.tensor(0.0, device=human_points_t.device)

    pts_canon = apply_inverse_similarity_batch(
        human_points_t,
        obj_T,
        obj_scale,
    )
    sdf_vals = query_sdf(obj_data.sdf_grid, pts_canon)
    penetration = F.relu(-sdf_vals)
    n_inside = (penetration > 0).sum().clamp(min=1)
    return penetration.sum() / n_inside.float()


def _compute_obj_obj_penetration_loss(
    obj_a_world_points: torch.Tensor,
    obj_b: ObjectData,
    obj_b_T: torch.Tensor,
    obj_b_scale: torch.Tensor,
) -> torch.Tensor:
    if obj_b.sdf_grid is None:
        return torch.tensor(0.0, device=obj_a_world_points.device)

    pts_in_b_canon = apply_inverse_similarity_batch(
        obj_a_world_points,
        obj_b_T,
        obj_b_scale,
    )
    sdf_vals = query_sdf(obj_b.sdf_grid, pts_in_b_canon)
    penetration = F.relu(-sdf_vals)
    n_inside = (penetration > 0).sum().clamp(min=1)
    return penetration.sum() / n_inside.float()


def _compute_smoothness_diagnostic(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    is_translational: bool,
    is_rotational: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = rotvecs.device
    num_frames = rotvecs.shape[0]
    loss = torch.tensor(0.0, device=device)
    per_frame = torch.zeros(num_frames, device=device)

    if num_frames < 2:
        return loss, per_frame

    R_mats = axis_angle_to_matrix(rotvecs)
    if is_rotational:
        if num_frames >= 3:
            geo_dists = []
            for t in range(1, num_frames - 1):
                mid = 0.5 * (rotvecs[t - 1] + rotvecs[t + 1])
                diff = rotvecs[t] - mid
                val = (diff ** 2).sum()
                geo_dists.append(val)
                per_frame[t] = per_frame[t] + val
            loss = loss + torch.stack(geo_dists).mean()
        else:
            diff = rotvecs[1:] - rotvecs[:-1]
            step_vals = (diff ** 2).mean(dim=1)
            per_frame[1:] = per_frame[1:] + step_vals
            loss = loss + step_vals.mean()
    else:
        rot_vals = []
        for t in range(num_frames - 1):
            gd_sq = geodesic_distance_sq(R_mats[t], R_mats[t + 1])
            val = 10.0 * gd_sq
            rot_vals.append(val)
            per_frame[t + 1] = per_frame[t + 1] + val
        loss = loss + torch.stack(rot_vals).mean()

    if is_translational:
        if num_frames >= 3:
            accel = trans[2:] + trans[:-2] - 2.0 * trans[1:-1]
            accel_vals = (accel ** 2).mean(dim=1)
            per_frame[1:-1] = per_frame[1:-1] + accel_vals
            loss = loss + accel_vals.mean()
        else:
            diff = trans[1:] - trans[:-1]
            step_vals = (diff ** 2).mean(dim=1)
            per_frame[1:] = per_frame[1:] + step_vals
            loss = loss + step_vals.mean()
    else:
        diff = trans[1:] - trans[:-1]
        step_vals = 10.0 * (diff ** 2).mean(dim=1)
        per_frame[1:] = per_frame[1:] + step_vals
        loss = loss + step_vals.mean()

    return loss, per_frame


def _compute_smoothness_loss(
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    is_translational: bool,
    is_rotational: bool,
) -> torch.Tensor:
    scalar, _ = _compute_smoothness_diagnostic(
        rotvecs,
        trans,
        is_translational,
        is_rotational,
    )
    return scalar


def _compute_bidirectional_2d_chamfer_diagnostic(
    observed_points,
    model_points_world: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame = torch.zeros(
        model_points_world.shape[0],
        dtype=model_points_world.dtype,
        device=model_points_world.device,
    )
    if observed_points is None:
        return torch.tensor(0.0, device=model_points_world.device), per_frame

    projected, valid = project_points_normalized_torch(
        model_points_world,
        k,
        width,
        height,
    )
    model_packed, model_lengths = pack_projected_points(projected, valid)
    valid_frames = (observed_points.lengths > 0) & (model_lengths > 0)
    if not torch.any(valid_frames):
        return torch.tensor(0.0, device=model_points_world.device), per_frame

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
    loss_fwd = masked_mean_from_lengths(obs_to_model, obs_lengths)
    loss_bwd = masked_mean_from_lengths(model_to_obs, model_lengths)
    local_fwd = masked_mean_per_lengths(obs_to_model, obs_lengths)
    local_bwd = masked_mean_per_lengths(model_to_obs, model_lengths)
    per_frame[valid_frames] = 0.5 * (local_fwd + local_bwd)
    return 0.5 * (loss_fwd + loss_bwd), per_frame


def _compute_bidirectional_2d_chamfer(
    observed_points,
    model_points_world: torch.Tensor,
    k: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    scalar, _ = _compute_bidirectional_2d_chamfer_diagnostic(
        observed_points,
        model_points_world,
        k,
        width,
        height,
    )
    return scalar


def penetration_weight_schedule(iteration: int, total_iters: int) -> float:
    return min(iteration / max(total_iters * 0.5, 1.0), 1.0)


def compute_all_losses(
    delta_rotvecs: dict[str, torch.Tensor],
    delta_trans: dict[str, torch.Tensor],
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    human_data: HumanData,
    human_delta_rotvecs: torch.Tensor,
    human_delta_trans: torch.Tensor,
    resolved_edges: list[ResolvedEdge],
    obj_keys: list[str],
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
    k: torch.Tensor,
    width: int,
    height: int,
) -> LossResult:
    device = human_data.base_verts.device
    num_frames = human_data.base_verts.shape[0]

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
            delta = compose_T(delta_rotvecs[slug][t], delta_trans[slug][t])
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
                bounded_log_scale_delta(
                    raw_scale_deltas[slug],
                    args.max_log_scale_delta,
                )
            )
        else:
            eff_scales[slug] = torch.tensor(1.0, device=device)

    if args.optimize_human:
        human_points_whole = apply_local_se3_sequence(
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
                human_part_cache[part_name] = apply_local_se3_sequence(
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
            object_points_cache[key] = apply_similarity_sequence(
                base_points,
                eff_T[slug],
                eff_scales[slug],
            )
        return object_points_cache[key]

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
            delta_log_scale = bounded_log_scale_delta(
                raw_scale_deltas[slug],
                args.max_log_scale_delta,
            )
            loss_object_scale_reg = (
                loss_object_scale_reg + delta_log_scale.pow(2)
            )
        loss_object_scale_reg = loss_object_scale_reg / len(obj_keys)

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
            pts_src,
            pts_dst,
            edge.is_continuous,
            edge.contact_reduction,
        )
        n_edges_contact += 1
    if n_edges_contact > 0:
        loss_contact = loss_contact / n_edges_contact

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
                pts_contact = _get_object_points(
                    obj_keys[edge.b_object_idx],
                    edge.b_part_name,
                )
        else:
            if edge.a_is_human:
                pts_contact = _get_human_part_points(edge.a_part_name)
            else:
                pts_contact = _get_object_points(
                    obj_keys[edge.a_object_idx],
                    edge.a_part_name,
                )

        loss_dynamics = loss_dynamics + _compute_dynamics_loss(
            pts_contact,
            ref_T,
            ref_scale,
            edge.is_rel_static,
        )
        n_edges_dyn += 1
    if n_edges_dyn > 0:
        loss_dynamics = loss_dynamics / n_edges_dyn

    pen_weight_schedule = penetration_weight_schedule(iteration, total_iters)

    loss_pen = torch.tensor(0.0, device=device)
    n_pen = 0
    for t in range(num_frames):
        human_sub = human_points_whole[t]
        for slug in obj_keys:
            loss_pen = loss_pen + _compute_penetration_loss(
                human_sub,
                objects[slug],
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

    loss_smooth = torch.tensor(0.0, device=device)
    for slug in obj_keys:
        loss_smooth = loss_smooth + _compute_smoothness_loss(
            eff_rotvecs[slug],
            eff_trans[slug],
            objects[slug].state.is_translational,
            objects[slug].state.is_rotational,
        )
    if obj_keys:
        loss_smooth = loss_smooth / len(obj_keys)

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
    )


def compute_final_loss_diagnostics(
    delta_rotvecs: dict[str, torch.Tensor],
    delta_trans: dict[str, torch.Tensor],
    raw_scale_deltas: dict[str, torch.Tensor],
    objects: dict[str, ObjectData],
    human_data: HumanData,
    human_delta_rotvecs: torch.Tensor,
    human_delta_trans: torch.Tensor,
    resolved_edges: list[ResolvedEdge],
    obj_keys: list[str],
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
    k: torch.Tensor,
    width: int,
    height: int,
) -> DiagnosticLossResult:
    device = human_data.base_verts.device
    num_frames = human_data.base_verts.shape[0]
    sequence = compute_all_losses(
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
        iteration,
        total_iters,
        k,
        width,
        height,
    )

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
            delta = compose_T(delta_rotvecs[slug][t], delta_trans[slug][t])
            T_eff = base @ delta
            T_list.append(T_eff)
            rvs.append(
                matrix_to_axis_angle(T_eff[:3, :3].unsqueeze(0)).squeeze(0)
            )
            trs.append(T_eff[:3, 3])
        eff_T[slug] = torch.stack(T_list, dim=0)
        eff_rotvecs[slug] = torch.stack(rvs)
        eff_trans[slug] = torch.stack(trs)
        if args.optimize_object_scale:
            eff_scales[slug] = torch.exp(
                bounded_log_scale_delta(
                    raw_scale_deltas[slug],
                    args.max_log_scale_delta,
                )
            )
        else:
            eff_scales[slug] = torch.tensor(1.0, device=device)

    if args.optimize_human:
        human_points_whole = apply_local_se3_sequence(
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
                human_part_cache[part_name] = apply_local_se3_sequence(
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
            object_points_cache[key] = apply_similarity_sequence(
                base_points,
                eff_T[slug],
                eff_scales[slug],
            )
        return object_points_cache[key]

    per_frame_raw = {
        key: torch.zeros(num_frames, device=device)
        for key in FRAME_DIAGNOSTIC_TERM_KEYS
    }
    global_raw = {
        "object_scale_reg": sequence.object_scale_reg.detach().clone(),
    }

    n_params = max(
        sum(
            delta_rotvecs[s].numel() + delta_trans[s].numel()
            for s in obj_keys
        ),
        1,
    )
    for slug in obj_keys:
        per_frame_raw["prior"] = per_frame_raw["prior"] + (
            delta_rotvecs[slug].pow(2).sum(dim=1)
            + delta_trans[slug].pow(2).sum(dim=1)
        )
    per_frame_raw["prior"] = per_frame_raw["prior"] / float(n_params)

    if args.optimize_human:
        human_prior_denom = float(
            human_delta_rotvecs.numel() + human_delta_trans.numel()
        )
        per_frame_raw["human_prior"] = (
            human_delta_rotvecs.pow(2).sum(dim=1)
            + human_delta_trans.pow(2).sum(dim=1)
        ) / human_prior_denom
        _, per_frame_raw["human_smooth"] = _compute_smoothness_diagnostic(
            human_delta_rotvecs,
            human_delta_trans,
            is_translational=True,
            is_rotational=True,
        )
        (
            _,
            per_frame_raw["human_mask_2d"],
        ) = _compute_bidirectional_2d_chamfer_diagnostic(
            human_data.mask_points_2d,
            human_points_whole,
            k,
            width,
            height,
        )

    if resolved_edges:
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
            per_frame_raw["contact"] = (
                per_frame_raw["contact"]
                + _compute_contact_per_frame(
                    pts_src,
                    pts_dst,
                    edge.contact_reduction,
                )
            )
        per_frame_raw["contact"] = (
            per_frame_raw["contact"] / len(resolved_edges)
        )

    num_edges_dyn = 0
    for edge in resolved_edges:
        if edge.canonical_obj_idx < 0:
            continue
        ref_slug = obj_keys[edge.canonical_obj_idx]
        if edge.canonical_obj_idx == edge.a_object_idx:
            if edge.b_is_human:
                pts_contact = _get_human_part_points(edge.b_part_name)
            else:
                pts_contact = _get_object_points(
                    obj_keys[edge.b_object_idx],
                    edge.b_part_name,
                )
        else:
            if edge.a_is_human:
                pts_contact = _get_human_part_points(edge.a_part_name)
            else:
                pts_contact = _get_object_points(
                    obj_keys[edge.a_object_idx],
                    edge.a_part_name,
                )
        _, dyn_frames = _compute_dynamics_diagnostic(
            pts_contact,
            eff_T[ref_slug],
            eff_scales[ref_slug],
            edge.is_rel_static,
        )
        per_frame_raw["dynamics"] = per_frame_raw["dynamics"] + dyn_frames
        num_edges_dyn += 1
    if num_edges_dyn > 0:
        per_frame_raw["dynamics"] = per_frame_raw["dynamics"] / num_edges_dyn

    if obj_keys:
        for slug in obj_keys:
            _, smooth_frames = _compute_smoothness_diagnostic(
                eff_rotvecs[slug],
                eff_trans[slug],
                objects[slug].state.is_translational,
                objects[slug].state.is_rotational,
            )
            per_frame_raw["smooth"] = per_frame_raw["smooth"] + smooth_frames
        per_frame_raw["smooth"] = per_frame_raw["smooth"] / len(obj_keys)

        for slug in obj_keys:
            _, mask_frames = _compute_bidirectional_2d_chamfer_diagnostic(
                objects[slug].mask_points_2d,
                _get_object_points(slug),
                k,
                width,
                height,
            )
            per_frame_raw["object_mask_2d"] = (
                per_frame_raw["object_mask_2d"] + mask_frames
            )
        per_frame_raw["object_mask_2d"] = (
            per_frame_raw["object_mask_2d"] / len(obj_keys)
        )

    num_part_terms = 0
    for slug in obj_keys:
        for part_name, packed_points in (
            objects[slug].part_mask_points_2d.items()
        ):
            _, part_frames = _compute_bidirectional_2d_chamfer_diagnostic(
                packed_points,
                _get_object_points(slug, part_name),
                k,
                width,
                height,
            )
            per_frame_raw["object_part_mask_2d"] = (
                per_frame_raw["object_part_mask_2d"] + part_frames
            )
            num_part_terms += 1
    if num_part_terms > 0:
        per_frame_raw["object_part_mask_2d"] = (
            per_frame_raw["object_part_mask_2d"] / num_part_terms
        )

    pen_weight = penetration_weight_schedule(iteration, total_iters)
    pen_counts = torch.zeros(num_frames, device=device)
    whole_object_points = {slug: _get_object_points(slug) for slug in obj_keys}
    for t in range(num_frames):
        human_sub = human_points_whole[t]
        for slug in obj_keys:
            per_frame_raw["penetration"][t] = (
                per_frame_raw["penetration"][t]
                + _compute_penetration_loss(
                    human_sub,
                    objects[slug],
                    eff_T[slug][t],
                    eff_scales[slug],
                )
            )
            pen_counts[t] = pen_counts[t] + 1.0
        for i in range(len(obj_keys)):
            for j in range(i + 1, len(obj_keys)):
                per_frame_raw["penetration"][t] = (
                    per_frame_raw["penetration"][t]
                    + _compute_obj_obj_penetration_loss(
                        whole_object_points[obj_keys[i]][t],
                        objects[obj_keys[j]],
                        eff_T[obj_keys[j]][t],
                        eff_scales[obj_keys[j]],
                    )
                )
                pen_counts[t] = pen_counts[t] + 1.0
                per_frame_raw["penetration"][t] = (
                    per_frame_raw["penetration"][t]
                    + _compute_obj_obj_penetration_loss(
                        whole_object_points[obj_keys[j]][t],
                        objects[obj_keys[i]],
                        eff_T[obj_keys[i]][t],
                        eff_scales[obj_keys[i]],
                    )
                )
                pen_counts[t] = pen_counts[t] + 1.0
    valid_pen = pen_counts > 0
    per_frame_raw["penetration"][valid_pen] = (
        per_frame_raw["penetration"][valid_pen] / pen_counts[valid_pen]
    )
    per_frame_raw["penetration"] = per_frame_raw["penetration"] * pen_weight

    return DiagnosticLossResult(
        sequence=sequence,
        per_frame_raw=per_frame_raw,
        global_raw=global_raw,
    )
