"""Optimization loop for joint human-object refinement."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import cv2
import numpy as np
import torch

from debug_utils import (
    build_final_loss_summary_row,
    build_frame_loss_rows,
    build_loss_row,
    format_loss_log,
)
from geometry import bounded_log_scale_delta, compose_T_sequence
from losses import compute_all_losses, compute_final_loss_diagnostics
from models import OptimizationResult, ProblemContext


def _print_optimization_header(
    context: ProblemContext,
    args: argparse.Namespace,
) -> None:
    print("=" * 60)
    print("Human-Object Mesh Refinement")
    print(f"  video:    {args.video_name}")
    print(f"  device:   {context.device}")
    print(f"  frames:   {context.num_frames}")
    print(f"  humans:   {', '.join(context.human_keys)}")
    print(f"  objects:  {', '.join(context.obj_keys)}")
    print(f"  edges:    {len(context.interaction_edges)}")
    print(
        f"  humans:   fixed  object_scale: "
        f"{'on' if args.optimize_object_scale else 'off'}"
    )
    print(
        f"  K: fx={context.k[0, 0]:.1f}  fy={context.k[1, 1]:.1f}  "
        f"cx={context.k[0, 2]:.1f}  cy={context.k[1, 2]:.1f}"
    )
    print(
        f"  reprojection_weight={args.tracking_weight}  "
        f"huber_delta_px={args.huber_delta_px}"
    )
    print(
        f"  init: PnP={'off' if args.disable_pnp_init else 'on'}  "
        f"ransac={args.pnp_ransac_thresh}px  "
        f"outlier={args.outlier_reproj_thresh_px}px/{args.outlier_max_fraction:.2f}"
    )
    print(
        "  schedules:"
        f" obj2d=({args.object_cd2d_weight_start},{args.object_cd2d_weight_end})"
        f" part2d=({args.object_part_cd2d_weight_start},{args.object_part_cd2d_weight_end})"
        f" smooth_t=({args.object_smooth_trans_weight_start},{args.object_smooth_trans_weight_end})"
        f" smooth_r=({args.object_smooth_rot_weight_start},{args.object_smooth_rot_weight_end})"
    )
    print(
        "             "
        f"scale=({args.object_scale_weight_start},{args.object_scale_weight_end})"
        f" intersect=({args.intersect_weight_start},{args.intersect_weight_end})"
        f" nocontact=({args.nocontact_weight_start},{args.nocontact_weight_end})"
        f" drift=({args.contact_drift_weight_start},{args.contact_drift_weight_end})"
    )
    print("=" * 60)


def _pnp_sequential_init(
    x0_cv: np.ndarray,
    obs_uv_tm: np.ndarray,
    vis_tm: np.ndarray,
    k: np.ndarray,
    ransac_thresh: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    t_frames = obs_uv_tm.shape[0]
    rotvecs = np.zeros((t_frames, 3), dtype=np.float32)
    trans = np.zeros((t_frames, 3), dtype=np.float32)
    info: list[dict[str, Any]] = []

    prev_rvec = np.zeros((3, 1), dtype=np.float64)
    prev_tvec = np.zeros((3, 1), dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)
    k64 = k.astype(np.float64)

    info.append({"frame": 0, "n_valid": int(x0_cv.shape[0]), "n_inliers": int(x0_cv.shape[0]), "pnp_ok": True})

    for t in range(1, t_frames):
        valid = vis_tm[t] > 0.5
        n_valid = int(valid.sum())
        if n_valid < 6:
            rotvecs[t] = prev_rvec.ravel().astype(np.float32)
            trans[t] = prev_tvec.ravel().astype(np.float32)
            info.append({"frame": t, "n_valid": n_valid, "n_inliers": 0, "pnp_ok": False})
            continue

        pts3d = x0_cv[valid].astype(np.float64)
        pts2d = obs_uv_tm[t, valid].astype(np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d,
            pts2d,
            k64,
            dist_coeffs,
            rvec=prev_rvec.copy(),
            tvec=prev_tvec.copy(),
            useExtrinsicGuess=(t > 1),
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
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()
        elif success:
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()

        rotvecs[t] = prev_rvec.ravel().astype(np.float32)
        trans[t] = prev_tvec.ravel().astype(np.float32)
        info.append({"frame": t, "n_valid": n_valid, "n_inliers": n_inliers, "pnp_ok": bool(success)})

    return rotvecs, trans, info


def _identify_outlier_tracks(
    x0: torch.Tensor,
    obs_uv: torch.Tensor,
    vis: torch.Tensor,
    rotvecs: torch.Tensor,
    trans: torch.Tensor,
    k: np.ndarray,
    threshold_px: float,
    max_fraction: float,
) -> torch.Tensor:
    with torch.no_grad():
        t_mats = compose_T_sequence(rotvecs, trans)
        r_all = t_mats[:, :3, :3]
        t_all = t_mats[:, :3, 3]

        xt = torch.einsum("tij,mj->tmi", r_all, x0) + t_all.unsqueeze(1)
        z = xt[..., 2].clamp(min=1e-6)
        pred_uv = torch.stack(
            [
                float(k[0, 0]) * xt[..., 0] / z + float(k[0, 2]),
                float(k[1, 1]) * xt[..., 1] / z + float(k[1, 2]),
            ],
            dim=-1,
        )
        err = torch.sqrt(((obs_uv - pred_uv) ** 2).sum(dim=-1).clamp(min=1e-12))
        vis_binary = (vis > 0.5).float()
        vis_count = vis_binary.sum(dim=0).clamp(min=1.0)
        mean_err = (err * vis_binary).sum(dim=0) / vis_count

        outlier = mean_err > threshold_px
        m = x0.shape[0]
        n_outlier = int(outlier.sum().item())
        max_reject = int(m * max_fraction)
        if n_outlier > max_reject and max_reject > 0:
            _, sorted_idx = mean_err.sort(descending=True)
            outlier = torch.zeros(m, dtype=torch.bool, device=x0.device)
            outlier[sorted_idx[:max_reject]] = True
    return outlier


def _initialise_object_pose_sequences(
    context: ProblemContext,
    args: argparse.Namespace,
) -> None:
    print("\n[Init] Building object pose initialisation from CoTracker observations...")
    for slug in context.obj_keys:
        od = context.objects[slug]
        t_frames = od.track_obs_uv.shape[0]
        if args.disable_pnp_init or t_frames <= 1:
            rot_init = torch.zeros((t_frames, 3), device=context.device)
            trans_init = torch.zeros((t_frames, 3), device=context.device)
            od.pnp_init_info = []
            print(f"  {slug}: identity initialisation")
        else:
            rot_np, trans_np, info = _pnp_sequential_init(
                od.track_points0_cv.detach().cpu().numpy(),
                od.track_obs_uv.detach().cpu().numpy(),
                od.track_vis_tm.detach().cpu().numpy(),
                context.k,
                float(args.pnp_ransac_thresh),
            )
            rot_init = torch.from_numpy(rot_np).float().to(context.device)
            trans_init = torch.from_numpy(trans_np).float().to(context.device)
            od.pnp_init_info = info

            ok_count = sum(1 for row in info[1:] if row["pnp_ok"]) if len(info) > 1 else 0
            max_trans = float(np.linalg.norm(trans_np, axis=1).max()) if trans_np.size else 0.0
            max_rot = float(np.linalg.norm(rot_np, axis=1).max()) if rot_np.size else 0.0
            print(
                f"  {slug}: PnP {ok_count}/{max(t_frames - 1, 1)} frames, "
                f"max_trans={max_trans:.4f}m, max_rot={np.degrees(max_rot):.1f}deg"
            )

            if float(args.outlier_reproj_thresh_px) > 0.0:
                outlier_mask = _identify_outlier_tracks(
                    od.track_points0_cv,
                    od.track_obs_uv,
                    od.track_vis_tm,
                    rot_init,
                    trans_init,
                    context.k,
                    float(args.outlier_reproj_thresh_px),
                    float(args.outlier_max_fraction),
                )
                n_outlier = int(outlier_mask.sum().item())
                if n_outlier > 0:
                    od.track_vis_tm[:, outlier_mask] = 0.0
                    print(
                        f"    outlier rejection: removed {n_outlier}/{od.track_points0_cv.shape[0]} tracks"
                    )

        tracked_poses = compose_T_sequence(rot_init, trans_init)
        od.tracked_rotvecs = rot_init.detach().clone()
        od.tracked_trans = trans_init.detach().clone()
        od.tracked_poses_torch = tracked_poses.detach().clone()
        od.tracked_poses = tracked_poses.detach().cpu().numpy().astype(np.float32)
        od.track_valid_count = int((od.track_vis_tm.max(dim=0).values > 0).sum().item())


def run_joint_optimization(
    context: ProblemContext,
    args: argparse.Namespace,
) -> OptimizationResult:
    device = context.device
    num_frames = context.num_frames
    _print_optimization_header(context, args)
    _initialise_object_pose_sequences(context, args)

    delta_rotvecs: dict[str, torch.Tensor] = {}
    delta_trans: dict[str, torch.Tensor] = {}
    raw_scale_deltas: dict[str, torch.Tensor] = {}
    params: list[torch.Tensor] = []

    for slug in context.obj_keys:
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

    total_iters = args.adam_iters
    optimizer = torch.optim.Adam(params, lr=args.adam_lr)
    iter_rows: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_iter = -1
    best_state: dict[str, dict[str, torch.Tensor]] = {}
    no_improve_iters = 0
    early_stop_triggered = False
    early_stop_enabled = args.early_stop_patience > 0

    print(f"\n[Adam] {args.adam_iters} iterations, lr={args.adam_lr}")
    optimisation_start = time.perf_counter()

    for it in range(args.adam_iters):
        optimizer.zero_grad(set_to_none=True)
        result = compute_all_losses(
            delta_rotvecs,
            delta_trans,
            raw_scale_deltas,
            context.objects,
            context.humans,
            context.interaction_edges,
            context.obj_keys,
            args,
            iteration=it,
            total_iters=total_iters,
            k=context.k_torch,
            width=context.width,
            height=context.height,
        )
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        loss_val = result.total.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            if best_state:
                for slug in context.obj_keys:
                    delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
                    delta_trans[slug].data.copy_(best_state[slug]["dt"])
                    raw_scale_deltas[slug].data.copy_(best_state[slug]["ds"])
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.5
                print(
                    f"  [{it:4d}] NaN detected; "
                    "restored best state and halved lr."
                )
            continue

        rel_improve = (
            (best_loss - loss_val) / max(abs(best_loss), 1e-8)
            if math.isfinite(best_loss)
            else float("inf")
        )
        if loss_val < best_loss:
            best_loss = loss_val
            best_iter = it
            best_state = {
                slug: {
                    "dr": delta_rotvecs[slug].detach().clone(),
                    "dt": delta_trans[slug].detach().clone(),
                    "ds": raw_scale_deltas[slug].detach().clone(),
                }
                for slug in context.obj_keys
            }
            if rel_improve > args.early_stop_rel_improve:
                no_improve_iters = 0
            elif early_stop_enabled and it >= args.early_stop_start:
                no_improve_iters += 1
        elif early_stop_enabled and it >= args.early_stop_start:
            no_improve_iters += 1

        if it % args.log_interval == 0 or it == args.adam_iters - 1:
            row = build_loss_row(it, result)
            iter_rows.append(row)
            if (
                args.verbose
                or it % (args.log_interval * 4) == 0
                or it == args.adam_iters - 1
            ):
                for line in format_loss_log(it, args.adam_iters, result):
                    print(line)

        if early_stop_enabled and no_improve_iters >= args.early_stop_patience:
            early_stop_triggered = True
            print(
                f"  Early stop at iter {it}: no relative improvement "
                f"greater than {args.early_stop_rel_improve:.1e} for "
                f"{args.early_stop_patience} iterations."
            )
            break

    if not best_state or best_iter < 0:
        raise RuntimeError("Optimisation did not produce a valid checkpoint.")

    for slug in context.obj_keys:
        delta_rotvecs[slug].data.copy_(best_state[slug]["dr"])
        delta_trans[slug].data.copy_(best_state[slug]["dt"])
        raw_scale_deltas[slug].data.copy_(best_state[slug]["ds"])

    with torch.no_grad():
        final_diagnostic = compute_final_loss_diagnostics(
            delta_rotvecs,
            delta_trans,
            raw_scale_deltas,
            context.objects,
            context.humans,
            context.interaction_edges,
            context.obj_keys,
            args,
            iteration=best_iter,
            total_iters=total_iters,
            k=context.k_torch,
            width=context.width,
            height=context.height,
        )
    best_loss = float(final_diagnostic.sequence.total.item())
    frame_rows = build_frame_loss_rows(0, final_diagnostic)
    final_loss_summary_row = build_final_loss_summary_row(
        best_iter,
        final_diagnostic.sequence,
    )
    optimisation_time_s = time.perf_counter() - optimisation_start

    print(
        f"\nOptimisation complete. Best loss: {best_loss:.6f} "
        f"(iter {best_iter})"
    )

    final_T_mats: dict[str, np.ndarray] = {}
    final_scales: dict[str, float] = {}
    object_delta_stats: dict[str, dict[str, Any]] = {}
    for slug in context.obj_keys:
        od = context.objects[slug]
        delta_T = compose_T_sequence(
            delta_rotvecs[slug].detach(),
            delta_trans[slug].detach(),
        )
        T_out = torch.matmul(od.tracked_poses_torch, delta_T).cpu().numpy()
        final_T_mats[slug] = T_out
        final_scales[slug] = float(
            torch.exp(
                bounded_log_scale_delta(
                    raw_scale_deltas[slug].detach(),
                    args.max_log_scale_delta,
                )
            ).item()
        )

        delta_log_scale = float(
            bounded_log_scale_delta(
                raw_scale_deltas[slug].detach(),
                args.max_log_scale_delta,
            ).item()
        )
        object_delta_stats[slug] = {
            "slug": slug,
            "cotracker_seed_points": int(od.track_seed_count),
            "cotracker_valid_points": int(od.track_valid_count),
            "pnp_frames_ok": int(
                sum(1 for row in od.pnp_init_info[1:] if row.get("pnp_ok", False))
            ),
            "max_delta_rot_deg": float(
                delta_rotvecs[slug].detach().norm(dim=-1).max().item()
                * 180.0
                / math.pi
            ),
            "mean_delta_rot_deg": float(
                delta_rotvecs[slug].detach().norm(dim=-1).mean().item()
                * 180.0
                / math.pi
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

    return OptimizationResult(
        best_loss=best_loss,
        best_iter=best_iter,
        optimisation_time_s=optimisation_time_s,
        early_stop_triggered=early_stop_triggered,
        iter_rows=iter_rows,
        frame_rows=frame_rows,
        final_loss_summary_row=final_loss_summary_row,
        final_diagnostic=final_diagnostic,
        final_T_mats=final_T_mats,
        final_scales=final_scales,
        final_human_verts_np_by_slug={
            slug: context.humans[slug].base_verts.detach().cpu().numpy().copy()
            for slug in context.human_keys
        },
        object_delta_stats=object_delta_stats,
    )
