"""Optimization loop for human-object mesh refinement."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

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
    print(f"  tracking_weight={args.tracking_weight}")
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


def run_joint_optimization(
    context: ProblemContext,
    args: argparse.Namespace,
) -> OptimizationResult:
    device = context.device
    num_frames = context.num_frames
    _print_optimization_header(context, args)

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
