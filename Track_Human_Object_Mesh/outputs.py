"""Output writing for joint human-object mesh refinement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from debug_utils import save_loss_plot_tree
from models import (
    FRAME_DIAGNOSTIC_TERM_KEYS,
    LOSS_TERM_KEYS,
    OptimizationResult,
    ProblemContext,
)
from utils import (
    close_ffmpeg,
    draw_overlay,
    ensure_dir,
    list_images,
    resolve_frames_dir,
    save_csv,
    start_ffmpeg_writer,
)


OVERLAY_FILL_ALPHA = 0.55
OVERLAY_CONTOUR_THICKNESS = 0
HUMAN_COLOR_BGR: tuple[int, int, int] = (255, 200, 100)


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


def _render_joint_overlay(
    frame_paths: list[Path],
    context: ProblemContext,
    result: OptimizationResult,
    fps: float,
    save_pngs: bool = False,
) -> None:
    num_frames = result.final_human_verts_np.shape[0]
    if not frame_paths or len(frame_paths) < num_frames:
        print("[WARN] Not enough frames for overlay rendering.")
        return

    overlays_dir = context.out_dir / "overlays"
    if save_pngs:
        ensure_dir(overlays_dir)

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        print("[WARN] Cannot read first frame for overlay.")
        return
    h, w = first_frame.shape[:2]

    writer = start_ffmpeg_writer(context.out_dir / "overlay.mp4", fps, (h, w))
    try:
        for t in range(num_frames):
            if t >= len(frame_paths):
                break
            frame = cv2.imread(str(frame_paths[t]))
            if frame is None:
                continue

            overlay = draw_overlay(
                frame_bgr=frame,
                verts_cv=result.final_human_verts_np[t],
                faces=context.human_faces,
                k=context.k,
                fill_alpha=OVERLAY_FILL_ALPHA * 0.6,
                contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                color_bgr=HUMAN_COLOR_BGR,
            )

            for slug in context.obj_keys:
                object_data = context.objects[slug]
                verts_t = (
                    (
                        object_data.template_verts.cpu().numpy()
                        * result.final_scales[slug]
                    )
                    @ result.final_T_mats[slug][t, :3, :3].T
                    + result.final_T_mats[slug][t, :3, 3][None, :]
                )
                overlay = draw_overlay(
                    frame_bgr=overlay,
                    verts_cv=verts_t.astype(np.float32),
                    faces=object_data.faces,
                    k=context.k,
                    fill_alpha=OVERLAY_FILL_ALPHA,
                    contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                    color_bgr=object_data.color_bgr,
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


def save_run_outputs(
    context: ProblemContext,
    result: OptimizationResult,
    args: argparse.Namespace,
) -> None:
    print("\nSaving outputs...")

    for slug in context.obj_keys:
        obj_dir = context.out_dir / slug
        ensure_dir(obj_dir)
        _save_transform_json(
            obj_dir / "transform_refined.json",
            result.final_scales[slug],
            result.final_T_mats[slug],
        )
        _save_mesh_sequence(
            context.objects[slug].template_verts.cpu().numpy(),
            context.objects[slug].faces,
            result.final_T_mats[slug],
            obj_dir / "meshes",
            global_scale=result.final_scales[slug],
        )
        with (obj_dir / "delta_stats.json").open("w", encoding="utf-8") as f:
            json.dump(result.object_delta_stats[slug], f, indent=2)
        stats = result.object_delta_stats[slug]
        print(
            f"  {slug}: max Δrot={stats['max_delta_rot_deg']:.2f}°, "
            f"max Δtrans={stats['max_delta_trans_m']:.4f}m, "
            f"scale={stats['global_scale']:.4f}"
        )

    human_out_dir = context.out_dir / "human" / "meshes"
    ensure_dir(human_out_dir)
    _save_human_mesh_sequence(
        result.final_human_verts_np,
        context.human_faces,
        human_out_dir,
    )
    with (context.out_dir / "human" / "delta_stats.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result.human_delta_stats, f, indent=2)

    debug_dir = context.out_dir / "debug"
    debug_csv_dir = debug_dir / "csv"
    debug_plot_dir = debug_dir / "plots"
    save_csv(debug_csv_dir / "iter_metrics.csv", result.iter_rows)
    save_csv(debug_csv_dir / "frame_loss_metrics.csv", result.frame_rows)
    save_csv(
        debug_csv_dir / "final_loss_summary.csv",
        [result.final_loss_summary_row],
    )
    save_loss_plot_tree(
        debug_plot_dir / "iter",
        result.iter_rows,
        x_key="iter",
        total_key="total",
        term_keys=LOSS_TERM_KEYS,
        x_label="Iteration",
        title_prefix="Iteration",
    )
    save_loss_plot_tree(
        debug_plot_dir / "frame",
        result.frame_rows,
        x_key="frame_idx",
        total_key="total_scaled_local",
        term_keys=FRAME_DIAGNOSTIC_TERM_KEYS,
        x_label="Frame",
        title_prefix="Per-Frame Local",
    )

    frames_dir = resolve_frames_dir(context.dirs)
    overlay_path = context.out_dir / "overlay.mp4"
    overlay_written = False
    if frames_dir is not None:
        frame_paths = list_images(frames_dir)
        if frame_paths:
            print("  Rendering overlay video...")
            _render_joint_overlay(
                frame_paths,
                context,
                result,
                args.fps,
                args.save_overlay_pngs,
            )
            overlay_written = True
            print(f"  → {overlay_path}")
    else:
        print("  [WARN] No frames directory found — skipping overlay.")

    summary = {
        "video_name": args.video_name,
        "status": "completed",
        "script": "track_human_object_mesh.py",
        "num_frames": context.num_frames,
        "num_objects": len(context.obj_keys),
        "num_edges": len(context.resolved_edges),
        "best_total_loss": result.best_loss,
        "optimisation_time_s": result.optimisation_time_s,
        "best_iter": result.best_iter,
        "inputs": {
            "aligned_mesh_dir": str(context.dirs["aligned"]),
            "tracked_object_dir": str(context.dirs["tracked"]),
            "segment_object_dir": str(context.dirs["seg_obj"]),
            "pag_file": str(context.pag_path),
            "smpl_seg_json": str(context.smpl_seg_path),
            "intrinsics_source": str(context.intr_path),
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
            "early_stop_triggered": result.early_stop_triggered,
        },
        "debug_outputs": {
            "frame_loss_semantics": "diagnostic_local",
            "csv": {
                "iter_metrics_csv": str(debug_csv_dir / "iter_metrics.csv"),
                "frame_loss_metrics_csv": str(
                    debug_csv_dir / "frame_loss_metrics.csv"
                ),
                "final_loss_summary_csv": str(
                    debug_csv_dir / "final_loss_summary.csv"
                ),
            },
            "plots": {
                "iter_dir": str(debug_plot_dir / "iter"),
                "frame_dir": str(debug_plot_dir / "frame"),
            },
            "iter_columns": (
                list(result.iter_rows[0].keys()) if result.iter_rows else []
            ),
            "frame_columns": (
                list(result.frame_rows[0].keys()) if result.frame_rows else []
            ),
            "final_summary_columns": list(
                result.final_loss_summary_row.keys()
            ),
            "global_only_terms": list(
                result.final_diagnostic.global_raw.keys()
            ),
        },
        "objects": {
            slug: {
                "name": context.objects[slug].name,
                "num_verts": int(
                    context.objects[slug].template_verts.shape[0]
                ),
                "num_faces": int(context.objects[slug].faces.shape[0]),
                "num_parts": len(context.objects[slug].part_vert_ids),
                "parts": list(context.objects[slug].part_vert_ids.keys()),
                "is_translational": (
                    context.objects[slug].state.is_translational
                ),
                "is_rotational": context.objects[slug].state.is_rotational,
                "final_scale": result.final_scales[slug],
            }
            for slug in context.obj_keys
        },
        "human": {
            "num_verts": int(context.human_data.base_verts.shape[1]),
            "num_faces": int(context.human_faces.shape[0]),
            "optimize_human": bool(args.optimize_human),
            "delta_stats": result.human_delta_stats,
        },
        "edges": [
            {
                "node_a": (
                    context.pag.edges[i].node_a
                    if i < len(context.pag.edges)
                    else "?"
                ),
                "node_b": (
                    context.pag.edges[i].node_b
                    if i < len(context.pag.edges)
                    else "?"
                ),
                "is_continuous": edge.is_continuous,
                "is_rel_static": edge.is_rel_static,
            }
            for i, edge in enumerate(context.resolved_edges)
        ],
        "conventions": {
            "coordinate_system": "OpenCV (X-right, Y-down, Z-forward)",
            "T_4x4": (
                "rigid component only; true object transform also uses "
                "global_scale"
            ),
        },
    }
    with (context.out_dir / "run_summary.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! Output: {context.out_dir}")
    print(f"  Summary:  {context.out_dir / 'run_summary.json'}")
    print(f"  Debug:    {debug_dir}")
    if overlay_written:
        print(f"  Overlay:  {overlay_path}")
    else:
        print("  Overlay:  skipped")
    for slug in context.obj_keys:
        print(f"  {slug}:  transform_refined.json, meshes/")
    print("  human:   meshes/")
    print(f"{'=' * 60}")
