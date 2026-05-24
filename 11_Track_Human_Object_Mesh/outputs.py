"""Output writing for human-object mesh refinement."""

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
HUMAN_COLORS_BGR: list[tuple[int, int, int]] = [
    (255, 200, 100),
    (100, 220, 255),
    (180, 255, 140),
    (255, 160, 220),
]


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


def _build_vertex_color_visual(
    vertex_colors: np.ndarray | None,
    num_verts: int,
) -> trimesh.visual.ColorVisuals | None:
    if vertex_colors is None:
        return None
    colors = np.asarray(vertex_colors)
    if colors.shape[0] != num_verts:
        return None
    if colors.ndim != 2 or colors.shape[1] not in (3, 4):
        return None
    if np.issubdtype(colors.dtype, np.floating):
        max_val = float(np.max(colors)) if colors.size else 0.0
        if max_val <= 1.0:
            colors = np.clip(colors * 255.0, 0.0, 255.0)
        colors = np.rint(colors).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8, copy=False)
    return trimesh.visual.ColorVisuals(vertex_colors=colors)


def _save_mesh_sequence(
    verts_template: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray | None,
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
        visual=_build_vertex_color_visual(vertex_colors, len(verts_template)),
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
    num_frames = context.num_frames
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

            overlay = frame
            for idx, human_slug in enumerate(context.human_keys):
                human_verts = result.final_human_verts_np_by_slug[human_slug]
                overlay = draw_overlay(
                    frame_bgr=overlay,
                    verts_cv=human_verts[t],
                    faces=context.humans[human_slug].faces,
                    k=context.k,
                    fill_alpha=OVERLAY_FILL_ALPHA * 0.6,
                    contour_thickness=OVERLAY_CONTOUR_THICKNESS,
                    color_bgr=HUMAN_COLORS_BGR[idx % len(HUMAN_COLORS_BGR)],
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


def _scheduled_weights_summary(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    return {
        "tracking": {"fixed": float(args.tracking_weight)},
        "object_cd2d": {
            "start": float(args.object_cd2d_weight_start),
            "end": float(args.object_cd2d_weight_end),
        },
        "object_part_cd2d": {
            "start": float(args.object_part_cd2d_weight_start),
            "end": float(args.object_part_cd2d_weight_end),
        },
        "object_smooth_trans": {
            "start": float(args.object_smooth_trans_weight_start),
            "end": float(args.object_smooth_trans_weight_end),
        },
        "object_smooth_rot": {
            "start": float(args.object_smooth_rot_weight_start),
            "end": float(args.object_smooth_rot_weight_end),
        },
        "object_scale": {
            "start": float(args.object_scale_weight_start),
            "end": float(args.object_scale_weight_end),
        },
        "intersect": {
            "start": float(args.intersect_weight_start),
            "end": float(args.intersect_weight_end),
        },
        "nocontact": {
            "start": float(args.nocontact_weight_start),
            "end": float(args.nocontact_weight_end),
        },
        "contact_drift": {
            "start": float(args.contact_drift_weight_start),
            "end": float(args.contact_drift_weight_end),
        },
    }


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
            context.objects[slug].vertex_colors,
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

    human_stats_by_slug: dict[str, dict[str, object]] = {}
    for human_slug in context.human_keys:
        human_dir = context.out_dir / human_slug
        human_mesh_dir = human_dir / "meshes"
        ensure_dir(human_mesh_dir)
        _save_human_mesh_sequence(
            result.final_human_verts_np_by_slug[human_slug],
            context.humans[human_slug].faces,
            human_mesh_dir,
        )
        human_stats = {
            "status": "fixed_copy",
            "source": "module_06_smplx_with_module_09_transform",
            "name": context.humans[human_slug].name,
            "slug": human_slug,
            "num_frames": int(result.final_human_verts_np_by_slug[human_slug].shape[0]),
            "num_verts": int(result.final_human_verts_np_by_slug[human_slug].shape[1]),
            "num_faces": int(context.humans[human_slug].faces.shape[0]),
        }
        with (human_dir / "fixed_input_stats.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(human_stats, f, indent=2)
        human_stats_by_slug[human_slug] = human_stats

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
        "interaction_name": args.interaction_name,
        "status": "completed",
        "script": "01_track_human_object_mesh.py",
        "num_frames": context.num_frames,
        "num_humans": len(context.human_keys),
        "num_objects": len(context.obj_keys),
        "num_edges": len(context.interaction_edges),
        "best_total_loss": result.best_loss,
        "optimisation_time_s": result.optimisation_time_s,
        "best_iter": result.best_iter,
        "inputs": {
            "aligned_mesh_dir": str(context.dirs["aligned"]),
            "human_motion_dir": str(context.dirs["human_motion"]),
            "tracked_object_dir": str(context.dirs["tracked"]),
            "segment_object_dir": str(context.dirs["seg_obj"]),
            "pag_file": str(context.pag_path),
            "smpl_seg_json": str(context.smpl_seg_path),
            "intrinsics_source": str(context.intr_path),
        },
        "weights": _scheduled_weights_summary(args),
        "optimisation": {
            "adam_iters": args.adam_iters,
            "adam_lr": args.adam_lr,
            "sdf_resolution": args.sdf_resolution,
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
        "humans": human_stats_by_slug,
        "edges": [
            {
                "node_a": edge.node_a.raw_node,
                "node_b": edge.node_b.raw_node,
                "is_continuous": edge.is_continuous,
                "is_rel_static": edge.is_rel_static,
            }
            for edge in context.interaction_edges
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
    for human_slug in context.human_keys:
        print(f"  {human_slug}:  meshes/ (fixed copy)")
    print(f"{'=' * 60}")
