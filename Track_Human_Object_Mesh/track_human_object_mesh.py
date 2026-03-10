"""Entry point for joint human-object mesh refinement."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Joint human-object mesh refinement with PAG constraints."
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
    p.add_argument(
        "--sdf_resolution",
        type=int,
        default=128,
        help="Voxel resolution for SDF grids.",
    )

    p.add_argument("--adam_iters", type=int, default=5000)
    p.add_argument("--adam_lr", type=float, default=1e-3)
    p.add_argument(
        "--early_stop_start",
        type=int,
        default=500,
        help="Iteration at which to begin checking early stopping.",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=300,
        help=(
            "Consecutive no-improvement iterations before stopping. "
            "Set 0 to disable."
        ),
    )
    p.add_argument(
        "--early_stop_rel_improve",
        type=float,
        default=1e-4,
        help=(
            "Minimum relative best-loss improvement required to "
            "reset patience."
        ),
    )

    p.add_argument(
        "--optimize_human",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimise per-frame global human SE(3) corrections.",
    )
    p.add_argument(
        "--optimize_object_scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimise one global uniform scale per object.",
    )
    p.add_argument(
        "--max_log_scale_delta",
        type=float,
        default=0.22,
        help="Maximum absolute log-scale correction per object.",
    )

    p.add_argument(
        "--lambda_prior",
        type=float,
        default=20.0,
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
        default=12.0,
        help="Temporal smoothness weight.",
    )
    p.add_argument(
        "--lambda_human_prior",
        type=float,
        default=100.0,
        help="Human correction prior weight.",
    )
    p.add_argument(
        "--lambda_human_smooth",
        type=float,
        default=30.0,
        help="Human correction smoothness weight.",
    )
    p.add_argument(
        "--lambda_human_mask_2d",
        type=float,
        default=40.0,
        help="2D human silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_mask_2d",
        type=float,
        default=60.0,
        help="2D object silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_part_mask_2d",
        type=float,
        default=120.0,
        help="2D object part silhouette chamfer weight.",
    )
    p.add_argument(
        "--lambda_object_scale",
        type=float,
        default=30.0,
        help="Object global scale regularisation weight.",
    )

    p.add_argument(
        "--num_mask_points_2d",
        type=int,
        default=2048,
        help="Maximum sampled 2D mask points per frame.",
    )
    p.add_argument(
        "--num_object_surface_points",
        type=int,
        default=4096,
        help="Whole-object sampled surface points.",
    )
    p.add_argument(
        "--num_part_surface_points",
        type=int,
        default=2048,
        help="Per-part sampled surface points.",
    )
    p.add_argument(
        "--num_human_surface_points",
        type=int,
        default=4096,
        help="Sampled human surface points per frame.",
    )

    p.add_argument("--fps", type=float, default=6.0)
    p.add_argument("--save_overlay_pngs", action="store_true")
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from data_loading import load_problem_context
    from optimizer import run_joint_optimization
    from outputs import save_run_outputs

    script_dir = Path(__file__).resolve().parent
    device = torch.device(args.device)
    context = load_problem_context(args, script_dir, device)
    result = run_joint_optimization(context, args)
    save_run_outputs(context, result, args)


if __name__ == "__main__":
    main()
