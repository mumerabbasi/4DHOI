"""Entry point for human-object mesh refinement."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Human-object mesh refinement with Original_Code shared losses."
    )

    p.add_argument("--interaction_name", type=str, default="interaction_01")
    p.add_argument(
        "--aligned_mesh_dir",
        type=str,
        default=None,
        help=(
            "09_Align_Meshes/output/<interaction> containing meshes/transforms.json "
            "(auto-resolved)."
        ),
    )
    p.add_argument(
        "--human_motion_dir",
        type=str,
        default=None,
        help=(
            "06_Estimate_Human_Motion/output/<interaction> containing "
            "humans/<person>/hmr4d_results.pt (auto-resolved)."
        ),
    )
    p.add_argument(
        "--smpl_folder",
        type=str,
        default="../../GVHMR/inputs/checkpoints/body_models/",
        help="SMPL-X body model folder used to reconstruct module-06 GVHMR output.",
    )
    p.add_argument(
        "--smplx_batch_size",
        type=int,
        default=32,
        help="Batch size for reconstructing SMPL-X vertices from module-06 params.",
    )
    p.add_argument(
        "--tracked_object_dir",
        type=str,
        default=None,
        help="10_Track_Object_Mesh/output/<interaction> (auto-resolved).",
    )
    p.add_argument(
        "--segment_object_dir",
        type=str,
        default=None,
        help="08_Segment_Object_Mesh/output/<interaction> (auto-resolved).",
    )
    p.add_argument(
        "--segment_video_dir",
        type=str,
        default=None,
        help="03_Segment_Video/output/<interaction> for frame images (auto-resolved).",
    )
    p.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="PAG JSON. Auto-resolved from 01_Generate_PAG/output/<interaction>.",
    )
    p.add_argument(
        "--smpl_seg_json",
        type=str,
        default=None,
        help=(
            "SMPL-X contact segmentation JSON. Auto-resolved from "
            "06_Estimate_Human_Motion/assets/smplx_vert_segmentation.json."
        ),
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Root output directory.",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device to use, e.g. cuda:0, cuda:1, or cpu.",
    )
    p.add_argument(
        "--sdf_resolution",
        type=int,
        default=128,
        help="Voxel resolution for SDF grids.",
    )

    p.add_argument("--adam_iters", type=int, default=10000)
    p.add_argument("--adam_lr", type=float, default=1e-4)
    p.add_argument(
        "--early_stop_start",
        type=int,
        default=15000,
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
        "--optimize_arm_pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Optimise PAG-selected SMPL-X arm-chain pose corrections.",
    )
    p.add_argument(
        "--optimize_object_scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimise one global uniform scale per object.",
    )
    p.add_argument(
        "--max_object_scale_delta",
        type=float,
        default=0.1,
        help="Maximum fractional object scale change around 1.0.",
    )

    p.add_argument(
        "--tracking_weight",
        type=float,
        default=20.0,
        help="Fixed weight for the custom tracked-pose anchor term.",
    )

    p.add_argument("--object_cd2d_weight_start", type=float, default=0)
    p.add_argument("--object_cd2d_weight_end", type=float, default=0)
    p.add_argument("--object_part_cd2d_weight_start", type=float, default=0)
    p.add_argument("--object_part_cd2d_weight_end", type=float, default=0)
    p.add_argument("--object_smooth_trans_weight_start", type=float, default=1e3)
    p.add_argument("--object_smooth_trans_weight_end", type=float, default=1e3)
    p.add_argument("--object_smooth_rot_weight_start", type=float, default=1e3)
    p.add_argument("--object_smooth_rot_weight_end", type=float, default=1e3)
    p.add_argument(
        "--static_object_smooth_multiplier",
        type=float,
        default=500.0,
        help=(
            "Extra multiplier applied to smoothness terms for PAG-static "
            "objects (non-translational/non-rotational)."
        ),
    )
    p.add_argument("--human_pose_weight_start", type=float, default=50)
    p.add_argument("--human_pose_weight_end", type=float, default=100)
    p.add_argument("--human_pose_smooth_weight_start", type=float, default=100)
    p.add_argument("--human_pose_smooth_weight_end", type=float, default=100)
    p.add_argument("--object_scale_weight_start", type=float, default=0)
    p.add_argument("--object_scale_weight_end", type=float, default=0)
    p.add_argument("--object_intersect_weight_start", type=float, default=0)
    p.add_argument("--object_intersect_weight_end", type=float, default=0)
    p.add_argument("--intersect_weight_start", type=float, default=20)
    p.add_argument("--intersect_weight_end", type=float, default=20)
    p.add_argument("--nocontact_weight_start", type=float, default=1e3)
    p.add_argument("--nocontact_weight_end", type=float, default=1e3)
    p.add_argument("--contact_drift_weight_start", type=float, default=1e4)
    p.add_argument("--contact_drift_weight_end", type=float, default=1e4)

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

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {args.device}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(device)

    script_dir = Path(__file__).resolve().parent
    context = load_problem_context(args, script_dir, device)
    result = run_joint_optimization(context, args)
    save_run_outputs(context, result, args)


if __name__ == "__main__":
    main()
