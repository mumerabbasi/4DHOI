"""First-frame multi-seed object meshes with shared canonical mesh.

Canonical mesh is generated from the first seed only.
Each seed contributes only pose, which is applied to that shared canonical mesh.
"""

import argparse

from generate_objects_meshes_first_frame_multiseed_common import (
    load_sam3d_pipeline,
    resolve_path,
    run_first_frame_multiseed,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate N posed meshes from only the first frame using multiple seeds. "
            "Canonical mesh is shared from first seed."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../Segment_Video/output/video_01",
        help="Segment_Video output dir with _frames/ and objects/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_first_frame_multiseed_shared_canonical",
        help="Output root (<output_dir>/<video>/<first_frame>/...).",
    )
    parser.add_argument("--num_seeds", type=int, default=8, help="Number of seeds to run.")
    parser.add_argument("--seed_start", type=int, default=42, help="Starting seed.")
    parser.add_argument("--seed_stride", type=int, default=10, help="Step between seeds.")
    parser.add_argument(
        "--focal_length",
        type=float,
        default=None,
        help="Focal length in mm for projection (default: auto from first frame).",
    )
    parser.add_argument(
        "--f_scale",
        type=float,
        default=0.9,
        help="Scale for auto-estimated intrinsics in auto focal mode.",
    )
    parser.add_argument(
        "--overlay_quality",
        type=str,
        default="quality",
        choices=["quality", "legacy"],
        help="Overlay renderer preset.",
    )
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    sam3d = load_sam3d_pipeline()

    run_first_frame_multiseed(
        mode="shared_canonical",
        input_dir=input_dir,
        sam3d=sam3d,
        mesh_output_root=output_dir,
        num_seeds=int(args.num_seeds),
        seed_start=int(args.seed_start),
        seed_stride=int(args.seed_stride),
        focal_length_mm=args.focal_length,
        f_scale=float(args.f_scale),
        overlay_quality=args.overlay_quality,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
