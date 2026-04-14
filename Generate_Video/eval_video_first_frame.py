from __future__ import annotations

import argparse
from pathlib import Path

from first_frame_eval_common import run_video_first_frame_eval


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Extract the first frame from a generated video and run geometry "
            "evaluation plus a 3D target overlay."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument(
        "--video",
        default=None,
        help=(
            "Optional explicit mp4 path. Defaults to the only mp4 in "
            "output/video_xx."
        ),
    )
    parser.add_argument(
        "--generated-root",
        default=None,
        help="Defaults to Generate_Video/output/<video_name>.",
    )
    parser.add_argument(
        "--selection-json",
        default=None,
        help="Path to target_selection.json.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Defaults to Select_Target_Instance/input_prompts/<video_name>.",
    )
    parser.add_argument(
        "--scannet-root",
        default=None,
        help="Defaults to the Select_Target_Instance convention.",
    )
    args = parser.parse_args()

    result = run_video_first_frame_eval(
        script_dir=script_dir,
        video_name=args.video_name,
        video=args.video,
        generated_root=args.generated_root,
        selection_json=args.selection_json,
        input_dir=args.input_dir,
        scannet_root=args.scannet_root,
    )
    outputs = result["outputs"]

    print(f"Saved extracted first frame: {outputs['first_frame_path']}")
    print(f"Reused target mask: {outputs['target_mask_path']}")
    print(f"Reused camera JSON: {outputs['camera_json_path']}")
    print(f"Saved 3D overlay: {outputs['target_3d_overlay_path']}")
    print(f"Saved overview panel: {outputs['overview_path']}")
    print(f"Saved mask overlay: {outputs['mask_overlay_path']}")
    print(f"Saved object crop panel: {outputs['object_crop_path']}")
    print(f"Saved geometry report: {outputs['report_path']}")


if __name__ == "__main__":
    main()
