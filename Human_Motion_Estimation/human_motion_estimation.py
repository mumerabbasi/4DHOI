from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="../Generate_Video/videos/frame_02_video_THUDM_CogVideoX_5b_I2V.mp4"
    )
    parser.add_argument("--outdir", default="./output_human_motion")
    parser.add_argument(
        "--gvhmr_path",
        default="../../GVHMR",
        help="Path to the cloned GVHMR repo."
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    outdir = Path(args.outdir).resolve()
    gvhmr_path = Path(args.gvhmr_path).resolve()

    # GVHMR's demo script location
    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    print(f"Running GVHMR on {video_path.name}")

    # Construct the command to run the GVHMR inference.
    # We use subprocess to run it within the GVHMR environment context.
    cmd = [
        sys.executable,  # Use the current python interpreter
        str(demo_script),
        f"--video={str(video_path)}",
        f"--output_root={str(outdir)}",
        # Optional: Skip visual odometry if camera is static.
        # HOI-PAGE videos are often static/slow pan.
        # Remove "-s" if you want camera motion estimation enabled.
        "-s",
    ]

    # Run GVHMR. Set cwd to gvhmr_path to find relative configs/checkpoints.
    subprocess.run(cmd, cwd=str(gvhmr_path), check=True)
    print(f"\nSuccess! Motion data saved to: {outdir}")
    print("Look for the .npz or .pkl files containing SMPL parameters.")


if __name__ == "__main__":
    main()
