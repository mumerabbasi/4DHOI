from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="../Generate_Video/videos/video_02",
    )
    parser.add_argument("--outdir", default="./output")
    parser.add_argument(
        "--gvhmr_path",
        default="../../GVHMR",
        help="Path to the cloned GVHMR repo.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--f_mm",
        type=float,
        default=24,
        help="Focal length in millimeters passed to GVHMR (--f_mm).",
    )
    args = parser.parse_args()

    video_dir = Path(args.video).resolve()
    outdir = Path(args.outdir).resolve()
    gvhmr_path = Path(args.gvhmr_path).resolve()

    video_files = [
        p
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ]
    if len(video_files) != 1:
        raise ValueError(
            f"--video must point to a directory containing exactly one video file, "
            f"but found {len(video_files)} video files in: {video_dir}"
        )
    video_path = video_files[0]

    # GVHMR's demo script location
    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    print(f"Running GVHMR on {video_path.name} (f_mm={args.f_mm})")

    # Construct the command to run the GVHMR inference.
    cmd = [
        sys.executable,
        str(demo_script),
        f"--video={video_path}",
        f"--output_root={outdir}",
        # Skip visual odometry (static camera for HOI-PAGE videos)
        "-s",
    ]
    if args.f_mm is not None:
        cmd.append(f"--f_mm={args.f_mm:g}")

    # Run GVHMR. Set cwd to gvhmr_path to find relative configs/checkpoints.
    subprocess.run(cmd, cwd=str(gvhmr_path), check=True)

    # GVHMR writes outputs to <outdir>/<video_stem>. Rename to <outdir>/<video_dir>.
    src_dir = outdir / video_path.stem
    dst_dir = outdir / video_dir.name
    if src_dir.exists() and src_dir != dst_dir:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        src_dir.rename(dst_dir)

    print(f"\nSuccess! Motion data saved to: {dst_dir}")
    print("Look for the .npz or .pkl files containing SMPL parameters.")


if __name__ == "__main__":
    main()
