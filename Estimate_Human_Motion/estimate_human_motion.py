from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


F_MM_AUTO = "auto"


def parse_f_mm_arg(raw: str) -> int | None | str:
    value = raw.strip().lower()
    if value in {"none", "null"}:
        return None
    if value == F_MM_AUTO:
        return F_MM_AUTO
    return int(value)


def resolve_f_mm(video_dir_name: str, cli_f_mm: int | None | str) -> int | None:
    if cli_f_mm != F_MM_AUTO:
        return cli_f_mm

    intrinsics_path = (
        Path(__file__).resolve().parents[1]
        / "Generate_Object_Mesh"
        / "output"
        / video_dir_name
        / "camera_intrinsics.json"
    )
    with intrinsics_path.open("r", encoding="utf-8") as f:
        return int(round(float(json.load(f)["focal_length_mm_recommended"])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default="../Generate_Video/videos/video_01",
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
        type=parse_f_mm_arg,
        default=F_MM_AUTO,
        help=(
            "Focal length in full-frame millimeters passed to GVHMR (--f_mm). "
            "Default: read focal_length_mm_recommended from "
            "Generate_Object_Mesh/output/<video_dir>/camera_intrinsics.json and round to int. "
            "You can also pass --f_mm auto explicitly. "
            "Use --f_mm None to omit --f_mm and let GVHMR infer intrinsics. "
            "Use --f_mm <int> to override."
        ),
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
    resolved_f_mm = resolve_f_mm(video_dir.name, args.f_mm)

    # GVHMR's demo script location
    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    print(f"Running GVHMR on {video_path.name} (f_mm={resolved_f_mm})")

    # Construct the command to run the GVHMR inference.
    cmd = [
        sys.executable,
        str(demo_script),
        f"--video={video_path}",
        f"--output_root={outdir}",
        # Skip visual odometry (static camera for HOI-PAGE videos)
        "-s",
    ]
    if resolved_f_mm is not None:
        cmd.append(f"--f_mm={resolved_f_mm}")

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
