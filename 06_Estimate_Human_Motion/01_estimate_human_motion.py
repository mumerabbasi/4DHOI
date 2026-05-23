from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from incam_stabilization import stabilize_result_file


F_MM_AUTO = "auto"


@dataclass(frozen=True)
class HumanTrackSpec:
    name: str
    masks_dir: Path | None


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
        / "04_Generate_Object_Mesh"
        / "output"
        / video_dir_name
        / "camera_intrinsics.json"
    )
    with intrinsics_path.open("r", encoding="utf-8") as f:
        return int(round(float(json.load(f)["blender_recommendation"]["lens_mm"])))


def build_default_paths(interaction_name: str) -> tuple[Path, Path]:
    """Build default video/input paths for a given interaction name."""
    script_dir = Path(__file__).parent.resolve()
    project_dir = script_dir.parent
    video_dir = project_dir / "02_Generate_Video" / "output" / interaction_name
    output_dir = script_dir / "output"
    return video_dir, output_dir


def discover_video_file(video_dir: Path) -> Path:
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
    return video_files[0]


def discover_segment_humans(interaction_name: str) -> list[HumanTrackSpec]:
    segment_humans_dir = (
        Path(__file__).resolve().parents[1]
        / "03_Segment_Video"
        / "output"
        / interaction_name
        / "humans"
    )
    if not segment_humans_dir.exists():
        return []

    human_specs: list[HumanTrackSpec] = []
    for human_dir in sorted(segment_humans_dir.iterdir()):
        masks_dir = human_dir / "masks"
        if human_dir.is_dir() and masks_dir.is_dir():
            human_specs.append(HumanTrackSpec(name=human_dir.name, masks_dir=masks_dir))
    return human_specs


def build_subprocess_env(gvhmr_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{gvhmr_path}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(gvhmr_path)
    return env


def _mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    margin = int(round(max(box_w, box_h) * max(float(margin_ratio), 0.0)))
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width, x2 + margin),
        min(height, y2 + margin),
    )


def _frame_bbox_from_mask(
    human_spec: HumanTrackSpec,
    frame_idx: int,
    width: int,
    height: int,
    last_bbox: tuple[int, int, int, int] | None,
    bbox_margin_ratio: float,
) -> tuple[int, int, int, int]:
    full_frame = (0, 0, width, height)

    mask_path = human_spec.masks_dir / f"frame_{frame_idx:04d}.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        bbox = last_bbox or full_frame
    else:
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        bbox = _mask_to_bbox(mask) or last_bbox or full_frame

    return _expand_bbox(bbox, width, height, bbox_margin_ratio)


def create_bbox_isolated_human_video(
    source_video_path: Path,
    human_spec: HumanTrackSpec,
    output_video_path: Path,
    bbox_margin_ratio: float,
) -> None:
    """Create a full-frame video with black pixels outside the expanded target bbox."""
    if human_spec.masks_dir is None:
        raise ValueError(f"No masks available for human track: {human_spec.name}")

    fps = probe_video_fps(source_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {source_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Could not read video size for: {source_video_path}")

    ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video_path),
    ]
    writer = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert writer.stdin is not None

    frame_idx = 0
    last_bbox: tuple[int, int, int, int] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            bbox = _frame_bbox_from_mask(
                human_spec=human_spec,
                frame_idx=frame_idx,
                width=width,
                height=height,
                last_bbox=last_bbox,
                bbox_margin_ratio=bbox_margin_ratio,
            )
            last_bbox = bbox
            x1, y1, x2, y2 = bbox
            composed = np.zeros_like(frame)
            composed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            writer.stdin.write(composed.tobytes())
            frame_idx += 1
    finally:
        cap.release()
        writer.stdin.close()

    stderr = writer.stderr.read() if writer.stderr is not None else b""
    ret = writer.wait()
    if ret != 0:
        msg = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed while writing {output_video_path}:\n{msg}")


def create_passthrough_human_video(
    source_video_path: Path,
    output_video_path: Path,
) -> None:
    """Copy the source video to a per-human filename without altering pixels."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    if output_video_path.exists():
        output_video_path.unlink()
    shutil.copy2(source_video_path, output_video_path)


def probe_video_fps(video_path: Path) -> float:
    """Read the source video fps using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    raw_fps = result.stdout.strip()
    if not raw_fps:
        raise RuntimeError(f"Could not determine fps for video: {video_path}")
    return float(Fraction(raw_fps))


def finalize_gvhmr_output(
    src_dir: Path,
    dst_dir: Path,
    allow_partial: bool,
) -> Path:
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected GVHMR output directory not found: {src_dir}")

    results_path = src_dir / "hmr4d_results.pt"
    if not results_path.exists() and not allow_partial:
        raise FileNotFoundError(f"Expected GVHMR result file not found: {results_path}")

    if src_dir != dst_dir:
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        src_dir.rename(dst_dir)
    return dst_dir


def build_gvhmr_demo_cmd(
    demo_script: Path,
    video_path: Path,
    output_root: Path,
    resolved_f_mm: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(demo_script),
        f"--video={video_path}",
        f"--output_root={output_root}",
        "-s",
    ]
    if resolved_f_mm is not None:
        cmd.append(f"--f_mm={resolved_f_mm}")
    return cmd


def run_gvhmr_inference(
    video_path: Path,
    output_root: Path,
    final_dir_name: str,
    gvhmr_path: Path,
    resolved_f_mm: int | None,
) -> Path:
    """Run GVHMR on one video and normalize the final output directory name."""
    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    print(f"Running GVHMR on {video_path.name} (f_mm={resolved_f_mm})")

    cmd = build_gvhmr_demo_cmd(demo_script, video_path, output_root, resolved_f_mm)
    env = build_subprocess_env(gvhmr_path)
    src_dir = output_root / video_path.stem
    dst_dir = output_root / final_dir_name

    allow_partial = False
    try:
        subprocess.run(cmd, cwd=str(gvhmr_path), check=True, env=env)
    except subprocess.CalledProcessError:
        # GVHMR may fail only at the final ffmpeg merge step after saving the result.
        if (src_dir / "hmr4d_results.pt").exists():
            print(
                f"Warning: GVHMR exited early for {final_dir_name}, "
                "but the main result file was saved. Continuing."
            )
            allow_partial = True
        else:
            raise

    return finalize_gvhmr_output(src_dir, dst_dir, allow_partial=allow_partial)


def rerender_stabilized_incam_outputs(
    video_path: Path,
    output_root: Path,
    result_dir: Path,
    gvhmr_path: Path,
    resolved_f_mm: int | None,
) -> None:
    """Regenerate incam videos after overwriting smpl_params_incam with stabilized motion."""
    incam_artifacts = sorted(result_dir.glob("*incam*.mp4"))
    for artifact in incam_artifacts:
        artifact.unlink()

    if not incam_artifacts:
        return

    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    cmd = build_gvhmr_demo_cmd(demo_script, video_path, output_root, resolved_f_mm)
    env = build_subprocess_env(gvhmr_path)
    subprocess.run(cmd, cwd=str(gvhmr_path), check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interaction_name",
        default="interaction_01",
        help="Interaction name used to build default paths for the other arguments.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Directory containing exactly one video file. "
        "Defaults to ../02_Generate_Video/output/<interaction_name>/.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Root output directory. Final outputs are written to <outdir>/<interaction_name>/.",
    )
    parser.add_argument(
        "--gvhmr_path",
        default="../../GVHMR",
        help="Path to the cloned GVHMR repo.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--stabilize_incam",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Project GVHMR's post-processed global trajectory back into the static "
            "camera frame, overwrite smpl_params_incam with the stabilized motion, "
            "preserve the raw camera-frame params under smpl_params_incam_raw, and "
            "rerender the incam videos."
        ),
    )
    parser.add_argument(
        "--bbox_margin_ratio",
        type=float,
        default=0.1,
        help="Extra margin added around each detected human bbox as a fraction of the larger bbox side.",
    )
    parser.add_argument(
        "--f_mm",
        type=parse_f_mm_arg,
        default=F_MM_AUTO,
        help=(
            "Focal length in full-frame millimeters passed to GVHMR (--f_mm). "
            "Default: read blender_recommendation.lens_mm from "
            "04_Generate_Object_Mesh/output/<video_dir>/camera_intrinsics.json and round to int. "
            "You can also pass --f_mm auto explicitly. "
            "Use --f_mm None to omit --f_mm and let GVHMR infer intrinsics. "
            "Use --f_mm <int> to override."
        ),
    )
    args = parser.parse_args()

    default_video_dir, default_outdir = build_default_paths(args.interaction_name)

    video_dir = Path(args.video).resolve() if args.video else default_video_dir
    outdir = Path(args.outdir).resolve() if args.outdir else default_outdir
    gvhmr_path = Path(args.gvhmr_path).resolve()

    video_path = discover_video_file(video_dir)
    resolved_f_mm = resolve_f_mm(args.interaction_name, args.f_mm)
    human_specs = discover_segment_humans(args.interaction_name)
    if human_specs:
        print(
            f"Found {len(human_specs)} segmented humans for {args.interaction_name}: "
            f"{[spec.name for spec in human_specs]}"
        )
    else:
        human_specs = [HumanTrackSpec(name="person_1", masks_dir=None)]
        print(
            f"No segmented humans found for {args.interaction_name}. "
            "Falling back to the original video as person_1."
        )

    multi_root = outdir / args.interaction_name / "humans"
    masked_videos_dir = outdir / args.interaction_name / "_masked_videos"
    final_dirs: list[Path] = []

    for human_spec in human_specs:
        if human_spec.masks_dir is not None:
            print(f"\nPreparing bbox-isolated video for {human_spec.name}...")
            masked_video_path = masked_videos_dir / f"{human_spec.name}.mp4"
            create_bbox_isolated_human_video(
                source_video_path=video_path,
                human_spec=human_spec,
                output_video_path=masked_video_path,
                bbox_margin_ratio=args.bbox_margin_ratio,
            )
            gvhmr_input_video = masked_video_path
        else:
            passthrough_video_path = masked_videos_dir / f"{human_spec.name}.mp4"
            create_passthrough_human_video(video_path, passthrough_video_path)
            gvhmr_input_video = passthrough_video_path

        final_output_dir = run_gvhmr_inference(
            video_path=gvhmr_input_video,
            output_root=multi_root,
            final_dir_name=human_spec.name,
            gvhmr_path=gvhmr_path,
            resolved_f_mm=resolved_f_mm,
        )
        if args.stabilize_incam:
            result_path = final_output_dir / "hmr4d_results.pt"
            print(f"Stabilizing camera-frame motion for {human_spec.name}...")
            stabilize_result_file(result_path, gvhmr_path)
            rerender_stabilized_incam_outputs(
                video_path=gvhmr_input_video,
                output_root=multi_root,
                result_dir=final_output_dir,
                gvhmr_path=gvhmr_path,
                resolved_f_mm=resolved_f_mm,
            )
        final_dirs.append(final_output_dir)

    print("\nSuccess! Motion data saved to:")
    for final_dir in final_dirs:
        print(f"  {final_dir}")
    print("Look for the .pt file containing SMPL parameters for each human.")


if __name__ == "__main__":
    main()
