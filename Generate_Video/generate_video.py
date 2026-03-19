from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers.utils import export_to_video

# Always-on camera lock prompt additions
CAMERA_LOCK_SUFFIX = (
    "Static locked-off camera on a tripod. "
    "No camera movement. No pan, tilt, zoom, dolly, orbit, or roll. "
    "No handheld shake. Fixed framing and fixed perspective. "
)

CAMERA_LOCK_NEGATIVE = (
    "camera movement, moving camera, pan, panning, tilt, tilting, zoom, zooming, "
    "dolly, dolly-in, dolly-out, tracking shot, orbit, rotation, roll, "
    "handheld, shaky cam, camera shake, jitter, parallax, perspective shift"
)


def model_suffix(model: str) -> str:
    return (
        model.replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def load_pag_prompt(path: Path) -> str:
    pag = json.loads(path.read_text(encoding="utf-8"))
    return pag["interaction"]


def resolve_pag_path(script_dir: Path, video_name: str, raw_pag: str | None) -> Path:
    if raw_pag:
        return Path(raw_pag).resolve()
    return next((script_dir.parent / "Generate_PAG" / "output" / video_name).glob("*.json"))


def resolve_frame_path(
    video_dir: Path,
    raw_frame: str | None,
    raw_selection_json: str | None,
) -> tuple[Path, Path | None]:
    if raw_frame:
        return Path(raw_frame).resolve(), None

    selection_path = (
        Path(raw_selection_json).resolve()
        if raw_selection_json
        else video_dir / "selected_first_frame.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_value = selection.get("selected_frame_path") or selection.get("selected_frame")
    if not selected_value:
        raise KeyError(f"No selected frame found in: {selection_path}")

    frame_path = Path(selected_value)
    if not frame_path.is_absolute():
        frame_path = (selection_path.parent / frame_path).resolve()
    return frame_path, selection_path


def load_and_prepare_image(path: Path, width: int, height: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size

    target_ratio = width / height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    return img.resize((width, height), resample=Image.BICUBIC)


def generate_video_wan(
    pipe,
    prompt: str,
    negative_prompt: str,
    image: Image.Image,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guidance_scale: float,
    generator: torch.Generator,
) -> list:
    """Generate video using Wan pipeline."""
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--frame", default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--pag", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--model", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    video_dir = script_dir / "output" / args.video_name
    frame_path, selection_path = resolve_frame_path(
        video_dir,
        args.frame,
        args.selection_json,
    )
    pag_path = resolve_pag_path(script_dir, args.video_name, args.pag)
    outdir = Path(args.outdir).resolve() if args.outdir else video_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Video name: {args.video_name}")
    if selection_path is not None:
        print(f"Frame selection JSON: {selection_path}")
    print(f"Input frame: {frame_path}")
    print(f"PAG file: {pag_path}")
    print(f"Output directory: {outdir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frames: {args.num_frames}")
    print(f"Steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"FPS: {args.fps}")

    prompt = load_pag_prompt(pag_path)

    # Always-on camera lock
    prompt = prompt.rstrip() + "\n\n" + CAMERA_LOCK_SUFFIX
    negative_prompt = CAMERA_LOCK_NEGATIVE

    print("Preparing input frame...")
    image = load_and_prepare_image(frame_path, args.width, args.height)

    print("Loading Wan image-to-video pipeline...")
    from diffusers import WanImageToVideoPipeline

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    pipe.enable_model_cpu_offload(device=args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print("Generating video frames...")
    frames = generate_video_wan(
        pipe,
        prompt,
        negative_prompt,
        image,
        args.width,
        args.height,
        args.num_frames,
        args.steps,
        args.guidance_scale,
        generator,
    )

    out_path = outdir / f"{frame_path.stem}_video_{model_suffix(args.model)}.mp4"
    print("Exporting MP4...")
    export_to_video(frames, out_path.as_posix(), fps=args.fps)
    print(f"Saved video: {out_path}")


if __name__ == "__main__":
    main()
