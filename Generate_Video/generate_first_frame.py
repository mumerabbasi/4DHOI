from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxPipeline

# Always-on framing prompt additions
FIRST_FRAME_FRAMING_SUFFIX = (
    "Wide shot. The human and the object are both fully visible and placed "
    "far enough from the camera to leave generous space around them. "
    "Keep them comfortably inside the frame so they can remain in view for "
    "the entire video without any camera movement."
)


def resolve_pag_path(script_dir: Path, video_name: str, raw_pag_dir: str | None) -> Path:
    pag_dir = (
        Path(raw_pag_dir).resolve()
        if raw_pag_dir
        else script_dir.parent / "Generate_PAG" / "output" / video_name
    )
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return candidates[0]


def load_pag_prompt(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pag = payload.get("pag", payload)
    return pag["interaction"]


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Sample FLUX.1-dev first-frame images from a PAG directory.",
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--pag-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pag_path = resolve_pag_path(script_dir, args.video_name, args.pag_dir)
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.video_name / "first_frames"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    prompt = load_pag_prompt(pag_path).rstrip() + "\n\n" + FIRST_FRAME_FRAMING_SUFFIX

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=dtype,
    ).to(args.device)

    generators = [
        torch.Generator(device=args.device).manual_seed(args.seed + i)
        for i in range(args.n)
    ]

    images = pipe(
        prompt=prompt,
        prompt_2=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=50,
        guidance_scale=3.5,
        num_images_per_prompt=args.n,
        generator=generators,
        max_sequence_length=512,
    ).images

    for i, img in enumerate(images):
        img.save(outdir / f"frame_{i:02d}.png")

    print(f"Loaded PAG: {pag_path}")
    print(f"Video name: {args.video_name}")
    print(f"Saved {len(images)} images to: {outdir}")


if __name__ == "__main__":
    main()
