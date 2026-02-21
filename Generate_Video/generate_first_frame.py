from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample FLUX.1-dev first-frame images from a PAG directory.",
    )
    parser.add_argument(
        "--pag-dir",
        default="../Generate_PAG/output/video_03",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    pag_dir = Path(args.pag_dir)
    pag_path = next(pag_dir.glob("*.json"))

    outdir = Path("./output") / pag_dir.name / "first_frames"
    outdir.mkdir(parents=True, exist_ok=True)

    pag = json.loads(pag_path.read_text(encoding="utf-8"))
    prompt = pag["interaction"]

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
    print(f"Saved {len(images)} images to: {outdir}")


if __name__ == "__main__":
    main()
