from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxFillPipeline

DEFAULT_VIDEO_NAME = "video_01"
DEFAULT_PROMPT = "a person pulling back the chair"
DEFAULT_MODEL = "black-forest-labs/FLUX.1-Fill-dev"
DEFAULT_IMAGE_PATH = Path(__file__).resolve().parent / "input" / "DSC08445.JPG"


def resolve_video_dir(script_dir: Path, video_name: str) -> Path:
    return script_dir / "output" / video_name


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug or "scene_fill")[:limit]


def load_scene_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_image(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Use FLUX.1-Fill-dev to add a human into an existing scene image.",
    )
    parser.add_argument("--video_name", default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--mask", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    args = parser.parse_args()

    video_dir = resolve_video_dir(script_dir, args.video_name)
    image_path = Path(args.image).expanduser().resolve()
    mask_path = (
        Path(args.mask).expanduser().resolve()
        if args.mask
        else (video_dir / "human_masks" / "human_mask.png").resolve()
    )

    outdir = Path(args.outdir).resolve() if args.outdir else video_dir
    outdir.mkdir(parents=True, exist_ok=True)

    scene_image = load_scene_image(image_path)
    prompt = DEFAULT_PROMPT.strip()
    mask_image = load_mask_image(mask_path)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = FluxFillPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generators = [torch.Generator(device="cpu").manual_seed(args.seed + i) for i in range(args.n)]
    edited_images = pipe(
        prompt=prompt,
        prompt_2=prompt,
        image=scene_image,
        mask_image=mask_image,
        height=scene_image.height,
        width=scene_image.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        num_images_per_prompt=args.n,
        max_sequence_length=args.max_sequence_length,
        generator=generators,
    ).images

    run_name = f"{image_path.stem}_{slugify(prompt)}_seed{args.seed}"
    input_path = outdir / f"{run_name}_input.png"
    output_paths = [outdir / f"{run_name}_frame_{i:02d}.png" for i in range(len(edited_images))]

    scene_image.save(input_path)
    for image, output_path in zip(edited_images, output_paths):
        image.save(output_path)

    print(f"Saved input copy: {input_path}")
    print(f"Saved {len(output_paths)} generated frame(s) to: {outdir}")


if __name__ == "__main__":
    main()
