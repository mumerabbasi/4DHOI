from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxFillPipeline

DEFAULT_PROMPT = "a person going to lift the blue chair"
DEFAULT_MODEL = "black-forest-labs/FLUX.1-Fill-dev"


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def default_image_path() -> Path:
    return Path(__file__).resolve().parent / "input" / "DSC08445.JPG"


def default_mask_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Get_Input"
        / "output"
        / "video_01"
        / "target_mask.png"
    )


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (slug or "scene_fill")[:limit]


def load_scene_image(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    return image, image.size


def load_mask_image(path: Path, expected_size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != expected_size:
        mask = mask.resize(expected_size, resample=Image.NEAREST)
    return mask


def build_scene_prompt(action_prompt: str) -> str:
    action_prompt = action_prompt.strip()
    '''return (
        "Add exactly one realistic full-body adult person into this real indoor "
        "photograph. The person should naturally fit the camera perspective, rest "
        "on the floor, and interact plausibly with the nearby furniture. "
        f"The intended action is: {action_prompt}. "
        "Preserve the room layout, desk arrangement, chairs, lighting, and camera "
        "viewpoint. Keep all background details consistent outside the edited area. "
        "No extra people, no floating body parts, no cropped body, no large scene "
        "changes, no text, no watermark."
    )
    '''
    return (action_prompt)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Use FLUX.1-Fill-dev to add a human into an existing scene image.",
    )
    parser.add_argument("--image", default=str(default_image_path()))
    parser.add_argument("--mask", default=str(default_mask_path()))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    mask_path = Path(args.mask).expanduser().resolve()
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    outdir = Path(args.outdir).resolve() if args.outdir else script_dir / "output" / "scene_first_frame"
    outdir.mkdir(parents=True, exist_ok=True)

    scene_image, original_size = load_scene_image(image_path)
    prompt = build_scene_prompt(args.prompt)
    mask_image = load_mask_image(mask_path, scene_image.size)

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = FluxFillPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(args.device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    edited_image = pipe(
        prompt=prompt,
        prompt_2=prompt,
        image=scene_image,
        mask_image=mask_image,
        height=scene_image.height,
        width=scene_image.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        max_sequence_length=args.max_sequence_length,
        generator=generator,
    ).images[0]

    run_name = f"{image_path.stem}_{slugify(args.prompt)}_seed{args.seed}"
    output_path = outdir / f"{run_name}.png"
    input_path = outdir / f"{run_name}_input.png"
    summary_path = outdir / f"{run_name}_run_summary.json"

    scene_image.save(input_path)
    edited_image.save(output_path)

    summary = {
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "original_size": list(original_size),
        "generation_size": list(scene_image.size),
        "prompt": args.prompt,
        "final_prompt": prompt,
        "model": args.model,
        "device": args.device,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "max_sequence_length": args.max_sequence_length,
        "saved_input_path": str(input_path),
        "saved_output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input image: {image_path}")
    print(f"Mask image: {mask_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model}")
    print(f"Saved input copy: {input_path}")
    print(f"Saved generated frame: {output_path}")
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
