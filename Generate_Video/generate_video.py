from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pag", default="../Generate_PAG/output_pag_deepseek_r1_32b.json")
    parser.add_argument("--frame", default="./first_frames/frame_00.png")
    parser.add_argument("--outdir", default="./videos")
    parser.add_argument("--model", default="THUDM/CogVideoX-5b-I2V")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frame_path = Path(args.frame)  # NEW
    prompt = load_pag_prompt(Path(args.pag))
    image = load_and_prepare_image(frame_path, args.width, args.height)  # CHANGED

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    ).to(args.device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    frames = pipe(
        prompt=prompt,
        image=image,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]

    out_path = outdir / f"{frame_path.stem}_video_{model_suffix(args.model)}.mp4"  # CHANGED
    export_to_video(frames, out_path.as_posix(), fps=args.fps)
    print(out_path)


if __name__ == "__main__":
    main()
