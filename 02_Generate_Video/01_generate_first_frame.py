from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

# Always-on framing prompt additions
FIRST_FRAME_FRAMING_SUFFIX = (
    "Wide shot. The human and the object are both fully visible and placed "
    "far enough from the camera to leave generous space around them. "
    "Keep them comfortably inside the frame so they can remain in view for "
    "the entire video without any camera movement."
)


def enable_cpu_offload(pipe: Flux2KleinPipeline, device: str) -> None:
    try:
        pipe.enable_model_cpu_offload(device=device)
    except TypeError:
        if device.startswith("cuda:"):
            pipe.enable_model_cpu_offload(gpu_id=int(device.split(":", 1)[1]))
        else:
            pipe.enable_model_cpu_offload()


def resolve_pag_path(script_dir: Path, interaction_name: str, raw_pag_dir: str | None) -> Path:
    pag_dir = (
        Path(raw_pag_dir).resolve()
        if raw_pag_dir
        else script_dir.parent / "01_Generate_PAG" / "output" / interaction_name
    )
    if not pag_dir.is_dir():
        pag_root = pag_dir.parent
        available = sorted(path.name for path in pag_root.glob("interaction_*") if path.is_dir())
        suggestion = difflib.get_close_matches(interaction_name, available, n=1)
        details = f" Available interactions: {', '.join(available)}." if available else ""
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        raise FileNotFoundError(
            f"PAG directory not found: {pag_dir}.{hint}{details}"
        )

    pag_files = sorted(pag_dir.glob("*.json"))
    if not pag_files:
        raise FileNotFoundError(f"No PAG JSON files found in: {pag_dir}")
    return pag_files[0]


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Sample FLUX.2 [klein] first-frame images from a PAG directory.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--pag-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.2-klein-9B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable model CPU offload and move the full FLUX.2 [klein] pipeline to --device.",
    )
    args = parser.parse_args()

    pag_path = resolve_pag_path(script_dir, args.interaction_name, args.pag_dir)
    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name / "first_frames"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    pag = json.loads(pag_path.read_text(encoding="utf-8"))
    prompt = pag["interaction"].rstrip() + "\n\n" + FIRST_FRAME_FRAMING_SUFFIX

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )

    if args.device.startswith("cuda") and not args.no_cpu_offload:
        enable_cpu_offload(pipe, args.device)
    else:
        pipe.to(args.device)

    generators = [
        torch.Generator(device=args.device).manual_seed(args.seed + i)
        for i in range(args.n)
    ]

    call_kwargs = {
        "prompt": prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "num_images_per_prompt": args.n,
        "generator": generators,
        "max_sequence_length": args.max_sequence_length,
    }

    images = pipe(
        **call_kwargs,
    ).images

    for i, img in enumerate(images):
        img.save(outdir / f"frame_{i:02d}.png")

    print(f"Model: {args.model_id}")
    print(f"Loaded PAG: {pag_path}")
    print(f"Interaction name: {args.interaction_name}")
    print(f"Saved {len(images)} images to: {outdir}")


if __name__ == "__main__":
    main()
