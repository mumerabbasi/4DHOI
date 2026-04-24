from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_pag_path(script_dir: Path, video_name: str, raw_pag: str | None) -> Path:
    if raw_pag:
        return Path(raw_pag).resolve()

    pag_dir = (script_dir.parent / "Generate_PAG" / "output" / video_name).resolve()
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return candidates[0]


def resolve_selection_path(script_dir: Path, video_name: str, raw_selection_json: str | None) -> Path:
    if raw_selection_json:
        return Path(raw_selection_json).resolve()
    return (script_dir.parent / "Select_Target_Instance" / "output" / video_name / "target_selection.json").resolve()


def resolve_scene_image_path(selection_root: Path) -> Path:
    path = (selection_root / "2d" / "scene_image.png").resolve()
    if not path.exists():
        raise FileNotFoundError(f"2d scene image not found: {path}")
    return path


def resolve_object_mask_path(
    selection_root: Path,
    selection_payload: dict[str, Any],
) -> Path:
    rel_path = str(selection_payload["target_selection_2d"]["mask_path"])

    path = (selection_root / rel_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"2d target mask path does not exist: {path}. rel_path={rel_path}"
        )
    return path


def load_pag_interaction(path: Path) -> str:
    payload = load_json(path)
    pag_block = payload.get("pag", payload)
    interaction = str(pag_block["interaction"]).strip()
    if not interaction:
        raise ValueError(f"PAG interaction is empty in: {path}")
    return interaction


def load_target_object(selection_payload: dict[str, Any]) -> str:
    target_object = str(selection_payload["target_selection"]["object"]).strip()
    if not target_object:
        raise ValueError("target_selection.object must be a non-empty string.")
    return target_object


def load_binary_mask(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(path).convert("L")
    if mask_img.size != expected_size:
        raise ValueError(
            f"Mask size mismatch for {path}: got {mask_img.size}, expected {expected_size}."
        )
    mask_array = np.asarray(mask_img, dtype=np.uint8)
    return mask_array > 0


def load_scene_preservation_instructions(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(interaction: str, scene_preservation_instructions: str) -> str:
    return f"{scene_preservation_instructions}\n\nInteraction: {interaction}"


def build_mask_conditioning_image(mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.bool_:
        raise TypeError(f"Expected boolean mask, got {mask.dtype}")
    gray = (mask.astype(np.uint8) * 255)
    return Image.fromarray(gray, mode="L").convert("RGB")


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    table: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in table:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return table[dtype_name]


def floor_to_multiple(value: int, divisor: int) -> int:
    return value - (value % divisor)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_system_prompt_path = (script_dir / "prompts" / "system_prompt_first_frame_generation.md").resolve()

    parser = argparse.ArgumentParser(
        description=(
            "Sample multiple first-frame candidates with Flux2 using direct conditioning "
            "from a scene image, a target-object mask image, and PAG interaction text."
        )
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--pag", default=None, help="Path to output_pag_*.json")
    parser.add_argument("--selection-json", default=None, help="Path to target_selection.json")
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--system-prompt",
        default=str(default_system_prompt_path),
        help="Path to system prompt markdown file.",
    )
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--model", default="diffusers/FLUX.2-dev-bnb-4bit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw Flux2 outputs at model resolution.",
    )
    args = parser.parse_args()

    selection_path = resolve_selection_path(script_dir, args.video_name, args.selection_json)
    selection_root = selection_path.parent
    selection_payload = load_json(selection_path)

    pag_path = resolve_pag_path(script_dir, args.video_name, args.pag)
    interaction = load_pag_interaction(pag_path)
    target_object = load_target_object(selection_payload)

    scene_image_path = resolve_scene_image_path(selection_root)
    object_mask_path = resolve_object_mask_path(
        selection_root=selection_root,
        selection_payload=selection_payload,
    )

    outdir = (
        Path(args.outdir).resolve()
        if args.outdir
        else (script_dir / "output" / args.video_name / "first_frames").resolve()
    )
    outdir.mkdir(parents=True, exist_ok=True)

    source_image = Image.open(scene_image_path).convert("RGB")
    width, height = source_image.size
    object_mask = load_binary_mask(object_mask_path, expected_size=(width, height))
    if not np.any(object_mask):
        raise ValueError("The target mask from target_selection_2d is empty.")
    mask_conditioning_image = build_mask_conditioning_image(object_mask)

    system_prompt_path = Path(args.system_prompt).resolve()
    scene_preservation_instructions = load_scene_preservation_instructions(system_prompt_path)
    prompt = build_prompt(
        interaction=interaction,
        scene_preservation_instructions=scene_preservation_instructions,
    )

    model_width = floor_to_multiple(width, 16)
    model_height = floor_to_multiple(height, 16)
    if model_width <= 0 or model_height <= 0:
        raise ValueError(
            f"Invalid model resolution derived from source size {(width, height)}: "
            f"{(model_width, model_height)}"
        )

    if (model_width, model_height) == (width, height):
        conditioning_image = source_image
        conditioning_mask_image = mask_conditioning_image
    else:
        conditioning_image = source_image.resize(
            (model_width, model_height),
            resample=Image.Resampling.BICUBIC,
        )
        conditioning_mask_image = mask_conditioning_image.resize(
            (model_width, model_height),
            resample=Image.Resampling.NEAREST,
        )
        print(
            "Adjusted Flux2 generation resolution to nearest multiple of 16: "
            f"source={(width, height)} -> model={(model_width, model_height)}"
        )

    print(f"Video name: {args.video_name}")
    print(f"Scene image: {scene_image_path}")
    print(f"Object mask: {object_mask_path}")
    print(f"System prompt: {system_prompt_path}")
    print(f"Selection JSON: {selection_path}")
    print(f"PAG file: {pag_path}")
    print(f"Output directory: {outdir}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Samples: {args.n}")
    print(f"Steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Model resolution: {model_width}x{model_height}")

    from diffusers import Flux2Pipeline

    torch_dtype = resolve_torch_dtype(args.torch_dtype)
    pipe = Flux2Pipeline.from_pretrained(args.model, torch_dtype=torch_dtype).to(args.device)

    for i in range(args.n):
        run_seed = args.seed + i
        generator = torch.Generator(device=args.device).manual_seed(run_seed)

        result = pipe(
            prompt=prompt,
            image=[conditioning_image, conditioning_mask_image],
            height=model_height,
            width=model_width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]

        final_image = result
        if result.size != (width, height):
            final_image = result.resize(
                (width, height),
                resample=Image.Resampling.BICUBIC,
            )

        frame_name = f"frame_{i:02d}.png"
        final_image.save(outdir / frame_name)

        if args.save_raw:
            result.save(outdir / f"raw_{frame_name}")

        print(f"Saved frame: {outdir / frame_name}")

    Image.fromarray((object_mask.astype(np.uint8) * 255), mode="L").save(outdir / "object_mask_used.png")
    conditioning_mask_image.save(outdir / "conditioning_mask_image.png")
    (outdir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    run_metadata = {
        "video_name": args.video_name,
        "scene_image": str(scene_image_path),
        "object_mask": str(object_mask_path),
        "system_prompt": str(system_prompt_path),
        "selection_json": str(selection_path),
        "pag_json": str(pag_path),
        "target_object": target_object,
        "n": int(args.n),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "guidance_scale": float(args.guidance_scale),
        "model": args.model,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "mask_source": "target_selection_2d.mask_path",
        "conditioning_inputs": ["scene_image", "target_mask_binary"],
    }
    (outdir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved run metadata, prompt, and masks.")


if __name__ == "__main__":
    main()
