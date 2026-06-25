from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LEGACY_CONTACT_DIR = PROJECT_DIR / "03_Estimate_Contact"
if str(LEGACY_CONTACT_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_CONTACT_DIR))

from common import (  # noqa: E402
    adjusted_intrinsics_for_crop,
    bbox_from_mask,
    build_contact_prompt,
    build_pinhole_intrinsics,
    build_sam3_processor,
    choose_contact_palette,
    classify_nearest_color,
    contact_palette_from_spec,
    crop_array,
    erode_binary_mask,
    fit_bbox_to_image_aspect,
    floor_contact_human_parts,
    get_default_sam3_device,
    keep_largest_components,
    load_binary_mask,
    load_json,
    load_rgb,
    normalize_label,
    normalize_overlay_to_canvas,
    pad_bbox,
    read_api_key,
    resolve_scannet_root,
    resolve_transforms_path,
    run_sam3_text_prompt,
    save_binary_mask,
    save_contact_spec,
    save_contact_visualization,
    save_json,
    save_rgb,
    save_text,
    select_highest_confidence_mask,
    slugify,
    target_object_human_parts,
)


NEXT_ROUND_BASE_REMINDER = """Use the original Canvas image as the base again.
If a previous composite is provided, use it only as correction context.
Do not use the previous composite or generated image as the base.
Keep the canvas unchanged.
Only fix the specified colored contact masks."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Agentic human-in-the-loop contact mask estimation with manual "
            "ChatGPT image generation and Gemini VLM feedback."
        ),
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--inpainted-frame", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--vlm-prompt", default=None)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--api-key-file",
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--gemini-retries", type=int, default=3)
    parser.add_argument("--gemini-retry-sleep-s", type=float, default=8.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--padding-frac", type=float, default=0.25)
    parser.add_argument(
        "--disable-aspect-ratio-crop",
        action="store_true",
        help="Skip expanding the human crop to the source image aspect ratio.",
    )
    parser.add_argument("--human-sam3-prompt", default="person")
    parser.add_argument("--floor-sam3-prompt", default="floor")
    parser.add_argument("--floor-mask-erode-pixels", type=int, default=3)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=None)
    parser.add_argument("--sam3-device", default=get_default_sam3_device())
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
    parser.add_argument("--color-max-distance", type=float, default=90.0)
    parser.add_argument("--min-component-area", type=int, default=0)
    parser.add_argument("--keep-components", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-unaccepted", action="store_true")
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else SCRIPT_DIR / "output" / args.interaction_name
    )
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "sig.json"
    )
    scene_image_path = (
        Path(args.scene_image).resolve()
        if args.scene_image
        else sig_json_path.parent / "scene_image.png"
    )
    inpainted_frame_path = (
        Path(args.inpainted_frame).resolve()
        if args.inpainted_frame
        else PROJECT_DIR
        / "02_Generate_Human_Frame"
        / "output"
        / args.interaction_name
        / "inpainted_frame_resized.png"
    )
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / args.interaction_name
    )
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else SCRIPT_DIR / "system_prompt_estimate_contact_agentic.md"
    )
    vlm_prompt_path = (
        Path(args.vlm_prompt).resolve()
        if args.vlm_prompt
        else SCRIPT_DIR / "vlm_prompt_evaluate_contact.md"
    )
    return {
        "output_root": output_root,
        "assets_dir": output_root / "assets",
        "sig_json": sig_json_path,
        "scene_image": scene_image_path,
        "inpainted_frame": inpainted_frame_path,
        "input_dir": input_dir,
        "system_prompt": system_prompt_path,
        "vlm_prompt": vlm_prompt_path,
    }


def target_object_label(sig_payload: dict[str, Any]) -> str:
    labels: list[str] = []
    target_objects = sig_payload.get("target_objects")
    if isinstance(target_objects, list):
        for target_object in target_objects:
            if isinstance(target_object, dict):
                label = str(target_object.get("label", "")).strip()
                if label:
                    labels.append(label)
    if labels:
        return ", ".join(labels)
    target_object = sig_payload.get("target_object")
    if isinstance(target_object, dict):
        label = str(target_object.get("label", "")).strip()
        if label:
            return label
    return "target object"


def color_mapping_text(palette: list[dict[str, Any]]) -> str:
    lines = []
    for item in palette:
        rgb = item["rgb"]
        lines.append(
            f"- {item['label']}: {item['hex']} / "
            f"RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) ({item['color_name']})"
        )
    return "\n".join(lines)


def open_rgb_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def response_chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    parts = []
    for candidate in getattr(chunk, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "".join(parts)


def gemini_generate_json(
    api_key: str,
    model: str,
    user_prompt: str,
    image_paths: list[Path],
    temperature: float,
    seed: int,
    max_output_tokens: int,
    raw_response_path: Path,
) -> str:
    from google import genai
    from google.genai import types

    raw_response_path.parent.mkdir(parents=True, exist_ok=True)
    content_chunks: list[str] = []

    def write_accumulated() -> None:
        raw_response_path.write_text("".join(content_chunks), encoding="utf-8")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        seed=int(seed),
        maxOutputTokens=int(max_output_tokens),
        responseMimeType="application/json",
    )
    contents = [user_prompt] + [open_rgb_image(path) for path in image_paths]
    write_accumulated()
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        text = response_chunk_text(chunk)
        if text:
            content_chunks.append(text)
            write_accumulated()

    content = "".join(content_chunks)
    if not content.strip():
        raise RuntimeError(
            "Gemini response did not contain text content. Raw response path: "
            f"{raw_response_path}."
        )
    return content.strip()


def parse_json_response(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON must be an object.")
    return parsed


def render_vlm_prompt(
    template: str,
    target_label: str,
    palette: list[dict[str, Any]],
) -> str:
    return template.replace("{target_object}", target_label).replace(
        "{color_mapping}", color_mapping_text(palette)
    )


def evaluate_round(
    args: argparse.Namespace,
    api_key: str,
    vlm_prompt_template: str,
    target_label: str,
    palette: list[dict[str, Any]],
    reference_path: Path,
    canvas_path: Path,
    composite_path: Path,
    raw_response_path: Path,
) -> tuple[dict[str, Any], str]:
    prompt = render_vlm_prompt(vlm_prompt_template, target_label, palette)
    raw_response = gemini_generate_json(
        api_key=api_key,
        model=args.model,
        user_prompt=prompt,
        image_paths=[reference_path, canvas_path, composite_path],
        temperature=float(args.temperature),
        seed=int(args.seed),
        max_output_tokens=int(args.max_output_tokens),
        raw_response_path=raw_response_path,
    )
    parsed = parse_json_response(raw_response)
    if "done" not in parsed:
        parsed["done"] = False
    if "confidence" not in parsed:
        parsed["confidence"] = 0.0
    if "correction_instruction" not in parsed:
        parsed["correction_instruction"] = ""
    parsed["done"] = bool(parsed["done"])
    try:
        parsed["confidence"] = float(parsed["confidence"])
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    parsed["correction_instruction"] = str(
        parsed["correction_instruction"]
    ).strip()
    return parsed, raw_response


def prepare_assets(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> dict[str, Any]:
    output_root = paths["output_root"]
    assets_dir = paths["assets_dir"]
    reference_crop_path = assets_dir / "reference_inpainted_crop.png"
    canvas_crop_path = assets_dir / "target_scene_crop.png"
    floor_mask_crop_path = output_root / "floor_mask_crop.png"
    contact_spec_path = output_root / "contact_spec.json"
    base_prompt_path = assets_dir / "base_prompt.md"
    sig_payload = load_json(paths["sig_json"])
    human_parts = target_object_human_parts(sig_payload)
    floor_parts = floor_contact_human_parts(sig_payload)

    if (
        not args.overwrite
        and reference_crop_path.exists()
        and canvas_crop_path.exists()
        and contact_spec_path.exists()
        and base_prompt_path.exists()
        and (not floor_parts or floor_mask_crop_path.exists())
    ):
        return {
            "sig_payload": sig_payload,
            "target_label": target_object_label(sig_payload),
            "human_parts": human_parts,
            "floor_parts": floor_parts,
            "reference_crop_path": reference_crop_path,
            "canvas_crop_path": canvas_crop_path,
            "floor_mask_crop_path": floor_mask_crop_path,
            "contact_spec_path": contact_spec_path,
            "base_prompt_path": base_prompt_path,
            "base_prompt": base_prompt_path.read_text(encoding="utf-8"),
            "vlm_prompt_path": paths["vlm_prompt"],
            "vlm_prompt_template": paths["vlm_prompt"].read_text(
                encoding="utf-8"
            ),
            "palette": contact_palette_from_spec(
                contact_spec_path,
                human_parts,
            ),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    input_payload = load_json(paths["input_dir"] / "input_scene.json")
    system_prompt = paths["system_prompt"].read_text(encoding="utf-8")
    vlm_prompt_template = paths["vlm_prompt"].read_text(encoding="utf-8")

    scene_rgb = load_rgb(paths["scene_image"])
    inpainted_rgb = load_rgb(paths["inpainted_frame"])
    if scene_rgb.shape != inpainted_rgb.shape:
        raise ValueError(
            "Scene image and resized inpainted frame must have the same "
            "shape: "
            f"scene={scene_rgb.shape}, inpainted={inpainted_rgb.shape}"
        )
    image_h, image_w = scene_rgb.shape[:2]

    from PIL import Image

    sam3_processor = build_sam3_processor(
        checkpoint_path=(
            Path(args.sam3_checkpoint).resolve()
            if args.sam3_checkpoint
            else None
        ),
        bpe_path=(
            Path(args.sam3_bpe_path).resolve() if args.sam3_bpe_path else None
        ),
        device=args.sam3_device,
        confidence_threshold=args.sam3_confidence_threshold,
        allow_hf_download=not args.no_sam3_hf_download,
    )
    human_predictions = run_sam3_text_prompt(
        processor=sam3_processor,
        image_rgb=Image.fromarray(inpainted_rgb),
        prompt=args.human_sam3_prompt,
    )
    selected_human = select_highest_confidence_mask(human_predictions)
    human_mask = selected_human["mask"]
    if human_mask.shape != (image_h, image_w):
        raise ValueError(
            "SAM3 human mask shape does not match image shape: "
            f"mask={human_mask.shape[::-1]}, image={image_w}x{image_h}"
        )
    human_bbox = bbox_from_mask(human_mask)
    if human_bbox is None:
        raise ValueError(
            f"SAM3 returned an empty human mask for prompt "
            f"'{args.human_sam3_prompt}'."
        )
    crop_xyxy = pad_bbox(
        human_bbox,
        image_width=image_w,
        image_height=image_h,
        padding_frac=args.padding_frac,
    )
    if not args.disable_aspect_ratio_crop:
        crop_xyxy = fit_bbox_to_image_aspect(
            crop_xyxy,
            image_width=image_w,
            image_height=image_h,
        )

    reference_crop = crop_array(inpainted_rgb, crop_xyxy)
    canvas_crop = crop_array(scene_rgb, crop_xyxy)

    save_rgb(reference_crop_path, reference_crop)
    save_rgb(canvas_crop_path, canvas_crop)
    if floor_parts:
        floor_predictions = run_sam3_text_prompt(
            processor=sam3_processor,
            image_rgb=Image.fromarray(scene_rgb),
            prompt=args.floor_sam3_prompt,
        )
        selected_floor = select_highest_confidence_mask(floor_predictions)
        floor_mask = selected_floor["mask"]
        if floor_mask.shape != (image_h, image_w):
            raise ValueError(
                "SAM3 floor mask shape does not match image shape: "
                f"mask={floor_mask.shape[::-1]}, image={image_w}x{image_h}"
            )
        floor_mask_crop = crop_array(floor_mask, crop_xyxy)
        if not floor_mask_crop.any():
            raise ValueError(
                f"SAM3 returned an empty floor mask inside the contact crop "
                f"for prompt '{args.floor_sam3_prompt}'."
            )
        floor_mask_crop = erode_binary_mask(
            floor_mask_crop,
            int(args.floor_mask_erode_pixels),
        )
        save_binary_mask(floor_mask_crop_path, floor_mask_crop)

    scannet_root = resolve_scannet_root(PROJECT_DIR, args.scannet_root)
    transforms_path = resolve_transforms_path(
        scannet_root, input_payload["scene_context"]
    )
    transforms_payload = load_json(transforms_path)
    source_intrinsics, metadata_w, metadata_h = build_pinhole_intrinsics(
        transforms_payload,
    )
    if (metadata_w, metadata_h) != (image_w, image_h):
        raise ValueError(
            "Scene image shape does not match ScanNet++ camera metadata: "
            f"image={image_w}x{image_h}, metadata={metadata_w}x{metadata_h}"
        )
    contact_intrinsics = adjusted_intrinsics_for_crop(
        source_intrinsics,
        crop_xyxy,
    )
    palette = choose_contact_palette(
        human_parts=human_parts,
        scene_rgb=canvas_crop,
        avoid_mask=None,
    )
    save_contact_spec(contact_spec_path, contact_intrinsics, palette)
    base_prompt = build_contact_prompt(system_prompt, human_parts, palette)
    save_text(base_prompt_path, base_prompt + "\n")

    return {
        "sig_payload": sig_payload,
        "target_label": target_object_label(sig_payload),
        "human_parts": human_parts,
        "floor_parts": floor_parts,
        "reference_crop_path": reference_crop_path,
        "canvas_crop_path": canvas_crop_path,
        "floor_mask_crop_path": floor_mask_crop_path,
        "contact_spec_path": contact_spec_path,
        "base_prompt_path": base_prompt_path,
        "base_prompt": base_prompt,
        "vlm_prompt_path": paths["vlm_prompt"],
        "vlm_prompt_template": vlm_prompt_template,
        "palette": palette,
    }


def build_round_prompt(
    base_prompt: str,
    correction_instruction: str | None,
) -> str:
    if not correction_instruction:
        return base_prompt.rstrip() + "\n"
    return (
        base_prompt.rstrip()
        + "\n\n"
        + "Correction instructions from the VLM evaluator:\n"
        + correction_instruction.strip()
        + "\n\n"
        + NEXT_ROUND_BASE_REMINDER
        + "\n"
    )


def write_round_prompt_package(
    prompt_dir: Path,
    prompt: str,
    reference_path: Path,
    canvas_path: Path,
    previous_composite_path: Path | None,
) -> Path:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    reference_copy = prompt_dir / "01_reference_image.png"
    canvas_copy = prompt_dir / "02_canvas_image.png"
    shutil.copy2(reference_path, reference_copy)
    shutil.copy2(canvas_path, canvas_copy)

    attachments = [
        {
            "order": 1,
            "path": reference_copy.name,
            "role": "Reference Image",
            "description": (
                "Human interacting with the target object; use only to infer "
                "contact locations."
            ),
        },
        {
            "order": 2,
            "path": canvas_copy.name,
            "role": "Canvas Image",
            "description": "Authoritative base image to preserve exactly.",
        },
    ]
    if (
        previous_composite_path is not None
        and previous_composite_path.exists()
    ):
        previous_copy = prompt_dir / "03_previous_composite.png"
        shutil.copy2(previous_composite_path, previous_copy)
        attachments.append(
            {
                "order": 3,
                "path": previous_copy.name,
                "role": "Previous Composite",
                "description": (
                    "Correction context only; do not use as the base image."
                ),
            }
        )

    attachment_lines = [
        "Attach the images from this folder in this exact order:",
        *[
            f"{item['order']}. {item['path']} - "
            f"{item['role']}: {item['description']}"
            for item in attachments
        ],
        "",
    ]
    prompt_path = prompt_dir / "prompt.md"
    save_text(
        prompt_path,
        "\n".join(attachment_lines) + prompt.rstrip() + "\n",
    )
    return prompt_path


def wait_for_generated_image(generated_path: Path) -> None:
    from PIL import Image

    while True:
        input(
            "\nSave the ChatGPT-generated image to:\n"
            f"{generated_path}\n"
            "Press Enter after saving it."
        )
        if not generated_path.exists():
            print(f"Generated image not found yet: {generated_path}")
            continue
        try:
            with Image.open(generated_path) as image:
                image.verify()
        except Exception as exc:
            print(f"Generated image is not readable ({exc}). Save it again.")
            continue
        return


def extract_contact_masks_from_overlay(
    overlay_path: Path,
    resized_overlay_path: Path,
    canvas_path: Path,
    human_parts: list[str],
    palette: list[dict[str, Any]],
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
) -> list[tuple[str, Any]]:
    overlay_rgb, _canvas_hw = normalize_overlay_to_canvas(
        overlay_path=overlay_path,
        canvas_path=canvas_path,
        resized_overlay_path=resized_overlay_path,
    )
    palette_by_part = {
        normalize_label(str(item["part"])): item for item in palette
    }
    contact_colors = [
        tuple(
            int(value)
            for value in palette_by_part[normalize_label(part)]["rgb"]
        )
        for part in human_parts
    ]
    nearest_color, color_accept = classify_nearest_color(
        overlay_rgb=overlay_rgb,
        target_colors_rgb=contact_colors,
        color_max_distance=color_max_distance,
    )

    masks_by_part: list[tuple[str, Any]] = []
    for part_idx, part in enumerate(human_parts):
        mask = (nearest_color == part_idx) & color_accept
        mask = keep_largest_components(
            mask,
            min_area=min_component_area,
            keep_components=keep_components,
        )
        masks_by_part.append((part, mask))

    return masks_by_part


def save_final_contact_masks(
    contact_masks_dir: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    floor_parts: list[str],
    floor_mask_path: Path,
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
) -> None:
    if contact_masks_dir.exists():
        shutil.rmtree(contact_masks_dir)
    contact_masks_dir.mkdir(parents=True, exist_ok=True)
    for part, mask in masks_by_part:
        save_binary_mask(contact_masks_dir / f"{slugify(part)}.png", mask)
    floor_mask_written = False
    if floor_parts:
        expected_hw = masks_by_part[0][1].shape if masks_by_part else None
        floor_mask = load_binary_mask(floor_mask_path, expected_hw=expected_hw)
        for part in floor_parts:
            save_binary_mask(
                contact_masks_dir / f"{slugify(part)}.png",
                floor_mask,
            )
        floor_mask_written = True
    saved_floor_mask_path = (
        str(floor_mask_path) if floor_mask_written else None
    )
    save_json(
        contact_masks_dir / "metadata.json",
        {
            "palette": palette,
            "color_max_distance": float(color_max_distance),
            "min_component_area": int(min_component_area),
            "keep_components": int(keep_components),
            "floor_parts": floor_parts,
            "floor_mask_path": saved_floor_mask_path,
        },
    )


def recompose_masks(
    canvas_path: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import numpy as np

    canvas_rgb = load_rgb(canvas_path)
    composite = canvas_rgb.copy()
    palette_by_part = {
        normalize_label(str(item["part"])): tuple(
            int(value) for value in item["rgb"]
        )
        for item in palette
    }
    for part, mask in masks_by_part:
        normalized = normalize_label(part)
        if normalized not in palette_by_part:
            continue
        composite[mask.astype(bool)] = np.asarray(
            palette_by_part[normalized], dtype=np.uint8
        )
    save_rgb(output_path, composite)


def publish_round_outputs(
    output_root: Path,
    composite_path: Path,
    canvas_path: Path,
    masks_by_part: list[tuple[str, Any]],
    palette: list[dict[str, Any]],
    floor_parts: list[str],
    floor_mask_path: Path,
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
    summary: dict[str, Any],
) -> None:
    root_masks_dir = output_root / "contact_masks"
    save_final_contact_masks(
        contact_masks_dir=root_masks_dir,
        masks_by_part=masks_by_part,
        palette=palette,
        floor_parts=floor_parts,
        floor_mask_path=floor_mask_path,
        color_max_distance=color_max_distance,
        min_component_area=min_component_area,
        keep_components=keep_components,
    )
    shutil.copy2(composite_path, output_root / "contact_overlay.png")
    save_contact_visualization(
        canvas_rgb=load_rgb(canvas_path),
        masks_by_part=masks_by_part,
        palette=palette,
        visualization_path=output_root / "contact_overlay_visualization.png",
    )
    save_json(output_root / "agentic_summary.json", summary)


def run_agentic_loop(args: argparse.Namespace, assets: dict[str, Any]) -> int:
    output_root = resolve_paths(args)["output_root"]
    rounds_root = output_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    api_key = read_api_key(Path(args.api_key_file).resolve())

    base_prompt = str(assets["base_prompt"])
    vlm_prompt_template = str(assets["vlm_prompt_template"])
    human_parts = list(assets["human_parts"])
    floor_parts = list(assets["floor_parts"])
    palette = list(assets["palette"])
    reference_path = Path(assets["reference_crop_path"])
    canvas_path = Path(assets["canvas_crop_path"])
    floor_mask_path = Path(assets["floor_mask_crop_path"])
    correction_instruction: str | None = None
    latest_summary: dict[str, Any] | None = None
    latest_masks_by_part: list[tuple[str, Any]] | None = None
    latest_composite_path: Path | None = None

    for round_index in range(1, int(args.max_rounds) + 1):
        round_dir = rounds_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_prompt_dir = round_dir / "prompt"
        generated_path = round_dir / "generated_contact_overlay.png"
        resized_overlay_path = (
            round_dir / "generated_contact_overlay_resized.png"
        )
        composite_path = round_dir / "composite.png"
        raw_response_path = round_dir / "gemini_raw_response.txt"
        stale_round_masks_dir = round_dir / "contact_masks"
        if stale_round_masks_dir.exists():
            shutil.rmtree(stale_round_masks_dir)

        prompt = build_round_prompt(base_prompt, correction_instruction)
        previous_composite = (
            rounds_root / f"round_{round_index - 1:02d}" / "composite.png"
            if round_index > 1
            else None
        )
        prompt_path = write_round_prompt_package(
            prompt_dir=round_prompt_dir,
            prompt=prompt,
            reference_path=reference_path,
            canvas_path=canvas_path,
            previous_composite_path=previous_composite,
        )
        print("\n" + "=" * 72)
        print(f"Round {round_index:02d}/{args.max_rounds}")
        print(f"Prompt: {prompt_path}")
        print(f"Upload image 1: {round_prompt_dir / '01_reference_image.png'}")
        print(f"Upload image 2: {round_prompt_dir / '02_canvas_image.png'}")
        if previous_composite is not None and previous_composite.exists():
            print(
                "Upload image 3: "
                f"{round_prompt_dir / '03_previous_composite.png'}"
            )
        print(f"Save ChatGPT result as: {generated_path}")
        wait_for_generated_image(generated_path)

        contact_masks = extract_contact_masks_from_overlay(
            overlay_path=generated_path,
            resized_overlay_path=resized_overlay_path,
            canvas_path=canvas_path,
            human_parts=human_parts,
            palette=palette,
            color_max_distance=float(args.color_max_distance),
            min_component_area=int(args.min_component_area),
            keep_components=int(args.keep_components),
        )
        recompose_masks(
            canvas_path=canvas_path,
            masks_by_part=contact_masks,
            palette=palette,
            output_path=composite_path,
        )

        print(f"Composite for VLM evaluation: {composite_path}")
        evaluation = None
        raw_response = None
        max_attempts = max(1, int(args.gemini_retries))
        for attempt_index in range(1, max_attempts + 1):
            print(
                f"Gemini evaluation attempt {attempt_index}/{max_attempts}; "
                f"raw response: {raw_response_path}"
            )
            try:
                raw_response_path.write_text("", encoding="utf-8")
                evaluation, raw_response = evaluate_round(
                    args=args,
                    api_key=api_key,
                    vlm_prompt_template=vlm_prompt_template,
                    target_label=str(assets["target_label"]),
                    palette=palette,
                    reference_path=reference_path,
                    canvas_path=canvas_path,
                    composite_path=composite_path,
                    raw_response_path=raw_response_path,
                )
                break
            except Exception as exc:
                print(
                    "Gemini evaluation attempt failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt_index >= max_attempts:
                    raise
                sleep_s = max(0.0, float(args.gemini_retry_sleep_s))
                print(f"Retrying Gemini evaluation in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
        if evaluation is None or raw_response is None:
            raise RuntimeError("Gemini evaluation did not produce a response.")
        print(
            "VLM result: "
            f"done={evaluation['done']} "
            f"confidence={evaluation['confidence']:.3f}"
        )
        if evaluation["correction_instruction"]:
            print(f"Correction: {evaluation['correction_instruction']}")

        latest_composite_path = composite_path
        latest_masks_by_part = contact_masks
        latest_summary = {
            "interaction_name": args.interaction_name,
            "done": bool(evaluation["done"]),
            "accepted_round": round_index if evaluation["done"] else None,
            "latest_round": round_index,
            "max_rounds": int(args.max_rounds),
            "provider": "gemini",
            "model": args.model,
            "target_object": assets["target_label"],
            "human_parts": human_parts,
            "floor_parts": floor_parts,
            "palette": palette,
            "composite_includes_floor_contacts": False,
            "reference_crop": str(reference_path),
            "canvas_crop": str(canvas_path),
            "contact_spec": str(assets["contact_spec_path"]),
            "vlm_prompt": str(assets["vlm_prompt_path"]),
            "latest_evaluation": evaluation,
        }

        if evaluation["done"]:
            publish_round_outputs(
                output_root=output_root,
                composite_path=composite_path,
                canvas_path=canvas_path,
                masks_by_part=contact_masks,
                palette=palette,
                floor_parts=floor_parts,
                floor_mask_path=floor_mask_path,
                color_max_distance=float(args.color_max_distance),
                min_component_area=int(args.min_component_area),
                keep_components=int(args.keep_components),
                summary=latest_summary,
            )
            print(f"Accepted contact masks at round {round_index:02d}.")
            return 0
        correction_instruction = evaluation["correction_instruction"] or (
            "The previous result was not accepted. Improve the colored "
            "contact "
            "mask locations and sizes while preserving the original canvas."
        )

    if (
        latest_summary is not None
        and latest_composite_path is not None
        and latest_masks_by_part is not None
    ):
        latest_summary["done"] = False
        latest_summary["accepted_round"] = None
        latest_summary["stopped_reason"] = "max_rounds_reached"
        publish_round_outputs(
            output_root=output_root,
            composite_path=latest_composite_path,
            canvas_path=canvas_path,
            masks_by_part=latest_masks_by_part,
            palette=palette,
            floor_parts=floor_parts,
            floor_mask_path=floor_mask_path,
            color_max_distance=float(args.color_max_distance),
            min_component_area=int(args.min_component_area),
            keep_components=int(args.keep_components),
            summary=latest_summary,
        )
    print(f"Reached max rounds ({args.max_rounds}) without VLM acceptance.")
    return 0 if args.allow_unaccepted else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_rounds) < 1:
        raise ValueError("--max-rounds must be at least 1")
    paths = resolve_paths(args)
    assets = prepare_assets(args, paths)
    print(f"Wrote/loaded reference crop: {assets['reference_crop_path']}")
    print(f"Wrote/loaded canvas crop: {assets['canvas_crop_path']}")
    if assets["floor_parts"]:
        print(
            "Wrote/loaded floor mask crop: "
            f"{assets['floor_mask_crop_path']}"
        )
    print(f"Wrote/loaded contact spec: {assets['contact_spec_path']}")
    print(f"Wrote/loaded base prompt: {assets['base_prompt_path']}")
    print(f"Loaded VLM prompt: {assets['vlm_prompt_path']}")
    if args.prepare_only:
        return 0
    return run_agentic_loop(args, assets)


if __name__ == "__main__":
    raise SystemExit(main())
