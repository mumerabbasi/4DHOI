from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    contact_palette_from_spec,
    floor_contact_human_parts,
    load_binary_mask,
    load_json,
    read_api_key,
    run_gemini_image_edit,
    save_binary_mask,
    save_contact_masks_from_overlay,
    slugify,
    target_object_human_parts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Estimate cropped scene contact masks with Gemini.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--canvas-image", default=None)
    parser.add_argument("--contact-spec", default=None)
    parser.add_argument("--target-mask-crop", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument(
        "--api-key-file",
        default=str(project_dir / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-2.5-flash-image")
    parser.add_argument("--overwrite-overlay", action="store_true")
    parser.add_argument("--color-max-distance", type=float, default=30.0)
    parser.add_argument("--color-hue-tolerance", type=float, default=10.0)
    parser.add_argument("--min-color-saturation", type=int, default=80)
    parser.add_argument("--min-color-value", type=int, default=40)
    parser.add_argument("--min-component-area", type=int, default=50)
    parser.add_argument("--keep-components", type=int, default=1)
    parser.add_argument("--target-mask-erode-pixels", type=int, default=0)
    parser.set_defaults(script_dir=script_dir, project_dir=project_dir)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name
    )
    prompt_dir = output_root / "prompt"
    prompt_path = (
        Path(args.prompt).resolve()
        if args.prompt
        else prompt_dir / "prompt.md"
    )
    reference_image_path = (
        Path(args.reference_image).resolve()
        if args.reference_image
        else prompt_dir / "reference_inpainted_crop.png"
    )
    canvas_image_path = (
        Path(args.canvas_image).resolve()
        if args.canvas_image
        else prompt_dir / "target_scene_crop.png"
    )
    target_mask_crop_path = (
        Path(args.target_mask_crop).resolve()
        if args.target_mask_crop
        else output_root / "target_mask_crop.png"
    )
    contact_spec_path = (
        Path(args.contact_spec).resolve()
        if args.contact_spec
        else output_root / "contact_spec.json"
    )
    floor_mask_crop_path = output_root / "floor_mask_crop.png"
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else project_dir
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "scene_interaction_graph.json"
    )

    from PIL import Image

    sig_payload = load_json(sig_json_path)
    human_parts = target_object_human_parts(sig_payload)
    floor_parts = floor_contact_human_parts(sig_payload)
    output_root.mkdir(parents=True, exist_ok=True)

    contact_overlay_path = output_root / "contact_overlay.png"
    resized_overlay_path = output_root / "contact_overlay_resized.png"
    visualization_path = output_root / "contact_overlay_visualization.png"
    contact_masks_dir = output_root / "contact_masks"
    contact_palette = contact_palette_from_spec(contact_spec_path, human_parts)

    if contact_overlay_path.exists() and not args.overwrite_overlay:
        print(
            "Skipping Gemini call; found existing contact overlay: "
            f"{contact_overlay_path}"
        )
    else:
        if contact_overlay_path.exists():
            print(f"Overwriting existing contact overlay: {contact_overlay_path}")
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {prompt_path}. "
                "Run 01_build_prompt.py first."
            )
        prompt = prompt_path.read_text(encoding="utf-8")
        api_key = read_api_key(Path(args.api_key_file).resolve())
        reference_image = Image.open(reference_image_path).convert("RGB")
        canvas_image = Image.open(canvas_image_path).convert("RGB")
        contact_overlay = run_gemini_image_edit(
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            reference_image=reference_image,
            canvas_image=canvas_image,
        )
        contact_overlay.save(contact_overlay_path)

    written_paths = save_contact_masks_from_overlay(
        overlay_path=contact_overlay_path,
        resized_overlay_path=resized_overlay_path,
        canvas_path=canvas_image_path,
        target_mask_crop_path=target_mask_crop_path,
        contact_masks_dir=contact_masks_dir,
        human_parts=human_parts,
        palette=contact_palette,
        visualization_path=visualization_path,
        color_max_distance=args.color_max_distance,
        color_hue_tolerance=args.color_hue_tolerance,
        min_color_saturation=args.min_color_saturation,
        min_color_value=args.min_color_value,
        min_component_area=args.min_component_area,
        keep_components=args.keep_components,
        target_mask_erode_pixels=args.target_mask_erode_pixels,
    )
    if floor_parts:
        if not floor_mask_crop_path.exists():
            raise FileNotFoundError(
                f"Floor mask crop not found: {floor_mask_crop_path}. "
                "Run 01_build_prompt.py first so SAM3 can create it."
            )
        canvas_w, canvas_h = Image.open(canvas_image_path).size
        floor_mask = load_binary_mask(
            floor_mask_crop_path,
            expected_hw=(canvas_h, canvas_w),
        )
        for part in floor_parts:
            mask_path = contact_masks_dir / f"{slugify(part)}.png"
            save_binary_mask(mask_path, floor_mask)
            written_paths.append(mask_path)

    print(f"Read contact overlay: {contact_overlay_path}")
    print(f"Wrote resized contact overlay: {resized_overlay_path}")
    print(f"Wrote contact overlay visualization: {visualization_path}")
    for path in written_paths:
        print(f"Wrote contact mask artifact: {path}")
    return contact_overlay_path


if __name__ == "__main__":
    main()
