from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    adjusted_intrinsics_for_crop,
    bbox_from_mask,
    build_contact_prompt,
    build_pinhole_intrinsics,
    build_sam3_processor,
    choose_contact_palette,
    crop_array,
    erode_binary_mask,
    fit_bbox_to_image_aspect,
    floor_contact_human_parts,
    get_default_sam3_device,
    load_json,
    load_rgb,
    pad_bbox,
    resolve_scannet_root,
    resolve_transforms_path,
    save_binary_mask,
    save_contact_spec,
    save_rgb,
    save_text,
    run_sam3_text_prompt,
    select_highest_confidence_mask,
    target_object_human_parts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Build the contact estimation Gemini prompt and cropped input "
            "images."
        ),
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--inpainted-frame", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--prompt-out", default=None)
    parser.add_argument("--padding-frac", type=float, default=0.25)
    parser.add_argument(
        "--disable-aspect-ratio-crop",
        action="store_true",
        help=(
            "Skip expanding the human crop to the source image aspect ratio. "
            "The crop will remain the padded tight human bbox."
        ),
    )
    parser.add_argument("--human-sam3-prompt", default="person")
    parser.add_argument("--floor-sam3-prompt", default="floor")
    parser.add_argument("--floor-mask-erode-pixels", type=int, default=3)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=None)
    parser.add_argument("--sam3-device", default=get_default_sam3_device())
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
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
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else project_dir / "01_Generate_SIG" / "input_prompts" / args.interaction_name
    )
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else project_dir
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "sig.json"
    )
    input_payload = load_json(input_dir / "input_scene.json")
    scannet_root = resolve_scannet_root(project_dir, args.scannet_root)
    scene_image_path = (
        Path(args.scene_image).resolve()
        if args.scene_image
        else sig_json_path.parent / "scene_image.png"
    )
    inpainted_frame_path = (
        Path(args.inpainted_frame).resolve()
        if args.inpainted_frame
        else project_dir
        / "02_Generate_Human_Frame"
        / "output"
        / args.interaction_name
        / "inpainted_frame_resized.png"
    )
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_estimate_contact.md"
    )
    prompt_path = (
        Path(args.prompt_out).resolve()
        if args.prompt_out
        else prompt_dir / "prompt.md"
    )

    sig_payload = load_json(sig_json_path)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    human_parts = target_object_human_parts(sig_payload)
    floor_parts = floor_contact_human_parts(sig_payload)

    scene_rgb = load_rgb(scene_image_path)
    inpainted_rgb = load_rgb(inpainted_frame_path)
    if scene_rgb.shape != inpainted_rgb.shape:
        raise ValueError(
            "Scene image and resized inpainted frame must have the same shape: "
            f"scene={scene_rgb.shape}, inpainted={inpainted_rgb.shape}"
        )
    image_h, image_w = scene_rgb.shape[:2]

    from PIL import Image

    sam3_processor = build_sam3_processor(
        checkpoint_path=Path(args.sam3_checkpoint).resolve()
        if args.sam3_checkpoint
        else None,
        bpe_path=Path(args.sam3_bpe_path).resolve()
        if args.sam3_bpe_path
        else None,
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
    floor_mask_crop = None
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

    reference_crop_path = prompt_dir / "reference_inpainted_crop.png"
    canvas_crop_path = prompt_dir / "target_scene_crop.png"
    floor_mask_crop_path = output_root / "floor_mask_crop.png"
    contact_spec_path = output_root / "contact_spec.json"

    save_rgb(reference_crop_path, reference_crop)
    save_rgb(canvas_crop_path, canvas_crop)
    if floor_mask_crop is not None:
        save_binary_mask(floor_mask_crop_path, floor_mask_crop)

    transforms_path = resolve_transforms_path(
        scannet_root,
        input_payload["scene_context"],
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
    contact_palette = choose_contact_palette(
        human_parts=human_parts,
        scene_rgb=canvas_crop,
        avoid_mask=None,
    )
    save_contact_spec(contact_spec_path, contact_intrinsics, contact_palette)
    prompt = build_contact_prompt(system_prompt, human_parts, contact_palette)
    save_text(prompt_path, prompt + "\n")

    print(f"Wrote contact estimation prompt: {prompt_path}")
    print(f"Wrote reference crop: {reference_crop_path}")
    print(f"Wrote canvas crop: {canvas_crop_path}")
    if floor_mask_crop is not None:
        print(f"Wrote floor mask crop: {floor_mask_crop_path}")
    print(f"Wrote contact spec JSON: {contact_spec_path}")
    return prompt_path


if __name__ == "__main__":
    main()
