from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
LEGACY_CONTACT_DIR = PROJECT_DIR / "03_Estimate_Contact"
if str(LEGACY_CONTACT_DIR) not in sys.path:
    sys.path.insert(0, str(LEGACY_CONTACT_DIR))

from common import (  # noqa: E402
    build_pinhole_intrinsics,
    build_sam3_processor,
    classify_nearest_color,
    contact_palette_from_spec,
    crop_array,
    erode_binary_mask,
    floor_contact_human_parts,
    get_default_sam3_device,
    keep_largest_components,
    load_json,
    load_rgb,
    normalize_label,
    resolve_scannet_root,
    resolve_transforms_path,
    resize_cover_center_crop_array,
    run_sam3_text_prompt,
    save_binary_mask,
    save_json,
    select_highest_confidence_mask,
    slugify,
)


DEFAULT_COLOR_MAX_DISTANCE = 90.0
DEFAULT_MIN_COMPONENT_AREA = 0
DEFAULT_KEEP_COMPONENTS = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GT contact masks from manually annotated overlays.",
    )
    parser.add_argument(
        "--interaction_name",
        default=None,
        help=(
            "Process one interaction, for example interaction_01. "
            "If omitted, process every interaction_* directory."
        ),
    )
    parser.add_argument(
        "--gt-output-dir",
        default=str(SCRIPT_DIR / "output"),
        help="Directory containing GT interaction outputs.",
    )
    parser.add_argument(
        "--overlay-name",
        default="contact_overlay_gt.png",
        help="Overlay filename inside each interaction directory.",
    )
    parser.add_argument(
        "--masks-dir-name",
        default="contact_masks_gt",
        help="Output mask directory name inside each interaction directory.",
    )
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--input-scene-json", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--floor-sam3-prompt", default="floor")
    parser.add_argument("--floor-mask-erode-pixels", type=int, default=3)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=None)
    parser.add_argument("--sam3-device", default=get_default_sam3_device())
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
    parser.add_argument(
        "--color-max-distance",
        type=float,
        default=DEFAULT_COLOR_MAX_DISTANCE,
        help="Maximum RGB distance accepted as a contact-mask color.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help="Minimum connected-component area to keep.",
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=DEFAULT_KEEP_COMPONENTS,
        help="Number of largest components to keep per body part.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing GT contact mask directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when required inputs are missing.",
    )
    return parser.parse_args(argv)


def interaction_dirs(
    gt_output_dir: Path,
    interaction_name: str | None,
) -> list[Path]:
    if interaction_name:
        return [gt_output_dir / interaction_name]
    return sorted(
        path
        for path in gt_output_dir.glob("interaction_*")
        if path.is_dir()
    )


def fail_or_warn(message: str, strict: bool) -> bool:
    if strict:
        raise FileNotFoundError(message)
    print(f"WARNING: {message}")
    return False


def palette_parts(contact_spec: dict[str, Any], contact_spec_path: Path) -> list[str]:
    palette = contact_spec.get("palette")
    if not isinstance(palette, dict):
        raise ValueError(f"Missing object field 'palette' in {contact_spec_path}")

    parts_payload = palette.get("parts")
    if not isinstance(parts_payload, list):
        raise ValueError(f"Missing list field 'palette.parts' in {contact_spec_path}")

    parts: list[str] = []
    for item in parts_payload:
        if not isinstance(item, dict):
            continue
        part = str(item.get("part", "")).strip()
        if part:
            parts.append(part)

    if not parts:
        raise ValueError(f"No contact parts found in {contact_spec_path}")
    return parts


def default_sig_json_path(interaction_name: str) -> Path:
    return (
        PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / interaction_name
        / "sig.json"
    )


def default_input_scene_json_path(interaction_name: str) -> Path:
    return (
        PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json"
    )


def default_scene_image_path(interaction_name: str) -> Path:
    return (
        PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / interaction_name
        / "scene_image.png"
    )


def resolve_optional_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).resolve() if raw_path else default_path


def contact_crop_xyxy_from_intrinsics(
    source_intrinsics: list[list[float]],
    contact_intrinsics: list[list[float]],
    canvas_shape: tuple[int, int],
    source_shape: tuple[int, int],
) -> list[int]:
    canvas_h, canvas_w = canvas_shape
    source_h, source_w = source_shape
    crop_x0 = int(round(float(source_intrinsics[0][2]) - float(contact_intrinsics[0][2])))
    crop_y0 = int(round(float(source_intrinsics[1][2]) - float(contact_intrinsics[1][2])))
    crop_x0 = max(0, min(source_w - canvas_w, crop_x0))
    crop_y0 = max(0, min(source_h - canvas_h, crop_y0))
    return [crop_x0, crop_y0, crop_x0 + canvas_w, crop_y0 + canvas_h]


def save_gt_contact_masks_from_overlay(
    overlay_path: Path,
    canvas_path: Path,
    contact_masks_dir: Path,
    human_parts: list[str],
    palette: list[dict[str, Any]],
    color_max_distance: float,
    min_component_area: int,
    keep_components: int,
) -> list[Path]:
    from PIL import Image

    canvas_rgb = load_rgb(canvas_path)
    overlay_rgb = load_rgb(overlay_path)
    overlay_rgb = resize_cover_center_crop_array(
        overlay_rgb,
        canvas_rgb.shape[:2],
        resampling=Image.Resampling.NEAREST,
    )

    palette_by_part = {
        normalize_label(str(item["part"])): item
        for item in palette
    }
    target_colors = [
        tuple(int(value) for value in palette_by_part[normalize_label(part)]["rgb"])
        for part in human_parts
    ]
    nearest_color, color_accept = classify_nearest_color(
        overlay_rgb=overlay_rgb,
        target_colors_rgb=target_colors,
        color_max_distance=color_max_distance,
    )

    contact_masks_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for part_idx, part in enumerate(human_parts):
        mask = (nearest_color == part_idx) & color_accept
        mask = keep_largest_components(
            mask,
            min_area=min_component_area,
            keep_components=keep_components,
        )
        mask_path = contact_masks_dir / f"{slugify(part)}.png"
        save_binary_mask(mask_path, mask)
        written_paths.append(mask_path)

    metadata_path = contact_masks_dir / "metadata.json"
    save_json(
        metadata_path,
        {
            "palette": palette,
            "color_max_distance": float(color_max_distance),
            "min_component_area": int(min_component_area),
            "keep_components": int(keep_components),
        },
    )
    written_paths.append(metadata_path)
    return written_paths


def save_floor_contact_masks(
    interaction_dir: Path,
    contact_masks_dir: Path,
    canvas_path: Path,
    contact_spec_path: Path,
    sig_payload: dict[str, Any],
    floor_parts: list[str],
    args: argparse.Namespace,
) -> list[Path]:
    from PIL import Image

    interaction_name = interaction_dir.name
    input_scene_path = resolve_optional_path(
        args.input_scene_json,
        default_input_scene_json_path(interaction_name),
    )
    scene_image_path = resolve_optional_path(
        args.scene_image,
        default_scene_image_path(interaction_name),
    )
    missing_inputs = [
        path
        for path in (input_scene_path, scene_image_path, canvas_path, contact_spec_path)
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required floor segmentation input(s): "
            + ", ".join(str(path) for path in missing_inputs)
        )

    input_payload = load_json(input_scene_path)
    transforms_path = resolve_transforms_path(
        resolve_scannet_root(PROJECT_DIR, args.scannet_root),
        input_payload["scene_context"],
    )
    transforms_payload = load_json(transforms_path)
    source_intrinsics, source_w, source_h = build_pinhole_intrinsics(
        transforms_payload,
    )

    contact_spec = load_json(contact_spec_path)
    contact_intrinsics = contact_spec["camera"]["intrinsics_3x3"]
    scene_rgb = load_rgb(scene_image_path)
    canvas_rgb = load_rgb(canvas_path)
    if scene_rgb.shape[:2] != (source_h, source_w):
        raise ValueError(
            "Scene image shape does not match ScanNet++ camera metadata: "
            f"image={scene_rgb.shape[1]}x{scene_rgb.shape[0]}, "
            f"metadata={source_w}x{source_h}"
        )
    crop_xyxy = contact_crop_xyxy_from_intrinsics(
        source_intrinsics=source_intrinsics,
        contact_intrinsics=contact_intrinsics,
        canvas_shape=canvas_rgb.shape[:2],
        source_shape=scene_rgb.shape[:2],
    )

    sam3_processor = getattr(args, "_sam3_processor", None)
    if sam3_processor is None:
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
        setattr(args, "_sam3_processor", sam3_processor)
    floor_predictions = run_sam3_text_prompt(
        processor=sam3_processor,
        image_rgb=Image.fromarray(scene_rgb),
        prompt=args.floor_sam3_prompt,
    )
    selected_floor = select_highest_confidence_mask(floor_predictions)
    floor_mask = selected_floor["mask"]
    if floor_mask.shape != scene_rgb.shape[:2]:
        raise ValueError(
            "SAM3 floor mask shape does not match image shape: "
            f"mask={floor_mask.shape[::-1]}, "
            f"image={scene_rgb.shape[1]}x{scene_rgb.shape[0]}"
        )
    floor_mask_crop = crop_array(floor_mask, crop_xyxy)
    if floor_mask_crop.shape != canvas_rgb.shape[:2]:
        raise ValueError(
            "Cropped floor mask shape does not match contact canvas: "
            f"mask={floor_mask_crop.shape[::-1]}, "
            f"canvas={canvas_rgb.shape[1]}x{canvas_rgb.shape[0]}"
        )
    if not floor_mask_crop.any():
        raise ValueError(
            f"SAM3 returned an empty floor mask inside the contact crop "
            f"for prompt '{args.floor_sam3_prompt}'."
        )
    floor_mask_crop = erode_binary_mask(
        floor_mask_crop,
        int(args.floor_mask_erode_pixels),
    )

    contact_masks_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    floor_mask_crop_path = interaction_dir / "floor_mask_crop.png"
    save_binary_mask(floor_mask_crop_path, floor_mask_crop)
    written_paths.append(floor_mask_crop_path)
    for part in floor_parts:
        mask_path = contact_masks_dir / f"{slugify(part)}.png"
        save_binary_mask(mask_path, floor_mask_crop)
        written_paths.append(mask_path)

    metadata_path = contact_masks_dir / "metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(
        {
            "floor_parts": floor_parts,
            "floor_mask_path": str(floor_mask_crop_path),
            "floor_sam3_prompt": str(args.floor_sam3_prompt),
            "floor_sam3_score": float(selected_floor["sam3_score"]),
            "floor_crop_xyxy": crop_xyxy,
            "floor_source_scene_image_path": str(scene_image_path),
            "floor_source_input_scene_json_path": str(input_scene_path),
            "floor_source_transforms_path": str(transforms_path),
        }
    )
    save_json(metadata_path, metadata)
    if metadata_path not in written_paths:
        written_paths.append(metadata_path)
    return written_paths


def write_gt_masks(
    interaction_dir: Path,
    args: argparse.Namespace,
) -> str:
    if not interaction_dir.exists():
        fail_or_warn(
            f"Interaction directory not found: {interaction_dir}",
            bool(args.strict),
        )
        return "skipped"

    overlay_path = interaction_dir / str(args.overlay_name)
    contact_masks_dir = interaction_dir / str(args.masks_dir_name)
    canvas_path = interaction_dir / "assets" / "target_scene_crop.png"
    contact_spec_path = interaction_dir / "contact_spec.json"
    sig_path = resolve_optional_path(
        args.sig_json,
        default_sig_json_path(interaction_dir.name),
    )

    if contact_masks_dir.exists() and not args.overwrite:
        print(f"Skipping existing GT masks: {contact_masks_dir}")
        return "skipped"

    missing_inputs = [
        path
        for path in (overlay_path, canvas_path, contact_spec_path)
        if not path.exists()
    ]
    if missing_inputs:
        fail_or_warn(
            "Missing required input(s): "
            + ", ".join(str(path) for path in missing_inputs),
            bool(args.strict),
        )
        return "skipped"

    if contact_masks_dir.exists():
        shutil.rmtree(contact_masks_dir)

    sig_payload = load_json(sig_path) if sig_path.exists() else {}
    floor_parts = floor_contact_human_parts(sig_payload) if sig_payload else []
    contact_spec = load_json(contact_spec_path)
    human_parts = palette_parts(contact_spec, contact_spec_path)
    palette = contact_palette_from_spec(contact_spec_path, human_parts)
    written_paths = save_gt_contact_masks_from_overlay(
        overlay_path=overlay_path,
        canvas_path=canvas_path,
        contact_masks_dir=contact_masks_dir,
        human_parts=human_parts,
        palette=palette,
        color_max_distance=float(args.color_max_distance),
        min_component_area=int(args.min_component_area),
        keep_components=int(args.keep_components),
    )

    metadata_path = contact_masks_dir / "metadata.json"
    metadata = load_json(metadata_path)
    metadata.update(
        {
            "source": "gt_contact_overlay",
            "source_overlay_path": str(overlay_path),
            "source_canvas_path": str(canvas_path),
            "source_contact_spec_path": str(contact_spec_path),
            "human_parts": human_parts,
            "source_sig_path": str(sig_path) if sig_path.exists() else None,
        }
    )
    save_json(metadata_path, metadata)
    if floor_parts:
        floor_written_paths = save_floor_contact_masks(
            interaction_dir=interaction_dir,
            contact_masks_dir=contact_masks_dir,
            canvas_path=canvas_path,
            contact_spec_path=contact_spec_path,
            sig_payload=sig_payload,
            floor_parts=floor_parts,
            args=args,
        )
        for path in floor_written_paths:
            if path not in written_paths:
                written_paths.append(path)

    print(f"Wrote GT contact masks: {contact_masks_dir}")
    for path in written_paths:
        print(f"Wrote artifact: {path}")
    return "written"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt_output_dir = Path(args.gt_output_dir).resolve()
    dirs = interaction_dirs(gt_output_dir, args.interaction_name)
    if not dirs:
        fail_or_warn(
            f"No interaction directories found under {gt_output_dir}",
            bool(args.strict),
        )
        return 1 if args.strict else 0

    counts = {"written": 0, "skipped": 0}
    for interaction_dir in dirs:
        result = write_gt_masks(interaction_dir, args)
        counts[result] = counts.get(result, 0) + 1

    print(
        "GT contact mask build complete: "
        f"{counts.get('written', 0)} written, "
        f"{counts.get('skipped', 0)} skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
