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
    contact_palette_from_spec,
    load_binary_mask,
    load_json,
    save_binary_mask,
    save_contact_masks_from_overlay,
    save_json,
    slugify,
)


DEFAULT_COLOR_MAX_DISTANCE = 90.0
DEFAULT_MIN_COMPONENT_AREA = 0
DEFAULT_KEEP_COMPONENTS = 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build binary contact masks from agentic round-1 contact overlays."
        ),
    )
    parser.add_argument(
        "--interaction_name",
        default=None,
        help=(
            "Process one interaction, for example interaction_22. "
            "If omitted, process every interaction_* directory."
        ),
    )
    parser.add_argument(
        "--agentic-output-dir",
        default=str(SCRIPT_DIR / "output"),
        help="Directory containing agentic interaction outputs.",
    )
    parser.add_argument(
        "--color-max-distance",
        type=float,
        default=None,
        help=(
            "Maximum RGB distance accepted as a contact-mask color. "
            f"Defaults to source metadata, then {DEFAULT_COLOR_MAX_DISTANCE}."
        ),
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=None,
        help=(
            "Minimum connected-component area to keep. "
            f"Defaults to source metadata, then {DEFAULT_MIN_COMPONENT_AREA}."
        ),
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=None,
        help=(
            "Number of largest components to keep per body part. "
            f"Defaults to source metadata, then {DEFAULT_KEEP_COMPONENTS}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing round_01/contact_masks directory.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of warning when required inputs are missing.",
    )
    return parser.parse_args(argv)


def metadata_default(
    metadata: dict[str, Any],
    cli_value: int | float | None,
    key: str,
    fallback: int | float,
) -> int | float:
    if cli_value is not None:
        return cli_value
    value = metadata.get(key)
    if isinstance(value, (int, float)):
        return value
    return fallback


def read_source_metadata(interaction_dir: Path) -> dict[str, Any]:
    metadata_path = interaction_dir / "contact_masks" / "metadata.json"
    if not metadata_path.exists():
        return {}
    return load_json(metadata_path)


def interaction_dirs(
    agentic_output_dir: Path,
    interaction_name: str | None,
) -> list[Path]:
    if interaction_name:
        return [agentic_output_dir / interaction_name]
    return sorted(
        path
        for path in agentic_output_dir.glob("interaction_*")
        if path.is_dir()
    )


def fail_or_warn(message: str, strict: bool) -> bool:
    if strict:
        raise FileNotFoundError(message)
    print(f"WARNING: {message}")
    return False


def required_summary_list(
    summary: dict[str, Any],
    key: str,
    path: Path,
) -> list[str]:
    value = summary.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Expected list field '{key}' in {path}")
    return [str(item) for item in value]


def write_round1_masks(
    interaction_dir: Path,
    args: argparse.Namespace,
) -> str:
    if not interaction_dir.exists():
        fail_or_warn(
            f"Interaction directory not found: {interaction_dir}",
            bool(args.strict),
        )
        return "skipped"

    round_dir = interaction_dir / "rounds" / "round_01"
    overlay_path = round_dir / "generated_contact_overlay.png"
    resized_overlay_path = round_dir / "generated_contact_overlay_resized.png"
    visualization_path = round_dir / "contact_overlay_visualization.png"
    contact_masks_dir = round_dir / "contact_masks"
    canvas_path = interaction_dir / "assets" / "target_scene_crop.png"
    contact_spec_path = interaction_dir / "contact_spec.json"
    summary_path = interaction_dir / "agentic_summary.json"
    floor_mask_path = interaction_dir / "floor_mask_crop.png"

    if contact_masks_dir.exists() and not args.overwrite:
        print(f"Skipping existing round-1 masks: {contact_masks_dir}")
        return "skipped"

    missing_inputs = [
        path
        for path in (overlay_path, canvas_path, contact_spec_path, summary_path)
        if not path.exists()
    ]
    if missing_inputs:
        fail_or_warn(
            "Missing required input(s): "
            + ", ".join(str(path) for path in missing_inputs),
            bool(args.strict),
        )
        return "skipped"

    summary = load_json(summary_path)
    human_parts = required_summary_list(summary, "human_parts", summary_path)
    floor_parts = required_summary_list(summary, "floor_parts", summary_path)
    if floor_parts and not floor_mask_path.exists():
        fail_or_warn(
            f"Missing floor mask crop for floor contact(s) {floor_parts}: "
            f"{floor_mask_path}",
            bool(args.strict),
        )
        return "skipped"

    if contact_masks_dir.exists():
        shutil.rmtree(contact_masks_dir)

    source_metadata = read_source_metadata(interaction_dir)
    color_max_distance = float(
        metadata_default(
            source_metadata,
            args.color_max_distance,
            "color_max_distance",
            DEFAULT_COLOR_MAX_DISTANCE,
        )
    )
    min_component_area = int(
        metadata_default(
            source_metadata,
            args.min_component_area,
            "min_component_area",
            DEFAULT_MIN_COMPONENT_AREA,
        )
    )
    keep_components = int(
        metadata_default(
            source_metadata,
            args.keep_components,
            "keep_components",
            DEFAULT_KEEP_COMPONENTS,
        )
    )

    palette = contact_palette_from_spec(contact_spec_path, human_parts)
    written_paths = save_contact_masks_from_overlay(
        overlay_path=overlay_path,
        resized_overlay_path=resized_overlay_path,
        canvas_path=canvas_path,
        contact_masks_dir=contact_masks_dir,
        human_parts=human_parts,
        palette=palette,
        visualization_path=visualization_path,
        color_max_distance=color_max_distance,
        min_component_area=min_component_area,
        keep_components=keep_components,
    )

    if floor_parts:
        from PIL import Image

        canvas_w, canvas_h = Image.open(canvas_path).size
        floor_mask = load_binary_mask(
            floor_mask_path,
            expected_hw=(canvas_h, canvas_w),
        )
        for part in floor_parts:
            mask_path = contact_masks_dir / f"{slugify(part)}.png"
            save_binary_mask(mask_path, floor_mask)
            written_paths.append(mask_path)

    metadata_path = contact_masks_dir / "metadata.json"
    metadata = load_json(metadata_path)
    metadata.update(
        {
            "source_round": 1,
            "source_overlay_path": str(overlay_path),
            "source_canvas_path": str(canvas_path),
            "source_contact_spec_path": str(contact_spec_path),
            "source_summary_path": str(summary_path),
            "floor_parts": floor_parts,
            "floor_mask_path": str(floor_mask_path) if floor_parts else None,
        }
    )
    save_json(metadata_path, metadata)

    print(f"Wrote round-1 contact masks: {contact_masks_dir}")
    print(f"Wrote resized round-1 overlay: {resized_overlay_path}")
    print(f"Wrote round-1 visualization: {visualization_path}")
    for path in written_paths:
        print(f"Wrote artifact: {path}")
    return "written"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agentic_output_dir = Path(args.agentic_output_dir).resolve()
    dirs = interaction_dirs(agentic_output_dir, args.interaction_name)
    if not dirs:
        fail_or_warn(
            f"No interaction directories found under {agentic_output_dir}",
            bool(args.strict),
        )
        return 1 if args.strict else 0

    counts = {"written": 0, "skipped": 0}
    for interaction_dir in dirs:
        result = write_round1_masks(interaction_dir, args)
        counts[result] = counts.get(result, 0) + 1

    print(
        "Round-1 contact mask build complete: "
        f"{counts.get('written', 0)} written, "
        f"{counts.get('skipped', 0)} skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
