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
    load_json,
    save_contact_masks_from_overlay,
    save_json,
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
    resized_overlay_path = interaction_dir / "contact_overlay_gt_resized.png"
    visualization_path = interaction_dir / "contact_overlay_gt_visualization.png"
    contact_masks_dir = interaction_dir / str(args.masks_dir_name)
    canvas_path = interaction_dir / "assets" / "target_scene_crop.png"
    contact_spec_path = interaction_dir / "contact_spec.json"

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

    contact_spec = load_json(contact_spec_path)
    human_parts = palette_parts(contact_spec, contact_spec_path)
    palette = contact_palette_from_spec(contact_spec_path, human_parts)
    written_paths = save_contact_masks_from_overlay(
        overlay_path=overlay_path,
        resized_overlay_path=resized_overlay_path,
        canvas_path=canvas_path,
        contact_masks_dir=contact_masks_dir,
        human_parts=human_parts,
        palette=palette,
        visualization_path=visualization_path,
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
        }
    )
    save_json(metadata_path, metadata)

    print(f"Wrote GT contact masks: {contact_masks_dir}")
    print(f"Wrote resized GT overlay: {resized_overlay_path}")
    print(f"Wrote GT visualization: {visualization_path}")
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
