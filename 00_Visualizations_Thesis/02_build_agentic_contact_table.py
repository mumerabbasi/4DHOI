from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# The order and basenames used by the table in Thesis/chapters/04_experiments.tex.
DEFAULT_INTERACTIONS: tuple[tuple[str, str], ...] = (
    ("table", "02"),
    ("shower", "09"),
    ("bicycle", "22"),
    ("drawer", "23"),
    ("cupboard", "24"),
    ("pullup", "26"),
    ("bathtub", "27"),
    ("ladder", "30"),
    ("blind", "07"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the human/single-shot/agentic contact table images with "
            "one fixed-aspect, human-centered crop per interaction."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "output" / "agentic_contact",
        help="Directory receiving <basename>_{human,singleshot,agentic}.png.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCRIPT_DIR / "output" / "agentic_contact_table_crops.json",
        help="JSON file recording source rounds and crop coordinates.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        help="Output crop width:height ratio (default: 1:1).",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=768,
        help="Output width in pixels; height follows --aspect-ratio.",
    )
    parser.add_argument(
        "--extra-padding",
        type=float,
        default=0.0,
        help=(
            "Additional fractional padding around the recovered agentic crop "
            "before fitting the requested aspect ratio."
        ),
    )
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8)).save(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def parse_aspect_ratio(raw_value: str) -> float:
    parts = raw_value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Aspect ratio must have the form W:H, got {raw_value!r}")
    width, height = (float(value) for value in parts)
    if width <= 0 or height <= 0:
        raise ValueError("Aspect-ratio components must be positive")
    return width / height


def normalized_part_slug(part: str) -> str:
    words = part.lower().replace("-", " ").replace("_", " ").split()
    return "_".join(words)


def locate_exact_crop(full_image: np.ndarray, crop: np.ndarray) -> tuple[int, int]:
    """Locate an unmodified RGB crop inside its full-resolution source image."""
    full_h, full_w = full_image.shape[:2]
    crop_h, crop_w = crop.shape[:2]
    if crop_h > full_h or crop_w > full_w:
        raise ValueError(
            f"Crop {crop_w}x{crop_h} is larger than source {full_w}x{full_h}"
        )

    matches = cv2.matchTemplate(full_image, crop, cv2.TM_SQDIFF)
    _min_value, _max_value, min_location, _max_location = cv2.minMaxLoc(matches)
    x0, y0 = (int(value) for value in min_location)
    candidate = full_image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if not np.array_equal(candidate, crop):
        max_error = int(
            np.abs(candidate.astype(np.int16) - crop.astype(np.int16)).max()
        )
        raise ValueError(
            "Agentic canvas is not an exact crop of the clean scene "
            f"(best location=({x0}, {y0}), max channel error={max_error})"
        )
    return x0, y0


def fit_box_to_aspect(
    box_xyxy: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    aspect_ratio: float,
    extra_padding: float,
) -> tuple[int, int, int, int]:
    """Use the human crop's vertical extent and normalize its horizontal view.

    The agentic crop was produced from a padded person mask, so its height is a
    useful proxy for the full human extent. Keeping that height avoids clipping
    tall poses. The width is then set from the requested aspect ratio; this
    trims excessive side context from landscape crops and adds context to
    portrait crops.
    """
    x0, y0, x1, y1 = box_xyxy
    height = float(y1 - y0) * (1.0 + 2.0 * extra_padding)
    width = height * aspect_ratio
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0

    scale = min(1.0, image_width / width, image_height / height)
    width *= scale
    height *= scale
    output_width = max(1, min(image_width, int(round(width))))
    output_height = max(1, min(image_height, int(round(output_width / aspect_ratio))))
    if output_height > image_height:
        output_height = image_height
        output_width = max(1, min(image_width, int(round(output_height * aspect_ratio))))

    new_x0 = int(round(center_x - output_width / 2.0))
    new_y0 = int(round(center_y - output_height / 2.0))
    new_x0 = min(max(0, new_x0), image_width - output_width)
    new_y0 = min(max(0, new_y0), image_height - output_height)
    return new_x0, new_y0, new_x0 + output_width, new_y0 + output_height


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image)
    return np.asarray(pil_image.resize(size, resample=Image.Resampling.LANCZOS))


def crop_and_resize_rgb(
    image: np.ndarray,
    box_xyxy: tuple[int, int, int, int],
    size: tuple[int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = box_xyxy
    return resize_rgb(image[y0:y1, x0:x1], size)


def load_palette(mask_dir: Path) -> list[dict[str, Any]]:
    metadata = load_json(mask_dir / "metadata.json")
    palette = metadata.get("palette")
    if not isinstance(palette, list):
        raise ValueError(f"Missing palette list in {mask_dir / 'metadata.json'}")
    return palette


def compose_native_canvas(
    canvas: np.ndarray,
    mask_dir: Path,
) -> np.ndarray:
    composite = canvas.copy()
    for palette_item in load_palette(mask_dir):
        part = str(palette_item["part"])
        mask_path = mask_dir / f"{normalized_part_slug(part)}.png"
        mask = load_mask(mask_path)
        if mask.shape != canvas.shape[:2]:
            raise ValueError(
                f"Mask shape mismatch for {mask_path}: "
                f"mask={mask.shape[::-1]}, canvas={canvas.shape[1]}x{canvas.shape[0]}"
            )
        composite[mask] = np.asarray(palette_item["rgb"], dtype=np.uint8)
    return composite


def compose_full_scene(
    full_scene: np.ndarray,
    source_crop_xy: tuple[int, int],
    source_crop_shape: tuple[int, int],
    mask_dir: Path,
) -> np.ndarray:
    composite = full_scene.copy()
    crop_x, crop_y = source_crop_xy
    crop_h, crop_w = source_crop_shape
    for palette_item in load_palette(mask_dir):
        part = str(palette_item["part"])
        mask_path = mask_dir / f"{normalized_part_slug(part)}.png"
        mask = load_mask(mask_path)
        if mask.shape != (crop_h, crop_w):
            raise ValueError(
                f"Mask shape mismatch for {mask_path}: "
                f"mask={mask.shape[::-1]}, crop={crop_w}x{crop_h}"
            )
        region = composite[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        region[mask] = np.asarray(palette_item["rgb"], dtype=np.uint8)
    return composite


def last_round_dir(rounds_dir: Path) -> Path:
    round_dirs = sorted(path for path in rounds_dir.glob("round_*") if path.is_dir())
    if not round_dirs:
        raise FileNotFoundError(f"No round directories found in {rounds_dir}")
    return round_dirs[-1]


def verify_native_composite(
    canvas: np.ndarray,
    mask_dir: Path,
    expected_path: Path,
) -> None:
    rebuilt = compose_native_canvas(canvas, mask_dir)
    expected = load_rgb(expected_path)
    if rebuilt.shape != expected.shape or not np.array_equal(rebuilt, expected):
        changed_pixels = int(np.any(rebuilt != expected, axis=2).sum())
        raise ValueError(
            f"Stored masks do not exactly rebuild {expected_path} "
            f"({changed_pixels} differing pixels)"
        )


def process_interaction(
    basename: str,
    interaction_id: str,
    output_dir: Path,
    aspect_ratio: float,
    output_size: tuple[int, int],
    extra_padding: float,
) -> dict[str, Any]:
    interaction_name = f"interaction_{interaction_id}"
    agentic_dir = (
        PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output" / interaction_name
    )
    rounds_dir = agentic_dir / "rounds"
    round_01_dir = rounds_dir / "round_01"
    final_round_dir = last_round_dir(rounds_dir)

    full_scene_path = (
        PROJECT_DIR / "01_Generate_SIG" / "output" / interaction_name / "scene_image.png"
    )
    full_human_path = (
        PROJECT_DIR
        / "02_Generate_Human_Frame"
        / "output"
        / interaction_name
        / "inpainted_frame_resized.png"
    )
    agentic_canvas_path = agentic_dir / "assets" / "target_scene_crop.png"
    round_01_masks_dir = round_01_dir / "contact_masks"
    final_masks_dir = agentic_dir / "contact_masks"

    full_scene = load_rgb(full_scene_path)
    full_human = load_rgb(full_human_path)
    canvas = load_rgb(agentic_canvas_path)
    if full_scene.shape != full_human.shape:
        raise ValueError(
            f"Full scene/human shape mismatch for {interaction_name}: "
            f"scene={full_scene.shape}, human={full_human.shape}"
        )

    crop_x, crop_y = locate_exact_crop(full_scene, canvas)
    crop_h, crop_w = canvas.shape[:2]
    source_box = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
    output_box = fit_box_to_aspect(
        source_box,
        image_width=full_scene.shape[1],
        image_height=full_scene.shape[0],
        aspect_ratio=aspect_ratio,
        extra_padding=extra_padding,
    )

    # Confirm that the stored masks/palette reproduce the original composites
    # before moving them into the full-image coordinate system.
    verify_native_composite(canvas, round_01_masks_dir, round_01_dir / "composite.png")
    verify_native_composite(canvas, final_masks_dir, final_round_dir / "composite.png")

    single_shot_full = compose_full_scene(
        full_scene,
        (crop_x, crop_y),
        (crop_h, crop_w),
        round_01_masks_dir,
    )
    agentic_full = compose_full_scene(
        full_scene,
        (crop_x, crop_y),
        (crop_h, crop_w),
        final_masks_dir,
    )

    outputs = {
        "human": output_dir / f"{basename}_human.png",
        "singleshot": output_dir / f"{basename}_singleshot.png",
        "agentic": output_dir / f"{basename}_agentic.png",
    }
    save_rgb(outputs["human"], crop_and_resize_rgb(full_human, output_box, output_size))
    save_rgb(
        outputs["singleshot"],
        crop_and_resize_rgb(single_shot_full, output_box, output_size),
    )
    save_rgb(
        outputs["agentic"],
        crop_and_resize_rgb(agentic_full, output_box, output_size),
    )

    return {
        "basename": basename,
        "interaction": interaction_name,
        "single_shot_round": round_01_dir.name,
        "agentic_round": final_round_dir.name,
        "source_size": [int(full_scene.shape[1]), int(full_scene.shape[0])],
        "agentic_source_crop_xyxy": list(source_box),
        "table_crop_xyxy": list(output_box),
        "output_size": list(output_size),
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
    }


def main() -> None:
    args = parse_args()
    if args.output_size <= 0:
        raise ValueError("--output-size must be positive")
    if args.extra_padding < 0:
        raise ValueError("--extra-padding must be non-negative")

    aspect_ratio = parse_aspect_ratio(args.aspect_ratio)
    output_width = int(args.output_size)
    output_height = max(1, int(round(output_width / aspect_ratio)))
    output_size = (output_width, output_height)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        process_interaction(
            basename=basename,
            interaction_id=interaction_id,
            output_dir=output_dir,
            aspect_ratio=aspect_ratio,
            output_size=output_size,
            extra_padding=float(args.extra_padding),
        )
        for basename, interaction_id in DEFAULT_INTERACTIONS
    ]

    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "aspect_ratio": args.aspect_ratio,
                "output_size": list(output_size),
                "interactions": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records) * 3} table images to {output_dir}")
    print(f"Wrote crop manifest to {manifest_path}")


if __name__ == "__main__":
    main()
