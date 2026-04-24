from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    if not ok:
        raise IOError(f"Failed to write mask: {path}")


def load_binary_mask(
    path: Path,
    shape_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if shape_hw is not None and mask.shape != shape_hw:
        raise ValueError(
            f"Mask shape mismatch for {path}: got {mask.shape[::-1]}, "
            f"expected {shape_hw[::-1]}"
        )
    return mask > 127


def resolve_contact_root(
    script_dir: Path,
    video_name: str,
    raw_contact_root: str | None,
) -> Path:
    if raw_contact_root:
        return Path(raw_contact_root).resolve()
    return (script_dir / "output" / video_name).resolve()


def resolve_contact_overlay_path(
    contact_root: Path,
    raw_contact_overlay: str | None,
) -> Path:
    candidate = Path(raw_contact_overlay) if raw_contact_overlay else Path(
        "frame_00_contact_overlay.png"
    )
    path = candidate if candidate.is_absolute() else contact_root / "contact_overlay" / candidate
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Contact overlay does not exist: {path}")
    return path


def resolve_contact_json_path(
    contact_root: Path,
    raw_contact_json: str | None,
) -> Path:
    if raw_contact_json:
        path = Path(raw_contact_json)
        path = path if path.is_absolute() else contact_root / path
        path = path.resolve()
    else:
        path = (contact_root / "contact_mask.json").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Contact JSON does not exist: {path}")
    return path


def resolve_target_mask_path(
    script_dir: Path,
    video_name: str,
    raw_target_mask: str | None,
) -> Path:
    if raw_target_mask:
        path = Path(raw_target_mask).resolve()
    else:
        path = (
            script_dir.parent
            / "Generate_Video"
            / "output"
            / video_name
            / "first_frames_resized"
            / "frame_00_target_mask.png"
        ).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resized target mask does not exist: {path}")
    return path


def resize_and_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid source image shape: {image.shape}")

    scale = target_width / float(src_w)
    scaled_height = int(round(src_h * scale))
    if scaled_height < target_height:
        raise ValueError(
            "Scaled height is smaller than the requested crop height: "
            f"src={(src_w, src_h)}, scaled_height={scaled_height}, "
            f"target_height={target_height}"
        )

    resized = cv2.resize(
        image,
        (target_width, scaled_height),
        interpolation=interpolation,
    )
    crop_top = (scaled_height - target_height) // 2
    return resized[crop_top:crop_top + target_height, :]


def create_mask_overlay(image_bgr: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    if image_bgr.shape[:2] != object_mask.shape:
        raise ValueError(
            f"Overlay shape mismatch: image={image_bgr.shape[:2]}, "
            f"mask={object_mask.shape}"
        )

    overlay = image_bgr.astype(np.float32).copy()
    alpha = 0.55
    mask_color = np.array([0, 0, 255], dtype=np.float32)
    overlay[object_mask] = (
        (1.0 - alpha) * overlay[object_mask] + alpha * mask_color
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    value = color_hex.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {color_hex!r}")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b, g, r)


def slugify_part(human_part: str) -> str:
    return "_".join(human_part.strip().lower().split())


def classify_nearest_color(
    overlay_bgr: np.ndarray,
    bgrs: list[tuple[int, int, int]],
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.array(bgrs, dtype=np.int16)
    diff = overlay_bgr.astype(np.int16)[..., None, :] - candidates[None, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    nearest = dist.argmin(axis=-1)
    accept = dist.min(axis=-1) < max_distance
    return nearest, accept


def run(
    script_dir: Path,
    video_name: str,
    contact_root: str | None,
    contact_overlay: str | None,
    contact_json: str | None,
    target_mask: str | None,
    target_width: int,
    target_height: int,
    color_max_distance: float,
) -> dict[str, Any]:
    resolved_contact_root = resolve_contact_root(
        script_dir,
        video_name,
        contact_root,
    )
    contact_overlay_path = resolve_contact_overlay_path(
        resolved_contact_root,
        contact_overlay,
    )
    contact_json_path = resolve_contact_json_path(
        resolved_contact_root,
        contact_json,
    )
    target_mask_path = resolve_target_mask_path(
        script_dir,
        video_name,
        target_mask,
    )

    contact_payload = load_json(contact_json_path)
    contact_nodes = contact_payload.get("contact_nodes", [])
    if not contact_nodes:
        raise ValueError(f"No contact_nodes found in: {contact_json_path}")

    overlay_bgr = read_bgr(contact_overlay_path)
    target_mask_bin = load_binary_mask(target_mask_path)

    resized_overlay_bgr = resize_and_center_crop(
        overlay_bgr,
        target_width,
        target_height,
        interpolation=cv2.INTER_CUBIC,
    )
    if resized_overlay_bgr.shape[:2] != (target_height, target_width):
        raise ValueError(
            "Resized overlay shape is unexpected: "
            f"got {resized_overlay_bgr.shape[:2]}, "
            f"expected {(target_height, target_width)}"
        )
    if target_mask_bin.shape != (target_height, target_width):
        raise ValueError(
            "Target mask resolution does not match requested output size: "
            f"target_mask={target_mask_bin.shape[::-1]}, "
            f"requested={(target_width, target_height)}"
        )

    final_dir = ensure_dir(resolved_contact_root / "contact_masks")
    debug_dir = ensure_dir(resolved_contact_root / "debug")

    debug_overlay_path = debug_dir / "frame_00_contact_overlay.png"
    debug_target_mask_path = debug_dir / "frame_00_target_mask.png"
    overlay_check_path = debug_dir / "contact_overlay_target_mask_overlay.png"

    save_image(debug_overlay_path, resized_overlay_bgr)
    save_mask(debug_target_mask_path, target_mask_bin)
    save_image(
        overlay_check_path,
        create_mask_overlay(resized_overlay_bgr, target_mask_bin),
    )

    slugs: list[str] = []
    bgrs: list[tuple[int, int, int]] = []
    for node in contact_nodes:
        human_part = str(node["human_part"])
        color_hex = str(node["color_hex"])
        slug = slugify_part(human_part)
        if slug in slugs:
            raise ValueError(
                f"Duplicate human_part slug '{slug}' in contact_nodes "
                f"({contact_json_path})"
            )
        slugs.append(slug)
        bgrs.append(hex_to_bgr(color_hex))

    nearest, accept = classify_nearest_color(
        overlay_bgr,
        bgrs,
        max_distance=color_max_distance,
    )

    final_mask_paths: list[Path] = []
    for index, slug in enumerate(slugs):
        part_mask_src = (nearest == index) & accept
        part_mask_resized = resize_and_center_crop(
            part_mask_src.astype(np.uint8) * 255,
            target_width,
            target_height,
            interpolation=cv2.INTER_NEAREST,
        ) > 127

        final_mask = np.logical_and(part_mask_resized, target_mask_bin)
        final_path = final_dir / f"{slug}.png"
        save_mask(final_path, final_mask)
        final_mask_paths.append(final_path)

    return {
        "contact_overlay_path": contact_overlay_path,
        "target_mask_path": target_mask_path,
        "debug_overlay_path": debug_overlay_path,
        "debug_target_mask_path": debug_target_mask_path,
        "overlay_check_path": overlay_check_path,
        "final_mask_paths": final_mask_paths,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Resize the Estimate_Contact overlay, intersect each per-color "
            "contact region with the pre-resized target mask, and save one "
            "binary contact mask per contact_node."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument(
        "--contact-root",
        default=None,
        help="Defaults to Estimate_Contact_Region/output/<video_name>.",
    )
    parser.add_argument(
        "--contact-overlay",
        default="frame_00.png",
        help="Contact overlay filename under contact_overlay/ or an absolute path.",
    )
    parser.add_argument(
        "--contact-json",
        default=None,
        help="Defaults to <contact_root>/contact_mask.json.",
    )
    parser.add_argument(
        "--target-mask",
        default=None,
        help=(
            "Defaults to Generate_Video/output/<video_name>/"
            "first_frames_resized/frame_00_target_mask.png."
        ),
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1280,
        help="Target output width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=720,
        help="Target output height.",
    )
    parser.add_argument(
        "--color-max-distance",
        type=float,
        default=150.0,
        help=(
            "Maximum BGR Euclidean distance for a pixel to be accepted as "
            "matching its nearest contact color. Gemini's overlays have "
            "meaningful color drift, so a generous default is used."
        ),
    )
    args = parser.parse_args()

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive.")
    if args.color_max_distance <= 0:
        raise ValueError("--color-max-distance must be positive.")

    result = run(
        script_dir=script_dir,
        video_name=args.video_name,
        contact_root=args.contact_root,
        contact_overlay=args.contact_overlay,
        contact_json=args.contact_json,
        target_mask=args.target_mask,
        target_width=args.target_width,
        target_height=args.target_height,
        color_max_distance=args.color_max_distance,
    )

    print(f"Contact overlay: {result['contact_overlay_path']}")
    print(f"Target mask: {result['target_mask_path']}")
    print(f"Saved debug overlay: {result['debug_overlay_path']}")
    print(f"Saved debug target mask: {result['debug_target_mask_path']}")
    print(f"Saved overlay check: {result['overlay_check_path']}")
    for path in result["final_mask_paths"]:
        print(f"Saved contact mask: {path}")


if __name__ == "__main__":
    main()
