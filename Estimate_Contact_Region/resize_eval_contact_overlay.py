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


def resolve_selection_path(
    script_dir: Path,
    video_name: str,
    raw_selection_json: str | None,
) -> Path:
    if raw_selection_json:
        return Path(raw_selection_json).resolve()
    return (
        script_dir.parent
        / "Select_Target_Instance"
        / "output"
        / video_name
        / "target_selection.json"
    ).resolve()


def resolve_contact_overlay_path(
    contact_root: Path,
    raw_contact_overlay: str | None,
) -> Path:
    candidate = Path(raw_contact_overlay) if raw_contact_overlay else Path(
        "frame_00_contact_overlay.png"
    )
    path = candidate if candidate.is_absolute() else contact_root / "first_frames" / candidate
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Contact overlay does not exist: {path}")
    return path


def resolve_optional_contact_mask_path(
    contact_root: Path,
    raw_contact_mask: str | None,
) -> Path | None:
    if raw_contact_mask:
        path = Path(raw_contact_mask)
        path = path if path.is_absolute() else contact_root / "first_frames" / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Contact mask does not exist: {path}")
        return path

    path = (contact_root / "first_frames" / "frame_00_contact_mask.png").resolve()
    return path if path.exists() else None


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


def resize_and_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> tuple[np.ndarray, dict[str, Any]]:
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
    cropped = resized[crop_top:crop_top + target_height, :]

    transform = {
        "source_width": int(src_w),
        "source_height": int(src_h),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "scale": float(scale),
        "scaled_height": int(scaled_height),
        "crop_top": int(crop_top),
    }
    return cropped, transform


def assert_same_aspect_ratio(
    first_shape_hw: tuple[int, int],
    second_shape_hw: tuple[int, int],
    first_label: str,
    second_label: str,
    tolerance: float = 0.005,
) -> None:
    first_h, first_w = first_shape_hw
    second_h, second_w = second_shape_hw
    first_ratio = first_w / float(first_h)
    second_ratio = second_w / float(second_h)
    if abs(first_ratio - second_ratio) > tolerance:
        raise ValueError(
            f"{first_label} and {second_label} must have matching aspect ratios: "
            f"{first_label}={(first_w, first_h)} ratio={first_ratio:.6f}, "
            f"{second_label}={(second_w, second_h)} ratio={second_ratio:.6f}"
        )


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


def run_eval(
    script_dir: Path,
    video_name: str,
    contact_root: str | None,
    contact_overlay: str | None,
    contact_mask: str | None,
    selection_json: str | None,
    target_width: int,
    target_height: int,
) -> dict[str, Path | None]:
    resolved_contact_root = resolve_contact_root(
        script_dir,
        video_name,
        contact_root,
    )
    contact_overlay_path = resolve_contact_overlay_path(
        resolved_contact_root,
        contact_overlay,
    )
    contact_mask_path = resolve_optional_contact_mask_path(
        resolved_contact_root,
        contact_mask,
    )

    selection_path = resolve_selection_path(
        script_dir,
        video_name,
        selection_json,
    )
    selection_root = selection_path.parent
    selection_payload = load_json(selection_path)
    scene_image_path = resolve_scene_image_path(selection_root)
    object_mask_path = resolve_object_mask_path(selection_root, selection_payload)

    scene_bgr = read_bgr(scene_image_path)
    contact_overlay_bgr = read_bgr(contact_overlay_path)
    assert_same_aspect_ratio(
        contact_overlay_bgr.shape[:2],
        scene_bgr.shape[:2],
        "contact overlay",
        "selected scene image",
    )

    object_mask = load_binary_mask(
        object_mask_path,
        shape_hw=scene_bgr.shape[:2],
    )

    resized_contact_overlay_bgr, overlay_transform = resize_and_center_crop(
        contact_overlay_bgr,
        target_width,
        target_height,
        interpolation=cv2.INTER_CUBIC,
    )
    resized_mask_u8, mask_transform = resize_and_center_crop(
        object_mask.astype(np.uint8) * 255,
        target_width,
        target_height,
        interpolation=cv2.INTER_NEAREST,
    )
    if (
        overlay_transform["crop_top"] != mask_transform["crop_top"]
        or overlay_transform["scaled_height"] != mask_transform["scaled_height"]
    ):
        raise ValueError("Contact overlay and target mask transforms diverged.")

    resized_mask = resized_mask_u8 > 127
    resized_dir = ensure_dir(resolved_contact_root / "first_frames_resized")
    debug_dir = ensure_dir(resolved_contact_root / "debug")

    resized_contact_overlay_path = resized_dir / contact_overlay_path.name
    target_mask_path = resized_dir / "target_mask.png"
    overlay_check_path = debug_dir / "contact_overlay_target_mask_overlay.png"

    save_image(resized_contact_overlay_path, resized_contact_overlay_bgr)
    save_mask(target_mask_path, resized_mask)
    save_image(
        overlay_check_path,
        create_mask_overlay(resized_contact_overlay_bgr, resized_mask),
    )

    resized_contact_mask_path: Path | None = None
    if contact_mask_path is not None:
        contact_mask_bgr = read_bgr(contact_mask_path)
        assert_same_aspect_ratio(
            contact_mask_bgr.shape[:2],
            contact_overlay_bgr.shape[:2],
            "contact mask",
            "contact overlay",
        )
        resized_contact_mask_bgr, _ = resize_and_center_crop(
            contact_mask_bgr,
            target_width,
            target_height,
            interpolation=cv2.INTER_NEAREST,
        )
        resized_contact_mask_path = resized_dir / contact_mask_path.name
        save_image(resized_contact_mask_path, resized_contact_mask_bgr)

    return {
        "contact_overlay_path": contact_overlay_path,
        "resized_contact_overlay_path": resized_contact_overlay_path,
        "target_mask_path": target_mask_path,
        "overlay_check_path": overlay_check_path,
        "contact_mask_path": contact_mask_path,
        "resized_contact_mask_path": resized_contact_mask_path,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Resize Estimate_Contact overlays and evaluate them against the "
            "selected target mask."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument(
        "--contact-root",
        default=None,
        help="Defaults to Estimate_Contact/output/<video_name>.",
    )
    parser.add_argument(
        "--contact-overlay",
        default="frame_00_contact_overlay.png",
        help="Contact overlay filename under first_frames/ or an absolute path.",
    )
    parser.add_argument(
        "--contact-mask",
        default=None,
        help=(
            "Optional contact mask filename under first_frames/ or an absolute "
            "path. If omitted, frame_00_contact_mask.png is used when present."
        ),
    )
    parser.add_argument(
        "--selection-json",
        default=None,
        help="Path to target_selection.json.",
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
    args = parser.parse_args()

    if args.target_width <= 0 or args.target_height <= 0:
        raise ValueError("--target-width and --target-height must be positive.")

    result = run_eval(
        script_dir=script_dir,
        video_name=args.video_name,
        contact_root=args.contact_root,
        contact_overlay=args.contact_overlay,
        contact_mask=args.contact_mask,
        selection_json=args.selection_json,
        target_width=args.target_width,
        target_height=args.target_height,
    )

    print(f"Contact overlay: {result['contact_overlay_path']}")
    print(f"Saved resized contact overlay: {result['resized_contact_overlay_path']}")
    print(f"Saved resized target mask: {result['target_mask_path']}")
    print(f"Saved contact overlay check: {result['overlay_check_path']}")
    if result["resized_contact_mask_path"] is not None:
        print(f"Saved resized contact mask: {result['resized_contact_mask_path']}")
    else:
        print("No contact mask found; skipped contact mask resize.")


if __name__ == "__main__":
    main()
