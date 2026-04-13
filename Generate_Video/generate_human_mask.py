from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_VIDEO_NAME = "video_01"
DEFAULT_IMAGE_PATH = "input/DSC08445.JPG"

# Edit this rectangle manually to choose where FLUX may generate the human.
DEFAULT_RECT_X1 = 1300
DEFAULT_RECT_Y1 = 500
DEFAULT_RECT_X2 = 1600
DEFAULT_RECT_Y2 = 1168
OVERLAY_COLOR_BGR = (0, 255, 255)


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def resolve_video_dir(script_dir: Path, video_name: str) -> Path:
    return script_dir / "output" / video_name


def resolve_human_masks_dir(script_dir: Path, video_name: str) -> Path:
    return resolve_video_dir(script_dir, video_name) / "human_masks"


def build_rect_mask(size: tuple[int, int], x1: int, y1: int, x2: int, y2: int) -> Image.Image:
    width, height = size
    left = clamp(min(x1, x2), 0, width)
    top = clamp(min(y1, y2), 0, height)
    right = clamp(max(x1, x2), 0, width)
    bottom = clamp(max(y1, y2), 0, height)

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle((left, top, right, bottom), fill=255)
    return mask


def build_overlay(image_rgb: Image.Image, mask: Image.Image) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    selected_mask = np.asarray(mask, dtype=np.uint8) > 0

    overlay = image_bgr.copy()
    color_arr = np.array(OVERLAY_COLOR_BGR, dtype=np.float32)
    overlay_f = overlay.astype(np.float32)
    overlay_f[selected_mask] = 0.55 * overlay_f[selected_mask] + 0.45 * color_arr
    overlay = np.clip(overlay_f, 0.0, 255.0).astype(np.uint8)

    ys, xs = np.where(selected_mask)
    if xs.size > 0:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        cv2.rectangle(overlay, (x0, y0), (x1, y1), OVERLAY_COLOR_BGR, 2)
        cv2.putText(
            overlay,
            "human mask",
            (x0, max(24, y0 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Create a manual rectangular human mask for FLUX fill.",
    )
    parser.add_argument("--video_name", default=DEFAULT_VIDEO_NAME)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH))
    parser.add_argument("--x1", type=int, default=DEFAULT_RECT_X1)
    parser.add_argument("--y1", type=int, default=DEFAULT_RECT_Y1)
    parser.add_argument("--x2", type=int, default=DEFAULT_RECT_X2)
    parser.add_argument("--y2", type=int, default=DEFAULT_RECT_Y2)
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    human_masks_dir = resolve_human_masks_dir(script_dir, args.video_name)
    human_masks_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    mask = build_rect_mask(
        image.size,
        x1=args.x1,
        y1=args.y1,
        x2=args.x2,
        y2=args.y2,
    )

    mask_path = human_masks_dir / "human_mask.png"
    overlay_path = human_masks_dir / "human_mask_overlay.png"

    mask.save(mask_path)
    overlay_bgr = build_overlay(image, mask)
    cv2.imwrite(str(overlay_path), overlay_bgr)

    print(f"Input image: {image_path}")
    print(f"Saved human mask: {mask_path}")
    print(f"Saved mask overlay: {overlay_path}")
    print(f"Rectangle: ({args.x1}, {args.y1}, {args.x2}, {args.y2})")


if __name__ == "__main__":
    main()
