from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


IMAGE_SOURCE_TO_IMAGE_REL_PATH: dict[str, str] = {
    "dslr_resized_undistorted": "dslr/resized_undistorted_images",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"Gemini API key file not found: {path}. "
            "Create it with a single API key line."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Gemini API key file is empty: {path}")
    return key


def resolve_scene_image_path(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> Path:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_IMAGE_REL_PATH:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_IMAGE_REL_PATH)}"
        )

    image_rel_path = IMAGE_SOURCE_TO_IMAGE_REL_PATH[camera_source]
    return scannet_root / scene_id / image_rel_path / camera_name


def extract_response_image(response: Any) -> Image.Image:
    parts = getattr(response, "parts", None)
    if parts is None and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
    if parts is None:
        raise RuntimeError("Gemini response did not contain parts.")

    for part in parts:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            try:
                image = as_image()
                if image is not None:
                    return image.convert("RGB")
            except Exception:
                pass
        inline_data = getattr(part, "inline_data", None) or getattr(
            part,
            "inlineData",
            None,
        )
        if inline_data is not None and hasattr(part, "as_image"):
            return part.as_image().convert("RGB")
    raise RuntimeError("Gemini response did not include an image.")


def build_prompt(system_prompt: str, interaction: str) -> str:
    return f"{system_prompt.strip()}\n\nInteraction:\n{interaction}"


def run_gemini_image_edit(
    api_key: str,
    model: str,
    prompt: str,
    scene_image: Image.Image,
) -> Image.Image:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, scene_image],
    )
    return extract_response_image(response)


def load_binary_mask(path: Path) -> np.ndarray:
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask > 127


def resize_cover_center_crop(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    from PIL import Image, ImageOps

    if image.size == target_size:
        return image
    return ImageOps.fit(
        image,
        target_size,
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def save_target_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    path: Path,
    color_rgb: tuple[int, int, int] = (255, 0, 0),
    alpha: int = 120,
) -> None:
    import numpy as np
    from PIL import Image

    mask_image = Image.fromarray((mask.astype(np.uint8) * 255)).convert("L")
    if mask_image.size != image.size:
        mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)

    base = image.convert("RGBA")
    transparent = Image.new("RGBA", image.size, color_rgb + (0,))
    colored = Image.new("RGBA", image.size, color_rgb + (int(alpha),))
    overlay = Image.composite(colored, transparent, mask_image)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(base, overlay).convert("RGB").save(path)
