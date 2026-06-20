from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


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
