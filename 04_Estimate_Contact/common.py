from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image


BODY_PART_COLOR_MAP: dict[str, tuple[str, str]] = {
    "left hand": ("#FF00FF", "pure magenta"),
    "right hand": ("#00FF00", "pure lime green"),
    "left arm": ("#00FFFF", "bright cyan"),
    "right arm": ("#FF8C00", "bright orange"),
    "left shoulder": ("#FFD700", "bright gold"),
    "right shoulder": ("#1E90FF", "bright dodger blue"),
    "left leg": ("#FF1493", "bright deep pink"),
    "right leg": ("#7FFF00", "bright chartreuse"),
    "left foot": ("#00BFFF", "bright sky blue"),
    "right foot": ("#FF4500", "bright orange red"),
    "head": ("#FFFF00", "pure yellow"),
    "hips": ("#9400D3", "bright violet"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def normalize_label(text: str) -> str:
    return " ".join(
        text.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def normalize_scene_element(text: str) -> str:
    raw = str(text).strip().lower()
    normalized = normalize_label(text)
    if raw == "target_object" or normalized in {"target object", "object"}:
        return "target_object"
    return normalized


def slugify(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def color_for_part(part_name: str) -> tuple[str, str]:
    normalized = normalize_label(part_name)
    if normalized not in BODY_PART_COLOR_MAP:
        raise KeyError(
            f"Unsupported human body part for contact color: '{part_name}'. "
            f"Supported parts: {sorted(BODY_PART_COLOR_MAP)}"
        )
    return BODY_PART_COLOR_MAP[normalized]


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.strip().lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {hex_color}")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def title_label(text: str) -> str:
    return normalize_label(text).title()


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


def target_object_human_parts(sig_payload: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()
    for edge in sig_payload.get("interaction_edges", []):
        if not isinstance(edge, dict):
            continue
        scene_element = normalize_scene_element(
            str(edge.get("scene_element", ""))
        )
        if scene_element != "target_object":
            continue
        part = normalize_label(str(edge.get("human_part", "")))
        if not part or part in seen:
            continue
        color_for_part(part)
        seen.add(part)
        parts.append(part)
    if not parts:
        raise ValueError(
            "No target_object contact edges found in interaction_edges."
        )
    return parts


def build_contact_prompt(system_prompt: str, human_parts: list[str]) -> str:
    color_lines = []
    for part in human_parts:
        _hex_color, color_name = color_for_part(part)
        color_lines.append(
            f"{title_label(part)}: Mark this contact region on the object "
            f"surface using a precise {color_name} overlay."
        )
    return (
        f"{system_prompt.strip()}\n\n"
        "Color Mapping for Segmentation Overlays:\n"
        + "\n".join(color_lines)
    )


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


def run_gemini_image_edit(
    api_key: str,
    model: str,
    prompt: str,
    scene_image: Image.Image,
    target_render_image: Image.Image,
) -> Image.Image:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, scene_image, target_render_image],
    )
    return extract_response_image(response)


def resize_cover_center_crop_array(
    image_rgb: Any,
    target_shape: tuple[int, int],
) -> Any:
    import numpy as np
    from PIL import Image, ImageOps

    target_h, target_w = target_shape
    if image_rgb.shape[:2] == (target_h, target_w):
        return image_rgb

    image = Image.fromarray(image_rgb)
    resized = ImageOps.fit(
        image,
        (target_w, target_h),
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )
    return np.asarray(resized.convert("RGB"))


def resize_cover_center_crop_mask(
    mask: Any,
    target_shape: tuple[int, int],
) -> Any:
    import numpy as np
    from PIL import Image, ImageOps

    target_h, target_w = target_shape
    if mask.shape[:2] == (target_h, target_w):
        return mask.astype(bool)

    image = Image.fromarray(mask.astype(np.uint8) * 255).convert("L")
    resized = ImageOps.fit(
        image,
        (target_w, target_h),
        method=Image.Resampling.NEAREST,
        centering=(0.5, 0.5),
    )
    return np.asarray(resized) > 127


def keep_largest_components(
    mask: Any,
    min_area: int,
    keep_components: int,
) -> Any:
    import cv2
    import numpy as np

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    components: list[tuple[int, int]] = []
    for label_id in range(1, labels_count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            components.append((area, label_id))

    components.sort(reverse=True)
    if keep_components > 0:
        components = components[: int(keep_components)]

    cleaned = np.zeros(mask.shape, dtype=bool)
    for _area, label_id in components:
        cleaned |= labels == label_id
    return cleaned


def erode_binary_mask(mask: Any, erode_pixels: int) -> Any:
    if erode_pixels <= 0:
        return mask

    import cv2

    kernel_size = 2 * int(erode_pixels) + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    eroded = cv2.erode(mask.astype("uint8"), kernel, iterations=1) > 0
    if not eroded.any():
        return mask
    return eroded


def classify_nearest_color(
    overlay_rgb: Any,
    target_colors_rgb: list[tuple[int, int, int]],
    color_max_distance: float,
) -> tuple[Any, Any]:
    import numpy as np

    candidates = np.asarray(target_colors_rgb, dtype=np.int16)
    diff = (
        overlay_rgb.astype(np.int16)[..., None, :]
        - candidates[None, None, :, :]
    )
    distances = np.linalg.norm(diff, axis=-1)
    nearest = distances.argmin(axis=-1)
    accept = distances.min(axis=-1) <= float(color_max_distance)
    return nearest, accept


def save_contact_masks_from_overlay(
    overlay_path: Path,
    target_render_path: Path,
    contact_masks_dir: Path,
    human_parts: list[str],
    color_max_distance: float = 90.0,
    min_component_area: int = 50,
    keep_components: int = 1,
    erode_pixels: int = 1,
) -> list[Path]:
    import cv2

    target_bgr = cv2.imread(str(target_render_path), cv2.IMREAD_COLOR)
    if target_bgr is None:
        raise FileNotFoundError(
            f"Failed to read target render: {target_render_path}"
        )
    overlay_bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if overlay_bgr is None:
        raise FileNotFoundError(
            f"Failed to read contact overlay: {overlay_path}"
        )

    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    target_colors = [
        hex_to_rgb(color_for_part(part)[0])
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
        mask_at_overlay_size = (
            (nearest_color == part_idx)
            & color_accept
        )
        mask = resize_cover_center_crop_mask(
            mask_at_overlay_size,
            target_rgb.shape[:2],
        )
        mask = keep_largest_components(
            mask,
            min_area=min_component_area,
            keep_components=keep_components,
        )
        mask = erode_binary_mask(mask, erode_pixels=erode_pixels)
        mask_path = contact_masks_dir / f"{slugify(part)}.png"
        cv2.imwrite(str(mask_path), mask.astype("uint8") * 255)
        written_paths.append(mask_path)

    metadata_path = contact_masks_dir / "metadata.json"
    save_json(
        metadata_path,
        {
            "overlay": str(overlay_path),
            "target_render": str(target_render_path),
            "human_parts": human_parts,
            "mask_shape_hw": list(target_rgb.shape[:2]),
            "color_max_distance": float(color_max_distance),
            "min_component_area": int(min_component_area),
            "keep_components": int(keep_components),
            "erode_pixels": int(erode_pixels),
        },
    )
    written_paths.append(metadata_path)
    return written_paths
