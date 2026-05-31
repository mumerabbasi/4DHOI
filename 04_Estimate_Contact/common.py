from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
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

CONTACT_COLOR_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("#FF00FF", "pure magenta"),
    ("#00FF00", "pure green"),
    ("#0000FF", "pure blue"),
    ("#FFFF00", "pure yellow"),
    ("#FF0000", "pure red"),
    ("#00FFFF", "pure cyan"),
    ("#FF8000", "vivid orange"),
    ("#8000FF", "electric violet"),
    ("#00FF80", "spring green"),
    ("#FF0080", "hot rose"),
    ("#0080FF", "azure blue"),
    ("#80FF00", "chartreuse green"),
)

SAM3_CHECKPOINT = None
SAM3_BPE_PATH = "/my_workspace/4DHHOI/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

IMAGE_SOURCE_TO_TRANSFORMS_REL_PATH: dict[str, str] = {
    "dslr_resized_undistorted": "dslr/nerfstudio/transforms_undistorted.json",
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


def rgb_to_hex(color_rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*color_rgb)


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


def contact_human_parts(
    sig_payload: dict[str, Any],
    scene_element: str | None = None,
) -> list[str]:
    parts: list[str] = []
    seen: set[str] = set()
    requested_scene_element = (
        normalize_scene_element(scene_element)
        if scene_element is not None
        else None
    )
    for edge in sig_payload.get("interaction_edges", []):
        if not isinstance(edge, dict):
            continue
        edge_scene_element = normalize_scene_element(
            str(edge.get("scene_element", ""))
        )
        if (
            requested_scene_element is not None
            and edge_scene_element != requested_scene_element
        ):
            continue
        part = normalize_label(str(edge.get("human_part", "")))
        if not part or part in seen:
            continue
        color_for_part(part)
        seen.add(part)
        parts.append(part)
    return parts


def target_object_human_parts(sig_payload: dict[str, Any]) -> list[str]:
    parts = contact_human_parts(sig_payload, scene_element="target_object")
    if not parts:
        raise ValueError(
            "No target_object contact edges found in interaction_edges."
        )
    return parts


def floor_contact_human_parts(sig_payload: dict[str, Any]) -> list[str]:
    return contact_human_parts(sig_payload, scene_element="floor")


def all_contact_human_parts(sig_payload: dict[str, Any]) -> list[str]:
    return contact_human_parts(sig_payload, scene_element=None)


def build_static_contact_palette(human_parts: list[str]) -> list[dict[str, Any]]:
    palette: list[dict[str, Any]] = []
    for part in human_parts:
        hex_color, color_name = color_for_part(part)
        palette.append(
            {
                "part": normalize_label(part),
                "label": title_label(part),
                "hex": hex_color.upper(),
                "rgb": list(hex_to_rgb(hex_color)),
                "color_name": color_name,
            }
        )
    return palette


def _scene_color_distance_score(
    candidate_rgb: tuple[int, int, int],
    scene_rgb: np.ndarray,
    avoid_mask: np.ndarray | None,
) -> float:
    import numpy as np

    if avoid_mask is not None and np.any(avoid_mask):
        pixels = scene_rgb[avoid_mask.astype(bool)]
    else:
        pixels = scene_rgb.reshape(-1, 3)
    if pixels.size == 0:
        return 441.0
    step = max(1, int(pixels.shape[0] // 50000))
    sampled = pixels[::step].astype(np.float32)
    candidate = np.asarray(candidate_rgb, dtype=np.float32)
    distances = np.linalg.norm(sampled - candidate[None, :], axis=1)
    return float(np.percentile(distances, 5))


def choose_contact_palette(
    human_parts: list[str],
    scene_rgb: np.ndarray,
    avoid_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    import numpy as np

    for part in human_parts:
        color_for_part(part)

    candidates = [
        (hex_color.upper(), name, hex_to_rgb(hex_color))
        for hex_color, name in CONTACT_COLOR_CANDIDATES
    ]
    if len(human_parts) > len(candidates):
        raise ValueError(
            f"Need {len(human_parts)} contact colors, but only "
            f"{len(candidates)} candidates are configured."
        )

    scene_scores = {
        hex_color: _scene_color_distance_score(color_rgb, scene_rgb, avoid_mask)
        for hex_color, _name, color_rgb in candidates
    }
    selected: list[tuple[str, str, tuple[int, int, int]]] = []
    available = candidates[:]
    for _part in human_parts:
        best_idx = 0
        best_score: tuple[float, float, float] | None = None
        for idx, (hex_color, _name, color_rgb) in enumerate(available):
            if selected:
                pair_distance = min(
                    float(
                        np.linalg.norm(
                            np.asarray(color_rgb, dtype=np.float32)
                            - np.asarray(selected_rgb, dtype=np.float32)
                        )
                    )
                    for _selected_hex, _selected_name, selected_rgb in selected
                )
            else:
                pair_distance = 441.0
            scene_distance = scene_scores[hex_color]
            score = (
                min(pair_distance, scene_distance),
                pair_distance,
                scene_distance,
            )
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        selected.append(available.pop(best_idx))

    palette: list[dict[str, Any]] = []
    for part, (hex_color, color_name, color_rgb) in zip(human_parts, selected):
        palette.append(
            {
                "part": normalize_label(part),
                "label": title_label(part),
                "hex": hex_color,
                "rgb": list(color_rgb),
                "color_name": color_name,
            }
        )
    return palette


def build_contact_spec(
    intrinsics_3x3: list[list[float]],
    palette: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "camera": {
            "intrinsics_3x3": intrinsics_3x3,
        },
        "palette": {
            "parts": palette,
        },
    }


def save_contact_spec(
    path: Path,
    intrinsics_3x3: list[list[float]],
    palette: list[dict[str, Any]],
) -> None:
    save_json(path, build_contact_spec(intrinsics_3x3, palette))


def load_contact_spec(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Contact spec JSON not found: {path}")
    payload = load_json(path)
    if not isinstance(payload.get("camera"), dict):
        raise ValueError(f"Contact spec missing camera object: {path}")
    if not isinstance(payload.get("palette"), dict):
        raise ValueError(f"Contact spec missing palette object: {path}")
    return payload


def contact_palette_from_spec(
    path: Path,
    human_parts: list[str],
) -> list[dict[str, Any]]:
    payload = load_contact_spec(path)
    raw_parts = payload["palette"].get("parts")
    if not isinstance(raw_parts, list):
        raise ValueError(f"Contact spec palette missing parts list: {path}")

    by_part: dict[str, dict[str, Any]] = {}
    for item in raw_parts:
        if not isinstance(item, dict):
            continue
        part = normalize_label(str(item.get("part", "")))
        hex_color = str(item.get("hex", "")).upper()
        color_rgb = hex_to_rgb(hex_color)
        by_part[part] = {
            "part": part,
            "label": str(item.get("label") or title_label(part)),
            "hex": rgb_to_hex(color_rgb),
            "rgb": list(color_rgb),
            "color_name": str(item.get("color_name") or hex_color),
        }

    palette: list[dict[str, Any]] = []
    for part in human_parts:
        normalized = normalize_label(part)
        if normalized not in by_part:
            raise ValueError(
                f"Contact palette {path} does not define a color for "
                f"'{normalized}'."
            )
        palette.append(by_part[normalized])
    return palette


def build_contact_prompt(
    system_prompt: str,
    human_parts: list[str],
    palette: list[dict[str, Any]] | None = None,
) -> str:
    if palette is None:
        palette = build_static_contact_palette(human_parts)
    palette_by_part = {
        normalize_label(str(item["part"])): item
        for item in palette
    }
    color_lines = []
    for part in human_parts:
        part_label = normalize_label(part)
        color_info = palette_by_part[part_label]
        rgb = color_info["rgb"]
        color_lines.append(
            f"{title_label(part)}: The {part_label} is in contact with the "
            f"target object. Copy the visible {part_label} mask shape from "
            f"the Reference Image onto the matching location in the returned "
            f"overlay image using solid "
            f"{color_info['hex']} / RGB({rgb[0]}, {rgb[1]}, {rgb[2]}) "
            f"({color_info['color_name']})."
        )
    return (
        f"{system_prompt.strip()}\n\n"
        "Color Mapping for Segmentation Masks:\n"
        + "\n".join(color_lines)
    )


def resolve_scannet_root(project_dir: Path, raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (project_dir.parent / "Scannet++" / "data").resolve()


def resolve_transforms_path(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> Path:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    if camera_source not in IMAGE_SOURCE_TO_TRANSFORMS_REL_PATH:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_TRANSFORMS_REL_PATH)}"
        )
    return (
        scannet_root
        / scene_id
        / IMAGE_SOURCE_TO_TRANSFORMS_REL_PATH[camera_source]
    )


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[list[list[float]], int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = [
        [float(transforms_payload["fl_x"]), 0.0, float(transforms_payload["cx"])],
        [0.0, float(transforms_payload["fl_y"]), float(transforms_payload["cy"])],
        [0.0, 0.0, 1.0],
    ]
    return intrinsics, width, height


def load_rgb(path: Path) -> np.ndarray:
    import cv2

    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def load_binary_mask(path: Path, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    import cv2

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if expected_hw is not None and mask.shape != expected_hw:
        raise ValueError(
            f"Mask shape mismatch at {path}: got {mask.shape[::-1]}, "
            f"expected {expected_hw[::-1]}"
        )
    return mask > 127


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype("uint8") * 255)


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    import numpy as np

    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def union_bboxes(bboxes: list[list[int] | None]) -> list[int]:
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        raise ValueError("Cannot compute a crop from empty bounding boxes.")
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def pad_bbox(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding_frac: float,
) -> list[int]:
    import math

    x0, y0, x1, y1 = bbox
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    pad_x = int(math.ceil(bbox_w * max(0.0, float(padding_frac))))
    pad_y = int(math.ceil(bbox_h * max(0.0, float(padding_frac))))
    return [
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(int(image_width), x1 + pad_x),
        min(int(image_height), y1 + pad_y),
    ]


def crop_array(image: np.ndarray, crop_xyxy: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = crop_xyxy
    return image[y0:y1, x0:x1].copy()


def get_default_sam3_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_sam3_processor(
    checkpoint_path: Path | None,
    bpe_path: Path | None,
    device: str,
    confidence_threshold: float,
    allow_hf_download: bool,
) -> Any:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    resolved_checkpoint_path = (
        str(checkpoint_path) if checkpoint_path is not None else SAM3_CHECKPOINT
    )
    resolved_bpe_path = str(bpe_path) if bpe_path is not None else SAM3_BPE_PATH

    model = build_sam3_image_model(
        bpe_path=resolved_bpe_path,
        checkpoint_path=resolved_checkpoint_path,
        device=device,
        load_from_HF=allow_hf_download,
    )
    return Sam3Processor(
        model=model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def run_sam3_text_prompt(
    processor: Any,
    image_rgb: Image.Image,
    prompt: str,
) -> list[dict[str, Any]]:
    import numpy as np

    predictions: list[dict[str, Any]] = []
    state = processor.set_image(image_rgb)
    state = processor.set_text_prompt(prompt=prompt, state=state)

    mask_tensor = state["masks"]
    box_tensor = state["boxes"]
    score_tensor = state["scores"]

    for mask_index in range(mask_tensor.shape[0]):
        mask = mask_tensor[mask_index, 0].detach().cpu().numpy().astype(bool)
        if not np.any(mask):
            continue
        predictions.append(
            {
                "prompt": prompt,
                "mask_index": int(mask_index),
                "mask": mask,
                "bbox_xyxy": [
                    float(value)
                    for value in box_tensor[mask_index].detach().cpu().numpy().tolist()
                ],
                "sam3_score": float(score_tensor[mask_index].detach().cpu().item()),
            }
        )
    return predictions


def select_highest_confidence_mask(
    sam3_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    if not sam3_predictions:
        raise ValueError("SAM3 did not return any masks.")
    return max(
        sam3_predictions,
        key=lambda item: (
            float(item["sam3_score"]),
            -int(np.count_nonzero(item["mask"])),
        ),
    )


def adjusted_intrinsics_for_crop(
    intrinsics: list[list[float]],
    crop_xyxy: list[int],
) -> list[list[float]]:
    x0, y0, _x1, _y1 = crop_xyxy
    adjusted = [row[:] for row in intrinsics]
    adjusted[0][2] = float(adjusted[0][2]) - float(x0)
    adjusted[1][2] = float(adjusted[1][2]) - float(y0)
    return adjusted


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
    reference_image: Image.Image,
    canvas_image: Image.Image,
) -> Image.Image:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, reference_image, canvas_image],
    )
    return extract_response_image(response)


def resize_cover_center_crop_array(
    image_rgb: np.ndarray,
    target_shape: tuple[int, int],
    resampling: int | None = None,
) -> np.ndarray:
    import numpy as np
    from PIL import Image, ImageOps

    if resampling is None:
        resampling = Image.Resampling.BICUBIC
    target_h, target_w = target_shape
    if image_rgb.shape[:2] == (target_h, target_w):
        return image_rgb

    image = Image.fromarray(image_rgb)
    resized = ImageOps.fit(
        image,
        (target_w, target_h),
        method=resampling,
        centering=(0.5, 0.5),
    )
    return np.asarray(resized.convert("RGB"))


def resize_cover_center_crop_mask(
    mask: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
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


def normalize_overlay_to_canvas(
    overlay_path: Path,
    canvas_path: Path,
    resized_overlay_path: Path,
) -> tuple[np.ndarray, tuple[int, int]]:
    from PIL import Image

    canvas_rgb = load_rgb(canvas_path)
    overlay_rgb = load_rgb(overlay_path)
    normalized = resize_cover_center_crop_array(
        overlay_rgb,
        canvas_rgb.shape[:2],
        resampling=Image.Resampling.NEAREST,
    )
    save_rgb(resized_overlay_path, normalized)
    return normalized, canvas_rgb.shape[:2]


def keep_largest_components(
    mask: np.ndarray,
    min_area: int,
    keep_components: int,
) -> np.ndarray:
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


def erode_binary_mask(mask: np.ndarray, erode_pixels: int) -> np.ndarray:
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
    overlay_rgb: np.ndarray,
    target_colors_rgb: list[tuple[int, int, int]],
    color_max_distance: float,
    color_hue_tolerance: float = 12.0,
    min_color_saturation: int = 80,
    min_color_value: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    import numpy as np

    candidates = np.asarray(target_colors_rgb, dtype=np.int16)
    diff = (
        overlay_rgb.astype(np.int16)[..., None, :]
        - candidates[None, None, :, :]
    )
    distances = np.linalg.norm(diff, axis=-1)
    nearest_rgb = distances.argmin(axis=-1)
    rgb_accept = distances.min(axis=-1) <= float(color_max_distance)

    overlay_hsv = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2HSV)
    candidate_rgb = np.asarray(target_colors_rgb, dtype=np.uint8).reshape(-1, 1, 3)
    candidate_hsv = cv2.cvtColor(candidate_rgb, cv2.COLOR_RGB2HSV)[:, 0, :]
    hue_diff = np.abs(
        overlay_hsv[..., 0].astype(np.int16)[..., None]
        - candidate_hsv[:, 0].astype(np.int16)[None, None, :]
    )
    hue_distance = np.minimum(hue_diff, 180 - hue_diff)
    nearest_hue = hue_distance.argmin(axis=-1)
    hue_accept = (
        (hue_distance.min(axis=-1) <= float(color_hue_tolerance))
        & (overlay_hsv[..., 1] >= int(min_color_saturation))
        & (overlay_hsv[..., 2] >= int(min_color_value))
    )

    nearest = nearest_rgb.copy()
    hue_only = ~rgb_accept & hue_accept
    nearest[hue_only] = nearest_hue[hue_only]
    accept = rgb_accept | hue_accept
    return nearest, accept


def save_contact_visualization(
    canvas_rgb: np.ndarray,
    masks_by_part: list[tuple[str, np.ndarray]],
    palette: list[dict[str, Any]],
    visualization_path: Path,
    alpha: float = 0.65,
) -> None:
    import numpy as np

    palette_by_part = {
        normalize_label(str(item["part"])): tuple(int(v) for v in item["rgb"])
        for item in palette
    }
    visual = canvas_rgb.astype(np.float32).copy()
    blend_alpha = min(1.0, max(0.0, float(alpha)))
    for part, mask in masks_by_part:
        color = np.asarray(palette_by_part[normalize_label(part)], dtype=np.float32)
        visual[mask] = (
            visual[mask] * (1.0 - blend_alpha)
            + color[None, :] * blend_alpha
        )
    save_rgb(visualization_path, np.clip(visual, 0, 255).astype(np.uint8))


def save_contact_masks_from_overlay(
    overlay_path: Path,
    resized_overlay_path: Path,
    canvas_path: Path,
    target_mask_crop_path: Path,
    contact_masks_dir: Path,
    human_parts: list[str],
    palette: list[dict[str, Any]] | None = None,
    visualization_path: Path | None = None,
    color_max_distance: float = 90.0,
    color_hue_tolerance: float = 12.0,
    min_color_saturation: int = 80,
    min_color_value: int = 40,
    min_component_area: int = 0,
    keep_components: int = 3,
    target_mask_erode_pixels: int = 2,
) -> list[Path]:
    overlay_rgb, canvas_hw = normalize_overlay_to_canvas(
        overlay_path=overlay_path,
        canvas_path=canvas_path,
        resized_overlay_path=resized_overlay_path,
    )
    canvas_rgb = load_rgb(canvas_path)
    target_mask = load_binary_mask(target_mask_crop_path, expected_hw=canvas_hw)
    target_mask = erode_binary_mask(target_mask, target_mask_erode_pixels)

    if palette is None:
        palette = build_static_contact_palette(human_parts)
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
        color_hue_tolerance=color_hue_tolerance,
        min_color_saturation=min_color_saturation,
        min_color_value=min_color_value,
    )

    contact_masks_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    masks_by_part: list[tuple[str, np.ndarray]] = []
    for part_idx, part in enumerate(human_parts):
        mask = (
            (nearest_color == part_idx)
            & color_accept
            & target_mask
        )
        mask = keep_largest_components(
            mask,
            min_area=min_component_area,
            keep_components=keep_components,
        )
        mask_path = contact_masks_dir / f"{slugify(part)}.png"
        save_binary_mask(mask_path, mask)
        masks_by_part.append((part, mask))
        written_paths.append(mask_path)

    if visualization_path is not None:
        save_contact_visualization(
            canvas_rgb=canvas_rgb,
            masks_by_part=masks_by_part,
            palette=palette,
            visualization_path=visualization_path,
        )

    metadata_path = contact_masks_dir / "metadata.json"
    save_json(
        metadata_path,
        {
            "palette": palette,
            "color_max_distance": float(color_max_distance),
            "color_hue_tolerance": float(color_hue_tolerance),
            "min_color_saturation": int(min_color_saturation),
            "min_color_value": int(min_color_value),
            "min_component_area": int(min_component_area),
            "keep_components": int(keep_components),
            "target_mask_erode_pixels": int(target_mask_erode_pixels),
        },
    )
    written_paths.append(metadata_path)
    return written_paths
