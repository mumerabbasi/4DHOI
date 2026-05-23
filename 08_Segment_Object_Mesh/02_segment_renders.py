"""
Segment rendered object parts using Qwen prompt enrichment + SAM3 text prompting.

Pipeline:
1. Load PAG file to get objects and part names.
2. For each object, use a few representative renders to generate concise SAM3-ready
   text prompts for each PAG part and cache them in sam3_prompts.json.
3. For each render/view and part, run SAM3 text prompting.
4. Filter SAM3 predictions by confidence, minimum size, and object silhouette
   overlap derived from the face-ID EXR render.
5. For missing part/view masks, ask Qwen-VL for a bbox and seed SAM3 with it.
6. Merge the top valid masks per part, make part masks exclusive, and save
   per-view masks/visualizations compatible with 04_segment_meshes.py.

Usage:
    python 02_segment_renders.py --interaction_name interaction_01
"""

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image


OLLAMA_HOST = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"
QWEN_MODEL = "qwen3.6:27b"

SAM3_CHECKPOINT = None  # None -> auto-download from HuggingFace
SAM3_BPE_PATH = "/my_workspace/4DHHOI/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

MIN_MASK_PIXELS = 50  # minimum mask area to accept


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return (base_dir / p).resolve() if not p.is_absolute() else p.resolve()


def resolve_default_dirs(args, script_dir: Path) -> tuple[Path, Path]:
    output_root = _resolve_path(args.output_root, script_dir)
    return output_root, (output_root / args.interaction_name).resolve()


def resolve_pag_path(args, script_dir: Path) -> Path:
    if args.pag_file is not None:
        pag = _resolve_path(args.pag_file, script_dir)
        if not pag.exists():
            raise FileNotFoundError(f"PAG file not found: {pag}")
        return pag
    for subdir in ("output", "pags"):
        d = (script_dir.parent / "01_Generate_PAG" / subdir / args.interaction_name).resolve()
        if d.exists():
            cands = sorted(d.glob("output_pag_*.json"))
            if cands:
                return cands[0]
    raise FileNotFoundError(
        f"No output_pag_*.json found for {args.interaction_name} under 01_Generate_PAG/"
    )


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_").replace("-", "_")


def parse_pag_objects_and_parts(pag_path: Path) -> dict[str, list[str]]:
    with pag_path.open("r", encoding="utf-8") as f:
        pag = json.load(f)
    seen, objects = set(), []
    for item in pag.get("object states", []):
        if not isinstance(item, dict):
            continue
        name = (item.get("name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            objects.append(name)
    parts_map: dict[str, list[str]] = {n: [] for n in objects}
    for node in pag.get("object part nodes", []):
        if not isinstance(node, str):
            continue
        pieces = node.split(", ", 1)
        if len(pieces) != 2:
            continue
        obj, part = pieces[0].strip(), pieces[1].strip()
        if obj and part:
            parts_map.setdefault(obj, [])
            if part not in parts_map[obj]:
                parts_map[obj].append(part)
    return {n: p for n, p in parts_map.items() if p}


def _import_sam3():
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        return build_sam3_image_model, Sam3Processor
    except ImportError:
        repo = Path(__file__).resolve().parents[2] / "sam3"
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        return build_sam3_image_model, Sam3Processor


def load_sam3(
    checkpoint_path: str | None, bpe_path: str | None, device: str | None,
) -> dict:
    print("Loading SAM3 image model ...")
    if device is None:
        device = _select_default_torch_device()
    _set_current_cuda_device(device)
    build_fn, ProcessorCls = _import_sam3()
    model = build_fn(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=True,
        load_from_HF=(checkpoint_path is None),
    )
    # The local SAM3 builder only moves the model when device == "cuda" exactly.
    # Explicitly move here as well so devices like "cuda:1" work correctly.
    model = model.to(device=device)
    processor = ProcessorCls(model=model, device=device)
    print(f"SAM3 ready on {device}")
    return {"model": model, "processor": processor, "device": device}


def _set_current_cuda_device(device: str) -> None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    torch.cuda.set_device(torch.device(device))


def _select_default_torch_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _parse_qwen_response(
    text: str, names: list[str], img_w: int, img_h: int,
) -> dict[str, list[int] | None]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result: dict[str, list[int] | None] = {n: None for n in names}
    pattern = (
        r"<ref>([^<]+)</ref>\s*<box>\s*"
        r"(\[\s*\[?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]?\s*\]|null)"
        r"\s*</box>"
    )
    for ref, box_str in re.findall(pattern, text, re.IGNORECASE):
        ref = ref.strip()
        matched = next((n for n in names if n.lower() == ref.lower()), None)
        if matched is None or box_str.strip().lower() == "null":
            continue
        m = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", box_str)
        if m:
            result[matched] = [
                int(int(m.group(1)) * img_w / 1000),
                int(int(m.group(2)) * img_h / 1000),
                int(int(m.group(3)) * img_w / 1000),
                int(int(m.group(4)) * img_h / 1000),
            ]
    return result


def detect_parts_qwen(
    client: OpenAI,
    model: str,
    image_path: str,
    parts: list[str],
    object_name: str,
    img_w: int,
    img_h: int,
    reasoning_effort: str | None,
) -> dict[str, list[int] | None]:
    b64 = _encode_image_b64(image_path)
    parts_list = "\n".join(f"- {p}" for p in parts)
    prompt = (
        f"You are analyzing an isolated synthetic render of a {object_name} on a plain background. "
        f"Detect bounding boxes for these parts:\n{parts_list}\n\n"
        f"Only use the visible rendered object pixels. "
        f"Make each box tightly enclose the visible region of that part.\n\n"
        f"For each part, output in this format:\n"
        f"<ref>part_name</ref><box>[[x1,y1,x2,y2]]</box>\n\n"
        f"Coordinates should be on a 0-1000 normalized scale (not pixels).\n"
        f"If a part is not visible, output: <ref>part_name</ref><box>null</box>\n\n"
        f"Detect all parts listed above."
    )
    image_ext = Path(image_path).suffix.lower()
    image_mime = "image/png" if image_ext == ".png" else "image/jpeg"
    image_url = f"data:{image_mime};base64,{b64}"
    try:
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(
            **kwargs,
        )
        text = _extract_text_content(response.choices[0].message.content)
        return _parse_qwen_response(text, parts, img_w, img_h)
    except Exception as e:
        print(f"      Qwen-VL error: {e}")
        return {p: None for p in parts}


def _clamp_bbox(bbox: list[int] | tuple[int, ...] | None, w: int, h: int) -> list[int] | None:
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x1_raw, y1_raw, x2_raw, y2_raw = [int(v) for v in bbox]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(x1_raw, w - 1))
    y1 = max(0, min(y1_raw, h - 1))
    x2 = max(x1 + 1, min(x2_raw, w))
    y2 = max(y1 + 1, min(y2_raw, h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def _bbox_area(bbox: list[int] | tuple[int, ...] | None) -> int:
    if bbox is None or len(bbox) != 4:
        return 0
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection_area(
    bbox_a: list[int] | tuple[int, ...] | None,
    bbox_b: list[int] | tuple[int, ...] | None,
) -> int:
    if bbox_a is None or bbox_b is None:
        return 0
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def _bbox_is_inner(
    inner_bbox: list[int] | tuple[int, ...] | None,
    outer_bbox: list[int] | tuple[int, ...] | None,
) -> bool:
    """Heuristic: inner bbox is mostly covered by outer bbox and centered inside it."""
    inner_area = _bbox_area(inner_bbox)
    outer_area = _bbox_area(outer_bbox)
    if inner_area <= 0 or outer_area <= inner_area:
        return False

    ix1, iy1, ix2, iy2 = inner_bbox
    ox1, oy1, ox2, oy2 = outer_bbox
    cx = 0.5 * (ix1 + ix2)
    cy = 0.5 * (iy1 + iy2)
    center_inside = ox1 <= cx <= ox2 and oy1 <= cy <= oy2
    coverage = _bbox_intersection_area(inner_bbox, outer_bbox) / inner_area
    return center_inside and coverage >= 0.75


def _exclusive_priority_order(
    parts: list[str],
    masks: dict[str, np.ndarray | None],
    bboxes: dict[str, list[int] | None],
) -> list[str]:
    """
    Order parts so nested/smaller parts claim overlap first.

    Internal parts are approximated by bbox nesting depth, then by smaller bbox area,
    then by smaller mask area.
    """
    ranked_parts = []
    for idx, part in enumerate(parts):
        bbox = bboxes.get(part)
        mask = masks.get(part)
        nested_depth = sum(
            1
            for other in parts
            if other != part and _bbox_is_inner(bbox, bboxes.get(other))
        )
        bbox_area = _bbox_area(bbox) or sys.maxsize
        mask_area = int(mask.sum()) if mask is not None and np.any(mask) else sys.maxsize
        ranked_parts.append((part, nested_depth, bbox_area, mask_area, idx))

    ranked_parts.sort(key=lambda item: (-item[1], item[2], item[3], item[4]))
    return [part for part, *_ in ranked_parts]


def make_masks_exclusive(
    parts: list[str],
    masks: dict[str, np.ndarray | None],
    bboxes: dict[str, list[int] | None],
) -> dict[str, np.ndarray | None]:
    """Remove overlaps so higher-priority internal parts carve pixels from outer parts."""
    reference_mask = next(
        (mask for mask in masks.values() if mask is not None and np.any(mask)),
        None,
    )
    if reference_mask is None:
        return {part: None for part in parts}

    claimed = np.zeros(reference_mask.shape, dtype=bool)
    resolved: dict[str, np.ndarray | None] = {part: None for part in parts}
    priority_order = _exclusive_priority_order(parts, masks, bboxes)
    overlap_found = False

    for part in priority_order:
        mask = masks.get(part)
        if mask is None or not np.any(mask):
            continue

        mask_bool = mask.astype(bool)
        overlap = mask_bool & claimed
        trimmed = mask_bool & ~claimed

        overlap_px = int(overlap.sum())
        if overlap_px > 0:
            overlap_found = True
            print(f"        overlap trim for {part}: removed {overlap_px}px")

        trimmed_area = int(trimmed.sum())
        if trimmed_area >= MIN_MASK_PIXELS:
            resolved[part] = trimmed.astype(np.uint8)
            claimed |= trimmed
        else:
            if trimmed_area > 0 or overlap_px > 0:
                print(f"        {part}: dropped after exclusivity (area={trimmed_area}px)")
            resolved[part] = None

    if overlap_found:
        print(f"        exclusivity priority: {' > '.join(priority_order)}")

    return resolved


def _mask_output_path(masks_dir: Path, image_stem: str, part: str) -> Path:
    return masks_dir / f"{image_stem}_{_sanitize(part)}.png"


def _write_view_masks(
    masks_dir: Path,
    image_stem: str,
    parts: list[str],
    masks: dict[str, np.ndarray | None],
) -> None:
    for part in parts:
        mask_path = _mask_output_path(masks_dir, image_stem, part)
        if mask_path.exists():
            mask_path.unlink()

        mask = masks.get(part)
        if mask is None or not np.any(mask):
            continue

        cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))



def segment_with_box(
    sam3: dict,
    image_pil: Image.Image,
    bbox: list[int],
    h: int,
    w: int,
) -> np.ndarray:
    """Run SAM3 box-prompt segmentation. Returns (H, W) uint8 binary mask."""
    _set_current_cuda_device(sam3["device"])
    model = sam3["model"]
    processor = sam3["processor"]

    # Set image fresh for each box prompt (matches SAM3 example notebook pattern)
    inference_state = processor.set_image(image_pil)

    box_np = np.array(bbox, dtype=np.float32).reshape(1, 4)

    # Request a single mask directly.
    masks, _, _ = model.predict_inst(
        inference_state,
        point_coords=None,
        point_labels=None,
        box=box_np,
        multimask_output=False,
    )

    if masks is None or masks.size == 0:
        print("        predict_inst returned empty masks")
        return np.zeros((h, w), dtype=np.uint8)

    print(f"        masks.shape={masks.shape}")

    # For single-mask mode this is usually (1, H, W) or (H, W).
    if masks.ndim == 3:
        mask = masks[0]
    elif masks.ndim == 2:
        mask = masks
    else:
        # Handle unexpected shape defensively.
        mask = masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        mask = mask[0]

    # mask is bool/float, convert to binary uint8
    mask = (mask > 0).astype(np.uint8)

    # Resize if dimensions don't match (shouldn't happen but safety net)
    if mask.shape != (h, w):
        print(f"        resizing mask from {mask.shape} to ({h}, {w})")
        mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.uint8)

    return mask


_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
]


def draw_bboxes(image: np.ndarray, boxes: dict[str, list[int] | None]) -> np.ndarray:
    out = image.copy()
    for i, (name, bbox) in enumerate(boxes.items()):
        if bbox is None:
            continue
        c = _PALETTE[i % len(_PALETTE)]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (tw, th), _ = cv2.getTextSize(name, font, scale, thick)
        ly = y1 - 5 if y1 > th + 10 else y2 + th + 5
        cv2.rectangle(out, (x1, ly - th - 2), (x1 + tw + 4, ly + 2), c, -1)
        cv2.putText(out, name, (x1 + 2, ly), font, scale, (0, 0, 0), thick)
    return out


def draw_masks(image: np.ndarray, masks: dict[str, np.ndarray | None]) -> np.ndarray:
    out = image.copy()
    h, w = image.shape[:2]
    legend_items: list[tuple[str, tuple[int, int, int]]] = []
    for i, (name, mask) in enumerate(masks.items()):
        if mask is None or not np.any(mask):
            continue
        c = _PALETTE[i % len(_PALETTE)]
        # Ensure 2D mask matching image dims
        m = mask.astype(bool) if mask.shape == (h, w) else (
            cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        )
        overlay = out.copy()
        overlay[m] = c
        out = cv2.addWeighted(overlay, 0.4, out, 0.6, 0)
        legend_items.append((name, c))

    if legend_items:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_line_type = cv2.LINE_AA
        line_height = 24
        x_start = 10
        y_start = 24

        max_text_w = max(
            cv2.getTextSize(name, font, font_scale, thickness)[0][0]
            for name, _ in legend_items
        )
        legend_w = 20 + max_text_w + 10
        overlay_bg = out.copy()
        cv2.rectangle(
            overlay_bg,
            (x_start - 5, y_start - line_height + 2),
            (x_start + legend_w, y_start + (len(legend_items) - 1) * line_height + 10),
            (0, 0, 0),
            -1,
        )
        out = cv2.addWeighted(overlay_bg, 0.5, out, 0.5, 0)

        for idx, (name, color) in enumerate(legend_items):
            y = y_start + idx * line_height
            cv2.rectangle(out, (x_start, y - 10), (x_start + 14, y + 4), color, -1)
            cv2.putText(
                out,
                name,
                (x_start + 20, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                text_line_type,
            )
    return out

PROMPT_CACHE_FILENAME = "sam3_prompts.json"
DEBUG_DIRNAME = "debug"
RAW_PREDICTIONS_DIRNAME = "sam3_raw_predictions"
FALLBACK_BBOX_DIRNAME = "bboxes_fallback"
BBOX_CACHE_FILENAME = "part_bboxes.json"
DEFAULT_PROMPT_VIEW_STEMS = ("az010_el+10", "az100_el+10", "az190_el+10")
MIN_SILHOUETTE_OVERLAP = 0.5
DEDUP_IOU_THRESHOLD = 0.95


def _strip_thinking_and_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def _extract_json_object(text: str) -> dict | None:
    cleaned = _strip_thinking_and_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start: end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _prompt_image_content(image_path: Path) -> dict:
    image_ext = image_path.suffix.lower()
    image_mime = "image/png" if image_ext == ".png" else "image/jpeg"
    image_url = f"data:{image_mime};base64,{_encode_image_b64(str(image_path))}"
    return {"type": "image_url", "image_url": {"url": image_url}}


def _normalize_prompt_text(object_name: str, part_name: str, raw_text: str | None) -> str:
    text = (raw_text or "").strip()
    if not text:
        return f"{part_name} of the {object_name}"

    text = re.sub(r"\s+", " ", text)
    if part_name.lower() not in text.lower():
        text = f"{part_name} of the {object_name}"
    elif object_name.lower() not in text.lower():
        text = f"{text} on the {object_name}"

    if len(text.split()) > 24:
        text = f"{part_name} of the {object_name}"
    return text


def _build_default_prompts(object_name: str, parts: list[str]) -> dict[str, dict[str, str]]:
    return {
        part: {
            "sam3_prompt": f"{part} of the {object_name}",
        }
        for part in parts
    }


def _select_prompt_views(image_files: list[Path], num_views: int) -> list[Path]:
    by_stem = {path.stem: path for path in image_files}
    selected: list[Path] = []

    for stem in DEFAULT_PROMPT_VIEW_STEMS:
        path = by_stem.get(stem)
        if path is not None and path not in selected:
            selected.append(path)
        if len(selected) >= num_views:
            return selected

    for image_path in image_files:
        if image_path not in selected:
            selected.append(image_path)
        if len(selected) >= num_views:
            break
    return selected


def _generate_prompts_with_qwen(
    client: OpenAI,
    model: str,
    object_name: str,
    parts: list[str],
    prompt_images: list[Path],
    reasoning_effort: str | None,
) -> dict[str, dict[str, str]]:
    parts_list = "\n".join(f"- {part}" for part in parts)
    prompt = (
        f"You are helping write concise segmentation prompts for SAM3.\n"
        f"Object: {object_name}\n"
        f"PAG parts:\n{parts_list}\n\n"
        f"You will see a few renders of the same object. For each part, return a short text prompt "
        f"that will help SAM3 segment that part across different views.\n"
        f"Requirements:\n"
        f"- Return JSON only.\n"
        f"- Keep the exact part string unchanged inside the prompt.\n"
        f"- Mention the object name.\n"
        f"- Keep each prompt concise, ideally under 20 words.\n"
        f"- Do not mention camera angles, colors, or background.\n"
        f"- Focus on the physical region/shape of the part.\n\n"
        f"Return this schema exactly:\n"
        f'{{"prompts": {{"<part_name>": "<short prompt>"}}}}'
    )

    content = [{"type": "text", "text": prompt}]
    for idx, image_path in enumerate(prompt_images, start=1):
        content.append({"type": "text", "text": f"Reference render {idx}: {image_path.stem}"})
        content.append(_prompt_image_content(image_path))

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.1,
        "max_tokens": 1536,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        response = client.chat.completions.create(**kwargs)
        text = _extract_text_content(response.choices[0].message.content)
        payload = _extract_json_object(text) or {}
        prompt_map = payload.get("prompts", payload)
        if not isinstance(prompt_map, dict):
            raise ValueError("Qwen response did not contain a prompt mapping")

        prompts = {}
        for part in parts:
            prompts[part] = {
                "sam3_prompt": _normalize_prompt_text(
                    object_name, part, prompt_map.get(part)
                ),
            }
        return prompts
    except Exception as e:
        print(f"  Qwen prompt-generation error for {object_name}: {e}")
        print("  Using simple object+part prompts.")
        return _build_default_prompts(object_name, parts)


def _load_prompt_cache(
    cache_path: Path,
    object_name: str,
    parts: list[str],
) -> dict[str, dict[str, str]] | None:
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"  Failed to load prompt cache ({cache_path}): {e}")
        return None

    if not isinstance(payload, dict):
        return None

    prompt_map = payload.get("prompts")
    if not isinstance(prompt_map, dict):
        return None

    prompts: dict[str, dict[str, str]] = {}
    for part in parts:
        raw_entry = prompt_map.get(part)
        if not isinstance(raw_entry, dict):
            return None
        sam3_prompt = _normalize_prompt_text(object_name, part, raw_entry.get("sam3_prompt"))
        prompts[part] = {
            "sam3_prompt": sam3_prompt,
        }
    return prompts


def _save_prompt_cache(
    cache_path: Path,
    object_name: str,
    parts: list[str],
    prompt_views: list[Path],
    prompts: dict[str, dict[str, str]],
) -> None:
    payload = {
        "object_name": object_name,
        "parts": parts,
        "prompt_views": [path.name for path in prompt_views],
        "prompts": prompts,
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved SAM3 prompt cache -> {cache_path}")


def _coerce_cached_bbox(raw_bbox) -> list[int] | None:
    if raw_bbox is None or not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        return [int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])]
    except (TypeError, ValueError):
        return None


def _load_fallback_bbox_cache(cache_path: Path) -> dict[str, dict[str, list[int] | None]]:
    if not cache_path.exists():
        return {}

    try:
        with cache_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"  Failed to load fallback bbox cache ({cache_path}): {e}")
        return {}

    views = payload.get("views") if isinstance(payload, dict) else None
    if not isinstance(views, list):
        print(f"  Invalid fallback bbox cache format in {cache_path}; ignoring it.")
        return {}

    cache: dict[str, dict[str, list[int] | None]] = {}
    for view in views:
        if not isinstance(view, dict):
            continue
        image_name = view.get("image")
        part_boxes = view.get("parts")
        if not isinstance(image_name, str) or not isinstance(part_boxes, dict):
            continue
        cache[image_name] = {
            str(part): _coerce_cached_bbox(bbox)
            for part, bbox in part_boxes.items()
        }

    print(f"  Using fallback bbox cache: {cache_path}")
    return cache


def _save_fallback_bbox_cache(
    cache_path: Path,
    object_name: str,
    fallback_cache: dict[str, dict[str, list[int] | None]],
    image_sizes: dict[str, list[int]],
) -> None:
    payload = {
        "object_name": object_name,
        "views": [],
    }
    for image_name in sorted(fallback_cache):
        payload["views"].append(
            {
                "image": image_name,
                "image_size": image_sizes.get(image_name, [0, 0]),
                "parts": fallback_cache[image_name],
            }
        )

    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def ensure_object_prompt_cache(
    object_dir: Path,
    object_name: str,
    parts: list[str],
    qwen_client: OpenAI,
    qwen_model: str,
    reasoning_effort: str | None,
    prompt_views_count: int,
) -> dict[str, dict[str, str]] | None:
    renders_dir = object_dir / "renders"
    rgb_dir = renders_dir / "rgb"
    image_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    if not image_files:
        image_files = sorted(renders_dir.glob("rgb_*.png"))
    if not image_files:
        print(f"  No render images found for {object_name}")
        return None

    cache_path = object_dir / PROMPT_CACHE_FILENAME
    cached = _load_prompt_cache(cache_path, object_name, parts)
    if cached is not None:
        print(f"  Using cached SAM3 prompts: {cache_path}")
        return cached

    prompt_views = _select_prompt_views(image_files, max(1, prompt_views_count))
    print(
        f"  Generating SAM3 prompts for {object_name} from views: "
        f"{', '.join(path.stem for path in prompt_views)}"
    )
    prompts = _generate_prompts_with_qwen(
        qwen_client,
        qwen_model,
        object_name,
        parts,
        prompt_views,
        reasoning_effort,
    )
    _save_prompt_cache(cache_path, object_name, parts, prompt_views, prompts)
    return prompts


def _view_name_from_face_id_file(face_id_file: Path) -> str:
    stem = face_id_file.stem
    return stem[:-4] if stem.endswith("0001") else stem


def _collect_face_id_files(face_id_dir: Path) -> dict[str, Path]:
    return {
        _view_name_from_face_id_file(face_id_file): face_id_file
        for face_id_file in sorted(face_id_dir.glob("*.exr"))
    }


def _load_face_id_silhouette(exr_path: Path) -> np.ndarray:
    image = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(
            f"Failed to read face-id EXR: {exr_path}. "
            "Ensure OpenCV has OpenEXR enabled in the current environment."
        )

    # Blender writes the face ID into the EXR R channel; OpenCV loads EXR images in BGR order.
    channel = image[..., 2] if image.ndim == 3 else image
    return np.isfinite(channel) & (channel > 1e-6)


def _extract_candidates_from_state(state: dict) -> list[dict]:
    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")
    if masks is None or boxes is None or scores is None:
        return []

    if torch.is_tensor(masks):
        masks_np = masks.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    if masks_np.ndim == 4:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    if torch.is_tensor(boxes):
        boxes_np = boxes.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    if torch.is_tensor(scores):
        scores_np = scores.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    candidates: list[dict] = []
    for idx in range(min(len(masks_np), len(boxes_np), len(scores_np))):
        candidates.append(
            {
                "mask": (masks_np[idx] > 0).astype(np.uint8),
                "box": [float(v) for v in boxes_np[idx].tolist()],
                "score": float(scores_np[idx]),
            }
        )
    return candidates


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    union = int((a | b).sum())
    if union == 0:
        return 0.0
    return float((a & b).sum()) / float(union)


def _bbox_from_mask(mask: np.ndarray | None) -> list[int] | None:
    if mask is None or not np.any(mask):
        return None
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _resize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    if mask.shape == (h, w):
        return mask
    resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def _filter_and_merge_candidates(
    candidates: list[dict],
    silhouette_mask: np.ndarray,
    max_masks_per_part: int,
) -> tuple[np.ndarray | None, list[dict]]:
    kept: list[dict] = []
    silhouette = silhouette_mask.astype(bool)

    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        mask = candidate["mask"].astype(bool)
        original_area = int(mask.sum())
        if original_area < MIN_MASK_PIXELS:
            continue

        clipped = mask & silhouette
        clipped_area = int(clipped.sum())
        if clipped_area < MIN_MASK_PIXELS:
            continue

        overlap_ratio = clipped_area / max(1, original_area)
        if overlap_ratio < MIN_SILHOUETTE_OVERLAP:
            continue

        clipped_mask = clipped.astype(np.uint8)
        if any(_mask_iou(clipped_mask, prev["mask"]) >= DEDUP_IOU_THRESHOLD for prev in kept):
            continue

        kept.append(
            {
                "mask": clipped_mask,
                "box": candidate["box"],
                "score": candidate["score"],
                "original_area": original_area,
                "clipped_area": clipped_area,
                "silhouette_overlap": overlap_ratio,
            }
        )
        if len(kept) >= max_masks_per_part:
            break

    if not kept:
        return None, []

    merged = np.zeros_like(kept[0]["mask"], dtype=np.uint8)
    for item in kept:
        merged |= item["mask"]
    return merged, kept


def _serialize_candidates(candidates: list[dict]) -> list[dict]:
    serialized = []
    for item in candidates:
        entry = {k: v for k, v in item.items() if k != "mask"}
        serialized.append(entry)
    return serialized


def _run_sam3_text_prompt(
    sam3: dict,
    image_pil: Image.Image,
    prompt: str,
    confidence_threshold: float,
) -> list[dict]:
    processor = sam3["processor"]
    processor.set_confidence_threshold(confidence_threshold)
    state = processor.set_image(image_pil)
    state = processor.set_text_prompt(prompt=prompt, state=state)
    return _extract_candidates_from_state(state)


def _segment_part_with_prompts(
    sam3: dict,
    image_pil: Image.Image,
    silhouette_mask: np.ndarray,
    prompt_info: dict[str, str],
    confidence_threshold: float,
    max_masks_per_part: int,
) -> tuple[np.ndarray | None, dict]:
    prompt_text = prompt_info["sam3_prompt"]
    try:
        raw_candidates = _run_sam3_text_prompt(
            sam3, image_pil, prompt_text, confidence_threshold
        )
    except Exception as e:
        return None, {
            "used_prompt": None,
            "prompt": prompt_text,
            "error": str(e),
            "raw_candidates": [],
            "kept_candidates": [],
        }

    merged_mask, kept = _filter_and_merge_candidates(
        raw_candidates, silhouette_mask, max_masks_per_part
    )
    debug_info = {
        "used_prompt": prompt_text if merged_mask is not None and np.any(merged_mask) else None,
        "prompt": prompt_text,
        "raw_candidates": _serialize_candidates(raw_candidates),
        "kept_candidates": _serialize_candidates(kept),
    }
    if merged_mask is not None and np.any(merged_mask):
        return merged_mask, debug_info

    return None, {
        "used_prompt": None,
        "prompt": prompt_text,
        "raw_candidates": _serialize_candidates(raw_candidates),
        "kept_candidates": _serialize_candidates(kept),
    }


def _detect_missing_bboxes_with_cache(
    qwen_client: OpenAI,
    qwen_model: str,
    image_path: Path,
    object_name: str,
    missing_parts: list[str],
    w: int,
    h: int,
    reasoning_effort: str | None,
    qwen_retries: int,
    fallback_cache: dict[str, dict[str, list[int] | None]],
) -> tuple[dict[str, list[int] | None], dict[str, str | None], bool]:
    image_cache = fallback_cache.setdefault(image_path.name, {})
    boxes: dict[str, list[int] | None] = {}
    errors: dict[str, str | None] = {}
    uncached_parts = [part for part in missing_parts if part not in image_cache]

    if uncached_parts:
        max_attempts = max(0, qwen_retries) + 1
        remaining = list(uncached_parts)
        detected: dict[str, list[int] | None] = {part: None for part in uncached_parts}

        for attempt in range(1, max_attempts + 1):
            print(
                f"        Qwen bbox fallback attempt {attempt}/{max_attempts}: "
                f"{remaining}"
            )
            attempt_boxes = detect_parts_qwen(
                qwen_client,
                qwen_model,
                str(image_path),
                remaining,
                object_name,
                w,
                h,
                reasoning_effort,
            )
            for part in remaining:
                bbox = _clamp_bbox(attempt_boxes.get(part), w, h)
                if bbox is not None:
                    detected[part] = bbox

            remaining = [part for part in remaining if detected[part] is None]
            if not remaining:
                break

        for part in uncached_parts:
            image_cache[part] = detected[part]

    for part in missing_parts:
        bbox = image_cache.get(part)
        boxes[part] = _clamp_bbox(bbox, w, h)
        errors[part] = None if boxes[part] is not None else "no bbox from Qwen-VL"

    return boxes, errors, bool(uncached_parts)


def _segment_missing_parts_with_bboxes(
    sam3: dict,
    image_pil: Image.Image,
    silhouette_mask: np.ndarray,
    missing_parts: list[str],
    fallback_bboxes: dict[str, list[int] | None],
    h: int,
    w: int,
) -> tuple[dict[str, np.ndarray | None], dict[str, str | None]]:
    masks: dict[str, np.ndarray | None] = {}
    errors: dict[str, str | None] = {}
    silhouette = silhouette_mask.astype(bool)

    for part in missing_parts:
        bbox = fallback_bboxes.get(part)
        if bbox is None:
            masks[part] = None
            errors[part] = "missing bbox"
            continue

        try:
            mask = segment_with_box(sam3, image_pil, bbox, h, w).astype(bool)
        except Exception as e:
            masks[part] = None
            errors[part] = f"SAM3 box prompt failed: {e}"
            continue

        clipped = mask & silhouette
        area = int(clipped.sum())
        if area < MIN_MASK_PIXELS:
            masks[part] = None
            errors[part] = f"bbox mask too small after silhouette clip: {area}px"
            continue

        masks[part] = clipped.astype(np.uint8)
        errors[part] = None

    return masks, errors


def process_object(
    object_dir: Path,
    object_name: str,
    parts: list[str],
    prompts: dict[str, dict[str, str]],
    sam3: dict,
    qwen_client: OpenAI,
    qwen_model: str,
    reasoning_effort: str | None,
    qwen_retries: int,
    confidence_threshold: float,
    max_masks_per_part: int,
    make_masks_exclusive_flag: bool,
    enable_bbox_fallback: bool,
) -> None:
    renders_dir = object_dir / "renders"
    rgb_dir = renders_dir / "rgb"
    face_id_dir = renders_dir / "face_id"
    masks_dir = object_dir / "masks"
    viz_dir = object_dir / "visualizations"
    raw_dir = object_dir.parent / DEBUG_DIRNAME / RAW_PREDICTIONS_DIRNAME / object_dir.name
    bboxes_dir = object_dir / FALLBACK_BBOX_DIRNAME
    for directory in (masks_dir, viz_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if enable_bbox_fallback:
        bboxes_dir.mkdir(parents=True, exist_ok=True)
    for old_debug_json in raw_dir.glob("*.json"):
        old_debug_json.unlink()

    image_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    if not image_files:
        image_files = sorted(renders_dir.glob("rgb_*.png"))
    if not image_files:
        print(f"  No render images found for {object_name}")
        return

    face_id_files = _collect_face_id_files(face_id_dir) if face_id_dir.exists() else {}
    fallback_cache_path = bboxes_dir / BBOX_CACHE_FILENAME
    fallback_cache = (
        _load_fallback_bbox_cache(fallback_cache_path)
        if enable_bbox_fallback
        else {}
    )
    fallback_image_sizes: dict[str, list[int]] = {}

    print(f"\n  Processing {object_name}: parts={parts}, views={len(image_files)}")
    print(f"  Debug raw predictions: {raw_dir}")
    if enable_bbox_fallback:
        print(f"  BBox fallback cache: {fallback_cache_path}")
    else:
        print("  BBox fallback: disabled")
    print("  Running SAM3 text segmentation using cached prompts ...")
    for idx, img_path in enumerate(image_files, start=1):
        print(f"\n    [SAM3 {idx}/{len(image_files)}] {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"      Could not read {img_path}")
            continue

        face_id_path = face_id_files.get(img_path.stem)
        if face_id_path is None:
            print(f"      Missing face-id EXR for {img_path.stem}; skipping view")
            continue

        silhouette_mask = _load_face_id_silhouette(face_id_path)
        h, w = image_bgr.shape[:2]
        if silhouette_mask.shape != (h, w):
            silhouette_mask = _resize_mask(silhouette_mask.astype(np.uint8), h, w).astype(bool)
        fallback_image_sizes[img_path.name] = [w, h]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        all_masks: dict[str, np.ndarray | None] = {}
        all_bboxes: dict[str, list[int] | None] = {}
        fallback_viz_bboxes: dict[str, list[int] | None] = {}
        raw_payload = {
            "object_name": object_name,
            "image": img_path.name,
            "face_id_path": str(face_id_path),
            "parts": {},
        }

        for part in parts:
            prompt_info = prompts.get(part, {"sam3_prompt": part})
            print(f"      {part}: {prompt_info['sam3_prompt']}")
            merged_mask, debug_info = _segment_part_with_prompts(
                sam3,
                image_pil,
                silhouette_mask,
                prompt_info,
                confidence_threshold,
                max_masks_per_part,
            )
            if merged_mask is not None and int(merged_mask.sum()) >= MIN_MASK_PIXELS:
                all_masks[part] = merged_mask
                all_bboxes[part] = _bbox_from_mask(merged_mask)
                text_mask_found = True
                print(
                    f"        kept area={int(merged_mask.sum())}px "
                    f"via prompt"
                )
            else:
                all_masks[part] = None
                all_bboxes[part] = None
                text_mask_found = False
                print("        no valid mask")

            raw_payload["parts"][part] = {
                "sam3_prompt": prompt_info["sam3_prompt"],
                **debug_info,
                "text_mask_found": text_mask_found,
                "fallback_used": False,
                "fallback_bbox": None,
                "fallback_error": None,
                "final_source": "text" if text_mask_found else "missing",
            }

        missing_parts = [
            part
            for part in parts
            if all_masks.get(part) is None or int(all_masks[part].sum()) < MIN_MASK_PIXELS
        ]
        if missing_parts and enable_bbox_fallback:
            print(f"      Missing after text prompt, trying bbox fallback: {missing_parts}")
            fallback_bboxes, bbox_errors, cache_changed = _detect_missing_bboxes_with_cache(
                qwen_client,
                qwen_model,
                img_path,
                object_name,
                missing_parts,
                w,
                h,
                reasoning_effort,
                qwen_retries,
                fallback_cache,
            )
            fallback_viz_bboxes = dict(fallback_bboxes)
            if cache_changed:
                _save_fallback_bbox_cache(
                    fallback_cache_path,
                    object_name,
                    fallback_cache,
                    fallback_image_sizes,
                )

            fallback_masks, mask_errors = _segment_missing_parts_with_bboxes(
                sam3,
                image_pil,
                silhouette_mask,
                missing_parts,
                fallback_bboxes,
                h,
                w,
            )
            for part in missing_parts:
                raw_payload["parts"][part]["fallback_bbox"] = fallback_bboxes.get(part)
                fallback_mask = fallback_masks.get(part)
                if fallback_mask is not None and int(fallback_mask.sum()) >= MIN_MASK_PIXELS:
                    all_masks[part] = fallback_mask
                    all_bboxes[part] = fallback_bboxes.get(part)
                    raw_payload["parts"][part]["fallback_used"] = True
                    raw_payload["parts"][part]["final_source"] = "bbox_fallback"
                    print(
                        f"        {part}: recovered area={int(fallback_mask.sum())}px "
                        "via bbox fallback"
                    )
                else:
                    error = mask_errors.get(part) or bbox_errors.get(part) or "bbox fallback failed"
                    raw_payload["parts"][part]["fallback_error"] = error
                    raw_payload["parts"][part]["final_source"] = "missing"
                    print(f"        {part}: bbox fallback failed ({error})")

            if any(bbox is not None for bbox in fallback_viz_bboxes.values()):
                cv2.imwrite(
                    str(bboxes_dir / f"{img_path.stem}_bboxes.png"),
                    draw_bboxes(image_bgr, fallback_viz_bboxes),
                )
        elif missing_parts:
            for part in missing_parts:
                raw_payload["parts"][part]["fallback_error"] = "bbox fallback disabled"

        if make_masks_exclusive_flag:
            all_masks = make_masks_exclusive(parts, all_masks, all_bboxes)

        _write_view_masks(masks_dir, img_path.stem, parts, all_masks)
        with (raw_dir / f"{img_path.stem}.json").open("w", encoding="utf-8") as f:
            json.dump(raw_payload, f, indent=2)

        if any(m is not None and np.any(m) for m in all_masks.values()):
            cv2.imwrite(
                str(viz_dir / f"{img_path.stem}_segmented.png"),
                draw_masks(image_bgr, all_masks),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment rendered object parts using Qwen-enriched SAM3 text prompts."
    )
    parser.add_argument(
        "--interaction_name",
        type=str,
        default="interaction_01",
        help="Interaction name used to resolve default input paths.",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help=(
            "PAG JSON path (default: first output_pag_*.json in "
            "../01_Generate_PAG/output/<interaction_name>, with ../01_Generate_PAG/pags/<interaction_name> "
            "checked if needed)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output",
        help=(
            "Output root containing rendered object directories "
            "(default: ./output, relative to this script)."
        ),
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=OLLAMA_HOST,
        help=f"Ollama-compatible OpenAI endpoint (default: {OLLAMA_HOST}).",
    )
    parser.add_argument(
        "--ollama_api_key",
        type=str,
        default=OLLAMA_API_KEY,
        help="API key sent to the Ollama-compatible endpoint.",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=QWEN_MODEL,
        help=f"Qwen-VL model name used for prompt generation (default: {QWEN_MODEL}).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="high",
        help=(
            "Reasoning control for Ollama's OpenAI-compatible endpoint. "
            "Use 'none' to omit the field."
        ),
    )
    parser.add_argument(
        "--sam3_checkpoint",
        type=str,
        default=SAM3_CHECKPOINT,
        help="SAM3 checkpoint path (default: auto-download from HuggingFace).",
    )
    parser.add_argument(
        "--sam3_bpe_path",
        type=str,
        default=SAM3_BPE_PATH,
        help=f"SAM3 BPE vocab path (default: {SAM3_BPE_PATH}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for SAM3 inference, for example cpu, cuda, or cuda:1 "
            "(default: auto-select cuda if available, else cpu)."
        ),
    )
    parser.add_argument(
        "--not_make_masks_exclusive",
        action="store_true",
        help="Disable post-processing that removes mask overlaps by prioritizing internal parts.",
    )
    parser.add_argument(
        "--sam3_confidence_threshold",
        type=float,
        default=0.5,
        help="Confidence threshold applied inside SAM3 text prompting (default: 0.5).",
    )
    parser.add_argument(
        "--max_masks_per_part",
        type=int,
        default=4,
        help="Maximum number of valid SAM3 masks to union per part/view (default: 4).",
    )
    parser.add_argument(
        "--prompt_views",
        type=int,
        default=3,
        help="Number of representative renders used for Qwen prompt generation (default: 3).",
    )
    parser.add_argument(
        "--qwen_retries",
        type=int,
        default=1,
        help="Number of retries for Qwen-VL bbox fallback detection (default: 1).",
    )
    parser.add_argument(
        "--disable_bbox_fallback",
        action="store_true",
        help="Disable Qwen-VL bbox fallback for missing SAM3 text masks.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    _, objects_video_dir = resolve_default_dirs(args, script_dir)
    pag_path = resolve_pag_path(args, script_dir)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    objects_parts = parse_pag_objects_and_parts(pag_path)
    if not objects_parts:
        print(f"No object parts found in PAG: {pag_path}")
        return
    if not objects_video_dir.exists():
        raise NotADirectoryError(
            f"Objects directory not found: {objects_video_dir}. "
            "Run 01_render_mesh_views.py first."
        )

    print(f"\n{'=' * 60}")
    print("02_segment_renders.py -- Qwen prompt enrichment + SAM3 text mode")
    print(f"  video:   {args.interaction_name}")
    print(f"  pag:     {pag_path}")
    print(f"  ollama:  {args.ollama_host}")
    print(f"  model:   {args.qwen_model}")
    print(f"  reasoning effort: {args.reasoning_effort}")
    print(f"  device:  {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  SAM3 confidence threshold: {args.sam3_confidence_threshold}")
    print(f"  max masks per part: {args.max_masks_per_part}")
    print(f"  prompt views: {args.prompt_views}")
    print(f"  exclusive masks: {'on' if not args.not_make_masks_exclusive else 'off'}")
    print(f"  bbox fallback: {'off' if args.disable_bbox_fallback else 'on'}")
    if not args.disable_bbox_fallback:
        print(f"  qwen bbox retries: {max(0, args.qwen_retries)}")
    print(f"  objects: {objects_video_dir}")
    for name, parts in objects_parts.items():
        print(f"    {name}: {parts}")
    print(f"{'=' * 60}\n")

    valid_objects: list[tuple[str, str, list[str], Path]] = []
    for obj_name, parts in objects_parts.items():
        slug = _sanitize(obj_name)
        obj_dir = objects_video_dir / slug
        renders = obj_dir / "renders"
        if not obj_dir.exists():
            print(f"[SKIP] {slug}: directory not found: {obj_dir}")
            continue
        if not renders.exists():
            print(f"[SKIP] {slug}: renders not found: {renders}")
            continue
        valid_objects.append((obj_name, slug, parts, obj_dir))

    if not valid_objects:
        print("No valid objects to process.")
        return

    qwen_client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key)

    print(f"\n{'=' * 60}")
    print("Stage 1/2: Qwen prompt generation for all objects")
    print(f"{'=' * 60}")
    object_prompts: dict[Path, dict[str, dict[str, str]]] = {}
    for obj_name, slug, parts, obj_dir in valid_objects:
        print(f"\n{'=' * 60}")
        print(f"Object (Qwen): {obj_name} ({slug})  parts: {parts}")
        print(f"{'=' * 60}")
        prompts = ensure_object_prompt_cache(
            obj_dir,
            obj_name,
            parts,
            qwen_client,
            args.qwen_model,
            reasoning_effort,
            args.prompt_views,
        )
        if prompts is not None:
            object_prompts[obj_dir] = prompts

    sam3 = load_sam3(args.sam3_checkpoint, args.sam3_bpe_path, args.device)
    print(f"\n{'=' * 60}")
    print("Stage 2/2: SAM3 text segmentation for all objects")
    print(f"{'=' * 60}")
    for obj_name, slug, parts, obj_dir in valid_objects:
        prompts = object_prompts.get(obj_dir)
        if prompts is None:
            print(f"[SKIP] {slug}: prompt generation failed")
            continue
        print(f"\n{'=' * 60}")
        print(f"Object (SAM3): {obj_name} ({slug})  parts: {parts}")
        print(f"{'=' * 60}")
        process_object(
            obj_dir,
            obj_name,
            parts,
            prompts,
            sam3,
            qwen_client,
            args.qwen_model,
            reasoning_effort,
            args.qwen_retries,
            confidence_threshold=args.sam3_confidence_threshold,
            max_masks_per_part=max(1, args.max_masks_per_part),
            make_masks_exclusive_flag=not args.not_make_masks_exclusive,
            enable_bbox_fallback=not args.disable_bbox_fallback,
        )
    print("Done!")


if __name__ == "__main__":
    main()
