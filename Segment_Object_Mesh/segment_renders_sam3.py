"""
Segment rendered object parts using Qwen prompt enrichment + SAM3 text prompting.

Pipeline:
1. Load PAG file to get objects and part names.
2. For each object, use a few representative renders to generate concise SAM3-ready
   text prompts for each PAG part and cache them in sam3_prompts.json.
3. For each render/view and part, run SAM3 text prompting.
4. Filter SAM3 predictions by confidence, minimum size, and object silhouette
   overlap derived from the face-ID EXR render.
5. Merge the top valid masks per part, make part masks exclusive, and save
   per-view masks/visualizations compatible with segment_meshes.py.

Usage:
    python segment_renders_sam3.py --video_name video_01
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image

from segment_renders import (  # type: ignore
    MIN_MASK_PIXELS,
    OLLAMA_API_KEY,
    OLLAMA_HOST,
    QWEN_MODEL,
    SAM3_BPE_PATH,
    SAM3_CHECKPOINT,
    _encode_image_b64,
    _extract_text_content,
    _sanitize,
    _write_view_masks,
    draw_masks,
    load_sam3,
    make_masks_exclusive,
    parse_pag_objects_and_parts,
    resolve_default_dirs,
    resolve_pag_path,
)


PROMPT_CACHE_FILENAME = "sam3_prompts.json"
RAW_PREDICTIONS_DIRNAME = "sam3_raw_predictions"
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
        payload = json.loads(cleaned[start : end + 1])
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
            "fallback_prompt": part,
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
                "fallback_prompt": part,
            }
        return prompts
    except Exception as e:
        print(f"  Qwen prompt-generation error for {object_name}: {e}")
        print("  Falling back to simple object+part prompts.")
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
        fallback_prompt = raw_entry.get("fallback_prompt")
        if not isinstance(fallback_prompt, str) or not fallback_prompt.strip():
            fallback_prompt = part
        prompts[part] = {
            "sam3_prompt": sam3_prompt,
            "fallback_prompt": fallback_prompt.strip(),
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
    attempts = [
        ("sam3_prompt", prompt_info["sam3_prompt"]),
        ("fallback_prompt", prompt_info["fallback_prompt"]),
    ]
    debug_attempts: list[dict] = []

    for attempt_idx, (prompt_type, prompt_text) in enumerate(attempts):
        if attempt_idx > 0 and prompt_text == attempts[0][1]:
            continue

        try:
            raw_candidates = _run_sam3_text_prompt(
                sam3, image_pil, prompt_text, confidence_threshold
            )
        except Exception as e:
            raw_candidates = []
            debug_attempts.append(
                {
                    "prompt_type": prompt_type,
                    "prompt": prompt_text,
                    "error": str(e),
                    "raw_candidates": [],
                    "kept_candidates": [],
                }
            )
            continue

        merged_mask, kept = _filter_and_merge_candidates(
            raw_candidates, silhouette_mask, max_masks_per_part
        )
        debug_attempts.append(
            {
                "prompt_type": prompt_type,
                "prompt": prompt_text,
                "raw_candidates": _serialize_candidates(raw_candidates),
                "kept_candidates": _serialize_candidates(kept),
            }
        )
        if merged_mask is not None and np.any(merged_mask):
            return merged_mask, {
                "used_prompt_type": prompt_type,
                "used_prompt": prompt_text,
                "attempts": debug_attempts,
            }

    return None, {
        "used_prompt_type": None,
        "used_prompt": None,
        "attempts": debug_attempts,
    }


def process_object(
    object_dir: Path,
    object_name: str,
    parts: list[str],
    prompts: dict[str, dict[str, str]],
    sam3: dict,
    confidence_threshold: float,
    max_masks_per_part: int,
    make_masks_exclusive_flag: bool,
) -> None:
    renders_dir = object_dir / "renders"
    rgb_dir = renders_dir / "rgb"
    face_id_dir = renders_dir / "face_id"
    masks_dir = object_dir / "masks"
    viz_dir = object_dir / "visualizations"
    raw_dir = object_dir / RAW_PREDICTIONS_DIRNAME
    for directory in (masks_dir, viz_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    image_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    if not image_files:
        image_files = sorted(renders_dir.glob("rgb_*.png"))
    if not image_files:
        print(f"  No render images found for {object_name}")
        return

    face_id_files = _collect_face_id_files(face_id_dir) if face_id_dir.exists() else {}

    print(f"\n  Processing {object_name}: parts={parts}, views={len(image_files)}")
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

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        all_masks: dict[str, np.ndarray | None] = {}
        all_bboxes: dict[str, list[int] | None] = {}
        raw_payload = {
            "object_name": object_name,
            "image": img_path.name,
            "face_id_path": str(face_id_path),
            "parts": {},
        }

        for part in parts:
            prompt_info = prompts.get(part, {"sam3_prompt": part, "fallback_prompt": part})
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
                print(
                    f"        kept area={int(merged_mask.sum())}px "
                    f"via {debug_info['used_prompt_type']}"
                )
            else:
                all_masks[part] = None
                all_bboxes[part] = None
                print("        no valid mask")

            raw_payload["parts"][part] = {
                "sam3_prompt": prompt_info["sam3_prompt"],
                "fallback_prompt": prompt_info["fallback_prompt"],
                **debug_info,
            }

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
        "--video_name",
        type=str,
        default="video_01",
        help="Video name used to resolve default input paths.",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help=(
            "PAG JSON path (default: first output_pag_*.json in "
            "../Generate_PAG/output/<video_name>, with ../Generate_PAG/pags/<video_name> "
            "as a fallback)."
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
            "Run render_mesh_views.py first."
        )

    print(f"\n{'=' * 60}")
    print("segment_renders_sam3.py -- Qwen prompt enrichment + SAM3 text mode")
    print(f"  video:   {args.video_name}")
    print(f"  pag:     {pag_path}")
    print(f"  ollama:  {args.ollama_host}")
    print(f"  model:   {args.qwen_model}")
    print(f"  reasoning effort: {args.reasoning_effort}")
    print(f"  device:  {args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"  SAM3 confidence threshold: {args.sam3_confidence_threshold}")
    print(f"  max masks per part: {args.max_masks_per_part}")
    print(f"  prompt views: {args.prompt_views}")
    print(f"  exclusive masks: {'on' if not args.not_make_masks_exclusive else 'off'}")
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
            confidence_threshold=args.sam3_confidence_threshold,
            max_masks_per_part=max(1, args.max_masks_per_part),
            make_masks_exclusive_flag=not args.not_make_masks_exclusive,
        )
    print("Done!")


if __name__ == "__main__":
    main()
