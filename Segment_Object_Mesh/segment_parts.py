"""
Segment rendered object parts using Qwen-VL detection + SAM3 box-prompt segmentation.

Pipeline:
1. Load PAG file to get objects and their parts.
2. For each object's rendered views:
   a. Detect part bounding boxes with Qwen-VL (qwen3-vl:32b via Ollama).
   b. Segment each detected part with SAM3 box prompts.
3. Save per-view masks, bbox visualizations, and segmentation overlays.

Usage:
    python segment_parts.py --video_name video_01
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from openai import OpenAI
from PIL import Image


OLLAMA_HOST = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"
QWEN_MODEL = "qwen3-vl:32b"

SAM3_CHECKPOINT = None  # None -> auto-download from HuggingFace
SAM3_BPE_PATH = "/my_workspace/4DHHOI/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

MIN_MASK_PIXELS = 50  # minimum mask area to accept


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    return (base_dir / p).resolve() if not p.is_absolute() else p.resolve()


def resolve_default_dirs(args, script_dir: Path) -> tuple[Path, Path]:
    output_root = _resolve_path(args.output_root, script_dir)
    return output_root, (output_root / args.video_name).resolve()


def resolve_pag_path(args, script_dir: Path) -> Path:
    if args.pag_file is not None:
        pag = _resolve_path(args.pag_file, script_dir)
        if not pag.exists():
            raise FileNotFoundError(f"PAG file not found: {pag}")
        return pag
    for subdir in ("output", "pags"):
        d = (script_dir.parent / "Generate_PAG" / subdir / args.video_name).resolve()
        if d.exists():
            cands = sorted(d.glob("output_pag_*.json"))
            if cands:
                return cands[0]
    raise FileNotFoundError(
        f"No output_pag_*.json found for {args.video_name} under Generate_PAG/"
    )


def _sanitize(name: str) -> str:
    return name.strip().replace(" ", "_")


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


def load_sam3(checkpoint_path: str | None, bpe_path: str | None) -> dict:
    print("Loading SAM3 image model ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_fn, ProcessorCls = _import_sam3()
    model = build_fn(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        eval_mode=True,
        enable_inst_interactivity=True,
        load_from_HF=(checkpoint_path is None),
    )
    processor = ProcessorCls(model=model, device=device)
    print(f"SAM3 ready on {device}")
    return {"model": model, "processor": processor, "device": device}


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
) -> dict[str, list[int] | None]:
    b64 = _encode_image_b64(image_path)
    parts_list = "\n".join(f"- {p}" for p in parts)
    prompt = (
        f"You are analyzing a rendered image of a {object_name}. "
        f"Detect bounding boxes for these parts:\n{parts_list}\n\n"
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
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        text = _extract_text_content(response.choices[0].message.content)
        return _parse_qwen_response(text, parts, img_w, img_h)
    except Exception as e:
        print(f"      Qwen-VL error: {e}")
        return {p: None for p in parts}


def _clamp_bbox(bbox: list[int], w: int, h: int) -> list[int] | None:
    x1 = max(0, min(bbox[0], w - 1))
    y1 = max(0, min(bbox[1], h - 1))
    x2 = max(x1 + 1, min(bbox[2], w))
    y2 = max(y1 + 1, min(bbox[3], h))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [x1, y1, x2, y2]


def segment_with_box(
    sam3: dict,
    image_pil: Image.Image,
    bbox: list[int],
    h: int,
    w: int,
) -> np.ndarray:
    """Run SAM3 box-prompt segmentation. Returns (H, W) uint8 binary mask."""
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

    print(f"        masks.shape={masks.shape} dtype={masks.dtype}")

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


def process_object(
    object_dir: Path,
    object_name: str,
    parts: list[str],
    qwen_client: OpenAI,
    qwen_model: str,
    sam3: dict,
) -> dict:
    renders_dir = object_dir / "renders"
    rgb_dir = renders_dir / "rgb"
    masks_dir = object_dir / "masks"
    bboxes_dir = object_dir / "bboxes"
    viz_dir = object_dir / "visualizations"
    for d in (masks_dir, bboxes_dir, viz_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted(rgb_dir.glob("*.png")) if rgb_dir.exists() else []
    if not image_files:
        image_files = sorted(renders_dir.glob("rgb_*.png"))
    if not image_files:
        print(f"  No render images found for {object_name}")
        return {}

    print(f"\n  Processing {object_name}: parts={parts}, views={len(image_files)}")
    results = {"object_name": object_name, "parts": parts, "views": []}

    for img_path in image_files:
        print(f"\n    View: {img_path.name}")
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"      Could not read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        h, w = image_bgr.shape[:2]

        view = {"image": img_path.name, "image_size": [w, h], "parts": {}}
        all_masks: dict[str, np.ndarray | None] = {}
        all_bboxes: dict[str, list[int] | None] = {}

        # Detect all parts with Qwen-VL
        print("      Detecting parts with Qwen-VL ...")
        qwen_boxes = detect_parts_qwen(
            qwen_client,
            qwen_model,
            str(img_path),
            parts,
            object_name,
            w,
            h,
        )

        for part in parts:
            raw_bbox = qwen_boxes.get(part)
            bbox = _clamp_bbox(raw_bbox, w, h) if raw_bbox else None
            all_bboxes[part] = bbox

            if bbox is None:
                print(f"      {part}: not detected by Qwen-VL")
                all_masks[part] = None
                view["parts"][part] = {"detected": False, "bbox": None, "mask_file": None}
                continue

            print(f"      {part}: bbox={bbox}")
            mask = segment_with_box(sam3, image_pil, bbox, h, w)
            area = int(mask.sum())
            print(f"        area={area}px")

            if area >= MIN_MASK_PIXELS:
                all_masks[part] = mask
                fname = f"{img_path.stem}_{part.replace(' ', '_')}.png"
                cv2.imwrite(str(masks_dir / fname), (mask * 255).astype(np.uint8))
                view["parts"][part] = {
                    "detected": True, "bbox": bbox, "mask_file": fname, "area": area,
                }
            else:
                print("        mask too small, discarding")
                all_masks[part] = None
                view["parts"][part] = {"detected": False, "bbox": bbox, "mask_file": None}

        # Save bbox visualization
        if any(b is not None for b in all_bboxes.values()):
            cv2.imwrite(
                str(bboxes_dir / f"{img_path.stem}_bboxes.png"),
                draw_bboxes(image_bgr, all_bboxes),
            )
        # Save mask overlay visualization
        if any(m is not None and np.any(m) for m in all_masks.values()):
            cv2.imwrite(
                str(viz_dir / f"{img_path.stem}_segmented.png"),
                draw_masks(image_bgr, all_masks),
            )

        results["views"].append(view)

    results_path = bboxes_dir / "part_labels.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Segment rendered object parts using Qwen-VL + SAM3."
    )
    parser.add_argument("--video_name", type=str, default="video_01")
    parser.add_argument("--pag_file", type=str, default=None)
    parser.add_argument("--output_root", type=str, default="./output")
    parser.add_argument("--ollama_host", type=str, default=OLLAMA_HOST)
    parser.add_argument("--ollama_api_key", type=str, default=OLLAMA_API_KEY)
    parser.add_argument("--qwen_model", type=str, default=QWEN_MODEL)
    parser.add_argument("--sam3_checkpoint", type=str, default=SAM3_CHECKPOINT)
    parser.add_argument("--sam3_bpe_path", type=str, default=SAM3_BPE_PATH)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    _, objects_video_dir = resolve_default_dirs(args, script_dir)
    pag_path = resolve_pag_path(args, script_dir)

    objects_parts = parse_pag_objects_and_parts(pag_path)
    if not objects_parts:
        print(f"No object parts found in PAG: {pag_path}")
        return
    if not objects_video_dir.exists():
        raise NotADirectoryError(
            f"Objects directory not found: {objects_video_dir}. "
            "Run render_views_blender.py first."
        )

    print(f"\n{'=' * 60}")
    print("segment_parts.py -- Qwen-VL + SAM3 part segmentation")
    print(f"  video:   {args.video_name}")
    print(f"  pag:     {pag_path}")
    print(f"  ollama:  {args.ollama_host}")
    print(f"  model:   {args.qwen_model}")
    print(f"  objects: {objects_video_dir}")
    for name, parts in objects_parts.items():
        print(f"    {name}: {parts}")
    print(f"{'=' * 60}\n")

    qwen_client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key)
    sam3 = load_sam3(args.sam3_checkpoint, args.sam3_bpe_path)

    for obj_name, parts in objects_parts.items():
        slug = _sanitize(obj_name)
        obj_dir = objects_video_dir / slug
        renders = obj_dir / "renders"
        if not obj_dir.exists():
            reason = f"directory not found: {obj_dir}"
            print(f"[SKIP] {slug}: {reason}")
            continue
        if not renders.exists():
            reason = f"renders not found: {renders}"
            print(f"[SKIP] {slug}: {reason}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Object: {obj_name} ({slug})  parts: {parts}")
        print(f"{'=' * 60}")
        process_object(obj_dir, obj_name, parts, qwen_client, args.qwen_model, sam3)
    print("Done!")


if __name__ == "__main__":
    main()
