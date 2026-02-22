"""Segment first video frame into object and human masks using SAM3.

Run this script with the sam3 environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_pag_objects(pag_path: Path) -> list[str]:
    """Extract unique object names from PAG file."""
    with pag_path.open("r", encoding="utf-8") as f:
        pag = json.load(f)

    objects = set()
    for node in pag.get("object part nodes", []):
        obj_name = node.split(", ", 1)[0]
        objects.add(obj_name)

    return sorted(objects)


def load_sam3_processor(
    confidence_threshold: float = 0.25,
    device: str = "cuda",
) -> Sam3Processor:
    """Load SAM3 image processor."""
    print(f"Loading SAM3 model on {device}...")
    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("SAM3 model loaded successfully")
    return processor


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return (inter_area / union) if union > 0 else 0.0


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    return float(inter / union) if union > 0 else 0.0


def _has_collision(
    candidate: dict,
    used_candidates: list[dict],
    bbox_iou_threshold: float = 0.80,
    mask_iou_threshold: float = 0.65,
) -> bool:
    candidate_bbox = candidate.get("bbox")
    candidate_mask = candidate.get("mask")

    for used in used_candidates:
        used_bbox = used.get("bbox")
        if candidate_bbox is not None and used_bbox is not None:
            if _bbox_iou(candidate_bbox, used_bbox) >= bbox_iou_threshold:
                return True

        used_mask = used.get("mask")
        if candidate_mask is not None and used_mask is not None:
            if _mask_iou(candidate_mask, used_mask) >= mask_iou_threshold:
                return True

    return False


def _pick_best_available_candidate(
    candidates: list[dict],
    used_candidates: list[dict],
) -> tuple[dict | None, int]:
    if not candidates:
        return None, -1

    for idx, candidate in enumerate(candidates):
        if not _has_collision(candidate, used_candidates):
            return candidate, idx

    return candidates[0], 0


def _deduplicate_candidates(
    candidates: list[dict],
    iou_threshold: float = 0.85,
) -> list[dict]:
    if not candidates:
        return []

    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("score", 0.0),
        reverse=True,
    )
    kept: list[dict] = []
    for candidate in sorted_candidates:
        bbox = candidate.get("bbox")
        if bbox is None:
            continue
        is_duplicate = any(
            _bbox_iou(bbox, other["bbox"]) >= iou_threshold
            for other in kept
            if other.get("bbox") is not None
        )
        if not is_duplicate:
            kept.append(candidate)
    return kept


def _to_binary_mask(mask_like: np.ndarray | torch.Tensor) -> np.ndarray:
    mask = mask_like
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    if getattr(mask, "ndim", 0) == 3:
        mask = mask.squeeze(0)
    return (mask > 0.5).astype(np.uint8)


def segment_candidates_with_sam3(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
) -> list[dict]:
    """Segment prompt with SAM3 and return candidate masks with score and bbox."""
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or len(masks) == 0:
        processor.reset_all_prompts(inference_state)
        return []

    candidates = []
    for idx in range(len(masks)):
        mask = _to_binary_mask(masks[idx])
        if mask.sum() == 0:
            continue

        bbox = None
        if boxes is not None and len(boxes) > idx:
            box = boxes[idx]
            if hasattr(box, "detach"):
                box = box.detach()
            if hasattr(box, "cpu"):
                box = box.cpu().numpy()
            bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
        if bbox is None:
            bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue

        score = 0.0
        if scores is not None and len(scores) > idx:
            score_item = scores[idx]
            if hasattr(score_item, "item"):
                score = float(score_item.item())
            else:
                score = float(score_item)

        candidates.append({
            "mask": mask,
            "bbox": bbox,
            "score": score,
        })

    processor.reset_all_prompts(inference_state)
    candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return candidates


def extract_first_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from video: {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def resolve_single_video(video_dir: Path) -> Path:
    video_files = [
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if len(video_files) != 1:
        raise ValueError(
            f"--video_dir must contain exactly one video file, found {len(video_files)} in {video_dir}"
        )
    return video_files[0]


def save_mask(mask: np.ndarray, path: Path) -> None:
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))


def save_bbox_image(image_rgb: np.ndarray, bbox: list[int], label: str, path: Path) -> None:
    image_bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        image_bgr,
        label,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(path), image_bgr)


def resolve_pag_path(script_dir: Path, video_name: str, pag_file: str | None) -> Path:
    if pag_file is not None:
        path = Path(pag_file)
        if not path.is_absolute():
            path = script_dir / path
        return path.resolve()

    pag_dir = (script_dir.parent / "Generate_PAG" / "output" / video_name).resolve()
    pag_files = sorted(pag_dir.glob("*.json"))
    if len(pag_files) != 1:
        raise ValueError(
            f"Expected exactly one PAG JSON in {pag_dir}, found {len(pag_files)}."
        )
    return pag_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment first frame objects and human using SAM3 (run with sam3 env)."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="../Generate_Video/output/video_03",
        help="Directory containing exactly one video file.",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="Optional PAG JSON; default is ../Generate_PAG/output/<video_xx>/*.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Segmentation output dir; default is ./output/<video_xx>.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="SAM3 confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda, cpu, cuda:0, etc.).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = script_dir / video_dir
    video_dir = video_dir.resolve()

    video_name = video_dir.name
    video_path = resolve_single_video(video_dir)
    pag_path = resolve_pag_path(script_dir, video_name, args.pag_file)

    if args.output_dir is None:
        output_root = script_dir / "output" / video_name
    else:
        output_root = Path(args.output_dir)
        if not output_root.is_absolute():
            output_root = script_dir / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frame_name = "frame_00"
    image_rgb = extract_first_frame(video_path)
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    frame_path = output_root / f"{frame_name}.png"
    cv2.imwrite(str(frame_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    objects = parse_pag_objects(pag_path)
    print(f"Objects to process: {objects}")
    print(f"Saved source frame: {frame_path}")

    processor = load_sam3_processor(
        confidence_threshold=args.confidence,
        device=args.device,
    )

    object_results = []
    selected_object_candidates = []
    for obj_name in objects:
        print(f"\n{'=' * 50}")
        print(f"Processing object: {obj_name}")

        obj_slug = obj_name.replace(" ", "_")
        obj_output_dir = output_root / obj_slug
        mask_dir = obj_output_dir / "mask"
        bbox_dir = obj_output_dir / "bbox"
        mask_dir.mkdir(parents=True, exist_ok=True)
        bbox_dir.mkdir(parents=True, exist_ok=True)

        candidates = segment_candidates_with_sam3(processor, image_pil, obj_name)
        candidates = _deduplicate_candidates(candidates)
        selected, selected_idx = _pick_best_available_candidate(
            candidates,
            selected_object_candidates,
        )
        if selected is not None:
            selected_object_candidates.append(selected)
            if selected_idx > 0:
                print(
                    f"  Top candidate collided; using alternate candidate rank {selected_idx + 1}."
                )

        result = {
            "object": obj_name,
            "success": False,
            "output_dir": str(obj_output_dir),
            "mask_file": f"mask/{frame_name}.png",
        }

        if selected is None:
            print(f"  Failed to segment: {obj_name}")
            object_results.append(result)
            continue

        mask = selected["mask"]
        bbox = selected.get("bbox")
        score = selected.get("score")

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

        mask_path = mask_dir / f"{frame_name}.png"
        save_mask(mask, mask_path)
        print(f"  Saved mask: {mask_path}")

        result["success"] = True
        if score is not None:
            result["confidence"] = float(score)

        if bbox is not None:
            bbox_img_path = bbox_dir / f"{frame_name}.png"
            save_bbox_image(image_rgb, bbox, obj_name, bbox_img_path)
            bbox_json_path = bbox_dir / f"{frame_name}.json"
            with bbox_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "object": obj_name,
                        "source_frame": frame_name,
                        "source_image": str(frame_path),
                        "source_video": str(video_path),
                        "image_size": [w, h],
                        "bbox": bbox,
                        "confidence": score,
                    },
                    f,
                    indent=2,
                )
            result["bbox_file"] = f"bbox/{frame_name}.png"
            result["bbox_json"] = f"bbox/{frame_name}.json"

        object_results.append(result)

    print(f"\n{'=' * 50}")
    print("Processing human prompt: person")
    human_output_dir = output_root / "human"
    human_mask_dir = human_output_dir / "mask"
    human_bbox_dir = human_output_dir / "bbox"
    human_mask_dir.mkdir(parents=True, exist_ok=True)
    human_bbox_dir.mkdir(parents=True, exist_ok=True)
    human_result = {
        "success": False,
        "output_dir": str(human_output_dir),
        "mask_file": f"mask/{frame_name}.png",
        "prompt": "person",
    }

    human_candidates = segment_candidates_with_sam3(processor, image_pil, "person")
    human_candidates = _deduplicate_candidates(human_candidates)
    if human_candidates:
        best_human = human_candidates[0]
        human_mask_path = human_mask_dir / f"{frame_name}.png"
        save_mask(best_human["mask"], human_mask_path)
        human_result["success"] = True

        human_bbox = best_human.get("bbox")
        if human_bbox is not None:
            x1, y1, x2, y2 = human_bbox
            human_bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            human_bbox_img_path = human_bbox_dir / f"{frame_name}.png"
            save_bbox_image(image_rgb, human_bbox, "human", human_bbox_img_path)
            human_bbox_json_path = human_bbox_dir / f"{frame_name}.json"
            with human_bbox_json_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "object": "human",
                        "source_frame": frame_name,
                        "source_image": str(frame_path),
                        "source_video": str(video_path),
                        "image_size": [w, h],
                        "bbox": human_bbox,
                        "confidence": best_human.get("score"),
                        "prompt": "person",
                    },
                    f,
                    indent=2,
                )
            human_result["bbox_file"] = f"bbox/{frame_name}.png"
            human_result["bbox_json"] = f"bbox/{frame_name}.json"

        human_score = best_human.get("score")
        if human_score is not None:
            human_result["confidence"] = float(human_score)
        print(f"  Saved human mask: {human_mask_path}")
    else:
        print("  Failed to segment human.")

    summary_path = output_root / f"{frame_name}_segmentation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_frame": frame_name,
                "source_image": str(frame_path),
                "source_video": str(video_path),
                "pag_file": str(pag_path),
                "objects": object_results,
                "human": human_result,
            },
            f,
            indent=2,
        )

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
