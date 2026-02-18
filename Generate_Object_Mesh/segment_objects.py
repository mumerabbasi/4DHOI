"""Segment objects using SAM3 with text prompts.

This script uses SAM3 (official Meta implementation) to segment objects
from the first frame of a video based on text prompts from a PAG file.

Run this script with the sam3 environment.

Pipeline:
    1. Parse PAG file to get object names
    2. Read the first frame from a video
    3. Use SAM3 to segment each object with text prompts
    4. Save outputs to:
       - objects/<object_name>/bbox/frame_xx.png  (bbox visualization)
       - objects/<object_name>/bbox/frame_xx.json (bbox metadata)
       - objects/<object_name>/mask/frame_xx.png  (binary mask)
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# SAM3 imports (official Meta implementation)
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def parse_pag_objects(pag_path: str) -> list[str]:
    """Extract unique object names from PAG file."""
    with open(pag_path) as f:
        pag = json.load(f)

    objects = set()
    for node in pag.get("object part nodes", []):
        obj_name = node.split(", ", 1)[0]
        objects.add(obj_name)

    return sorted(objects)


def load_sam3_model(
    confidence_threshold: float = 0.25,
    device: str = "cuda",
):
    """
    Load SAM3 model from official Meta repository.

    Args:
        confidence_threshold: Minimum confidence for detections.
        device: Device to run model on ('cuda', 'cpu', or 'cuda:0', etc.).

    Returns:
        Tuple of (model, processor).
    """
    print(f"Loading SAM3 model on {device}...")
    model = build_sam3_image_model()
    model = model.to(device)
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    print("SAM3 model loaded successfully")
    return model, processor


def _bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    """Compute [x1, y1, x2, y2] bbox from binary mask."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU between two xyxy boxes."""
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


def _deduplicate_candidates(
    candidates: list[dict],
    iou_threshold: float = 0.85,
) -> list[dict]:
    """Remove near-duplicate candidates while preserving best score."""
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


def segment_candidates_with_sam3(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
) -> list[dict]:
    """
    Segment object using SAM3 and return all candidate detections.

    Returns:
        List of dicts with keys: mask, bbox, score
    """
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or len(masks) == 0:
        processor.reset_all_prompts(inference_state)
        return []

    candidates = []
    num_masks = len(masks)
    for idx in range(num_masks):
        mask = masks[idx]
        if hasattr(mask, "detach"):
            mask = mask.detach()
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        mask = (mask > 0.5).astype(np.uint8)
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


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
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
    """Check if candidate overlaps an already assigned instance."""
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
    """Select the best non-colliding candidate; fallback to top candidate."""
    if not candidates:
        return None, -1

    for idx, candidate in enumerate(candidates):
        if not _has_collision(candidate, used_candidates):
            return candidate, idx

    return candidates[0], 0


def save_mask_image(mask: np.ndarray, output_path: Path) -> None:
    """Save binary mask as PNG."""
    cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))


def save_bbox_image(
    image_rgb: np.ndarray,
    bbox: list[int],
    object_name: str,
    output_path: Path,
) -> None:
    """Save image with bounding box drawn."""
    result = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        result,
        object_name,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output_path), result)


def find_default_pag_file(video_name: str, script_dir: Path) -> Path | None:
    """Find default PAG file for a video directory name."""
    pag_dir = (script_dir.parent / "Generate_PAG" / "pags" / video_name).resolve()
    if not pag_dir.exists():
        return None

    output_pag_files = sorted(pag_dir.glob("output_pag*.json"))
    if output_pag_files:
        return output_pag_files[0]

    json_files = sorted(pag_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    return None


def find_single_video(video_dir: Path) -> Path | None:
    """Find the single MP4 in a video directory."""
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if len(mp4_files) != 1:
        return None
    return mp4_files[0]


def extract_first_frame(video_path: Path) -> np.ndarray | None:
    """Read the first frame from a video as an RGB image."""
    capture = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = capture.read()
    capture.release()

    if not ok or frame_bgr is None:
        return None

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects using SAM3 (run with sam3 env)."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="../Generate_Video/videos/video_02",
        help="Directory containing exactly one MP4 (e.g., .../video_01).",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help=(
            "PAG JSON file. Default: "
            "../Generate_PAG/pags/<video_xx>/output_pag*.json"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for masks. Default: ./output/<video_xx>.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="SAM3 confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda, cpu, cuda:0, etc.).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    video_dir = Path(args.video_dir)
    if not video_dir.is_absolute():
        video_dir = script_dir / video_dir
    video_dir = video_dir.resolve()
    video_name = video_dir.name

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"Error: Video directory not found: {video_dir}")
        return

    video_path = find_single_video(video_dir)
    if video_path is None:
        mp4_files = sorted(video_dir.glob("*.mp4"))
        print(
            f"Error: Expected exactly one MP4 in {video_dir}, "
            f"found {len(mp4_files)}."
        )
        return

    if args.pag_file is not None:
        pag_path = Path(args.pag_file)
        if not pag_path.is_absolute():
            pag_path = script_dir / pag_path
        pag_path = pag_path.resolve()
    else:
        pag_path = find_default_pag_file(video_name, script_dir)
        if pag_path is None:
            print(
                "Error: Could not resolve default PAG file. Expected one of:\n"
                f"  1) ../Generate_PAG/pags/{video_name}/output_pag*.json\n"
                f"  2) A single ../Generate_PAG/pags/{video_name}/*.json\n"
                "Please pass --pag_file explicitly."
            )
            return

    if args.output_dir is not None:
        output_root = Path(args.output_dir)
        if not output_root.is_absolute():
            output_root = script_dir / output_root
    else:
        output_root = script_dir / "output" / video_name
    output_root = output_root.resolve()

    if not pag_path.exists():
        print(f"Error: PAG file not found: {pag_path}")
        return

    frame_name = "frame_00"
    image_rgb = extract_first_frame(video_path)
    if image_rgb is None:
        print(f"Error: Could not read first frame from video: {video_path}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    source_image_path = output_root / f"{frame_name}.png"
    cv2.imwrite(
        str(source_image_path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
    )
    image_pil = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    # Get objects from PAG
    objects = parse_pag_objects(str(pag_path))
    print(f"Objects to process: {objects}")
    print(f"Source frame: {frame_name}")
    print(f"Source video: {video_path}")
    print(f"Saved source frame: {source_image_path}")

    # Load SAM3 model
    _, sam3_processor = load_sam3_model(
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Process each object
    results_summary = []
    selected_candidates = []
    for obj_name in objects:
        print(f"\n{'='*50}")
        print(f"Processing: {obj_name}")

        dir_name = obj_name.replace(" ", "_")
        obj_output_dir = output_root / dir_name
        obj_output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for new structure
        bbox_dir = obj_output_dir / "bbox"
        mask_dir = obj_output_dir / "mask"
        bbox_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Segment with SAM3 and resolve collisions using alternate candidates
        print("  Segmenting with SAM3...")
        candidates = segment_candidates_with_sam3(sam3_processor, image_pil, obj_name)
        candidates = _deduplicate_candidates(candidates)
        selected, selected_idx = _pick_best_available_candidate(
            candidates,
            selected_candidates,
        )

        if selected is not None:
            selected_candidates.append(selected)
            if selected_idx > 0:
                print(
                    f"  Top candidate collided; using alternate candidate rank {selected_idx + 1}."
                )

        if selected is None:
            mask, bbox, score = None, None, None
        else:
            mask = selected.get("mask")
            bbox = selected.get("bbox")
            score = selected.get("score")

        if mask is None:
            print(f"  Failed to segment {obj_name}")
            results_summary.append({
                "object": obj_name,
                "success": False,
            })
            continue

        print(f"  Detection confidence: {score:.3f}" if score else "  Detected")

        # Clamp bbox if present
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
            print(f"  Bbox: {bbox}")

        # Save mask to mask/ subdir
        mask_filename = f"{frame_name}.png"
        save_mask_image(mask, mask_dir / mask_filename)
        print(f"Saved: mask/{mask_filename}")

        # Save bbox visualization to bbox/ subdir
        bbox_img_filename = None
        if bbox is not None:
            bbox_img_filename = f"{frame_name}.png"
            save_bbox_image(image_rgb, bbox, obj_name, bbox_dir / bbox_img_filename)
            print(f"Saved: bbox/{bbox_img_filename}")

            # Save bbox metadata JSON to bbox/ subdir
            bbox_metadata = {
                "object": obj_name,
                "source_frame": frame_name,
                "source_image": str(source_image_path),
                "source_video": str(video_path),
                "image_size": [w, h],
                "bbox": bbox,
                "confidence": score,
            }
            bbox_json_filename = f"{frame_name}.json"
            with open(bbox_dir / bbox_json_filename, "w") as f:
                json.dump(bbox_metadata, f, indent=2)
            print(f"    Saved: bbox/{bbox_json_filename}")

        results_summary.append({
            "object": obj_name,
            "success": True,
            "output_dir": str(obj_output_dir),
            "mask_file": f"mask/{mask_filename}",
            "bbox_file": f"bbox/{bbox_img_filename}" if bbox_img_filename else None,
            "bbox_json": f"bbox/{frame_name}.json" if bbox is not None else None,
        })

    # Save overall summary
    summary_path = output_root / f"{frame_name}_segmentation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "source_frame": frame_name,
            "source_image": str(source_image_path),
            "source_video": str(video_path),
            "pag_file": str(pag_path),
            "objects": results_summary,
        }, f, indent=2)
    print(f"\n{'='*50}")
    print(f"Summary saved to: {summary_path}")
    print("Done! Now run generate_meshes.py with sam3d-objects env.")


if __name__ == "__main__":
    main()
