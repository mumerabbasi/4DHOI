"""Segment objects using SAM3 with text prompts.

This script uses SAM3 (official Meta implementation) to segment objects
from an image based on text prompts from a PAG file.

Run this script with the sam3 environment.

Pipeline:
    1. Parse PAG file to get object names
    2. Use SAM3 to segment each object with text prompts
    3. Save outputs to:
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


def segment_with_sam3(
    processor: Sam3Processor,
    image: Image.Image,
    object_name: str,
) -> tuple[np.ndarray | None, list[int] | None, float | None]:
    """
    Segment object using SAM3 with text prompt.

    Args:
        processor: SAM3 processor.
        image: PIL Image to segment.
        object_name: Name of the object to segment.

    Returns:
        Tuple of (mask, bbox, score) or (None, None, None) if not detected.
    """
    inference_state = processor.set_image(image)

    # Create descriptive prompt for better detection
    if object_name.lower() == "iron":
        prompt = "clothes iron appliance"
    elif object_name.lower() == "ironing board":
        prompt = "ironing board furniture"
    else:
        prompt = object_name

    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    if masks is None or len(masks) == 0:
        return None, None, None

    # Get the best detection
    best_idx = 0
    if scores is not None and len(scores) > 0:
        best_idx = scores.argmax().item() if hasattr(scores, 'argmax') else 0

    # Extract mask
    mask = masks[best_idx]
    if hasattr(mask, 'cpu'):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    mask = (mask > 0.5).astype(np.uint8)

    # Extract bbox
    bbox = None
    if boxes is not None and len(boxes) > best_idx:
        box = boxes[best_idx]
        if hasattr(box, 'cpu'):
            box = box.cpu().numpy()
        bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

    # Extract score
    score = None
    if scores is not None and len(scores) > best_idx:
        score = scores[best_idx]
        if hasattr(score, 'item'):
            score = score.item()

    processor.reset_all_prompts(inference_state)

    return mask, bbox, score


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


def save_masked_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save image with mask applied."""
    result = image_rgb.copy()
    result[mask == 0] = 0
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects using SAM3 (run with sam3 env)."
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default="../Generate_PAG/output_pag_deepseek_r1_32b.json",
        help="PAG JSON file.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="../Generate_Video/first_frames/frame_00.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="objects",
        help="Output directory for masks.",
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

    # Resolve paths
    image_path = Path(args.image).resolve()
    pag_path = Path(args.pag_file).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(__file__).parent / output_root

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    if not pag_path.exists():
        print(f"Error: PAG file not found: {pag_path}")
        return

    # Extract frame name from image path for reference
    frame_name = image_path.stem  # e.g., "frame_00"

    # Get objects from PAG
    objects = parse_pag_objects(str(pag_path))
    print(f"Objects to process: {objects}")
    print(f"Source frame: {frame_name}")

    # Load SAM3 model
    print(f"\nLoading SAM3 on {args.device}...")
    _, sam3_processor = load_sam3_model(
        confidence_threshold=args.confidence,
        device=args.device,
    )

    # Load image
    image_pil = Image.open(str(image_path)).convert("RGB")
    image_rgb = np.array(image_pil)
    h, w = image_rgb.shape[:2]

    # Process each object
    results_summary = []
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

        # Segment with SAM3
        print("  Segmenting with SAM3...")
        mask, bbox, score = segment_with_sam3(sam3_processor, image_pil, obj_name)

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
        print(f"    Saved: mask/{mask_filename}")

        # Save bbox visualization to bbox/ subdir
        bbox_img_filename = None
        if bbox is not None:
            bbox_img_filename = f"{frame_name}.png"
            save_bbox_image(image_rgb, bbox, obj_name, bbox_dir / bbox_img_filename)
            print(f"    Saved: bbox/{bbox_img_filename}")

            # Save bbox metadata JSON to bbox/ subdir
            bbox_metadata = {
                "object": obj_name,
                "source_frame": frame_name,
                "source_image": str(image_path),
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
            "source_image": str(image_path),
            "pag_file": str(pag_path),
            "objects": results_summary,
        }, f, indent=2)
    print(f"\n{'='*50}")
    print(f"Summary saved to: {summary_path}")
    print("Done! Now run generate_meshes.py with sam3d-objects env.")


if __name__ == "__main__":
    main()
