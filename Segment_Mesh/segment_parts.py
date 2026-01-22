"""
Segment object parts using Qwen-VL for detection and SAM2 for segmentation.

Pipeline:
1. Load PAG file to get all objects and their parts
2. For each object, load rendered images from objects/<name>/renders/
3. Use Qwen-VL (via Ollama) to detect bounding boxes for each part
4. Use SAM2 to generate segmentation masks from bounding boxes
5. Save masks to objects/<name>/masks/


Usage:
    python segment_parts.py --pag_file ../Generate_PAG/output_pag_deepseek_r1_32b.json
    python segment_parts.py --object_dir objects/iron --parts "handle,soleplate"
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

# SAM2 imports
sys.path.insert(0, "/my_workspace/4DHHOI/sam2")  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402


# Ollama configuration
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL = "qwen3-vl:32b"

# SAM2 configuration
SAM2_CHECKPOINT = "/my_workspace/4DHHOI/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_parts_qwen(
    image_path: str,
    parts: list[str],
    object_name: str,
) -> dict[str, list[int] | None]:
    """
    Use Qwen-VL to detect bounding boxes for object parts.

    Args:
        image_path: Path to the image.
        parts: List of part names to detect (e.g., ["handle", "soleplate"]).
        object_name: Name of the object (e.g., "iron").

    Returns:
        Dictionary mapping part names to bounding boxes [x1, y1, x2, y2] or None.
    """
    image_b64 = encode_image_base64(image_path)

    # Build prompt for all parts at once
    parts_str = ", ".join(parts)
    prompt = f"""You are analyzing an image of a {object_name}. Detect bounding boxes
    for these parts: {parts_str}

    For each part that is VISIBLE in the image, provide the bounding box coordinates
    as [x1, y1, x2, y2] where:
    - x1, y1 = top-left corner (pixels from left/top)
    - x2, y2 = bottom-right corner (pixels from left/top)

    Respond with a JSON object in this exact format:
    {{
        "part_name": [x1, y1, x2, y2],
        "another_part": [x1, y1, x2, y2],
        "not_visible_part": null
    }}

    If a part is not visible or cannot be detected, set its value to null.
    Coordinates should be integers representing pixel positions.

    Detect bounding boxes for: {parts_str}"""

    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")

        # Remove think tags if present
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            boxes = json.loads(json_match.group())
            # Normalize keys and validate
            normalized = {}
            for part in parts:
                # Try exact match first, then case-insensitive
                if part in boxes:
                    normalized[part] = boxes[part]
                else:
                    for key, value in boxes.items():
                        if key.lower() == part.lower():
                            normalized[part] = value
                            break
                    else:
                        normalized[part] = None
            return normalized
        else:
            print(f"  Warning: Could not parse JSON from response: {response_text[:200]}")
            return {part: None for part in parts}

    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {part: None for part in parts}


def load_sam2_predictor() -> SAM2ImagePredictor:
    """Load SAM2 model."""
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_model = build_sam2(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=device,
    )
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"SAM2 loaded on {device}")
    return predictor


def segment_with_sam2(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """
    Segment object using SAM2 with bounding box prompt.

    Args:
        predictor: SAM2 predictor.
        image: RGB image as numpy array.
        bbox: Bounding box [x1, y1, x2, y2].

    Returns:
        Binary mask as numpy array (H, W).
    """
    predictor.set_image(image)

    # Convert to numpy array for SAM2
    box = np.array(bbox)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],  # Add batch dimension
        multimask_output=False,
    )

    # Return the mask (squeeze batch and channel dims)
    return masks[0].astype(np.uint8)


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG."""
    # Scale to 0-255 for visualization
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_img)


def process_object(
    object_dir: Path,
    parts: list[str],
    sam_predictor: SAM2ImagePredictor,
) -> dict:
    """
    Process all renders of an object.

    Args:
        object_dir: Path to object directory.
        parts: List of part names to segment.
        sam_predictor: SAM2 predictor.

    Returns:
        Dictionary with segmentation results.
    """
    renders_dir = object_dir / "renders"
    masks_dir = object_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    object_name = object_dir.name

    # Get all render images
    image_files = sorted(renders_dir.glob("rgb_*.png"))
    if not image_files:
        print(f"No render images found in {renders_dir}")
        return {}

    print(f"\nProcessing {object_name} with parts: {parts}")
    print(f"Found {len(image_files)} images")

    results = {
        "object_name": object_name,
        "parts": parts,
        "views": []
    }

    for img_path in image_files:
        print(f"\n  Processing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Detect bounding boxes with Qwen-VL
        print("Detecting parts with Qwen-VL...")
        boxes = detect_parts_qwen(str(img_path), parts, object_name)

        view_result = {
            "image": img_path.name,
            "image_size": [w, h],
            "parts": {}
        }

        # Segment each detected part
        for part_name, bbox in boxes.items():
            if bbox is None:
                print(f"    {part_name}: not detected")
                view_result["parts"][part_name] = {
                    "detected": False,
                    "bbox": None,
                    "mask_file": None
                }
                continue

            # Validate and clamp bbox
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            bbox = [x1, y1, x2, y2]

            print(f"    {part_name}: bbox={bbox}")

            # Segment with SAM2
            mask = segment_with_sam2(sam_predictor, image_rgb, bbox)

            # Save mask
            mask_filename = f"{img_path.stem}_{part_name.replace(' ', '_')}.png"
            mask_path = masks_dir / mask_filename
            save_mask(mask, str(mask_path))
            print(f"      Saved: {mask_filename}")

            view_result["parts"][part_name] = {
                "detected": True,
                "bbox": bbox,
                "mask_file": mask_filename
            }

        results["views"].append(view_result)

    # Save results JSON
    results_path = masks_dir / "part_labels.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    return results


def parse_pag_file(pag_path: str) -> dict[str, list[str]]:
    """
    Parse PAG file to extract objects and their parts.

    Args:
        pag_path: Path to PAG JSON file.

    Returns:
        Dictionary mapping object names to list of part names.
        e.g., {"iron": ["handle", "soleplate"], "ironing board": ["surface", "support"]}
    """
    with open(pag_path) as f:
        pag = json.load(f)

    objects_parts = {}
    for node in pag.get("object part nodes", []):
        # Format: "object_name, part_name"
        parts = node.split(", ", 1)
        if len(parts) == 2:
            obj_name, part_name = parts
            if obj_name not in objects_parts:
                objects_parts[obj_name] = []
            objects_parts[obj_name].append(part_name)

    return objects_parts


def main():
    parser = argparse.ArgumentParser(
        description="Segment object parts using Qwen-VL + SAM2."
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="PAG JSON file to extract all objects and parts automatically.",
    )
    parser.add_argument(
        "--objects_root",
        type=str,
        default="objects",
        help="Root directory containing object folders (default: objects).",
    )
    parser.add_argument(
        "--object_dir",
        type=str,
        default=None,
        help="Path to single object directory (e.g., objects/iron).",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default=None,
        help="Comma-separated list of part names (e.g., 'handle,soleplate').",
    )

    args = parser.parse_args()

    # Determine what to process
    if args.pag_file:
        # Process all objects from PAG file
        pag_path = Path(args.pag_file).resolve()
        if not pag_path.exists():
            print(f"Error: PAG file not found: {pag_path}")
            return

        objects_parts = parse_pag_file(str(pag_path))
        if not objects_parts:
            print("No object parts found in PAG file")
            return

        print(f"Found {len(objects_parts)} objects in PAG file:")
        for obj, parts in objects_parts.items():
            print(f"  {obj}: {parts}")

        # Get objects root directory
        objects_root = Path(args.objects_root).resolve()
        if not objects_root.exists():
            print(f"Error: Objects root directory not found: {objects_root}")
            return

        # Load SAM2 once for all objects
        sam_predictor = load_sam2_predictor()

        # Process each object
        for obj_name, parts in objects_parts.items():
            # Convert object name to directory name (replace spaces with underscores)
            dir_name = obj_name.replace(" ", "_")
            object_dir = objects_root / dir_name

            if not object_dir.exists():
                print(f"\nWarning: Object directory not found: {object_dir}, skipping...")
                continue

            if not (object_dir / "renders").exists():
                print(f"\nWarning: No renders found for {obj_name}, skipping...")
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {obj_name}")
            print(f"{'='*60}")
            process_object(object_dir, parts, sam_predictor)

    elif args.object_dir and args.parts:
        # Process single object with specified parts
        object_dir = Path(args.object_dir).resolve()
        if not object_dir.exists():
            print(f"Error: Object directory not found: {object_dir}")
            return

        parts = [p.strip() for p in args.parts.split(",")]
        print(f"Object: {object_dir.name}")
        print(f"Parts to segment: {parts}")

        sam_predictor = load_sam2_predictor()
        process_object(object_dir, parts, sam_predictor)

    else:
        print("Error: Provide either --pag_file or both --object_dir and --parts")
        parser.print_help()
        return

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
