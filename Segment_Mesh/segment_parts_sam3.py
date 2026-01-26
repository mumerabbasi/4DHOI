"""
Segment object parts using SAM3 with text prompts.

Pipeline:
1. Load PAG file to get all objects and their parts
2. For each object, load rendered images from objects/<name>/renders/
3. Use SAM3 to segment parts directly from text prompts
4. Save masks to objects/<name>/masks/

Usage:
    python segment_parts_sam3.py --pag_file ../Generate_PAG/output_pag_deepseek_r1_32b.json
    python segment_parts_sam3.py --object_dir objects/iron --parts "handle,soleplate"
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ultralytics.models.sam import SAM3SemanticPredictor


# SAM3 configuration
SAM3_CHECKPOINT = "/my_workspace/4DHHOI/models/sam3.pt"


def load_sam3_predictor() -> SAM3SemanticPredictor:
    """Load SAM3 model with semantic prediction capabilities."""
    print("Loading SAM3 model...")

    overrides = dict(
        conf=0.25,
        task="segment",
        mode="predict",
        model=SAM3_CHECKPOINT,
        half=True,  # Use FP16 for faster inference
        save=False,  # We'll handle saving ourselves
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print("SAM3 loaded")
    return predictor


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG."""
    # Scale to 0-255 for visualization
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_img)


def draw_masks_on_image(
    image: np.ndarray,
    masks: dict[str, np.ndarray | None],
) -> np.ndarray:
    """
    Draw segmentation masks on image with labels.

    Args:
        image: BGR image as numpy array.
        masks: Dictionary mapping part names to masks or None.

    Returns:
        Image with masks overlaid.
    """
    result = image.copy()

    # Color palette for different parts
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Light green
        (255, 128, 0),  # Light blue
    ]

    for i, (part_name, mask) in enumerate(masks.items()):
        if mask is None:
            continue

        color = colors[i % len(colors)]

        # Create colored overlay
        overlay = result.copy()
        overlay[mask > 0] = color

        # Blend with original
        alpha = 0.4
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Find centroid for label placement
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))

            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                part_name, font, font_scale, thickness
            )

            # Draw text background
            cv2.rectangle(
                result,
                (cx - 2, cy - text_h - 4),
                (cx + text_w + 4, cy + 4),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                result, part_name, (cx, cy),
                font, font_scale, (0, 0, 0), thickness
            )

    return result


def segment_parts_sam3(
    predictor: SAM3SemanticPredictor,
    image_path: str,
    parts: list[str],
    object_name: str,
) -> dict[str, np.ndarray | None]:
    """
    Segment object parts using SAM3 with text prompts.

    Args:
        predictor: SAM3 predictor.
        image_path: Path to the image.
        parts: List of part names to segment (e.g., ["handle", "soleplate"]).
        object_name: Name of the object (e.g., "iron").

    Returns:
        Dictionary mapping part names to binary masks or None if not detected.
    """
    # Set the image for prediction
    predictor.set_image(image_path)

    # Create descriptive prompts for each part
    text_prompts = [f"{part} of {object_name}" for part in parts]

    # Query SAM3 with all parts at once
    results = predictor(text=text_prompts)

    # Process results
    masks = {}
    if results and len(results) > 0:
        result = results[0]  # Get first result

        if hasattr(result, 'masks') and result.masks is not None:
            # SAM3 returns masks for each text prompt
            for i, part in enumerate(parts):
                if i < len(result.masks.data):
                    mask = result.masks.data[i].cpu().numpy()
                    # Check if mask has any content
                    if mask.sum() > 0:
                        masks[part] = mask.astype(np.uint8)
                    else:
                        masks[part] = None
                else:
                    masks[part] = None
        else:
            # No masks found
            for part in parts:
                masks[part] = None
    else:
        for part in parts:
            masks[part] = None

    return masks


def process_object(
    object_dir: Path,
    parts: list[str],
    sam_predictor: SAM3SemanticPredictor,
) -> dict:
    """
    Process all renders of an object.

    Args:
        object_dir: Path to object directory.
        parts: List of part names to segment.
        sam_predictor: SAM3 predictor.

    Returns:
        Dictionary with segmentation results.
    """
    renders_dir = object_dir / "renders"
    masks_dir = object_dir / "masks_sam3"
    viz_dir = object_dir / "visualizations"
    masks_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    object_name = object_dir.name.replace("_", " ")

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

        # Load image for visualization
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]

        # Segment parts with SAM3
        print("    Segmenting parts with SAM3...")
        masks = segment_parts_sam3(sam_predictor, str(img_path), parts, object_name)

        # Save visualization with masks overlaid
        viz_image = draw_masks_on_image(image, masks)
        viz_filename = f"{img_path.stem}_segmented.png"
        viz_path = viz_dir / viz_filename
        cv2.imwrite(str(viz_path), viz_image)
        print(f"    Saved visualization: {viz_filename}")

        view_result = {
            "image": img_path.name,
            "image_size": [w, h],
            "parts": {}
        }

        # Save each part mask
        for part_name, mask in masks.items():
            if mask is None:
                print(f"    {part_name}: not detected")
                view_result["parts"][part_name] = {
                    "detected": False,
                    "mask_file": None
                }
                continue

            print(f"    {part_name}: detected (mask area: {mask.sum()} pixels)")

            # Save mask
            mask_filename = f"{img_path.stem}_{part_name.replace(' ', '_')}.png"
            mask_path = masks_dir / mask_filename
            save_mask(mask, str(mask_path))
            print(f"      Saved: {mask_filename}")

            view_result["parts"][part_name] = {
                "detected": True,
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
        description="Segment object parts using SAM3 with text prompts."
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default="../Generate_PAG/output_pag_deepseek_r1_32b.json",
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

        # Load SAM3 once for all objects
        sam_predictor = load_sam3_predictor()

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

        sam_predictor = load_sam3_predictor()
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
