"""
Segment objects and parts from generated video using Qwen-VL + SAM2 video predictor.

Algorithm (from 3.3):
1. Detect objects in frame 1 using open-vocabulary detection (Qwen-VL)
2. Use SAM2 to track/segment objects across frames → object masks {M^o_t}
3. Similarly obtain part masks {M^p_t} for each object part

Output structure:
    Segment_Video/
    └── video_xx/
        └── <object_name>/
            ├── object_segmentation/
            │   ├── masks/       # Binary masks per frame
            │   └── visualizations/  # Overlay visualizations
            └── parts_segmentation/
                ├── masks/       # Binary masks per frame per part
                └── visualizations/  # Overlay visualizations

Usage:
    python segment_video.py --video_path ../Generate_Video/videos/video_01 \
                            --pag_file ../Generate_PAG/pags/video_01/output_pag_deepseek_r1_32b.json

    python segment_video.py --video_path ../Generate_Video/videos/video_01 \
                            --object iron --parts "handle,soleplate"
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
from sam2.build_sam import build_sam2_video_predictor  # noqa: E402


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


def encode_numpy_image_base64(image: np.ndarray) -> str:
    """Encode numpy BGR image to base64 string."""
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def detect_objects_qwen(
    image: np.ndarray,
    objects: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for objects in the first frame.

    Args:
        image: BGR image as numpy array.
        objects: List of object names to detect.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Dictionary mapping object names to bounding boxes or None.
    """
    image_b64 = encode_numpy_image_base64(image)

    # Build detection request
    objects_request = "\n".join([f"- {obj}" for obj in objects])
    prompt = f"""You are analyzing an image. Detect bounding boxes for these objects:
{objects_request}

For each object, output in this format:
<ref>object_name</ref><box>[[x1,y1,x2,y2]]</box>

Coordinates should be on a 0-1000 normalized scale (not pixels).
If an object is not visible, output: <ref>object_name</ref><box>null</box>

Detect all objects listed above."""

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
        response_text = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        )

        # Parse all <ref>object</ref><box>[[x1,y1,x2,y2]]</box> patterns
        normalized = {obj: None for obj in objects}

        pattern = (
            r"<ref>([^<]+)</ref>\s*<box>\s*"
            r"(\[\s*\[?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]?\s*\]|null)"
            r"\s*</box>"
        )
        matches = re.findall(pattern, response_text, re.IGNORECASE)

        for ref_name, box_str in matches:
            ref_name = ref_name.strip()
            matched_obj = None
            for obj in objects:
                if obj.lower() == ref_name.lower():
                    matched_obj = obj
                    break

            if matched_obj is None:
                continue

            if box_str.strip().lower() == "null":
                normalized[matched_obj] = None
            else:
                coord_match = re.search(
                    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", box_str
                )
                if coord_match:
                    x1 = int(int(coord_match.group(1)) * image_width / 1000)
                    y1 = int(int(coord_match.group(2)) * image_height / 1000)
                    x2 = int(int(coord_match.group(3)) * image_width / 1000)
                    y2 = int(int(coord_match.group(4)) * image_height / 1000)
                    normalized[matched_obj] = [x1, y1, x2, y2]

        return normalized

    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {obj: None for obj in objects}


def detect_parts_qwen(
    image: np.ndarray,
    parts: list[str],
    object_name: str,
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for object parts.

    Args:
        image: BGR image as numpy array.
        parts: List of part names to detect.
        object_name: Name of the object (e.g., "iron").
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Dictionary mapping part names to bounding boxes or None.
    """
    image_b64 = encode_numpy_image_base64(image)

    parts_request = "\n".join([f"- {part}" for part in parts])
    prompt = f"""You are analyzing an image of a {object_name}. \
Detect bounding boxes for these parts:
{parts_request}

For each part, output in this format:
<ref>part_name</ref><box>[[x1,y1,x2,y2]]</box>

Coordinates should be on a 0-1000 normalized scale (not pixels).
If a part is not visible, output: <ref>part_name</ref><box>null</box>

Detect all parts listed above."""

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

        response_text = re.sub(
            r"<think>.*?</think>", "", response_text, flags=re.DOTALL
        )

        normalized = {part: None for part in parts}

        pattern = (
            r"<ref>([^<]+)</ref>\s*<box>\s*"
            r"(\[\s*\[?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]?\s*\]|null)"
            r"\s*</box>"
        )
        matches = re.findall(pattern, response_text, re.IGNORECASE)

        for ref_name, box_str in matches:
            ref_name = ref_name.strip()
            matched_part = None
            for part in parts:
                if part.lower() == ref_name.lower():
                    matched_part = part
                    break

            if matched_part is None:
                continue

            if box_str.strip().lower() == "null":
                normalized[matched_part] = None
            else:
                coord_match = re.search(
                    r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", box_str
                )
                if coord_match:
                    x1 = int(int(coord_match.group(1)) * image_width / 1000)
                    y1 = int(int(coord_match.group(2)) * image_height / 1000)
                    x2 = int(int(coord_match.group(3)) * image_width / 1000)
                    y2 = int(int(coord_match.group(4)) * image_height / 1000)
                    normalized[matched_part] = [x1, y1, x2, y2]

        return normalized

    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {part: None for part in parts}


def load_sam2_video_predictor():
    """Load SAM2 video predictor."""
    print("Loading SAM2 video predictor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(
        SAM2_CONFIG,
        SAM2_CHECKPOINT,
        device=device,
    )
    print(f"SAM2 video predictor loaded on {device}")
    return predictor


def extract_video_frames(video_path: str, output_dir: Path) -> tuple[list[Path], int, int]:
    """Extract frames from video file to JPEG files.

    Args:
        video_path: Path to video file.
        output_dir: Directory to save frames.

    Returns:
        Tuple of (list of frame paths, video width, video height).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = output_dir / f"frame_{frame_idx:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_idx += 1

    cap.release()
    print(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths, width, height


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG."""
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_img)


def create_visualization(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    alpha: float = 0.4,
) -> np.ndarray:
    """Create visualization with masks overlaid on frame.

    Args:
        frame: BGR frame as numpy array.
        masks: Dictionary mapping names to binary masks.
        alpha: Transparency for mask overlay.

    Returns:
        Visualization image.
    """
    result = frame.copy()

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

    for i, (name, mask) in enumerate(masks.items()):
        if mask is None:
            continue

        color = colors[i % len(colors)]
        overlay = result.copy()
        overlay[mask > 0] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Draw label at centroid
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(name, font, font_scale, thickness)
            cv2.rectangle(
                result,
                (cx - 2, cy - text_h - 4),
                (cx + text_w + 4, cy + 4),
                color,
                -1
            )
            cv2.putText(result, name, (cx, cy), font, font_scale, (0, 0, 0), thickness)

    return result


def track_with_sam2(
    predictor,
    frames_dir: Path,
    bboxes: dict[str, list[int] | None],
    output_masks_dir: Path,
    output_viz_dir: Path,
    item_type: str = "object",
    use_subdirs: bool = True,
) -> dict:
    """Track objects/parts across video frames using SAM2.

    Args:
        predictor: SAM2 video predictor.
        frames_dir: Directory containing extracted JPEG frames.
        bboxes: Dictionary mapping names to bounding boxes (detected in frame 0).
        output_masks_dir: Directory to save masks.
        output_viz_dir: Directory to save visualizations.
        item_type: "object" or "part" for logging.
        use_subdirs: If True, create subdirectories per item inside masks dir.

    Returns:
        Dictionary with tracking results.
"""
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_viz_dir.mkdir(parents=True, exist_ok=True)

    # Initialize inference state from frames directory
    inference_state = predictor.init_state(video_path=str(frames_dir))
    num_frames = inference_state["num_frames"]
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]

    print(f"  Video: {num_frames} frames, {video_width}x{video_height}")

    # Add bounding box prompts for each detected item on frame 0
    obj_id_to_name = {}
    for obj_id, (name, bbox) in enumerate(bboxes.items()):
        if bbox is None:
            print(f"    Skipping {name}: not detected in frame 0")
            continue

        # Clamp bbox to valid range
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, video_width - 1))
        y1 = max(0, min(y1, video_height - 1))
        x2 = max(x1 + 1, min(x2, video_width))
        y2 = max(y1 + 1, min(y2, video_height))
        bbox = [x1, y1, x2, y2]

        print(f"    Adding {item_type} '{name}' with bbox {bbox}")
        obj_id_to_name[obj_id] = name

        # Add box prompt on frame 0
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=bbox,
        )

    if not obj_id_to_name:
        print(f"  No {item_type}s detected, skipping tracking")
        return {"tracked": False, "items": {}}

    # Propagate through video
    print(f"  Propagating masks through {num_frames} frames...")
    results = {"tracked": True, "items": {}, "num_frames": num_frames}

    # Get frame file list for visualization
    frame_files = sorted(frames_dir.glob("*.png"))

    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
        # video_res_masks shape: (num_objects, 1, H, W)
        masks_dict = {}

        for i, obj_id in enumerate(obj_ids):
            if obj_id not in obj_id_to_name:
                continue

            name = obj_id_to_name[obj_id]
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            masks_dict[name] = mask

            # Determine mask output directory
            if use_subdirs:
                item_mask_dir = output_masks_dir / name.replace(" ", "_")
                item_mask_dir.mkdir(exist_ok=True)
            else:
                item_mask_dir = output_masks_dir

            # Save mask
            mask_filename = f"frame_{frame_idx:04d}.png"
            save_mask(mask, str(item_mask_dir / mask_filename))

            # Store in results
            if name not in results["items"]:
                results["items"][name] = {"masks": [], "bbox_frame0": bboxes[name]}
            results["items"][name]["masks"].append(mask_filename)

        # Create and save visualization
        if frame_idx < len(frame_files):
            frame = cv2.imread(str(frame_files[frame_idx]))
            viz = create_visualization(frame, masks_dict)
            viz_filename = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(output_viz_dir / viz_filename), viz)

    # Reset state for next use
    predictor.reset_state(inference_state)

    return results


def parse_pag_file(pag_path: str) -> dict[str, list[str]]:
    """Parse PAG file to extract objects and their parts.

    Args:
        pag_path: Path to PAG JSON file.

    Returns:
        Dictionary mapping object names to list of part names.
    """
    with open(pag_path) as f:
        pag = json.load(f)

    objects_parts = {}
    for node in pag.get("object part nodes", []):
        parts = node.split(", ", 1)
        if len(parts) == 2:
            obj_name, part_name = parts
            if obj_name not in objects_parts:
                objects_parts[obj_name] = []
            objects_parts[obj_name].append(part_name)

    return objects_parts


def find_video_file(video_dir: Path) -> Path | None:
    """Find the first video file in directory."""
    video_extensions = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV"]
    for ext in video_extensions:
        videos = list(video_dir.glob(f"*{ext}"))
        if videos:
            return videos[0]
    return None


def process_video(
    video_path: Path,
    objects_parts: dict[str, list[str]],
    output_root: Path,
    predictor,
) -> None:
    """Process a video to segment objects and their parts.

    Args:
        video_path: Path to video file.
        objects_parts: Dictionary mapping object names to their parts.
        output_root: Root output directory (e.g., Segment_Video/video_01).
        predictor: SAM2 video predictor.
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"{'='*60}")

    # Extract frames
    frames_dir = output_root / "_frames"
    if not frames_dir.exists() or not list(frames_dir.glob("*.png")):
        frame_paths, width, height = extract_video_frames(str(video_path), frames_dir)
    else:
        frame_paths = sorted(frames_dir.glob("*.png"))
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        print(f"Using existing {len(frame_paths)} frames from {frames_dir}")

    # Load first frame for detection
    first_frame = cv2.imread(str(frames_dir / "frame_0000.png"))
    if first_frame is None:
        print("Error: Could not load first frame")
        return

    # Get all object names
    objects = list(objects_parts.keys())
    print(f"\nObjects to segment: {objects}")

    # Step 1: Detect all objects in frame 0
    print("\n[Step 1] Detecting objects in frame 0 with Qwen-VL...")
    object_bboxes = detect_objects_qwen(first_frame, objects, width, height)

    for obj, bbox in object_bboxes.items():
        if bbox:
            print(f"  {obj}: {bbox}")
        else:
            print(f"  {obj}: NOT DETECTED")

    # Process each object
    for obj_name, parts in objects_parts.items():
        obj_dir_name = obj_name.replace(" ", "_")
        obj_output_dir = output_root / obj_dir_name

        print(f"\n{'='*40}")
        print(f"Processing object: {obj_name}")
        print(f"{'='*40}")

        # Step 2: Track object across frames
        print("\n[Step 2] Tracking object across frames with SAM2...")
        obj_seg_dir = obj_output_dir / "object_segmentation"

        if object_bboxes.get(obj_name) is not None:
            obj_results = track_with_sam2(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes={obj_name: object_bboxes[obj_name]},
                output_masks_dir=obj_seg_dir / "masks",
                output_viz_dir=obj_seg_dir / "visualizations",
                item_type="object",
                use_subdirs=False,  # No subdirs needed for single object
            )

            # Save object segmentation results
            results_path = obj_seg_dir / "segmentation_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(obj_results, f, indent=2)
            print(f"  Saved object results to: {results_path}")
        else:
            print(f"  Skipping object tracking: {obj_name} not detected")

        # Step 3: Detect and track parts
        print(f"\n[Step 3] Detecting parts {parts} in frame 0...")
        part_bboxes = detect_parts_qwen(first_frame, parts, obj_name, width, height)

        for part, bbox in part_bboxes.items():
            if bbox:
                print(f"  {part}: {bbox}")
            else:
                print(f"  {part}: NOT DETECTED")

        # Track parts across frames
        print("\n[Step 4] Tracking parts across frames with SAM2...")
        parts_seg_dir = obj_output_dir / "parts_segmentation"

        # Filter out undetected parts
        detected_parts = {p: b for p, b in part_bboxes.items() if b is not None}

        if detected_parts:
            parts_results = track_with_sam2(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes=detected_parts,
                output_masks_dir=parts_seg_dir / "masks",
                output_viz_dir=parts_seg_dir / "visualizations",
                item_type="part",
                use_subdirs=True,  # Subdirs for each part
            )

            # Save parts segmentation results
            results_path = parts_seg_dir / "segmentation_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(parts_results, f, indent=2)
            print(f"  Saved parts results to: {results_path}")
        else:
            print("  Skipping parts tracking: no parts detected")


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects and parts from video using Qwen-VL + SAM2."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="../Generate_Video/videos/video_01",
        help="Path to video directory (e.g., ../Generate_Video/videos/video_01).",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="PAG JSON file to extract objects and parts. "
        "If not specified, will look in ../Generate_PAG/pags/video_xx/.",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="Single object name to segment (e.g., 'iron').",
    )
    parser.add_argument(
        "--parts",
        type=str,
        default=None,
        help="Comma-separated list of part names (e.g., 'handle,soleplate').",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root output directory. Defaults to Segment_Video/<video_name>/.",
    )

    args = parser.parse_args()

    # Resolve video path
    video_dir = Path(args.video_path).resolve()
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return

    # Find video file
    video_file = find_video_file(video_dir)
    if video_file is None:
        print(f"Error: No video file found in {video_dir}")
        return

    print(f"Video file: {video_file}")

    # Determine video name (e.g., video_01)
    video_name = video_dir.name

    # Determine output root
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        script_dir = Path(__file__).parent.resolve()
        output_root = script_dir / video_name

    print(f"Output directory: {output_root}")

    # Get objects and parts
    if args.object and args.parts:
        # Single object mode
        parts = [p.strip() for p in args.parts.split(",")]
        objects_parts = {args.object: parts}
        print(f"Single object mode: {args.object} with parts {parts}")
    else:
        # PAG file mode
        if args.pag_file:
            pag_path = Path(args.pag_file).resolve()
        else:
            # Try to find PAG file automatically
            pag_dir = Path(__file__).parent.parent / "Generate_PAG" / "pags" / video_name
            pag_candidates = list(pag_dir.glob("output_pag_*.json"))
            if pag_candidates:
                pag_path = pag_candidates[0]
            else:
                print("Error: No PAG file found. Specify --pag_file or --object + --parts")
                return

        if not pag_path.exists():
            print(f"Error: PAG file not found: {pag_path}")
            return

        print(f"PAG file: {pag_path}")
        objects_parts = parse_pag_file(str(pag_path))

        if not objects_parts:
            print("No object parts found in PAG file")
            return

        print(f"Found {len(objects_parts)} objects:")
        for obj, parts in objects_parts.items():
            print(f"  {obj}: {parts}")

    # Load SAM2 video predictor
    predictor = load_sam2_video_predictor()

    # Process video
    process_video(
        video_path=video_file,
        objects_parts=objects_parts,
        output_root=output_root,
        predictor=predictor,
    )

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
