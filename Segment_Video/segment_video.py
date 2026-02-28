"""
Segment objects, parts, and humans from generated video using Qwen-VL + SAM2.

1. Parse PAG file to extract objects (with parts) and humans (with descriptions)
2. Detect objects and humans in frame 0 using open-vocabulary detection (Qwen-VL)
3. Use SAM2 to track/segment objects and humans across frames → masks
4. Similarly obtain part masks for each object part
5. No part segmentation for humans

Directory structure:
    video_xx/
        _frames/frame_xxxx.jpg
        objects/<object_name>/
            object_segmentation/masks/frame_xxxx.png
            object_segmentation/visualizations/frame_xxxx.png
            parts_segmentation/masks/<part_name>/frame_xxxx.png
            parts_segmentation/visualizations/frame_xxxx.png
        humans/<person_name>/
            masks/frame_xxxx.png
            visualizations/frame_xxxx.png
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


def _parse_bbox_response(
    response_text: str,
    names: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Parse Qwen-VL bounding box response into pixel coordinates.

    Args:
        response_text: Raw response text from Qwen-VL.
        names: List of item names to match against.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Dictionary mapping names to bounding boxes [x1, y1, x2, y2] or None.
    """
    # Remove think tags if present
    response_text = re.sub(
        r"<think>.*?</think>", "", response_text, flags=re.DOTALL
    )

    normalized = {name: None for name in names}

    pattern = (
        r"<ref>([^<]+)</ref>\s*<box>\s*"
        r"(\[\s*\[?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]?\s*\]|null)"
        r"\s*</box>"
    )
    matches = re.findall(pattern, response_text, re.IGNORECASE)

    for ref_name, box_str in matches:
        ref_name = ref_name.strip()
        matched_name = None
        for name in names:
            if name.lower() == ref_name.lower():
                matched_name = name
                break

        if matched_name is None:
            continue

        if box_str.strip().lower() == "null":
            normalized[matched_name] = None
        else:
            coord_match = re.search(
                r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", box_str
            )
            if coord_match:
                x1 = int(int(coord_match.group(1)) * image_width / 1000)
                y1 = int(int(coord_match.group(2)) * image_height / 1000)
                x2 = int(int(coord_match.group(3)) * image_width / 1000)
                y2 = int(int(coord_match.group(4)) * image_height / 1000)
                normalized[matched_name] = [x1, y1, x2, y2]

    return normalized


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

    objects_request = "\n".join([f"- {name}" for name in objects])

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
        return _parse_bbox_response(
            response_text, objects, image_width, image_height
        )
    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {name: None for name in objects}


def detect_humans_qwen(
    image: np.ndarray,
    humans: dict[str, str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for humans in the first frame.

    Args:
        image: BGR image as numpy array.
        humans: Dictionary mapping human names to descriptions.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.

    Returns:
        Dictionary mapping human names to bounding boxes or None.
    """
    image_b64 = encode_numpy_image_base64(image)

    items_list = []
    for name, desc in humans.items():
        if desc:
            items_list.append(f"- {name}: {desc}")
        else:
            items_list.append(f"- {name}")
    humans_request = "\n".join(items_list)

    prompt = f"""You are analyzing an image. Detect bounding boxes for these people:
{humans_request}

For each person, output in this format (use the person name only, not the description):
<ref>person_name</ref><box>[[x1,y1,x2,y2]]</box>

The bounding box should tightly enclose the entire person (head to feet).
Coordinates should be on a 0-1000 normalized scale (not pixels).
If a person is not visible, output: <ref>person_name</ref><box>null</box>

Detect all people listed above."""

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
        return _parse_bbox_response(
            response_text, list(humans.keys()), image_width, image_height
        )
    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {name: None for name in humans}


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
        return _parse_bbox_response(
            response_text, parts, image_width, image_height
        )
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

    Frames are saved as <nnnn>.jpg (e.g. 0000.jpg, 0001.jpg) because SAM2
    requires JPEG images with purely numeric filenames.

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

        # SAM2 expects JPEG files with numeric-only names
        frame_path = output_dir / f"{frame_idx:04d}.jpg"
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

    # Draw legend in top-left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 24
    x_start = 10
    y_start = 24

    visible_items = [
        (name, colors[i % len(colors)])
        for i, (name, mask) in enumerate(masks.items())
        if mask is not None and np.any(mask > 0)
    ]

    if visible_items:
        # Draw semi-transparent background for legend
        max_text_w = max(
            cv2.getTextSize(name, font, font_scale, thickness)[0][0]
            for name, _ in visible_items
        )
        legend_w = 20 + max_text_w + 10  # color square + text + padding
        # legend_h = len(visible_items) * line_height + 10
        overlay_bg = result.copy()
        cv2.rectangle(
            overlay_bg,
            (x_start - 5, y_start - line_height + 2),
            (x_start + legend_w, y_start + (len(visible_items) - 1) * line_height + 10),
            (0, 0, 0),
            -1,
        )
        result = cv2.addWeighted(overlay_bg, 0.5, result, 0.5, 0)

        for idx, (name, color) in enumerate(visible_items):
            y = y_start + idx * line_height
            # Color square
            cv2.rectangle(result, (x_start, y - 10), (x_start + 14, y + 4), color, -1)
            # Label text
            cv2.putText(result, name, (x_start + 20, y), font, font_scale, (255, 255, 255), thickness)

    return result


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological smoothing to a binary mask.

    Performs closing (fill holes) → opening (remove noise) → optional
    Gaussian blur + re-threshold for smoother edges.

    Args:
        mask: Binary mask (uint8, 0 or 1).
        kernel_size: Size of the morphological kernel.

    Returns:
        Smoothed binary mask (uint8, 0 or 1).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Close: fill small holes inside the mask
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open: remove small noise outside the mask
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    # Gaussian blur + re-threshold for smoother edges
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (kernel_size, kernel_size), 0)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed


def resolve_overlapping_masks(
    logits: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Resolve overlapping masks using argmax over logit scores.

    For pixels claimed by multiple items, assigns to the one with the
    highest SAM2 logit score. Non-overlapping pixels are kept as-is.

    Args:
        logits: Dictionary mapping names to raw logit arrays (float, H x W).

    Returns:
        Dictionary mapping names to non-overlapping binary masks (uint8, 0 or 1).
    """
    if not logits:
        return {}

    names = list(logits.keys())
    # Stack logits: (N, H, W)
    stacked = np.stack([logits[n] for n in names], axis=0)

    # Build a mask of which pixels are positive for each item
    positive = stacked > 0.0  # (N, H, W)
    any_positive = positive.any(axis=0)  # (H, W)

    # For each pixel, find which item has the highest logit
    best_idx = np.argmax(stacked, axis=0)  # (H, W)

    result = {}
    for i, name in enumerate(names):
        # A pixel belongs to this item if:
        # - it had a positive logit AND
        # - this item has the highest logit at that pixel
        mask = ((best_idx == i) & any_positive).astype(np.uint8)
        result[name] = mask

    return result


def track_with_sam2(
    predictor,
    frames_dir: Path,
    bboxes: dict[str, list[int] | None],
    output_masks_dir: Path,
    output_viz_dir: Path,
    item_type: str = "object",
    use_subdirs: bool = True,
    resolve_overlaps: bool = False,
) -> dict:
    """Track objects/parts/humans across video frames using SAM2.

    Args:
        predictor: SAM2 video predictor.
        frames_dir: Directory containing extracted JPEG frames (<nnnn>.jpg).
        bboxes: Dictionary mapping names to bounding boxes (detected in frame 0).
        output_masks_dir: Directory to save masks.
        output_viz_dir: Directory to save visualizations.
        item_type: "object", "part", or "human" for logging.
        use_subdirs: If True, create subdirectories per item inside masks dir.
        resolve_overlaps: If True, resolve overlapping masks using logit argmax
            and apply morphological smoothing. Recommended for part segmentation.

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
    frame_files = sorted(frames_dir.glob("*.jpg"))

    for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
        # video_res_masks shape: (num_objects, 1, H, W)

        if resolve_overlaps and len(obj_id_to_name) > 1:
            # Collect raw logits for overlap resolution
            logits_dict = {}
            for i, obj_id in enumerate(obj_ids):
                if obj_id not in obj_id_to_name:
                    continue
                name = obj_id_to_name[obj_id]
                logits_dict[name] = video_res_masks[i, 0].cpu().numpy()

            # Resolve overlaps via argmax, then smooth
            masks_dict = resolve_overlapping_masks(logits_dict)
            for name in masks_dict:
                masks_dict[name] = smooth_mask(masks_dict[name])
        else:
            # Standard: threshold and smooth each mask independently
            masks_dict = {}
            for i, obj_id in enumerate(obj_ids):
                if obj_id not in obj_id_to_name:
                    continue
                name = obj_id_to_name[obj_id]
                mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                mask = smooth_mask(mask)
                masks_dict[name] = mask

        # Save masks
        for name, mask in masks_dict.items():
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


def parse_pag_file(pag_path: str) -> tuple[dict[str, dict], list[dict]]:
    """Parse PAG file to extract objects (with parts and descriptions) and humans.

    Args:
        pag_path: Path to PAG JSON file.

    Returns:
        Tuple of:
        - objects_info: dict mapping object name to
            {"description": str, "parts": list[str]}
        - humans_info: list of {"name": str, "description": str}
    """
    with open(pag_path) as f:
        pag = json.load(f)

    # Build object descriptions from "object states"
    obj_descriptions = {}
    for obj_state in pag.get("object states", []):
        name = obj_state.get("name", "")
        desc = obj_state.get("description", "")
        if name:
            obj_descriptions[name] = desc

    # Build objects_info from "object part nodes"
    objects_info: dict[str, dict] = {}
    for node in pag.get("object part nodes", []):
        parts = node.split(", ", 1)
        if len(parts) == 2:
            obj_name, part_name = parts
            if obj_name not in objects_info:
                objects_info[obj_name] = {
                    "description": obj_descriptions.get(obj_name, ""),
                    "parts": [],
                }
            objects_info[obj_name]["parts"].append(part_name)

    # Also add objects that have states but no parts listed
    for name, desc in obj_descriptions.items():
        if name not in objects_info:
            objects_info[name] = {"description": desc, "parts": []}

    # Extract humans from "human states"
    humans_info = []
    for human_state in pag.get("human states", []):
        humans_info.append({
            "name": human_state.get("name", ""),
            "description": human_state.get("description", ""),
        })

    return objects_info, humans_info


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
    objects_info: dict[str, dict],
    humans_info: list[dict],
    output_root: Path,
    predictor,
) -> None:
    """Process a video to segment objects, parts, and humans.

    Args:
        video_path: Path to video file.
        objects_info: Dict mapping object names to
            {"description": str, "parts": list[str]}.
        humans_info: List of {"name": str, "description": str}.
        output_root: Root output directory (e.g., Segment_Video/video_01).
        predictor: SAM2 video predictor.
    """
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"{'='*60}")

    # Extract frames (SAM2 requires <nnnn>.jpg naming during tracking)
    frames_dir = output_root / "_frames"
    existing_jpgs = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    if not existing_jpgs:
        frame_paths, width, height = extract_video_frames(str(video_path), frames_dir)
    else:
        # If frames were already renamed to frame_xxxx.jpg from a previous run,
        # temporarily rename back to numeric <nnnn>.jpg for SAM2 compatibility.
        needs_numeric_rename = any(p.stem.startswith("frame_") for p in existing_jpgs)
        if needs_numeric_rename:
            print("Renaming existing frame_xxxx.jpg back to <nnnn>.jpg for SAM2...")
            for jpg_path in existing_jpgs:
                stem = jpg_path.stem
                if stem.startswith("frame_"):
                    try:
                        idx = int(stem.replace("frame_", ""))
                        new_path = frames_dir / f"{idx:04d}.jpg"
                        jpg_path.rename(new_path)
                    except ValueError:
                        pass

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        first_frame = cv2.imread(str(frame_paths[0]))
        height, width = first_frame.shape[:2]
        print(f"Using existing {len(frame_paths)} frames from {frames_dir}")

    # Load first frame for detection
    first_frame = cv2.imread(str(frames_dir / "0000.jpg"))
    if first_frame is None:
        print("Error: Could not load first frame")
        return

    # ─── Human segmentation ───────────────────────────────────────
    if humans_info:
        print(f"\nHumans to segment: {[h['name'] for h in humans_info]}")

        # Build name → description dict for detection
        humans_dict = {h["name"]: h["description"] for h in humans_info}

        print("\n[Human Step 1] Detecting humans in frame 0 with Qwen-VL...")
        human_bboxes = detect_humans_qwen(first_frame, humans_dict, width, height)

        for name, bbox in human_bboxes.items():
            if bbox:
                print(f"  {name}: {bbox}")
            else:
                print(f"  {name}: NOT DETECTED")

        # Process each human individually
        for human_name, human_bbox in human_bboxes.items():
            human_dir_name = human_name.replace(" ", "_")
            human_output_dir = output_root / "humans" / human_dir_name

            print(f"\n{'─'*40}")
            print(f"Processing human: {human_name}")
            print(f"{'─'*40}")

            if human_bbox is None:
                print(f"  Skipping {human_name}: not detected in frame 0")
                continue

            print(f"\n[Human Step 2] Tracking {human_name} across frames with SAM2...")
            track_with_sam2(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes={human_name: human_bbox},
                output_masks_dir=human_output_dir / "masks",
                output_viz_dir=human_output_dir / "visualizations",
                item_type="human",
                use_subdirs=False,
            )
    else:
        print("\nNo humans specified in PAG, skipping human segmentation")

    # ─── Object segmentation ──────────────────────────────────────
    if not objects_info:
        print("\nNo objects specified, skipping object segmentation")
        return

    # Build list of object names for detection
    objects = list(objects_info.keys())
    print(f"\nObjects to segment: {objects}")

    # Detect all objects in frame 0
    print("\n[Object Step 1] Detecting objects in frame 0 with Qwen-VL...")
    object_bboxes = detect_objects_qwen(first_frame, objects, width, height)

    for obj, bbox in object_bboxes.items():
        if bbox:
            print(f"  {obj}: {bbox}")
        else:
            print(f"  {obj}: NOT DETECTED")

    # Process each object
    for obj_name, obj_info in objects_info.items():
        parts = obj_info["parts"]
        obj_dir_name = obj_name.replace(" ", "_")
        obj_output_dir = output_root / "objects" / obj_dir_name

        print(f"\n{'='*40}")
        print(f"Processing object: {obj_name}")
        print(f"{'='*40}")

        # Track object across frames
        print("\n[Object Step 2] Tracking object across frames with SAM2...")
        obj_seg_dir = obj_output_dir / "object_segmentation"

        if object_bboxes.get(obj_name) is not None:
            track_with_sam2(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes={obj_name: object_bboxes[obj_name]},
                output_masks_dir=obj_seg_dir / "masks",
                output_viz_dir=obj_seg_dir / "visualizations",
                item_type="object",
                use_subdirs=False,
            )
        else:
            print(f"  Skipping object tracking: {obj_name} not detected")

        # Detect and track parts
        if not parts:
            print(f"  No parts defined for {obj_name}, skipping part segmentation")
            continue

        print(f"\n[Object Step 3] Detecting parts {parts} in frame 0...")
        part_bboxes = detect_parts_qwen(
            first_frame, parts, obj_name, width, height
        )

        for part, bbox in part_bboxes.items():
            if bbox:
                print(f"  {part}: {bbox}")
            else:
                print(f"  {part}: NOT DETECTED")

        # Track parts across frames
        print("\n[Object Step 4] Tracking parts across frames with SAM2...")
        parts_seg_dir = obj_output_dir / "parts_segmentation"

        # Filter out undetected parts
        detected_parts = {p: b for p, b in part_bboxes.items() if b is not None}

        if detected_parts:
            track_with_sam2(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes=detected_parts,
                output_masks_dir=parts_seg_dir / "masks",
                output_viz_dir=parts_seg_dir / "visualizations",
                item_type="part",
                use_subdirs=True,
                resolve_overlaps=True,
            )
        else:
            print("  Skipping parts tracking: no parts detected")

    # ─── Rename _frames to uniform frame_xxxx.jpg naming ─────────
    print("\nRenaming frames to frame_xxxx.jpg naming...")
    for jpg_path in sorted(frames_dir.glob("*.jpg")):
        stem = jpg_path.stem
        # Skip if already renamed
        if stem.startswith("frame_"):
            continue
        try:
            idx = int(stem)
            new_path = frames_dir / f"frame_{idx:04d}.jpg"
            jpg_path.rename(new_path)
        except ValueError:
            pass
    print("  Frames renamed.")


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects, parts, and humans from video using Qwen-VL + SAM2."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="../Generate_Video/output/video_01",
        help="Path to video directory (e.g., ../Generate_Video/output/video_01).",
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help="PAG JSON file to extract objects, parts, and humans. "
        "If not specified, will look in ../Generate_PAG/output/video_xx/.",
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
        output_root = script_dir / "output" / video_name

    print(f"Output directory: {output_root}")

    # Get objects, parts, and humans
    if args.object and args.parts:
        # Single object mode (no human segmentation)
        parts = [p.strip() for p in args.parts.split(",")]
        objects_info = {args.object: {"description": "", "parts": parts}}
        humans_info = []
        print(f"Single object mode: {args.object} with parts {parts}")
    else:
        # PAG file mode
        if args.pag_file:
            pag_path = Path(args.pag_file).resolve()
        else:
            # Try to find PAG file automatically
            pag_dir = Path(__file__).parent.parent / "Generate_PAG" / "output" / video_name
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
        objects_info, humans_info = parse_pag_file(str(pag_path))

        if not objects_info and not humans_info:
            print("No objects or humans found in PAG file")
            return

        if objects_info:
            print(f"Found {len(objects_info)} objects:")
            for obj, info in objects_info.items():
                desc = f" ({info['description']})" if info["description"] else ""
                print(f"  {obj}{desc}: parts={info['parts']}")

        if humans_info:
            print(f"Found {len(humans_info)} humans:")
            for h in humans_info:
                desc = f" ({h['description']})" if h["description"] else ""
                print(f"  {h['name']}{desc}")

    # Load SAM2 video predictor
    predictor = load_sam2_video_predictor()

    # Process video
    process_video(
        video_path=video_file,
        objects_info=objects_info,
        humans_info=humans_info,
        output_root=output_root,
        predictor=predictor,
    )

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
