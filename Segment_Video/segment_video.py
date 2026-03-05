"""
Segment objects, parts, and humans from generated video using Qwen-VL + SAM3.

1. Parse PAG file to extract objects (with parts) and humans (with descriptions)
2. Detect objects and humans in frame 0 using open-vocabulary detection (Qwen-VL)
3. Use SAM3 to track/segment objects and humans across frames -> masks
4. Similarly obtain part masks for each object part
5. No part segmentation for humans
"""

import argparse
import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI


# Ollama configuration
OLLAMA_HOST = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"
QWEN_MODEL = "qwen3-vl:32b"

# SAM3 configuration
SAM3_CHECKPOINT = None
SAM3_BPE_PATH = "/my_workspace/4DHHOI/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

VIZ_COLORS = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 255, 0),  # Light green
    (255, 128, 0),  # Light blue
]


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
    """Parse Qwen-VL bounding box response into pixel coordinates."""
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


def _detect_bboxes_qwen(
    client: OpenAI,
    model: str,
    image_b64: str,
    prompt: str,
    names: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        response_text = _extract_text_content(response.choices[0].message.content)
        return _parse_bbox_response(response_text, names, image_width, image_height)
    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return {name: None for name in names}


def detect_objects_qwen(
    client: OpenAI,
    model: str,
    image: np.ndarray,
    objects: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for objects in the first frame."""
    image_b64 = encode_numpy_image_base64(image)
    objects_request = "\n".join([f"- {name}" for name in objects])

    prompt = f"""You are analyzing an image. Detect bounding boxes for these objects:
{objects_request}

For each object, output in this format:
<ref>object_name</ref><box>[[x1,y1,x2,y2]]</box>

Coordinates should be on a 0-1000 normalized scale (not pixels).
If an object is not visible, output: <ref>object_name</ref><box>null</box>

Detect all objects listed above."""
    return _detect_bboxes_qwen(
        client=client,
        model=model,
        image_b64=image_b64,
        prompt=prompt,
        names=objects,
        image_width=image_width,
        image_height=image_height,
    )


def detect_humans_qwen(
    client: OpenAI,
    model: str,
    image: np.ndarray,
    humans: dict[str, str],
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for humans in the first frame."""
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
    return _detect_bboxes_qwen(
        client=client,
        model=model,
        image_b64=image_b64,
        prompt=prompt,
        names=list(humans.keys()),
        image_width=image_width,
        image_height=image_height,
    )


def detect_parts_qwen(
    client: OpenAI,
    model: str,
    image: np.ndarray,
    parts: list[str],
    object_name: str,
    image_width: int,
    image_height: int,
) -> dict[str, list[int] | None]:
    """Use Qwen-VL to detect bounding boxes for object parts."""
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
    return _detect_bboxes_qwen(
        client=client,
        model=model,
        image_b64=image_b64,
        prompt=prompt,
        names=parts,
        image_width=image_width,
        image_height=image_height,
    )


def load_sam3_tracker():
    """Load SAM3 video model and return its tracker module."""
    import torch
    from sam3.model_builder import build_sam3_video_model

    print("Loading SAM3 video tracker...")
    model = build_sam3_video_model(
        checkpoint_path=SAM3_CHECKPOINT,
        bpe_path=SAM3_BPE_PATH,
    )
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    tracker.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM3 video tracker loaded on {device}")
    return tracker


def extract_video_frames(video_path: str, output_dir: Path) -> tuple[list[Path], int, int]:
    """Extract frames from video file to numeric JPEG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_frame in output_dir.glob("*.jpg"):
        existing_frame.unlink()

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_paths = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
    color_map: dict[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """Create visualization with masks overlaid on frame."""
    result = frame.copy()

    def get_color(name: str, index: int) -> tuple[int, int, int]:
        if color_map is not None and name in color_map:
            return color_map[name]
        return VIZ_COLORS[index % len(VIZ_COLORS)]

    for i, (name, mask) in enumerate(masks.items()):
        if mask is None:
            continue

        color = get_color(name, i)
        overlay = result.copy()
        overlay[mask > 0] = color
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text_line_type = cv2.LINE_AA
    line_height = 24
    x_start = 10
    y_start = 24

    visible_items = [
        (name, get_color(name, i))
        for i, (name, mask) in enumerate(masks.items())
        if mask is not None and np.any(mask > 0)
    ]

    if visible_items:
        max_text_w = max(
            cv2.getTextSize(name, font, font_scale, thickness)[0][0]
            for name, _ in visible_items
        )
        legend_w = 20 + max_text_w + 10
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
            cv2.rectangle(result, (x_start, y - 10), (x_start + 14, y + 4), color, -1)
            cv2.putText(
                result,
                name,
                (x_start + 20, y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                text_line_type,
            )

    return result


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply morphological smoothing to a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
    blurred = cv2.GaussianBlur(smoothed.astype(np.float32), (kernel_size, kernel_size), 0)
    smoothed = (blurred > 0.5).astype(np.uint8)
    return smoothed


def resolve_overlapping_masks(
    masks: dict[str, np.ndarray],
    confidences: dict[str, float],
) -> dict[str, np.ndarray]:
    """Resolve overlaps by assigning each pixel to the highest-confidence mask."""
    if not masks:
        return {}

    names = sorted(masks.keys(), key=lambda n: confidences.get(n, 0.0), reverse=True)
    stacked = np.stack([(masks[n] > 0) for n in names], axis=0)
    any_positive = stacked.any(axis=0)

    first_idx = np.argmax(stacked.astype(np.uint8), axis=0)
    result = {}
    for i, name in enumerate(names):
        result[name] = ((first_idx == i) & any_positive).astype(np.uint8)
    return result


def maybe_postprocess_masks(
    masks: dict[str, np.ndarray],
    apply_mask_postprocess: bool,
) -> dict[str, np.ndarray]:
    """Optionally apply morphology-based postprocessing to each mask."""
    if not apply_mask_postprocess:
        return masks
    return {name: smooth_mask(mask) for name, mask in masks.items()}


def track_with_sam3(
    predictor,
    frames_dir: Path,
    bboxes: dict[str, list[int] | None],
    output_masks_dir: Path,
    output_viz_dir: Path,
    item_type: str = "object",
    use_subdirs: bool = True,
    resolve_overlaps: bool = False,
    apply_mask_postprocess: bool = False,
) -> dict:
    """Track objects/parts/humans across video frames using SAM3 tracker."""
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    output_viz_dir.mkdir(parents=True, exist_ok=True)

    inference_state = predictor.init_state(video_path=str(frames_dir))
    num_frames = inference_state["num_frames"]
    video_height = inference_state["video_height"]
    video_width = inference_state["video_width"]

    print(f"  Video: {num_frames} frames, {video_width}x{video_height}")

    obj_id_to_name = {}
    for obj_id, (name, bbox) in enumerate(bboxes.items()):
        if bbox is None:
            print(f"    Skipping {name}: not detected in frame 0")
            continue

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, video_width - 1))
        y1 = max(0, min(y1, video_height - 1))
        x2 = max(x1 + 1, min(x2, video_width))
        y2 = max(y1 + 1, min(y2, video_height))
        bbox = [x1, y1, x2, y2]

        print(f"    Adding {item_type} '{name}' with bbox {bbox}")
        obj_id_to_name[obj_id] = name

        norm_box = np.array(
            [
                x1 / video_width,
                y1 / video_height,
                x2 / video_width,
                y2 / video_height,
            ],
            dtype=np.float32,
        )
        _, _, _, _ = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=obj_id,
            box=norm_box,
            rel_coordinates=True,
        )

    if not obj_id_to_name:
        print(f"  No {item_type}s detected, skipping tracking")
        return {"tracked": False, "items": {}}

    color_map = {
        name: VIZ_COLORS[i % len(VIZ_COLORS)]
        for i, name in enumerate(sorted(obj_id_to_name.values()))
    }

    print(f"  Propagating masks through {num_frames} frames...")
    results = {"tracked": True, "items": {}, "num_frames": num_frames}

    frame_files = sorted(frames_dir.glob("*.jpg"))

    for frame_idx, obj_ids, _, video_res_masks, obj_scores in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=0,
        max_frame_num_to_track=num_frames,
        reverse=False,
        propagate_preflight=True,
    ):
        masks_dict: dict[str, np.ndarray] = {}
        conf_dict: dict[str, float] = {}
        for i, obj_id in enumerate(obj_ids):
            if obj_id not in obj_id_to_name:
                continue
            name = obj_id_to_name[obj_id]
            mask = (video_res_masks[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
            masks_dict[name] = mask
            conf_dict[name] = float(obj_scores[i].item())

        if resolve_overlaps and len(masks_dict) > 1:
            masks_dict = resolve_overlapping_masks(masks_dict, conf_dict)

        masks_dict = maybe_postprocess_masks(
            masks=masks_dict,
            apply_mask_postprocess=apply_mask_postprocess,
        )

        for name, mask in masks_dict.items():
            if use_subdirs:
                item_mask_dir = output_masks_dir / name.replace(" ", "_")
                item_mask_dir.mkdir(exist_ok=True)
            else:
                item_mask_dir = output_masks_dir

            mask_filename = f"frame_{frame_idx:04d}.png"
            save_mask(mask, str(item_mask_dir / mask_filename))

            if name not in results["items"]:
                results["items"][name] = {"masks": [], "bbox_frame0": bboxes[name]}
            results["items"][name]["masks"].append(mask_filename)

        if frame_idx < len(frame_files):
            frame = cv2.imread(str(frame_files[frame_idx]))
            viz = create_visualization(frame, masks_dict, color_map=color_map)
            viz_filename = f"frame_{frame_idx:04d}.png"
            cv2.imwrite(str(output_viz_dir / viz_filename), viz)

    predictor.clear_all_points_in_video(inference_state)
    return results


def parse_pag_file(pag_path: str) -> tuple[dict[str, dict], list[dict]]:
    """Parse PAG file to extract objects (with parts and descriptions) and humans."""
    with open(pag_path) as f:
        pag = json.load(f)

    obj_descriptions = {}
    for obj_state in pag.get("object states", []):
        name = obj_state.get("name", "")
        desc = obj_state.get("description", "")
        if name:
            obj_descriptions[name] = desc

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

    for name, desc in obj_descriptions.items():
        if name not in objects_info:
            objects_info[name] = {"description": desc, "parts": []}

    humans_info = []
    for human_state in pag.get("human states", []):
        humans_info.append({
            "name": human_state.get("name", ""),
            "description": human_state.get("description", ""),
        })

    return objects_info, humans_info


def find_video_file(video_dir: Path) -> Path:
    """Find the first video file in directory."""
    video_extensions = [".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV"]
    for ext in video_extensions:
        videos = sorted(video_dir.glob(f"*{ext}"))
        if videos:
            return videos[0]
    raise FileNotFoundError(f"No video file found in {video_dir}")


def rename_numeric_frames_to_prefixed(frames_dir: Path) -> None:
    """Rename numeric tracker frames to frame_xxxx.jpg for final output."""
    for jpg_path in sorted(frames_dir.glob("*.jpg")):
        stem = jpg_path.stem
        if stem.startswith("frame_"):
            continue
        try:
            idx = int(stem)
            new_path = frames_dir / f"frame_{idx:04d}.jpg"
            jpg_path.rename(new_path)
        except ValueError:
            pass


def process_video(
    video_path: Path,
    objects_info: dict[str, dict],
    humans_info: list[dict],
    output_root: Path,
    predictor,
    qwen_client: OpenAI,
    qwen_model: str,
    mask_postprocess: bool = False,
) -> None:
    """Process a video to segment objects, parts, and humans."""
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path.name}")
    print(f"{'='*60}")

    frames_dir = output_root / "_frames"
    frame_paths, width, height = extract_video_frames(str(video_path), frames_dir)
    first_frame = cv2.imread(str(frame_paths[0]))

    if humans_info:
        print(f"\nHumans to segment: {[h['name'] for h in humans_info]}")
        humans_dict = {h["name"]: h["description"] for h in humans_info}

        print("\n[Human Step 1] Detecting humans in frame 0 with Qwen-VL...")
        human_bboxes = detect_humans_qwen(
            qwen_client, qwen_model, first_frame, humans_dict, width, height
        )

        for name, bbox in human_bboxes.items():
            if bbox:
                print(f"  {name}: {bbox}")
            else:
                print(f"  {name}: NOT DETECTED")

        for human_name, human_bbox in human_bboxes.items():
            human_dir_name = human_name.replace(" ", "_")
            human_output_dir = output_root / "humans" / human_dir_name

            print(f"\n{'─'*40}")
            print(f"Processing human: {human_name}")
            print(f"{'─'*40}")

            if human_bbox is None:
                print(f"  Skipping {human_name}: not detected in frame 0")
                continue

            print(f"\n[Human Step 2] Tracking {human_name} across frames with SAM3...")
            track_with_sam3(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes={human_name: human_bbox},
                output_masks_dir=human_output_dir / "masks",
                output_viz_dir=human_output_dir / "visualizations",
                item_type="human",
                use_subdirs=False,
                apply_mask_postprocess=mask_postprocess,
            )
    else:
        print("\nNo humans specified in PAG, skipping human segmentation")

    if not objects_info:
        print("\nNo objects specified, skipping object segmentation")
        return

    objects = list(objects_info.keys())
    print(f"\nObjects to segment: {objects}")

    print("\n[Object Step 1] Detecting objects in frame 0 with Qwen-VL...")
    object_bboxes = detect_objects_qwen(
        qwen_client, qwen_model, first_frame, objects, width, height
    )

    for obj, bbox in object_bboxes.items():
        if bbox:
            print(f"  {obj}: {bbox}")
        else:
            print(f"  {obj}: NOT DETECTED")

    for obj_name, obj_info in objects_info.items():
        parts = obj_info["parts"]
        obj_dir_name = obj_name.replace(" ", "_")
        obj_output_dir = output_root / "objects" / obj_dir_name

        print(f"\n{'='*40}")
        print(f"Processing object: {obj_name}")
        print(f"{'='*40}")

        print("\n[Object Step 2] Tracking object across frames with SAM3...")
        obj_seg_dir = obj_output_dir / "object_segmentation"

        if object_bboxes.get(obj_name) is not None:
            track_with_sam3(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes={obj_name: object_bboxes[obj_name]},
                output_masks_dir=obj_seg_dir / "masks",
                output_viz_dir=obj_seg_dir / "visualizations",
                item_type="object",
                use_subdirs=False,
                apply_mask_postprocess=mask_postprocess,
            )
        else:
            print(f"  Skipping object tracking: {obj_name} not detected")

        if not parts:
            print(f"  No parts defined for {obj_name}, skipping part segmentation")
            continue

        print(f"\n[Object Step 3] Detecting parts {parts} in frame 0...")
        part_bboxes = detect_parts_qwen(
            qwen_client, qwen_model, first_frame, parts, obj_name, width, height
        )

        for part, bbox in part_bboxes.items():
            if bbox:
                print(f"  {part}: {bbox}")
            else:
                print(f"  {part}: NOT DETECTED")

        print("\n[Object Step 4] Tracking parts across frames with SAM3...")
        parts_seg_dir = obj_output_dir / "parts_segmentation"
        detected_parts = {p: b for p, b in part_bboxes.items() if b is not None}

        if detected_parts:
            track_with_sam3(
                predictor=predictor,
                frames_dir=frames_dir,
                bboxes=detected_parts,
                output_masks_dir=parts_seg_dir / "masks",
                output_viz_dir=parts_seg_dir / "visualizations",
                item_type="part",
                use_subdirs=True,
                resolve_overlaps=True,
                apply_mask_postprocess=mask_postprocess,
            )
        else:
            print("  Skipping parts tracking: no parts detected")

    print("\nRenaming frames to frame_xxxx.jpg naming...")
    rename_numeric_frames_to_prefixed(frames_dir)
    print("  Frames renamed.")


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects, parts, and humans from video using Qwen-VL + SAM3."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="../Generate_Video/output/video_03",
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
        "--output_root",
        type=str,
        default=None,
        help="Root output directory. Defaults to Segment_Video/<video_name>/.",
    )
    parser.add_argument(
        "--mask_postprocess",
        action="store_true",
        help="Enable mask morphology post-processing (close -> open -> blur-threshold).",
    )
    parser.add_argument(
        "--ollama_host",
        type=str,
        default=OLLAMA_HOST,
        help="OpenAI-compatible Ollama host (e.g., http://127.0.0.1:11434/v1).",
    )
    parser.add_argument(
        "--ollama_api_key",
        type=str,
        default=OLLAMA_API_KEY,
        help="API key used by OpenAI client for Ollama (default: ollama).",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=QWEN_MODEL,
        help="Qwen-VL model name served by Ollama.",
    )

    args = parser.parse_args()

    video_dir = Path(args.video_path).resolve()
    video_file = find_video_file(video_dir)

    print(f"Video file: {video_file}")
    video_name = video_dir.name

    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        script_dir = Path(__file__).parent.resolve()
        output_root = script_dir / "output" / video_name

    print(f"Output directory: {output_root}")
    print(f"Ollama host: {args.ollama_host}")
    print(f"Qwen model: {args.qwen_model}")

    if args.pag_file:
        pag_path = Path(args.pag_file).resolve()
    else:
        pag_dir = Path(__file__).parent.parent / "Generate_PAG" / "output" / video_name
        pag_path = next(pag_dir.glob("output_pag_*.json"))

    print(f"PAG file: {pag_path}")
    objects_info, humans_info = parse_pag_file(str(pag_path))

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

    predictor = load_sam3_tracker()
    qwen_client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key)

    process_video(
        video_path=video_file,
        objects_info=objects_info,
        humans_info=humans_info,
        output_root=output_root,
        predictor=predictor,
        qwen_client=qwen_client,
        qwen_model=args.qwen_model,
        mask_postprocess=args.mask_postprocess,
    )

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
