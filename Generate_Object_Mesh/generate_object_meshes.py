"""Generate 3D object meshes using SAM2 and SAM 3D Objects.

Pipeline:
    1. Parse PAG file to get object names
    2. Use Qwen-VL to detect each object's bounding box
    3. Use SAM2 to segment the object
    4. Use SAM 3D Objects to generate 3D mesh
    5. Save outputs to objects/<object_name>/

Usage:
    python generate_object_meshes.py
    python generate_object_meshes.py --pag_file pag.json --image img.png
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
sys.path.insert(0, "/my_workspace/4DHHOI/sam2")
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

# Configuration
SAM2_CHECKPOINT = "/my_workspace/4DHHOI/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM3D_PATH = "/my_workspace/4DHHOI/sam-3d-objects"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL = "qwen3-vl:32b"


def parse_pag_objects(pag_path: str) -> list[str]:
    """Extract unique object names from PAG file."""
    with open(pag_path) as f:
        pag = json.load(f)

    objects = set()
    for node in pag.get("object part nodes", []):
        obj_name = node.split(", ", 1)[0]
        objects.add(obj_name)

    return sorted(objects)


def detect_bbox(
    image_path: str,
    object_name: str,
    image_width: int,
    image_height: int,
) -> list[int] | None:
    """Use Qwen-VL to detect object bounding box."""
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Handle common ambiguities
    object_description = object_name
    if object_name.lower() == "iron":
        object_description = (
            "clothes iron (the household appliance being used to "
            "press clothes on the ironing board)"
        )
    elif object_name.lower() == "ironing board":
        object_description = (
            "ironing board (the large foldable board/table with "
            "metal legs that the person is using)"
        )

    # Use Qwen's detection format with normalized coordinates (0-1000)
    prompt = f"""Detect the {object_description} in this image.

Output the bounding box in the format: \
<ref>{object_name}</ref><box>[[x1,y1,x2,y2]]</box>
where coordinates are normalized to 0-1000 scale.

If not visible, output: <ref>{object_name}</ref><box>null</box>"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": QWEN_MODEL,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=180,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        print(f"    Qwen response: {text[:500]}")

        # Try to parse Qwen's box format: <box>[[x1,y1,x2,y2]]</box>
        box_pattern = (
            r"<box>\s*\[\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,"
            r"\s*(\d+)\s*,\s*(\d+)\s*\]?\s*\]?\s*</box>"
        )
        box_match = re.search(box_pattern, text)
        if box_match:
            x1 = int(int(box_match.group(1)) * image_width / 1000)
            y1 = int(int(box_match.group(2)) * image_height / 1000)
            x2 = int(int(box_match.group(3)) * image_width / 1000)
            y2 = int(int(box_match.group(4)) * image_height / 1000)
            return [x1, y1, x2, y2]

        # Fallback: try JSON format
        json_match = re.search(r'\{[^{}]*"bbox"[^{}]*\}', text, re.DOTALL)
        if json_match:
            bbox = json.loads(json_match.group()).get("bbox")
            if bbox:
                if all(0 <= c <= 1000 for c in bbox):
                    return [
                        int(bbox[0] * image_width / 1000),
                        int(bbox[1] * image_height / 1000),
                        int(bbox[2] * image_width / 1000),
                        int(bbox[3] * image_height / 1000),
                    ]
                return bbox

    except Exception as e:
        print(f"    Detection error: {e}")

    return None


def segment_object(
    predictor: SAM2ImagePredictor,
    image: np.ndarray,
    bbox: list[int],
) -> np.ndarray:
    """Segment object using SAM2 with bounding box prompt."""
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array(bbox)[None, :],
        multimask_output=False,
    )
    return masks[0].astype(np.uint8)


def load_sam3d():
    """Load SAM 3D Objects inference pipeline."""
    notebook_path = Path(SAM3D_PATH) / "notebook"
    if str(notebook_path) not in sys.path:
        sys.path.insert(0, str(notebook_path))

    from inference import Inference  # noqa: E402

    # Try different checkpoint locations
    possible_paths = [
        Path(SAM3D_PATH) / "checkpoints" / "hf" / "pipeline.yaml",
        (Path(SAM3D_PATH) / "checkpoints" / "hf-download"
         / "checkpoints" / "pipeline.yaml"),
    ]

    config_path = None
    for p in possible_paths:
        if p.exists():
            config_path = p
            break

    if config_path is None:
        raise FileNotFoundError(
            f"SAM3D config not found. Tried: {[str(p) for p in possible_paths]}"
        )

    return Inference(str(config_path), compile=False)


def generate_mesh(inference, image: np.ndarray, mask: np.ndarray) -> dict:
    """Generate 3D mesh using SAM 3D Objects."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if mask.dtype != bool:
        mask = mask > 0

    return inference(image, mask, seed=42)


def save_outputs(output: dict, output_dir: Path, object_name: str) -> None:
    """Save mesh outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "gs" in output:
        output["gs"].save_ply(str(output_dir / f"{object_name}.ply"))
        print(f"    Saved: {object_name}.ply")

    if "glb" in output and output["glb"] is not None:
        output["glb"].export(str(output_dir / "mesh.glb"))
        print("    Saved: mesh.glb")


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


def save_mask_image(mask: np.ndarray, output_path: Path) -> None:
    """Save binary mask as PNG."""
    cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))


def save_masked_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save image with mask applied (for visual verification)."""
    result = image_rgb.copy()
    result[mask == 0] = 0
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Generate 3D meshes from PAG objects.")
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
    args = parser.parse_args()

    # Resolve paths
    image_path = Path(args.image).resolve()
    pag_path = Path(args.pag_file).resolve()
    output_root = Path(__file__).parent / "objects"

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    if not pag_path.exists():
        print(f"Error: PAG file not found: {pag_path}")
        return

    # Get objects from PAG
    objects = parse_pag_objects(str(pag_path))
    print(f"Objects to process: {objects}")

    # Load models
    print("\nLoading SAM2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2)

    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()

    # Load image once
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    # Process each object
    for obj_name in objects:
        print(f"\n{'='*50}")
        print(f"Processing: {obj_name}")

        dir_name = obj_name.replace(" ", "_")
        obj_output_dir = output_root / dir_name
        obj_output_dir.mkdir(parents=True, exist_ok=True)

        # Detect
        print("  Detecting...")
        bbox = detect_bbox(str(image_path), obj_name, w, h)
        if bbox is None:
            print(f"  Failed to detect {obj_name}")
            continue

        # Clamp bbox
        x1, y1, x2, y2 = bbox
        bbox = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
        print(f"  Bbox: {bbox}")

        # Save bbox visualization
        save_bbox_image(image_rgb, bbox, obj_name, obj_output_dir / "bbox.png")
        print("Saved: bbox.png")

        # Save bbox as JSON
        with open(obj_output_dir / "bbox.json", "w") as f:
            json.dump({"object": obj_name, "bbox": bbox}, f, indent=2)
        print("Saved: bbox.json")

        # Segment
        print("  Segmenting...")
        mask = segment_object(sam2_predictor, image_rgb, bbox)

        # Save mask
        save_mask_image(mask, obj_output_dir / "mask.png")
        print("Saved: mask.png")

        # Save masked image for visual verification
        save_masked_image(image_rgb, mask, obj_output_dir / "masked_image.png")
        print("Saved: masked_image.png")
        # Generate mesh
        print("  Generating mesh...")
        try:
            output = generate_mesh(sam3d, image_rgb, mask)
            save_outputs(output, obj_output_dir, dir_name)
        except Exception as e:
            print(f"  Mesh generation failed: {e}")

    print(f"\n{'='*50}")
    print("Done!")


if __name__ == "__main__":
    main()
