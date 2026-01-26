"""
Generate 3D object meshes from an image using SAM2 for segmentation and SAM 3D Objects
for 3D reconstruction.

Pipeline:
1. Load PAG file to get all objects from the scene
2. Load the first frame image used for video generation
3. Use SAM2 (with Qwen-VL for detection) to segment each object
4. Use SAM 3D Objects to generate 3D mesh for each segmented object
5. Export meshes as GLB files to objects/<name>/ directories

Prerequisites:
- SAM2 installed and checkpoints available
- SAM 3D Objects installed (from https://github.com/facebookresearch/sam-3d-objects)
  Follow the setup instructions at: doc/setup.md
- Qwen-VL model running via Ollama for object detection

Usage:
    python generate_object_meshes.py --pag_file ../Generate_PAG/output_pag_deepseek_r1_32b.json
    python generate_object_meshes.py --image ../Generate_Video/first_frames/frame_00.png \
        --object_name iron --output_dir objects/iron
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import torch

# SAM2 imports
SAM2_PATH = "/my_workspace/4DHHOI/sam2"
sys.path.insert(0, SAM2_PATH)  # noqa: E402
from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

# SAM 3D Objects imports - will be imported dynamically when available
SAM3D_PATH = "/my_workspace/4DHHOI/sam-3d-objects"

# Ollama configuration for object detection
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
QWEN_MODEL = "qwen3-vl:32b"

# SAM2 configuration
SAM2_CHECKPOINT = "/my_workspace/4DHHOI/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM 3D Objects configuration
SAM3D_CHECKPOINT_TAG = "hf"


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def detect_object_bbox_qwen(
    image_path: str,
    object_name: str,
) -> Optional[list[int]]:
    """
    Use Qwen-VL to detect bounding box for an object.

    Args:
        image_path: Path to the image.
        object_name: Name of the object to detect (e.g., "iron").

    Returns:
        Bounding box [x1, y1, x2, y2] or None if not detected.
    """
    image_b64 = encode_image_base64(image_path)

    prompt = f"""Detect the {object_name} in this image and provide its bounding box.

    Provide the bounding box coordinates as [x1, y1, x2, y2] where:
    - x1, y1 = top-left corner (pixels from left/top)
    - x2, y2 = bottom-right corner (pixels from left/top)

    Respond with ONLY a JSON object in this exact format:
    {{
        "bbox": [x1, y1, x2, y2]
    }}

    If the {object_name} is not visible or cannot be detected, respond with:
    {{
        "bbox": null
    }}

    Coordinates should be integers representing pixel positions."""

    payload = {
        "model": QWEN_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "")

        # Remove think tags if present
        response_text = re.sub(
            r'<think>.*?</think>', '', response_text, flags=re.DOTALL
        )

        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get("bbox")
        else:
            print("  Warning: Could not parse JSON from response")
            return None

    except Exception as e:
        print(f"  Error calling Qwen-VL: {e}")
        return None


def load_sam2_predictor(
    checkpoint: str = SAM2_CHECKPOINT,
    config: str = SAM2_CONFIG,
) -> SAM2ImagePredictor:
    """Load SAM2 model for segmentation."""
    print("Loading SAM2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"SAM2 loaded on {device}")
    return predictor


def segment_object_sam2(
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
    box = np.array(bbox)

    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
    )

    return masks[0].astype(np.uint8)


def load_sam3d_inference(
    sam3d_path: str = SAM3D_PATH,
    checkpoint_tag: str = SAM3D_CHECKPOINT_TAG,
):
    """
    Load SAM 3D Objects inference pipeline.

    Args:
        sam3d_path: Path to SAM 3D Objects repository.
        checkpoint_tag: Checkpoint tag (e.g., "hf").

    Returns:
        Inference object from SAM 3D Objects.
    """
    print("Loading SAM 3D Objects model...")

    # Add SAM 3D Objects to path
    notebook_path = Path(sam3d_path) / "notebook"
    if str(notebook_path) not in sys.path:
        sys.path.insert(0, str(notebook_path))

    # Import inference module
    from inference import Inference  # noqa: E402

    # Load model
    config_path = Path(sam3d_path) / "checkpoints" / checkpoint_tag / "pipeline.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"SAM 3D Objects config not found at {config_path}. "
            "Please follow the setup instructions at doc/setup.md to download checkpoints."
        )

    inference = Inference(str(config_path), compile=False)
    print("SAM 3D Objects model loaded!")
    return inference


def generate_mesh_sam3d(
    inference,
    image: np.ndarray,
    mask: np.ndarray,
    seed: int = 42,
) -> dict:
    """
    Generate 3D mesh using SAM 3D Objects.

    Args:
        inference: SAM 3D Objects inference pipeline.
        image: RGB image as numpy array (H, W, 3), uint8.
        mask: Binary mask as numpy array (H, W), bool or uint8.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing:
        - "gs": Gaussian splat object (can save to PLY)
        - "glb": GLB mesh object (can export to GLB)
        - Other SAM 3D outputs
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # Ensure mask is boolean
    if mask.dtype != bool:
        mask = mask > 0

    # Run inference
    output = inference(image, mask, seed=seed)
    return output


def save_mesh_outputs(
    output: dict,
    output_dir: Path,
    object_name: str,
) -> dict[str, Path]:
    """
    Save mesh outputs to files.

    Args:
        output: SAM 3D Objects output dictionary.
        output_dir: Directory to save outputs.
        object_name: Name of the object.

    Returns:
        Dictionary mapping output type to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    # Save Gaussian splat as PLY
    if "gs" in output:
        ply_path = output_dir / f"{object_name}.ply"
        output["gs"].save_ply(str(ply_path))
        saved_files["ply"] = ply_path
        print(f"  Saved Gaussian splat: {ply_path}")

    # Save GLB mesh
    if "glb" in output and output["glb"] is not None:
        glb_path = output_dir / "mesh.glb"
        output["glb"].export(str(glb_path))
        saved_files["glb"] = glb_path
        print(f"  Saved GLB mesh: {glb_path}")

    return saved_files


def save_mask(mask: np.ndarray, output_path: str) -> None:
    """Save binary mask as PNG."""
    mask_img = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_img)


def save_bbox_visualization(
    image: np.ndarray,
    bbox: list[int],
    object_name: str,
    output_path: str,
) -> None:
    """Save image with bounding box drawn."""
    result = image.copy()
    if result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, object_name, (x1, y1 - 10), font, 0.6, (0, 255, 0), 2)

    cv2.imwrite(output_path, result)


def parse_pag_file(pag_path: str) -> list[str]:
    """
    Parse PAG file to extract unique object names.

    Args:
        pag_path: Path to PAG JSON file.

    Returns:
        List of unique object names (e.g., ["iron", "ironing board"]).
    """
    with open(pag_path) as f:
        pag = json.load(f)

    object_names = set()
    for node in pag.get("object part nodes", []):
        # Format: "object_name, part_name"
        parts = node.split(", ", 1)
        if len(parts) >= 1:
            object_names.add(parts[0])

    return sorted(list(object_names))


def process_single_object(
    image_path: str,
    object_name: str,
    output_dir: Path,
    sam2_predictor: SAM2ImagePredictor,
    sam3d_inference,
    seed: int = 42,
    skip_existing: bool = False,
) -> dict:
    """
    Process a single object: detect, segment, and generate 3D mesh.

    Args:
        image_path: Path to input image.
        object_name: Name of the object to process.
        output_dir: Output directory for this object.
        sam2_predictor: SAM2 predictor for segmentation.
        sam3d_inference: SAM 3D Objects inference pipeline.
        seed: Random seed.
        skip_existing: Skip if mesh already exists.

    Returns:
        Dictionary with processing results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if mesh already exists
    glb_path = output_dir / "mesh.glb"
    if skip_existing and glb_path.exists():
        print(f"  Skipping {object_name}: mesh already exists")
        return {"status": "skipped", "object_name": object_name}

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    print(f"\n  Processing: {object_name}")
    print(f"  Image size: {w}x{h}")

    # Step 1: Detect object bounding box
    print("  Step 1: Detecting object with Qwen-VL...")
    bbox = detect_object_bbox_qwen(image_path, object_name)

    if bbox is None:
        print(f"  Warning: Could not detect {object_name}")
        return {"status": "detection_failed", "object_name": object_name}

    # Validate and clamp bbox
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    bbox = [x1, y1, x2, y2]
    print(f"  Detected bbox: {bbox}")

    # Save bbox visualization
    bbox_vis_path = output_dir / "bbox_detection.png"
    save_bbox_visualization(image_rgb, bbox, object_name, str(bbox_vis_path))
    print(f"  Saved bbox visualization: {bbox_vis_path}")

    # Step 2: Segment object with SAM2
    print("  Step 2: Segmenting with SAM2...")
    mask = segment_object_sam2(sam2_predictor, image_rgb, bbox)

    # Save mask
    mask_path = output_dir / "segmentation_mask.png"
    save_mask(mask, str(mask_path))
    print(f"  Saved segmentation mask: {mask_path}")

    # Step 3: Generate 3D mesh with SAM 3D Objects
    print("  Step 3: Generating 3D mesh with SAM 3D Objects...")
    try:
        output = generate_mesh_sam3d(sam3d_inference, image_rgb, mask, seed=seed)

        # Save outputs
        saved_files = save_mesh_outputs(output, output_dir, object_name)

        return {
            "status": "success",
            "object_name": object_name,
            "bbox": bbox,
            "saved_files": saved_files,
        }

    except Exception as e:
        print(f"  Error generating mesh: {e}")
        return {
            "status": "mesh_generation_failed",
            "object_name": object_name,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D meshes from image using SAM2 + SAM 3D Objects."
    )

    # Input options
    parser.add_argument(
        "--pag_file",
        type=str,
        default="../Generate_PAG/output_pag_deepseek_r1_32b.json",
        help="PAG JSON file to extract object names from.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="../Generate_Video/first_frames/frame_00.png",
        help="Input image path.",
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Single object name to process (overrides PAG file).",
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="objects",
        help="Root output directory for object meshes (default: objects).",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip objects that already have mesh.glb.",
    )

    # Model configuration
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default=SAM2_CHECKPOINT,
        help="Path to SAM2 checkpoint.",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default=SAM2_CONFIG,
        help="SAM2 model config name.",
    )
    parser.add_argument(
        "--sam3d_path",
        type=str,
        default=SAM3D_PATH,
        help="Path to SAM 3D Objects repository.",
    )
    parser.add_argument(
        "--sam3d_checkpoint_tag",
        type=str,
        default=SAM3D_CHECKPOINT_TAG,
        help="SAM 3D Objects checkpoint tag (default: hf).",
    )

    # Other options
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default=OLLAMA_URL,
        help="Ollama API URL for Qwen-VL.",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=QWEN_MODEL,
        help="Qwen-VL model name in Ollama.",
    )

    args = parser.parse_args()

    # Update global configurations from args
    global OLLAMA_URL, QWEN_MODEL
    OLLAMA_URL = args.ollama_url
    QWEN_MODEL = args.qwen_model

    # Resolve paths
    image_path = Path(args.image).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return

    output_root = Path(args.output_dir).resolve()

    # Determine objects to process
    if args.object_name:
        # Single object mode
        object_names = [args.object_name]
        print(f"Processing single object: {args.object_name}")
    else:
        # PAG file mode
        pag_path = Path(args.pag_file).resolve()
        if not pag_path.exists():
            print(f"Error: PAG file not found: {pag_path}")
            return

        object_names = parse_pag_file(str(pag_path))
        if not object_names:
            print("No objects found in PAG file")
            return

        print(f"Found {len(object_names)} objects in PAG file:")
        for name in object_names:
            print(f"  - {name}")

    # Load models
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    sam2_predictor = load_sam2_predictor(
        checkpoint=args.sam2_checkpoint,
        config=args.sam2_config,
    )

    sam3d_inference = load_sam3d_inference(
        sam3d_path=args.sam3d_path,
        checkpoint_tag=args.sam3d_checkpoint_tag,
    )

    # Process each object
    results = []
    for object_name in object_names:
        print("\n" + "=" * 60)
        print(f"Processing: {object_name}")
        print("=" * 60)

        # Create output directory (replace spaces with underscores)
        dir_name = object_name.replace(" ", "_")
        object_output_dir = output_root / dir_name

        result = process_single_object(
            image_path=str(image_path),
            object_name=object_name,
            output_dir=object_output_dir,
            sam2_predictor=sam2_predictor,
            sam3d_inference=sam3d_inference,
            seed=args.seed,
            skip_existing=args.skip_existing,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    success_count = sum(1 for r in results if r["status"] == "success")
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    failed_count = len(results) - success_count - skipped_count

    print(f"Total objects: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")

    if failed_count > 0:
        print("\nFailed objects:")
        for r in results:
            if r["status"] not in ["success", "skipped"]:
                print(f"  - {r['object_name']}: {r['status']}")

    # Save results JSON
    results_path = output_root / "mesh_generation_results.json"
    with open(results_path, "w") as f:
        # Convert Path objects to strings for JSON serialization
        serializable_results = []
        for r in results:
            r_copy = r.copy()
            if "saved_files" in r_copy:
                r_copy["saved_files"] = {
                    k: str(v) for k, v in r_copy["saved_files"].items()
                }
            serializable_results.append(r_copy)
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved results to: {results_path}")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
