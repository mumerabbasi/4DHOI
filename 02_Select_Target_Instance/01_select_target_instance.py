from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


SAM3_CHECKPOINT = None
SAM3_BPE_PATH = "/my_workspace/4DHHOI/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_path(script_dir: Path, interaction_name: str, raw_input_dir: str | None) -> Path:
    if raw_input_dir:
        return Path(raw_input_dir).resolve()
    return script_dir.parent / "01_Generate_SIG" / "input_prompts" / interaction_name


def resolve_output_dir(script_dir: Path, interaction_name: str, raw_outdir: str | None) -> Path:
    if raw_outdir:
        return Path(raw_outdir).resolve()
    return script_dir / "output" / interaction_name


def resolve_scannet_root(script_dir: Path, raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_paths(scannet_root: Path, scene_context: dict[str, Any]) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]

    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )

    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "image_path": scene_root / image_rel / camera_name,
        "transforms_path": scene_root / transforms_rel,
    }


def resolve_target_label(sig_payload: dict[str, Any], sig_json_path: Path) -> str:
    target = sig_payload.get("target_object", {})
    label = str(target.get("label", "")).strip()
    if not label:
        raise ValueError(
            f"SIG target_object.label is empty or missing: {sig_json_path}"
        )
    return label


def build_sam3_processor(
    checkpoint_path: Path | None,
    bpe_path: Path | None,
    device: str,
    confidence_threshold: float,
    allow_hf_download: bool,
) -> Any:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    resolved_checkpoint_path = str(checkpoint_path) if checkpoint_path is not None else SAM3_CHECKPOINT
    resolved_bpe_path = str(bpe_path) if bpe_path is not None else SAM3_BPE_PATH

    model = build_sam3_image_model(
        bpe_path=resolved_bpe_path,
        checkpoint_path=resolved_checkpoint_path,
        device=device,
        load_from_HF=allow_hf_download,
    )
    return Sam3Processor(
        model=model,
        device=device,
        confidence_threshold=confidence_threshold,
    )


def run_sam3_text_prompts(
    processor: Any,
    image_rgb: Image.Image,
    prompts: list[str],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []

    for prompt in prompts:
        state = processor.set_image(image_rgb)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        mask_tensor = state["masks"]
        box_tensor = state["boxes"]
        score_tensor = state["scores"]

        for mask_index in range(mask_tensor.shape[0]):
            mask = mask_tensor[mask_index, 0].detach().cpu().numpy().astype(bool)
            if not np.any(mask):
                continue

            predictions.append(
                {
                    "prompt": prompt,
                    "mask_index": int(mask_index),
                    "mask": mask,
                    "bbox_xyxy": [
                        float(value)
                        for value in box_tensor[mask_index].detach().cpu().numpy().tolist()
                    ],
                    "sam3_score": float(score_tensor[mask_index].detach().cpu().item()),
                }
            )

    return predictions


def get_default_sam3_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_mask_stats(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise ValueError("Selected mask is empty.")

    return {
        "visible_bbox_xyxy": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
        "mask_area_px": int(mask.sum()),
    }


def select_highest_confidence_mask(sam3_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not sam3_predictions:
        raise ValueError("SAM3 did not return any masks.")
    return max(
        sam3_predictions,
        key=lambda item: (float(item["sam3_score"]), -int(np.count_nonzero(item["mask"]))),
    )


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Run SAM3 on the SIG target object and save the single best target mask."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=None)
    parser.add_argument("--sam3-device", default=get_default_sam3_device())
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
    args = parser.parse_args()

    input_dir = resolve_input_path(script_dir, args.interaction_name, args.input_dir)
    output_root = resolve_output_dir(script_dir, args.interaction_name, args.outdir)
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else project_dir / "01_Generate_SIG" / "output" / args.interaction_name / "scene_interaction_graph.json"
    )

    input_payload = load_json(input_dir / "input_scene.json")
    sig_payload = load_json(sig_json_path)
    scene_paths = resolve_scene_paths(scannet_root, input_payload["scene_context"])
    image_bgr = cv2.imread(str(scene_paths["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {scene_paths['image_path']}")
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    target_label = resolve_target_label(sig_payload, sig_json_path)

    sam3_processor = build_sam3_processor(
        checkpoint_path=Path(args.sam3_checkpoint).resolve() if args.sam3_checkpoint else None,
        bpe_path=Path(args.sam3_bpe_path).resolve() if args.sam3_bpe_path else None,
        device=args.sam3_device,
        confidence_threshold=args.sam3_confidence_threshold,
        allow_hf_download=not args.no_sam3_hf_download,
    )
    sam3_predictions = run_sam3_text_prompts(
        processor=sam3_processor,
        image_rgb=image_rgb,
        prompts=[target_label],
    )
    selected = select_highest_confidence_mask(sam3_predictions)
    selected_mask = selected["mask"]
    selected_stats = build_mask_stats(selected_mask)

    output_root.mkdir(parents=True, exist_ok=True)
    scene_image_path = output_root / "scene_image.png"
    target_mask_path = output_root / "target_mask.png"
    selection_json_path = output_root / "target_selection.json"

    cv2.imwrite(str(scene_image_path), image_bgr)
    save_mask(target_mask_path, selected_mask)

    selection_payload = {
        "label": str(selected["prompt"]),
        "target_mask_score": float(selected["sam3_score"]),
        "mask_path": target_mask_path.name,
        "sig_json": str(sig_json_path),
    }
    selection_json_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input scene: {input_dir / 'input_scene.json'}")
    print(f"SIG: {sig_json_path}")
    print(f"Scene image: {scene_paths['image_path']}")
    print(f"Target label: {target_label}")
    print(f"Saved target mask: {target_mask_path}")
    print(f"Saved selection JSON: {selection_json_path}")
    print(
        "Selected target:",
        {
            "selected_label": selected["prompt"],
            "sam3_model_score": selected["sam3_score"],
            "mask_area_px": selected_stats["mask_area_px"],
        },
    )


if __name__ == "__main__":
    main()
