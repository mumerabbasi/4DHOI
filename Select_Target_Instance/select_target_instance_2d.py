from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openai import OpenAI
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


def normalize_label(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def resolve_input_path(script_dir: Path, video_name: str, raw_input_dir: str | None) -> Path:
    if raw_input_dir:
        return Path(raw_input_dir).resolve()
    return script_dir / "input_prompts" / video_name


def resolve_output_dir(script_dir: Path, video_name: str, raw_outdir: str | None) -> Path:
    if raw_outdir:
        return Path(raw_outdir).resolve()
    return script_dir / "output" / video_name


def resolve_stage_output_dir(output_root: Path) -> Path:
    return output_root / "2d"


def resolve_scannet_root(script_dir: Path, raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_paths(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> dict[str, Path]:
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


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_prompt_request(interaction: str) -> dict[str, Any]:
    return {"interaction": interaction}


def request_target_prompt_plan(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    text = strip_json_fence(response.choices[0].message.content)
    return json.loads(text)


def build_sam3_prompt_list(prompt_plan: dict[str, Any]) -> list[str]:
    prompt_candidates: list[str] = []

    raw_target_object_phrase = str(prompt_plan.get("target_object_phrase", "")).strip()
    if raw_target_object_phrase:
        prompt_candidates.append(raw_target_object_phrase)

    raw_primary = str(prompt_plan.get("sam3_prompt", "")).strip()
    if raw_primary:
        prompt_candidates.append(raw_primary)

    raw_fallbacks = prompt_plan.get("fallback_prompts", [])
    if isinstance(raw_fallbacks, list):
        for item in raw_fallbacks:
            text = str(item).strip()
            if text:
                prompt_candidates.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompt_candidates:
        norm = normalize_label(prompt)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(prompt)
    return deduped


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

            box_xyxy = box_tensor[mask_index].detach().cpu().numpy().tolist()
            predictions.append(
                {
                    "prompt": prompt,
                    "mask_index": int(mask_index),
                    "mask": mask,
                    "bbox_xyxy": [float(value) for value in box_xyxy],
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


def select_highest_confidence_mask(
    sam3_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
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

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2D target selection stage: build a SAM3 text prompt plan from the "
            "interaction, run SAM3, and save the single best mask."
        ),
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--host", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:27b")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="high",
        help="Reasoning control for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--interaction-override",
        default=None,
        help="Optional text override for interaction_context.interaction.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default=None,
        help="Optional local SAM3 checkpoint path. If omitted, SAM3 can download from Hugging Face.",
    )
    parser.add_argument(
        "--sam3-bpe-path",
        default=None,
        help="Optional local SAM3 BPE path. Defaults to the bundled path in the local sam3 checkout.",
    )
    parser.add_argument(
        "--sam3-device",
        default=get_default_sam3_device(),
        help="Device for SAM3 inference, for example 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--sam3-confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold used by Sam3Processor.",
    )
    parser.add_argument(
        "--no-sam3-hf-download",
        action="store_true",
        help="Disable automatic SAM3 checkpoint download from Hugging Face.",
    )
    args = parser.parse_args()

    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_target_instance_sam3.md"
    )
    input_dir = resolve_input_path(script_dir, args.video_name, args.input_dir)
    output_root = resolve_output_dir(script_dir, args.video_name, args.outdir)
    stage_output_dir = resolve_stage_output_dir(output_root)
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)

    input_payload = load_json(input_dir / "input_pag.json")
    scene_context = input_payload["scene_context"]
    interaction = args.interaction_override or input_payload["interaction_context"]["interaction"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)

    image_bgr = cv2.imread(str(scene_paths["image_path"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {scene_paths['image_path']}")
    image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    prompt_request = build_prompt_request(interaction=interaction)
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort
    client = OpenAI(base_url=args.host, api_key="ollama")
    prompt_plan = request_target_prompt_plan(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        user_payload=prompt_request,
        temperature=args.temperature,
        reasoning_effort=reasoning_effort,
    )

    sam3_prompts = build_sam3_prompt_list(prompt_plan)
    if not sam3_prompts:
        raise ValueError("No SAM3 prompts could be constructed from the prompt planner output.")

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
        prompts=sam3_prompts,
    )
    if not sam3_predictions:
        raise ValueError(
            "SAM3 did not return any masks for the generated prompt set. "
            f"Prompts tried: {sam3_prompts}"
        )

    selected_sam3_prediction = select_highest_confidence_mask(sam3_predictions)
    selected_mask = selected_sam3_prediction["mask"]
    selected_stats = build_mask_stats(selected_mask)

    output_root.mkdir(parents=True, exist_ok=True)
    stage_output_dir.mkdir(parents=True, exist_ok=True)
    scene_image_path = stage_output_dir / "scene_image.png"
    target_mask_path = stage_output_dir / "target_mask.png"
    selection_json_path = output_root / "target_selection.json"

    cv2.imwrite(str(scene_image_path), image_bgr)
    save_mask(target_mask_path, selected_mask)

    selection_payload = {
        "target_selection_2d": {
            "sam3_prompt": str(selected_sam3_prediction["prompt"]),
            "target_mask_score": float(selected_sam3_prediction["sam3_score"]),
            "mask_path": str(Path("2d") / target_mask_path.name),
        }
    }
    selection_json_path.write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {stage_output_dir}")
    print(f"Scene image: {scene_paths['image_path']}")
    print(f"System prompt: {system_prompt_path}")
    print(f"SAM3 prompts tried: {sam3_prompts}")
    print(f"Saved scene image copy: {scene_image_path}")
    print(f"Saved shared selection JSON: {selection_json_path}")
    print(f"Saved target mask: {target_mask_path}")
    print(
        "Selected 2D target:",
        {
            "selected_sam3_prompt": selected_sam3_prediction["prompt"],
            "sam3_model_score": selected_sam3_prediction["sam3_score"],
            "mask_area_px": selected_stats["mask_area_px"],
        },
    )


if __name__ == "__main__":
    main()
