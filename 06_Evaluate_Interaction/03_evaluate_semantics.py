from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def save_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "input_scene_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        "render_root": SCRIPT_DIR / "output" / interaction_name / "semantics",
        "output_root": SCRIPT_DIR / "output" / interaction_name / "semantics",
    }


def discover_interactions() -> list[str]:
    output_root = PROJECT_DIR / "05_Optimize_Static_Scene" / "output"
    names = [
        path.name
        for path in sorted(output_root.glob("interaction_*"))
        if (path / "meshes" / "frame_0000_world.ply").exists()
    ]
    if not names:
        raise RuntimeError(f"No optimized interactions found under {output_root}.")
    return names


def resolve_prompt(input_scene_json_path: Path) -> str:
    input_payload = load_json(input_scene_json_path)
    interaction_context = input_payload.get("interaction_context", {})
    prompt = ""
    if isinstance(interaction_context, dict):
        prompt = str(interaction_context.get("interaction", "")).strip()
    if not prompt:
        raise ValueError(
            f"Could not resolve interaction_context.interaction from "
            f"{input_scene_json_path}"
        )
    return prompt


def collect_render_paths(render_root: Path) -> list[Path]:
    render_dir = render_root / "renders"
    image_paths = sorted(
        path
        for path in render_dir.glob("view_*.png")
        if path.stem.removeprefix("view_").isdigit()
    )
    if image_paths:
        return image_paths
    return []


def parse_device(raw_device: str) -> torch.device:
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def compute_clip_score(
    image_paths: list[Path],
    prompt: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    if not image_paths:
        raise ValueError("compute_clip_score requires at least one image.")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds
        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        scores = image_features @ text_features.T
    score_values = scores[:, 0].detach().cpu().numpy().astype(float).tolist()
    per_view = [
        {
            "render_path": str(path),
            "clip_score": float(score),
        }
        for path, score in zip(image_paths, score_values)
    ]
    return float(sum(score_values) / len(score_values)), per_view


def evaluate_interaction_semantics(
    interaction_name: str,
    args: argparse.Namespace,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name)
    input_scene_json_path = resolve_path(
        args.input_scene_json,
        defaults["input_scene_json"],
    )
    render_root = resolve_path(args.render_root, defaults["render_root"])
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))

    render_paths = collect_render_paths(render_root)
    if not render_paths:
        raise FileNotFoundError(
            f"No renders found for {interaction_name} under {render_root / 'renders'}. "
            "Run 03a_render_interaction.py first."
        )
    prompt = resolve_prompt(input_scene_json_path)
    clip_score, per_view = compute_clip_score(
        image_paths=render_paths,
        prompt=prompt,
        model=model,
        processor=processor,
        device=device,
    )
    row = {
        "interaction_name": interaction_name,
        "clip_score": clip_score,
        "num_renders": int(len(per_view)),
    }
    save_csv_rows(
        output_root / "metrics.csv",
        [row],
        fieldnames=["interaction_name", "clip_score", "num_renders"],
    )
    save_json(
        output_root / "metrics.json",
        {
            "interaction_name": interaction_name,
            "prompt": prompt,
            "clip_score": clip_score,
            "renders": per_view,
        },
    )
    print(f"{interaction_name}: clip_score={clip_score:.6f}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate semantic consistency with CLIP image-text cosine "
            "similarity over Blender renders."
        )
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--render_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if any(
            value is not None
            for value in (args.input_scene_json, args.render_root, args.output_root)
        ):
            raise ValueError(
                "--all_interactions cannot be combined with per-interaction "
                "input/render/output overrides."
            )
        interaction_names = discover_interactions()
        combined_output_root = ensure_dir(SCRIPT_DIR / "output")
    else:
        interaction_names = [args.interaction_name]
        combined_output_root = None

    device = parse_device(args.device)
    print(f"Loading CLIP model: {args.clip_model}")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
    model.eval()

    rows = [
        evaluate_interaction_semantics(
            interaction_name=interaction_name,
            args=args,
            model=model,
            processor=processor,
            device=device,
        )
        for interaction_name in interaction_names
    ]

    if all_mode:
        mean_clip_score = float(sum(row["clip_score"] for row in rows) / len(rows))
        save_csv_rows(
            combined_output_root / "semantics.csv",
            rows
            + [
                {
                    "interaction_name": "__mean__",
                    "clip_score": mean_clip_score,
                    "num_renders": int(sum(int(row["num_renders"]) for row in rows)),
                }
            ],
            fieldnames=["interaction_name", "clip_score", "num_renders"],
        )
        save_json(
            combined_output_root / "semantics.json",
            {
                "interactions": rows,
                "aggregate": {
                    "num_interactions": int(len(rows)),
                    "mean_clip_score": mean_clip_score,
                },
            },
        )
        print(f"mean_clip_score={mean_clip_score:.6f}")


if __name__ == "__main__":
    main()
