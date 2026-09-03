#!/usr/bin/env python3
"""Evaluate selected GenZI renders with Module 06's CLIP metric."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import CLIPModel, CLIPProcessor

from genzi_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SELECTION_CONFIG,
    PROJECT_DIR,
    discover_genzi_interactions,
    ensure_dir,
    genzi_eval_root,
    load_python_module,
    save_csv_rows,
    save_json,
    validate_render_selection,
    write_selection_manifest,
)


BASE = load_python_module(
    "module06_semantics_for_genzi",
    PROJECT_DIR / "06_Evaluate_Interaction" / "03_evaluate_semantics.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected GenZI renders with Module 06's CLIP metric."
    )
    parser.add_argument("--interaction_name", default="interaction_02")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--selection_config", type=Path, default=DEFAULT_SELECTION_CONFIG
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--clip_model", type=str, default=BASE.DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def eval_one(
    interaction_name: str,
    args: argparse.Namespace,
    model: CLIPModel,
    processor: CLIPProcessor,
    device,
) -> dict:
    validate_render_selection(interaction_name, args.output_mode, args.selection_config)
    render_root = genzi_eval_root(args.output_mode) / interaction_name / "semantics"
    output_root = Path(args.output_root).resolve() if args.output_root else render_root
    shim = argparse.Namespace(
        output_mode=args.output_mode,
        input_scene_json=str(
            PROJECT_DIR
            / "01_Generate_SIG"
            / "input_prompts"
            / interaction_name
            / "input_scene.json"
        ),
        render_root=str(render_root),
        output_root=str(output_root),
    )
    return BASE.evaluate_interaction_semantics(
        interaction_name=interaction_name,
        args=shim,
        model=model,
        processor=processor,
        device=device,
    )


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError(
                "--all_interactions cannot be combined with --output_root."
            )
        names = discover_genzi_interactions(args.output_mode, args.selection_config)
    else:
        names = [args.interaction_name]
    write_selection_manifest(names, args.output_mode, args.selection_config)

    device = BASE.parse_device(args.device)
    model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()
    rows = [eval_one(name, args, model, processor, device) for name in names]

    if all_mode:
        root = ensure_dir(genzi_eval_root(args.output_mode))
        mean_score = sum(float(row["clip_score"]) for row in rows) / len(rows)
        combined_rows = rows + [
            {
                "interaction_name": "__mean__",
                "clip_score": mean_score,
                "num_renders": sum(int(row["num_renders"]) for row in rows),
            }
        ]
        save_csv_rows(
            root / "semantics.csv",
            combined_rows,
            ["interaction_name", "clip_score", "num_renders"],
        )
        save_json(
            root / "semantics.json",
            {
                "interactions": rows,
                "aggregate": {
                    "num_interactions": len(rows),
                    "mean_clip_score": mean_score,
                },
            },
        )


if __name__ == "__main__":
    main()
