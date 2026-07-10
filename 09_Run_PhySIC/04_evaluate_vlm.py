#!/usr/bin/env python3
"""Evaluate PhySIC renders with the same VLM rubric as module 06."""

from __future__ import annotations

import argparse
from pathlib import Path

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    ensure_dir,
    load_python_module,
    physic_eval_root,
    save_csv_rows,
)


BASE = load_python_module(
    "module06_vlm",
    PROJECT_DIR / "06_Evaluate_Interaction" / "04_evaluate_vlm.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySIC renders with the module-06 VLM verifier."
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--aggregate_evals",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vlm_provider",
        choices=BASE.VLM_PROVIDERS,
        default="gemini",
    )
    parser.add_argument("--qwen_model", type=str, default=BASE.DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemini_model", type=str, default=BASE.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--gemini_api_key_file",
        type=str,
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--gemini_max_output_tokens", type=int, default=4096)
    parser.add_argument("--gemini_retries", type=int, default=3)
    parser.add_argument("--gemini_retry_sleep_s", type=float, default=10.0)
    args = parser.parse_args()
    if args.model is not None:
        args.qwen_model = args.model
    return args


def make_base_args(interaction_name: str, args: argparse.Namespace) -> argparse.Namespace:
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode) / interaction_name / "vlm"
    )
    return argparse.Namespace(
        output_mode=args.output_mode,
        input_scene_json=str(
            PROJECT_DIR
            / "01_Generate_SIG"
            / "input_prompts"
            / interaction_name
            / "input_scene.json"
        ),
        render_root=str(physic_eval_root(args.output_mode) / interaction_name / "semantics"),
        output_root=str(output_root),
        vlm_provider=args.vlm_provider,
        qwen_model=args.qwen_model,
        gemini_model=args.gemini_model,
        ollama_host=args.ollama_host,
        gemini_api_key_file=args.gemini_api_key_file,
        max_image_side=args.max_image_side,
        temperature=args.temperature,
        seed=args.seed,
        timeout_s=args.timeout_s,
        gemini_max_output_tokens=args.gemini_max_output_tokens,
        gemini_retries=args.gemini_retries,
        gemini_retry_sleep_s=args.gemini_retry_sleep_s,
    )


def aggregate(output_mode: str) -> None:
    output_root = physic_eval_root(output_mode)
    metrics_csv_paths = sorted(output_root.glob("interaction_*/vlm/metrics.csv"))
    if not metrics_csv_paths:
        raise FileNotFoundError(
            f"No VLM metrics.csv files found under {output_root}/interaction_*/vlm."
        )
    rows = []
    for metrics_csv_path in metrics_csv_paths:
        row = {"interaction_name": metrics_csv_path.parent.parent.name}
        row.update(BASE.load_vlm_metrics_row(metrics_csv_path))
        rows.append(row)
    mean_row = {"interaction_name": "__mean__"}
    for fieldname in BASE.CSV_FIELDNAMES:
        values = [
            float(row[fieldname])
            for row in rows
            if isinstance(row.get(fieldname), int | float)
        ]
        mean_row[fieldname] = float(sum(values) / len(values)) if values else None
    ensure_dir(output_root)
    save_csv_rows(
        output_root / "vlm.csv",
        rows + [mean_row],
        BASE.AGGREGATE_CSV_FIELDNAMES,
    )
    print(f"Saved {output_root / 'vlm.csv'} with {len(rows)} interactions.")


def main() -> None:
    args = parse_args()
    if args.aggregate_evals:
        aggregate(args.output_mode)
        return

    prompt_template_override = (
        args.prompt_template
        if args.prompt_template is not None
        else args.system_prompt
    )
    prompt_template_path = BASE.resolve_path(
        prompt_template_override,
        BASE.DEFAULT_PROMPT_TEMPLATE_PATH,
    )
    prompt_template = BASE.load_text(prompt_template_path)
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError("--all_interactions cannot be combined with --output_root.")
        interaction_names = discover_physic_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    for interaction_name in interaction_names:
        BASE.evaluate_interaction_vlm(
            interaction_name=interaction_name,
            args=make_base_args(interaction_name, args),
            prompt_template=prompt_template,
            prompt_template_path=prompt_template_path,
        )

    if all_mode:
        aggregate(args.output_mode)


if __name__ == "__main__":
    main()
