#!/usr/bin/env python3
"""Evaluate PROX renders with the same VLM rubric as Module 06 and PhySIC."""

from __future__ import annotations

import argparse
from pathlib import Path

from prox_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_prox_interactions,
    ensure_dir,
    load_python_module,
    prox_eval_root,
    save_csv_rows,
)


BASE = load_python_module(
    "module06_vlm_for_prox",
    PROJECT_DIR / "06_Evaluate_Interaction" / "04_evaluate_vlm.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PROX renders with the Module 06 VLM verifier."
    )
    parser.add_argument("--interaction_name", default="interaction_02")
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
    parser.add_argument("--vlm_provider", choices=BASE.VLM_PROVIDERS, default="gemini")
    parser.add_argument("--qwen_model", default=BASE.DEFAULT_QWEN_MODEL)
    parser.add_argument("--gemini_model", default=BASE.DEFAULT_GEMINI_MODEL)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ollama_host", default="http://localhost:11434")
    parser.add_argument(
        "--gemini_api_key_file",
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--prompt_template", default=None)
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=BASE.DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--gemini_max_output_tokens", type=int, default=4096)
    parser.add_argument("--gemini_retries", type=int, default=3)
    parser.add_argument("--gemini_retry_sleep_s", type=float, default=10.0)
    args = parser.parse_args()
    if args.model is not None:
        args.qwen_model = args.model
    return args


def make_base_args(
    interaction_name: str,
    args: argparse.Namespace,
) -> argparse.Namespace:
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else prox_eval_root(args.output_mode) / interaction_name / "vlm"
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
        render_root=str(prox_eval_root(args.output_mode) / interaction_name / "semantics"),
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
    output_root = prox_eval_root(output_mode)
    paths = sorted(output_root.glob("interaction_*/vlm/metrics.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No VLM metrics.csv files found under {output_root}/interaction_*/vlm."
        )
    rows = []
    for path in paths:
        row = {"interaction_name": path.parent.parent.name}
        row.update(BASE.load_vlm_metrics_row(path))
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
    prompt_override = args.prompt_template if args.prompt_template else args.system_prompt
    prompt_path = BASE.resolve_path(prompt_override, BASE.DEFAULT_PROMPT_TEMPLATE_PATH)
    prompt = BASE.load_text(prompt_path)
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if args.output_root is not None:
            raise ValueError("--all_interactions cannot be combined with --output_root.")
        names = discover_prox_interactions(args.output_mode)
    else:
        names = [args.interaction_name]
    for name in names:
        BASE.evaluate_interaction_vlm(
            interaction_name=name,
            args=make_base_args(name, args),
            prompt_template=prompt,
            prompt_template_path=prompt_path,
        )
    if all_mode:
        aggregate(args.output_mode)


if __name__ == "__main__":
    main()
