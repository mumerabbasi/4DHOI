#!/usr/bin/env python3
"""Evaluate PROX humans with Module 06's authoritative evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from prox_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_prox_interactions,
    load_python_module,
    prox_eval_root,
    prox_output_root,
)


BASE = load_python_module(
    "module06_physical_for_prox",
    PROJECT_DIR / "06_Evaluate_Interaction" / "01_evaluate_physical_plausibility.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PROX by swapping its optimized human into Module 06."
    )
    parser.add_argument("--interaction_name", default="interaction_02")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        discover_prox_interactions(args.output_mode)
        if args.all_interactions or args.interaction_name == "all"
        else [args.interaction_name]
    )
    source_root = prox_output_root(args.output_mode)
    items = [
        BASE.ExternalHumanEvaluationInput(
            interaction_name=name,
            human_mesh_world=source_root / name / "final_smplx_world.ply",
            optimized_params_camera=source_root / name / "result.pkl",
        )
        for name in names
    ]
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else prox_eval_root(args.output_mode)
    )
    BASE.evaluate_external_world_humans(items, output_base, args.device)


if __name__ == "__main__":
    main()
