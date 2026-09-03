#!/usr/bin/env python3
"""Evaluate selected GenZI parameter diversity through Module 06."""

from __future__ import annotations

import argparse
from pathlib import Path

from genzi_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SELECTION_CONFIG,
    PROJECT_DIR,
    discover_genzi_interactions,
    genzi_eval_root,
    load_python_module,
    materialize_camera_params,
    select_genzi_candidate,
    write_selection_manifest,
)


BASE = load_python_module(
    "module06_diversity_for_genzi",
    PROJECT_DIR / "06_Evaluate_Interaction" / "02_evaluate_diversity.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate selected GenZI parameter diversity through Module 06."
    )
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument(
        "--selection_config", type=Path, default=DEFAULT_SELECTION_CONFIG
    )
    parser.add_argument("--num_clusters", type=int, default=15)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--no_standardize", action="store_true")
    args = parser.parse_args()

    names = discover_genzi_interactions(args.output_mode, args.selection_config)
    candidates = [
        select_genzi_candidate(name, args.output_mode, args.selection_config)
        for name in names
    ]
    write_selection_manifest(names, args.output_mode, args.selection_config)
    items = [
        (
            candidate.interaction_name,
            materialize_camera_params(candidate, args.output_mode),
        )
        for candidate in candidates
    ]
    BASE.evaluate_parameter_diversity(
        param_items=items,
        output_root=(
            args.output_root.resolve()
            if args.output_root is not None
            else genzi_eval_root(args.output_mode)
        ),
        num_clusters=args.num_clusters,
        max_iters=args.kmeans_iters,
        standardize=not args.no_standardize,
    )


if __name__ == "__main__":
    main()
