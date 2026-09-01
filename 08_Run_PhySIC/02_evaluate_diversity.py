#!/usr/bin/env python3
"""Evaluate PhySIC diversity with Module 06's authoritative metric."""

from __future__ import annotations

import argparse
from pathlib import Path

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    PROJECT_DIR,
    discover_physic_interactions,
    load_python_module,
    physic_eval_root,
    physic_output_root,
)


BASE = load_python_module(
    "module06_diversity_for_physic",
    PROJECT_DIR / "06_Evaluate_Interaction" / "02_evaluate_diversity.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySIC parameter diversity through Module 06."
    )
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--num_clusters", type=int, default=15)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--no_standardize", action="store_true")
    args = parser.parse_args()
    source_root = physic_output_root(args.output_mode)
    items = [
        (
            name,
            source_root / name / "debug" / "params" / "optimized_frame_0000.pt",
        )
        for name in discover_physic_interactions(args.output_mode)
    ]
    BASE.evaluate_parameter_diversity(
        param_items=items,
        output_root=(
            args.output_root.resolve()
            if args.output_root is not None
            else physic_eval_root(args.output_mode)
        ),
        num_clusters=args.num_clusters,
        max_iters=args.kmeans_iters,
        standardize=not args.no_standardize,
    )


if __name__ == "__main__":
    main()
