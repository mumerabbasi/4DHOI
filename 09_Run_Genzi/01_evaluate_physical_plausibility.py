#!/usr/bin/env python3
"""Evaluate selected GenZI humans with Module 06's authoritative evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from genzi_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    DEFAULT_SELECTION_CONFIG,
    PROJECT_DIR,
    REPO_DIR,
    discover_genzi_interactions,
    genzi_eval_root,
    load_python_module,
    materialize_camera_params,
    select_genzi_candidate,
    write_selection_manifest,
)


BASE = load_python_module(
    "module06_physical_for_genzi",
    PROJECT_DIR / "06_Evaluate_Interaction" / "01_evaluate_physical_plausibility.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate selected GenZI outputs through Module 06."
    )
    parser.add_argument("--interaction_name", default="interaction_02")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument(
        "--selection_config", type=Path, default=DEFAULT_SELECTION_CONFIG
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = (
        discover_genzi_interactions(args.output_mode, args.selection_config)
        if args.all_interactions or args.interaction_name == "all"
        else [args.interaction_name]
    )
    candidates = [
        select_genzi_candidate(name, args.output_mode, args.selection_config)
        for name in names
    ]
    write_selection_manifest(names, args.output_mode, args.selection_config)
    model_spec = BASE.SMPLXModelSpec(
        model_path=REPO_DIR / "GenZI" / "data" / "smpl-x" / "models_smplx_v1_1",
        gender="neutral",
        use_pca=True,
        num_pca_comps=12,
        flat_hand_mean=False,
    )
    items = [
        BASE.ExternalHumanEvaluationInput(
            interaction_name=candidate.interaction_name,
            human_mesh_world=candidate.human_mesh_world,
            optimized_params_camera=materialize_camera_params(
                candidate, args.output_mode
            ),
            smplx_model=model_spec,
        )
        for candidate in candidates
    ]
    output_base = (
        args.output_root.resolve()
        if args.output_root is not None
        else genzi_eval_root(args.output_mode)
    )
    BASE.evaluate_external_world_humans(items, output_base, args.device)


if __name__ == "__main__":
    main()
