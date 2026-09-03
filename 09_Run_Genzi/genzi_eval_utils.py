#!/usr/bin/env python3
"""Shared selection, path, and parameter adapters for GenZI evaluation."""

from __future__ import annotations

import csv
import importlib.util
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_OUTPUT_MODE = "output"
DEFAULT_SELECTION_CONFIG = SCRIPT_DIR / "evaluation_selection.json"
FINAL_CANDIDATE_COUNT = 4


@dataclass(frozen=True)
class GenZICandidate:
    interaction_name: str
    index: int
    candidate_dir: Path
    human_mesh_world: Path
    smplx_params_world: Path
    genzi_run_dir: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def genzi_output_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / output_mode


def genzi_interaction_root(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> Path:
    return genzi_output_root(output_mode) / interaction_name


def genzi_eval_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / "evaluation" / output_mode


def interaction_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 10**9, name


def resolve_selection_config(path: Path | None = None) -> Path:
    return (path or DEFAULT_SELECTION_CONFIG).resolve()


def load_evaluation_selections(path: Path | None = None) -> dict[str, int]:
    config_path = resolve_selection_config(path)
    payload = load_json(config_path)
    selections: dict[str, int] = {}
    for name, value in payload.items():
        if not isinstance(name, str) or not name.startswith("interaction_"):
            raise ValueError(f"Invalid interaction key in {config_path}: {name!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Selection for {name} must be an integer from 0 to 3.")
        if not 0 <= value < FINAL_CANDIDATE_COUNT:
            raise ValueError(f"Selection for {name} must be in [0, 3], got {value}.")
        selections[name] = value
    if not selections:
        raise ValueError(f"No interaction selections found in {config_path}.")
    return selections


def list_final_candidates(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> list[GenZICandidate]:
    interaction_root = genzi_interaction_root(interaction_name, output_mode)
    summary_path = interaction_root / "genzi_run_summary.json"
    summary = load_json(summary_path)
    run_dir = Path(str(summary["genzi_log_dir"])).resolve()
    scene_output_root = run_dir / interaction_name / interaction_name
    if not scene_output_root.is_dir():
        raise FileNotFoundError(
            f"Missing GenZI scene output directory: {scene_output_root}"
        )

    # Match the stable alphabetical order shown by file managers. Only final
    # stage-1 outputs participate; intermediate stage-0 folders are excluded.
    candidate_dirs = sorted(
        path
        for path in scene_output_root.glob("stage*_stage001_inpaint*")
        if path.is_dir()
        and (path / "optim_human.ply").is_file()
        and (path / "smplx.pkl").is_file()
    )
    if len(candidate_dirs) != FINAL_CANDIDATE_COUNT:
        raise RuntimeError(
            f"Expected exactly four final GenZI outputs for {interaction_name}, "
            f"found {len(candidate_dirs)} under {scene_output_root}."
        )
    return [
        GenZICandidate(
            interaction_name=interaction_name,
            index=index,
            candidate_dir=candidate_dir,
            human_mesh_world=candidate_dir / "optim_human.ply",
            smplx_params_world=candidate_dir / "smplx.pkl",
            genzi_run_dir=run_dir,
        )
        for index, candidate_dir in enumerate(candidate_dirs)
    ]


def select_genzi_candidate(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    selection_config: Path | None = None,
) -> GenZICandidate:
    selections = load_evaluation_selections(selection_config)
    if interaction_name not in selections:
        raise KeyError(
            f"No GenZI evaluation selection for {interaction_name} in "
            f"{resolve_selection_config(selection_config)}."
        )
    return list_final_candidates(interaction_name, output_mode)[
        selections[interaction_name]
    ]


def discover_genzi_interactions(
    output_mode: str = DEFAULT_OUTPUT_MODE,
    selection_config: Path | None = None,
) -> list[str]:
    selections = load_evaluation_selections(selection_config)
    names = sorted(selections, key=interaction_sort_key)
    for name in names:
        select_genzi_candidate(name, output_mode, selection_config)
    return names


def write_selection_manifest(
    interaction_names: list[str],
    output_mode: str = DEFAULT_OUTPUT_MODE,
    selection_config: Path | None = None,
) -> Path:
    config_path = resolve_selection_config(selection_config)
    selections = load_evaluation_selections(config_path)
    requested_names = set(interaction_names)
    unknown = sorted(requested_names - set(selections), key=interaction_sort_key)
    if unknown:
        raise KeyError(f"Selections are missing for: {', '.join(unknown)}")
    entries = []
    for name in sorted(selections, key=interaction_sort_key):
        candidates = list_final_candidates(name, output_mode)
        selected_index = selections[name]
        entries.append(
            {
                "interaction_name": name,
                "selected_index": selected_index,
                "selected_candidate_dir": str(candidates[selected_index].candidate_dir),
                "candidates": [
                    {
                        "index": candidate.index,
                        "directory_name": candidate.candidate_dir.name,
                        "candidate_dir": str(candidate.candidate_dir),
                        "human_mesh_world": str(candidate.human_mesh_world),
                        "smplx_params_world": str(candidate.smplx_params_world),
                    }
                    for candidate in candidates
                ],
            }
        )
    path = genzi_eval_root(output_mode) / "selection_manifest.json"
    save_json(
        path,
        {
            "selection_config": str(config_path),
            "candidate_order": "alphabetical final stage-1 directory order",
            "interactions": entries,
        },
    )
    return path


def validate_render_selection(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
    selection_config: Path | None = None,
) -> Path:
    candidate = select_genzi_candidate(interaction_name, output_mode, selection_config)
    render_config_path = (
        genzi_eval_root(output_mode)
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    render_config = load_json(render_config_path)
    rendered_index = render_config.get("selected_candidate_index")
    rendered_dir = Path(str(render_config.get("selected_candidate_dir", ""))).resolve()
    if (
        rendered_index != candidate.index
        or rendered_dir != candidate.candidate_dir.resolve()
    ):
        raise RuntimeError(
            f"Existing renders for {interaction_name} use GenZI candidate "
            f"{rendered_index} at {rendered_dir}, but the current selection "
            f"resolves to candidate {candidate.index} at {candidate.candidate_dir}. "
            "Run 03a_render_interactions.py again before semantic or VLM evaluation."
        )
    return render_config_path


def _load_smplx_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as file_obj:
        payload = pickle.load(file_obj, encoding="latin1")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an SMPL-X parameter dictionary in {path}.")
    required = ("vertices", "joints", "transl", "global_orient", "body_pose", "betas")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing GenZI SMPL-X values in {path}: {missing}")
    return payload


def validate_candidate_mesh(
    candidate: GenZICandidate, tolerance_m: float = 1e-5
) -> float:
    payload = _load_smplx_pickle(candidate.smplx_params_world)
    mesh = trimesh.load(str(candidate.human_mesh_world), force="mesh", process=False)
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float32)
    param_vertices = np.asarray(payload["vertices"], dtype=np.float32)
    if mesh_vertices.shape != param_vertices.shape:
        raise ValueError(
            f"GenZI mesh and smplx.pkl vertex shapes differ for "
            f"{candidate.interaction_name}: {mesh_vertices.shape} vs {param_vertices.shape}."
        )
    error = float(
        np.linalg.norm(mesh_vertices - param_vertices, axis=1).max(initial=0.0)
    )
    if not np.isfinite(error) or error > float(tolerance_m):
        raise RuntimeError(
            f"GenZI mesh does not match smplx.pkl for {candidate.interaction_name}: "
            f"max error={error:.9f}m."
        )
    return error


def _source_camera_transform(interaction_name: str) -> tuple[np.ndarray, np.ndarray]:
    input_path = (
        PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json"
    )
    scene_context = load_json(input_path)["scene_context"]
    scene_id = str(scene_context["scene_id"])
    camera_name = str(scene_context["camera"]["name"])
    camera_source = str(scene_context["camera"]["source"])
    if camera_source != "dslr_resized_undistorted":
        raise ValueError(
            f"Unsupported camera source for {interaction_name}: {camera_source!r}"
        )
    images_path = (
        REPO_DIR / "Scannet++" / "data" / scene_id / "dslr" / "colmap" / "images.txt"
    )
    for line in images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qw, qx, qy, qz = map(float, parts[1:5])
        rotation = np.asarray(
            [
                [
                    1 - 2 * (qy * qy + qz * qz),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx * qx + qz * qz),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx * qx + qy * qy),
                ],
            ],
            dtype=np.float32,
        )
        translation = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return rotation, translation
    raise ValueError(f"Could not find camera {camera_name!r} in {images_path}.")


def materialize_camera_params(
    candidate: GenZICandidate,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> Path:
    """Convert GenZI world-frame SMPL-X parameters to Module 06 camera space."""
    mesh_error = validate_candidate_mesh(candidate)
    payload = _load_smplx_pickle(candidate.smplx_params_world)
    rotation_w2c, translation_w2c = _source_camera_transform(candidate.interaction_name)

    transl_world = np.asarray(payload["transl"], dtype=np.float32).reshape(3)
    orient_world = torch.as_tensor(
        payload["global_orient"], dtype=torch.float32
    ).reshape(1, 3)
    rotation_world = axis_angle_to_matrix(orient_world)[0]
    rotation_camera = torch.from_numpy(rotation_w2c).float() @ rotation_world
    orient_camera = matrix_to_axis_angle(rotation_camera[None])[0].cpu().numpy()

    joints_world = np.asarray(payload["joints"], dtype=np.float32).reshape(-1, 3)
    root_joint_untranslated = joints_world[0] - transl_world
    transl_camera = (
        (root_joint_untranslated + transl_world) @ rotation_w2c.T
        + translation_w2c
        - root_joint_untranslated
    )

    converted: dict[str, Any] = {
        "transl": transl_camera.reshape(1, 3).astype(np.float32),
        "global_orient": orient_camera.reshape(1, 3).astype(np.float32),
        "body_pose": np.asarray(payload["body_pose"], dtype=np.float32),
        "betas": np.asarray(payload["betas"], dtype=np.float32),
        "scale": 1.0,
        "coordinate_frame": "scannet_camera",
    }
    for key in (
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
    ):
        if key in payload:
            converted[key] = np.asarray(payload[key], dtype=np.float32)

    target_dir = ensure_dir(
        genzi_eval_root(output_mode) / "_selected_inputs" / candidate.interaction_name
    )
    target_path = target_dir / "smplx_camera.pkl"
    with target_path.open("wb") as file_obj:
        pickle.dump(converted, file_obj)
    save_json(
        target_dir / "selection.json",
        {
            "interaction_name": candidate.interaction_name,
            "selected_index": candidate.index,
            "candidate_dir": str(candidate.candidate_dir),
            "human_mesh_world": str(candidate.human_mesh_world),
            "smplx_params_world": str(candidate.smplx_params_world),
            "smplx_params_camera": str(target_path),
            "coordinate_frame": "scannet_camera",
            "source_mesh_parameter_max_error_m": mesh_error,
        },
    )
    return target_path
