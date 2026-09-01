#!/usr/bin/env python3
"""Utilities shared by the PROX evaluators.

The evaluation scene and cameras always come from Module 06.  PROX contributes
only its optimized SMPL-X result.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_MODE = "output"
CONTACT_SEGMENT_BY_BODY_SEGMENT = {
    "left_hand": "left_hand_contact",
    "right_hand": "right_hand_contact",
    "left_arm": "left_arm_contact",
    "right_arm": "right_arm_contact",
    "left_leg": "left_leg_contact",
    "right_leg": "right_leg_contact",
    "left_foot": "left_foot_contact",
    "right_foot": "right_foot_contact",
    "head": "head_contact",
    "hips": "hips_contact",
    "back": "back_contact",
}
METRIC_CSV_FIELDNAMES = [
    "node_a",
    "node_b",
    "min_distance_m",
    "max_distance_m",
    "mean_distance_m",
    "ncs",
    "mean_penetration_m",
    "max_penetration_m",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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
    spec.loader.exec_module(module)
    return module


def prox_output_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / output_mode


def prox_interaction_root(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> Path:
    return prox_output_root(output_mode) / interaction_name


def prox_eval_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / "evaluation" / output_mode


def interaction_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 10**9, name


def discover_prox_interactions(output_mode: str = DEFAULT_OUTPUT_MODE) -> list[str]:
    root = prox_output_root(output_mode)
    names = [
        path.name
        for path in root.glob("interaction_*")
        if path.is_dir()
        and (path / "final_smplx_world.ply").is_file()
        and (path / "result.pkl").is_file()
    ]
    names.sort(key=interaction_sort_key)
    if not names:
        raise RuntimeError(f"No completed PROX interactions found under {root}.")
    return names


def normalize_label(text: str) -> str:
    return " ".join(
        text.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def load_smplx_segments() -> dict[str, Any]:
    path = (
        PROJECT_DIR
        / "04_Estimate_Human_Pose"
        / "assets"
        / "smplx_vert_segmentation.json"
    )
    raw = load_json(path)
    return {
        "vertex_count": int(raw["vertex_count"]),
        "segments": {
            str(key): np.unique(np.asarray(value, dtype=np.int64))
            for key, value in raw["segments"].items()
        },
    }


def contact_segment_id_for_part(human_part: str) -> str:
    segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(slugify_segment_name(human_part))
    if segment_id is None:
        raise KeyError(f"No SMPL-X contact segment mapping for '{human_part}'.")
    return segment_id


def load_sig_edges(interaction_name: str) -> list[dict[str, str]]:
    path = PROJECT_DIR / "01_Generate_SIG" / "output" / interaction_name / "sig.json"
    payload = load_json(path)
    parsed = []
    for edge in payload.get("interaction_edges", []):
        if not isinstance(edge, dict):
            continue
        human_part = normalize_label(str(edge.get("human_part", "")))
        scene_element = normalize_label(str(edge.get("scene_element", "")))
        if human_part and scene_element:
            parsed.append(
                {
                    "human_part": human_part,
                    "scene_element": scene_element,
                    "node_a": f"person 1, {human_part}",
                    "node_b": scene_element,
                }
            )
    if not parsed:
        raise RuntimeError(f"No usable interaction edges found in {path}")
    return parsed


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {path}, got {type(loaded)!r}")
    return loaded


def module06_render_config(interaction_name: str) -> tuple[Path, dict[str, Any]]:
    path = (
        PROJECT_DIR
        / "06_Evaluate_Interaction"
        / "output"
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing shared Module 06 render config: {path}")
    return path, load_json(path)


def shared_scene_camera(interaction_name: str) -> dict[str, Any]:
    """Load Module 06's scene crop and source camera, never PROX scene data."""
    config_path, config = module06_render_config(interaction_name)
    scene_path = Path(config["scene_crop_ply"]).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing shared Module 06 scene crop: {scene_path}")
    views = config.get("views", [])
    if not views:
        raise RuntimeError(f"Module 06 render config has no views: {config_path}")
    source_view = views[0]
    matrix_world_blender = np.asarray(
        source_view["camera_matrix_world"], dtype=np.float32
    )
    if matrix_world_blender.shape != (4, 4):
        raise ValueError("Module 06 source camera matrix must be 4x4.")
    # Blender camera coordinates are x-right, y-up, z-back. Convert the saved
    # camera-to-world rotation to OpenCV x-right, y-down, z-forward.
    cv_to_blender = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    rotation_camera_to_world = matrix_world_blender[:3, :3] @ cv_to_blender
    rotation_world_to_camera = rotation_camera_to_world.T
    camera_center_world = matrix_world_blender[:3, 3]
    translation_world_to_camera = -rotation_world_to_camera @ camera_center_world
    return {
        "render_config_path": config_path,
        "render_config": config,
        "scene_path": scene_path,
        "scene_world": load_mesh(scene_path),
        "intrinsics": np.asarray(source_view["intrinsics"], dtype=np.float32),
        "width": int(source_view["width"]),
        "height": int(source_view["height"]),
        "rotation_world_to_camera": rotation_world_to_camera,
        "translation_world_to_camera": translation_world_to_camera,
    }


def world_to_camera(vertices: np.ndarray, context: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(vertices, dtype=np.float32)
        @ np.asarray(context["rotation_world_to_camera"], dtype=np.float32).T
        + np.asarray(context["translation_world_to_camera"], dtype=np.float32)[None]
    )


def project_vertices(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = vertices[:, 2]
    valid = z > 1e-6
    uv = np.full((vertices.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid, 0] = intrinsics[0, 0] * vertices[valid, 0] / z[valid] + intrinsics[0, 2]
    uv[valid, 1] = intrinsics[1, 1] * vertices[valid, 1] / z[valid] + intrinsics[1, 2]
    return uv, valid


def mask_vertex_ids_for_part(
    interaction_name: str,
    human_part: str,
    scene_vertices: np.ndarray,
    intrinsics: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    mask_path = (
        PROJECT_DIR
        / "00_Annotate_GT_Contact"
        / "output"
        / interaction_name
        / "contact_masks_gt"
        / f"{slugify_segment_name(human_part)}.png"
    )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read contact mask: {mask_path}")
    if mask.shape != image_hw:
        mask = cv2.resize(mask, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_NEAREST)
    uv, valid = project_vertices(scene_vertices, intrinsics)
    xy = np.zeros_like(uv, dtype=np.int64)
    xy[valid] = np.rint(uv[valid]).astype(np.int64)
    in_bounds = (
        valid
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < image_hw[1])
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < image_hw[0])
    )
    ids = np.where(in_bounds)[0]
    selected = ids[mask[xy[ids, 1], xy[ids, 0]] > 127]
    return selected.astype(np.int64)


def nearest_distance_stats(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> dict[str, float]:
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        raise RuntimeError("Contact-distance inputs cannot be empty.")
    distances, _ = cKDTree(target_points).query(source_points, k=1, workers=-1)
    return {
        "min_distance_m": float(np.min(distances)),
        "max_distance_m": float(np.max(distances)),
        "mean_distance_m": float(np.mean(distances)),
    }


def sample_scene_points(
    scene_mesh: trimesh.Trimesh,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        points, _ = trimesh.sample.sample_surface(scene_mesh, num_samples)
    finally:
        np.random.set_state(state)
    return points.astype(np.float32)


def compute_mesh_penetration(
    scene_mesh: trimesh.Trimesh,
    human_mesh: trimesh.Trimesh,
    num_samples: int,
    seed: int,
) -> dict[str, float]:
    scene_points = sample_scene_points(scene_mesh, num_samples, seed)
    try:
        inside = human_mesh.contains(scene_points)
    except Exception:
        bounds = np.asarray(human_mesh.bounds)
        inside = np.all(
            (scene_points >= bounds[0][None]) & (scene_points <= bounds[1][None]),
            axis=1,
        )
    if np.any(inside):
        try:
            _, distances, _ = trimesh.proximity.closest_point(
                human_mesh, scene_points[inside]
            )
        except Exception:
            distances, _ = cKDTree(np.asarray(human_mesh.vertices)).query(
                scene_points[inside], k=1, workers=-1
            )
        penetration = np.asarray(distances, dtype=np.float64)
    else:
        penetration = np.asarray([], dtype=np.float64)
    count = int(scene_points.shape[0])
    inside_count = int(np.count_nonzero(inside))
    return {
        "ncs": float((count - inside_count) / count),
        "mean_penetration_m": float(np.mean(penetration)) if penetration.size else 0.0,
        "max_penetration_m": float(np.max(penetration)) if penetration.size else 0.0,
    }


def part_colors_from_contact_spec(
    interaction_name: str,
) -> dict[str, tuple[int, int, int]]:
    path = (
        PROJECT_DIR
        / "00_Annotate_GT_Contact"
        / "output"
        / interaction_name
        / "contact_spec.json"
    )
    payload = load_json(path)
    colors = {}
    for item in payload.get("palette", {}).get("parts", []):
        if isinstance(item, dict) and isinstance(item.get("rgb"), list):
            colors[slugify_segment_name(str(item.get("part", "")))] = tuple(
                int(value) for value in item["rgb"]
            )
    return colors


def write_contact_debug_scene(
    interaction_name: str,
    scene_mesh: trimesh.Trimesh,
    part_to_vertex_ids: dict[str, np.ndarray],
    output_dir: Path,
) -> tuple[Path, Path]:
    ensure_dir(output_dir)
    colors = np.tile(
        np.asarray([188, 188, 188, 255], dtype=np.uint8),
        (scene_mesh.vertices.shape[0], 1),
    )
    palette = part_colors_from_contact_spec(interaction_name)
    fallback = [(255, 0, 0), (0, 170, 0), (0, 80, 255), (255, 140, 0)]
    legend = []
    for index, (part, vertex_ids) in enumerate(part_to_vertex_ids.items()):
        rgb = palette.get(slugify_segment_name(part), fallback[index % len(fallback)])
        colors[vertex_ids, :3] = np.asarray(rgb, dtype=np.uint8)
        legend.append(
            {"human_part": part, "vertex_count": int(vertex_ids.size), "rgb": list(rgb)}
        )
    debug_mesh = scene_mesh.copy()
    debug_mesh.visual.vertex_colors = colors
    ply_path = output_dir / "projected_contact_scene.ply"
    json_path = output_dir / "projected_contact_scene.json"
    debug_mesh.export(ply_path)
    save_json(
        json_path,
        {
            "ply_path": str(ply_path),
            "coordinate_frame": "module06_source_camera",
            "scene_source": "Module 06 shared scene crop",
            "edges": legend,
        },
    )
    return ply_path, json_path


def physical_summary_row(
    interaction_name: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "interaction_name": interaction_name,
        "num_edges": len(rows),
        "mean_min_contact_distance_m": float(np.mean([row["min_distance_m"] for row in rows])),
        "mean_max_contact_distance_m": float(np.mean([row["max_distance_m"] for row in rows])),
        "mean_contact_distance_m": float(np.mean([row["mean_distance_m"] for row in rows])),
        "ncs": float(np.mean([row["ncs"] for row in rows])),
        "mean_penetration_m": float(np.mean([row["mean_penetration_m"] for row in rows])),
        "max_penetration_m": float(np.mean([row["max_penetration_m"] for row in rows])),
    }


def aggregate_physical_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"interaction_name": "__mean__", "num_edges": sum(row["num_edges"] for row in rows)}
    for key in (
        "mean_min_contact_distance_m",
        "mean_max_contact_distance_m",
        "mean_contact_distance_m",
        "ncs",
        "mean_penetration_m",
        "max_penetration_m",
    ):
        result[key] = float(np.mean([float(row[key]) for row in rows]))
    return result
