from __future__ import annotations

import argparse
import base64
import csv
import json
import pickle
import shutil
import zipfile
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCANNET_ROOT = PROJECT_DIR.parent / "Scannet++" / "data"
DEFAULT_SMPL_SEG_JSON = PROJECT_DIR / "05_Estimate_Human_Pose" / "assets" / "smplx_vert_segmentation.json"


CONTACT_PROMPT = SCRIPT_DIR / "system_prompt_contact.md"
POSE_PROMPT = SCRIPT_DIR / "system_prompt_pose.md"
PENETRATION_PROMPT = SCRIPT_DIR / "system_prompt_penetration.md"


def log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_label(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").replace("-", " ").split())


def slugify(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one completed 4DHSI optimized static scene with "
            "deterministic metrics and optional VLM crop checks."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--optimizer-output-root", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--input-scene-json", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--contact-camera-json", default=None)
    parser.add_argument("--contact-canvas-image", default=None)
    parser.add_argument("--scannet-root", default=str(DEFAULT_SCANNET_ROOT))
    parser.add_argument("--smpl-seg-json", default=str(DEFAULT_SMPL_SEG_JSON))
    parser.add_argument("--contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--severe-penetration-min-sdf-m", type=float, default=-0.03)
    parser.add_argument("--severe-penetration-inside-points", type=int, default=1000)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama-api-key", default="ollama")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--vlm-thinking-effort", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--render-image-size", default="720x720")
    parser.add_argument("--render-device", default="cuda:0")
    parser.add_argument("--crop-padding-frac", type=float, default=1.75)
    parser.add_argument("--crop-min-size-px", type=int, default=320)
    parser.add_argument("--context-padding-frac", type=float, default=0.35)
    parser.add_argument("--contact-view-count", type=int, default=4)
    parser.add_argument("--contact-context-output-size", default="720x720")
    parser.add_argument("--contact-local-output-size", default="720x720")
    parser.add_argument("--contact-context-fill-frac", type=float, default=0.94)
    parser.add_argument("--contact-local-fill-frac", type=float, default=0.94)
    parser.add_argument("--view-planner-image-size", default="720x720")
    parser.add_argument("--candidate-yaws-deg", default="-60,-30,0,30,60,120,-120,150,-150")
    parser.add_argument("--candidate-elevations-deg", default="-10,0,10")
    parser.add_argument("--candidate-radius-scales", default="0.7, 0.6, 0.5")
    parser.add_argument("--min-human-visible-over-total-ratio", type=float, default=0.50)
    parser.add_argument("--min-target-visible-over-total-ratio", type=float, default=0.75)
    parser.add_argument("--max-target-self-occluded-ratio", type=float, default=0.10)
    parser.add_argument("--min-view-angular-separation-deg", type=float, default=15.0)
    parser.add_argument("--occlusion-depth-eps-m", type=float, default=0.01)
    return parser.parse_args()


def default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "optimizer_output_root": PROJECT_DIR /
        "06_Optimize_Static_Scene" /
        "output" /
        interaction_name,
        "sig_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "output" /
        interaction_name /
        "scene_interaction_graph.json",
        "input_scene_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "input_prompts" /
        interaction_name /
        "input_scene.json",
        "outdir": SCRIPT_DIR /
        "output" /
        interaction_name,
        "contact_camera_json": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "contact_camera.json",
        "contact_canvas_image": PROJECT_DIR /
        "04_Estimate_Contact" /
        "output" /
        interaction_name /
        "prompt" /
        "target_scene_crop.png",
    }


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return Path(raw_path).resolve() if raw_path else default_path.resolve()


def parse_size(text: str) -> tuple[int, int]:
    normalized = str(text).lower().replace(" ", "")
    if "x" not in normalized:
        raise ValueError(f"Expected size formatted like 720x720, got {text!r}")
    width_text, height_text = normalized.split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"Size must be positive, got {text!r}")
    return width, height


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def rank_failure_tags(tags: list[str]) -> list[str]:
    priority = [
        "missing_contact",
        "severe_penetration",
        "implausible_pose",
        "wrong_interaction",
        "wrong_target",
        "needs_review",
    ]
    normalized = [slugify(tag) for tag in tags if tag]
    return sorted(
        set(normalized),
        key=lambda tag: (priority.index(tag) if tag in priority else 999, tag),
    )


def read_final_loss_summary(csv_path: Path) -> dict[str, Any]:
    if not csv_path.exists():
        return {"available": False, "path": str(csv_path)}
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    if not rows:
        return {"available": False, "path": str(csv_path), "error": "empty CSV"}

    parsed: dict[str, Any] = {"available": True, "path": str(csv_path)}
    for key, value in rows[-1].items():
        if value == "":
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value) if "." in value or "e" in value.lower() else int(value)
        except ValueError:
            parsed[key] = value
    return parsed


def load_optimized_params(params_path: Path) -> dict[str, Any]:
    if not params_path.exists():
        return {"available": False, "path": str(params_path)}

    try:
        if zipfile.is_zipfile(params_path):
            with zipfile.ZipFile(params_path) as archive:
                data_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
                payload = pickle.loads(archive.read(data_name))
            if isinstance(payload, dict):
                return {
                    "available": True,
                    "path": str(params_path),
                    "height_m": payload.get("height_m"),
                    "scale": payload.get("scale"),
                    "canonical_height_unscaled_m": payload.get("canonical_height_unscaled_m"),
                    "loader": "zip_pickle",
                }
    except Exception:
        pass

    try:
        import torch
    except Exception as error:
        return {
            "available": False,
            "path": str(params_path),
            "error": f"Could not import torch to read optimized params: {error}",
        }

    try:
        payload = torch.load(params_path, map_location="cpu", weights_only=False)
    except Exception as error:
        return {
            "available": False,
            "path": str(params_path),
            "error": f"Could not read optimized params: {error}",
        }
    if not isinstance(payload, dict):
        return {"available": False, "path": str(params_path), "error": "optimized params payload is not a dict"}
    return {
        "available": True,
        "path": str(params_path),
        "height_m": payload.get("height_m"),
        "scale": payload.get("scale"),
        "canonical_height_unscaled_m": payload.get("canonical_height_unscaled_m"),
        "loader": "torch",
    }


def collect_contact_metrics(alignment_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_edges = (
        alignment_summary.get("human", {})
        .get("final_frame_0", {})
        .get("interaction_edges", [])
    )
    if not isinstance(final_edges, list):
        raise ValueError("alignment_summary human.final_frame_0.interaction_edges must be a list")

    threshold_m = float(args.contact_threshold_m)
    edges: list[dict[str, Any]] = []
    distances: list[float] = []
    failure_tags: list[str] = []

    for index, edge in enumerate(final_edges):
        if not isinstance(edge, dict):
            continue
        distance = edge.get("nocontact_distance_m")
        distance_m = float(distance) if distance is not None else None
        passed = distance_m is not None and distance_m <= threshold_m
        if not passed:
            failure_tags.append("missing_contact")
        if distance_m is not None:
            distances.append(distance_m)
        edges.append(
            {
                "index": int(index),
                "moving_part_name": edge.get("moving_part_name"),
                "moving_segment_id": edge.get("moving_segment_id"),
                "fixed_part_name": edge.get("fixed_part_name"),
                "fixed_entity_name": edge.get("fixed_entity_name"),
                "fixed_point_count": edge.get("fixed_point_count"),
                "moving_vertex_count": edge.get("moving_vertex_count"),
                "reduction": edge.get("reduction"),
                "nocontact_raw": edge.get("nocontact_raw"),
                "nocontact_distance_m": distance_m,
                "threshold_m": threshold_m,
                "pass": bool(passed),
            }
        )

    return {
        "pass": bool(edges) and all(bool(edge["pass"]) for edge in edges),
        "failure_tags": rank_failure_tags(failure_tags),
        "edge_count": int(len(edges)),
        "edges": edges,
        "mean_distance_m": float(sum(distances) / len(distances)) if distances else None,
        "max_distance_m": float(max(distances)) if distances else None,
    }


def collect_penetration_metrics(alignment_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    final_debug = (
        alignment_summary.get("human", {})
        .get("optimization", {})
        .get("scene_intersect_debug", {})
        .get("final", {})
    )
    scene_points = final_debug.get("scene_points", {}) if isinstance(final_debug, dict) else {}
    min_sdf = scene_points.get("min_sdf_m")
    inside_points = scene_points.get("num_inside_points")
    min_sdf_f = float(min_sdf) if min_sdf is not None else None
    inside_i = int(inside_points) if inside_points is not None else None

    severe_by_sdf = min_sdf_f is not None and min_sdf_f < float(args.severe_penetration_min_sdf_m)
    severe_by_count = inside_i is not None and inside_i > int(args.severe_penetration_inside_points)
    severe = bool(severe_by_sdf or severe_by_count)

    return {
        "pass": not severe,
        "failure_tags": ["severe_penetration"] if severe else [],
        "severe_penetration": severe,
        "severe_by_min_sdf": bool(severe_by_sdf),
        "severe_by_inside_points": bool(severe_by_count),
        "thresholds": {
            "min_sdf_m": float(args.severe_penetration_min_sdf_m),
            "inside_points": int(args.severe_penetration_inside_points),
        },
        "scene_points": scene_points,
        "sdf_grid": final_debug.get("sdf_grid", {}) if isinstance(final_debug, dict) else {},
    }


def collect_metrics(
    sig_payload: dict[str, Any],
    alignment_summary: dict[str, Any],
    optimizer_root: Path,
    sig_json_path: Path,
    input_scene_json_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contact = collect_contact_metrics(alignment_summary, args)
    penetration = collect_penetration_metrics(alignment_summary, args)
    optimization = alignment_summary.get("human", {}).get("optimization", {})
    optimizer_config = alignment_summary.get("optimizer", {})
    target = alignment_summary.get("target_object", {})
    if not isinstance(target, dict):
        target = {}

    failure_tags: list[str] = []
    failure_tags.extend(contact.get("failure_tags", []))
    failure_tags.extend(penetration.get("failure_tags", []))
    ranked_tags = rank_failure_tags(failure_tags)

    return {
        "interaction_name": alignment_summary.get("interaction_name", args.interaction_name),
        "scene_id": alignment_summary.get("scene_id"),
        "target_object": target,
        "interaction": sig_payload.get("interaction", ""),
        "source_paths": {
            "sig_json": str(sig_json_path),
            "input_scene_json": str(input_scene_json_path),
            "optimizer_output_root": str(optimizer_root),
            "alignment_summary": str(optimizer_root / "alignment_summary.json"),
            "world_mesh": str(optimizer_root / "meshes" / "frame_0000_world.ply"),
        },
        "thresholds": {
            "contact_threshold_m": float(args.contact_threshold_m),
            "severe_penetration_min_sdf_m": float(args.severe_penetration_min_sdf_m),
            "severe_penetration_inside_points": int(args.severe_penetration_inside_points),
        },
        "contact": contact,
        "penetration": penetration,
        "optimization": {
            "final_total_loss": optimization.get("final_total_loss"),
            "final_iter": optimization.get("final_iter"),
            "stage_iters": optimization.get("stage_iters"),
            "scene_intersect_stats": optimization.get("scene_intersect_stats"),
            "final_loss_summary": read_final_loss_summary(optimizer_root / "debug" / "csv" / "final_loss_summary.csv"),
        },
        "height": {
            "optimized_params": load_optimized_params(optimizer_root / "debug" / "params" / "optimized_frame_0000.pt"),
            "height_prior": optimizer_config.get("height_prior"),
        },
        "deterministic": {
            "pass": not bool(ranked_tags),
            "failure_tags": ranked_tags,
        },
    }


def build_verification_summary(metrics: dict[str, Any], vlm_judgments: dict[str, Any] | None = None) -> dict[str, Any]:
    failure_tags = list(metrics.get("deterministic", {}).get("failure_tags", []))
    vlm_enabled = bool(vlm_judgments and vlm_judgments.get("enabled"))
    vlm_missing = False

    if vlm_enabled and vlm_judgments:
        for judgment in vlm_judgments.get("contact_edges", []):
            verdict = judgment.get("pass")
            if verdict is False:
                failure_tags.append("missing_contact")
            elif verdict is None:
                vlm_missing = True
                failure_tags.append("needs_review")

        pose = vlm_judgments.get("pose")
        if isinstance(pose, dict):
            verdict = pose.get("pass")
            if verdict is False:
                failure_tags.append("implausible_pose")
            elif verdict is None:
                vlm_missing = True
                failure_tags.append("needs_review")

        penetration = vlm_judgments.get("penetration")
        if isinstance(penetration, dict):
            verdict = penetration.get("pass")
            if verdict is False:
                failure_tags.append("severe_penetration")
            elif verdict is None:
                vlm_missing = True
                failure_tags.append("needs_review")

    ranked_tags = rank_failure_tags(failure_tags)
    if metrics.get("deterministic", {}).get("failure_tags"):
        status = "fail"
    elif vlm_enabled and any(tag != "needs_review" for tag in ranked_tags):
        status = "fail"
    elif vlm_enabled and vlm_missing:
        status = "needs_review"
    else:
        status = "pass"

    return {
        "interaction_name": metrics.get("interaction_name"),
        "scene_id": metrics.get("scene_id"),
        "target_object": metrics.get("target_object"),
        "status": status,
        "failure_tags": ranked_tags,
        "deterministic_pass": metrics.get("deterministic", {}).get("pass"),
        "vlm_enabled": vlm_enabled,
        "contact_pass": metrics.get("contact", {}).get("pass"),
        "penetration_pass": metrics.get("penetration", {}).get("pass"),
        "max_contact_distance_m": metrics.get("contact", {}).get("max_distance_m"),
        "mean_contact_distance_m": metrics.get("contact", {}).get("mean_distance_m"),
        "severe_penetration": metrics.get("penetration", {}).get("severe_penetration"),
        "final_total_loss": metrics.get("optimization", {}).get("final_total_loss"),
    }


def validate_required_inputs(optimizer_root: Path, sig_json_path: Path, input_scene_json_path: Path) -> None:
    required = [
        sig_json_path,
        input_scene_json_path,
        optimizer_root / "alignment_summary.json",
        optimizer_root / "meshes" / "frame_0000_world.ply",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required evaluation input(s): " + ", ".join(missing))


def import_render_deps() -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import trimesh
        from PIL import Image
        from pytorch3d.ops import interpolate_face_attributes
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        from pytorch3d.utils import cameras_from_opencv_projection
    except Exception as error:
        raise RuntimeError(
            "Render dependencies are unavailable. Run this inside the 4dhsi environment, e.g. "
            "`conda run -n 4dhsi python 07_Verify_Static_Scene/01_evaluate_static_scene.py ...`. "
            f"Original import error: {error}"
        ) from error
    return {
        "np": np,
        "torch": torch,
        "trimesh": trimesh,
        "Image": Image,
        "interpolate_face_attributes": interpolate_face_attributes,
        "MeshRasterizer": MeshRasterizer,
        "RasterizationSettings": RasterizationSettings,
        "cameras_from_opencv_projection": cameras_from_opencv_projection,
        "Meshes": Meshes,
    }


def colmap_qvec_to_rotmat(qvec: list[float]) -> Any:
    import numpy as np

    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def read_colmap_w2c(images_txt: Path, image_name: str) -> Any:
    import numpy as np

    if not images_txt.exists():
        raise FileNotFoundError(f"COLMAP images.txt not found: {images_txt}")
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 10 and parts[9] == image_name:
            rotation = colmap_qvec_to_rotmat([float(value) for value in parts[1:5]])
            translation = np.asarray([float(value) for value in parts[5:8]], dtype=np.float32)
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = rotation
            w2c[:3, 3] = translation
            return w2c
    raise FileNotFoundError(f"Camera pose for {image_name} not found in {images_txt}")


def read_camera(input_scene: dict[str, Any], scannet_root: Path) -> dict[str, Any]:
    scene_context = input_scene.get("scene_context", {})
    scene_id = scene_context.get("scene_id")
    camera = scene_context.get("camera", {})
    image_name = camera.get("name")
    source = camera.get("source", "dslr_resized_undistorted")
    if not scene_id or not image_name:
        raise ValueError("input_scene_json must contain scene_context.scene_id and scene_context.camera.name")
    if source != "dslr_resized_undistorted":
        raise ValueError(f"Only dslr_resized_undistorted camera source is supported for Module 07 v1; got {source!r}")

    scene_dir = scannet_root / str(scene_id)
    transforms_path = scene_dir / "dslr" / "nerfstudio" / "transforms_undistorted.json"
    colmap_images_path = scene_dir / "dslr" / "colmap" / "images.txt"
    image_path = scene_dir / "dslr" / "resized_undistorted_images" / str(image_name)
    scene_mesh_path = scene_dir / "scans" / "mesh_aligned_0.05.ply"
    transforms = load_json(transforms_path)
    frames = transforms.get("frames", []) + transforms.get("test_frames", [])
    frame = next((item for item in frames if Path(str(item.get("file_path", ""))).name == image_name), None)
    if frame is None:
        raise FileNotFoundError(f"Camera frame {image_name} not found in {transforms_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Original camera image not found: {image_path}")
    if not scene_mesh_path.exists():
        raise FileNotFoundError(f"ScanNet++ scene mesh not found: {scene_mesh_path}")

    return {
        "scene_id": scene_id,
        "image_name": image_name,
        "image_path": image_path,
        "scene_mesh_path": scene_mesh_path,
        "w2c": read_colmap_w2c(colmap_images_path, str(image_name)),
        "width": int(transforms["w"]),
        "height": int(transforms["h"]),
        "fx": float(transforms["fl_x"]),
        "fy": float(transforms["fl_y"]),
        "cx": float(transforms["cx"]),
        "cy": float(transforms["cy"]),
        "transforms_path": transforms_path,
        "colmap_images_path": colmap_images_path,
    }


def read_contact_crop_camera(
    base_camera: dict[str, Any],
    contact_camera_json: Path,
    contact_canvas_image: Path,
    Image: Any,
) -> dict[str, Any]:
    if not contact_camera_json.exists():
        raise FileNotFoundError(f"Contact camera JSON not found: {contact_camera_json}")
    if not contact_canvas_image.exists():
        raise FileNotFoundError(f"Contact canvas image not found: {contact_canvas_image}")

    payload = load_json(contact_camera_json)
    intrinsics = payload.get("intrinsics_3x3")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3:
        raise ValueError(f"Expected intrinsics_3x3 in {contact_camera_json}")

    with Image.open(contact_canvas_image) as image:
        width, height = image.size

    crop_camera = dict(base_camera)
    crop_camera.update(
        {
            "width": int(width),
            "height": int(height),
            "fx": float(intrinsics[0][0]),
            "fy": float(intrinsics[1][1]),
            "cx": float(intrinsics[0][2]),
            "cy": float(intrinsics[1][2]),
            "contact_camera_json": contact_camera_json,
            "contact_canvas_image": contact_canvas_image,
        }
    )
    return crop_camera


def choose_device(torch: Any, requested: str) -> Any:
    device = torch.device(str(requested))
    if device.type != "cuda":
        raise RuntimeError("--render-device must be a CUDA device like cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError(f"--render-device {requested} was requested, but CUDA is not available")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError(
            f"--render-device {requested} was requested, but only "
            f"{torch.cuda.device_count()} CUDA device(s) are available"
        )
    return device


def as_vertex_colors(mesh: Any, np: Any, color: tuple[float, float, float] | None = None) -> Any:
    vertex_count = len(mesh.vertices)
    if color is not None:
        return np.tile(np.asarray(color, dtype=np.float32), (vertex_count, 1))
    visual = getattr(mesh, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is None or len(vertex_colors) != vertex_count:
        return np.tile(np.asarray([0.62, 0.62, 0.62], dtype=np.float32), (vertex_count, 1))
    colors = np.asarray(vertex_colors[:, :3], dtype=np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return colors


def filter_scene_faces_to_contact_camera(
    vertices_world: Any,
    faces: Any,
    camera: dict[str, Any],
    np: Any,
    max_depth_m: float = 20.0,
    border_px: float = 96.0,
) -> Any:
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    camera_points = vertices_world @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    triangles = camera_points[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1)
    if max_depth_m is not None:
        positive &= np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return faces[:0].copy()

    z_safe = np.clip(z, 1e-6, None)
    u = float(camera["fx"]) * triangles[..., 0] / z_safe + float(camera["cx"]) - 0.5
    v = float(camera["fy"]) * triangles[..., 1] / z_safe + float(camera["cy"]) - 0.5
    u_min = np.min(u, axis=1)
    u_max = np.max(u, axis=1)
    v_min = np.min(v, axis=1)
    v_max = np.max(v, axis=1)
    overlaps = (
        positive
        & (u_max >= -float(border_px))
        & (u_min <= float(int(camera["width"]) - 1) + float(border_px))
        & (v_max >= -float(border_px))
        & (v_min <= float(int(camera["height"]) - 1) + float(border_px))
    )
    return faces[overlaps].astype(np.int64)


def compact_scene_crop(
    vertices: Any,
    faces: Any,
    colors: Any,
    np: Any,
) -> tuple[Any, Any, Any, Any]:
    if faces.shape[0] == 0:
        raise RuntimeError("No ScanNet scene faces remained after contact-camera crop filtering.")
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    crop_vertices = vertices[unique_vids].astype(np.float32)
    crop_faces = inverse.reshape(-1, 3).astype(np.int64)
    crop_colors = colors[unique_vids].astype(np.float32)
    return crop_vertices, crop_faces, crop_colors, unique_vids.astype(np.int64)


def shaded_human_vertex_colors(mesh: Any, np: Any, camera: dict[str, Any]) -> Any:
    base = np.asarray([0.78, 0.76, 0.72], dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    camera_forward_world = np.asarray(camera["w2c"], dtype=np.float32)[
        :3, :3].T @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    light_dir = -camera_forward_world
    light_dir = light_dir / max(float(np.linalg.norm(light_dir)), 1e-6)
    diffuse = np.maximum(normals @ light_dir, 0.0)
    fill = np.maximum(normals @ np.asarray([0.2, -0.4, 0.9], dtype=np.float32), 0.0)
    shade = 0.42 + 0.46 * diffuse + 0.18 * fill
    return np.clip(base[None, :] * shade[:, None], 0.0, 1.0)


def highlight_vertex_colors(vertex_colors: Any, vertex_ids: list[int], np: Any) -> Any:
    highlighted = np.asarray(vertex_colors, dtype=np.float32).copy()
    if vertex_ids:
        highlighted[np.asarray(vertex_ids, dtype=np.int64)] = np.asarray([1.0, 0.04, 0.02], dtype=np.float32)
    return highlighted


def filter_faces_for_camera(
        vertices: Any,
        faces: Any,
        w2c: Any,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        np: Any) -> Any:
    camera_points = vertices @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    z = camera_points[:, 2]
    valid_z = z > 0.02
    u = fx * (camera_points[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (camera_points[:, 1] / np.maximum(z, 1e-6)) + cy
    margin = 128
    in_view = valid_z & (u >= -margin) & (u <= width + margin) & (v >= -margin) & (v <= height + margin)
    face_mask = in_view[faces].any(axis=1) & valid_z[faces].any(axis=1)
    if face_mask.sum() < 10:
        return faces
    return faces[face_mask]


def rasterize_colored_mesh(
    vertices: Any,
    faces: Any,
    vertex_colors: Any,
    camera: dict[str, Any],
    image_size: tuple[int, int],
    deps: dict[str, Any],
    device: Any,
) -> tuple[Any, Any]:
    np = deps["np"]
    torch = deps["torch"]
    Meshes = deps["Meshes"]
    MeshRasterizer = deps["MeshRasterizer"]
    RasterizationSettings = deps["RasterizationSettings"]
    cameras_from_opencv_projection = deps["cameras_from_opencv_projection"]
    interpolate_face_attributes = deps["interpolate_face_attributes"]

    height, width = image_size
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    scale_x = width / float(camera["width"])
    scale_y = height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y

    faces = filter_faces_for_camera(vertices, faces, w2c, fx, fy, cx, cy, width, height, np)
    if len(faces) == 0:
        image = np.ones((height, width, 3), dtype=np.float32)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        return image, depth
    used_vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used_vertex_ids].astype(np.float32)
    vertex_colors = vertex_colors[used_vertex_ids].astype(np.float32)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    verts_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64, device=device)
    colors_t = torch.as_tensor(vertex_colors, dtype=torch.float32, device=device).clamp(0.0, 1.0)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    R = torch.as_tensor(w2c[:3, :3][None], dtype=torch.float32, device=device)
    T = torch.as_tensor(w2c[:3, 3][None], dtype=torch.float32, device=device)
    K = torch.as_tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    image_size_t = torch.as_tensor([[height, width]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size_t)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    fragments = rasterizer(mesh)
    face_attrs = colors_t[faces_t]
    pix_colors = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, face_attrs)[0, :, :, 0, :]
    mask = fragments.pix_to_face[0, :, :, 0] >= 0
    depth = fragments.zbuf[0, :, :, 0]
    image = torch.ones((height, width, 3), dtype=torch.float32, device=device)
    image[mask] = pix_colors[mask]
    depth_np = depth.detach().cpu().numpy()
    depth_np[~mask.detach().cpu().numpy()] = np.inf
    return image.detach().cpu().numpy(), depth_np


def rasterize_depth(
    vertices: Any,
    faces: Any,
    camera: dict[str, Any],
    image_size: tuple[int, int],
    deps: dict[str, Any],
    device: Any,
) -> Any:
    np = deps["np"]
    torch = deps["torch"]
    Meshes = deps["Meshes"]
    MeshRasterizer = deps["MeshRasterizer"]
    RasterizationSettings = deps["RasterizationSettings"]
    cameras_from_opencv_projection = deps["cameras_from_opencv_projection"]

    height, width = image_size
    if len(faces) == 0:
        return np.full((height, width), np.inf, dtype=np.float32)
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    scale_x = width / float(camera["width"])
    scale_y = height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y

    faces = filter_faces_for_camera(vertices, faces, w2c, fx, fy, cx, cy, width, height, np)
    if len(faces) == 0:
        return np.full((height, width), np.inf, dtype=np.float32)
    used_vertex_ids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    vertices = vertices[used_vertex_ids].astype(np.float32)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    verts_t = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces_t = torch.as_tensor(faces.astype(np.int64), dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts_t], faces=[faces_t])

    R = torch.as_tensor(w2c[:3, :3][None], dtype=torch.float32, device=device)
    T = torch.as_tensor(w2c[:3, 3][None], dtype=torch.float32, device=device)
    K = torch.as_tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)
    image_size_t = torch.as_tensor([[height, width]], dtype=torch.float32, device=device)
    cameras = cameras_from_opencv_projection(R=R, tvec=T, camera_matrix=K, image_size=image_size_t)
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
        ),
    )
    fragments = rasterizer(mesh)
    mask = fragments.pix_to_face[0, :, :, 0] >= 0
    depth = fragments.zbuf[0, :, :, 0]
    depth_np = depth.detach().cpu().numpy()
    depth_np[~mask.detach().cpu().numpy()] = np.inf
    return depth_np


def composite_scene_and_human(render_assets: dict[str, Any],
                              camera: dict[str, Any], image_size: tuple[int, int]) -> Any:
    np = render_assets["deps"]["np"]
    scene_rgb, scene_depth = rasterize_colored_mesh(
        vertices=render_assets["scene_vertices"],
        faces=render_assets["scene_faces"],
        vertex_colors=render_assets["scene_colors"],
        camera=camera,
        image_size=image_size,
        deps=render_assets["deps"],
        device=render_assets["device"],
    )
    human_rgb, human_depth = rasterize_colored_mesh(
        vertices=render_assets["human_vertices"],
        faces=render_assets["human_faces"],
        vertex_colors=render_assets["human_colors"],
        camera=camera,
        image_size=image_size,
        deps=render_assets["deps"],
        device=render_assets["device"],
    )
    composite = scene_rgb.copy()
    human_mask = np.isfinite(human_depth) & (human_depth < scene_depth)
    composite[human_mask] = human_rgb[human_mask]
    return (np.clip(composite, 0.0, 1.0) * 255).astype(np.uint8)


def render_scene_with_human(
    camera: dict[str, Any],
    contact_crop_camera: dict[str, Any],
    human_mesh_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deps = import_render_deps()
    np = deps["np"]
    torch = deps["torch"]
    trimesh = deps["trimesh"]
    device = choose_device(torch, args.render_device)

    scene_mesh = trimesh.load(camera["scene_mesh_path"], process=False, force="mesh")
    human_mesh = trimesh.load(human_mesh_path, process=False, force="mesh")
    full_scene_vertices = np.asarray(scene_mesh.vertices, dtype=np.float32)
    full_scene_faces = np.asarray(scene_mesh.faces, dtype=np.int64)
    full_scene_colors = as_vertex_colors(scene_mesh, np)
    crop_faces_world_ids = filter_scene_faces_to_contact_camera(
        full_scene_vertices,
        full_scene_faces,
        contact_crop_camera,
        np,
    )
    scene_vertices, scene_faces, scene_colors, scene_vertex_source_ids = compact_scene_crop(
        full_scene_vertices,
        crop_faces_world_ids,
        full_scene_colors,
        np,
    )
    log(
        "evidence",
        "rebuilt ScanNet contact crop "
        f"faces={scene_faces.shape[0]}/{full_scene_faces.shape[0]} "
        f"verts={scene_vertices.shape[0]}/{full_scene_vertices.shape[0]}",
    )
    render_w, render_h = parse_size(args.render_image_size)
    image_size = (render_h, render_w)

    render_assets = {
        "scene_vertices": scene_vertices,
        "scene_faces": scene_faces,
        "scene_colors": scene_colors,
        "scene_source_vertex_ids": scene_vertex_source_ids,
        "scene_crop": {
            "mode": "rebuilt_from_scannet_contact_camera_frustum",
            "source_scene_mesh": str(camera["scene_mesh_path"]),
            "contact_camera_json": str(contact_crop_camera["contact_camera_json"]),
            "contact_canvas_image": str(contact_crop_camera["contact_canvas_image"]),
            "full_scene_vertex_count": int(full_scene_vertices.shape[0]),
            "full_scene_face_count": int(full_scene_faces.shape[0]),
            "crop_vertex_count": int(scene_vertices.shape[0]),
            "crop_face_count": int(scene_faces.shape[0]),
        },
        "human_vertices": np.asarray(human_mesh.vertices, dtype=np.float32),
        "human_faces": np.asarray(human_mesh.faces, dtype=np.int64),
        "human_colors": shaded_human_vertex_colors(human_mesh, np, camera),
        "deps": deps,
        "device": device,
    }

    log("evidence", f"rendering original camera view at {render_w}x{render_h} on {device}")
    return {
        **render_assets,
        "image": composite_scene_and_human(render_assets, camera, image_size),
        "human_vertices": np.asarray(human_mesh.vertices, dtype=np.float32),
        "camera": camera,
        "render_width": render_w,
        "render_height": render_h,
        "deps": deps,
    }


def project_vertices(points_world: Any, camera: dict[str, Any], render_width: int, render_height: int, np: Any) -> Any:
    w2c = np.asarray(camera["w2c"], dtype=np.float32)
    points_camera = points_world @ w2c[:3, :3].T + w2c[:3, 3][None, :]
    z = points_camera[:, 2]
    scale_x = render_width / float(camera["width"])
    scale_y = render_height / float(camera["height"])
    fx = float(camera["fx"]) * scale_x
    fy = float(camera["fy"]) * scale_y
    cx = float(camera["cx"]) * scale_x
    cy = float(camera["cy"]) * scale_y
    u = fx * (points_camera[:, 0] / np.maximum(z, 1e-6)) + cx
    v = fy * (points_camera[:, 1] / np.maximum(z, 1e-6)) + cy
    return np.stack([u, v, z], axis=1)


def camera_center_from_w2c(w2c: Any, np: Any) -> Any:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return -rotation.T @ translation


def look_at_w2c(eye: Any, target: Any, np: Any) -> Any:
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    forward = np.asarray(target - eye, dtype=np.float32)
    forward = forward / np.maximum(np.linalg.norm(forward), 1e-6)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up)
    right = right / np.maximum(np.linalg.norm(right), 1e-6)
    down = np.cross(forward, right)
    down = down / np.maximum(np.linalg.norm(down), 1e-6)
    rotation = np.stack([right, down, forward], axis=0).astype(np.float32)
    translation = (-rotation @ eye.astype(np.float32)).astype(np.float32)
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = rotation
    w2c[:3, 3] = translation
    return w2c


def rotate_xy(vector: Any, yaw_deg: float, np: Any) -> Any:
    radians = np.deg2rad(float(yaw_deg))
    cos_v = np.cos(radians)
    sin_v = np.sin(radians)
    rotated = np.asarray(vector, dtype=np.float32).copy()
    x = rotated[0] * cos_v - rotated[1] * sin_v
    y = rotated[0] * sin_v + rotated[1] * cos_v
    rotated[0] = x
    rotated[1] = y
    return rotated


def apply_elevation_offset(offset: Any, elevation_deg: float, np: Any) -> Any:
    adjusted = np.asarray(offset, dtype=np.float32).copy()
    horizontal = float(np.linalg.norm(adjusted[:2]))
    radius = float(np.linalg.norm(adjusted))
    if horizontal < 1e-6 or radius < 1e-6:
        return adjusted
    current = float(np.arctan2(adjusted[2], horizontal))
    target_elevation = current + float(np.deg2rad(elevation_deg))
    new_horizontal = radius * float(np.cos(target_elevation))
    adjusted[:2] = adjusted[:2] / horizontal * new_horizontal
    adjusted[2] = radius * float(np.sin(target_elevation))
    return adjusted


def orbit_camera(camera: dict[str, Any], target: Any, yaw_deg: float,
                 radius_scale: float, elevation_deg: float, np: Any) -> dict[str, Any]:
    base_w2c = np.asarray(camera["w2c"], dtype=np.float32)
    base_eye = camera_center_from_w2c(base_w2c, np)
    offset = base_eye - target
    rotated_offset = rotate_xy(offset, yaw_deg, np)
    rotated_offset = rotated_offset * float(radius_scale)
    rotated_offset = apply_elevation_offset(rotated_offset, elevation_deg, np)
    eye = target + rotated_offset
    orbit = dict(camera)
    orbit["w2c"] = look_at_w2c(eye, target, np)
    orbit["orbit_yaw_deg"] = float(yaw_deg)
    orbit["orbit_elevation_deg"] = float(elevation_deg)
    orbit["orbit_radius_scale"] = float(radius_scale)
    orbit["orbit_pivot_world"] = [float(value) for value in target.tolist()]
    return orbit


def target_faces_for_segment(human_faces: Any, segment_vertex_ids: list[int], np: Any) -> Any:
    if not segment_vertex_ids:
        return human_faces[:0]
    segment_mask = np.zeros(int(human_faces.max()) + 1, dtype=bool)
    segment_mask[np.asarray(segment_vertex_ids, dtype=np.int64)] = True
    face_mask = segment_mask[human_faces].any(axis=1)
    return human_faces[face_mask]


def visible_surface_stats(
    candidate_depth: Any,
    occluder_depth: Any,
    np: Any,
    eps_m: float,
) -> dict[str, Any]:
    projected_mask = np.isfinite(candidate_depth)
    visible_mask = projected_mask & (
        ~np.isfinite(occluder_depth)
        | (candidate_depth <= occluder_depth + float(eps_m))
    )
    projected_pixels = int(projected_mask.sum())
    visible_pixels = int(visible_mask.sum())
    return {
        "visible_pixel_count": visible_pixels,
        "projected_pixel_count": projected_pixels,
        "visible_over_projected_surface_ratio": (
            float(visible_pixels / projected_pixels) if projected_pixels else 0.0
        ),
    }


def add_surface_visibility_ratios(
    metrics: dict[str, Any],
    rendered: dict[str, Any],
    camera: dict[str, Any],
    image_size: tuple[int, int],
    target_faces: Any,
    eps_m: float,
) -> dict[str, Any]:
    np = rendered["deps"]["np"]
    scene_depth = rasterize_depth(
        rendered["scene_vertices"],
        rendered["scene_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_depth = rasterize_depth(
        rendered["human_vertices"],
        rendered["human_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    target_depth = rasterize_depth(
        rendered["human_vertices"],
        target_faces,
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_stats = visible_surface_stats(
        human_depth,
        scene_depth,
        np,
        eps_m,
    )
    target_stats = visible_surface_stats(
        target_depth,
        np.minimum(scene_depth, human_depth),
        np,
        eps_m,
    )
    target_self_stats = visible_surface_stats(
        target_depth,
        human_depth,
        np,
        eps_m,
    )
    human_ratio = float(human_stats["visible_over_projected_surface_ratio"])
    target_ratio = float(target_stats["visible_over_projected_surface_ratio"])
    target_self_visible_ratio = float(target_self_stats["visible_over_projected_surface_ratio"])
    target_self_occluded_ratio = 1.0 - target_self_visible_ratio if int(target_self_stats["projected_pixel_count"]) else 1.0
    enriched = dict(metrics)
    enriched["human_visible_over_total_ratio"] = human_ratio
    enriched["target_visible_over_total_ratio"] = target_ratio
    enriched["human_visible_over_projected_surface_ratio"] = human_ratio
    enriched["target_visible_over_projected_surface_ratio"] = target_ratio
    enriched["human_visible_pixel_count"] = int(human_stats["visible_pixel_count"])
    enriched["human_projected_pixel_count"] = int(human_stats["projected_pixel_count"])
    enriched["target_visible_pixel_count"] = int(target_stats["visible_pixel_count"])
    enriched["target_projected_pixel_count"] = int(target_stats["projected_pixel_count"])
    enriched["target_self_visible_over_projected_surface_ratio"] = target_self_visible_ratio
    enriched["target_self_occluded_over_projected_surface_ratio"] = target_self_occluded_ratio
    enriched["target_self_visible_pixel_count"] = int(target_self_stats["visible_pixel_count"])
    enriched["target_self_projected_pixel_count"] = int(target_self_stats["projected_pixel_count"])
    enriched["visibility_metric"] = "rendered_surface_pixels"
    enriched["visibility_occlusion_model"] = "cropped_scene_depth_plus_human_self_depth"
    enriched["occlusion_depth_eps_m"] = float(eps_m)
    enriched["selection_score"] = float(4.0 * target_ratio + 2.0 * human_ratio)
    return enriched


def add_human_visibility_metrics(
    metrics: dict[str, Any],
    rendered: dict[str, Any],
    camera: dict[str, Any],
    image_size: tuple[int, int],
    eps_m: float,
) -> dict[str, Any]:
    np = rendered["deps"]["np"]
    scene_depth = rasterize_depth(
        rendered["scene_vertices"],
        rendered["scene_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_depth = rasterize_depth(
        rendered["human_vertices"],
        rendered["human_faces"],
        camera,
        image_size,
        rendered["deps"],
        rendered["device"],
    )
    human_stats = visible_surface_stats(
        human_depth,
        scene_depth,
        np,
        eps_m,
    )
    human_ratio = float(human_stats["visible_over_projected_surface_ratio"])
    height, width = image_size
    coverage = float(int(human_stats["visible_pixel_count"]) / max(width * height, 1))
    enriched = dict(metrics)
    enriched["human_visible_over_total_ratio"] = human_ratio
    enriched["human_visible_over_projected_surface_ratio"] = human_ratio
    enriched["human_visible_pixel_count"] = int(human_stats["visible_pixel_count"])
    enriched["human_projected_pixel_count"] = int(human_stats["projected_pixel_count"])
    enriched["human_visible_image_coverage_ratio"] = coverage
    enriched["visibility_metric"] = "rendered_surface_pixels"
    enriched["visibility_occlusion_model"] = "cropped_scene_depth"
    enriched["occlusion_depth_eps_m"] = float(eps_m)
    enriched["selection_score"] = float(4.0 * human_ratio + coverage)
    return enriched


def view_passes_visibility(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("human_visible_over_total_ratio", 0.0))
        >= float(args.min_human_visible_over_total_ratio)
        and float(metrics.get("target_visible_over_total_ratio", 0.0))
        >= float(args.min_target_visible_over_total_ratio)
    )


def global_view_passes_visibility(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("human_visible_over_total_ratio", 0.0))
        >= float(args.min_human_visible_over_total_ratio)
    )


def view_passes_self_occlusion(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(metrics.get("target_self_occluded_over_projected_surface_ratio", 1.0))
        <= float(args.max_target_self_occluded_ratio)
    )


VISUAL_SEGMENT_ALIASES = {
    "left_hand_inner": "left_hand",
    "right_hand_inner": "right_hand",
    "left_foot_bottom": "left_foot",
    "right_foot_bottom": "right_foot",
    "hips_contact": "hips",
}


def visual_segment_id(segment_id: str | None, part_name: str | None) -> str | None:
    if segment_id:
        segment_key = str(segment_id)
        if segment_key in VISUAL_SEGMENT_ALIASES:
            return VISUAL_SEGMENT_ALIASES[segment_key]
        return segment_key
    if part_name:
        return slugify(part_name)
    return None


def segment_vertices(segmentation: dict[str, Any], segment_id: str | None,
                     part_name: str | None) -> tuple[str | None, list[int]]:
    segments = segmentation.get("segments", {})
    candidates = []
    visual_id = visual_segment_id(segment_id, part_name)
    if visual_id:
        candidates.append(visual_id)
    if segment_id:
        candidates.append(str(segment_id))
    if part_name:
        part_slug = slugify(part_name)
        candidates.extend([part_slug, f"{part_slug}_inner", f"{part_slug}_bottom", f"{part_slug}_contact"])
    for candidate in candidates:
        if candidate in segments and isinstance(segments[candidate], list):
            return candidate, [int(index) for index in segments[candidate]]
    return visual_id, []


def bbox_from_projected(
    projected: Any,
    width: int,
    height: int,
    padding_frac: float,
    min_size: int,
    np: Any,
    aspect_ratio: float | None = None,
) -> list[int] | None:
    valid = projected[(projected[:, 2] > 0.02) & np.isfinite(projected[:, 0]) & np.isfinite(projected[:, 1])]
    if len(valid) == 0:
        return None
    x0, y0 = valid[:, :2].min(axis=0)
    x1, y1 = valid[:, :2].max(axis=0)
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)
    pad = max(box_w, box_h) * padding_frac
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    final_w = max(box_w + 2.0 * pad, float(min_size))
    final_h = max(box_h + 2.0 * pad, float(min_size))
    if aspect_ratio is not None and aspect_ratio > 0:
        current_aspect = final_w / max(final_h, 1.0)
        if current_aspect < aspect_ratio:
            final_w = final_h * aspect_ratio
        else:
            final_h = final_w / aspect_ratio
    bx0 = int(np.floor(cx - final_w * 0.5))
    by0 = int(np.floor(cy - final_h * 0.5))
    bx1 = int(np.ceil(cx + final_w * 0.5))
    by1 = int(np.ceil(cy + final_h * 0.5))
    return [bx0, by0, bx1, by1]


def contact_crop_min_size(part_name: str | None, segment_id: str | None, default_min_size: int) -> int:
    label = f"{part_name or ''} {segment_id or ''}".lower()
    if "hand" in label:
        return min(default_min_size, 180)
    if "foot" in label:
        return min(default_min_size, 240)
    if "hip" in label:
        return max(default_min_size, 360)
    return default_min_size


def optical_zoom_camera(
    camera: dict[str, Any],
    bbox: list[int],
    base_width: int,
    base_height: int,
    output_size: tuple[int, int],
    fill_frac: float,
) -> dict[str, Any]:
    output_width, output_height = output_size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    bbox_width = max(x1 - x0, 1.0)
    bbox_height = max(y1 - y0, 1.0)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5

    fx_base = float(camera["fx"]) * (float(base_width) / float(camera["width"]))
    fy_base = float(camera["fy"]) * (float(base_height) / float(camera["height"]))
    cx_base = float(camera["cx"]) * (float(base_width) / float(camera["width"]))
    cy_base = float(camera["cy"]) * (float(base_height) / float(camera["height"]))

    zoom = min(
        (float(output_width) * fill_frac) / bbox_width,
        (float(output_height) * fill_frac) / bbox_height,
    )
    zoom_camera = dict(camera)
    zoom_camera.update(
        {
            "width": int(output_width),
            "height": int(output_height),
            "fx": float(fx_base * zoom),
            "fy": float(fy_base * zoom),
            "cx": float(output_width * 0.5 + (cx_base - center_x) * zoom),
            "cy": float(output_height * 0.5 + (cy_base - center_y) * zoom),
            "optical_zoom_factor": float(zoom),
        }
    )
    return zoom_camera


def save_optical_zoom_render_with_visibility(
    rendered: dict[str, Any],
    bbox: list[int],
    path: Path,
    output_size: tuple[int, int],
    fill_frac: float,
    target_faces: Any,
    args: argparse.Namespace,
    base_camera: dict[str, Any],
    base_width: int,
    base_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    zoom_camera = optical_zoom_camera(base_camera, bbox, base_width, base_height, output_size, fill_frac)
    image = composite_scene_and_human(rendered, zoom_camera, (output_size[1], output_size[0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered["deps"]["Image"].fromarray(image).save(path)
    metadata = {
        "path": str(path),
        "output_size": [int(output_size[0]), int(output_size[1])],
        "optical_zoom_factor": zoom_camera["optical_zoom_factor"],
        "fx": zoom_camera["fx"],
        "fy": zoom_camera["fy"],
        "cx": zoom_camera["cx"],
        "cy": zoom_camera["cy"],
        "base_width": int(base_width),
        "base_height": int(base_height),
    }
    visibility = add_surface_visibility_ratios(
        {},
        rendered,
        zoom_camera,
        (output_size[1], output_size[0]),
        target_faces,
        float(args.occlusion_depth_eps_m),
    )
    return metadata, visibility


def select_contact_view_cameras(
    rendered: dict[str, Any],
    base_camera: dict[str, Any],
    human_vertices: Any,
    segment_vertex_ids: list[int],
    target_faces: Any,
    args: argparse.Namespace,
    np: Any,
) -> list[dict[str, Any]]:
    base_width, base_height = parse_size(args.view_planner_image_size)
    segment_vertices_np = human_vertices[np.asarray(segment_vertex_ids, dtype=np.int64)]
    contact_pivot = segment_vertices_np.mean(axis=0)
    original_depth = add_surface_visibility_ratios(
        {},
        rendered,
        base_camera,
        (base_height, base_width),
        target_faces,
        float(args.occlusion_depth_eps_m),
    )
    original_usable = view_passes_visibility(original_depth, args) and view_passes_self_occlusion(original_depth, args)
    candidates = []
    if original_usable:
        candidates.append(
            {
                "camera_kind": "original",
                "yaw_offset_deg": 0.0,
                "elevation_deg": 0.0,
                "camera": base_camera,
                "pivot_world": [float(value) for value in contact_pivot.tolist()],
                "orbit_radius_scale": 1.0,
                "visibility": original_depth,
                "selection_status": "selected",
            }
        )

    for yaw in parse_float_list(args.candidate_yaws_deg):
        for elevation in parse_float_list(args.candidate_elevations_deg):
            for radius_scale in parse_float_list(args.candidate_radius_scales):
                camera = orbit_camera(base_camera, contact_pivot, yaw, radius_scale, elevation, np)
                visibility = add_surface_visibility_ratios(
                    {},
                    rendered,
                    camera,
                    (base_height, base_width),
                    target_faces,
                    float(args.occlusion_depth_eps_m),
                )
                if not view_passes_visibility(visibility, args) or not view_passes_self_occlusion(visibility, args):
                    continue
                candidates.append(
                    {
                        "camera_kind": "orbit",
                        "yaw_offset_deg": float(yaw),
                        "elevation_deg": float(elevation),
                        "camera": camera,
                        "pivot_world": [float(value) for value in contact_pivot.tolist()],
                        "orbit_radius_scale": float(radius_scale),
                        "visibility": visibility,
                        "selection_status": "selected",
                    }
                )
    candidates.sort(key=lambda item: item["visibility"]["selection_score"], reverse=True)

    views = []
    selected_yaws: list[float] = []
    for candidate in candidates:
        if len(views) >= max(1, int(args.contact_view_count)):
            break
        yaw = float(candidate["yaw_offset_deg"])
        if any(
            abs(((yaw - used + 180.0) % 360.0) - 180.0) < float(args.min_view_angular_separation_deg)
            for used in selected_yaws
        ):
            continue
        candidate = dict(candidate)
        candidate["view_index"] = len(views)
        if candidate["camera_kind"] == "original":
            candidate["view_name"] = f"view_{len(views):02d}_original"
        else:
            candidate["view_name"] = (
                f"view_{len(views):02d}_orbit_yaw_{int(candidate['yaw_offset_deg']):+d}"
                f"_elev_{int(candidate['elevation_deg']):+d}_r_{candidate['orbit_radius_scale']:.2f}"
            )
        views.append(candidate)
        selected_yaws.append(yaw)
    return views


def build_view_evidence(
    rendered: dict[str, Any],
    view: dict[str, Any],
    segment_vertex_ids: list[int],
    target_faces: Any,
    edge_dir: Path,
    part_name: str | None,
    segment_id: str | None,
    args: argparse.Namespace,
    np: Any,
) -> dict[str, Any]:
    base_camera = view["camera"]
    base_width, base_height = parse_size(args.render_image_size)
    human_vertices = rendered["human_vertices"]
    segment_vertices_np = human_vertices[np.asarray(segment_vertex_ids, dtype=np.int64)]
    human_projected = project_vertices(human_vertices, base_camera, base_width, base_height, np)
    segment_projected = project_vertices(segment_vertices_np, base_camera, base_width, base_height, np)

    context_size = parse_size(args.contact_context_output_size)
    local_size = parse_size(args.contact_local_output_size)
    context_bbox = bbox_from_projected(
        human_projected,
        base_width,
        base_height,
        float(args.context_padding_frac),
        1,
        np,
        aspect_ratio=float(context_size[0]) / float(context_size[1]),
    )
    local_bbox = bbox_from_projected(
        segment_projected,
        base_width,
        base_height,
        float(args.crop_padding_frac),
        contact_crop_min_size(part_name, segment_id, int(args.crop_min_size_px)),
        np,
        aspect_ratio=float(local_size[0]) / float(local_size[1]),
    )
    if context_bbox is None:
        context_bbox = [0, 0, base_width, base_height]
    if local_bbox is None:
        local_bbox = context_bbox

    view_dir = edge_dir / view["view_name"]
    context_path = view_dir / "context.png"
    local_path = view_dir / "local_contact.png"
    context_render, final_context_visibility = save_optical_zoom_render_with_visibility(
        rendered,
        context_bbox,
        context_path,
        context_size,
        fill_frac=float(args.contact_context_fill_frac),
        target_faces=target_faces,
        args=args,
        base_camera=base_camera,
        base_width=base_width,
        base_height=base_height,
    )
    local_render, final_local_visibility = save_optical_zoom_render_with_visibility(
        rendered,
        local_bbox,
        local_path,
        local_size,
        fill_frac=float(args.contact_local_fill_frac),
        target_faces=target_faces,
        args=args,
        base_camera=base_camera,
        base_width=base_width,
        base_height=base_height,
    )
    return {
        "view_index": int(view["view_index"]),
        "view_name": view["view_name"],
        "camera_kind": view["camera_kind"],
        "yaw_offset_deg": float(view["yaw_offset_deg"]),
        "elevation_deg": float(view.get("elevation_deg", 0.0)),
        "pivot_world": view.get("pivot_world"),
        "orbit_radius_scale": view.get("orbit_radius_scale"),
        "candidate_camera": {
            "yaw_offset_deg": float(view["yaw_offset_deg"]),
            "elevation_deg": float(view.get("elevation_deg", 0.0)),
            "radius_scale": view.get("orbit_radius_scale"),
            "pivot_world": view.get("pivot_world"),
        },
        "planner_visibility": view["visibility"],
        "selection_status": view.get("selection_status", "selected"),
        "final_context_visibility": final_context_visibility,
        "final_local_visibility": final_local_visibility,
        "context_bbox_xyxy": context_bbox,
        "local_bbox_xyxy": local_bbox,
        "images": {
            "context": str(context_path),
            "local_contact": str(local_path),
        },
        "rendering": {
            "context": context_render,
            "local_contact": local_render,
        },
    }


def select_global_view_cameras(
    rendered: dict[str, Any],
    base_camera: dict[str, Any],
    args: argparse.Namespace,
    np: Any,
) -> list[dict[str, Any]]:
    base_width, base_height = parse_size(args.view_planner_image_size)
    human_vertices = rendered["human_vertices"]
    human_min = human_vertices.min(axis=0)
    human_max = human_vertices.max(axis=0)
    pivot = (human_min + human_max) * 0.5

    candidates = []
    original_visibility = add_human_visibility_metrics(
        {},
        rendered,
        base_camera,
        (base_height, base_width),
        float(args.occlusion_depth_eps_m),
    )
    if global_view_passes_visibility(original_visibility, args):
        candidates.append(
            {
                "camera_kind": "original",
                "yaw_offset_deg": 0.0,
                "elevation_deg": 0.0,
                "camera": base_camera,
                "pivot_world": [float(value) for value in pivot.tolist()],
                "orbit_radius_scale": 1.0,
                "visibility": original_visibility,
                "selection_status": "selected",
            }
        )

    for yaw in parse_float_list(args.candidate_yaws_deg):
        for elevation in parse_float_list(args.candidate_elevations_deg):
            for radius_scale in parse_float_list(args.candidate_radius_scales):
                camera = orbit_camera(base_camera, pivot, yaw, radius_scale, elevation, np)
                visibility = add_human_visibility_metrics(
                    {},
                    rendered,
                    camera,
                    (base_height, base_width),
                    float(args.occlusion_depth_eps_m),
                )
                if not global_view_passes_visibility(visibility, args):
                    continue
                candidates.append(
                    {
                        "camera_kind": "orbit",
                        "yaw_offset_deg": float(yaw),
                        "elevation_deg": float(elevation),
                        "camera": camera,
                        "pivot_world": [float(value) for value in pivot.tolist()],
                        "orbit_radius_scale": float(radius_scale),
                        "visibility": visibility,
                        "selection_status": "selected",
                    }
                )
    candidates.sort(key=lambda item: item["visibility"]["selection_score"], reverse=True)

    views = []
    selected_yaws: list[float] = []
    for candidate in candidates:
        if len(views) >= max(1, int(args.contact_view_count)):
            break
        yaw = float(candidate["yaw_offset_deg"])
        if any(
            abs(((yaw - used + 180.0) % 360.0) - 180.0) < float(args.min_view_angular_separation_deg)
            for used in selected_yaws
        ):
            continue
        candidate = dict(candidate)
        candidate["view_index"] = len(views)
        if candidate["camera_kind"] == "original":
            candidate["view_name"] = f"view_{len(views):02d}_original"
        else:
            candidate["view_name"] = (
                f"view_{len(views):02d}_orbit_yaw_{int(candidate['yaw_offset_deg']):+d}"
                f"_elev_{int(candidate['elevation_deg']):+d}_r_{candidate['orbit_radius_scale']:.2f}"
            )
        views.append(candidate)
        selected_yaws.append(yaw)
    return views


def build_global_view_evidence(
    rendered: dict[str, Any],
    view: dict[str, Any],
    views_dir: Path,
    args: argparse.Namespace,
    np: Any,
) -> dict[str, Any]:
    base_camera = view["camera"]
    base_width, base_height = parse_size(args.render_image_size)
    human_projected = project_vertices(rendered["human_vertices"], base_camera, base_width, base_height, np)
    output_size = parse_size(args.contact_context_output_size)
    bbox = bbox_from_projected(
        human_projected,
        base_width,
        base_height,
        float(args.context_padding_frac),
        1,
        np,
        aspect_ratio=float(output_size[0]) / float(output_size[1]),
    )
    if bbox is None:
        bbox = [0, 0, base_width, base_height]

    view_dir = views_dir / view["view_name"]
    image_path = view_dir / "view.png"
    zoom_camera = optical_zoom_camera(
        base_camera,
        bbox,
        base_width,
        base_height,
        output_size,
        float(args.contact_context_fill_frac),
    )
    image = composite_scene_and_human(rendered, zoom_camera, (output_size[1], output_size[0]))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    rendered["deps"]["Image"].fromarray(image).save(image_path)
    final_visibility = add_human_visibility_metrics(
        {},
        rendered,
        zoom_camera,
        (output_size[1], output_size[0]),
        float(args.occlusion_depth_eps_m),
    )
    return {
        "view_index": int(view["view_index"]),
        "view_name": view["view_name"],
        "camera_kind": view["camera_kind"],
        "yaw_offset_deg": float(view["yaw_offset_deg"]),
        "elevation_deg": float(view.get("elevation_deg", 0.0)),
        "pivot_world": view.get("pivot_world"),
        "orbit_radius_scale": view.get("orbit_radius_scale"),
        "candidate_camera": {
            "yaw_offset_deg": float(view["yaw_offset_deg"]),
            "elevation_deg": float(view.get("elevation_deg", 0.0)),
            "radius_scale": view.get("orbit_radius_scale"),
            "pivot_world": view.get("pivot_world"),
        },
        "planner_visibility": view["visibility"],
        "selection_status": view.get("selection_status", "selected"),
        "final_visibility": final_visibility,
        "bbox_xyxy": bbox,
        "image": str(image_path),
        "rendering": {
            "path": str(image_path),
            "output_size": [int(output_size[0]), int(output_size[1])],
            "optical_zoom_factor": zoom_camera["optical_zoom_factor"],
            "fx": zoom_camera["fx"],
            "fy": zoom_camera["fy"],
            "cx": zoom_camera["cx"],
            "cy": zoom_camera["cy"],
            "base_width": int(base_width),
            "base_height": int(base_height),
        },
    }


def read_prompt(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else fallback


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def generate_evidence(
    metrics: dict[str, Any],
    input_scene: dict[str, Any],
    optimizer_root: Path,
    outdir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    deps = import_render_deps()
    np = deps["np"]
    Image = deps["Image"]

    evidence_dir = outdir / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    contact_dir = evidence_dir / "contact"
    pose_dir = evidence_dir / "pose"
    penetration_dir = evidence_dir / "penetration"
    contact_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    penetration_dir.mkdir(parents=True, exist_ok=True)

    camera = read_camera(input_scene, Path(args.scannet_root).resolve())
    defaults = default_paths(args.interaction_name)
    contact_camera_json = resolve_path(args.contact_camera_json, defaults["contact_camera_json"])
    contact_canvas_image = resolve_path(args.contact_canvas_image, defaults["contact_canvas_image"])
    contact_crop_camera = read_contact_crop_camera(
        camera,
        contact_camera_json,
        contact_canvas_image,
        Image,
    )
    human_mesh_path = optimizer_root / "meshes" / "frame_0000_world.ply"
    log("load", f"source ScanNet scene mesh: {camera['scene_mesh_path']}")
    log("load", f"contact crop camera: {contact_camera_json}")
    log("load", f"contact crop canvas: {contact_canvas_image}")
    log("load", f"camera image: {camera['image_path']}")
    log("load", f"camera transforms: {camera['transforms_path']}")
    log("load", f"camera extrinsics: {camera['colmap_images_path']}")
    log("load", f"SMPL-X segmentation: {Path(args.smpl_seg_json).resolve()}")
    segmentation = load_json(Path(args.smpl_seg_json).resolve())

    rendered = render_scene_with_human(camera, contact_crop_camera, human_mesh_path, args)
    image = rendered["image"]
    full_scene_path = evidence_dir / "full_scene.png"
    Image.fromarray(image).save(full_scene_path)

    global_views_dir = pose_dir / "views"
    global_views = []
    selected_global_views = select_global_view_cameras(rendered, camera, args, np)
    for view in selected_global_views:
        view_evidence = build_global_view_evidence(rendered, view, global_views_dir, args, np)
        global_views.append(view_evidence)
    log("evidence", f"global pose/penetration views={len(global_views)}")

    contact_entries = []
    manifest_files = [str(full_scene_path)]
    manifest_files.extend([view["image"] for view in global_views])
    for edge in metrics.get("contact", {}).get("edges", []):
        index = int(edge["index"])
        part_name = edge.get("moving_part_name")
        segment_id = edge.get("moving_segment_id")
        target = edge.get("fixed_entity_name") or edge.get("fixed_part_name")
        crop_segment_id, segment_ids = segment_vertices(segmentation, segment_id, part_name)
        edge_slug = f"edge_{index:02d}_{slugify(part_name or 'part')}_to_{slugify(target or 'target')}"
        edge_dir = contact_dir / edge_slug
        views: list[dict[str, Any]] = []
        if not segment_ids:
            log("warn", f"edge={index} no segmentation vertices for {segment_id or part_name}")
        else:
            target_faces = target_faces_for_segment(rendered["human_faces"], segment_ids, np)
            highlighted_rendered = {
                **rendered,
                "human_colors": highlight_vertex_colors(rendered["human_colors"], segment_ids, np),
            }
            selected_views = select_contact_view_cameras(
                highlighted_rendered,
                camera,
                rendered["human_vertices"],
                segment_ids,
                target_faces,
                args,
                np,
            )
            for view in selected_views:
                view_evidence = build_view_evidence(
                    highlighted_rendered,
                    view,
                    segment_ids,
                    target_faces,
                    edge_dir,
                    part_name,
                    segment_id,
                    args,
                    np,
                )
                views.append(view_evidence)
                manifest_files.extend([view_evidence["images"]["context"], view_evidence["images"]["local_contact"]])

        entry = {
            "edge_index": index,
            "body_part": part_name,
            "moving_segment_id": segment_id,
            "crop_segment_id": crop_segment_id,
            "target": target,
            "contact_distance_m": edge.get("nocontact_distance_m"),
            "contact_threshold_m": edge.get("threshold_m"),
            "deterministic_pass": edge.get("pass"),
            "view_count": len(views),
            "views": views,
        }
        json_path = edge_dir / f"{edge_slug}_evidence.json"
        save_json(json_path, entry)
        manifest_files.append(str(json_path))
        contact_entries.append(entry)
        log(
            "evidence",
            f"edge={index} part={part_name} crop_segment={crop_segment_id} target={target} "
            f"distance={edge.get('nocontact_distance_m')} views={len(views)}",
        )

    pose_entry = {
        "interaction": metrics.get("interaction"),
        "height": metrics.get("height"),
        "optimization": metrics.get("optimization"),
        "view_count": len(global_views),
        "views": global_views,
        "images": {
            "full_scene": str(full_scene_path),
            "views": [view["image"] for view in global_views],
        },
    }
    pose_json_path = pose_dir / "pose_evidence.json"
    save_json(pose_json_path, pose_entry)
    manifest_files.append(str(pose_json_path))

    penetration_entry = {
        "penetration": metrics.get("penetration"),
        "view_count": len(global_views),
        "views": global_views,
        "images": {
            "full_scene": str(full_scene_path),
            "views": [view["image"] for view in global_views],
        },
    }
    penetration_json_path = penetration_dir / "penetration_evidence.json"
    save_json(penetration_json_path, penetration_entry)
    manifest_files.append(str(penetration_json_path))

    manifest = {
        "evidence_dir": str(evidence_dir),
        "render_mode": "scannet_colored_scene_neutral_smplx_human",
        "camera_image": str(camera["image_path"]),
        "scene_mesh": str(camera["scene_mesh_path"]),
        "scene_crop": rendered.get("scene_crop"),
        "human_mesh": str(human_mesh_path),
        "files": manifest_files,
        "contact_edges": contact_entries,
        "pose": pose_entry,
        "penetration": penetration_entry,
    }
    manifest_path = evidence_dir / "evidence_manifest.json"
    save_json(manifest_path, manifest)
    log("evidence", f"wrote manifest: {manifest_path}")
    return manifest


def extract_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(raw[start: end + 1])
    if not isinstance(payload, dict):
        raise ValueError("VLM response JSON must be an object")
    return payload


def normalize_judgment(payload: dict[str, Any], edge_index: int | None = None) -> dict[str, Any]:
    passed = payload.get("pass")
    if not isinstance(passed, bool):
        passed = None
    result = {
        "pass": passed,
        "reason": str(payload.get("reason", ""))[:1000],
    }
    if edge_index is not None:
        result = {"edge_index": int(edge_index), **result}
    return result


def vlm_request(
    client: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
    timeout: float,
    thinking_effort: str,
) -> str:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": task_text,
        }
    ]
    for path in image_paths:
        if path.exists():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image(path)}"},
                }
            )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0,
        timeout=timeout,
        extra_body={"reasoning_effort": thinking_effort},
    )
    return response.choices[0].message.content or ""


def judge_with_vlm(
    judge_name: str,
    client: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
    args: argparse.Namespace,
    edge_index: int | None = None,
) -> dict[str, Any]:
    for attempt in range(int(args.retries) + 1):
        try:
            suffix = f" edge={edge_index}" if edge_index is not None else ""
            log("vlm", f"start judge={judge_name}{suffix} attempt={attempt + 1}")
            text = vlm_request(
                client,
                args.model,
                system_prompt,
                task_text,
                image_paths,
                args.timeout,
                args.vlm_thinking_effort,
            )
            payload = extract_json_object(text)
            judgment = normalize_judgment(payload, edge_index=edge_index)
            if judgment["pass"] is None:
                raise ValueError("VLM JSON did not contain boolean pass")
            log("vlm", f"done judge={judge_name}{suffix} pass={judgment['pass']}")
            return judgment
        except Exception as error:
            log("warn", f"judge={judge_name} failed attempt={attempt + 1}: {error}")
    return normalize_judgment(
        {
            "pass": None,
            "reason": f"VLM judge failed after retries: {judge_name}",
        },
        edge_index=edge_index,
    )


def contact_task_text(edge: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            "Task: Decide whether the highlighted red human body part is in plausible contact with the target.",
            f"Interaction: {interaction}",
            f"Body part: {edge.get('body_part')}",
            f"Moving segment: {edge.get('moving_segment_id')}",
            f"Target: {edge.get('target')}",
            f"Number of rendered views: {edge.get('view_count', 0)}",
            "Each view contributes two images in order: context image, then local contact image.",
            'Return only strict JSON: {"pass": true, "reason": "short concrete reason"}',
        ]
    )


def pose_task_text(pose: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            "Task: Decide whether the optimized human pose is plausible for the interaction.",
            f"Interaction: {interaction}",
            f"Number of rendered views: {pose.get('view_count', 0)}",
            "Use all images together. A problem does not need to be visible in every view.",
            'Return only strict JSON: {"pass": true, "reason": "short concrete reason"}',
        ]
    )


def penetration_task_text(penetration: dict[str, Any], interaction: str) -> str:
    return "\n".join(
        [
            "Task: Decide whether there is visible serious human-scene penetration.",
            f"Interaction: {interaction}",
            f"Number of rendered views: {penetration.get('view_count', 0)}",
            "Use all images together. A problem does not need to be visible in every view.",
            'Return only strict JSON: {"pass": true, "reason": "short concrete reason"}',
        ]
    )


def run_vlm_judgments(evidence_manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_vlm:
        log("vlm", "skipped because --skip-vlm was set")
        return {
            "enabled": False,
            "model": None,
            "thinking_effort": args.vlm_thinking_effort,
            "contact_edges": [],
            "pose": {},
            "penetration": {},
        }

    try:
        from openai import OpenAI
    except Exception as error:
        raise RuntimeError(f"Could not import OpenAI client for VLM judging: {error}") from error

    client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key, max_retries=0)
    log(
        "vlm",
        f"enabled model={args.model} host={args.ollama_host} retries={args.retries} "
        f"thinking_effort={args.vlm_thinking_effort}",
    )
    contact_prompt = read_prompt(
        CONTACT_PROMPT,
        (
            "Return strict JSON judging whether the crop shows the requested "
            "body part in plausible contact with the target."
        ),
    )
    pose_prompt = read_prompt(
        POSE_PROMPT,
        "Return strict JSON judging whether the optimized human pose is plausible for the interaction.")
    penetration_prompt = read_prompt(PENETRATION_PROMPT,
                                     "Return strict JSON judging whether the human visibly penetrates scene geometry.")

    pose = evidence_manifest.get("pose", {})
    interaction = str(pose.get("interaction", ""))
    contact_judgments = []
    for edge in evidence_manifest.get("contact_edges", []):
        edge_index = int(edge.get("edge_index", len(contact_judgments)))
        image_paths: list[Path] = []
        for view in edge.get("views", []):
            images = view.get("images", {})
            for key in ["context", "local_contact"]:
                path = images.get(key)
                if path:
                    image_paths.append(Path(path))
        judgment = judge_with_vlm(
            "contact",
            client,
            args.model,
            contact_prompt,
            contact_task_text(edge, interaction),
            image_paths,
            args,
            edge_index=edge_index,
        )
        contact_judgments.append(
            {
                "edge_index": edge_index,
                "body_part": edge.get("body_part"),
                "moving_segment_id": edge.get("moving_segment_id"),
                "crop_segment_id": edge.get("crop_segment_id"),
                "target": edge.get("target"),
                "view_count": int(edge.get("view_count", 0)),
                "image_paths": [str(path) for path in image_paths],
                **judgment,
            }
        )

    pose_images = [Path(path) for path in pose.get("images", {}).get("views", []) if path]
    pose_judgment = judge_with_vlm(
        "pose",
        client,
        args.model,
        pose_prompt,
        pose_task_text(pose, interaction),
        pose_images,
        args,
    )

    penetration = evidence_manifest.get("penetration", {})
    penetration_images = [Path(path) for path in penetration.get("images", {}).get("views", []) if path]
    penetration_judgment = judge_with_vlm(
        "penetration",
        client,
        args.model,
        penetration_prompt,
        penetration_task_text(penetration, interaction),
        penetration_images,
        args)

    return {
        "enabled": True,
        "model": args.model,
        "host": args.ollama_host,
        "thinking_effort": args.vlm_thinking_effort,
        "contact_edges": contact_judgments,
        "pose": pose_judgment,
        "penetration": penetration_judgment,
    }


def main() -> None:
    args = parse_args()
    defaults = default_paths(args.interaction_name)
    optimizer_root = resolve_path(args.optimizer_output_root, defaults["optimizer_output_root"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    outdir = resolve_path(args.outdir, defaults["outdir"])

    log("load", f"interaction={args.interaction_name} outdir={outdir}")
    validate_required_inputs(optimizer_root, sig_json_path, input_scene_json_path)
    log("load", f"SIG: {sig_json_path}")
    log("load", f"input scene: {input_scene_json_path}")
    log("load", f"optimizer summary: {optimizer_root / 'alignment_summary.json'}")
    log("load", f"mesh: {optimizer_root / 'meshes' / 'frame_0000_world.ply'}")

    sig_payload = load_json(sig_json_path)
    input_scene = load_json(input_scene_json_path)
    alignment_summary = load_json(optimizer_root / "alignment_summary.json")
    metrics = collect_metrics(
        sig_payload=sig_payload,
        alignment_summary=alignment_summary,
        optimizer_root=optimizer_root,
        sig_json_path=sig_json_path,
        input_scene_json_path=input_scene_json_path,
        args=args,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / "metrics.json", metrics)
    log(
        "metrics",
        f"contact_pass={metrics['contact']['pass']} penetration_pass={metrics['penetration']['pass']} "
        f"edges={metrics['contact']['edge_count']} deterministic_pass={metrics['deterministic']['pass']}",
    )

    log("evidence", "generating original-camera scene render and per-edge crops")
    evidence_manifest = generate_evidence(metrics, input_scene, optimizer_root, outdir, args)
    vlm_judgments = run_vlm_judgments(evidence_manifest, args)
    save_json(outdir / "vlm_judgments.json", vlm_judgments)

    summary = build_verification_summary(metrics, vlm_judgments)
    save_json(outdir / "verification_summary.json", summary)

    log("summary", f"status={summary['status']} failure_tags={summary['failure_tags']}")
    log("summary", f"wrote metrics: {outdir / 'metrics.json'}")
    log("summary", f"wrote VLM judgments: {outdir / 'vlm_judgments.json'}")
    log("summary", f"wrote verification summary: {outdir / 'verification_summary.json'}")


def run_cli() -> None:
    try:
        main()
    except Exception as error:
        raise SystemExit(f"[error] {error}") from None


if __name__ == "__main__":
    run_cli()
