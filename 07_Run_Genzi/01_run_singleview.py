from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from omegaconf import OmegaConf
from PIL import Image, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"

if str(GENZI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZI_ROOT))


def ensure_alphapose_import_path() -> Path | None:
    try:
        alphapose = importlib.import_module("alphapose")
    except Exception:
        return None
    alphapose_root = Path(alphapose.__file__).resolve().parent.parent
    if str(alphapose_root) not in sys.path:
        sys.path.insert(0, str(alphapose_root))
    return alphapose_root


IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


@dataclass
class Paths:
    generated_root: Path
    input_scene_json: Path
    sig_json: Path
    contact_masks_dir: Path
    contact_canvas_path: Path
    contact_spec: Path
    output_root: Path


@dataclass
class Camera:
    intrinsics: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    width: int
    height: int


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_label(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def normalize_scene_element(text: str, target_labels: set[str] | None = None) -> str:
    raw = str(text).strip().lower()
    normalized = normalize_label(text)
    labels = target_labels or set()
    if (
        raw == "target_object"
        or raw.startswith("target_object_")
        or normalized in {"target object", "object", "target object 1", "target object 2"}
        or normalized in labels
    ):
        return "target_object"
    return normalized


def resolve_sig_target_label(sig_payload: dict[str, Any]) -> str:
    target_objects = sig_payload.get("target_objects")
    if isinstance(target_objects, list) and target_objects:
        first_target = target_objects[0]
        if isinstance(first_target, dict):
            label = str(first_target.get("label", "")).strip()
            if label:
                return label
    target_object = sig_payload.get("target_object", {})
    if not isinstance(target_object, dict):
        return ""
    return str(target_object.get("label", "")).strip()


def resolve_scannet_root(raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (WORKSPACE_ROOT / "Scannet++" / "data").resolve()


def build_paths(interaction_name: str, output_base: Path | None = None) -> Paths:
    if output_base is None:
        output_base = DEFAULT_OUTPUT_BASE
    agentic_root = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output" / interaction_name
    return Paths(
        generated_root=PROJECT_DIR / "02_Generate_Human_Frame" / "output" / interaction_name,
        input_scene_json=PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        sig_json=PROJECT_DIR / "01_Generate_SIG" / "output" / interaction_name / "sig.json",
        contact_masks_dir=agentic_root / "contact_masks",
        contact_canvas_path=agentic_root / "assets" / "target_scene_crop.png",
        contact_spec=agentic_root / "contact_spec.json",
        output_root=output_base / interaction_name,
    )


def required_input_paths(paths: Paths) -> dict[str, Path]:
    return {
        "inpainted_frame": paths.generated_root / "inpainted_frame_resized.png",
        "input_scene_json": paths.input_scene_json,
        "sig_json": paths.sig_json,
    }


def discover_interactions(output_base: Path | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    names = sorted(
        path.name
        for path in (PROJECT_DIR / "02_Generate_Human_Frame" / "output").glob("interaction_*")
        if path.is_dir()
    )
    runnable: list[str] = []
    skipped: list[dict[str, Any]] = []
    for name in names:
        paths = build_paths(name, output_base=output_base)
        missing = [
            {"name": key, "path": str(path)}
            for key, path in required_input_paths(paths).items()
            if not path.exists()
        ]
        if missing:
            skipped.append({"interaction_name": name, "missing": missing})
        else:
            runnable.append(name)
    return runnable, skipped


def build_pinhole_intrinsics(payload: dict[str, Any]) -> tuple[np.ndarray, int, int]:
    width = int(payload["w"])
    height = int(payload["h"])
    intrinsics = np.array(
        [
            [float(payload["fl_x"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fl_y"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, width, height


def colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def load_colmap_pose(colmap_images_path: Path, camera_name: str) -> tuple[np.ndarray, np.ndarray]:
    for line in colmap_images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(f"Could not find camera '{camera_name}' in {colmap_images_path}")


def resolve_scene_paths(scannet_root: Path, scene_context: dict[str, Any]) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera_payload = scene_context["camera"]
    source = camera_payload["source"]
    if source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(f"Unsupported camera source '{source}'.")
    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[source]
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "image_path": scene_root / image_rel / camera_payload["name"],
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
    }


def load_scannet_camera(scene_paths: dict[str, Path], scene_context: dict[str, Any]) -> Camera:
    transforms_payload = load_json(scene_paths["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )
    return Camera(
        intrinsics=intrinsics,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        width=width,
        height=height,
    )


def transform_world_to_camera(points_world: np.ndarray, camera: Camera) -> np.ndarray:
    return points_world @ camera.rotation_world_to_camera.T + camera.translation_world_to_camera[None]


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def load_contact_camera(contact_spec: Path, contact_canvas_path: Path, camera: Camera) -> Camera:
    payload = load_json(contact_spec)
    contact_intrinsics = np.asarray(payload["camera"]["intrinsics_3x3"], dtype=np.float32)
    image = cv2.imread(str(contact_canvas_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read contact canvas: {contact_canvas_path}")
    height, width = image.shape[:2]
    return Camera(
        intrinsics=contact_intrinsics,
        rotation_world_to_camera=camera.rotation_world_to_camera,
        translation_world_to_camera=camera.translation_world_to_camera,
        width=int(width),
        height=int(height),
    )


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(points_camera[:, 2], 1e-6, None)
    u = intrinsics[0, 0] * points_camera[:, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / z + intrinsics[1, 2]
    return u, v


def contact_mask_names(sig_payload: dict[str, Any], target_only: bool) -> list[str]:
    target_label = resolve_sig_target_label(sig_payload)
    target_norm = normalize_label(target_label)
    names: list[str] = []
    for edge in sig_payload.get("interaction_edges", []):
        if not isinstance(edge, dict):
            continue
        part = normalize_label(str(edge.get("human_part", "")))
        scene_element = normalize_scene_element(
            str(edge.get("scene_element", "")), {target_norm}
        )
        if target_only and scene_element != "target_object":
            continue
        if part:
            names.append(slugify_segment_name(part))
    return sorted(set(names))


def derive_interaction_point(
    sig_payload: dict[str, Any],
    contact_masks_dir: Path,
    contact_camera: Camera,
    scene_vertices_world: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices_camera = transform_world_to_camera(scene_vertices_world, contact_camera)
    u, v = project_points(vertices_camera, contact_camera.intrinsics)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_frame = (
        (vertices_camera[:, 2] > 1e-6)
        & (ui >= 0)
        & (ui < contact_camera.width)
        & (vi >= 0)
        & (vi < contact_camera.height)
    )

    attempts = [
        ("target_object", contact_mask_names(sig_payload, target_only=True)),
        (
            "all_contact_masks",
            [
                path.stem
                for path in sorted(contact_masks_dir.glob("*.png"))
                if path.stem not in {"metadata"}
            ],
        ),
    ]
    used_masks: list[dict[str, Any]] = []
    for source, names in attempts:
        hit_flags = np.zeros((scene_vertices_world.shape[0],), dtype=bool)
        for name in names:
            mask_path = contact_masks_dir / f"{name}.png"
            if not mask_path.exists():
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != (contact_camera.height, contact_camera.width):
                continue
            hits = in_frame.copy()
            hits[hits] &= mask[vi[hits], ui[hits]] > 127
            hit_flags |= hits
            used_masks.append(
                {
                    "source": source,
                    "mask": str(mask_path),
                    "hit_vertices": int(hits.sum()),
                }
            )
        if hit_flags.any():
            point = np.mean(scene_vertices_world[hit_flags], axis=0).astype(np.float32)
            return point, {
                "source": source,
                "num_vertices": int(hit_flags.sum()),
                "used_masks": used_masks,
            }

    visible = scene_vertices_world[in_frame]
    if visible.shape[0] == 0:
        point = np.mean(scene_vertices_world, axis=0).astype(np.float32)
        source = "scene_centroid"
    else:
        point = np.mean(visible, axis=0).astype(np.float32)
        source = "visible_scene_centroid"
    return point, {"source": source, "num_vertices": int(visible.shape[0]), "used_masks": used_masks}


def stage_single_view_external_inpaint(
    paths: Paths,
    interaction_name: str,
    image_size: int,
    num_stages: int,
) -> tuple[Path, dict[str, Any]]:
    src = paths.generated_root / "inpainted_frame_resized.png"
    external_root = ensure_dir(paths.output_root / "singleview_external_inpaints")
    staged_images: list[Path] = []
    with Image.open(src) as image:
        image = image.convert("RGB")
        original_size = image.size
        staged = ImageOps.fit(
            image,
            (int(image_size), int(image_size)),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        for stage_idx in range(int(num_stages)):
            dst_dir = ensure_dir(
                external_root
                / interaction_name
                / interaction_name
                / f"stage{stage_idx:03d}_inpaint000"
            )
            dst = dst_dir / "view000.png"
            staged.save(dst)
            staged_images.append(dst)
    return external_root, {
        "source": src,
        "staged_images": staged_images,
        "original_size": list(original_size),
        "staged_size": [int(image_size), int(image_size)],
        "num_stages": int(num_stages),
        "note": "Center-cropped/resized to GenZI render.image_size because stock GenZI normalizes AlphaPose joints by that size.",
    }


def camera_eye_up_fov(camera: Camera) -> tuple[np.ndarray, np.ndarray, float]:
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float32)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float32)
    eye = -(rotation.T @ translation)
    up = -(rotation.T @ np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    up = up / np.linalg.norm(up)
    fov_y = math.degrees(2.0 * math.atan(float(camera.height) / (2.0 * float(camera.intrinsics[1, 1]))))
    return eye.astype(np.float32), up.astype(np.float32), float(fov_y)


def default_render_args() -> dict[str, Any]:
    return {
        "bg_color": [0.5, 0.5, 0.5, 0.0],
        "ambient_light": [0.0, 0.0, 0.0],
        "dir_light_color": [1.0, 1.0, 1.0],
        "dir_light_intensity": 5.0,
        "pt_light_color": [1.0, 1.0, 1.0],
        "pt_light_intensity": 5.0,
        "pt_light_position": [0.0, 0.0, 20.0],
        "normal_pbr": True,
        "no_lighting": False,
        "all_solid": False,
        "cull_faces": False,
        "shadows": False,
    }


def configure_single_view_external_inpaint(cfg: dict[str, Any]) -> int:
    num_stages = len(cfg["optim.steps"])
    cfg["loss.inpaints_per_view"] = [1 for _ in range(num_stages)]
    cfg["loss.inpaint_min_views"] = 1
    cfg["data.max_views"] = 1
    return num_stages


def load_run_genzi_helpers() -> Any:
    script_path = MODULE_DIR / "03_run_genzi.py"
    spec = importlib.util.spec_from_file_location("dhsi_run_genzi_helpers", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def configure_cuda_visibility(args: argparse.Namespace) -> None:
    device = str(args.device)
    args.physical_device = device
    if not device.startswith("cuda:"):
        args.genzi_gpu_id = 0
        return
    requested_gpu = int(device.split(":")[-1])
    if requested_gpu != 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(requested_gpu)
        args.physical_device = f"cuda:{requested_gpu}"
        args.device = "cuda:0"
        args.genzi_gpu_id = 0
        return
    args.physical_device = device
    args.genzi_gpu_id = requested_gpu


def load_module02_look_at(output_root: Path) -> tuple[np.ndarray, dict[str, Any]] | None:
    summary_path = output_root / "sdf" / "sdf_summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    look_at = summary.get("look_at")
    if not isinstance(look_at, list) or len(look_at) != 3:
        return None
    return np.asarray(look_at, dtype=np.float32), {
        "source": "module02_sdf_summary",
        "summary_path": summary_path,
        "module02_look_at_stats": summary.get("look_at_stats", {}),
    }


def load_cfg(run_cfg: Path) -> dict[str, Any]:
    from genzi.misc import omegaconf_to_dotdict

    cfg = OmegaConf.load(run_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict)
    cfg["run_cfg"] = str(run_cfg)

    def absolutize(value: str) -> str:
        if not isinstance(value, str):
            return value
        if value.startswith("./"):
            return str((GENZI_ROOT / value[2:]).resolve())
        return value

    cfg = OmegaConf.create(cfg)
    flat = omegaconf_to_dotdict(cfg)
    for key, value in list(flat.items()):
        if key.endswith("_path") or key in {"path_prefix", "log_dir"}:
            if isinstance(value, str):
                flat[key] = absolutize(value)
    return flat


def ensure_smplx_symlink(cfg: dict[str, Any]) -> None:
    model_path = Path(cfg["smplx.model_path"])
    expected_dir = model_path / "smplx"
    expected_npz = expected_dir / "SMPLX_NEUTRAL.npz"
    if expected_npz.exists():
        return
    source_dir = WORKSPACE_ROOT / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx"
    source_npz = source_dir / "SMPLX_NEUTRAL.npz"
    if not source_npz.exists():
        return
    model_path.mkdir(parents=True, exist_ok=True)
    if expected_dir.exists() or expected_dir.is_symlink():
        return
    expected_dir.symlink_to(source_dir, target_is_directory=True)


def preflight(cfg: dict[str, Any], strict: bool = True) -> dict[str, Any]:
    ensure_smplx_symlink(cfg)
    ensure_alphapose_import_path()
    imports = [
        "torch",
        "smplx",
        "human_body_prior",
        "alphapose",
        "detector.apis",
        "pytorch3d",
        "nvdiffrast",
        "mesh_intersection",
        "pyrender",
    ]
    missing_imports = []
    versions = {}
    for module_name in imports:
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "ok")
        except Exception as exc:
            missing_imports.append(
                {
                    "module": module_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    missing_assets: list[dict[str, str]] = []
    for key in ("vposer.ckpt_path", "smplx.model_path"):
        path = Path(cfg[key])
        if not path.exists():
            missing_assets.append({"name": key, "path": str(path)})
    if Path(cfg["smplx.model_path"]).exists():
        smplx_npz = Path(cfg["smplx.model_path"]) / "smplx" / "SMPLX_NEUTRAL.npz"
        if not smplx_npz.exists():
            missing_assets.append({"name": "SMPLX_NEUTRAL.npz", "path": str(smplx_npz)})

    alphapose_paths: dict[str, str] = {}
    try:
        alphapose = importlib.import_module("alphapose")
        alphapose_root = Path(alphapose.__file__).resolve().parent.parent
        with Path(cfg["pose2d.args_path"]).open("r", encoding="utf-8") as file_obj:
            alphapose_cfg = yaml.safe_load(file_obj)
        for name in ("cfg", "checkpoint"):
            ap_path = alphapose_root / alphapose_cfg[name]
            alphapose_paths[name] = str(ap_path)
            if not ap_path.exists():
                missing_assets.append({"name": f"alphapose.{name}", "path": str(ap_path)})
        pose_cfg_path = alphapose_root / alphapose_cfg["cfg"]
        if pose_cfg_path.exists():
            with pose_cfg_path.open("r", encoding="utf-8") as file_obj:
                pose_cfg = yaml.safe_load(file_obj)
            detector_cfg = pose_cfg.get("DETECTOR", {})
            for name in ("CONFIG", "WEIGHTS"):
                if name not in detector_cfg:
                    continue
                ap_path = alphapose_root / detector_cfg[name]
                asset_name = f"alphapose.detector.{name.lower()}"
                alphapose_paths[f"detector.{name.lower()}"] = str(ap_path)
                if not ap_path.exists():
                    missing_assets.append({"name": asset_name, "path": str(ap_path)})
    except Exception as exc:
        alphapose_paths["error"] = f"{type(exc).__name__}: {exc}"

    result = {
        "ok": not missing_imports and not missing_assets,
        "imports": versions,
        "missing_imports": missing_imports,
        "missing_assets": missing_assets,
        "alphapose_paths": alphapose_paths,
    }
    if strict and not result["ok"]:
        lines = ["GenZI single-view preflight failed."]
        if missing_imports:
            lines.append("Missing imports:")
            lines.extend(f"  - {x['module']}: {x['error']}" for x in missing_imports)
        if missing_assets:
            lines.append("Missing assets:")
            lines.extend(f"  - {x['name']}: {x['path']}" for x in missing_assets)
        raise RuntimeError("\n".join(lines))
    return result


def maybe_limit_steps(cfg: dict[str, Any], max_steps: int | None) -> None:
    if max_steps is None:
        return
    for stage_idx, steps in enumerate(cfg["optim.steps"]):
        total = sum(int(x) for x in steps)
        if total <= int(max_steps):
            continue
        raw = np.asarray(steps, dtype=np.float64)
        scaled = np.maximum(1, np.floor(raw / raw.sum() * int(max_steps))).astype(int)
        while int(scaled.sum()) < int(max_steps):
            scaled[int(np.argmax(raw))] += 1
        while int(scaled.sum()) > int(max_steps):
            idx = int(np.argmax(scaled))
            if scaled[idx] > 1:
                scaled[idx] -= 1
            else:
                break
        cfg["optim.steps"][stage_idx] = scaled.tolist()


def run_interaction(
    interaction_name: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    paths = build_paths(
        interaction_name,
        output_base=Path(args.output_base).resolve() if args.output_base else None,
    )
    for name, path in required_input_paths(paths).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    output_root = ensure_dir(paths.output_root)
    input_payload = load_json(paths.input_scene_json)
    sig_payload = load_json(paths.sig_json)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(resolve_scannet_root(args.scannet_root), scene_context)
    for name in ("transforms_path", "colmap_images_path", "mesh_path"):
        if not scene_paths[name].exists():
            raise FileNotFoundError(f"Missing scene {name}: {scene_paths[name]}")

    scene_id = str(scene_context["scene_id"])
    sdf_meta_path = output_root / "sdf" / f"{scene_id}.json"
    sdf_npy_path = output_root / "sdf" / f"{scene_id}_sdf.npy"
    if not sdf_meta_path.exists() or not sdf_npy_path.exists():
        raise FileNotFoundError(
            "Missing SDF generated by 02_render_multiview.py. Expected "
            f"{sdf_meta_path} and {sdf_npy_path}"
        )
    sdf_metadata = load_json(sdf_meta_path)
    if sdf_metadata.get("method") != "selected_view_depth_tsdf":
        raise RuntimeError(
            "Single-view baseline expects the depth TSDF from 02_render_multiview.py. "
            f"Found method={sdf_metadata.get('method')!r} at {sdf_meta_path}"
        )

    camera = load_scannet_camera(scene_paths, scene_context)
    module02_look_at = load_module02_look_at(output_root)
    if module02_look_at is not None:
        interaction_point, interaction_point_stats = module02_look_at
    else:
        for name, path in {
            "contact_masks_dir": paths.contact_masks_dir,
            "contact_canvas_path": paths.contact_canvas_path,
            "contact_spec": paths.contact_spec,
        }.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {name}: {path}. Re-run 02_render_multiview.py so "
                    "sdf/sdf_summary.json is available, or provide the contact outputs."
                )
        contact_camera = load_contact_camera(paths.contact_spec, paths.contact_canvas_path, camera)
        scene_vertices_world, _scene_faces = load_mesh(scene_paths["mesh_path"])
        interaction_point, interaction_point_stats = derive_interaction_point(
            sig_payload=sig_payload,
            contact_masks_dir=paths.contact_masks_dir,
            contact_camera=contact_camera,
            scene_vertices_world=scene_vertices_world,
        )

    run_cfg = dict(cfg)
    num_stages = configure_single_view_external_inpaint(run_cfg)
    maybe_limit_steps(run_cfg, args.max_steps)

    image_size = int(run_cfg["render.image_size"])
    external_root, staged_inpaint = stage_single_view_external_inpaint(
        paths=paths,
        interaction_name=interaction_name,
        image_size=image_size,
        num_stages=num_stages,
    )
    eye, up_dir, fov_y = camera_eye_up_fov(camera)
    stage_viewpoints = [[eye.astype(float).tolist()] for _ in range(num_stages)]
    prompt = input_payload.get("interaction_context", {}).get("interaction", "")
    interaction_label = sig_payload.get("interaction", prompt)
    scene_yaml_payload = {
        "scene": {
            "mesh_path": str(scene_paths["mesh_path"].resolve()),
            "sdf_path": str(sdf_meta_path.resolve()),
            "subd_mesh_path": "",
        },
        "render": {
            **default_render_args(),
            "up_dir": up_dir.astype(float).tolist(),
        },
        "prompt_prefix": "",
        "prompt_suffix": "",
        "prompt_ids": [interaction_name],
        "prompts": [prompt],
        "neg_prompts": [""],
        "token_indices": [],
        "lookats": [interaction_point.astype(float).tolist()],
        "viewpoints": [stage_viewpoints],
        "interactions": [interaction_label],
    }
    scene_config_root = ensure_dir(output_root / "singleview_scene_config")
    scene_config_path = scene_config_root / f"{interaction_name}_v1.yml"
    with scene_config_path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(to_jsonable(scene_yaml_payload), file_obj, sort_keys=False)

    run_cfg["data.root_dir"] = str(scene_config_root.resolve())
    run_cfg["data.scenes"] = [interaction_name]
    run_cfg["data.cfg_suffix"] = "_v1.yml"
    run_cfg["data.max_views"] = 1
    run_cfg["data.fov"] = fov_y
    run_cfg["vlm.inpaint_dir"] = str(external_root.resolve())
    run_cfg["log_dir"] = str(
        Path(args.log_dir).resolve()
        if args.log_dir
        else (output_root / "genzi_singleview_runs").resolve()
    )
    if args.exp_name:
        run_cfg["exp_time"] = args.exp_name
    run_cfg["gpus"] = [int(args.genzi_gpu_id)]
    run_cfg["seed"] = int(args.seed)

    ensure_alphapose_import_path()
    ensure_smplx_symlink(run_cfg)
    run_genzi_helpers = load_run_genzi_helpers()
    run_genzi_helpers.configure_runtime(args.opengl_platform, args.wandb_mode)
    run_genzi_helpers.install_diffusers_compat_shims()

    from genzi.misc import seeding
    import genzi.generation as generation

    run_genzi_helpers.install_textureless_smplx_obj_export(generation)
    if args.skip_ldm_load:
        generation.get_ldm_inpaint = lambda _path, _device: None

    seeding(int(run_cfg["seed"]))
    generation.cfg = run_cfg
    app = generation.GenZI(run_cfg)
    app.run_scenes()

    summary = {
        "interaction_name": interaction_name,
        "output_root": output_root,
        "mode": "stock_genzi_single_view_external_inpaint",
        "scene_config": scene_config_path,
        "external_inpaint_root": external_root,
        "staged_inpaint": staged_inpaint,
        "log_dir": run_cfg["log_dir"],
        "scene_mesh": scene_paths["mesh_path"],
        "sdf_meta": sdf_meta_path,
        "interaction_point": interaction_point,
        "interaction_point_stats": interaction_point_stats,
        "camera": {
            "eye": eye,
            "up_dir": up_dir,
            "fov_y": fov_y,
            "source_camera": scene_context["camera"],
        },
        "device": str(args.device),
        "physical_device": str(args.physical_device),
        "steps": run_cfg["optim.steps"],
        "loss_inpaint_min_views": run_cfg["loss.inpaint_min_views"],
        "loss_inpaints_per_view": run_cfg["loss.inpaints_per_view"],
    }
    save_json(output_root / "genzi_singleview_summary.json", summary)
    return to_jsonable(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GenZI Ours-Single View baseline on 4DHSI inpainted frames."
    )
    parser.add_argument("--interaction_name", default="interaction_10")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--run_cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output_base", default=None)
    parser.add_argument("--scannet_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--skip_ldm_load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--opengl_platform", default="egl")
    parser.add_argument("--wandb_mode", default="offline")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_cuda_visibility(args)
    cfg = load_cfg(Path(args.run_cfg).resolve())

    preflight_result = None
    if not args.skip_preflight:
        preflight_result = preflight(cfg, strict=not args.preflight_only)
        print(json.dumps(to_jsonable(preflight_result), indent=2))
    if args.preflight_only:
        return

    if args.all_interactions:
        output_base = Path(args.output_base).resolve() if args.output_base else None
        interaction_names, skipped = discover_interactions(output_base=output_base)
        output_root = output_base or DEFAULT_OUTPUT_BASE
        save_json(output_root / "genzi_singleview_skip_report.json", skipped)
    else:
        interaction_names = [args.interaction_name]
        skipped = []

    summaries = []
    failures = []
    for interaction_name in interaction_names:
        started = time.time()
        try:
            print(f"\n[*] Running GenZI single-view baseline for {interaction_name}")
            summary = run_interaction(interaction_name, args, cfg)
            summary["elapsed_s"] = time.time() - started
            summaries.append(summary)
        except Exception as exc:
            failure = {
                "interaction_name": interaction_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[!] Failed {interaction_name}: {failure['error']}")
            if not args.all_interactions:
                raise

    output_base = Path(args.output_base).resolve() if args.output_base else DEFAULT_OUTPUT_BASE
    save_json(
        output_base / "genzi_singleview_batch_summary.json",
        {
            "summaries": summaries,
            "failures": failures,
            "skipped": skipped,
            "preflight": preflight_result,
        },
    )
    print(f"\n[*] Finished. successes={len(summaries)} failures={len(failures)} skipped={len(skipped)}")


if __name__ == "__main__":
    main()
