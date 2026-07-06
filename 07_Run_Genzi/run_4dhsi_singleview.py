from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import os.path as osp
import pickle
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"

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


@dataclass
class Paths:
    generated_root: Path
    input_scene_json: Path
    sig_json: Path
    contact_masks_dir: Path
    contact_canvas_path: Path
    contact_spec: Path
    comparison_mesh_world: Path
    output_root: Path


@dataclass
class Camera:
    intrinsics: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    width: int
    height: int


class CalibratedCameraProjector:
    def __init__(
        self,
        intrinsics: np.ndarray,
        rotation_world_to_camera: np.ndarray,
        translation_world_to_camera: np.ndarray,
        width: int,
        height: int,
        device: torch.device,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.intrinsics = torch.as_tensor(
            intrinsics, dtype=torch.float32, device=device
        )
        self.rotation_world_to_camera = torch.as_tensor(
            rotation_world_to_camera, dtype=torch.float32, device=device
        )
        self.translation_world_to_camera = torch.as_tensor(
            translation_world_to_camera, dtype=torch.float32, device=device
        )

    def project(self, vertices: torch.Tensor, camera_ids: list[int] | None = None) -> torch.Tensor:
        del camera_ids
        cam = vertices @ self.rotation_world_to_camera.t()
        cam = cam + self.translation_world_to_camera[None]
        z = torch.clamp(cam[:, 2], min=1e-6)
        x = self.intrinsics[0, 0] * cam[:, 0] / z + self.intrinsics[0, 2]
        y = self.intrinsics[1, 1] * cam[:, 1] / z + self.intrinsics[1, 2]
        return torch.stack((x, y), dim=1).unsqueeze(0)

    def project_normalized(self, vertices: torch.Tensor) -> torch.Tensor:
        projected = self.project(vertices)
        scale = projected.new_tensor([self.width, self.height]).view(1, 1, 2)
        return projected / scale


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def save_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
        output_base = PROJECT_DIR / "05_Optimize_Static_Scene" / "output_genzi_singleview"
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
        comparison_mesh_world=PROJECT_DIR
        / "05_Optimize_Static_Scene"
        / "output"
        / interaction_name
        / "meshes"
        / "frame_0000_world.ply",
        output_root=output_base / interaction_name,
    )


def required_input_paths(paths: Paths) -> dict[str, Path]:
    return {
        "inpainted_frame": paths.generated_root / "inpainted_frame_resized.png",
        "input_scene_json": paths.input_scene_json,
        "sig_json": paths.sig_json,
        "contact_masks_dir": paths.contact_masks_dir,
        "contact_canvas_path": paths.contact_canvas_path,
        "contact_spec": paths.contact_spec,
        "comparison_mesh_world": paths.comparison_mesh_world,
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


def transform_camera_to_world(points_camera: np.ndarray, camera: Camera) -> np.ndarray:
    return (points_camera - camera.translation_world_to_camera[None]) @ camera.rotation_world_to_camera


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


def opencv_signed_distance_sign() -> float:
    import open3d as o3d

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    sphere.compute_vertex_normals()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(sphere))
    query = o3d.core.Tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=o3d.core.Dtype.Float32)
    distances = scene.compute_signed_distance(query).numpy()
    center, outside = float(distances[0]), float(distances[1])
    if center < 0.0 and outside > 0.0:
        return 1.0
    if center > 0.0 and outside < 0.0:
        return -1.0
    raise RuntimeError(
        "Open3D signed-distance sanity check failed: "
        f"center={center:.6f}, outside={outside:.6f}"
    )


def ensure_scene_sdf(
    scene_id: str,
    mesh_path: Path,
    sdf_dir: Path,
    dim: int,
    padding_m: float,
    force: bool,
) -> Path:
    sdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = sdf_dir / f"{scene_id}.json"
    sdf_path = sdf_dir / f"{scene_id}_sdf.npy"
    if meta_path.exists() and sdf_path.exists() and not force:
        return meta_path

    import open3d as o3d

    print(f"[*] Building SDF for {scene_id} at dim={dim}: {mesh_path}")
    sign = opencv_signed_distance_sign()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to read scene mesh: {mesh_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    bbox_min = vertices.min(axis=0) - float(padding_m)
    bbox_max = vertices.max(axis=0) + float(padding_m)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    xs = np.linspace(bbox_min[0], bbox_max[0], int(dim), dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], int(dim), dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], int(dim), dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    sdf = np.empty((grid.shape[0],), dtype=np.float32)
    chunk_size = 262144
    for start in range(0, grid.shape[0], chunk_size):
        end = min(start + chunk_size, grid.shape[0])
        query = o3d.core.Tensor(grid[start:end], dtype=o3d.core.Dtype.Float32)
        sdf[start:end] = scene.compute_signed_distance(query).numpy() * sign
    sdf = sdf.reshape(int(dim), int(dim), int(dim)).astype(np.float32)
    np.save(sdf_path, sdf)
    save_json(
        meta_path,
        {
            "dim": int(dim),
            "min": bbox_min.astype(float).tolist(),
            "max": bbox_max.astype(float).tolist(),
            "mesh_path": str(mesh_path),
            "signed_distance_sign_multiplier": float(sign),
        },
    )
    return meta_path


def stage_single_view_image(paths: Paths) -> Path:
    src = paths.generated_root / "inpainted_frame_resized.png"
    dst_dir = ensure_dir(paths.output_root / "genzi_inputs" / "stage000_inpaint000")
    dst = dst_dir / "view000.png"
    shutil.copy2(src, dst)
    return dst


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
        "open3d",
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
    steps = list(cfg["optim.steps"][0])
    total = sum(int(x) for x in steps)
    if total <= int(max_steps):
        return
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
    cfg["optim.steps"][0] = scaled.tolist()


def draw_joint_overlay(
    image_rgb: np.ndarray,
    joints2d: np.ndarray,
    projected2d: np.ndarray,
    output_path: Path,
) -> None:
    image = image_rgb[:, :, ::-1].copy()
    for point in joints2d:
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 4, (0, 255, 0), -1)
    for point in projected2d:
        cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 3, (0, 0, 255), -1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def run_interaction(
    interaction_name: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    ensure_alphapose_import_path()
    from genzi.loss import HSILoss
    from genzi.misc import get_rotation_matrix, save_mesh, seeding, to_numpy
    from genzi.optim import LearnableParams, OptimWrapper, SmplxParams
    from genzi.pose2d import (
        Pose2DPipeline,
        smplx_alphapose_limb_corrs,
        smplx_alphapose_torso_corrs,
    )
    from genzi.scene import Scene
    from genzi.vposer_compat import load_vposer_model
    import smplx

    seeding(int(args.seed))
    paths = build_paths(
        interaction_name,
        output_base=Path(args.output_base).resolve() if args.output_base else None,
    )
    for name, path in required_input_paths(paths).items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    output_root = ensure_dir(paths.output_root)
    meshes_dir = ensure_dir(output_root / "meshes")
    debug_dir = ensure_dir(output_root / "debug")
    params_dir = ensure_dir(debug_dir / "params")
    overlays_dir = ensure_dir(debug_dir / "overlays")
    csv_dir = ensure_dir(debug_dir / "csv")

    stage_image_path = stage_single_view_image(paths)
    input_payload = load_json(paths.input_scene_json)
    sig_payload = load_json(paths.sig_json)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(resolve_scannet_root(args.scannet_root), scene_context)
    camera = load_scannet_camera(scene_paths, scene_context)
    contact_camera = load_contact_camera(paths.contact_spec, paths.contact_canvas_path, camera)

    image_rgb = np.array(Image.open(stage_image_path).convert("RGB"))
    if image_rgb.shape[:2] != (camera.height, camera.width):
        raise ValueError(
            "Inpainted frame shape does not match ScanNet++ camera metadata: "
            f"image={image_rgb.shape[1]}x{image_rgb.shape[0]}, "
            f"metadata={camera.width}x{camera.height}"
        )

    scene_vertices_world, _scene_faces = load_mesh(scene_paths["mesh_path"])
    interaction_point, interaction_point_stats = derive_interaction_point(
        sig_payload=sig_payload,
        contact_masks_dir=paths.contact_masks_dir,
        contact_camera=contact_camera,
        scene_vertices_world=scene_vertices_world,
    )

    sdf_meta_path = ensure_scene_sdf(
        scene_id=str(scene_context["scene_id"]),
        mesh_path=scene_paths["mesh_path"],
        sdf_dir=GENZI_ROOT / "data" / "4dhsi_sdf",
        dim=int(args.sdf_dim or cfg["data.sdf_dim"]),
        padding_m=float(args.sdf_padding_m if args.sdf_padding_m is not None else cfg["data.sdf_padding_m"]),
        force=bool(args.force_sdf),
    )

    device = torch.device(args.device)
    vposer, _ = load_vposer_model(
        cfg["vposer.ckpt_path"],
        remove_words_in_model_weights="vp_model.",
        disable_grad=True,
    )
    vposer.to(device)
    vposer.eval()

    smplx_model = smplx.create(
        model_path=cfg["smplx.model_path"],
        model_type=cfg["smplx.model_type"],
        batch_size=cfg["smplx.batch_size"],
        gender=cfg["smplx.gender"],
        num_pca_comps=cfg["smplx.num_pca_comps"],
    )
    smplx_model.to(device)
    smplx_model.requires_grad_(False)

    pose2d = Pose2DPipeline(args_path=cfg["pose2d.args_path"], outputpath=str(debug_dir / "alphapose"))
    pose2d_dict = pose2d(image=image_rgb, im_name="pose2d000.png")
    if len(pose2d_dict["result"]) == 0:
        raise RuntimeError(f"AlphaPose failed to detect a human for {interaction_name}")
    pose_result = pose2d_dict["result"][0]
    joint_scores = pose_result["kp_score"]
    joints = pose_result["keypoints"]
    torso_scores = joint_scores[smplx_alphapose_torso_corrs[:, 1], 0].float().to(device)
    limb_scores = joint_scores[smplx_alphapose_limb_corrs[:, 1], 0].float().to(device)
    num_valid_joints = int(
        torch.sum(torso_scores >= cfg["pose2d.score_thresh"]).item()
        + torch.sum(limb_scores >= cfg["pose2d.score_thresh"]).item()
    )
    if num_valid_joints < int(cfg["pose2d.min_num_joints"]):
        raise RuntimeError(
            f"Only {num_valid_joints} valid AlphaPose joints for {interaction_name}; "
            f"need {cfg['pose2d.min_num_joints']}"
        )

    scale = torch.as_tensor([camera.width, camera.height], dtype=torch.float32, device=device)
    joints_torso_px = joints[smplx_alphapose_torso_corrs[:, 1], :].float().to(device)
    joints_limb_px = joints[smplx_alphapose_limb_corrs[:, 1], :].float().to(device)
    joints2d_torso = (joints_torso_px / scale).unsqueeze(0)
    joints2d_limb = (joints_limb_px / scale).unsqueeze(0)
    joints2d_torso_scores = torso_scores.unsqueeze(0)
    joints2d_limb_scores = limb_scores.unsqueeze(0)

    scene3d = Scene(
        mesh_path=str(scene_paths["mesh_path"]),
        sdf_path=str(sdf_meta_path),
        subd_mesh_path="",
    )
    projector = CalibratedCameraProjector(
        intrinsics=camera.intrinsics,
        rotation_world_to_camera=camera.rotation_world_to_camera,
        translation_world_to_camera=camera.translation_world_to_camera,
        width=camera.width,
        height=camera.height,
        device=device,
    )

    smplx_rots = np.identity(4, dtype=np.float32)
    for idx, axis in enumerate(cfg["smplx.rotation_axes"]):
        smplx_rots = get_rotation_matrix(axis, cfg["smplx.rotation_angles"][idx]) @ smplx_rots
    smplx_params = SmplxParams(
        smplx_model,
        transl=torch.as_tensor(interaction_point, dtype=torch.float32, device=device),
        global_orient=torch.as_tensor(smplx_rots[:3, :3], dtype=torch.float32, device=device),
        use_latent_pose=cfg["smplx.use_latent_pose"],
        use_shape_params=cfg["smplx.use_shape_params"],
        use_continous_rot_repr=cfg["smplx.use_continous_rot_repr"],
    )
    smplx_params.to(device)
    isparams = LearnableParams(init_val=torch.zeros(1).float(), func=torch.sigmoid)
    isparams.to(device)
    loss_fn = HSILoss(cfg=cfg, stage_idx=0).to(device=device)

    optim_args = {
        "optim_steps": cfg["optim.steps"][0],
        "optim_type": cfg["optim.type"],
    }
    optimizers = [
        OptimWrapper(smplx_params.get_transl_params(), lrs=cfg["optim.transl_lrs"][0], name="transl", **optim_args),
        OptimWrapper(smplx_params.get_orient_params(), lrs=cfg["optim.orient_lrs"][0], name="orient", **optim_args),
        OptimWrapper(smplx_params.get_pose_params(), lrs=cfg["optim.pose_lrs"][0], name="pose", **optim_args),
        OptimWrapper(smplx_params.get_shape_params(), lrs=cfg["optim.shape_lrs"][0], name="shape", **optim_args),
        OptimWrapper(list(isparams.parameters()), lrs=cfg["optim.is_lrs"][0], name="is", **optim_args),
    ]

    rows: list[dict[str, Any]] = []
    reproj_first = math.nan
    reproj_last = math.nan
    num_steps = sum(cfg["optim.steps"][0])
    look_at = torch.as_tensor(interaction_point, dtype=torch.float32, device=device)
    for iter_idx in range(num_steps):
        for optimizer in optimizers:
            optimizer.step_lr()
            optimizer.zero_grad()

        sdict = smplx_params(smplx_model, vposer)
        inpaint_scores = isparams()
        torso_proj = projector.project_normalized(
            sdict["joints"][smplx_alphapose_torso_corrs[:, 0], :]
        )
        limb_proj = projector.project_normalized(
            sdict["joints"][smplx_alphapose_limb_corrs[:, 0], :]
        )
        loss_dict = loss_fn(
            iter_idx=iter_idx,
            inpaint_scores=inpaint_scores,
            joints2d_torso=joints2d_torso,
            joints2d_torso_scores=joints2d_torso_scores,
            joints2d_torso_proj=torso_proj,
            joints2d_limb=joints2d_limb,
            joints2d_limb_scores=joints2d_limb_scores,
            joints2d_limb_proj=limb_proj,
            joints3d=sdict["joints"],
            joints3d_init=None,
            body_pose=sdict["body_pose"],
            body_pose_latent=sdict["body_pose_latent"],
            betas=sdict["betas"],
            vertices=sdict["vertices"],
            faces=sdict["faces"],
            transl=sdict["transl"],
            scene3d=scene3d,
            look_at=look_at,
        )
        loss = loss_dict["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at iter {iter_idx}: {loss.item()}")
        loss.backward()
        if cfg["optim.grad_clip"] > 0:
            torch.nn.utils.clip_grad_value_(smplx_params.parameters(), cfg["optim.grad_clip"])
        for optimizer in optimizers:
            optimizer.step_params()

        with torch.no_grad():
            torso_err = torch.norm((torso_proj - joints2d_torso) * scale.view(1, 1, 2), dim=-1)
            limb_err = torch.norm((limb_proj - joints2d_limb) * scale.view(1, 1, 2), dim=-1)
            reproj_error = torch.mean(torch.cat([torso_err.reshape(-1), limb_err.reshape(-1)])).item()
        if iter_idx == 0:
            reproj_first = float(reproj_error)
        reproj_last = float(reproj_error)
        if iter_idx == 0 or iter_idx == num_steps - 1 or (iter_idx + 1) % int(args.log_every) == 0:
            row = {"iter": iter_idx, "reprojection_error_px": float(reproj_error)}
            row.update({key: float(value.detach().cpu().item()) for key, value in loss_dict.items()})
            rows.append(row)
            print(
                f"  iter {iter_idx + 1:04d}/{num_steps}: "
                f"loss={row['loss']:.6f} reproj={reproj_error:.2f}px"
            )

    with torch.no_grad():
        final_dict = smplx_params(smplx_model, vposer)
    smplx_dict = to_numpy(final_dict)
    smplx_dict.update(
        {
            "prompt": input_payload["interaction_context"]["interaction"],
            "scene_name": scene_context["scene_id"],
            "interaction": sig_payload.get("interaction", ""),
            "interaction_name": interaction_name,
            "camera": {
                "intrinsics": camera.intrinsics,
                "rotation_world_to_camera": camera.rotation_world_to_camera,
                "translation_world_to_camera": camera.translation_world_to_camera,
                "width": camera.width,
                "height": camera.height,
            },
            "interaction_point": interaction_point,
        }
    )
    with (output_root / "smplx.pkl").open("wb") as file_obj:
        pickle.dump(smplx_dict, file_obj)
    torch.save(smplx_params.state_dict(), output_root / "params.pth")

    vertices_world = smplx_dict["vertices"].astype(np.float32)
    faces = smplx_dict["faces"].astype(np.int32)
    vertices_camera = transform_world_to_camera(vertices_world, camera).astype(np.float32)
    save_mesh(str(meshes_dir / "frame_0000_world.ply"), vertices_world, faces)
    save_mesh(str(meshes_dir / "frame_0000_camera.ply"), vertices_camera, faces)
    save_mesh(str(output_root / "optim_human.ply"), vertices_world, faces)

    optimized_params_payload = {
        "transl": final_dict["transl"].detach().cpu().reshape(-1),
        "global_orient": final_dict["global_orient"].detach().cpu().reshape(-1),
        "body_pose": final_dict["body_pose"].detach().cpu().reshape(-1),
        "betas": final_dict["betas"].detach().cpu().reshape(-1),
        "scale": torch.tensor(1.0, dtype=torch.float32),
    }
    torch.save(optimized_params_payload, params_dir / "optimized_frame_0000.pt")
    save_csv_rows(csv_dir / "iter_metrics.csv", rows)

    final_projected_px = np.concatenate(
        [
            to_numpy(projector.project(final_dict["joints"][smplx_alphapose_torso_corrs[:, 0], :])[0]),
            to_numpy(projector.project(final_dict["joints"][smplx_alphapose_limb_corrs[:, 0], :])[0]),
        ],
        axis=0,
    )
    detected_px = np.concatenate(
        [to_numpy(joints_torso_px), to_numpy(joints_limb_px)],
        axis=0,
    )
    draw_joint_overlay(
        image_rgb=image_rgb,
        joints2d=detected_px,
        projected2d=final_projected_px,
        output_path=overlays_dir / "frame_0000_pose2d_genzi_projection.png",
    )

    summary = {
        "interaction_name": interaction_name,
        "output_root": output_root,
        "world_mesh": meshes_dir / "frame_0000_world.ply",
        "camera_mesh": meshes_dir / "frame_0000_camera.ply",
        "smplx_pkl": output_root / "smplx.pkl",
        "params_pth": output_root / "params.pth",
        "optimized_params": params_dir / "optimized_frame_0000.pt",
        "stage_image": stage_image_path,
        "scene_mesh": scene_paths["mesh_path"],
        "sdf_meta": sdf_meta_path,
        "num_valid_alphapose_joints": num_valid_joints,
        "reprojection_error_px_first": reproj_first,
        "reprojection_error_px_last": reproj_last,
        "interaction_point": interaction_point,
        "interaction_point_stats": interaction_point_stats,
        "steps": num_steps,
    }
    save_json(output_root / "genzi_singleview_summary.json", summary)
    return to_jsonable(summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GenZI Ours-Single View baseline on 4DHSI inpainted frames."
    )
    parser.add_argument("--interaction_name", default="interaction_10")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--run_cfg", default=str(GENZI_ROOT / "config" / "4dhsi_singleview.yml"))
    parser.add_argument("--output_base", default=None)
    parser.add_argument("--scannet_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sdf_dim", type=int, default=None)
    parser.add_argument("--sdf_padding_m", type=float, default=None)
    parser.add_argument("--force_sdf", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_cfg(Path(args.run_cfg).resolve())
    maybe_limit_steps(cfg, args.max_steps)

    preflight_result = None
    if not args.skip_preflight:
        preflight_result = preflight(cfg, strict=not args.preflight_only)
        print(json.dumps(to_jsonable(preflight_result), indent=2))
    if args.preflight_only:
        return

    if args.all_interactions:
        output_base = Path(args.output_base).resolve() if args.output_base else None
        interaction_names, skipped = discover_interactions(output_base=output_base)
        output_root = output_base or PROJECT_DIR / "05_Optimize_Static_Scene" / "output_genzi_singleview"
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

    output_base = Path(args.output_base).resolve() if args.output_base else PROJECT_DIR / "05_Optimize_Static_Scene" / "output_genzi_singleview"
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
