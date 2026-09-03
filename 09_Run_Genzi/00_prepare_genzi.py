"""Prepare GenZI with a Module-03 crop and the PROX visibility TSDF."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
MODULE_05_OUTPUT = PROJECT_DIR / "05_Optimize_Static_Scene" / "output"
MODULE_08_OUTPUT = PROJECT_DIR / "08_Run_Prox" / "output"
MODULE_03_OUTPUT = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"
MODULE_04_OUTPUT = PROJECT_DIR / "04_Estimate_Human_Pose" / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"
SDF_GRID_DIM = 384
SDF_TRUNCATION_M = 0.20
SDF_NEGATIVE_BAND_M = 0.20
MAX_TSDF_VIEWS = 64
DEPTH_RENDER_WIDTH = 2048
ROI_PADDING_M = 0.25
COVERAGE_RENDER_WIDTH = 320
COVERAGE_SURFACE_SAMPLES = 12_000
COVERAGE_DEPTH_TOLERANCE_M = 0.04
COVERAGE_MIN_VISIBLE_SAMPLES = 12
POSE_DEDUP_TRANSLATION_M = 0.08
POSE_DEDUP_ROTATION_DEG = 4.0
TSDF_RANDOM_SEED = 24017
DEFAULT_PROMPT_PREFIX = "a woman"
DEFAULT_PROMPT_SUFFIX = "wearing a white shirt and blue pants, full body"
DEFAULT_TOKEN_INDICES = "2"
SAM3_ROOT = WORKSPACE_ROOT / "sam3"
DEFAULT_SAM3_BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_GENZI_PYTHON = Path("/root/miniconda3/envs/genzi/bin/python")
DEFAULT_SAM3_PYTHON = Path("/root/miniconda3/envs/sam3/bin/python")
ANCHOR_METHOD = "sam3_visible_object_centroid_v1"
SAM3_CONFIDENCE_THRESHOLD = 0.0
SAM3_LOW_CONFIDENCE_WARNING = 0.5
TSDF_METHOD = "module03_scene_crop_coverage_visibility_tsdf_384_depth2048_pose_dedup_v2"
SCENE_SCOPE = "module03_crop"
SIGN_CONVENTION = (
    "positive is directly observed free space in front of rendered depth; "
    "negative is the symmetric truncation band behind an observed surface; positive "
    "free-space evidence overrides negative evidence"
)
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}

if str(GENZI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZI_ROOT))
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))


@dataclass(frozen=True)
class Camera:
    name: str
    intrinsics: Any
    rotation_world_to_camera: Any
    translation_world_to_camera: Any
    width: int
    height: int


@dataclass
class InteractionPaths:
    input_scene_json: Path
    sig_json: Path
    output_root: Path


def configure_headless_rendering(opengl_platform: str | None) -> None:
    if opengl_platform is None:
        return
    platform = str(opengl_platform).strip()
    if not platform:
        return
    os.environ.setdefault("PYOPENGL_PLATFORM", platform)


def log(message: str) -> None:
    print(message, flush=True)


def parse_token_indices(raw: str | None) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def link_or_copy(source: Path, destination: Path) -> str:
    """Materialize an immutable prepared artifact without duplicating it when possible."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def validate_module08_scene_artifacts(
    interaction_name: str,
    scene_id: str,
) -> dict[str, Any]:
    """Return validated PROX TSDF artifacts and their crop bounds for GenZI."""
    import numpy as np

    root = MODULE_08_OUTPUT / interaction_name
    metadata_path = root / "metadata.json"
    sdf_meta_path = root / "sdf" / f"{scene_id}.json"
    sdf_path = root / "sdf" / f"{scene_id}_sdf.npy"
    observed_path = root / "sdf" / f"{scene_id}_observed.npy"
    required = (metadata_path, sdf_meta_path, sdf_path, observed_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing module-08 crop/TSDF artifact(s): " + "; ".join(missing))

    run_metadata = load_json(metadata_path)
    sdf_metadata = load_json(sdf_meta_path)
    if str(run_metadata.get("scene_id")) != scene_id:
        raise ValueError(
            f"Module-08 scene ID mismatch for {interaction_name}: "
            f"{run_metadata.get('scene_id')!r} versus {scene_id!r}"
        )
    if sdf_metadata.get("method") != TSDF_METHOD:
        raise ValueError(
            f"Module-08 TSDF method mismatch for {interaction_name}: "
            f"{sdf_metadata.get('method')!r}"
        )
    if int(sdf_metadata.get("dim", -1)) != SDF_GRID_DIM:
        raise ValueError(
            f"Module-08 TSDF dimension must be {SDF_GRID_DIM}; "
            f"got {sdf_metadata.get('dim')!r}"
        )
    if not np.isclose(float(sdf_metadata.get("trunc_m", -1.0)), SDF_TRUNCATION_M):
        raise ValueError("Module-08 TSDF truncation does not match the GenZI contract")
    if not np.isclose(
        float(sdf_metadata.get("negative_band_m", -1.0)), SDF_NEGATIVE_BAND_M
    ):
        raise ValueError("Module-08 TSDF negative band does not match the GenZI contract")

    sdf = np.load(sdf_path, mmap_mode="r")
    observed = np.load(observed_path, mmap_mode="r")
    expected_shape = (SDF_GRID_DIM, SDF_GRID_DIM, SDF_GRID_DIM)
    if sdf.shape != expected_shape or observed.shape != expected_shape:
        raise ValueError(
            f"Module-08 TSDF shapes must be {expected_shape}; "
            f"got sdf={sdf.shape}, observed={observed.shape}"
        )
    if sdf.dtype != np.float32:
        raise ValueError(f"Module-08 TSDF must be float32; got {sdf.dtype}")
    bbox_min = np.asarray(sdf_metadata["min"], dtype=np.float32)
    bbox_max = np.asarray(sdf_metadata["max"], dtype=np.float32)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,) or not np.all(bbox_min < bbox_max):
        raise ValueError("Module-08 TSDF has invalid world-space bounds")

    crop_metadata = run_metadata.get("scene_crop")
    if not isinstance(crop_metadata, dict):
        raise ValueError(f"Module-08 metadata has no scene_crop block: {metadata_path}")
    if not np.allclose(crop_metadata.get("bbox_min"), bbox_min, atol=1e-6) or not np.allclose(
        crop_metadata.get("bbox_max"), bbox_max, atol=1e-6
    ):
        raise ValueError("Module-08 crop bounds and TSDF bounds disagree")

    return {
        "root": root,
        "metadata_path": metadata_path,
        "run_metadata": run_metadata,
        "crop_metadata": crop_metadata,
        "sdf_meta_path": sdf_meta_path,
        "sdf_path": sdf_path,
        "observed_path": observed_path,
        "sdf_metadata": sdf_metadata,
    }


def materialize_module08_scene_artifacts(
    artifacts: dict[str, Any],
    output_root: Path,
    scene_id: str,
    full_scene_mesh: Any,
    full_scene_mesh_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    """Reuse PROX's TSDF and create GenZI's colored crop from the original mesh."""
    import numpy as np

    mesh_path = output_root / "scene" / "mesh" / f"{scene_id}.ply"
    bbox_min = np.asarray(artifacts["sdf_metadata"]["min"], dtype=np.float32)
    bbox_max = np.asarray(artifacts["sdf_metadata"]["max"], dtype=np.float32)
    if full_scene_mesh.visual.kind not in {"vertex", "face"}:
        raise ValueError(
            f"Original ScanNet++ mesh has no usable colors: {full_scene_mesh_path}"
        )
    crop_mesh = crop_mesh_to_bounds(full_scene_mesh, bbox_min, bbox_max)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    crop_mesh.export(mesh_path)

    sdf_dir = output_root / "scene" / "sdf"
    sdf_meta_path = sdf_dir / f"{scene_id}.json"
    methods: dict[str, str] = {
        "mesh": "cropped_from_original_scannet_mesh",
        "sdf": link_or_copy(Path(artifacts["sdf_path"]), sdf_dir / f"{scene_id}_sdf.npy"),
        "observed": link_or_copy(
            Path(artifacts["observed_path"]), sdf_dir / f"{scene_id}_observed.npy"
        ),
    }
    diagnostic_names = (
        "free_vote_count",
        "negative_vote_count",
        "surface_vote_count",
        "occluded_vote_count",
    )
    for name in diagnostic_names:
        source = Path(artifacts["root"]) / "sdf" / f"{scene_id}_{name}.npy"
        if source.is_file():
            methods[name] = link_or_copy(source, sdf_dir / source.name)

    metadata = dict(artifacts["sdf_metadata"])
    metadata.update(
        {
            "mesh_path": str(mesh_path.resolve()),
            "observed_mask_path": str((sdf_dir / f"{scene_id}_observed.npy").resolve()),
            "free_vote_count_path": str((sdf_dir / f"{scene_id}_free_vote_count.npy").resolve()),
            "negative_vote_count_path": str((sdf_dir / f"{scene_id}_negative_vote_count.npy").resolve()),
            "surface_vote_count_path": str((sdf_dir / f"{scene_id}_surface_vote_count.npy").resolve()),
            "occluded_vote_count_path": str((sdf_dir / f"{scene_id}_occluded_vote_count.npy").resolve()),
            "scene_scope": SCENE_SCOPE,
            "artifact_source": "module08_prox_tsdf",
            "mesh_artifact_source": "module09_original_scannet_color_crop",
            "original_mesh_path": str(full_scene_mesh_path.resolve()),
            "source_metadata_path": str(Path(artifacts["metadata_path"]).resolve()),
            "source_sdf_meta_path": str(Path(artifacts["sdf_meta_path"]).resolve()),
        }
    )
    save_json(sdf_meta_path, metadata)
    provenance = {
        "mode": "module08_prox_tsdf_with_module09_color_crop",
        "source_root": Path(artifacts["root"]),
        "source_metadata": Path(artifacts["metadata_path"]),
        "source_mesh": full_scene_mesh_path,
        "source_sdf_meta": Path(artifacts["sdf_meta_path"]),
        "source_sdf": Path(artifacts["sdf_path"]),
        "crop_bounds_source": "module08_tsdf_metadata",
        "crop_bbox_min": bbox_min,
        "crop_bbox_max": bbox_max,
        "crop_mesh_vertices": int(len(crop_mesh.vertices)),
        "crop_mesh_faces": int(len(crop_mesh.faces)),
        "crop_mesh_visual_kind": crop_mesh.visual.kind,
        "materialization": methods,
    }
    return mesh_path, sdf_meta_path, metadata, provenance


def interaction_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"interaction_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def discover_module05_interactions() -> list[str]:
    """Return interactions with module 05's final completion artifact."""
    names = sorted(
        (
            path.parent.name
            for path in MODULE_05_OUTPUT.glob("interaction_*/alignment_summary.json")
        ),
        key=interaction_sort_key,
    )
    if not names:
        raise RuntimeError(
            "No processed interactions were found under module 05 output: "
            f"{MODULE_05_OUTPUT}"
        )
    return names


def resolve_input_scene_json(interaction_name: str) -> Path:
    prompt_root = PROJECT_DIR / "01_Generate_SIG" / "input_prompts"
    exact = prompt_root / interaction_name / "input_scene.json"
    if exact.exists():
        return exact
    matches = sorted(prompt_root.glob(f"{interaction_name}_*/input_scene.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No input_scene.json found for {interaction_name}")
    raise RuntimeError(
        f"Ambiguous input_scene.json for {interaction_name}: "
        + ", ".join(str(path) for path in matches)
    )


def build_interaction_paths(interaction_name: str, output_base: Path) -> InteractionPaths:
    return InteractionPaths(
        input_scene_json=resolve_input_scene_json(interaction_name),
        sig_json=PROJECT_DIR / "01_Generate_SIG" / "output" / interaction_name / "sig.json",
        output_root=output_base / interaction_name,
    )


def resolve_sig_target(sig_payload: dict[str, Any]) -> tuple[str, str]:
    targets = sig_payload.get("target_objects")
    if isinstance(targets, list) and targets and isinstance(targets[0], dict):
        target_id = str(targets[0].get("id") or "target_object_1")
        label = str(targets[0].get("label") or "").strip()
    else:
        target = sig_payload.get("target_object")
        if not isinstance(target, dict):
            raise ValueError("SIG does not define a target object.")
        target_id = "target_object_1"
        label = str(target.get("label") or "").strip()
    if not label:
        raise ValueError("The primary SIG target object has no label.")
    return target_id, label


def resolve_scannet_root(raw_root: str | None) -> Path:
    return Path(raw_root).resolve() if raw_root else (WORKSPACE_ROOT / "Scannet++" / "data").resolve()


def resolve_scene_paths(scannet_root: Path, scene_context: dict[str, Any]) -> dict[str, Path]:
    scene_id = str(scene_context["scene_id"])
    camera_payload = scene_context["camera"]
    source = str(camera_payload["source"])
    if source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(f"Unsupported camera source {source!r}")
    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[source]
    scene_root = scannet_root / scene_id
    return {
        "image_path": scene_root / image_rel / str(camera_payload["name"]),
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
    }


def build_pinhole_intrinsics(payload: dict[str, Any]) -> tuple[Any, int, int]:
    import numpy as np

    width = int(payload["w"])
    height = int(payload["h"])
    matrix = np.asarray(
        [
            [float(payload["fl_x"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fl_y"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return matrix, width, height


def colmap_qvec_to_rotmat(qvec: Any) -> Any:
    import numpy as np

    qw, qx, qy, qz = np.asarray(qvec, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def load_colmap_pose(path: Path, camera_name: str) -> tuple[Any, Any]:
    import numpy as np

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        rotation = colmap_qvec_to_rotmat(np.asarray(parts[1:5], dtype=np.float32))
        translation = np.asarray(parts[5:8], dtype=np.float32)
        return rotation, translation
    raise ValueError(f"Camera {camera_name!r} is absent from {path}")


def load_scannet_camera(scene_paths: dict[str, Path], scene_context: dict[str, Any]) -> Camera:
    intrinsics, width, height = build_pinhole_intrinsics(load_json(scene_paths["transforms_path"]))
    rotation, translation = load_colmap_pose(
        scene_paths["colmap_images_path"], str(scene_context["camera"]["name"])
    )
    return Camera(
        name=str(scene_context["camera"]["name"]),
        intrinsics=intrinsics,
        rotation_world_to_camera=rotation,
        translation_world_to_camera=translation,
        width=width,
        height=height,
    )


def load_colmap_cameras(
    path: Path,
    intrinsics: Any,
    width: int,
    height: int,
) -> list[Camera]:
    import numpy as np

    cameras: list[Camera] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10 or parts[0].startswith("#"):
            continue
        try:
            int(parts[0])
            qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float32)
            translation = np.asarray(
                [float(value) for value in parts[5:8]], dtype=np.float32
            )
            int(parts[8])
        except ValueError:
            continue
        cameras.append(
            Camera(
                name=parts[9],
                intrinsics=np.asarray(intrinsics, dtype=np.float32).copy(),
                rotation_world_to_camera=colmap_qvec_to_rotmat(qvec),
                translation_world_to_camera=translation,
                width=width,
                height=height,
            )
        )
    if not cameras:
        raise ValueError(f"No COLMAP image poses were found in {path}.")
    return cameras


def camera_to_world(points_camera: Any, camera: Camera) -> Any:
    import numpy as np

    points = np.asarray(points_camera, dtype=np.float32)
    return (
        points - camera.translation_world_to_camera.reshape(1, 3)
    ) @ camera.rotation_world_to_camera


def world_to_camera(points_world: Any, camera: Camera) -> Any:
    import numpy as np

    points = np.asarray(points_world, dtype=np.float32)
    return (
        points @ camera.rotation_world_to_camera.T
        + camera.translation_world_to_camera.reshape(1, 3)
    )


def camera_to_world_transform(camera: Camera) -> Any:
    import numpy as np

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = camera.rotation_world_to_camera.T
    transform[:3, 3] = -(
        camera.rotation_world_to_camera.T @ camera.translation_world_to_camera
    )
    return transform


def scaled_camera(camera: Camera, width: int) -> Camera:
    import numpy as np

    height = max(1, round(width * camera.height / camera.width))
    scale_x = float(width) / float(camera.width)
    scale_y = float(height) / float(camera.height)
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float32).copy()
    intrinsics[0, :] *= scale_x
    intrinsics[1, :] *= scale_y
    intrinsics[2, :] = [0.0, 0.0, 1.0]
    return Camera(
        name=camera.name,
        intrinsics=intrinsics,
        rotation_world_to_camera=camera.rotation_world_to_camera,
        translation_world_to_camera=camera.translation_world_to_camera,
        width=width,
        height=height,
    )


def pyrender_camera_pose(camera: Camera) -> Any:
    import numpy as np

    camera_to_world_cv = camera_to_world_transform(camera).astype(np.float64)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return camera_to_world_cv @ cv_to_gl


def render_scene_depths(
    mesh: Any,
    cameras: list[Camera],
    debug_dir: Path,
) -> tuple[list[Camera], Any]:
    import numpy as np
    import pyrender

    render_cameras = [scaled_camera(camera, DEPTH_RENDER_WIDTH) for camera in cameras]
    height = render_cameras[0].height
    debug_dir.mkdir(parents=True, exist_ok=True)
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.8, 0.8, 0.8]
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    renderer = pyrender.OffscreenRenderer(
        viewport_width=DEPTH_RENDER_WIDTH,
        viewport_height=height,
    )
    depths: list[Any] = []
    view_manifest: list[dict[str, Any]] = []
    try:
        for index, camera in enumerate(render_cameras, start=1):
            intrinsics = camera.intrinsics
            projection = pyrender.IntrinsicsCamera(
                fx=float(intrinsics[0, 0]),
                fy=float(intrinsics[1, 1]),
                cx=float(intrinsics[0, 2]),
                cy=float(intrinsics[1, 2]),
                znear=0.05,
                zfar=20.0,
            )
            node = scene.add(projection, pose=pyrender_camera_pose(camera))
            try:
                depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            finally:
                scene.remove_node(node)
            depth = np.asarray(depth, dtype=np.float32)
            depths.append(depth)
            depth_path = debug_dir / f"{index:03d}_{camera.name}_depth_m.npy"
            np.save(depth_path, depth)
            view_manifest.append(
                {
                    "index": index,
                    "camera_name": camera.name,
                    "depth_m_npy": depth_path,
                    "valid_depth_pixels": int(
                        np.count_nonzero(np.isfinite(depth) & (depth > 0.0))
                    ),
                    "intrinsics": camera.intrinsics,
                    "rotation_world_to_camera": camera.rotation_world_to_camera,
                    "translation_world_to_camera": camera.translation_world_to_camera,
                }
            )
            log(f"    rendered TSDF view {index}/{len(render_cameras)}: {camera.name}")
    finally:
        renderer.delete()
    save_json(
        debug_dir / "manifest.json",
        {
            "description": "Exact full-scene mesh depth maps used for TSDF fusion",
            "render_width": DEPTH_RENDER_WIDTH,
            "num_views": len(render_cameras),
            "views": view_manifest,
        },
    )
    return render_cameras, np.stack(depths, axis=0)


def load_mesh(path: Path) -> tuple[Any, Any, Any]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    return mesh, np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def load_cfg(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from genzi.misc import omegaconf_to_dotdict

    cfg = OmegaConf.load(path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict)

    def absolutize(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if value.startswith("./"):
            return str((GENZI_ROOT / value[2:]).resolve())
        return value

    flat = omegaconf_to_dotdict(OmegaConf.create(cfg))
    for key, value in list(flat.items()):
        if key.endswith("_path") or key in {"path_prefix", "log_dir"}:
            flat[key] = absolutize(value)
    flat["run_cfg"] = str(path)
    return flat


def default_render_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bg_color": [0.5, 0.5, 0.5, 0.0],
        "ambient_light": [0.0, 0.0, 0.0],
        "dir_light_color": [1.0, 1.0, 1.0],
        "dir_light_intensity": float(args.dir_light_intensity),
        "pt_light_color": [1.0, 1.0, 1.0],
        "pt_light_intensity": float(args.pt_light_intensity),
        "pt_light_position": [0.0, 0.0, 20.0],
        "normal_pbr": True,
        "no_lighting": False,
        "all_solid": False,
        "cull_faces": False,
        "shadows": False,
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_jsonable(payload), sort_keys=False), encoding="utf-8")


def build_sam3_processor(args: argparse.Namespace) -> Any:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    checkpoint = str(Path(args.sam3_checkpoint).resolve()) if args.sam3_checkpoint else None
    bpe_path = str(Path(args.sam3_bpe_path).resolve())
    processor_device = str(args.sam3_device)
    # SAM3's builder only transfers weights for the literal value "cuda".
    # CUDA_VISIBLE_DEVICES maps each worker's physical GPU to logical cuda:0.
    builder_device = "cuda" if processor_device.startswith("cuda:") else processor_device
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint,
        device=builder_device,
        load_from_HF=not args.no_sam3_hf_download,
    )
    return Sam3Processor(
        model=model,
        device=processor_device,
        confidence_threshold=SAM3_CONFIDENCE_THRESHOLD,
    )


def sam3_predictions_from_state(state: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    import numpy as np

    predictions = []
    for index in range(int(state["masks"].shape[0])):
        mask = state["masks"][index, 0].detach().cpu().numpy().astype(bool)
        if not mask.any():
            continue
        predictions.append(
            {
                "candidate": int(index),
                "prompt": prompt,
                "mask": mask,
                "bbox_xyxy": [float(value) for value in state["boxes"][index].detach().cpu().tolist()],
                "score": float(state["scores"][index].detach().cpu().item()),
                "area_pixels": int(np.count_nonzero(mask)),
            }
        )
    return predictions


def run_sam3(
    processor: Any,
    image: Any,
    prompt: str,
    target_box: list[float] | None,
) -> list[dict[str, Any]]:
    state = processor.set_image(image)
    if target_box is None:
        state = processor.set_text_prompt(prompt=prompt, state=state)
    else:
        x0, y0, x1, y1 = target_box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError(
                f"--target-box must lie inside the {image.width}x{image.height} image; got {target_box}"
            )
        normalized_cxcywh = [
            ((x0 + x1) * 0.5) / image.width,
            ((y0 + y1) * 0.5) / image.height,
            (x1 - x0) / image.width,
            (y1 - y0) / image.height,
        ]
        state = processor.add_geometric_prompt(
            state=state,
            box=normalized_cxcywh,
            label=True,
        )
    return sam3_predictions_from_state(state, prompt)


def save_sam3_prediction_archive(path: Path, predictions: list[dict[str, Any]]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if predictions:
        masks = np.stack([item["mask"] for item in predictions], axis=0).astype(bool)
        boxes = np.asarray([item["bbox_xyxy"] for item in predictions], dtype=np.float32)
        scores = np.asarray([item["score"] for item in predictions], dtype=np.float32)
    else:
        masks = np.zeros((0, 0, 0), dtype=bool)
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
    np.savez_compressed(path, masks=masks, boxes_xyxy=boxes, scores=scores)


def load_sam3_prediction_archive(path: Path, prompt: str) -> list[dict[str, Any]]:
    import numpy as np

    archive = np.load(path)
    predictions = []
    for index in range(int(archive["scores"].shape[0])):
        mask = archive["masks"][index].astype(bool)
        predictions.append(
            {
                "candidate": index,
                "prompt": prompt,
                "mask": mask,
                "bbox_xyxy": [float(value) for value in archive["boxes_xyxy"][index].tolist()],
                "score": float(archive["scores"][index]),
                "area_pixels": int(np.count_nonzero(mask)),
            }
        )
    return predictions


def run_sam3_subprocess(
    args: argparse.Namespace,
    image_path: Path,
    prompt: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    command = [
        str(Path(args.sam3_python).resolve()),
        str(Path(__file__).resolve()),
        "--_sam3-worker",
        "--_sam3-image",
        str(image_path.resolve()),
        "--_sam3-output",
        str(output_path.resolve()),
        "--target-prompt",
        prompt,
        "--sam3-bpe-path",
        str(Path(args.sam3_bpe_path).resolve()),
        "--sam3-device",
        str(args.sam3_device),
    ]
    if args.sam3_checkpoint:
        command.extend(["--sam3-checkpoint", str(Path(args.sam3_checkpoint).resolve())])
    if args.no_sam3_hf_download:
        command.append("--no-sam3-hf-download")
    if args.target_box is not None:
        command.extend(["--target-box", *(str(value) for value in args.target_box)])
    log(f"  running SAM 3 in {args.sam3_python}")
    subprocess.run(command, check=True, cwd=str(WORKSPACE_ROOT))
    return load_sam3_prediction_archive(output_path, prompt)


def save_sam3_candidates(image: Any, predictions: list[dict[str, Any]], path: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    base = np.asarray(image.convert("RGB"), dtype=np.uint8)
    colors = [
        (255, 60, 60), (40, 200, 70), (60, 120, 255), (255, 180, 30),
        (190, 70, 240), (40, 210, 210), (255, 90, 180), (160, 210, 40),
    ]
    overlay = base.astype(np.float32)
    for index, prediction in enumerate(predictions):
        color = np.asarray(colors[index % len(colors)], dtype=np.float32)
        mask = prediction["mask"]
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
    canvas = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for index, prediction in enumerate(predictions):
        color = colors[index % len(colors)]
        x0, y0, x1, y1 = prediction["bbox_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
        draw.text((x0 + 4, max(0, y0 - 18)), f"{index}: {prediction['score']:.3f}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def select_sam3_candidate(
    predictions: list[dict[str, Any]],
    requested_index: int | None,
) -> tuple[int, dict[str, Any]]:
    if not predictions:
        raise RuntimeError("SAM 3 returned no target-object masks.")
    ranked = sorted(enumerate(predictions), key=lambda item: item[1]["score"], reverse=True)
    if requested_index is not None:
        if requested_index < 0 or requested_index >= len(predictions):
            raise ValueError(
                f"--sam3-candidate={requested_index} is invalid; available indices are 0..{len(predictions)-1}"
            )
        return requested_index, predictions[requested_index]
    return ranked[0]


def largest_connected_component(mask: Any) -> Any:
    import cv2
    import numpy as np

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        raise RuntimeError("Selected SAM 3 mask is empty.")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def render_source_depth(mesh: Any, camera: Camera) -> Any:
    import numpy as np
    import pyrender

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.0, 0.0, 0.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64)
    pyrender_camera = pyrender.IntrinsicsCamera(
        fx=float(intrinsics[0, 0]), fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]), cy=float(intrinsics[1, 2]),
        znear=0.05, zfar=100.0,
    )
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float64)
    camera_to_world_cv = np.eye(4, dtype=np.float64)
    camera_to_world_cv[:3, :3] = rotation.T
    camera_to_world_cv[:3, 3] = -(rotation.T @ translation)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    scene.add(pyrender_camera, pose=camera_to_world_cv @ cv_to_gl)
    renderer = pyrender.OffscreenRenderer(viewport_width=camera.width, viewport_height=camera.height)
    try:
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        renderer.delete()
    return np.asarray(depth, dtype=np.float32)


def unproject_depth_pixel(camera: Camera, pixel_xy: tuple[int, int], depth: float) -> Any:
    import numpy as np

    u, v = pixel_xy
    ray = np.linalg.inv(np.asarray(camera.intrinsics, dtype=np.float64)) @ np.asarray([u, v, 1.0])
    point_camera = ray * float(depth)
    rotation = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
    translation = np.asarray(camera.translation_world_to_camera, dtype=np.float64)
    return (rotation.T @ (point_camera - translation)).astype(np.float32)


def save_anchor_overlay(image: Any, mask: Any, pixel_xy: tuple[int, int], path: Path) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    array = np.asarray(image.convert("RGB"), dtype=np.uint8).astype(np.float32)
    array[mask] = 0.55 * array[mask] + 0.45 * np.asarray([30, 220, 80], dtype=np.float32)
    canvas = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    x, y = pixel_xy
    radius = max(7, round(min(image.size) * 0.008))
    draw.line((x - radius, y, x + radius, y), fill=(255, 30, 30), width=4)
    draw.line((x, y - radius, x, y + radius), fill=(255, 30, 30), width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def derive_sam3_object_anchor(
    image_path: Path,
    mesh: Any,
    camera: Camera,
    target_id: str,
    target_label: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, dict[str, Any]]:
    import cv2
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    if image.size != (camera.width, camera.height):
        raise ValueError(
            f"Scene image size {image.size} does not match camera metadata {(camera.width, camera.height)}"
        )
    prompt = str(args.target_prompt or target_label).strip()
    prediction_archive = output_dir / "sam3_candidates.npz"
    predictions = run_sam3_subprocess(args, image_path, prompt, prediction_archive)
    candidate_path = output_dir / "sam3_candidates.png"
    save_sam3_candidates(image, predictions, candidate_path)
    selected_index, selected = select_sam3_candidate(predictions, args.sam3_candidate)
    low_confidence = float(selected["score"]) < SAM3_LOW_CONFIDENCE_WARNING
    if low_confidence:
        log(
            "  warning: highest-scoring SAM 3 mask has low confidence "
            f"({float(selected['score']):.3f} < {SAM3_LOW_CONFIDENCE_WARNING:.3f})"
        )
    object_mask = largest_connected_component(selected["mask"])
    mask = object_mask.copy()
    ys, xs = np.nonzero(object_mask)
    bbox_short_side = max(1, min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)))
    erosion_pixels = int(args.anchor_mask_erode_pixels)
    if erosion_pixels < 0:
        erosion_pixels = max(2, min(8, round(bbox_short_side * 0.01)))
    if erosion_pixels > 0:
        size = 2 * erosion_pixels + 1
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
        if eroded.any():
            mask = eroded

    depth = render_source_depth(mesh, camera)
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    if int(valid.sum()) < int(args.anchor_min_depth_pixels):
        raise RuntimeError(
            f"Only {int(valid.sum())} selected-object pixels have reconstructed scene depth; "
            f"minimum is {args.anchor_min_depth_pixels}."
        )
    valid_y, valid_x = np.nonzero(valid)
    centroid_x = float(valid_x.mean())
    centroid_y = float(valid_y.mean())
    nearest = int(np.argmin((valid_x - centroid_x) ** 2 + (valid_y - centroid_y) ** 2))
    pixel_xy = (int(valid_x[nearest]), int(valid_y[nearest]))
    anchor = unproject_depth_pixel(camera, pixel_xy, float(depth[pixel_xy[1], pixel_xy[0]]))

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray((object_mask.astype(np.uint8) * 255)).save(output_dir / "target_mask.png")
    Image.fromarray((mask.astype(np.uint8) * 255)).save(output_dir / "anchor_interior_mask.png")
    save_anchor_overlay(image, object_mask, pixel_xy, output_dir / "target_overlay.png")
    metadata = {
        "method": ANCHOR_METHOD,
        "sig_target_id": target_id,
        "sig_target_label": target_label,
        "sam3_prompt": prompt,
        "sam3_selection_mode": "box" if args.target_box is not None else "text",
        "sam3_candidate_policy": (
            "explicit_candidate" if args.sam3_candidate is not None else "highest_confidence"
        ),
        "sam3_confidence_filter": SAM3_CONFIDENCE_THRESHOLD,
        "sam3_low_confidence_warning_threshold": SAM3_LOW_CONFIDENCE_WARNING,
        "sam3_low_confidence": low_confidence,
        "selected_candidate": int(selected_index),
        "sam3_score": float(selected["score"]),
        "sam3_bbox_xyxy": selected["bbox_xyxy"],
        "candidate_count": len(predictions),
        "object_mask_pixels": int(object_mask.sum()),
        "anchor_interior_mask_pixels": int(mask.sum()),
        "valid_depth_pixels": int(valid.sum()),
        "erosion_pixels": erosion_pixels,
        "anchor_pixel_xy": list(pixel_xy),
        "anchor_depth_m": float(depth[pixel_xy[1], pixel_xy[0]]),
        "anchor_world_xyz": anchor,
        "source_image": image_path,
        "target_mask": output_dir / "target_mask.png",
        "anchor_interior_mask": output_dir / "anchor_interior_mask.png",
        "target_overlay": output_dir / "target_overlay.png",
        "candidate_overlay": candidate_path,
        "candidate_archive": prediction_archive,
        "uses_contact_information": False,
    }
    save_json(output_dir / "anchor.json", metadata)
    return anchor, metadata


def load_module03_crop(
    interaction_name: str,
    source_intrinsics: Any,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    interaction_root = MODULE_03_OUTPUT / interaction_name
    spec_path = interaction_root / "contact_spec.json"
    image_path = interaction_root / "assets" / "target_scene_crop.png"
    if not spec_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(
            f"Missing Module 03 crop inputs for {interaction_name}: "
            f"{spec_path}, {image_path}"
        )
    payload = load_json(spec_path)
    crop_intrinsics = np.asarray(
        payload["camera"]["intrinsics_3x3"], dtype=np.float32
    )
    with Image.open(image_path) as image:
        crop_width, crop_height = image.size
    source_intrinsics = np.asarray(source_intrinsics, dtype=np.float32)
    crop_x0 = float(source_intrinsics[0, 2] - crop_intrinsics[0, 2])
    crop_y0 = float(source_intrinsics[1, 2] - crop_intrinsics[1, 2])
    return {
        "spec_path": spec_path,
        "image_path": image_path,
        "intrinsics": crop_intrinsics,
        "width": int(crop_width),
        "height": int(crop_height),
        "xyxy_source_pixels": [
            crop_x0,
            crop_y0,
            crop_x0 + float(crop_width),
            crop_y0 + float(crop_height),
        ],
    }


def unproject_depth_crop(
    depth: Any,
    render_camera: Camera,
    crop_xyxy_source: list[float],
    source_camera: Camera,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    scale_x = float(render_camera.width) / float(source_camera.width)
    scale_y = float(render_camera.height) / float(source_camera.height)
    x0, y0, x1, y1 = crop_xyxy_source
    rx0 = max(0, int(np.floor(x0 * scale_x)))
    ry0 = max(0, int(np.floor(y0 * scale_y)))
    rx1 = min(render_camera.width, int(np.ceil(x1 * scale_x)))
    ry1 = min(render_camera.height, int(np.ceil(y1 * scale_y)))
    if rx1 <= rx0 or ry1 <= ry0:
        raise RuntimeError("The Module 03 crop does not overlap the rendered source view.")

    crop_depth = np.asarray(depth[ry0:ry1, rx0:rx1], dtype=np.float32)
    valid_y, valid_x = np.nonzero(np.isfinite(crop_depth) & (crop_depth > 0.0))
    if valid_x.size == 0:
        raise RuntimeError("The Module 03 crop contains no rendered ScanNet++ surface.")
    u = valid_x.astype(np.float32) + float(rx0)
    v = valid_y.astype(np.float32) + float(ry0)
    z = crop_depth[valid_y, valid_x]
    intrinsics = render_camera.intrinsics
    points_camera = np.stack(
        [
            (u - intrinsics[0, 2]) / intrinsics[0, 0] * z,
            (v - intrinsics[1, 2]) / intrinsics[1, 1] * z,
            z,
        ],
        axis=1,
    )
    points_world = camera_to_world(points_camera, render_camera)
    return points_world.astype(np.float32), {
        "render_crop_xyxy": [rx0, ry0, rx1, ry1],
        "valid_depth_pixels": int(points_world.shape[0]),
        "depth_min_m": float(z.min()),
        "depth_max_m": float(z.max()),
        "depth_median_m": float(np.median(z)),
    }


def crop_mesh_to_bounds(mesh: Any, bbox_min: Any, bbox_max: Any) -> Any:
    import numpy as np
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    inside = np.all(
        (vertices >= np.asarray(bbox_min)[None])
        & (vertices <= np.asarray(bbox_max)[None]),
        axis=1,
    )
    keep_faces = np.any(inside[faces], axis=1)
    if not np.any(keep_faces):
        raise RuntimeError("The Module 03 crop volume contains no scene faces.")
    used_vertices, inverse = np.unique(
        faces[keep_faces].reshape(-1), return_inverse=True
    )
    crop_kwargs: dict[str, Any] = {}
    visual_kind = mesh.visual.kind
    if visual_kind == "vertex":
        vertex_colors = np.asarray(mesh.visual.vertex_colors)
        if vertex_colors.shape[0] != vertices.shape[0]:
            raise ValueError(
                "The original scene mesh vertex-color count does not match its vertices."
            )
        crop_kwargs["vertex_colors"] = vertex_colors[used_vertices].copy()
    elif visual_kind == "face":
        face_colors = np.asarray(mesh.visual.face_colors)
        if face_colors.shape[0] != faces.shape[0]:
            raise ValueError(
                "The original scene mesh face-color count does not match its faces."
            )
        crop_kwargs["face_colors"] = face_colors[keep_faces].copy()

    crop_mesh = trimesh.Trimesh(
        vertices=vertices[used_vertices],
        faces=inverse.reshape(-1, 3),
        process=False,
        **crop_kwargs,
    )
    if visual_kind in {"vertex", "face"} and crop_mesh.visual.kind != visual_kind:
        raise RuntimeError(
            f"Scene crop lost its {visual_kind} colors during mesh construction."
        )
    return crop_mesh


def sample_mesh_surface(mesh: Any, count: int, seed: int) -> Any:
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = np.isfinite(areas) & (areas > 1e-10)
    triangles = triangles[valid]
    areas = areas[valid]
    if triangles.shape[0] == 0:
        raise RuntimeError("The crop mesh has no non-degenerate triangles.")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(triangles.shape[0], size=int(count), p=areas / areas.sum())
    tri = triangles[chosen]
    r1 = rng.random(int(count), dtype=np.float32)
    r2 = rng.random(int(count), dtype=np.float32)
    sr1 = np.sqrt(r1)
    return (
        (1.0 - sr1)[:, None] * tri[:, 0]
        + (sr1 * (1.0 - r2))[:, None] * tri[:, 1]
        + (sr1 * r2)[:, None] * tri[:, 2]
    ).astype(np.float32)


def camera_center(camera: Camera) -> Any:
    return -camera.rotation_world_to_camera.T @ camera.translation_world_to_camera


def projected_crop_samples(
    camera: Camera,
    samples: Any,
) -> tuple[Any, Any, Any, Any]:
    import numpy as np

    points_camera = world_to_camera(samples, camera)
    z = points_camera[:, 2]
    safe_z = np.maximum(z, 1e-6)
    intrinsics = camera.intrinsics
    u = intrinsics[0, 0] * points_camera[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points_camera[:, 1] / safe_z + intrinsics[1, 2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    projected = (
        (z > 0.05)
        & (z < 20.0)
        & (ui >= 0)
        & (ui < camera.width)
        & (vi >= 0)
        & (vi < camera.height)
    )
    return z, ui, vi, projected


def deduplicate_camera_poses(
    cameras: list[Camera],
    projected_counts: list[int],
    source_camera: Camera,
) -> tuple[list[int], dict[str, Any]]:
    import numpy as np

    if len(cameras) != len(projected_counts):
        raise ValueError("Camera and projected-count lists must have equal length.")
    if not cameras:
        return [], {
            "method": "translation_and_full_rotation_greedy_v1",
            "translation_threshold_m": POSE_DEDUP_TRANSLATION_M,
            "rotation_threshold_deg": POSE_DEDUP_ROTATION_DEG,
            "input_camera_count": 0,
            "retained_camera_count": 0,
            "removed_camera_count": 0,
        }

    centers = np.stack([camera_center(camera) for camera in cameras]).astype(np.float32)
    rotations = np.stack(
        [np.asarray(camera.rotation_world_to_camera, dtype=np.float32) for camera in cameras]
    )
    source_index = next(
        (index for index, camera in enumerate(cameras) if camera.name == source_camera.name),
        None,
    )
    order = sorted(
        range(len(cameras)), key=lambda index: (-int(projected_counts[index]), index)
    )
    if source_index is not None:
        order.remove(source_index)
        order.insert(0, source_index)

    retained: list[int] = []
    rotation_threshold_rad = np.deg2rad(POSE_DEDUP_ROTATION_DEG)
    for index in order:
        if not retained:
            retained.append(index)
            continue
        retained_array = np.asarray(retained, dtype=np.int64)
        translations = np.linalg.norm(
            centers[retained_array] - centers[index][None], axis=1
        )
        traces = np.einsum("nij,ij->n", rotations[retained_array], rotations[index])
        rotation_angles = np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0))
        is_duplicate = np.any(
            (translations <= POSE_DEDUP_TRANSLATION_M)
            & (rotation_angles <= rotation_threshold_rad)
        )
        if not is_duplicate:
            retained.append(index)
    retained.sort()
    return retained, {
        "method": "translation_and_full_rotation_greedy_v1",
        "translation_threshold_m": POSE_DEDUP_TRANSLATION_M,
        "rotation_threshold_deg": POSE_DEDUP_ROTATION_DEG,
        "representative_priority": (
            "source camera first, then descending projected crop-sample count"
        ),
        "input_camera_count": len(cameras),
        "retained_camera_count": len(retained),
        "removed_camera_count": len(cameras) - len(retained),
    }


def select_cameras_by_surface_coverage(
    full_scene_mesh: Any,
    crop_mesh: Any,
    cameras: list[Camera],
    source_camera: Camera,
    roi_center: Any,
    debug_dir: Path,
) -> tuple[list[Camera], dict[str, Any]]:
    import numpy as np
    import pyrender

    samples = sample_mesh_surface(
        crop_mesh, COVERAGE_SURFACE_SAMPLES, TSDF_RANDOM_SEED
    )
    scaled = [scaled_camera(camera, COVERAGE_RENDER_WIDTH) for camera in cameras]
    projected_counts_all: list[int] = []
    projection_candidate_indices: list[int] = []
    for index, camera in enumerate(scaled):
        _, _, _, projected = projected_crop_samples(camera, samples)
        projected_count = int(projected.sum())
        projected_counts_all.append(projected_count)
        if projected_count >= COVERAGE_MIN_VISIBLE_SAMPLES:
            projection_candidate_indices.append(index)

    projection_cameras = [cameras[index] for index in projection_candidate_indices]
    projection_counts = [projected_counts_all[index] for index in projection_candidate_indices]
    retained_local_indices, deduplication = deduplicate_camera_poses(
        projection_cameras, projection_counts, source_camera
    )
    prepass_indices = [
        projection_candidate_indices[index] for index in retained_local_indices
    ]
    deduplication.update(
        {
            "registered_camera_count": len(cameras),
            "projection_candidate_camera_count": len(projection_candidate_indices),
            "retained_cameras": [cameras[index].name for index in prepass_indices],
        }
    )
    save_json(debug_dir / "camera_pose_dedup.json", deduplication)
    if not prepass_indices:
        raise RuntimeError("No registered camera projects the Module 03 scene crop.")

    height = scaled[prepass_indices[0]].height
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    scene.add(pyrender.Mesh.from_trimesh(full_scene_mesh, smooth=False))
    renderer = pyrender.OffscreenRenderer(
        viewport_width=COVERAGE_RENDER_WIDTH, viewport_height=height
    )
    eligible_cameras: list[Camera] = []
    visibility_rows: list[Any] = []
    projected_counts: list[int] = []
    visible_counts: list[int] = []
    try:
        for camera_index in prepass_indices:
            original = cameras[camera_index]
            camera = scaled[camera_index]
            z, ui, vi, projected = projected_crop_samples(camera, samples)
            projected_count = projected_counts_all[camera_index]
            intrinsics = camera.intrinsics
            projection = pyrender.IntrinsicsCamera(
                fx=float(intrinsics[0, 0]),
                fy=float(intrinsics[1, 1]),
                cx=float(intrinsics[0, 2]),
                cy=float(intrinsics[1, 2]),
                znear=0.05,
                zfar=20.0,
            )
            node = scene.add(projection, pose=pyrender_camera_pose(camera))
            try:
                depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
            finally:
                scene.remove_node(node)
            sampled_depth = np.zeros(samples.shape[0], dtype=np.float32)
            sampled_depth[projected] = depth[vi[projected], ui[projected]]
            visible = (
                projected
                & (sampled_depth > 0.0)
                & (np.abs(sampled_depth - z) <= COVERAGE_DEPTH_TOLERANCE_M)
            )
            visible_count = int(visible.sum())
            if visible_count >= COVERAGE_MIN_VISIBLE_SAMPLES or original.name == source_camera.name:
                eligible_cameras.append(original)
                visibility_rows.append(visible)
                projected_counts.append(projected_count)
                visible_counts.append(visible_count)
    finally:
        renderer.delete()

    if not visibility_rows:
        raise RuntimeError("No registered camera visibly covers the Module 03 scene crop.")
    visibility = np.stack(visibility_rows, axis=0)
    source_index = next(
        (
            index
            for index, camera in enumerate(eligible_cameras)
            if camera.name == source_camera.name
        ),
        None,
    )
    if source_index is None:
        raise RuntimeError("The source camera has no visible Module 03 crop samples.")

    centers = np.stack([camera_center(camera) for camera in eligible_cameras], axis=0)
    directions = centers - np.asarray(roi_center, dtype=np.float32)[None]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    selected_indices = [int(source_index)]
    covered = visibility[source_index].copy()
    while len(selected_indices) < min(MAX_TSDF_VIEWS, len(eligible_cameras)):
        uncovered = ~covered
        if not np.any(uncovered):
            break
        gains = visibility[:, uncovered].sum(axis=1).astype(np.int64)
        gains[selected_indices] = -1
        best_gain = int(gains.max())
        if best_gain <= 0:
            break
        tied = np.flatnonzero(gains == best_gain)
        if tied.size == 1:
            next_index = int(tied[0])
        else:
            selected_directions = directions[np.asarray(selected_indices)]
            diversity = np.min(1.0 - directions[tied] @ selected_directions.T, axis=1)
            next_index = int(tied[int(np.argmax(diversity))])
        selected_indices.append(next_index)
        covered |= visibility[next_index]

    selected = [eligible_cameras[index] for index in selected_indices]
    selected_set = set(selected_indices)
    manifest = {
        "method": "greedy_visible_crop_surface_coverage_pose_dedup_v2",
        "coverage_render_width": COVERAGE_RENDER_WIDTH,
        "surface_samples": COVERAGE_SURFACE_SAMPLES,
        "visibility_depth_tolerance_m": COVERAGE_DEPTH_TOLERANCE_M,
        "registered_camera_count": len(cameras),
        "projection_candidate_camera_count": len(projection_candidate_indices),
        "depth_prepass_camera_count": len(prepass_indices),
        "pose_deduplication": deduplication,
        "eligible_camera_count": len(eligible_cameras),
        "selected_camera_count": len(selected),
        "covered_samples": int(covered.sum()),
        "covered_fraction": float(covered.mean()),
        "source_camera_always_included": True,
        "selected_cameras": [camera.name for camera in selected],
        "eligible_cameras": [
            {
                "name": camera.name,
                "projected_samples": projected_counts[index],
                "visible_samples": visible_counts[index],
                "selected": index in selected_set,
            }
            for index, camera in enumerate(eligible_cameras)
        ],
    }
    save_json(debug_dir / "camera_coverage.json", manifest)
    return selected, manifest


def derive_crop_volume(
    interaction_name: str,
    full_scene_mesh: Any,
    source_camera: Camera,
    initial_vertices_world: Any,
    initial_body_source: str,
    debug_dir: Path,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np

    crop = load_module03_crop(interaction_name, source_camera.intrinsics)
    rendered_cameras, depths = render_scene_depths(
        full_scene_mesh, [source_camera], debug_dir / "source_crop_view"
    )
    crop_points, depth_stats = unproject_depth_crop(
        depths[0], rendered_cameras[0], crop["xyxy_source_pixels"], source_camera
    )
    initial_vertices_world = np.asarray(initial_vertices_world, dtype=np.float32)
    combined_min = np.minimum(crop_points.min(axis=0), initial_vertices_world.min(axis=0))
    combined_max = np.maximum(crop_points.max(axis=0), initial_vertices_world.max(axis=0))
    bbox_min = combined_min - ROI_PADDING_M
    bbox_max = combined_max + ROI_PADDING_M
    crop_mesh = crop_mesh_to_bounds(full_scene_mesh, bbox_min, bbox_max)
    roi_center = (bbox_min + bbox_max) * 0.5
    metadata = {
        "method": "module03_source_crop_visible_surfaces_union_initial_body_v1",
        "module03_crop_image": crop["image_path"],
        "module03_contact_spec": crop["spec_path"],
        "crop_intrinsics": crop["intrinsics"],
        "crop_image_size": [crop["width"], crop["height"]],
        "crop_xyxy_source_pixels": crop["xyxy_source_pixels"],
        "depth": depth_stats,
        "padding_m": ROI_PADDING_M,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "bbox_extent_m": bbox_max - bbox_min,
        "roi_center_world": roi_center,
        "crop_mesh_vertices": int(len(crop_mesh.vertices)),
        "crop_mesh_faces": int(len(crop_mesh.faces)),
        "initial_body_source": initial_body_source,
        "uses_module03_contact_masks": False,
        "uses_module03_contact_parts": False,
    }
    save_json(debug_dir / "scene_crop_volume.json", metadata)
    return bbox_min, bbox_max, crop_mesh, metadata


def build_visibility_tsdf(
    scene_id: str,
    cameras: list[Camera],
    candidate_view_count: int,
    depths: Any,
    bbox_min: Any,
    bbox_max: Any,
    output_dir: Path,
    mesh_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Duplicate Module 08's scalar visibility-TSDF calculation locally."""
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    dim = SDF_GRID_DIM
    truncation = SDF_TRUNCATION_M
    negative_band = SDF_NEGATIVE_BAND_M
    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    axes = [
        np.linspace(bbox_min[index], bbox_max[index], dim, dtype=np.float32)
        for index in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    total = grid.shape[0]
    positive_sum = np.zeros(total, dtype=np.float32)
    negative_sum = np.zeros(total, dtype=np.float32)
    positive_count = np.zeros(total, dtype=np.uint16)
    negative_count = np.zeros(total, dtype=np.uint16)
    surface_count = np.zeros(total, dtype=np.uint16)
    occluded_count = np.zeros(total, dtype=np.uint16)
    chunk_size = 65536

    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        points = grid[start:stop]
        chunk_positive_sum = np.zeros(stop - start, dtype=np.float32)
        chunk_negative_sum = np.zeros(stop - start, dtype=np.float32)
        chunk_positive_count = np.zeros(stop - start, dtype=np.uint16)
        chunk_negative_count = np.zeros(stop - start, dtype=np.uint16)
        chunk_surface_count = np.zeros(stop - start, dtype=np.uint16)
        chunk_occluded_count = np.zeros(stop - start, dtype=np.uint16)
        for camera_index, camera in enumerate(cameras):
            points_camera = world_to_camera(points, camera)
            point_depth = points_camera[:, 2]
            intrinsics = camera.intrinsics
            safe_depth = np.maximum(point_depth, 1e-6)
            u = intrinsics[0, 0] * points_camera[:, 0] / safe_depth + intrinsics[0, 2]
            v = intrinsics[1, 1] * points_camera[:, 1] / safe_depth + intrinsics[1, 2]
            ui = np.rint(u).astype(np.int64)
            vi = np.rint(v).astype(np.int64)
            valid = (
                (point_depth > 0.05)
                & (point_depth < 20.0)
                & (ui >= 0)
                & (ui < camera.width)
                & (vi >= 0)
                & (vi < camera.height)
            )
            sampled = np.zeros(stop - start, dtype=np.float32)
            sampled[valid] = depths[camera_index, vi[valid], ui[valid]]
            valid &= np.isfinite(sampled) & (sampled > 0.0)
            signed = sampled - point_depth
            free = valid & (signed > 0.0)
            negative = valid & (signed < 0.0) & (signed >= -negative_band)
            surface = valid & (signed == 0.0)
            occluded = valid & (signed < -negative_band)
            chunk_positive_sum[free] += np.minimum(signed[free], truncation)
            chunk_negative_sum[negative] += np.maximum(
                signed[negative], -negative_band
            )
            chunk_positive_count[free] += 1
            chunk_negative_count[negative] += 1
            chunk_surface_count[surface] += 1
            chunk_occluded_count[occluded] += 1
        positive_sum[start:stop] = chunk_positive_sum
        negative_sum[start:stop] = chunk_negative_sum
        positive_count[start:stop] = chunk_positive_count
        negative_count[start:stop] = chunk_negative_count
        surface_count[start:stop] = chunk_surface_count
        occluded_count[start:stop] = chunk_occluded_count

    free_observed = positive_count > 0
    negative_observed = (~free_observed) & (negative_count > 0)
    surface_observed = (~free_observed) & (~negative_observed) & (surface_count > 0)
    observed = free_observed | negative_observed | surface_observed
    sdf = np.full(total, truncation, dtype=np.float32)
    sdf[free_observed] = (
        positive_sum[free_observed] / positive_count[free_observed].astype(np.float32)
    )
    sdf[negative_observed] = (
        negative_sum[negative_observed] / negative_count[negative_observed].astype(np.float32)
    )
    sdf[surface_observed] = 0.0
    shape = (dim, dim, dim)
    sdf = sdf.reshape(shape)
    observed = observed.reshape(shape)
    positive_count = positive_count.reshape(shape)
    negative_count = negative_count.reshape(shape)
    surface_count = surface_count.reshape(shape)
    occluded_count = occluded_count.reshape(shape)
    voxel_size = (bbox_max - bbox_min) / float(dim - 1)

    meta_path = output_dir / f"{scene_id}.json"
    sdf_path = output_dir / f"{scene_id}_sdf.npy"
    observed_path = output_dir / f"{scene_id}_observed.npy"
    positive_count_path = output_dir / f"{scene_id}_free_vote_count.npy"
    negative_count_path = output_dir / f"{scene_id}_negative_vote_count.npy"
    surface_count_path = output_dir / f"{scene_id}_surface_vote_count.npy"
    occluded_count_path = output_dir / f"{scene_id}_occluded_vote_count.npy"
    np.save(sdf_path, sdf)
    np.save(observed_path, observed)
    np.save(positive_count_path, positive_count)
    np.save(negative_count_path, negative_count)
    np.save(surface_count_path, surface_count)
    np.save(occluded_count_path, occluded_count)
    metadata = {
        "dim": dim,
        "grid_resolution": [dim, dim, dim],
        "voxel_size_m": voxel_size,
        "min": bbox_min,
        "max": bbox_max,
        "trunc_m": truncation,
        "negative_band_m": negative_band,
        "method": TSDF_METHOD,
        "mesh_path": mesh_path,
        "num_views": len(cameras),
        "candidate_view_count": candidate_view_count,
        "selected_views": [camera.name for camera in cameras],
        "all_registered_views_considered": True,
        "candidate_definition": (
            "human anchor projects inside the camera image and the human grid "
            "overlaps its frustum"
        ),
        "view_selection": (
            f"greedy farthest-view-direction sampling capped at {MAX_TSDF_VIEWS}"
        ),
        "source_camera_always_included": True,
        "view_weight": 1.0,
        "fusion": (
            "uniform; positive free-space evidence overrides negative evidence "
            f"within the {negative_band:.3f}m behind-surface band"
        ),
        "depth_image_size": [cameras[0].height, cameras[0].width],
        "unknown_value": truncation,
        "unknown_semantics": "unobserved; stored positive for PROX penetration compatibility",
        "observed_mask_path": observed_path.resolve(),
        "free_vote_count_path": positive_count_path.resolve(),
        "negative_vote_count_path": negative_count_path.resolve(),
        "surface_vote_count_path": surface_count_path.resolve(),
        "occluded_vote_count_path": occluded_count_path.resolve(),
        "observed_voxels": int(observed.sum()),
        "total_voxels": int(observed.size),
        "observed_fraction": float(observed.mean()),
        "negative_voxels": int(np.count_nonzero(observed & (sdf < 0.0))),
        "positive_observed_voxels": int(np.count_nonzero(observed & (sdf > 0.0))),
        "neutral_observed_voxels": int(np.count_nonzero(observed & (sdf == 0.0))),
        "unknown_voxels": int(np.count_nonzero(~observed)),
        "conflicting_observed_voxels": int(
            np.count_nonzero((positive_count > 0) & (negative_count > 0))
        ),
        "sign_convention": SIGN_CONVENTION,
    }
    save_json(meta_path, metadata)
    return meta_path, metadata


def generate_local_scene_artifacts(
    interaction_name: str,
    scene_id: str,
    full_scene_mesh: Any,
    source_camera: Camera,
    scene_paths: dict[str, Path],
    output_root: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Generate the crop and scalar TSDF without importing Module 08 code."""
    import numpy as np

    prox_initial_path = MODULE_08_OUTPUT / interaction_name / "initial_smplx_camera.ply"
    module04_body_path = MODULE_04_OUTPUT / interaction_name / "first_frame_smplx_world.ply"
    if prox_initial_path.is_file():
        body_path = prox_initial_path
        initial_body_source = "module08_prox_initial_smplx_camera"
        body_mesh, body_vertices_camera, _ = load_mesh(body_path)
        body_vertices = camera_to_world(body_vertices_camera, source_camera)
    elif module04_body_path.is_file():
        body_path = module04_body_path
        initial_body_source = "module04_first_frame_smplx_world"
        body_mesh, body_vertices, _ = load_mesh(body_path)
    else:
        raise FileNotFoundError(
            "Local crop generation requires an initial human body; neither "
            f"{prox_initial_path} nor {module04_body_path} exists"
        )
    del body_mesh
    debug_dir = output_root / "debug" / "scene_crop"
    bbox_min, bbox_max, crop_mesh, crop_metadata = derive_crop_volume(
        interaction_name,
        full_scene_mesh,
        source_camera,
        body_vertices,
        initial_body_source,
        debug_dir,
    )
    mesh_path = output_root / "scene" / "mesh" / f"{scene_id}.ply"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    crop_mesh.export(mesh_path)

    intrinsics = np.asarray(source_camera.intrinsics, dtype=np.float32)
    cameras = load_colmap_cameras(
        scene_paths["colmap_images_path"],
        intrinsics,
        source_camera.width,
        source_camera.height,
    )
    observing, coverage_metadata = select_cameras_by_surface_coverage(
        full_scene_mesh,
        crop_mesh,
        cameras,
        source_camera,
        np.asarray(crop_metadata["roi_center_world"], dtype=np.float32),
        debug_dir,
    )
    render_cameras, depths = render_scene_depths(
        full_scene_mesh, observing, output_root / "debug" / "tsdf_views"
    )
    sdf_meta_path, sdf_metadata = build_visibility_tsdf(
        scene_id,
        render_cameras,
        int(coverage_metadata["eligible_camera_count"]),
        depths,
        bbox_min,
        bbox_max,
        output_root / "scene" / "sdf",
        mesh_path,
    )
    sdf_metadata.update(
        {
            "method": TSDF_METHOD,
            "volume_source": crop_metadata["method"],
            "candidate_definition": (
                "pose-deduplicated registered camera visibly covers sampled "
                "crop-mesh surfaces"
            ),
            "view_selection": coverage_metadata["method"],
            "view_coverage_fraction": coverage_metadata["covered_fraction"],
            "module03_crop_image": crop_metadata["module03_crop_image"],
            "scene_scope": SCENE_SCOPE,
            "artifact_source": "module09_local_generation",
        }
    )
    save_json(sdf_meta_path, sdf_metadata)
    provenance = {
        "mode": "module09_local_generation",
        "initial_body_source": body_path,
        "initial_body_source_kind": initial_body_source,
        "coverage": coverage_metadata,
    }
    return mesh_path, sdf_meta_path, sdf_metadata, crop_metadata, provenance


def write_binary_ply_points(path: Path, points: Any, colors: Any) -> None:
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("PLY points and colors must have the same length")

    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(points.shape[0], dtype=vertex_dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def write_tsdf_debug(
    sdf_meta_path: Path,
    output_dir: Path,
    anchor_world: Any,
    max_category_points: int,
    surface_band_m: float | None,
    seed: int,
) -> dict[str, Any]:
    """Export the lossless TSDF and inspectable sign/surface artifacts."""
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    metadata = load_json(sdf_meta_path)
    sdf_path = sdf_meta_path.with_name(sdf_meta_path.stem + "_sdf.npy")
    observed_path = Path(metadata["observed_mask_path"])
    sdf = np.load(sdf_path).astype(np.float32)
    observed = np.load(observed_path).astype(bool)
    if sdf.shape != observed.shape:
        raise ValueError(f"TSDF/observation shapes differ: {sdf.shape} versus {observed.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_min = np.asarray(metadata["min"], dtype=np.float32)
    bbox_max = np.asarray(metadata["max"], dtype=np.float32)
    anchor = np.asarray(anchor_world, dtype=np.float32)
    trunc_m = float(metadata["trunc_m"])
    dim = np.asarray(sdf.shape, dtype=np.int64)
    voxel_size = (bbox_max - bbox_min) / np.maximum(dim - 1, 1)
    band = (
        float(surface_band_m)
        if surface_band_m is not None
        else max(float(voxel_size.max()), trunc_m * 0.08)
    )

    boundary = np.zeros_like(observed)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower_t = tuple(lower)
        upper_t = tuple(upper)
        crosses = (
            observed[lower_t]
            & observed[upper_t]
            & (((sdf[lower_t] < 0) & (sdf[upper_t] >= 0)) | ((sdf[upper_t] < 0) & (sdf[lower_t] >= 0)))
        )
        boundary[lower_t] |= crosses
        boundary[upper_t] |= crosses

    negative = observed & (sdf < 0.0)
    positive = observed & (sdf > 0.0)
    neutral = observed & ((np.abs(sdf) <= band) | boundary)
    unknown = ~observed
    flat_sdf = sdf.reshape(-1)
    flat_observed = observed.reshape(-1)
    flat_boundary = boundary.reshape(-1)
    rng = np.random.default_rng(int(seed))

    flat_indices = np.arange(sdf.size, dtype=np.int64)
    ijk = np.column_stack(np.unravel_index(flat_indices, sdf.shape)).astype(np.float32)
    all_points = bbox_min[None] + ijk * voxel_size[None]

    normalized = np.clip(flat_sdf / max(trunc_m, 1e-8), -1.0, 1.0)
    all_colors = np.zeros((sdf.size, 3), dtype=np.uint8)
    neg_strength = np.clip(-normalized, 0.0, 1.0)
    pos_strength = np.clip(normalized, 0.0, 1.0)
    all_colors[:, 0] = (255 * neg_strength + 245 * (1 - np.maximum(neg_strength, pos_strength))).astype(np.uint8)
    all_colors[:, 1] = (75 * neg_strength + 245 * (1 - np.maximum(neg_strength, pos_strength)) + 110 * pos_strength).astype(np.uint8)
    all_colors[:, 2] = (45 * neg_strength + 255 * pos_strength + 210 * (1 - np.maximum(neg_strength, pos_strength))).astype(np.uint8)
    all_colors[~flat_observed] = np.asarray([150, 150, 150], dtype=np.uint8)
    all_colors[flat_boundary] = np.asarray([255, 220, 0], dtype=np.uint8)

    full_npz_path = output_dir / "tsdf_full.npz"
    np.savez_compressed(
        full_npz_path,
        tsdf=sdf,
        observed=observed,
        boundary_crossings=boundary,
        grid_min=bbox_min,
        grid_max=bbox_max,
        voxel_size=voxel_size,
        truncation_m=np.asarray(trunc_m, dtype=np.float32),
        anchor_world=anchor,
    )
    full_ply_path = output_dir / "tsdf_full_colored.ply"
    write_binary_ply_points(full_ply_path, all_points, all_colors)

    def selected_indices(mask: Any) -> Any:
        indices = np.flatnonzero(np.asarray(mask).reshape(-1))
        limit = int(max_category_points)
        if limit > 0 and indices.shape[0] > limit:
            indices = np.sort(rng.choice(indices, size=limit, replace=False))
        return indices

    category_specs = {
        "negative": (negative, (220, 40, 40), output_dir / "tsdf_negative.ply"),
        "positive_observed": (positive, (45, 125, 255), output_dir / "tsdf_positive_observed.ply"),
        "neutral_boundary": (neutral, (255, 220, 0), output_dir / "tsdf_boundary_crossings.ply"),
        "unknown": (unknown, (150, 150, 150), output_dir / "tsdf_unknown.ply"),
    }
    category_manifest = {}
    for name, (mask, color, path) in category_specs.items():
        indices = selected_indices(mask)
        colors = np.empty((indices.shape[0], 3), dtype=np.uint8)
        colors[:] = np.asarray(color, dtype=np.uint8)
        write_binary_ply_points(path, all_points[indices], colors)
        category_manifest[name] = {
            "path": path,
            "full_voxel_count": int(np.count_nonzero(mask)),
            "written_points": int(indices.shape[0]),
            "rgb": list(color),
        }

    isosurface_path = output_dir / "tsdf_zero_isosurface.ply"
    isosurface_error = None
    try:
        from skimage.measure import marching_cubes

        surface_vertices, surface_faces, _normals, _values = marching_cubes(
            sdf,
            level=0.0,
            spacing=tuple(float(value) for value in voxel_size),
            mask=observed,
        )
        surface_vertices = surface_vertices + bbox_min[None]
        trimesh.Trimesh(
            vertices=surface_vertices,
            faces=surface_faces,
            process=False,
        ).export(isosurface_path)
    except Exception as exc:
        isosurface_error = f"{type(exc).__name__}: {exc}"

    anchor_ijk = np.rint((anchor - bbox_min) / np.maximum(voxel_size, 1e-8)).astype(np.int64)
    anchor_ijk = np.clip(anchor_ijk, 0, dim - 1)

    def slice_rgb(values: Any, known: Any) -> Any:
        values = np.asarray(values, dtype=np.float32)
        known = np.asarray(known, dtype=bool)
        norm = np.clip(values / max(trunc_m, 1e-8), -1.0, 1.0)
        rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
        neg = np.clip(-norm, 0.0, 1.0)
        pos = np.clip(norm, 0.0, 1.0)
        strength = np.maximum(neg, pos)
        rgb[..., 0] = (255 * neg + 245 * (1 - strength)).astype(np.uint8)
        rgb[..., 1] = (70 * neg + 245 * (1 - strength) + 100 * pos).astype(np.uint8)
        rgb[..., 2] = (40 * neg + 255 * pos + 210 * (1 - strength)).astype(np.uint8)
        rgb[~known] = np.asarray([150, 150, 150], dtype=np.uint8)
        return rgb

    slice_specs = {
        "xy": (sdf[:, :, anchor_ijk[2]], observed[:, :, anchor_ijk[2]], (anchor_ijk[0], anchor_ijk[1])),
        "xz": (sdf[:, anchor_ijk[1], :], observed[:, anchor_ijk[1], :], (anchor_ijk[0], anchor_ijk[2])),
        "yz": (sdf[anchor_ijk[0], :, :], observed[anchor_ijk[0], :, :], (anchor_ijk[1], anchor_ijk[2])),
    }
    slice_paths = {}
    for name, (values, known, marker) in slice_specs.items():
        rgb = np.transpose(slice_rgb(values, known), (1, 0, 2))[::-1]
        image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        mx = int(round(marker[0] / max(values.shape[0] - 1, 1) * 511))
        my = 511 - int(round(marker[1] / max(values.shape[1] - 1, 1) * 511))
        draw.line((mx - 8, my, mx + 8, my), fill=(0, 255, 0), width=3)
        draw.line((mx, my - 8, mx, my + 8), fill=(0, 255, 0), width=3)
        slice_path = output_dir / f"slices_{name}.png"
        image.save(slice_path)
        slice_paths[name] = slice_path

    manifest = {
        "method": TSDF_METHOD,
        "source_sdf_meta": sdf_meta_path,
        "source_sdf_npy": sdf_path,
        "source_observed_npy": observed_path,
        "full_tsdf_npz": full_npz_path,
        "full_colored_ply": full_ply_path,
        "zero_isosurface_ply": isosurface_path if isosurface_error is None else None,
        "zero_isosurface_error": isosurface_error,
        "slices": slice_paths,
        "categories": category_manifest,
        "grid_shape": list(sdf.shape),
        "grid_min": bbox_min,
        "grid_max": bbox_max,
        "voxel_size": voxel_size,
        "truncation_m": trunc_m,
        "surface_band_m": band,
        "anchor_world": anchor,
        "anchor_ijk": anchor_ijk,
        "sign_convention": metadata["sign_convention"],
        "color_legend": {
            "negative": "red",
            "positive_observed": "blue",
            "neutral_or_boundary_crossing": "yellow",
            "unknown": "gray",
            "anchor_in_slices": "green cross",
        },
    }
    save_json(output_dir / "manifest.json", manifest)
    return manifest


def render_interaction(
    interaction_name: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from genzi.render import Renderer
    from genzi.scene import Scene

    output_root = Path(args.output_base).resolve() / interaction_name
    log(f"  output_root={output_root}")
    paths = build_interaction_paths(interaction_name, Path(args.output_base).resolve())
    required = {
        "input_scene_json": paths.input_scene_json,
        "sig_json": paths.sig_json,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    log("  loading 4DHSI metadata")
    input_payload = load_json(paths.input_scene_json)
    sig_payload = load_json(paths.sig_json)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(resolve_scannet_root(args.scannet_root), scene_context)
    for name in ("image_path", "transforms_path", "colmap_images_path", "mesh_path"):
        if not scene_paths[name].exists():
            raise FileNotFoundError(f"Missing scene {name}: {scene_paths[name]}")

    mesh, _scene_vertices_world, _scene_faces = load_mesh(scene_paths["mesh_path"])
    camera = load_scannet_camera(scene_paths, scene_context)
    target_id, target_label = resolve_sig_target(sig_payload)
    log(f"  segmenting primary target with SAM 3: {target_label!r}")
    look_at, look_at_stats = derive_sam3_object_anchor(
        image_path=scene_paths["image_path"],
        mesh=mesh,
        camera=camera,
        target_id=target_id,
        target_label=target_label,
        args=args,
        output_dir=output_root / "anchor",
    )

    scene_id = str(scene_context["scene_id"])
    log(
        f"  interaction point={np.asarray(look_at).round(4).tolist()} "
        f"source={look_at_stats.get('method')} "
        f"sam3_score={look_at_stats.get('sam3_score', 0.0):.3f}"
    )
    sdf_meta_path = output_root / "scene" / "sdf" / f"{scene_id}.json"
    scene_mesh_path = output_root / "scene" / "mesh" / f"{scene_id}.ply"
    sdf_metadata: dict[str, Any] | None = None
    scene_artifact_provenance: dict[str, Any] = {"mode": "generated_locally"}
    crop_metadata: dict[str, Any] | None = None
    reused_module08 = False
    if args.scene_source in {"auto", "module08"}:
        try:
            artifacts = validate_module08_scene_artifacts(interaction_name, scene_id)
            (
                scene_mesh_path,
                sdf_meta_path,
                sdf_metadata,
                scene_artifact_provenance,
            ) = materialize_module08_scene_artifacts(
                artifacts,
                output_root,
                scene_id,
                mesh,
                scene_paths["mesh_path"],
            )
            crop_metadata = dict(artifacts["crop_metadata"])
            reused_module08 = True
            log(
                "  using a fresh colored crop of the original ScanNet++ mesh "
                "with the validated module-08 TSDF: "
                f"{artifacts['root']}"
            )
        except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
            if args.scene_source == "module08":
                raise
            log(f"  module-08 crop/TSDF unavailable; generating locally: {exc}")

    if not reused_module08:
        log("  deriving Module 03 crop and PROX-style TSDF locally")
        (
            scene_mesh_path,
            sdf_meta_path,
            sdf_metadata,
            crop_metadata,
            scene_artifact_provenance,
        ) = generate_local_scene_artifacts(
            interaction_name,
            scene_id,
            mesh,
            camera,
            scene_paths,
            output_root,
        )
    assert sdf_metadata is not None
    assert crop_metadata is not None
    view_selection_sdf = sdf_meta_path
    view_selection_sdf_source = (
        "module08_prox_real_tsdf" if reused_module08 else "module09_local_real_tsdf"
    )
    bbox_min = np.asarray(sdf_metadata["min"], dtype=np.float32)
    bbox_max = np.asarray(sdf_metadata["max"], dtype=np.float32)
    if not np.all((look_at >= bbox_min) & (look_at <= bbox_max)):
        raise ValueError(
            f"SAM3 anchor {look_at.tolist()} lies outside the Module 03 crop bounds "
            f"{bbox_min.tolist()} .. {bbox_max.tolist()}"
        )

    log(f"  initializing GenZI renderer image_size={int(cfg['render.image_size'])}")
    renderer = Renderer(image_size=int(cfg["render.image_size"]))
    log("  loading GenZI scene + SDF")
    scene3d = Scene(
        mesh_path=str(scene_mesh_path),
        sdf_path=str(view_selection_sdf),
        subd_mesh_path="",
    )
    up_dir = np.asarray(args.up_dir, dtype=np.float32)
    view_distance_m = float(
        args.view_distance_m
        if args.view_distance_m is not None
        else cfg["data.view_distances"][0]
    )
    num_viewpoints = int(args.num_viewpoints or cfg["data.num_viewpoints"])
    max_views = int(args.max_views or cfg["data.max_views"])
    log(
        "  selecting GenZI viewpoints "
        f"num_candidates={num_viewpoints} max_views={max_views} "
        f"distance={view_distance_m:.2f}m"
    )
    started = time.time()
    viewpoints, selected_look_at = scene3d.get_viewpoints(
        renderer=renderer,
        at=np.asarray(look_at, dtype=np.float32),
        up=up_dir,
        fov=float(cfg["data.fov"]),
        num_viewpoints=num_viewpoints,
        distance=view_distance_m,
        max_views=max_views,
        radius=float(cfg["data.patch_radius"]),
        use_at_normal=bool(cfg["data.use_at_normal"]),
        vpid=interaction_name,
        cache_path=None,
    )
    viewpoint_selection = {
        "source": "genzi.Scene.get_viewpoints",
        "elapsed_s": float(time.time() - started),
    }
    viewpoints = np.asarray(viewpoints, dtype=np.float32)
    selected_look_at = np.asarray(selected_look_at, dtype=np.float32)
    if viewpoints.ndim != 2 or viewpoints.shape[0] == 0:
        raise RuntimeError(f"GenZI selected no usable views for {interaction_name}")
    log(
        "  selected "
        f"{viewpoints.shape[0]} views with GenZI Scene.get_viewpoints "
        f"elapsed={viewpoint_selection['elapsed_s']:.1f}s"
    )
    render_args = default_render_args(args)
    debug_views_dir = output_root / "debug" / "selected_views"
    debug_view_paths = []
    renderer.set_cameras(
        eyes=viewpoints,
        at=selected_look_at,
        up=up_dir,
        fov=float(cfg["data.fov"]),
    )
    used_views = list(range(renderer.num_cameras()))
    log("  rendering selected scene views/depths with GenZI")
    scene_images, _scene_masks, _scene_depths = renderer.render(
        tri_meshes=[scene3d.get_trimesh()],
        camera_ids=used_views,
        **render_args,
    )

    sdf_visualization = {
        "mode": (
            "module08_prox_artifacts_reused"
            if reused_module08
            else "module09_local_prox_style_generation"
        ),
        "source_debug_manifest": (
            Path(scene_artifact_provenance["source_root"])
            / "debug"
            / "tsdf"
            / "manifest.json"
            if reused_module08
            else output_root / "debug" / "tsdf_views" / "manifest.json"
        ),
    }

    if args.save_debug_renders:
        assert scene_images is not None
        debug_views_dir.mkdir(parents=True, exist_ok=True)
        log("  writing debug scene views")
        for idx, view_id in enumerate(used_views):
            image = Image.fromarray((scene_images[idx] * 255).astype(np.uint8))
            view_path = debug_views_dir / f"view{view_id:03d}.png"
            image.save(view_path)
            debug_view_paths.append(view_path)
        np.savez(
            debug_views_dir / "views.npz",
            viewpoints=viewpoints,
            look_at=selected_look_at,
            up_dir=up_dir,
            fov=np.asarray(float(cfg["data.fov"]), dtype=np.float32),
        )
        log(f"  wrote {len(debug_view_paths)} debug render(s) to {debug_views_dir}")

    prompt = input_payload.get("interaction_context", {}).get("interaction", "")
    interaction_label = sig_payload.get("interaction", prompt)
    scene_yaml_payload = {
        "scene": {
            "mesh_path": str(scene_mesh_path.resolve()),
            "sdf_path": str(Path(sdf_meta_path).resolve()),
            "subd_mesh_path": "",
        },
        "render": {
            **render_args,
            "up_dir": up_dir.astype(float).tolist(),
        },
        "prompt_prefix": args.prompt_prefix,
        "prompt_suffix": args.prompt_suffix,
        "prompt_ids": [interaction_name],
        "prompts": [prompt],
        "neg_prompts": [args.negative_prompt],
        "token_indices": parse_token_indices(args.token_indices),
        "lookats": [selected_look_at.astype(float).tolist()],
        "viewpoints": [[viewpoints.astype(float).tolist() for _ in cfg["optim.steps"]]],
        "interactions": [interaction_label],
    }
    audit_scene_root = output_root / "scene" / "config"
    genzi_scene_root = Path(args.output_base).resolve() / "_scene_configs"
    genzi_scene_config_path = genzi_scene_root / f"{interaction_name}_v1.yml"
    write_yaml(genzi_scene_config_path, scene_yaml_payload)
    audit_scene_config_path = audit_scene_root / f"{interaction_name}_v1.yml"
    write_yaml(audit_scene_config_path, scene_yaml_payload)
    log(f"  wrote GenZI scene config: {genzi_scene_config_path}")

    manifest = {
        "interaction_name": interaction_name,
        "scene_id": scene_id,
        "artifact_mode": "complete_native_genzi_preparation",
        "scene_scope": SCENE_SCOPE,
        "scene_source": scene_artifact_provenance,
        "genzi_scene_config": genzi_scene_config_path,
        "audit_scene_config": audit_scene_config_path,
        "debug_views_dir": debug_views_dir,
        "debug_view_paths": debug_view_paths,
        "mesh_path": scene_mesh_path,
        "full_scene_mesh_path": scene_paths["mesh_path"],
        "crop": crop_metadata,
        "sdf_meta": sdf_meta_path,
        "sdf_stats": sdf_metadata,
        "sdf_visualization": sdf_visualization,
        "view_selection_sdf_source": view_selection_sdf_source,
        "view_selection_sdf_note": "The real crop TSDF is used for viewpoint selection.",
        "look_at": selected_look_at,
        "look_at_stats": look_at_stats,
        "viewpoints": viewpoints,
        "up_dir": up_dir,
        "fov": float(cfg["data.fov"]),
        "num_candidate_viewpoints": num_viewpoints,
        "max_views": max_views,
        "selected_views": int(viewpoints.shape[0]),
        "view_distance": view_distance_m,
        "patch_radius": float(cfg["data.patch_radius"]),
        "use_at_normal": bool(cfg["data.use_at_normal"]),
        "viewpoint_selection": viewpoint_selection,
        "anchor": look_at_stats,
    }
    save_json(output_root / "preparation_summary.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ScanNet++ interactions for the native multiview GenZI baseline."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--interaction-name", "--interaction_name", dest="interaction_name", default="interaction_01")
    selection.add_argument(
        "--all-interactions",
        "--all_interactions",
        dest="all_interactions",
        action="store_true",
        help=(
            "Prepare every interaction with an alignment_summary.json under "
            "05_Optimize_Static_Scene/output."
        ),
    )
    parser.add_argument("--run-cfg", dest="run_cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output-base", dest="output_base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--scannet-root", dest="scannet_root", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the selected interaction output directory/directories.")
    parser.add_argument(
        "--no-root-summary",
        action="store_true",
        help="Do not update the shared batch summary (used by parallel launchers).",
    )
    parser.add_argument(
        "--scene-source",
        choices=("auto", "module08", "generate"),
        default="auto",
        help=(
            "Crop/TSDF source: reuse validated module-08 PROX artifacts when available "
            "(auto), require them (module08), or generate locally (generate)."
        ),
    )

    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=str(DEFAULT_SAM3_BPE))
    parser.add_argument("--sam3-python", default=str(DEFAULT_SAM3_PYTHON))
    parser.add_argument("--genzi-python", default=str(DEFAULT_GENZI_PYTHON))
    parser.add_argument("--sam3-device", default="cuda:0")
    parser.add_argument("--sam3-candidate", type=int, default=None)
    parser.add_argument("--target-prompt", default=None)
    parser.add_argument("--target-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    parser.add_argument("--no-sam3-hf-download", action="store_true")
    parser.add_argument(
        "--anchor-mask-erode-pixels",
        type=int,
        default=-1,
        help="Mask erosion radius; -1 chooses 1%% of the shorter object-box side, clamped to 2..8 pixels.",
    )
    parser.add_argument("--anchor-min-depth-pixels", type=int, default=25)

    parser.add_argument("--num-viewpoints", dest="num_viewpoints", type=int, default=None)
    parser.add_argument("--max-views", dest="max_views", type=int, default=None)
    parser.add_argument(
        "--view-distance-m",
        dest="view_distance_m",
        type=float,
        default=None,
        help="Camera distance from the coarse object anchor; default uses the GenZI config.",
    )
    parser.add_argument("--up-dir", dest="up_dir", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument("--save-debug-renders", dest="save_debug_renders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dir-light-intensity", dest="dir_light_intensity", type=float, default=5.0)
    parser.add_argument("--pt-light-intensity", dest="pt_light_intensity", type=float, default=5.0)

    parser.add_argument("--prompt-prefix", dest="prompt_prefix", default=DEFAULT_PROMPT_PREFIX)
    parser.add_argument("--prompt-suffix", dest="prompt_suffix", default=DEFAULT_PROMPT_SUFFIX)
    parser.add_argument("--negative-prompt", dest="negative_prompt", default="")
    parser.add_argument("--token-indices", dest="token_indices", default=DEFAULT_TOKEN_INDICES)
    parser.add_argument("--opengl-platform", dest="opengl_platform", default="egl")
    parser.add_argument("--_runtime-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_sam3-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_sam3-image", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_sam3-output", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.no_sam3_hf_download and not args.sam3_checkpoint:
        raise ValueError("--no-sam3-hf-download requires --sam3-checkpoint; refusing to run SAM 3 with random weights.")
    if args._sam3_worker:
        if not args._sam3_image or not args._sam3_output or not args.target_prompt:
            raise ValueError("Internal SAM 3 worker requires image, output, and target prompt.")
        from PIL import Image

        processor = build_sam3_processor(args)
        image = Image.open(args._sam3_image).convert("RGB")
        predictions = run_sam3(processor, image, args.target_prompt, args.target_box)
        save_sam3_prediction_archive(Path(args._sam3_output), predictions)
        return

    requested_genzi_python = Path(args.genzi_python).resolve()
    if not args._runtime_child and Path(sys.executable).resolve() != requested_genzi_python:
        if not requested_genzi_python.exists():
            raise FileNotFoundError(f"GenZI Python does not exist: {requested_genzi_python}")
        command = [requested_genzi_python, Path(__file__).resolve(), *(argv if argv is not None else sys.argv[1:]), "--_runtime-child"]
        completed = subprocess.run([str(value) for value in command], cwd=str(WORKSPACE_ROOT))
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    configure_headless_rendering(args.opengl_platform)
    interaction_names = (
        discover_module05_interactions()
        if args.all_interactions
        else [args.interaction_name]
    )
    if args.all_interactions:
        log(
            f"[*] Selected {len(interaction_names)} interaction(s) processed by module 05: "
            + ", ".join(interaction_names)
        )
    if args.all_interactions and any(
        value is not None for value in (args.sam3_candidate, args.target_prompt, args.target_box)
    ):
        raise ValueError("SAM 3 candidate/prompt/box overrides are supported only for a single interaction.")

    output_base = Path(args.output_base).resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    for interaction_name in interaction_names:
        interaction_output = output_base / interaction_name
        if interaction_output.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"Output already exists: {interaction_output}. Pass --overwrite to replace it."
                )
            shutil.rmtree(interaction_output)
        runtime_config = output_base / "_scene_configs" / f"{interaction_name}_v1.yml"
        if runtime_config.exists():
            runtime_config.unlink()

    cfg = load_cfg(Path(args.run_cfg).resolve())

    summaries: list[dict[str, Any]] = []
    for interaction_name in interaction_names:
        started = time.time()
        log(f"\n[*] Preparing {interaction_name}")
        summary = render_interaction(interaction_name, args, cfg)
        summary["preparation_elapsed_s"] = float(time.time() - started)
        summaries.append(summary)

    batch_summary = {
        "interactions": interaction_names,
        "preparations": summaries,
        "run_cfg": args.run_cfg,
        "anchor_method": ANCHOR_METHOD,
        "tsdf_method": TSDF_METHOD,
        "scene_scope": SCENE_SCOPE,
        "scene_source_policy": args.scene_source,
        "uses_contact_information": False,
    }
    if not args.no_root_summary:
        save_json(output_base / "preparation_summary.json", batch_summary)
    for summary in summaries:
        save_json(output_base / summary["interaction_name"] / "preparation_summary.json", summary)
    log(f"\n[*] Preparation finished for {len(interaction_names)} interaction(s).")
    log("[*] Run native GenZI separately with 01_run_genzi.py.")


if __name__ == "__main__":
    main()
