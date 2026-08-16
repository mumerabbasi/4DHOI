"""Prepare ScanNet++ interactions for the native GenZI baseline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"
DEFAULT_SDF_DIM = 192
DEFAULT_SDF_PADDING_M = 0.5
DEFAULT_PROMPT_PREFIX = "a woman"
DEFAULT_PROMPT_SUFFIX = "wearing a white shirt and blue pants, full body"
DEFAULT_TOKEN_INDICES = "2"
SAM3_ROOT = WORKSPACE_ROOT / "sam3"
DEFAULT_SAM3_BPE = SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
DEFAULT_GENZI_PYTHON = Path("/root/miniconda3/envs/genzi/bin/python")
DEFAULT_SAM3_PYTHON = Path("/root/miniconda3/envs/sam3/bin/python")
ANCHOR_METHOD = "sam3_visible_object_centroid_v1"
TSDF_METHOD = "selected_view_depth_tsdf_v2"
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


@dataclass
class Camera:
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


def interaction_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"interaction_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def discover_interactions() -> list[str]:
    names = sorted(
        (path.parent.name for path in (PROJECT_DIR / "01_Generate_SIG" / "output").glob("interaction_*/sig.json")),
        key=interaction_sort_key,
    )
    if not names:
        raise RuntimeError("No generated SIG interactions were found.")
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
    return Camera(intrinsics, rotation, translation, width, height)


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
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint,
        device=args.sam3_device,
        load_from_HF=not args.no_sam3_hf_download,
    )
    return Sam3Processor(
        model=model,
        device=args.sam3_device,
        confidence_threshold=float(args.sam3_confidence_threshold),
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
        state = processor.add_geometric_prompt(state=state, box=normalized_cxcywh, label=True)
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
        "--sam3-confidence-threshold",
        str(args.sam3_confidence_threshold),
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
    ambiguity_margin: float,
    candidate_path: Path,
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
    if len(ranked) > 1:
        score_gap = float(ranked[0][1]["score"] - ranked[1][1]["score"])
        if score_gap < float(ambiguity_margin):
            raise RuntimeError(
                "SAM 3 target selection is ambiguous "
                f"(top score gap {score_gap:.3f} < {ambiguity_margin:.3f}). "
                f"Inspect {candidate_path} and rerun with --sam3-candidate INDEX or --target-box."
            )
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
    selected_index, selected = select_sam3_candidate(
        predictions, args.sam3_candidate, args.sam3_ambiguity_margin, candidate_path
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


def write_neutral_sdf(
    scene_id: str,
    scene_vertices_world: Any,
    sdf_dir: Path,
    dim: int,
    padding_m: float,
) -> Path:
    import numpy as np

    sdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = sdf_dir / f"{scene_id}.json"
    sdf_path = sdf_dir / f"{scene_id}_sdf.npy"
    vertices = np.asarray(scene_vertices_world, dtype=np.float32)
    bbox_min = vertices.min(axis=0) - float(padding_m)
    bbox_max = vertices.max(axis=0) + float(padding_m)
    sdf = np.ones((int(dim), int(dim), int(dim)), dtype=np.float32)
    np.save(sdf_path, sdf)
    save_json(
        meta_path,
        {
            "dim": int(dim),
            "min": bbox_min.astype(float).tolist(),
            "max": bbox_max.astype(float).tolist(),
            "method": "temporary_neutral_placeholder_for_view_selection",
            "note": "Used only because GenZI Scene requires an sdf_path before get_viewpoints.",
        },
    )
    return meta_path


def build_depth_tsdf_sdf(
    scene_id: str,
    scene_vertices_world: Any,
    renderer: Any,
    scene_depths: Any,
    sdf_dir: Path,
    dim: int,
    padding_m: float,
    trunc_m: float,
) -> tuple[Path, dict[str, Any]]:
    import numpy as np

    sdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = sdf_dir / f"{scene_id}.json"
    sdf_path = sdf_dir / f"{scene_id}_sdf.npy"
    observed_path = sdf_dir / f"{scene_id}_observed.npy"
    vertices = np.asarray(scene_vertices_world, dtype=np.float32)
    bbox_min = vertices.min(axis=0) - float(padding_m)
    bbox_max = vertices.max(axis=0) + float(padding_m)
    dim = int(dim)
    trunc_m = float(trunc_m)

    xs = np.linspace(bbox_min[0], bbox_max[0], dim, dtype=np.float32)
    ys = np.linspace(bbox_min[1], bbox_max[1], dim, dtype=np.float32)
    zs = np.linspace(bbox_min[2], bbox_max[2], dim, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    sdf = np.full((grid.shape[0],), trunc_m, dtype=np.float32)

    depths = np.asarray(scene_depths, dtype=np.float32)
    num_views, height, width = depths.shape
    camera_ids = list(range(num_views))
    znear = float(renderer.camera_args.get("znear", 0.1))
    zfar = float(renderer.camera_args.get("zfar", 20.0))
    observed = np.zeros((grid.shape[0],), dtype=bool)

    chunk_size = 65536
    for start in range(0, grid.shape[0], chunk_size):
        end = min(start + chunk_size, grid.shape[0])
        points = grid[start:end]
        screen = renderer.project(points, camera_ids=camera_ids)
        best_abs = np.full((points.shape[0],), np.inf, dtype=np.float32)
        best_sdf = np.full((points.shape[0],), trunc_m, dtype=np.float32)

        hom = np.concatenate(
            (points, np.ones((points.shape[0], 1), dtype=np.float32)),
            axis=1,
        )
        for view_idx, modelview in enumerate(renderer.modelviews):
            cam = hom @ modelview.T
            point_depth = -cam[:, 2]
            uv = screen[view_idx]
            ui = np.rint(uv[:, 0]).astype(np.int64)
            vi = np.rint(uv[:, 1]).astype(np.int64)
            valid = (
                (point_depth > znear)
                & (point_depth < zfar)
                & (ui >= 0)
                & (ui < width)
                & (vi >= 0)
                & (vi < height)
            )
            if not valid.any():
                continue
            sampled_depth = np.zeros((points.shape[0],), dtype=np.float32)
            sampled_depth[valid] = depths[view_idx, vi[valid], ui[valid]]
            valid &= sampled_depth > 0.0
            if not valid.any():
                continue

            signed_distance = sampled_depth - point_depth
            signed_distance = np.clip(signed_distance, -trunc_m, trunc_m).astype(np.float32)
            abs_distance = np.abs(signed_distance)
            update = valid & (abs_distance < best_abs)
            best_abs[update] = abs_distance[update]
            best_sdf[update] = signed_distance[update]

        observed_chunk = np.isfinite(best_abs)
        sdf[start:end] = best_sdf
        observed[start:end] = observed_chunk

    sdf = sdf.reshape(dim, dim, dim).astype(np.float32)
    observed_volume = observed.reshape(dim, dim, dim)
    np.save(sdf_path, sdf)
    np.save(observed_path, observed_volume)
    metadata = {
        "dim": dim,
        "min": bbox_min.astype(float).tolist(),
        "max": bbox_max.astype(float).tolist(),
        "mesh_path": "",
        "method": TSDF_METHOD,
        "trunc_m": trunc_m,
        "num_views": int(num_views),
        "depth_image_size": [int(height), int(width)],
        "unknown_value": trunc_m,
        "observed_mask_path": str(observed_path.resolve()),
        "observed_voxels": int(observed.sum()),
        "total_voxels": int(observed.shape[0]),
        "observed_fraction": float(observed.mean()),
        "negative_voxels": int(np.count_nonzero(observed_volume & (sdf < 0.0))),
        "positive_observed_voxels": int(np.count_nonzero(observed_volume & (sdf > 0.0))),
        "neutral_observed_voxels": int(np.count_nonzero(observed_volume & (sdf == 0.0))),
        "unknown_voxels": int(np.count_nonzero(~observed_volume)),
        "sign_convention": "positive is free space in front of rendered depth; negative is behind an observed scene surface",
    }
    save_json(meta_path, metadata)
    return meta_path, metadata


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

    mesh, scene_vertices_world, _scene_faces = load_mesh(scene_paths["mesh_path"])
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
    sdf_dir = output_root / "scene" / "sdf"
    sdf_meta_path = sdf_dir / f"{scene_id}.json"

    sdf_dim = int(args.sdf_dim or cfg.get("data.sdf_dim", DEFAULT_SDF_DIM))
    sdf_padding_m = float(
        args.sdf_padding_m
        if args.sdf_padding_m is not None
        else cfg.get("data.sdf_padding_m", DEFAULT_SDF_PADDING_M)
    )
    bootstrap_tmpdir = tempfile.TemporaryDirectory(prefix="genzi_sdf_bootstrap_")
    view_selection_sdf = write_neutral_sdf(
        scene_id=scene_id,
        scene_vertices_world=scene_vertices_world,
        sdf_dir=Path(bootstrap_tmpdir.name),
        dim=int(args.bootstrap_sdf_dim),
        padding_m=sdf_padding_m,
    )
    view_selection_sdf_source = "temporary_neutral_bootstrap"

    log(f"  initializing GenZI renderer image_size={int(cfg['render.image_size'])}")
    renderer = Renderer(image_size=int(cfg["render.image_size"]))
    log("  loading GenZI scene + SDF")
    scene3d = Scene(
        mesh_path=str(scene_paths["mesh_path"]),
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
    if bootstrap_tmpdir is not None:
        bootstrap_tmpdir.cleanup()
        bootstrap_tmpdir = None
        log("  removed temporary neutral SDF")

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
    scene_images, _scene_masks, scene_depths = renderer.render(
        tri_meshes=[scene3d.get_trimesh()],
        camera_ids=used_views,
        **render_args,
    )

    log(
        "  building fresh selected-view depth TSDF "
        f"dim={sdf_dim} trunc={float(args.depth_sdf_trunc_m):.3f}m"
    )
    sdf_meta_path, sdf_metadata = build_depth_tsdf_sdf(
        scene_id=scene_id,
        scene_vertices_world=scene_vertices_world,
        renderer=renderer,
        scene_depths=scene_depths,
        sdf_dir=sdf_dir,
        dim=sdf_dim,
        padding_m=sdf_padding_m,
        trunc_m=float(args.depth_sdf_trunc_m),
    )
    sdf_metadata.update(
        {
            "anchor_method": ANCHOR_METHOD,
            "anchor_world_xyz": selected_look_at,
            "mesh_path": str(scene_paths["mesh_path"].resolve()),
        }
    )
    save_json(sdf_meta_path, sdf_metadata)
    log(
        "  wrote depth-based TSDF "
        f"observed={sdf_metadata.get('observed_fraction', 0.0):.3f} path={sdf_meta_path}"
    )

    log("  exporting full TSDF debug artifacts")
    sdf_visualization = write_tsdf_debug(
        sdf_meta_path=sdf_meta_path,
        output_dir=output_root / "debug" / "tsdf",
        anchor_world=selected_look_at,
        max_category_points=int(args.tsdf_debug_max_points),
        surface_band_m=args.tsdf_surface_band_m,
        seed=int(args.seed),
    )

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
            "mesh_path": str(scene_paths["mesh_path"].resolve()),
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
        "genzi_scene_config": genzi_scene_config_path,
        "audit_scene_config": audit_scene_config_path,
        "debug_views_dir": debug_views_dir,
        "debug_view_paths": debug_view_paths,
        "mesh_path": scene_paths["mesh_path"],
        "sdf_meta": sdf_meta_path,
        "sdf_stats": sdf_metadata,
        "sdf_visualization": sdf_visualization,
        "view_selection_sdf_source": view_selection_sdf_source,
        "view_selection_sdf_note": "Temporary bootstrap removed after viewpoint selection.",
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
    if bootstrap_tmpdir is not None:
        bootstrap_tmpdir.cleanup()
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ScanNet++ interactions for the native multiview GenZI baseline."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--interaction-name", "--interaction_name", dest="interaction_name", default="interaction_01")
    selection.add_argument("--all-interactions", "--all_interactions", dest="all_interactions", action="store_true")
    parser.add_argument("--run-cfg", dest="run_cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output-base", dest="output_base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--scannet-root", dest="scannet_root", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", help="Replace only the selected interaction output directory/directories.")

    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument("--sam3-bpe-path", default=str(DEFAULT_SAM3_BPE))
    parser.add_argument("--sam3-python", default=str(DEFAULT_SAM3_PYTHON))
    parser.add_argument("--genzi-python", default=str(DEFAULT_GENZI_PYTHON))
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--sam3-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-ambiguity-margin", type=float, default=0.05)
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

    parser.add_argument("--sdf-dim", dest="sdf_dim", type=int, default=DEFAULT_SDF_DIM)
    parser.add_argument("--sdf-padding-m", dest="sdf_padding_m", type=float, default=None)
    parser.add_argument(
        "--depth-sdf-trunc-m",
        dest="depth_sdf_trunc_m",
        type=float,
        default=0.25,
        help="Truncation distance for the selected-view depth TSDF used by GenZI.",
    )
    parser.add_argument("--bootstrap-sdf-dim", dest="bootstrap_sdf_dim", type=int, default=16)
    parser.add_argument("--tsdf-debug-max-points", type=int, default=0, help="Per-category PLY limit; 0 writes every voxel.")
    parser.add_argument(
        "--tsdf-surface-band-m",
        dest="tsdf_surface_band_m",
        type=float,
        default=None,
        help="Absolute TSDF band included with sign-changing voxels in the yellow boundary PLY.",
    )

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
    interaction_names = discover_interactions() if args.all_interactions else [args.interaction_name]
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
        "uses_contact_information": False,
    }
    save_json(output_base / "preparation_summary.json", batch_summary)
    for summary in summaries:
        save_json(output_base / summary["interaction_name"] / "preparation_summary.json", summary)
    log(f"\n[*] Preparation finished for {len(interaction_names)} interaction(s).")
    log("[*] Run native GenZI separately with 01_run_genzi.py.")


if __name__ == "__main__":
    main()
