#!/usr/bin/env python3
"""Run PROX with a Module-03 scene-crop TSDF on ScanNet++ interactions.

The Module 03 image crop defines the 3D optimization volume. Registered DSLR
cameras are pose-deduplicated, tested for real crop-surface visibility, and then
selected greedily for TSDF coverage before the original PROX optimization.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
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
PROX_ROOT = WORKSPACE_ROOT / "PROX"
OPENPOSE_ROOT = WORKSPACE_ROOT / "openpose"
SCANNET_ROOT = WORKSPACE_ROOT / "Scannet++" / "data"
OUTPUT_ROOT = MODULE_DIR / "output"
MODULE03_OUTPUT = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"

PROX_CONFIG = PROX_ROOT / "cfg_files" / "PROX.yaml"
PROX_MODELS = PROX_ROOT / "models"
BODY_SEGMENTS_DIR = PROX_MODELS / "body_segments"
VPOSER_DIR = PROX_MODELS / "vposer_v1_0"
PART_SEGMENTATION = PROX_MODELS / "smplx_parts_segm.pkl"
OPENPOSE_EXECUTABLE = (
    OPENPOSE_ROOT / "build" / "examples" / "openpose" / "openpose.bin"
)

CONTACT_BODY_PARTS = [
    "L_Leg",
    "R_Leg",
    "L_Hand",
    "R_Hand",
    "gluteus",
    "back",
    "thighs",
]
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
RANDOM_SEED = 24017
TSDF_METHOD = (
    "module03_scene_crop_coverage_visibility_tsdf_384_depth2048_pose_dedup_v2"
)
SIGN_CONVENTION = (
    "positive is directly observed free space in front of rendered depth; "
    "negative is the symmetric truncation band behind an observed surface; positive "
    "free-space evidence overrides negative evidence"
)


@dataclass(frozen=True)
class Camera:
    name: str
    intrinsics: Any
    rotation_world_to_camera: Any
    translation_world_to_camera: Any
    width: int
    height: int


@dataclass(frozen=True)
class InteractionInputs:
    name: str
    human_image: Path
    input_scene: Path
    scene_id: str
    camera_name: str
    transforms: Path
    poses: Path
    mesh: Path


def log(message: str) -> None:
    print(message, flush=True)


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
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def interaction_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"interaction_(\d+)", name)
    return (int(match.group(1)) if match else 10**9, name)


def eligible_interactions() -> list[str]:
    names = sorted(
        (path.name for path in MODULE03_OUTPUT.glob("interaction_*") if path.is_dir()),
        key=interaction_sort_key,
    )
    if not names:
        raise FileNotFoundError(
            f"No interaction outputs were found under {MODULE03_OUTPUT}."
        )
    return names


def resolve_interactions(args: argparse.Namespace) -> list[str]:
    eligible = eligible_interactions()
    if args.all_interactions:
        return eligible
    if args.interaction_name not in eligible:
        raise ValueError(
            f"{args.interaction_name!r} has no Module 03 Agentic output under "
            f"{MODULE03_OUTPUT}."
        )
    return [args.interaction_name]


def resolve_interaction_inputs(name: str) -> InteractionInputs:
    input_scene = (
        PROJECT_DIR / "01_Generate_SIG" / "input_prompts" / name / "input_scene.json"
    )
    payload = load_json(input_scene)
    scene_context = payload["scene_context"]
    if scene_context["camera"]["source"] != "dslr_resized_undistorted":
        raise ValueError(
            "PROX ScanNet++ wrapper supports only dslr_resized_undistorted cameras."
        )
    scene_id = str(scene_context["scene_id"])
    camera_name = str(scene_context["camera"]["name"])
    scene_root = SCANNET_ROOT / scene_id
    human_root = PROJECT_DIR / "02_Generate_Human_Frame" / "output" / name
    inputs = InteractionInputs(
        name=name,
        human_image=human_root / "inpainted_frame_resized.png",
        input_scene=input_scene,
        scene_id=scene_id,
        camera_name=camera_name,
        transforms=scene_root / "dslr" / "nerfstudio" / "transforms_undistorted.json",
        poses=scene_root / "dslr" / "colmap" / "images.txt",
        mesh=scene_root / "scans" / "mesh_aligned_0.05.ply",
    )
    missing = [
        str(path)
        for path in (
            inputs.human_image,
            inputs.input_scene,
            inputs.transforms,
            inputs.poses,
            inputs.mesh,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing interaction input(s): " + "; ".join(missing))
    return inputs


def validate_prox_assets(config: dict[str, Any]) -> None:
    gender = str(config["gender"])
    required = [
        PROX_CONFIG,
        PROX_MODELS / "smplx" / f"SMPLX_{gender.upper()}.npz",
        VPOSER_DIR,
        PART_SEGMENTATION,
        BODY_SEGMENTS_DIR / "body_mask.json",
        *(BODY_SEGMENTS_DIR / f"{part}.json" for part in CONTACT_BODY_PARTS),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing official PROX runtime asset(s):\n  "
            + "\n  ".join(missing)
            + "\nPlace the official body_segments, VPoser v1 model, and "
            "SMPL-X/segmentation assets under PROX/models."
        )


def validate_openpose() -> None:
    required = [
        OPENPOSE_EXECUTABLE,
        OPENPOSE_ROOT / "models" / "pose" / "body_25" / "pose_iter_584000.caffemodel",
        OPENPOSE_ROOT / "models" / "face" / "pose_iter_116000.caffemodel",
        OPENPOSE_ROOT / "models" / "hand" / "pose_iter_102000.caffemodel",
    ]
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            "The official OpenPose executable/model file(s) are missing:\n  "
            + "\n  ".join(str(path) for path in missing)
            + "\n"
            "Build the official CMU OpenPose checkout at "
            f"{OPENPOSE_ROOT}; the wrapper requires its native BODY_25 detector."
        )


def colmap_qvec_to_rotmat(qvec: Any) -> Any:
    import numpy as np

    qw, qx, qy, qz = np.asarray(qvec, dtype=np.float64)
    return np.asarray(
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


def load_intrinsics(path: Path) -> tuple[Any, int, int]:
    import numpy as np

    payload = load_json(path)
    intrinsics = np.asarray(
        [
            [float(payload["fl_x"]), 0.0, float(payload["cx"])],
            [0.0, float(payload["fl_y"]), float(payload["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics, int(payload["w"]), int(payload["h"])


def load_colmap_cameras(path: Path, intrinsics: Any, width: int, height: int) -> list[Camera]:
    import numpy as np

    cameras: list[Camera] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 10 or parts[0].startswith("#"):
            continue
        try:
            image_id = int(parts[0])
            qvec = np.asarray([float(value) for value in parts[1:5]], dtype=np.float32)
            translation = np.asarray(
                [float(value) for value in parts[5:8]], dtype=np.float32
            )
            int(parts[8])
        except ValueError:
            continue
        del image_id
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


def camera_by_name(cameras: list[Camera], name: str) -> Camera:
    for camera in cameras:
        if camera.name == name:
            return camera
    raise ValueError(f"Camera {name!r} was not found in the COLMAP image list.")


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


def run_openpose(image_path: Path, output_dir: Path) -> dict[str, Any]:
    """Run the exact OpenPose keypoint frontend used by original PROX."""
    import numpy as np

    json_dir = output_dir / "json"
    overlay_dir = output_dir / "overlay"
    input_dir = output_dir / "input"
    json_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    openpose_input = input_dir / image_path.name
    shutil.copy2(image_path, openpose_input)
    command = [
        str(OPENPOSE_EXECUTABLE),
        "--image_dir",
        str(input_dir),
        "--model_folder",
        str(OPENPOSE_ROOT / "models"),
        "--model_pose",
        "BODY_25",
        "--hand",
        "--face",
        "--number_people_max",
        "1",
        "--num_gpu",
        "1",
        "--num_gpu_start",
        "0",
        "--display",
        "0",
        "--render_pose",
        "1",
        "--write_json",
        str(json_dir),
        "--write_images",
        str(overlay_dir),
    ]
    log("  running official OpenPose BODY_25 detector")
    subprocess.run(command, cwd=OPENPOSE_ROOT, check=True)
    keypoint_path = json_dir / f"{openpose_input.stem}_keypoints.json"
    if not keypoint_path.is_file():
        raise FileNotFoundError(f"OpenPose did not write {keypoint_path}.")
    payload = load_json(keypoint_path)
    people = payload.get("people", [])
    if not people:
        raise RuntimeError(f"OpenPose detected no person in {image_path}.")
    person = people[0]
    body = np.asarray(person["pose_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    left_hand = np.asarray(
        person["hand_left_keypoints_2d"], dtype=np.float32
    ).reshape(-1, 3)
    right_hand = np.asarray(
        person["hand_right_keypoints_2d"], dtype=np.float32
    ).reshape(-1, 3)
    face = np.asarray(person["face_keypoints_2d"], dtype=np.float32).reshape(-1, 3)
    if body.shape != (25, 3):
        raise ValueError(f"OpenPose returned {body.shape[0]} body joints, expected BODY_25.")
    if left_hand.shape != (21, 3) or right_hand.shape != (21, 3):
        raise ValueError("OpenPose did not return its native 21 joints for each hand.")
    if face.shape[0] < 68:
        raise ValueError("OpenPose did not return the face landmarks required by PROX.")
    # This is exactly PROX data_parser.read_keypoints: BODY_25, both complete
    # OpenPose hands, then FLAME-compatible face landmarks 17:68.
    keypoints = np.concatenate([body, left_hand, right_hand, face[17:68]], axis=0)
    overlay_candidates = sorted(overlay_dir.glob(f"{openpose_input.stem}*"))
    return {
        "keypoints": keypoints,
        "body25": body,
        "json": keypoint_path,
        "overlay": overlay_candidates[0] if overlay_candidates else None,
        "command": command,
    }


def load_interaction_data(inputs: InteractionInputs) -> dict[str, Any]:
    from PIL import Image

    with Image.open(inputs.human_image) as image:
        target_size = image.size
    intrinsics, camera_width, camera_height = load_intrinsics(inputs.transforms)
    if target_size != (camera_width, camera_height):
        raise ValueError(
            f"Human image size {target_size} does not match ScanNet++ camera "
            f"size {(camera_width, camera_height)}."
        )
    cameras = load_colmap_cameras(
        inputs.poses, intrinsics, camera_width, camera_height
    )
    source_camera = camera_by_name(cameras, inputs.camera_name)
    return {
        "intrinsics": intrinsics,
        "image_size": target_size,
        "cameras": cameras,
        "source_camera": source_camera,
    }


def create_body_model(device: Any, torch: Any, config: dict[str, Any]) -> Any:
    import smplx
    from misc_utils import JointMapper, smpl_to_openpose

    mapper = JointMapper(
        smpl_to_openpose(
            model_type="smplx",
            use_hands=True,
            use_face=True,
            use_face_contour=False,
            openpose_format="coco25",
        )
    )
    model = smplx.create(
        model_path=str(PROX_MODELS),
        model_type="smplx",
        gender=str(config["gender"]),
        ext="npz",
        num_betas=10,
        use_pca=bool(config["use_pca"]),
        num_pca_comps=int(config["num_pca_comps"]),
        flat_hand_mean=bool(config["flat_hand_mean"]),
        joint_mapper=mapper,
        create_global_orient=True,
        create_body_pose=False,
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=True,
        batch_size=1,
        dtype=torch.float32,
    ).to(device)
    model.reset_params()
    return model


def load_vposer(device: Any) -> tuple[Any, Any]:
    import torch
    from human_body_prior.tools.model_loader import load_vposer

    patch_torchgeometry_for_modern_torch(torch)
    vposer, _ = load_vposer(str(VPOSER_DIR), vp_model="snapshot")
    vposer = vposer.to(device).eval()
    embedding = torch.zeros(
        (1, 32), dtype=torch.float32, device=device, requires_grad=True
    )
    return vposer, embedding


def decoded_body_pose(vposer: Any, pose_embedding: Any) -> Any:
    return vposer.decode(pose_embedding, output_type="aa").reshape(1, -1)


def body_vertices_and_joints(body_model: Any, body_pose: Any, torch: Any) -> tuple[Any, Any]:
    with torch.no_grad():
        output = body_model(body_pose=body_pose, return_verts=True)
    return (
        output.vertices[0].detach().cpu().numpy().astype("float32"),
        output.joints[0].detach().cpu().numpy().astype("float32"),
    )


def sdf_grid_dimension(bbox_min: Any, bbox_max: Any) -> int:
    """Return the fixed cubic resolution used by the PROX scene SDF."""
    del bbox_min, bbox_max
    return SDF_GRID_DIM


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
    from PIL import Image

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
                color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            finally:
                scene.remove_node(node)
            depth = np.asarray(depth, dtype=np.float32)
            depths.append(depth)
            stem = f"{index:03d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', camera.name)}"
            color_path = debug_dir / f"{stem}_scene.png"
            depth_path = debug_dir / f"{stem}_depth_m.npy"
            depth_visual_path = debug_dir / f"{stem}_depth.png"
            Image.fromarray(np.asarray(color[:, :, :3], dtype=np.uint8)).save(color_path)
            np.save(depth_path, depth)
            valid_depth = depth[np.isfinite(depth) & (depth > 0.0)]
            depth_visual = np.zeros(depth.shape, dtype=np.uint8)
            if valid_depth.size:
                near, far = np.percentile(valid_depth, [2.0, 98.0])
                far = max(float(far), float(near) + 1e-6)
                normalized = np.clip((depth - near) / (far - near), 0.0, 1.0)
                depth_visual[depth > 0.0] = (
                    255.0 * (1.0 - normalized[depth > 0.0])
                ).astype(np.uint8)
            else:
                near = far = 0.0
            Image.fromarray(depth_visual).save(depth_visual_path)
            view_manifest.append(
                {
                    "index": index,
                    "camera_name": camera.name,
                    "scene_png": color_path,
                    "depth_visual_png": depth_visual_path,
                    "depth_m_npy": depth_path,
                    "depth_visual_near_m": float(near),
                    "depth_visual_far_m": float(far),
                    "valid_depth_pixels": int(valid_depth.size),
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
            "description": (
                "Synthetic scene RGB previews and exact mesh-rendered depth maps "
                "used for TSDF fusion; original DSLR RGB files are not required"
            ),
            "render_width": DEPTH_RENDER_WIDTH,
            "num_views": len(render_cameras),
            "views": view_manifest,
        },
    )
    return render_cameras, np.stack(depths, axis=0)


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
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    dim = sdf_grid_dimension(bbox_min, bbox_max)
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
    gradient = np.stack(
        np.gradient(sdf, *[float(value) for value in voxel_size], edge_order=1),
        axis=-1,
    ).astype(np.float32)
    gradient_norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    normals = np.divide(
        gradient,
        np.maximum(gradient_norm, 1e-8),
        out=np.zeros_like(gradient),
    )
    # Released PROX multiplies penetrating SDF values by these normals and then
    # evaluates an explicit sqrt(sum(x**2)).  A zero normal makes that expression
    # numerically undefined at zero even though its intended scalar magnitude is
    # simply abs(SDF).  Any unit vector has exactly the same magnitude in PROX's
    # loss, so give constant TSDF regions a deterministic unit fallback.
    zero_normal = gradient_norm[..., 0] <= 1e-8
    normals[zero_normal, 0] = 1.0

    meta_path = output_dir / f"{scene_id}.json"
    sdf_path = output_dir / f"{scene_id}_sdf.npy"
    normals_path = output_dir / f"{scene_id}_normals.npy"
    observed_path = output_dir / f"{scene_id}_observed.npy"
    positive_count_path = output_dir / f"{scene_id}_free_vote_count.npy"
    negative_count_path = output_dir / f"{scene_id}_negative_vote_count.npy"
    surface_count_path = output_dir / f"{scene_id}_surface_vote_count.npy"
    occluded_count_path = output_dir / f"{scene_id}_occluded_vote_count.npy"
    np.save(sdf_path, sdf)
    np.save(normals_path, normals)
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
        "normals_path": normals_path.resolve(),
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


def write_binary_ply_points(path: Path, points: Any, colors: Any) -> None:
    import numpy as np

    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
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
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {points.shape[0]}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def write_tsdf_debug(sdf_meta_path: Path, output_dir: Path, anchor_world: Any) -> dict[str, Any]:
    import numpy as np
    import trimesh
    from PIL import Image, ImageDraw

    metadata = load_json(sdf_meta_path)
    sdf_path = sdf_meta_path.with_name(sdf_meta_path.stem + "_sdf.npy")
    observed_path = Path(metadata["observed_mask_path"])
    sdf = np.load(sdf_path).astype(np.float32)
    observed = np.load(observed_path).astype(bool)
    bbox_min = np.asarray(metadata["min"], dtype=np.float32)
    bbox_max = np.asarray(metadata["max"], dtype=np.float32)
    anchor = np.asarray(anchor_world, dtype=np.float32)
    truncation = float(metadata["trunc_m"])
    negative_band = float(metadata.get("negative_band_m", truncation))
    dim = np.asarray(sdf.shape, dtype=np.int64)
    voxel_size = (bbox_max - bbox_min) / np.maximum(dim - 1, 1)
    surface_band = max(float(voxel_size.max()), truncation * 0.08)
    boundary = np.zeros_like(observed)
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower_tuple, upper_tuple = tuple(lower), tuple(upper)
        crosses = (
            observed[lower_tuple]
            & observed[upper_tuple]
            & (
                ((sdf[lower_tuple] < 0) & (sdf[upper_tuple] >= 0))
                | ((sdf[upper_tuple] < 0) & (sdf[lower_tuple] >= 0))
            )
        )
        boundary[lower_tuple] |= crosses
        boundary[upper_tuple] |= crosses
    negative = observed & (sdf < 0.0)
    positive = observed & (sdf > 0.0)
    neutral = observed & ((np.abs(sdf) <= surface_band) | boundary)
    unknown = ~observed
    flat_indices = np.arange(sdf.size, dtype=np.int64)
    ijk = np.column_stack(np.unravel_index(flat_indices, sdf.shape)).astype(np.float32)
    points = bbox_min[None] + ijk * voxel_size[None]
    normalized = np.clip(sdf.reshape(-1) / max(truncation, 1e-8), -1.0, 1.0)
    negative_strength = np.clip(-normalized, 0.0, 1.0)
    positive_strength = np.clip(normalized, 0.0, 1.0)
    strength = np.maximum(negative_strength, positive_strength)
    colors = np.zeros((sdf.size, 3), dtype=np.uint8)
    colors[:, 0] = (255 * negative_strength + 245 * (1 - strength)).astype(np.uint8)
    colors[:, 1] = (
        75 * negative_strength + 245 * (1 - strength) + 110 * positive_strength
    ).astype(np.uint8)
    colors[:, 2] = (
        45 * negative_strength + 255 * positive_strength + 210 * (1 - strength)
    ).astype(np.uint8)
    colors[~observed.reshape(-1)] = [150, 150, 150]
    colors[boundary.reshape(-1)] = [255, 220, 0]
    output_dir.mkdir(parents=True, exist_ok=True)
    full_npz = output_dir / "tsdf_full.npz"
    np.savez_compressed(
        full_npz,
        tsdf=sdf,
        observed=observed,
        boundary_crossings=boundary,
        grid_min=bbox_min,
        grid_max=bbox_max,
        voxel_size=voxel_size,
        truncation_m=np.asarray(truncation, dtype=np.float32),
        negative_band_m=np.asarray(negative_band, dtype=np.float32),
        max_voxel_spacing_m=np.asarray(voxel_size.max(), dtype=np.float32),
        anchor_world=anchor,
    )
    full_ply = output_dir / "tsdf_full_colored.ply"
    write_binary_ply_points(full_ply, points, colors)
    categories = {
        "negative": (negative, (220, 40, 40), output_dir / "tsdf_negative.ply"),
        "positive_observed": (
            positive,
            (45, 125, 255),
            output_dir / "tsdf_positive_observed.ply",
        ),
        "neutral_boundary": (
            neutral,
            (255, 220, 0),
            output_dir / "tsdf_boundary_crossings.ply",
        ),
        "unknown": (unknown, (150, 150, 150), output_dir / "tsdf_unknown.ply"),
    }
    category_manifest: dict[str, Any] = {}
    for name, (mask, color, path) in categories.items():
        indices = np.flatnonzero(mask.reshape(-1))
        point_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(indices), 1))
        write_binary_ply_points(path, points[indices], point_colors)
        category_manifest[name] = {
            "path": path,
            "full_voxel_count": int(len(indices)),
            "written_points": int(len(indices)),
            "rgb": list(color),
        }
    isosurface_path = output_dir / "tsdf_zero_isosurface.ply"
    isosurface_error = None
    try:
        from skimage.measure import marching_cubes

        surface_vertices, surface_faces, _, _ = marching_cubes(
            sdf,
            level=0.0,
            spacing=tuple(float(value) for value in voxel_size),
            mask=observed,
        )
        surface_vertices += bbox_min[None]
        trimesh.Trimesh(
            vertices=surface_vertices,
            faces=surface_faces,
            process=False,
        ).export(isosurface_path)
    except Exception as error:  # The manifest records grids with no zero crossing.
        isosurface_error = f"{type(error).__name__}: {error}"
    anchor_ijk = np.rint((anchor - bbox_min) / np.maximum(voxel_size, 1e-8)).astype(np.int64)
    anchor_ijk = np.clip(anchor_ijk, 0, dim - 1)

    def slice_rgb(values: Any, known: Any) -> Any:
        values = np.asarray(values, dtype=np.float32)
        known = np.asarray(known, dtype=bool)
        norm = np.clip(values / max(truncation, 1e-8), -1.0, 1.0)
        neg = np.clip(-norm, 0.0, 1.0)
        pos = np.clip(norm, 0.0, 1.0)
        local_strength = np.maximum(neg, pos)
        rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (255 * neg + 245 * (1 - local_strength)).astype(np.uint8)
        rgb[..., 1] = (
            70 * neg + 245 * (1 - local_strength) + 100 * pos
        ).astype(np.uint8)
        rgb[..., 2] = (
            40 * neg + 255 * pos + 210 * (1 - local_strength)
        ).astype(np.uint8)
        rgb[~known] = [150, 150, 150]
        return rgb

    slice_specs = {
        "xy": (
            sdf[:, :, anchor_ijk[2]],
            observed[:, :, anchor_ijk[2]],
            (anchor_ijk[0], anchor_ijk[1]),
        ),
        "xz": (
            sdf[:, anchor_ijk[1], :],
            observed[:, anchor_ijk[1], :],
            (anchor_ijk[0], anchor_ijk[2]),
        ),
        "yz": (
            sdf[anchor_ijk[0], :, :],
            observed[anchor_ijk[0], :, :],
            (anchor_ijk[1], anchor_ijk[2]),
        ),
    }
    slice_paths: dict[str, Path] = {}
    for name, (values, known, marker) in slice_specs.items():
        rgb = np.transpose(slice_rgb(values, known), (1, 0, 2))[::-1]
        image = Image.fromarray(rgb).resize((512, 512), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        marker_x = int(round(marker[0] / max(values.shape[0] - 1, 1) * 511))
        marker_y = 511 - int(round(marker[1] / max(values.shape[1] - 1, 1) * 511))
        draw.line((marker_x - 8, marker_y, marker_x + 8, marker_y), fill=(0, 255, 0), width=3)
        draw.line((marker_x, marker_y - 8, marker_x, marker_y + 8), fill=(0, 255, 0), width=3)
        path = output_dir / f"slices_{name}.png"
        image.save(path)
        slice_paths[name] = path
    manifest = {
        "method": TSDF_METHOD,
        "source_sdf_meta": sdf_meta_path,
        "source_sdf_npy": sdf_path,
        "source_observed_npy": observed_path,
        "full_tsdf_npz": full_npz,
        "full_colored_ply": full_ply,
        "zero_isosurface_ply": isosurface_path if isosurface_error is None else None,
        "zero_isosurface_error": isosurface_error,
        "slices": slice_paths,
        "categories": category_manifest,
        "grid_shape": list(sdf.shape),
        "grid_min": bbox_min,
        "grid_max": bbox_max,
        "voxel_size": voxel_size,
        "max_voxel_spacing_m": float(voxel_size.max()),
        "truncation_m": truncation,
        "negative_band_m": negative_band,
        "surface_band_m": surface_band,
        "anchor_world": anchor,
        "anchor_ijk": anchor_ijk,
        "sign_convention": SIGN_CONVENTION,
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


def export_mesh(path: Path, vertices: Any, faces: Any) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def save_keypoint_overlay(image_path: Path, keypoints: Any, output_path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for index, (x, y, confidence) in enumerate(keypoints):
        if confidence <= 0.0:
            continue
        radius = 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 60, 40))
        draw.text((x + 5, y - 5), str(index), fill=(255, 255, 0))
    image.save(output_path)


def render_body_overlay(
    image_path: Path,
    vertices_camera: Any,
    faces: Any,
    intrinsics: Any,
    output_path: Path,
) -> None:
    import numpy as np
    import pyrender
    import trimesh
    from PIL import Image

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]
    body = trimesh.Trimesh(vertices=vertices_camera, faces=faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        baseColorFactor=(0.2, 0.75, 0.95, 1.0),
    )
    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[0.8, 0.8, 0.8],
    )
    scene.add(pyrender.Mesh.from_trimesh(body, material=material, smooth=True))
    camera = pyrender.IntrinsicsCamera(
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
        znear=0.05,
        zfar=20.0,
    )
    scene.add(camera, pose=np.diag([1.0, -1.0, -1.0, 1.0]))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    mask = depth > 0.0
    composite = image.copy()
    composite[mask] = (
        0.35 * image[mask].astype(np.float32) + 0.65 * color[mask, :3].astype(np.float32)
    ).astype(np.uint8)
    Image.fromarray(composite).save(output_path)


def patch_torchgeometry_for_modern_torch(torch: Any) -> None:
    """Make VPoser v1's torchgeometry 0.1.2 work with PyTorch 2.

    The released conversion uses arithmetic on boolean masks. PyTorch 2 rejects
    ``1 - bool_tensor``; logical masks are otherwise mathematically identical.
    """
    import torchgeometry as tgm
    from torchgeometry.core import conversions

    if getattr(conversions.rotation_matrix_to_quaternion, "_prox_fixed", False):
        return

    def rotation_matrix_to_quaternion(rotation_matrix: Any, eps: float = 1e-6) -> Any:
        if not torch.is_tensor(rotation_matrix):
            raise TypeError(f"Expected a torch.Tensor, got {type(rotation_matrix)}")
        if rotation_matrix.ndim != 3 or rotation_matrix.shape[-2:] != (3, 4):
            raise ValueError(
                f"Expected an N x 3 x 4 rotation matrix, got {rotation_matrix.shape}"
            )
        matrix = rotation_matrix.transpose(1, 2)
        d2_negative = matrix[:, 2, 2] < eps
        d0_larger = matrix[:, 0, 0] > matrix[:, 1, 1]
        d0_less_negative_d1 = matrix[:, 0, 0] < -matrix[:, 1, 1]

        t0 = 1 + matrix[:, 0, 0] - matrix[:, 1, 1] - matrix[:, 2, 2]
        q0 = torch.stack(
            [
                matrix[:, 1, 2] - matrix[:, 2, 1],
                t0,
                matrix[:, 0, 1] + matrix[:, 1, 0],
                matrix[:, 2, 0] + matrix[:, 0, 2],
            ],
            dim=-1,
        )
        t1 = 1 - matrix[:, 0, 0] + matrix[:, 1, 1] - matrix[:, 2, 2]
        q1 = torch.stack(
            [
                matrix[:, 2, 0] - matrix[:, 0, 2],
                matrix[:, 0, 1] + matrix[:, 1, 0],
                t1,
                matrix[:, 1, 2] + matrix[:, 2, 1],
            ],
            dim=-1,
        )
        t2 = 1 - matrix[:, 0, 0] - matrix[:, 1, 1] + matrix[:, 2, 2]
        q2 = torch.stack(
            [
                matrix[:, 0, 1] - matrix[:, 1, 0],
                matrix[:, 2, 0] + matrix[:, 0, 2],
                matrix[:, 1, 2] + matrix[:, 2, 1],
                t2,
            ],
            dim=-1,
        )
        t3 = 1 + matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
        q3 = torch.stack(
            [
                t3,
                matrix[:, 1, 2] - matrix[:, 2, 1],
                matrix[:, 2, 0] - matrix[:, 0, 2],
                matrix[:, 0, 1] - matrix[:, 1, 0],
            ],
            dim=-1,
        )

        masks = [
            d2_negative & d0_larger,
            d2_negative & ~d0_larger,
            ~d2_negative & d0_less_negative_d1,
            ~d2_negative & ~d0_less_negative_d1,
        ]
        masks = [mask[:, None].type_as(q0) for mask in masks]
        quaternion = q0 * masks[0] + q1 * masks[1] + q2 * masks[2] + q3 * masks[3]
        denominator = (
            t0[:, None].repeat(1, 4) * masks[0]
            + t1[:, None].repeat(1, 4) * masks[1]
            + t2[:, None].repeat(1, 4) * masks[2]
            + t3[:, None].repeat(1, 4) * masks[3]
        )
        return quaternion / torch.sqrt(denominator) * 0.5

    rotation_matrix_to_quaternion._prox_fixed = True
    conversions.rotation_matrix_to_quaternion = rotation_matrix_to_quaternion
    tgm.rotation_matrix_to_quaternion = rotation_matrix_to_quaternion


def patch_prox_grid_sample(fitting: Any, torch: Any) -> None:
    """Preserve PROX's PyTorch-1.0 SDF sampling coordinate convention."""
    native_grid_sample = torch.nn.functional.grid_sample

    def prox_grid_sample(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("align_corners", True)
        return native_grid_sample(*args, **kwargs)

    class ProxFunctional:
        grid_sample = staticmethod(prox_grid_sample)

    fitting.F = ProxFunctional


def patch_prox_numerical_stability(fitting: Any, torch: Any) -> None:
    """Define released-PROX contact behavior at floating-point edge cases.

    The original contact implementation can feed a value just above one to
    asin, and takes the mean of an empty tensor when no contact normals pass
    its angle filter.  Both cases turn the complete LBFGS parameter vector into
    NaNs.  These patches leave every ordinary, non-empty loss unchanged.
    """
    if not getattr(fitting.torch, "_prox_safe_asin", False):
        native_torch = torch

        class ProxTorchProxy:
            _prox_safe_asin = True

            def __getattr__(self, name: str) -> Any:
                return getattr(native_torch, name)

            @staticmethod
            def asin(value: Any) -> Any:
                return native_torch.asin(value.clamp(min=-1.0, max=1.0))

        fitting.torch = ProxTorchProxy()

    robustifier_class = fitting.utils.GMoF_unscaled
    if not getattr(robustifier_class.forward, "_prox_empty_safe", False):
        native_forward = robustifier_class.forward

        def empty_safe_forward(self: Any, residual: Any) -> Any:
            if residual.numel() == 0:
                # A differentiable zero makes "no compatible contact" contribute
                # zero loss rather than NaN, which is the mathematical empty sum.
                return residual.sum().reshape(1)
            return native_forward(self, residual)

        empty_safe_forward._prox_empty_safe = True
        robustifier_class.forward = empty_safe_forward


def create_fixed_camera(intrinsics: Any, torch: Any, device: Any) -> Any:
    from camera import create_camera

    camera = create_camera(
        focal_length_x=float(intrinsics[0, 0]),
        focal_length_y=float(intrinsics[1, 1]),
        center=torch.as_tensor(
            intrinsics[:2, 2], dtype=torch.float32
        ).reshape(1, 2),
        batch_size=1,
        dtype=torch.float32,
    ).to(device)
    with torch.no_grad():
        camera.rotation.copy_(torch.eye(3, device=device).reshape(1, 3, 3))
        camera.translation.zero_()
    camera.rotation.requires_grad_(False)
    camera.translation.requires_grad_(False)
    return camera


def initialize_original_prox(
    body_model: Any,
    vposer: Any,
    pose_embedding: Any,
    keypoints: Any,
    intrinsics: Any,
    config: dict[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Run PROX's initialization once to localize the human-centered TSDF."""
    import fitting
    from optimizers import optim_factory

    patch_prox_grid_sample(fitting, torch)
    body_model.reset_params()
    with torch.no_grad():
        pose_embedding.zero_()
    pose_embedding.requires_grad_(False)
    camera = create_fixed_camera(intrinsics, torch, device)
    gt_joints = torch.as_tensor(
        keypoints[None, :, :2], dtype=torch.float32, device=device
    )
    edge_indices = [[5, 12], [2, 9]]
    init_translation = fitting.guess_init(
        body_model,
        gt_joints,
        edge_indices,
        use_vposer=True,
        vposer=vposer,
        pose_embedding=pose_embedding,
        model_type="smplx",
        focal_length=float(intrinsics[0, 0]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        body_model.transl.copy_(init_translation)
    body_model.transl.requires_grad_(True)
    body_model.global_orient.requires_grad_(True)
    camera_loss = fitting.create_loss(
        "camera_init",
        trans_estimation=init_translation,
        init_joints_idxs=torch.as_tensor(
            [9, 12, 2, 5], dtype=torch.long, device=device
        ),
        depth_loss_weight=100.0,
        camera_mode="fixed",
        dtype=torch.float32,
    ).to(device)
    camera_loss.trans_estimation[:] = init_translation
    optimizer_options = {
        "optim_type": config["optim_type"],
        "lr": float(config["lr"]),
        "maxiters": int(config["maxiters"]),
        "ftol": float(config["ftol"]),
        "gtol": float(config["gtol"]),
    }
    monitor_options = {
        "model_type": "smplx",
        "maxiters": int(config["maxiters"]),
        "ftol": float(config["ftol"]),
        "gtol": float(config["gtol"]),
        "visualize": False,
    }
    parameters = [body_model.transl, body_model.global_orient]
    with fitting.FittingMonitor(**monitor_options) as monitor:
        optimizer, create_graph = optim_factory.create_optimizer(
            parameters, **optimizer_options
        )
        closure = monitor.create_fitting_closure(
            optimizer,
            body_model,
            camera,
            gt_joints,
            camera_loss,
            create_graph=create_graph,
            use_vposer=True,
            vposer=vposer,
            pose_embedding=pose_embedding,
            return_full_pose=False,
            return_verts=False,
        )
        before = float(closure(backward=False).detach().item())
        value = monitor.run_fitting(
            optimizer,
            closure,
            parameters,
            body_model,
            use_vposer=True,
            pose_embedding=pose_embedding,
            vposer=vposer,
        )
        after = (
            float(value)
            if value is not None
            else float(closure(backward=False).detach().item())
        )
    pose_embedding.requires_grad_(True)
    return {
        "estimated_translation": init_translation.detach().cpu().numpy(),
        "aligned_translation": body_model.transl.detach().cpu().numpy(),
        "aligned_global_orient": body_model.global_orient.detach().cpu().numpy(),
        "loss_before": before,
        "loss_after": after,
    }


def create_prox_runtime(
    intrinsics: Any,
    config: dict[str, Any],
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    """Create the objects that upstream PROX's ``main.py`` normally creates."""
    from camera import create_camera
    from prior import create_prior
    camera = create_camera(
        focal_length_x=float(intrinsics[0, 0]),
        focal_length_y=float(intrinsics[1, 1]),
        center=torch.as_tensor(intrinsics[:2, 2], dtype=torch.float32).reshape(1, 2),
        batch_size=1,
        dtype=torch.float32,
    ).to(device)
    camera.rotation.requires_grad_(False)
    camera.translation.requires_grad_(False)

    def prior(name: str, fallback: str = "l2", **kwargs: Any) -> Any:
        return create_prior(
            prior_type=str(config.get(name, fallback)),
            dtype=torch.float32,
            **kwargs,
        ).to(device)

    joint_weights = torch.ones((1, 118), dtype=torch.float32, device=device)
    joint_weights[:, [int(index) for index in config["joints_to_ign"]]] = 0.0
    return {
        "camera": camera,
        "joint_weights": joint_weights,
        "body_pose_prior": prior("body_prior_type"),
        "jaw_prior": prior("jaw_prior_type"),
        "left_hand_prior": prior(
            "left_hand_prior_type",
            use_left_hand=True,
            num_gaussians=int(config["num_pca_comps"]),
        ),
        "right_hand_prior": prior(
            "right_hand_prior_type",
            use_right_hand=True,
            num_gaussians=int(config["num_pca_comps"]),
        ),
        "shape_prior": prior("shape_prior_type"),
        "expr_prior": prior("expr_prior_type"),
        "angle_prior": create_prior(prior_type="angle", dtype=torch.float32).to(device),
    }


def run_upstream_prox(
    image_path: Path,
    keypoints: Any,
    body_model: Any,
    intrinsics: Any,
    scene_id: str,
    scene_dir: Path,
    sdf_dir: Path,
    cam2world_dir: Path,
    result_path: Path,
    mesh_path: Path,
    config: dict[str, Any],
    torch: Any,
    device: Any,
) -> None:
    """Call the released PROX optimizer; no loss or stage is reimplemented here."""
    import numpy as np
    from PIL import Image
    import fitting
    from fit_single_frame import fit_single_frame

    patch_prox_grid_sample(fitting, torch)
    patch_prox_numerical_stability(fitting, torch)
    runtime = create_prox_runtime(intrinsics, config, torch, device)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    kwargs = dict(config)
    kwargs.update(
        {
            "batch_size": 1,
            "body_tri_idxs": [[5, 12], [2, 9]],
            "init_joints_idxs": [9, 12, 2, 5],
            "side_view_thsh": 25.0,
            "depth_loss_weight": 100.0,
            "model_type": "smplx",
            "camera_mode": "fixed",
            "focal_length_x": float(intrinsics[0, 0]),
            "focal_length_y": float(intrinsics[1, 1]),
            "use_cuda": True,
            "use_vposer": True,
            "use_joints_conf": True,
            "use_hands": True,
            "use_face": True,
            "interactive": False,
            "render_results": False,
            "visualize": False,
            "save_meshes": True,
            "vposer_ckpt": str(VPOSER_DIR),
            "part_segm_fn": str(PART_SEGMENTATION),
            "contact_body_parts": CONTACT_BODY_PARTS,
            "body_segments_dir": str(BODY_SEGMENTS_DIR),
            "scene_dir": str(scene_dir),
            "sdf_dir": str(sdf_dir),
            "cam2world_dir": str(cam2world_dir),
            "load_scene": True,
            "contact": True,
            "sdf_penetration": True,
            "dtype": torch.float32,
        }
    )
    # These are wrapper/main-only configuration keys, not fit_single_frame inputs.
    for key in (
        "output_folder",
        "model_folder",
        "prior_folder",
        "result_folder",
        "gender",
        "dataset",
        "flip",
        "float_dtype",
        "joints_to_ign",
    ):
        kwargs.pop(key, None)
    fit_single_frame(
        image,
        np.asarray(keypoints, dtype=np.float32)[None],
        None,
        None,
        scene_id,
        body_model,
        runtime.pop("camera"),
        runtime.pop("joint_weights"),
        result_fn=str(result_path),
        mesh_fn=str(mesh_path),
        **runtime,
        **kwargs,
    )


def load_selected_prox_mesh(
    result_path: Path,
    body_model: Any,
    torch: Any,
) -> tuple[dict[str, Any], Any, Any]:
    """Rebuild the mesh from PROX's selected result parameter dictionary.

    The released function writes the selected orientation to ``result.pkl``
    but can leave its convenience PLY at the last tried orientation. Treating
    the selected parameter dictionary as authoritative avoids that mismatch
    without changing the optimizer.
    """
    import numpy as np

    with result_path.open("rb") as handle:
        result = pickle.load(handle, encoding="latin1")
    non_finite = [
        key
        for key, value in result.items()
        if isinstance(value, np.ndarray) and not np.isfinite(value).all()
    ]
    if non_finite:
        raise RuntimeError(f"Upstream PROX returned non-finite parameter(s): {non_finite}")
    named = dict(body_model.named_parameters())
    parameters = {
        key: torch.as_tensor(value, dtype=torch.float32, device=body_model.transl.device)
        for key, value in result.items()
        if key in named
    }
    body_model.reset_params(**parameters)
    body_pose = torch.as_tensor(
        result["body_pose"], dtype=torch.float32, device=body_model.transl.device
    )
    with torch.no_grad():
        output = body_model(body_pose=body_pose, return_verts=True)
    return (
        result,
        output.vertices[0].detach().cpu().numpy().astype(np.float32),
        np.asarray(body_model.faces, dtype=np.int64),
    )


def load_trimesh(path: Path) -> Any:
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Scene mesh is empty: {path}")
    return mesh


def load_module03_crop(
    interaction_name: str,
    source_intrinsics: Any,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    interaction_root = MODULE03_OUTPUT / interaction_name
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
    return trimesh.Trimesh(
        vertices=vertices[used_vertices],
        faces=inverse.reshape(-1, 3),
        process=False,
    )


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
    """Project sampled crop surfaces without doing an occlusion render."""
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
    """Keep one high-coverage representative per near-identical camera pose."""
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

    centers = np.stack([camera_center(camera) for camera in cameras]).astype(
        np.float32
    )
    rotations = np.stack(
        [
            np.asarray(camera.rotation_world_to_camera, dtype=np.float32)
            for camera in cameras
        ]
    )
    source_index = next(
        (
            index
            for index, camera in enumerate(cameras)
            if camera.name == source_camera.name
        ),
        None,
    )
    order = sorted(
        range(len(cameras)),
        key=lambda index: (-int(projected_counts[index]), index),
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
        traces = np.einsum(
            "nij,ij->n", rotations[retained_array], rotations[index]
        )
        rotation_angles = np.arccos(
            np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
        )
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

    samples = sample_mesh_surface(crop_mesh, COVERAGE_SURFACE_SAMPLES, RANDOM_SEED)
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
    projection_counts = [
        projected_counts_all[index] for index in projection_candidate_indices
    ]
    retained_local_indices, deduplication = deduplicate_camera_poses(
        projection_cameras,
        projection_counts,
        source_camera,
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
    log(
        "    pose dedup kept "
        f"{len(prepass_indices)}/{len(projection_candidate_indices)} projected "
        f"camera(s) (removed {deduplication['removed_camera_count']})"
    )
    if not prepass_indices:
        raise RuntimeError("No registered camera projects the Module 03 scene crop.")

    height = scaled[prepass_indices[0]].height
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0])
    scene.add(pyrender.Mesh.from_trimesh(full_scene_mesh, smooth=False))
    renderer = pyrender.OffscreenRenderer(
        viewport_width=COVERAGE_RENDER_WIDTH,
        viewport_height=height,
    )
    eligible_cameras: list[Camera] = []
    visibility_rows: list[Any] = []
    projected_counts: list[int] = []
    visible_counts: list[int] = []
    try:
        for prepass_number, camera_index in enumerate(prepass_indices, start=1):
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
            if (
                visible_count >= COVERAGE_MIN_VISIBLE_SAMPLES
                or original.name == source_camera.name
            ):
                eligible_cameras.append(original)
                visibility_rows.append(visible)
                projected_counts.append(projected_count)
                visible_counts.append(visible_count)
            if (
                prepass_number % 100 == 0
                or prepass_number == len(prepass_indices)
            ):
                log(
                    f"    coverage prepass {prepass_number}/{len(prepass_indices)} "
                    f"eligible={len(eligible_cameras)}"
                )
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

    centers = np.stack(
        [camera_center(camera) for camera in eligible_cameras], axis=0
    )
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
            diversity = np.min(
                1.0 - directions[tied] @ selected_directions.T,
                axis=1,
            )
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
    source_intrinsics: Any,
    initial_vertices_world: Any,
    debug_dir: Path,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np

    crop = load_module03_crop(interaction_name, source_intrinsics)
    rendered_cameras, depths = render_scene_depths(
        full_scene_mesh,
        [source_camera],
        debug_dir / "source_crop_view",
    )
    crop_points, depth_stats = unproject_depth_crop(
        depths[0],
        rendered_cameras[0],
        crop["xyxy_source_pixels"],
        source_camera,
    )
    initial_vertices_world = np.asarray(initial_vertices_world, dtype=np.float32)
    combined_min = np.minimum(
        crop_points.min(axis=0), initial_vertices_world.min(axis=0)
    )
    combined_max = np.maximum(
        crop_points.max(axis=0), initial_vertices_world.max(axis=0)
    )
    bbox_min = combined_min - ROI_PADDING_M
    bbox_max = combined_max + ROI_PADDING_M
    crop_mesh = crop_mesh_to_bounds(full_scene_mesh, bbox_min, bbox_max)
    roi_center = (bbox_min + bbox_max) * 0.5

    preview_count = min(100_000, crop_points.shape[0])
    if crop_points.shape[0] > preview_count:
        indices = np.linspace(0, crop_points.shape[0] - 1, preview_count).astype(
            np.int64
        )
        preview = crop_points[indices]
    else:
        preview = crop_points
    write_binary_ply_points(
        debug_dir / "source_visible_crop_points.ply",
        preview,
        np.tile(np.asarray([40, 220, 80], dtype=np.uint8), (len(preview), 1)),
    )
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
        "uses_module03_contact_masks": False,
        "uses_module03_contact_parts": False,
    }
    save_json(debug_dir / "scene_crop_volume.json", metadata)
    return bbox_min, bbox_max, crop_mesh, metadata


def run_interaction(name: str, torch: Any, config: dict[str, Any]) -> None:
    import numpy as np

    started = time.time()
    inputs = resolve_interaction_inputs(name)
    final_root = OUTPUT_ROOT / name
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True)
    log(f"\n[*] Running crop-volume PROX for {name}")
    data = load_interaction_data(inputs)
    openpose = run_openpose(inputs.human_image, final_root / "openpose")
    device = torch.device("cuda:0")
    body_model = create_body_model(device, torch, config)
    vposer, pose_embedding = load_vposer(device)
    initialization = initialize_original_prox(
        body_model,
        vposer,
        pose_embedding,
        openpose["keypoints"],
        data["intrinsics"],
        config,
        torch,
        device,
    )
    initial_pose = decoded_body_pose(vposer, pose_embedding)
    initial_vertices_camera, _ = body_vertices_and_joints(
        body_model, initial_pose, torch
    )
    source_camera = data["source_camera"]
    initial_vertices_world = camera_to_world(initial_vertices_camera, source_camera)
    export_mesh(
        final_root / "initial_smplx_camera.ply",
        initial_vertices_camera,
        body_model.faces,
    )
    if openpose["overlay"] is not None:
        shutil.copy2(openpose["overlay"], final_root / "keypoints_overlay.png")
    else:
        save_keypoint_overlay(
            inputs.human_image,
            openpose["body25"],
            final_root / "keypoints_overlay.png",
        )
    log("  deriving Module 03 crop volume from source-visible scene surfaces")
    full_scene_mesh = load_trimesh(inputs.mesh)
    crop_debug = final_root / "debug" / "scene_crop"
    bbox_min, bbox_max, scene_mesh, crop_metadata = derive_crop_volume(
        name,
        full_scene_mesh,
        source_camera,
        data["intrinsics"],
        initial_vertices_world,
        crop_debug,
    )
    roi_center = np.asarray(crop_metadata["roi_center_world"], dtype=np.float32)
    log(
        "  selecting DSLR views by visible crop-surface coverage "
        f"from {len(data['cameras'])} registered camera(s)"
    )
    observing, coverage_metadata = select_cameras_by_surface_coverage(
        full_scene_mesh,
        scene_mesh,
        data["cameras"],
        source_camera,
        roi_center,
        crop_debug,
    )
    log(
        f"  selected {len(observing)} TSDF view(s), "
        f"coverage={coverage_metadata['covered_fraction']:.3f}"
    )
    render_cameras, depths = render_scene_depths(
        full_scene_mesh,
        observing,
        final_root / "debug" / "tsdf_views",
    )
    log(
        f"  building crop-volume TSDF dim={SDF_GRID_DIM} "
        f"trunc={SDF_TRUNCATION_M:.3f}m"
    )
    sdf_meta_path, sdf_metadata = build_visibility_tsdf(
        inputs.scene_id,
        render_cameras,
        int(coverage_metadata["eligible_camera_count"]),
        depths,
        bbox_min,
        bbox_max,
        final_root / "sdf",
        inputs.mesh,
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
        }
    )
    save_json(sdf_meta_path, sdf_metadata)
    del depths
    log("  exporting complete TSDF debug artifacts")
    tsdf_debug = write_tsdf_debug(
        sdf_meta_path,
        final_root / "debug" / "tsdf",
        roi_center,
    )
    scene_dir = final_root / "scene"
    cam2world_dir = final_root / "cam2world"
    scene_dir.mkdir()
    cam2world_dir.mkdir()
    scene_mesh.export(scene_dir / f"{inputs.scene_id}.ply")
    save_json(
        cam2world_dir / f"{inputs.scene_id}.json",
        camera_to_world_transform(source_camera),
    )
    final_camera_path = final_root / "final_smplx_camera.ply"
    result_path = final_root / "result.pkl"
    log("  calling upstream PROX fit_single_frame()")
    run_upstream_prox(
        inputs.human_image,
        openpose["keypoints"],
        body_model,
        data["intrinsics"],
        inputs.scene_id,
        scene_dir,
        final_root / "sdf",
        cam2world_dir,
        result_path,
        final_camera_path,
        config,
        torch,
        device,
    )
    result, final_vertices_camera, final_faces = load_selected_prox_mesh(
        result_path, body_model, torch
    )
    export_mesh(final_camera_path, final_vertices_camera, final_faces)
    final_vertices_world = camera_to_world(final_vertices_camera, source_camera)
    export_mesh(
        final_root / "final_smplx_world.ply",
        final_vertices_world,
        final_faces,
    )
    render_body_overlay(
        inputs.human_image,
        final_vertices_camera,
        final_faces,
        data["intrinsics"],
        final_root / "overlay.png",
    )
    metadata = {
        "interaction_name": name,
        "variant": "module03_scene_crop_coverage_views_pose_dedup_v2",
        "scene_id": inputs.scene_id,
        "camera_name": inputs.camera_name,
        "inputs": {
            "human_image": inputs.human_image,
            "input_scene": inputs.input_scene,
            "scene_mesh": inputs.mesh,
            "module03_crop": crop_metadata["module03_crop_image"],
            "module03_contact_spec": crop_metadata["module03_contact_spec"],
        },
        "intrinsics": data["intrinsics"],
        "camera_to_world": camera_to_world_transform(source_camera),
        "initialization": "original PROX zero-VPoser initialization",
        "contact_body_parts": CONTACT_BODY_PARTS,
        "uses_module03_contact_information": False,
        "module03_role": "2D scene crop and adjusted crop intrinsics only",
        "scene_crop": crop_metadata,
        "camera_coverage": coverage_metadata,
        "sdf": sdf_metadata,
        "tsdf_debug_manifest": tsdf_debug,
        "tsdf_localization_initialization": initialization,
        "runtime_seconds": float(time.time() - started),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "optimization_config": config,
    }
    save_json(final_root / "metadata.json", metadata)
    log(f"  wrote {final_root}")
    del body_model, vposer, pose_embedding, scene_mesh, full_scene_mesh
    gc.collect()
    torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PROX with Module 03 crop-volume TSDF camera coverage."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--interaction_name",
        "--interaction-name",
        dest="interaction_name",
        default="interaction_02",
    )
    selection.add_argument(
        "--all_interactions",
        "--all-interactions",
        dest="all_interactions",
        action="store_true",
    )
    return parser.parse_args(argv)


def load_original_prox_config() -> dict[str, Any]:
    """Load PROX.yaml through upstream's typed command-line parser."""
    from cmd_parser import parse_config

    original_argv = sys.argv
    try:
        # parse_config's argv parameter is unused in the released PROX code, so
        # temporarily provide the same command line used by upstream main.py.
        sys.argv = ["SMPLifyX", "--config", str(PROX_CONFIG)]
        return parse_config()
    finally:
        sys.argv = original_argv


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if str(PROX_ROOT / "prox") not in sys.path:
        sys.path.insert(0, str(PROX_ROOT / "prox"))
    # Import upstream PROX/psbody before selecting EGL for our headless debug
    # renderers. This keeps PROX's real Mesh implementation instead of stubbing it.
    import fit_single_frame as _upstream_fit_single_frame

    del _upstream_fit_single_frame

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PROX requires CUDA, but CUDA is unavailable.")
    names = resolve_interactions(args)
    config = load_original_prox_config()
    validate_prox_assets(config)
    validate_openpose()
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"[*] Selected {len(names)} interaction(s): " + ", ".join(names))
    for name in names:
        run_interaction(name, torch, config)
    log(f"\n[*] Crop-volume PROX finished for {len(names)} interaction(s).")


if __name__ == "__main__":
    main()
