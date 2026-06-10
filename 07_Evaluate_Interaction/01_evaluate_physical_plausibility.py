from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import smplx
import torch
import trimesh
from VolumetricSMPL import attach_volume
from pytorch3d.ops import knn_points
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


CONTACT_SEGMENT_BY_BODY_SEGMENT = {
    "left_hand": "left_hand_inner",
    "right_hand": "right_hand_inner",
    "left_leg": "left_leg_contact",
    "right_leg": "right_leg_contact",
    "left_foot": "left_foot_bottom",
    "right_foot": "right_foot_bottom",
    "hips": "hips_contact",
}
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}
BILATERAL_SWAP_MIN_IMPROVEMENT_M = 0.02
CONTACT_SURFACE_SAMPLES_PER_EDGE = 2048
CONTACT_SURFACE_SAMPLE_SEED = 17017
NON_COLLISION_SURFACE_SAMPLE_SEED = 24017
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
COMBINED_CSV_FIELDNAMES = [
    "interaction_name",
    "num_edges",
    "mean_contact_distance_m",
    "ncs",
    "mean_penetration_m",
]

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass
class IdentityCameraContext:
    intrinsics: np.ndarray
    width: int
    height: int
    camera: Any


@dataclass
class InteractionNode:
    raw_node: str
    entity_name: str
    part_name: str
    is_human: bool


@dataclass
class DynamicInteractionEdge:
    node_a: InteractionNode
    node_b: InteractionNode
    moving_node: InteractionNode
    fixed_node: InteractionNode
    moving_part_name: str
    moving_segment_id: str
    moving_segment_name: str
    moving_vertex_ids: np.ndarray
    fixed_points: np.ndarray
    reduction: str
    fixed_face_ids: np.ndarray | None = None
    fixed_vertex_ids: np.ndarray | None = None


@dataclass
class SmplxSegmentCatalog:
    vertex_count: int
    segments: dict[str, np.ndarray]
    body_segment_ids: list[str]
    contact_segment_ids: list[str]

    def get_indices(self, segment_id: str) -> np.ndarray:
        indices = self.segments.get(segment_id)
        if indices is None:
            raise KeyError(f"Unknown SMPL-X segment id '{segment_id}'.")
        return indices

    def get_display_name(self, segment_id: str) -> str:
        if segment_id not in self.segments:
            raise KeyError(f"Unknown SMPL-X segment id '{segment_id}'.")
        return segment_id.replace("_", " ")

    def get_body_segment_id(self, sig_part_name: str) -> str:
        segment_id = slugify_segment_name(sig_part_name)
        if segment_id not in self.body_segment_ids:
            raise KeyError(f"Missing body segment mapping for '{sig_part_name}'.")
        return segment_id

    def get_contact_or_body_segment_id(self, sig_part_name: str) -> str:
        body_segment_id = slugify_segment_name(sig_part_name)
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(body_segment_id)
        if segment_id is not None:
            if segment_id not in self.contact_segment_ids:
                raise KeyError(f"Missing contact segment mapping for '{sig_part_name}'.")
            return segment_id
        return self.get_body_segment_id(sig_part_name)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def normalize_label(text: str) -> str:
    return " ".join(
        text.strip().lower().replace("_", " ").replace("-", " ").split()
    )


def normalize_scene_element(text: str, target_labels: set[str] | None = None) -> str:
    raw = str(text).strip().lower()
    normalized = normalize_label(text)
    labels = target_labels or set()
    if raw == "target_object" or normalized in {"target object", "object"} or normalized in labels:
        return "target_object"
    return normalized


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def resolve_sig_target_label(sig_payload: dict[str, Any]) -> str:
    target_object = sig_payload.get("target_object", {})
    if not isinstance(target_object, dict):
        raise ValueError("SIG must contain target_object.")
    label = str(target_object.get("label", "")).strip()
    if label:
        return label
    raise ValueError("SIG target_object.label must be non-empty.")


def build_default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "input_scene_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        "human_pose_root": PROJECT_DIR
        / "05_Estimate_Human_Pose"
        / "output"
        / interaction_name,
        "sig_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "output"
        / interaction_name
        / "scene_interaction_graph.json",
        "smpl_seg_json": PROJECT_DIR
        / "05_Estimate_Human_Pose"
        / "assets"
        / "smplx_vert_segmentation.json",
        "contact_masks_dir": PROJECT_DIR
        / "04_Estimate_Contact"
        / "output"
        / interaction_name
        / "contact_masks",
        "contact_canvas_path": PROJECT_DIR
        / "04_Estimate_Contact"
        / "output"
        / interaction_name
        / "prompt"
        / "target_scene_crop.png",
        "contact_spec": PROJECT_DIR
        / "04_Estimate_Contact"
        / "output"
        / interaction_name
        / "contact_spec.json",
        "human_mesh_camera": PROJECT_DIR
        / "06_Optimize_Static_Scene"
        / "output"
        / interaction_name
        / "meshes"
        / "frame_0000_camera.ply",
        "optimized_params": PROJECT_DIR
        / "06_Optimize_Static_Scene"
        / "output"
        / interaction_name
        / "debug"
        / "params"
        / "optimized_frame_0000.pt",
        "output_root": SCRIPT_DIR
        / "output"
        / interaction_name
        / "physical_plausibility",
        "smpl_folder": PROJECT_DIR.parent
        / "GVHMR"
        / "inputs"
        / "checkpoints"
        / "body_models",
    }


def resolve_scannet_root(raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (PROJECT_DIR.parent / "Scannet++" / "data").resolve()


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = np.array(
        [
            [float(transforms_payload["fl_x"]), 0.0, float(transforms_payload["cx"])],
            [0.0, float(transforms_payload["fl_y"]), float(transforms_payload["cy"])],
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


def load_colmap_pose(
    colmap_images_path: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for line in colmap_images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(
        f"Could not find camera '{camera_name}' in {colmap_images_path}"
    )


def resolve_scene_paths(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )

    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "image_path": scene_root / image_rel / camera_name,
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
    }


def load_scannet_camera(
    scene_paths: dict[str, Path],
    scene_context: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    transforms_payload = load_json(scene_paths["transforms_path"])
    intrinsics, width, height = build_pinhole_intrinsics(transforms_payload)
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )
    return (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    )


def load_contact_camera(
    contact_spec_path: Path,
    contact_image_path: Path,
) -> tuple[np.ndarray, int, int]:
    if not contact_spec_path.exists():
        raise FileNotFoundError(f"Contact spec JSON not found: {contact_spec_path}")
    if not contact_image_path.exists():
        raise FileNotFoundError(f"Contact canvas image not found: {contact_image_path}")

    payload = load_json(contact_spec_path)
    camera_payload = payload.get("camera")
    if not isinstance(camera_payload, dict):
        raise ValueError(f"Expected camera object in {contact_spec_path}")
    intrinsics = np.asarray(camera_payload["intrinsics_3x3"], dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 intrinsics in {contact_spec_path}, got "
            f"{intrinsics.shape}"
        )

    image = cv2.imread(str(contact_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Failed to read contact canvas image: {contact_image_path}")
    height, width = image.shape[:2]
    return intrinsics, int(width), int(height)


def parse_device(raw_device: str) -> torch.device:
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_mesh(path: Path, process: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh", process=process)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh: {path}")

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if (
        verts.ndim != 2
        or verts.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
    ):
        raise ValueError(
            f"Unexpected mesh shapes for {path}: {verts.shape}, {faces.shape}"
        )
    return verts, faces


def transform_world_to_camera(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return points_world @ rotation_world_to_camera.T + translation_world_to_camera[None]


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be > 0")
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Cannot sample surface points from an empty mesh.")

    triangles = verts[faces]
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_sum = float(np.sum(areas))
    if not np.isfinite(area_sum) or area_sum <= 1e-8:
        raise ValueError("Cannot sample surface points from a zero-area mesh.")

    rng = np.random.default_rng(int(seed))
    probs = areas / area_sum
    face_indices = rng.choice(
        faces.shape[0],
        size=int(num_samples),
        replace=True,
        p=probs,
    )
    tri = triangles[face_indices]

    r1 = rng.random(int(num_samples), dtype=np.float32)
    r2 = rng.random(int(num_samples), dtype=np.float32)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2
    samples = (
        w0[:, None] * tri[:, 0, :]
        + w1[:, None] * tri[:, 1, :]
        + w2[:, None] * tri[:, 2, :]
    )
    return samples.astype(np.float32)


def build_identity_camera(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> IdentityCameraContext:
    rotation = torch.eye(3, dtype=torch.float32, device=device)[None]
    translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    camera_matrix = torch.from_numpy(intrinsics.astype(np.float32))[None].to(device)
    image_size = torch.tensor([[height, width]], dtype=torch.float32, device=device)
    camera = cameras_from_opencv_projection(
        R=rotation,
        tvec=translation,
        camera_matrix=camera_matrix,
        image_size=image_size,
    )
    return IdentityCameraContext(
        intrinsics=intrinsics.astype(np.float32),
        width=int(width),
        height=int(height),
        camera=camera,
    )


def build_rasterizer(camera_ctx: IdentityCameraContext) -> MeshRasterizer:
    device = camera_ctx.camera.device
    raster_settings = RasterizationSettings(
        image_size=(camera_ctx.height, camera_ctx.width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0 if device.type == "cuda" else None,
        max_faces_per_bin=400000 if device.type == "cuda" else None,
    )
    return MeshRasterizer(
        cameras=camera_ctx.camera,
        raster_settings=raster_settings,
    )


def to_meshes(
    verts: np.ndarray,
    faces: np.ndarray,
    device: torch.device,
) -> Meshes:
    verts_t = torch.from_numpy(verts.astype(np.float32)).to(device=device)
    faces_t = torch.from_numpy(faces.astype(np.int64)).to(device=device)
    return Meshes(verts=[verts_t], faces=[faces_t])


def rasterize_depth_and_mask(
    verts: np.ndarray,
    faces: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = to_meshes(verts, faces, device=device)
    rasterizer = build_rasterizer(camera_ctx)
    with torch.no_grad():
        fragments = rasterizer(mesh)
    pix_to_face = (
        fragments.pix_to_face[0, ..., 0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    depth = fragments.zbuf[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    mask = pix_to_face >= 0
    depth[~mask] = 0.0
    return depth, mask, pix_to_face


def filter_faces_to_camera_view(
    verts_camera: np.ndarray,
    faces: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float | None = None,
    border_px: float = 64.0,
) -> np.ndarray:
    triangles = verts_camera[faces]
    z = triangles[..., 2]
    positive = np.any(z > 1e-6, axis=1)
    if max_depth_m is not None:
        positive &= np.any(z < float(max_depth_m), axis=1)
    if not np.any(positive):
        return faces[:0].copy()

    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics[0, 0] * triangles[..., 0] / z_safe + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * triangles[..., 1] / z_safe + intrinsics[1, 2] - 0.5
    u_min = np.min(u, axis=1)
    u_max = np.max(u, axis=1)
    v_min = np.min(v, axis=1)
    v_max = np.max(v, axis=1)

    overlaps = (
        positive
        & (u_max >= -float(border_px))
        & (u_min <= float(width - 1) + float(border_px))
        & (v_max >= -float(border_px))
        & (v_min <= float(height - 1) + float(border_px))
    )
    return faces[overlaps].astype(np.int64)


def compact_mesh_with_vertex_ids(
    verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.shape[0] == 0:
        raise RuntimeError("Cannot compact an empty mesh.")
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_verts = verts[unique_vids].astype(np.float32)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64)
    return compact_verts, compact_faces, unique_vids.astype(np.int64)


def load_smpl_segment_catalog(seg_path: Path) -> SmplxSegmentCatalog:
    raw = load_json(seg_path)
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, dict):
        raise KeyError(
            f"Expected a 'segments' mapping in {seg_path}, but it was not found."
        )

    body_segment_ids = raw.get("body_segment_ids")
    if not isinstance(body_segment_ids, list):
        raise KeyError(
            f"Expected 'body_segment_ids' in {seg_path}, but it was not found."
        )

    contact_segment_ids = raw.get("contact_segment_ids")
    if not isinstance(contact_segment_ids, list):
        raise KeyError(
            f"Expected 'contact_segment_ids' in {seg_path}, but it was not found."
        )

    segments: dict[str, np.ndarray] = {}
    vertex_count = int(raw["vertex_count"])
    for segment_id, indices in raw_segments.items():
        indices_array = np.unique(np.asarray(indices, dtype=np.int64))
        if indices_array.size == 0:
            raise ValueError(f"Segment '{segment_id}' is empty.")
        if indices_array[0] < 0 or indices_array[-1] >= vertex_count:
            raise ValueError(f"Segment '{segment_id}' has out-of-range ids.")
        segments[str(segment_id)] = indices_array

    body_segment_ids = [str(segment_id) for segment_id in body_segment_ids]
    contact_segment_ids = [str(segment_id) for segment_id in contact_segment_ids]
    for segment_id in body_segment_ids + contact_segment_ids:
        if segment_id not in segments:
            raise KeyError(f"Missing SMPL-X segment '{segment_id}' in {seg_path}.")
    for body_segment_id, contact_segment_id in CONTACT_SEGMENT_BY_BODY_SEGMENT.items():
        if body_segment_id not in body_segment_ids:
            raise KeyError(f"Missing body segment '{body_segment_id}' in {seg_path}.")
        if contact_segment_id not in contact_segment_ids:
            raise KeyError(
                f"Missing contact segment '{contact_segment_id}' in {seg_path}."
            )

    return SmplxSegmentCatalog(
        vertex_count=vertex_count,
        segments=segments,
        body_segment_ids=body_segment_ids,
        contact_segment_ids=contact_segment_ids,
    )


def iter_sig_interactions(sig_payload: dict[str, Any]) -> list[dict[str, Any]]:
    interactions = sig_payload.get("interaction_edges", [])
    if not isinstance(interactions, list):
        raise ValueError("SIG must contain a list field named 'interaction_edges'.")
    edges: list[dict[str, Any]] = []
    for interaction in interactions:
        if not isinstance(interaction, dict):
            continue
        body_part = normalize_label(str(interaction.get("human_part", "")))
        scene_element = normalize_scene_element(str(interaction.get("scene_element", "")))
        if not body_part or not scene_element:
            continue
        edges.append({**interaction, "body_part": body_part, "scene_element": scene_element})
    return edges


def load_contact_mask_for_part(
    contact_masks_dir: Path,
    human_part: str,
    expected_hw: tuple[int, int],
) -> np.ndarray:
    slug = slugify_segment_name(human_part)
    path = (contact_masks_dir / f"{slug}.png").resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing contact mask for human part '{human_part}': "
            f"expected file '{path}'."
        )
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Failed to read contact mask: {path}")
    if mask.shape != expected_hw:
        raise ValueError(
            f"Contact mask shape mismatch for '{human_part}' at {path}: "
            f"got {mask.shape[::-1]}, expected {expected_hw[::-1]}"
        )
    return mask > 127


def _component_bbox_gap_px(
    component_a: dict[str, Any],
    component_b: dict[str, Any],
) -> float:
    dx = max(
        int(component_a["x_min"]) - int(component_b["x_max"]) - 1,
        int(component_b["x_min"]) - int(component_a["x_max"]) - 1,
        0,
    )
    dy = max(
        int(component_a["y_min"]) - int(component_b["y_max"]) - 1,
        int(component_b["y_min"]) - int(component_a["y_max"]) - 1,
        0,
    )
    return float(math.hypot(dx, dy))


def split_depth_continuous_mask_components(
    candidate_mask: np.ndarray,
    depth: np.ndarray,
    depth_jump_m: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if candidate_mask.shape != depth.shape:
        raise ValueError(
            "candidate_mask and depth shapes disagree: "
            f"candidate={candidate_mask.shape}, depth={depth.shape}"
        )
    threshold = float(depth_jump_m)
    if threshold < 0.0:
        raise ValueError(f"depth_jump_m must be >= 0, got {depth_jump_m}.")

    labels = np.full(candidate_mask.shape, -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    height, width = candidate_mask.shape
    neighbor_offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    seed_ys, seed_xs = np.nonzero(candidate_mask)
    for seed_y, seed_x in zip(seed_ys.tolist(), seed_xs.tolist()):
        if labels[seed_y, seed_x] >= 0:
            continue

        component_id = len(components)
        labels[seed_y, seed_x] = component_id
        stack = [(int(seed_y), int(seed_x))]
        pixels_y: list[int] = []
        pixels_x: list[int] = []
        while stack:
            y, x = stack.pop()
            pixels_y.append(y)
            pixels_x.append(x)
            center_depth = float(depth[y, x])
            for dy, dx in neighbor_offsets:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if labels[ny, nx] >= 0 or not bool(candidate_mask[ny, nx]):
                    continue
                if abs(float(depth[ny, nx]) - center_depth) > threshold:
                    continue
                labels[ny, nx] = component_id
                stack.append((ny, nx))

        ys = np.asarray(pixels_y, dtype=np.int32)
        xs = np.asarray(pixels_x, dtype=np.int32)
        component_depths = depth[ys, xs].astype(np.float32)
        components.append(
            {
                "id": int(component_id),
                "pixel_count": int(ys.size),
                "median_depth_m": float(np.median(component_depths)),
                "mean_depth_m": float(np.mean(component_depths)),
                "y_min": int(ys.min()),
                "y_max": int(ys.max()),
                "x_min": int(xs.min()),
                "x_max": int(xs.max()),
            }
        )

    return labels, components


def project_mask_to_depth_filtered_scene_faces(
    mask_bool: np.ndarray,
    scene_verts_camera: np.ndarray,
    scene_faces_compact: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
    depth_jump_m: float,
    min_component_pixels: int,
    nearby_depth_m: float,
    max_component_gap_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    depth, _, pix_to_face = rasterize_depth_and_mask(
        scene_verts_camera,
        scene_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
    )
    if pix_to_face.shape != mask_bool.shape:
        raise ValueError(
            "Rasterized pix_to_face and contact mask shapes disagree: "
            f"pix_to_face={pix_to_face.shape}, mask={mask_bool.shape}"
        )
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    candidate_mask = mask_bool & (pix_to_face >= 0) & valid_depth
    selected = pix_to_face[candidate_mask]
    if selected.size == 0:
        raise RuntimeError("Contact mask did not project onto any visible scene face.")

    labels, components = split_depth_continuous_mask_components(
        candidate_mask,
        depth,
        depth_jump_m=depth_jump_m,
    )
    if not components:
        raise RuntimeError("Contact mask has no depth-continuous components.")

    min_pixels = max(int(min_component_pixels), 1)
    nearby_depth = max(float(nearby_depth_m), 0.0)
    max_gap = max(float(max_component_gap_px), 0.0)
    main_component = max(components, key=lambda component: int(component["pixel_count"]))
    kept_component_ids = [int(main_component["id"])]
    component_summaries: list[dict[str, Any]] = []
    for component in components:
        component_id = int(component["id"])
        depth_delta = abs(
            float(component["median_depth_m"])
            - float(main_component["median_depth_m"])
        )
        gap_px = _component_bbox_gap_px(component, main_component)
        keep = component_id == int(main_component["id"]) or (
            int(component["pixel_count"]) >= min_pixels
            and depth_delta <= nearby_depth
            and gap_px <= max_gap
        )
        if keep and component_id not in kept_component_ids:
            kept_component_ids.append(component_id)
        component_summaries.append(
            {
                "id": component_id,
                "pixels": int(component["pixel_count"]),
                "median_depth_m": float(component["median_depth_m"]),
                "depth_delta_from_main_m": float(depth_delta),
                "bbox_gap_from_main_px": float(gap_px),
                "kept": bool(keep),
            }
        )

    kept_mask = np.isin(labels, np.asarray(kept_component_ids, dtype=np.int32))
    kept_faces = np.unique(pix_to_face[kept_mask].astype(np.int64))
    projected_faces = np.unique(selected.astype(np.int64))
    if kept_faces.size == 0:
        raise RuntimeError("Depth filtering removed all projected contact faces.")

    stats = {
        "projected_faces": int(projected_faces.size),
        "filtered_faces": int(kept_faces.size),
        "dropped_faces": int(projected_faces.size - kept_faces.size),
        "candidate_pixels": int(candidate_mask.sum()),
        "kept_pixels": int(kept_mask.sum()),
        "num_depth_components": int(len(components)),
        "kept_depth_components": int(len(kept_component_ids)),
        "main_component_pixels": int(main_component["pixel_count"]),
        "main_component_median_depth_m": float(main_component["median_depth_m"]),
        "depth_jump_m": float(depth_jump_m),
        "nearby_depth_m": float(nearby_depth),
        "min_component_pixels": int(min_pixels),
        "max_component_gap_px": float(max_gap),
        "components": component_summaries,
    }
    return kept_faces, stats


def expand_face_set_along_surface(
    face_indices: np.ndarray,
    verts_camera: np.ndarray,
    faces_compact: np.ndarray,
    num_rings: int,
) -> np.ndarray:
    if face_indices.size == 0 or num_rings <= 0:
        return np.unique(face_indices.astype(np.int64))

    mesh = trimesh.Trimesh(vertices=verts_camera, faces=faces_compact, process=False)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return np.unique(face_indices.astype(np.int64))

    num_faces = int(faces_compact.shape[0])
    neighbor_offsets = np.zeros(num_faces + 1, dtype=np.int64)
    pairs = np.concatenate([adjacency, adjacency[:, ::-1]], axis=0)
    order = np.argsort(pairs[:, 0], kind="stable")
    pairs_sorted = pairs[order]
    np.add.at(neighbor_offsets, pairs_sorted[:, 0] + 1, 1)
    np.cumsum(neighbor_offsets, out=neighbor_offsets)
    neighbor_flat = pairs_sorted[:, 1]

    in_set = np.zeros(num_faces, dtype=bool)
    in_set[face_indices.astype(np.int64)] = True
    frontier = face_indices.astype(np.int64)
    for _ in range(int(num_rings)):
        if frontier.size == 0:
            break
        starts = neighbor_offsets[frontier]
        ends = neighbor_offsets[frontier + 1]
        candidate = np.concatenate(
            [neighbor_flat[s:e] for s, e in zip(starts, ends)]
        ) if frontier.size > 0 else np.zeros((0,), dtype=np.int64)
        if candidate.size == 0:
            break
        new_mask = ~in_set[candidate]
        new_faces = np.unique(candidate[new_mask])
        if new_faces.size == 0:
            break
        in_set[new_faces] = True
        frontier = new_faces

    return np.flatnonzero(in_set).astype(np.int64)


def face_set_to_unique_vertex_ids(
    face_indices: np.ndarray,
    faces_compact: np.ndarray,
) -> np.ndarray:
    if face_indices.size == 0:
        raise RuntimeError("Cannot collect scene vertices from an empty face set.")
    selected_faces = faces_compact[face_indices.astype(np.int64)]
    return np.unique(selected_faces.reshape(-1)).astype(np.int64)


def sample_face_set_surface_points(
    face_indices: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if face_indices.size == 0:
        raise ValueError("Cannot sample scene contact points from an empty face set.")
    if int(num_samples) <= 0:
        raise ValueError("num_samples must be > 0")

    selected_faces = faces[face_indices.astype(np.int64)]
    triangles = verts[selected_faces]
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    positive = np.isfinite(areas) & (areas > 1e-8)
    rng = np.random.default_rng(int(seed))

    if not np.any(positive):
        raise ValueError("Cannot sample scene contact points from zero-area faces.")

    valid_triangles = triangles[positive]
    weights = areas[positive]
    weights = weights / float(weights.sum())
    face_ids = rng.choice(
        valid_triangles.shape[0],
        size=int(num_samples),
        replace=True,
        p=weights,
    )
    tri = valid_triangles[face_ids]

    r1 = rng.random(int(num_samples), dtype=np.float32)
    r2 = rng.random(int(num_samples), dtype=np.float32)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2
    samples = (
        w0[:, None] * tri[:, 0, :]
        + w1[:, None] * tri[:, 1, :]
        + w2[:, None] * tri[:, 2, :]
    )
    return samples.astype(np.float32)


def _get_reduction(nodes: tuple[InteractionNode, InteractionNode]) -> str:
    for node in nodes:
        if node.is_human and node.part_name.split(" ")[-1] in (
            "hand",
            "leg",
            "foot",
            "hips",
        ):
            return "mean"
    return "min"


def _edge_centroid(points: np.ndarray) -> np.ndarray:
    return points.astype(np.float32).mean(axis=0)


def _swap_fixed_region_assignment(
    edge_a: DynamicInteractionEdge,
    edge_b: DynamicInteractionEdge,
) -> None:
    edge_a.fixed_points, edge_b.fixed_points = edge_b.fixed_points, edge_a.fixed_points
    edge_a.fixed_face_ids, edge_b.fixed_face_ids = (
        edge_b.fixed_face_ids,
        edge_a.fixed_face_ids,
    )
    edge_a.fixed_vertex_ids, edge_b.fixed_vertex_ids = (
        edge_b.fixed_vertex_ids,
        edge_a.fixed_vertex_ids,
    )


def spatially_disambiguate_bilateral_interaction_edges(
    interaction_edges: list[DynamicInteractionEdge],
    init_verts_camera: np.ndarray,
) -> None:
    if len(interaction_edges) < 2:
        return

    edge_by_key: dict[tuple[str, str, str], DynamicInteractionEdge] = {}
    for edge in interaction_edges:
        part_tokens = normalize_label(edge.moving_part_name).split()
        if len(part_tokens) < 2 or part_tokens[0] not in {"left", "right"}:
            continue
        side = part_tokens[0]
        base_part = " ".join(part_tokens[1:])
        group_key = (normalize_label(edge.fixed_node.raw_node), base_part, side)
        edge_by_key[group_key] = edge

    checked_pairs: set[tuple[str, str]] = set()
    for fixed_node_key, base_part, side in list(edge_by_key):
        if side != "left":
            continue
        pair_key = (fixed_node_key, base_part)
        if pair_key in checked_pairs:
            continue
        checked_pairs.add(pair_key)

        left_edge = edge_by_key.get((fixed_node_key, base_part, "left"))
        right_edge = edge_by_key.get((fixed_node_key, base_part, "right"))
        if left_edge is None or right_edge is None:
            continue

        left_moving = _edge_centroid(init_verts_camera[left_edge.moving_vertex_ids])
        right_moving = _edge_centroid(init_verts_camera[right_edge.moving_vertex_ids])
        left_fixed = _edge_centroid(left_edge.fixed_points)
        right_fixed = _edge_centroid(right_edge.fixed_points)

        current_cost = float(
            np.linalg.norm(left_moving - left_fixed)
            + np.linalg.norm(right_moving - right_fixed)
        )
        swapped_cost = float(
            np.linalg.norm(left_moving - right_fixed)
            + np.linalg.norm(right_moving - left_fixed)
        )
        if swapped_cost + BILATERAL_SWAP_MIN_IMPROVEMENT_M >= current_cost:
            continue

        _swap_fixed_region_assignment(left_edge, right_edge)
        print(
            "  spatially swapped bilateral contact regions for "
            f"{base_part}: current_cost={current_cost:.4f}, "
            f"swapped_cost={swapped_cost:.4f}"
        )


def build_dynamic_interaction_edges(
    sig_payload: dict[str, Any],
    target_object_name: str,
    segment_catalog: SmplxSegmentCatalog,
    contact_masks_dir: Path,
    scene_verts_camera: np.ndarray,
    scene_faces_compact: np.ndarray,
    scene_vertex_source_ids: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
    expand_rings: int,
    surface_sample_seed: int,
    init_verts_camera: np.ndarray,
    contact_projection_depth_jump_m: float,
    contact_projection_nearby_depth_m: float,
    contact_projection_min_component_pixels: int,
    contact_projection_max_component_gap_px: float,
) -> list[DynamicInteractionEdge]:
    target_object_norm = normalize_label(target_object_name)
    image_hw = (camera_ctx.height, camera_ctx.width)
    interaction_edges: list[DynamicInteractionEdge] = []
    seen: set[tuple[str, str]] = set()

    for interaction in iter_sig_interactions(sig_payload):
        moving_part_name = normalize_label(str(interaction["body_part"]))
        scene_element = normalize_scene_element(
            str(interaction["scene_element"]),
            {target_object_norm},
        )
        if scene_element not in {"target_object", "floor"}:
            continue

        fixed_name = target_object_name if scene_element == "target_object" else "floor"
        dedup_key = (moving_part_name, scene_element)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        moving_node = InteractionNode(
            raw_node=f"human.{moving_part_name.replace(' ', '_')}",
            entity_name="human",
            part_name=moving_part_name,
            is_human=True,
        )
        fixed_node = InteractionNode(
            raw_node=fixed_name,
            entity_name=fixed_name,
            part_name=fixed_name,
            is_human=False,
        )
        moving_segment_id = segment_catalog.get_contact_or_body_segment_id(
            moving_part_name
        )
        part_vert_ids = segment_catalog.get_indices(moving_segment_id)
        moving_segment_name = segment_catalog.get_display_name(moving_segment_id)

        contact_mask = load_contact_mask_for_part(
            contact_masks_dir,
            moving_part_name,
            expected_hw=image_hw,
        )
        if not np.any(contact_mask):
            print(
                f"  skipping interaction edge '{moving_part_name}' -> "
                f"{scene_element}: contact mask is empty"
            )
            continue
        seed_face_ids, projection_filter_stats = (
            project_mask_to_depth_filtered_scene_faces(
                contact_mask,
                scene_verts_camera,
                scene_faces_compact,
                camera_ctx=camera_ctx,
                device=device,
                depth_jump_m=contact_projection_depth_jump_m,
                min_component_pixels=contact_projection_min_component_pixels,
                nearby_depth_m=contact_projection_nearby_depth_m,
                max_component_gap_px=contact_projection_max_component_gap_px,
            )
        )
        if seed_face_ids.size == 0:
            raise RuntimeError(
                f"Contact mask for '{moving_part_name}' projects to no "
                f"visible scene faces (mask path under {contact_masks_dir}). "
                "Check camera/mesh alignment or mask coverage."
            )
        projected_face_count = int(projection_filter_stats["projected_faces"])
        expanded_face_ids = expand_face_set_along_surface(
            seed_face_ids,
            scene_verts_camera,
            scene_faces_compact,
            num_rings=int(expand_rings),
        )
        fixed_vertex_ids = face_set_to_unique_vertex_ids(
            expanded_face_ids,
            scene_faces_compact,
        )
        fixed_vertex_ids = scene_vertex_source_ids[fixed_vertex_ids]
        if fixed_vertex_ids.size == 0:
            raise RuntimeError(
                f"Empty scene vertex set for '{moving_part_name}' after "
                f"expansion ({expand_rings} rings)."
            )
        fixed_points_part = sample_face_set_surface_points(
            expanded_face_ids,
            verts=scene_verts_camera,
            faces=scene_faces_compact,
            num_samples=CONTACT_SURFACE_SAMPLES_PER_EDGE,
            seed=(
                CONTACT_SURFACE_SAMPLE_SEED
                + int(surface_sample_seed)
                + 97 * len(interaction_edges)
            ),
        )
        print(
            f"  interaction edge '{moving_part_name}' -> {scene_element}: "
            f"projected_faces={projected_face_count} -> "
            f"filtered_faces={projection_filter_stats['filtered_faces']} "
            f"dropped_faces={projection_filter_stats['dropped_faces']} -> "
            f"depth_components={projection_filter_stats['num_depth_components']} "
            f"kept_components={projection_filter_stats['kept_depth_components']} -> "
            f"expanded_faces={expanded_face_ids.size} "
            f"scene_vertices={fixed_vertex_ids.size} "
            f"scene_surface_points={fixed_points_part.shape[0]}"
        )

        interaction_edges.append(
            DynamicInteractionEdge(
                node_a=moving_node,
                node_b=fixed_node,
                moving_node=moving_node,
                fixed_node=fixed_node,
                moving_part_name=moving_part_name,
                moving_segment_id=moving_segment_id,
                moving_segment_name=moving_segment_name,
                moving_vertex_ids=np.unique(np.asarray(part_vert_ids, dtype=np.int64)),
                fixed_points=fixed_points_part,
                reduction=_get_reduction((moving_node, fixed_node)),
                fixed_face_ids=expanded_face_ids,
                fixed_vertex_ids=fixed_vertex_ids,
            )
        )

    if not interaction_edges:
        raise RuntimeError(
            "No usable SIG interaction edges found for the human. "
            "All contact masks may be empty or unavailable."
        )
    spatially_disambiguate_bilateral_interaction_edges(
        interaction_edges,
        init_verts_camera=init_verts_camera,
    )
    for edge in interaction_edges:
        print(
            f"  final correspondence '{edge.moving_part_name}' -> "
            f"'{edge.fixed_node.raw_node}': "
            f"human_vertices={edge.moving_vertex_ids.size} "
            f"scene_vertices={0 if edge.fixed_vertex_ids is None else int(edge.fixed_vertex_ids.size)} "
            f"scene_surface_points={edge.fixed_points.shape[0]}"
        )
    return interaction_edges


def load_first_frame_smplx_params(
    result_dir: Path,
    param_key: str,
) -> dict[str, torch.Tensor]:
    result_path = result_dir / "hmr4d_results.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Could not find hmr4d_results.pt in: {result_dir}")
    payload = torch.load(result_path, weights_only=True)
    if param_key not in payload:
        raise KeyError(
            f"Could not find '{param_key}' in {result_path}. "
            f"Available keys: {sorted(payload.keys())}"
        )
    params = payload[param_key]
    return {
        "transl": params["transl"][0].detach().clone().float(),
        "global_orient": params["global_orient"][0].detach().clone().float(),
        "body_pose": params["body_pose"][0].detach().clone().float(),
        "betas": params["betas"][0].detach().clone().float(),
    }


def build_smplx_layer(smpl_folder: Path, device: torch.device) -> Any:
    layer = smplx.create(
        str(smpl_folder),
        model_type="smplx",
        gender="neutral",
        num_pca_comps=12,
        flat_hand_mean=False,
        create_body_pose=False,
        create_betas=False,
        create_global_orient=False,
        create_transl=False,
        return_full_pose=True,
    )
    layer = attach_volume(layer, pretrained=True, device=device)
    layer = layer.to(device)
    layer.requires_grad_(False)
    return layer


def load_optimized_smplx_params(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Optimized SMPL-X params not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = ("transl", "global_orient", "body_pose", "betas", "scale")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    return {
        "transl": torch.as_tensor(payload["transl"], dtype=torch.float32, device=device),
        "global_orient": torch.as_tensor(
            payload["global_orient"],
            dtype=torch.float32,
            device=device,
        ),
        "body_pose": torch.as_tensor(payload["body_pose"], dtype=torch.float32, device=device),
        "betas": torch.as_tensor(payload["betas"], dtype=torch.float32, device=device),
        "scale": torch.as_tensor(float(payload["scale"]), dtype=torch.float32, device=device),
    }


def build_optimized_smplx_current(
    smplx_layer: Any,
    optimized_params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        smplx_output = smplx_layer(
            transl=optimized_params["transl"].view(1, 3),
            global_orient=optimized_params["global_orient"].view(1, 3),
            body_pose=optimized_params["body_pose"].view(1, -1),
            betas=optimized_params["betas"].view(1, -1),
            return_full_pose=True,
        )
    transl = optimized_params["transl"].view(3)
    scale = optimized_params["scale"].reshape(())
    verts_unscaled = smplx_output.vertices[0]
    joints_unscaled = smplx_output.joints[0]
    return {
        "smplx_output": smplx_output,
        "transl": transl,
        "scale": scale,
        "verts": transl[None] + scale * (verts_unscaled - transl[None]),
        "joints": transl[None] + scale * (joints_unscaled - transl[None]),
    }


def clear_smplx_volume_cache(smplx_layer: Any) -> None:
    volume = getattr(smplx_layer, "volume", None)
    detach_cache = getattr(volume, "detach_cache", None)
    if callable(detach_cache):
        detach_cache()


def query_human_sdf_at_points(
    current: dict[str, torch.Tensor],
    smplx_layer: Any,
    query_points: torch.Tensor,
    chunk_size: int = 65536,
) -> torch.Tensor:
    scale = current["scale"].reshape(())
    transl = current["transl"].reshape(1, 3)
    sdf_chunks: list[torch.Tensor] = []
    clear_smplx_volume_cache(smplx_layer)
    try:
        for start in range(0, query_points.shape[0], int(chunk_size)):
            query_chunk = query_points[start:start + int(chunk_size)]
            query_unscaled = transl + (query_chunk - transl) / scale
            sdf_unscaled = smplx_layer.volume.query_fast(
                query_unscaled.unsqueeze(0),
                current["smplx_output"],
            )[0]
            sdf_chunks.append(sdf_unscaled * scale)
        if not sdf_chunks:
            raise RuntimeError("SMPL-X SDF query received zero query points.")
        return torch.cat(sdf_chunks, dim=0)
    finally:
        clear_smplx_volume_cache(smplx_layer)


def load_initial_smplx_vertices_camera(
    human_pose_root: Path,
    smpl_param_key: str,
    device: torch.device,
    smplx_layer: Any,
) -> np.ndarray:
    init_params = load_first_frame_smplx_params(human_pose_root, smpl_param_key)
    with torch.no_grad():
        out = smplx_layer(
            transl=init_params["transl"].view(1, 3).to(device),
            global_orient=init_params["global_orient"].view(1, 3).to(device),
            body_pose=init_params["body_pose"].view(1, -1).to(device),
            betas=init_params["betas"].view(1, -1).to(device),
            return_full_pose=True,
        )
    return out.vertices[0].detach().cpu().numpy().astype(np.float32)


def compute_contact_metrics(
    evaluated_vertices: np.ndarray,
    edges: list[DynamicInteractionEdge],
    device: torch.device,
) -> list[dict[str, Any]]:
    vertex_t = torch.from_numpy(evaluated_vertices.astype(np.float32)).to(device)
    rows: list[dict[str, Any]] = []
    for edge in edges:
        moving_points = vertex_t[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points = torch.from_numpy(edge.fixed_points.astype(np.float32)).to(device)
        if fixed_points.shape[0] == 0:
            raise RuntimeError(
                f"Cannot compute contact metrics for '{edge.moving_part_name}' -> "
                f"'{edge.fixed_node.raw_node}': edge has no fixed scene points."
            )
        fixed_points = fixed_points.unsqueeze(0)
        with torch.no_grad():
            nnres = knn_points(p1=moving_points, p2=fixed_points, norm=2, K=1)
            sq_dists = nnres.dists[0, :, 0]
            dists = torch.sqrt(torch.clamp(sq_dists, min=0.0))

        rows.append(
            {
                "node_a": edge.node_a.raw_node,
                "node_b": edge.node_b.raw_node,
                "min_distance_m": float(torch.min(dists).detach().cpu().item()),
                "max_distance_m": float(torch.max(dists).detach().cpu().item()),
                "mean_distance_m": float(torch.mean(dists).detach().cpu().item()),
            }
        )

    if not rows:
        raise RuntimeError("Cannot write contact metrics without rows.")
    return rows


def compute_interaction_non_collision_metrics(
    scene_points_camera: np.ndarray,
    smplx_layer: Any,
    current: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    if scene_points_camera.shape[0] == 0:
        raise RuntimeError("Cannot compute non-collision metrics without scene points.")
    scene_points_t = torch.from_numpy(scene_points_camera.astype(np.float32)).to(device)
    with torch.no_grad():
        sdf = query_human_sdf_at_points(
            current=current,
            smplx_layer=smplx_layer,
            query_points=scene_points_t,
        )
        penetrating = sdf < 0.0
        penetration = torch.clamp(-sdf[penetrating], min=0.0)

    num_points = int(sdf.shape[0])
    num_penetrating = int(penetrating.sum().detach().cpu().item())
    if num_penetrating > 0:
        mean_penetration_m = float(penetration.mean().detach().cpu().item())
        max_penetration_m = float(penetration.max().detach().cpu().item())
    else:
        mean_penetration_m = 0.0
        max_penetration_m = 0.0

    return {
        "ncs": float((num_points - num_penetrating) / num_points),
        "mean_penetration_m": mean_penetration_m,
        "max_penetration_m": max_penetration_m,
    }


def save_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_optimized_interactions() -> list[str]:
    output_root = PROJECT_DIR / "06_Optimize_Static_Scene" / "output"
    if not output_root.is_dir():
        raise FileNotFoundError(f"Optimization output directory not found: {output_root}")

    interaction_names: list[str] = []
    for interaction_dir in sorted(output_root.iterdir()):
        if not interaction_dir.is_dir():
            continue
        if not (interaction_dir / "meshes" / "frame_0000_camera.ply").exists():
            continue
        if not (interaction_dir / "debug" / "params" / "optimized_frame_0000.pt").exists():
            continue
        interaction_names.append(interaction_dir.name)
    if not interaction_names:
        raise RuntimeError(f"No optimized interactions found under {output_root}.")
    return interaction_names


def write_interaction_metric_files(
    output_root: Path,
    contact_rows: list[dict[str, Any]],
    interaction_metrics: dict[str, Any],
) -> tuple[Path, Path, list[dict[str, Any]]]:
    csv_path = output_root / "metrics.csv"
    json_path = output_root / "metrics.json"
    csv_rows = [
        {
            **row,
            "ncs": interaction_metrics["ncs"],
            "mean_penetration_m": interaction_metrics["mean_penetration_m"],
            "max_penetration_m": interaction_metrics["max_penetration_m"],
        }
        for row in contact_rows
    ]
    json_rows = [
        {
            "node_a": row["node_a"],
            "node_b": row["node_b"],
            "contact": {
                "min_distance_m": row["min_distance_m"],
                "max_distance_m": row["max_distance_m"],
                "mean_distance_m": row["mean_distance_m"],
            },
            "collision": {
                "ncs": interaction_metrics["ncs"],
                "mean_penetration_m": interaction_metrics["mean_penetration_m"],
                "max_penetration_m": interaction_metrics["max_penetration_m"],
            },
        }
        for row in contact_rows
    ]
    save_csv_rows(csv_path, csv_rows, fieldnames=METRIC_CSV_FIELDNAMES)
    save_json(json_path, json_rows)
    return csv_path, json_path, csv_rows


def write_combined_metric_files(
    output_root: Path,
    interaction_summaries: list[dict[str, Any]],
) -> tuple[Path, Path]:
    if not interaction_summaries:
        raise RuntimeError("Cannot write combined metrics without interaction summaries.")

    combined_csv_path = output_root / "physical_plausibility.csv"
    combined_json_path = output_root / "physical_plausibility.json"
    mean_ncs = float(np.mean([row["ncs"] for row in interaction_summaries]))
    mean_contact = float(
        np.mean([row["mean_contact_distance_m"] for row in interaction_summaries])
    )
    mean_penetration = float(
        np.mean([row["mean_penetration_m"] for row in interaction_summaries])
    )
    combined_rows = list(interaction_summaries)
    combined_rows.append(
        {
            "interaction_name": "__mean__",
            "num_edges": int(sum(int(row["num_edges"]) for row in interaction_summaries)),
            "mean_contact_distance_m": mean_contact,
            "ncs": mean_ncs,
            "mean_penetration_m": mean_penetration,
        }
    )
    save_csv_rows(
        combined_csv_path,
        combined_rows,
        fieldnames=COMBINED_CSV_FIELDNAMES,
    )
    save_json(
        combined_json_path,
        {
            "interactions": combined_rows[:-1],
            "aggregate": {
                "num_interactions": int(len(interaction_summaries)),
                "mean_ncs": mean_ncs,
                "mean_of_mean_contact_distance_m": mean_contact,
                "mean_penetration_m": mean_penetration,
            },
        },
    )
    return combined_csv_path, combined_json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate contact distances for one static interaction using "
            "standalone SIG/contact-mask semantics."
        )
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate every optimized interaction found under "
            "06_Optimize_Static_Scene/output."
        ),
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--human_pose_root", type=str, default=None)
    parser.add_argument("--sig-json", dest="sig_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--scannet_root", type=str, default=None)
    parser.add_argument("--smpl_folder", type=str, default=None)
    parser.add_argument("--contact_masks_dir", type=str, default=None)
    parser.add_argument("--contact_canvas_path", type=str, default=None)
    parser.add_argument("--contact_spec", type=str, default=None)
    parser.add_argument("--human_mesh_camera", type=str, default=None)
    parser.add_argument("--optimized_params", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--smpl_param_key", type=str, default="smpl_params_incam")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non_collision_surface_samples", type=int, default=700000)
    parser.add_argument("--contact_region_expand_rings", type=int, default=0)
    parser.add_argument(
        "--contact_projection_depth_jump_m",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--contact_projection_nearby_depth_m",
        type=float,
        default=0.05,
    )
    parser.add_argument("--contact_projection_min_component_pixels", type=int, default=16)
    parser.add_argument(
        "--contact_projection_max_component_gap_px",
        type=float,
        default=48.0,
    )
    return parser.parse_args()


def evaluate_interaction(
    args: argparse.Namespace,
    interaction_name: str,
    device: torch.device,
    smplx_layer: Any,
    output_root: Path,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name)
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    human_pose_root = resolve_path(args.human_pose_root, defaults["human_pose_root"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    smpl_seg_json_path = resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    contact_masks_dir = resolve_path(args.contact_masks_dir, defaults["contact_masks_dir"])
    contact_canvas_path = resolve_path(args.contact_canvas_path, defaults["contact_canvas_path"])
    contact_spec_path = resolve_path(args.contact_spec, defaults["contact_spec"])
    human_mesh_camera_path = resolve_path(args.human_mesh_camera, defaults["human_mesh_camera"])
    optimized_params_path = resolve_path(args.optimized_params, defaults["optimized_params"])
    scannet_root = resolve_scannet_root(args.scannet_root)
    output_root = ensure_dir(output_root)

    if not contact_masks_dir.is_dir():
        raise FileNotFoundError(f"Contact masks directory not found: {contact_masks_dir}")
    if not human_mesh_camera_path.exists():
        raise FileNotFoundError(f"Evaluated human mesh not found: {human_mesh_camera_path}")
    if not optimized_params_path.exists():
        raise FileNotFoundError(f"Optimized SMPL-X params not found: {optimized_params_path}")
    if not human_pose_root.is_dir():
        raise FileNotFoundError(f"Static GVHMR result directory not found: {human_pose_root}")

    input_payload = load_json(input_scene_json_path)
    sig_payload = load_json(sig_json_path)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)
    (
        _intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        _width,
        _height,
    ) = load_scannet_camera(scene_paths, scene_context)
    (
        target_intrinsics,
        target_width,
        target_height,
    ) = load_contact_camera(contact_spec_path, contact_canvas_path)
    contact_camera_ctx = build_identity_camera(
        intrinsics=target_intrinsics,
        width=target_width,
        height=target_height,
        device=device,
    )

    target_object_name = resolve_sig_target_label(sig_payload)
    segment_catalog = load_smpl_segment_catalog(smpl_seg_json_path)

    print(f"\nProcessing {interaction_name}")
    print(f"Loading ScanNet scene mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces = load_mesh(scene_paths["mesh_path"])
    scene_verts_camera = transform_world_to_camera(
        scene_verts_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    contact_scene_faces_in_view = filter_faces_to_camera_view(
        verts_camera=scene_verts_camera,
        faces=scene_faces,
        intrinsics=target_intrinsics,
        width=target_width,
        height=target_height,
        max_depth_m=20.0,
        border_px=96.0,
    )
    (
        contact_scene_verts_camera,
        contact_scene_faces_render,
        contact_scene_vertex_source_ids,
    ) = compact_mesh_with_vertex_ids(scene_verts_camera, contact_scene_faces_in_view)
    if contact_scene_faces_render.shape[0] == 0:
        raise RuntimeError(
            "No scene faces remained after contact crop camera filtering."
        )

    print("Loading initial SMPL-X vertices for bilateral contact disambiguation")
    init_verts_camera = load_initial_smplx_vertices_camera(
        human_pose_root=human_pose_root,
        smpl_param_key=args.smpl_param_key,
        device=device,
        smplx_layer=smplx_layer,
    )
    if init_verts_camera.shape[0] != segment_catalog.vertex_count:
        raise ValueError(
            "Initial SMPL-X vertex count does not match segmentation: "
            f"vertices={init_verts_camera.shape[0]}, "
            f"segmentation={segment_catalog.vertex_count}"
        )

    print(f"Loading evaluated human mesh from: {human_mesh_camera_path}")
    evaluated_vertices, _evaluated_faces = load_mesh(
        human_mesh_camera_path,
        process=False,
    )
    if evaluated_vertices.shape[0] != segment_catalog.vertex_count:
        raise ValueError(
            "Evaluated human mesh vertex count does not match segmentation: "
            f"vertices={evaluated_vertices.shape[0]}, "
            f"segmentation={segment_catalog.vertex_count}"
        )

    interaction_edges = build_dynamic_interaction_edges(
        sig_payload=sig_payload,
        target_object_name=target_object_name,
        segment_catalog=segment_catalog,
        contact_masks_dir=contact_masks_dir,
        scene_verts_camera=contact_scene_verts_camera,
        scene_faces_compact=contact_scene_faces_render,
        scene_vertex_source_ids=contact_scene_vertex_source_ids,
        camera_ctx=contact_camera_ctx,
        device=device,
        expand_rings=int(args.contact_region_expand_rings),
        surface_sample_seed=int(args.seed),
        init_verts_camera=init_verts_camera,
        contact_projection_depth_jump_m=float(args.contact_projection_depth_jump_m),
        contact_projection_nearby_depth_m=float(args.contact_projection_nearby_depth_m),
        contact_projection_min_component_pixels=int(
            args.contact_projection_min_component_pixels
        ),
        contact_projection_max_component_gap_px=float(
            args.contact_projection_max_component_gap_px
        ),
    )

    contact_rows = compute_contact_metrics(
        evaluated_vertices=evaluated_vertices,
        edges=interaction_edges,
        device=device,
    )

    print("Computing contact-crop non-collision score")
    non_collision_scene_points = sample_mesh_surface_points(
        verts=contact_scene_verts_camera,
        faces=contact_scene_faces_render,
        num_samples=int(args.non_collision_surface_samples),
        seed=NON_COLLISION_SURFACE_SAMPLE_SEED + int(args.seed),
    )
    optimized_params = load_optimized_smplx_params(optimized_params_path, device=device)
    optimized_current = build_optimized_smplx_current(
        smplx_layer=smplx_layer,
        optimized_params=optimized_params,
    )
    interaction_metrics = compute_interaction_non_collision_metrics(
        scene_points_camera=non_collision_scene_points,
        smplx_layer=smplx_layer,
        current=optimized_current,
        device=device,
    )

    csv_path, json_path, csv_rows = write_interaction_metric_files(
        output_root=output_root,
        contact_rows=contact_rows,
        interaction_metrics=interaction_metrics,
    )

    print("\nContact metrics")
    for row in contact_rows:
        print(
            f"  {row['node_a']} -> {row['node_b']}: "
            f"min={row['min_distance_m']:.5f}m "
            f"max={row['max_distance_m']:.5f}m "
            f"mean={row['mean_distance_m']:.5f}m"
        )
    print("\nInteraction non-collision")
    print(
        f"  ncs={interaction_metrics['ncs']:.6f} "
        f"mean_penetration={interaction_metrics['mean_penetration_m']:.5f}m "
        f"max_penetration={interaction_metrics['max_penetration_m']:.5f}m"
    )
    print(f"\nDone. Evaluation outputs saved to: {output_root}")
    mean_contact = float(np.mean([row["mean_distance_m"] for row in contact_rows]))
    return {
        "interaction_name": interaction_name,
        "num_edges": int(len(contact_rows)),
        "mean_contact_distance_m": mean_contact,
        "ncs": interaction_metrics["ncs"],
        "mean_penetration_m": interaction_metrics["mean_penetration_m"],
    }


def _ensure_all_mode_compatible_args(args: argparse.Namespace) -> None:
    per_interaction_overrides = [
        "input_scene_json",
        "human_pose_root",
        "sig_json",
        "contact_masks_dir",
        "contact_canvas_path",
        "contact_spec",
        "human_mesh_camera",
        "optimized_params",
    ]
    used = [name for name in per_interaction_overrides if getattr(args, name) is not None]
    if used:
        raise ValueError(
            "--all_interactions cannot be combined with per-interaction path "
            f"overrides: {used}"
        )


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or normalize_label(args.interaction_name) == "all"
    if all_mode:
        _ensure_all_mode_compatible_args(args)
        interaction_names = discover_optimized_interactions()
        output_base = resolve_path(args.output_root, SCRIPT_DIR / "output")
    else:
        interaction_names = [args.interaction_name]
        output_base = None

    device = parse_device(args.device)
    smpl_defaults = build_default_paths(interaction_names[0])
    smpl_folder = resolve_path(args.smpl_folder, smpl_defaults["smpl_folder"])
    smplx_layer = build_smplx_layer(smpl_folder, device)

    summaries: list[dict[str, Any]] = []
    for interaction_name in interaction_names:
        if all_mode:
            interaction_output_root = (
                output_base / interaction_name / "physical_plausibility"
            )
        else:
            defaults = build_default_paths(interaction_name)
            interaction_output_root = resolve_path(args.output_root, defaults["output_root"])
        summaries.append(
            evaluate_interaction(
                args=args,
                interaction_name=interaction_name,
                device=device,
                smplx_layer=smplx_layer,
                output_root=interaction_output_root,
            )
        )

    if all_mode:
        combined_csv, combined_json = write_combined_metric_files(
            output_root=ensure_dir(output_base),
            interaction_summaries=summaries,
        )
        mean_ncs = float(np.mean([row["ncs"] for row in summaries]))
        mean_contact = float(np.mean([row["mean_contact_distance_m"] for row in summaries]))
        mean_penetration = float(np.mean([row["mean_penetration_m"] for row in summaries]))
        print("\nCombined metrics")
        print(f"  interactions={len(summaries)}")
        print(f"  mean_ncs={mean_ncs:.6f}")
        print(f"  mean_of_mean_contact_distance_m={mean_contact:.5f}m")
        print(f"  mean_penetration_m={mean_penetration:.5f}m")
        print(f"  csv={combined_csv}")
        print(f"  json={combined_json}")


if __name__ == "__main__":
    main()
