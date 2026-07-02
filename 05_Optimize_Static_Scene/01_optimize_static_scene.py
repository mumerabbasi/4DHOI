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
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
from VolumetricSMPL import attach_volume
from pytorch3d.ops import knn_points
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

from mesh_intersection.bvh_search_tree import BVH
from mesh_intersection.loss import DistanceFieldPenetrationLoss


LOSS_TERM_KEYS = (
    "orient_gvhmr",
    "pose_gvhmr",
    "height_prior",
    "scene_intersect",
    "human_scene_depth",
    "nocontact",
    "self_intersect",
)
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INCH_TO_M = 0.0254
DEFAULT_HEIGHT_PRIOR_TARGET_M = 72.0 * INCH_TO_M
DEFAULT_HEIGHT_PRIOR_MIN_M = 70.0 * INCH_TO_M
DEFAULT_HEIGHT_PRIOR_MAX_M = 74.0 * INCH_TO_M
SMPLX_SDF_DEBUG_GRID_RESOLUTION = 64
SMPLX_SDF_DEBUG_CLAMP_M = 0.05
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


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
class InteractionEdge:
    node_a: InteractionNode
    node_b: InteractionNode


@dataclass
class SmplxSegmentCatalog:
    vertex_count: int
    segments: dict[str, np.ndarray]
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

    def get_contact_segment_id(self, sig_part_name: str) -> str:
        body_segment_id = slugify_segment_name(sig_part_name)
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(body_segment_id)
        if segment_id is None or segment_id not in self.contact_segment_ids:
            raise KeyError(f"Missing contact segment mapping for '{sig_part_name}'.")
        return segment_id

    def get_contact_or_body_segment_id(self, sig_part_name: str) -> str:
        return self.get_contact_segment_id(sig_part_name)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(path), force="mesh")
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


def normalize_label(text: str) -> str:
    return " ".join(
        text.strip().lower().replace("_", " ").replace("-", " ").split()
    )


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


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


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
        raise ValueError("SIG must contain target_objects.")
    label = str(target_object.get("label", "")).strip()
    if label:
        return label
    raise ValueError("SIG target_object.label must be non-empty.")


def resolve_scannet_root(
    script_dir: Path,
    raw_scannet_root: str | None,
) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


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


def build_pinhole_intrinsics(
    transforms_payload: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    width = int(transforms_payload["w"])
    height = int(transforms_payload["h"])
    intrinsics = np.array(
        [
            [
                float(transforms_payload["fl_x"]),
                0.0,
                float(transforms_payload["cx"]),
            ],
            [
                0.0,
                float(transforms_payload["fl_y"]),
                float(transforms_payload["cy"]),
            ],
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
        raise FileNotFoundError(
            f"Contact spec JSON not found: {contact_spec_path}"
        )
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path if raw_path is None else Path(raw_path).resolve()


def build_shared_default_paths(interaction_name: str) -> dict[str, Path]:
    return {
        "generated_root": PROJECT_DIR /
        "02_Generate_Human_Frame" /
        "output" /
        interaction_name,
        "input_scene_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "input_prompts" /
        interaction_name /
        "input_scene.json",
        "human_pose_root": PROJECT_DIR /
        "04_Estimate_Human_Pose" /
        "output" /
        interaction_name,
        "sig_json": PROJECT_DIR /
        "01_Generate_SIG" /
        "output" /
        interaction_name /
        "sig.json",
        "smpl_seg_json": PROJECT_DIR /
        "04_Estimate_Human_Pose" /
        "assets" /
        "smplx_vert_segmentation.json",
        "output_root": SCRIPT_DIR /
        "output" /
        interaction_name,
        "contact_masks_dir": PROJECT_DIR /
        "03_Estimate_Contact_Agentic" /
        "output" /
        interaction_name /
        "contact_masks",
        "contact_canvas_path": PROJECT_DIR /
        "03_Estimate_Contact_Agentic" /
        "output" /
        interaction_name /
        "assets" /
        "target_scene_crop.png",
        "contact_spec": PROJECT_DIR /
        "03_Estimate_Contact_Agentic" /
        "output" /
        interaction_name /
        "contact_spec.json",
    }


def parse_device(raw_device: str) -> torch.device:
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False,
                    indent=2) + "\n", encoding="utf-8")


def save_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def linear_weight(
        start: float,
        end: float,
        step_id: int,
        n_steps: int) -> float:
    if n_steps <= 0:
        return float(start)
    return float(start) + (float(end) - float(start)) * float(step_id) / float(
        n_steps
    )


def human_scene_depth_loss_enabled(args: argparse.Namespace) -> bool:
    return (
        float(args.human_scene_depth_weight_start) != 0.0
        or float(args.human_scene_depth_weight_end) != 0.0
    )


def save_loss_plot_tree(
    plot_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    total_key: str,
    term_keys: tuple[str, ...] | list[str],
    x_label: str,
    title_prefix: str,
) -> None:
    ensure_dir(plot_dir)
    raw_dir = ensure_dir(plot_dir / "raw")
    scaled_dir = ensure_dir(plot_dir / "scaled")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [float(row[x_key]) for row in rows]

    def save_plot(
        keys: list[str],
        labels: list[str],
        title: str,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label in zip(keys, labels):
            values = [float(row.get(key, 0.0)) for row in rows]
            ax.plot(xs, values, linewidth=1.3, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.set_title(title)
        if labels:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=140)
        plt.close(fig)

    save_plot(
        [total_key],
        ["total"],
        f"{title_prefix} Total Loss",
        plot_dir / "loss_total.png",
    )

    raw_keys = [f"{key}_raw" for key in term_keys]
    scaled_keys = [f"{key}_scaled" for key in term_keys]
    save_plot(
        raw_keys,
        list(term_keys),
        f"{title_prefix} Raw Loss Terms",
        raw_dir / "loss_all_terms.png",
    )
    save_plot(
        scaled_keys,
        list(term_keys),
        f"{title_prefix} Scaled Loss Terms",
        scaled_dir / "loss_all_terms.png",
    )

    for key in term_keys:
        save_plot(
            [f"{key}_raw"],
            [key],
            f"{title_prefix} Raw: {key}",
            raw_dir / f"{key}.png",
        )
        save_plot(
            [f"{key}_scaled"],
            [key],
            f"{title_prefix} Scaled: {key}",
            scaled_dir / f"{key}.png",
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


def load_smpl_segment_catalog(seg_path: Path) -> SmplxSegmentCatalog:
    raw = load_json(seg_path)
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, dict):
        raise KeyError(
            f"Expected a 'segments' mapping in {seg_path}, but it was not found."
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

    contact_segment_ids = [
        str(segment_id) for segment_id in contact_segment_ids
    ]
    for segment_id in contact_segment_ids:
        if segment_id not in segments:
            raise KeyError(
                f"Missing SMPL-X segment '{segment_id}' in {seg_path}."
            )
    for _body_segment_id, contact_segment_id in (
        CONTACT_SEGMENT_BY_BODY_SEGMENT.items()
    ):
        if contact_segment_id not in contact_segment_ids:
            raise KeyError(
                f"Missing contact segment '{contact_segment_id}' in {seg_path}."
            )

    return SmplxSegmentCatalog(
        vertex_count=vertex_count,
        segments=segments,
        contact_segment_ids=contact_segment_ids,
    )


def _get_reduction(nodes: tuple[InteractionNode, InteractionNode]) -> str:
    for node in nodes:
        if node.is_human and node.part_name.split(" ")[-1] in (
            "hand",
            "foot",
            "hips",
        ):
            return "mean"
    return "min"


def pcd_distance(
    p1: torch.Tensor,
    p2: torch.Tensor,
    reduction: str = "min",
) -> torch.Tensor:
    if p1.ndim != 3 or p2.ndim != 3:
        raise ValueError("pcd_distance expects tensors with shape [F, N, 3].")
    nnres = knn_points(p1=p1, p2=p2, norm=2, K=1)
    nn_dists = nnres.dists[..., 0]
    if reduction == "min":
        return torch.min(nn_dists, dim=1)[0]
    if reduction == "mean":
        return torch.mean(nn_dists, dim=1)
    raise RuntimeError(f"Unknown reduction: {reduction}")


def save_depth_visualization(path: Path, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0.0)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        vals = depth[valid]
        lo = float(np.percentile(vals, 2.0))
        hi = float(np.percentile(vals, 98.0))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        denom = max(hi - lo, 1e-6)
        vis_f = np.clip((depth - lo) / denom, 0.0, 1.0)
        vis[valid] = np.round(vis_f[valid] * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    cv2.imwrite(str(path), color)


def build_identity_camera(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    device: torch.device,
) -> IdentityCameraContext:
    rotation = torch.eye(3, dtype=torch.float32, device=device)[None]
    translation = torch.zeros((1, 3), dtype=torch.float32, device=device)
    camera_matrix = torch.from_numpy(intrinsics.astype(np.float32))[
        None].to(device)
    image_size = torch.tensor(
        [[height, width]], dtype=torch.float32, device=device)
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
        raster_settings=raster_settings)


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


def transform_world_to_camera(
    points_world: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return points_world @ rotation_world_to_camera.T + \
        translation_world_to_camera[None]


def transform_camera_to_world(
    points_camera: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (
        points_camera -
        translation_world_to_camera[None]) @ rotation_world_to_camera


def sample_mesh_surface_points(
    verts: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Cannot sample surface points from an empty mesh.")

    triangles = verts[faces]
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_sum = float(np.sum(areas))
    rng = np.random.default_rng(seed)

    if not np.isfinite(area_sum) or area_sum <= 1e-8:
        raise ValueError("Cannot sample surface points from a zero-area mesh.")

    probs = areas / area_sum
    face_indices = rng.choice(faces.shape[0], size=int(
        num_samples), replace=True, p=probs)
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


def write_ascii_ply(
        path: Path,
        vertices: np.ndarray,
        faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_colored_ascii_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors_uint8: np.ndarray,
) -> None:
    if vertex_colors_uint8.shape != (vertices.shape[0], 3):
        raise ValueError(
            "vertex_colors_uint8 must have shape (V, 3); got "
            f"{vertex_colors_uint8.shape} for {vertices.shape[0]} vertices."
        )
    colors = vertex_colors_uint8.astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in zip(vertices, colors):
            f.write(
                f"{vertex[0]} {vertex[1]} {vertex[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def write_colored_point_cloud_ply(
    path: Path,
    points: np.ndarray,
    colors_uint8: np.ndarray,
) -> None:
    if colors_uint8.shape != (points.shape[0], 3):
        raise ValueError(
            "colors_uint8 must have shape (N, 3); got "
            f"{colors_uint8.shape} for {points.shape[0]} points."
        )
    colors = colors_uint8.astype(np.uint8)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]} {point[1]} {point[2]} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


CONTACT_PALETTE_RGB: tuple[tuple[int, int, int], ...] = (
    (220, 30, 30),    # red
    (30, 180, 30),    # green
    (30, 90, 230),    # blue
    (220, 30, 220),   # magenta
    (30, 200, 220),   # cyan
    (240, 200, 30),   # yellow
    (240, 130, 30),   # orange
    (140, 60, 220),   # purple
    (160, 220, 30),   # lime
    (30, 200, 160),   # teal
)

CONTACT_PART_PALETTE_INDEX: dict[str, int] = {
    "right_hand": 0,
    "left_hand": 1,
    "right_foot": 2,
    "left_foot": 3,
}
BILATERAL_SWAP_MIN_IMPROVEMENT_M = 0.02
CONTACT_SURFACE_SAMPLES_PER_EDGE = 2048
CONTACT_SURFACE_SAMPLE_SEED = 17017


def palette_color_for_edge(index: int) -> tuple[int, int, int]:
    return CONTACT_PALETTE_RGB[index % len(CONTACT_PALETTE_RGB)]


def assign_interaction_palette_indices(interaction_edges: list[DynamicInteractionEdge]) -> None:
    used_indices: set[int] = set()
    for edge in interaction_edges:
        part_key = slugify_segment_name(edge.moving_part_name)
        palette_index = CONTACT_PART_PALETTE_INDEX.get(
            part_key,
            abs(hash(part_key)) % len(CONTACT_PALETTE_RGB),
        )
        edge.palette_index = int(palette_index)
        used_indices.add(int(palette_index))


def _edge_centroid(points: np.ndarray) -> np.ndarray:
    return points.astype(np.float32).mean(axis=0)


def _swap_fixed_region_assignment(
    edge_a: DynamicInteractionEdge,
    edge_b: DynamicInteractionEdge,
) -> None:
    (
        edge_a.fixed_points,
        edge_b.fixed_points,
    ) = (
        edge_b.fixed_points,
        edge_a.fixed_points,
    )
    (
        edge_a.fixed_face_ids,
        edge_b.fixed_face_ids,
    ) = (
        edge_b.fixed_face_ids,
        edge_a.fixed_face_ids,
    )
    (
        edge_a.fixed_vertex_ids,
        edge_b.fixed_vertex_ids,
    ) = (
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
        group_key = (
            normalize_label(edge.fixed_node.raw_node),
            base_part,
            side,
        )
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

        left_moving = _edge_centroid(
            init_verts_camera[left_edge.moving_vertex_ids]
        )
        right_moving = _edge_centroid(
            init_verts_camera[right_edge.moving_vertex_ids]
        )
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
    main_component = max(
        components,
        key=lambda component: int(component["pixel_count"]),
    )
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


def sample_scene_surface_points(
    scene_verts_camera: np.ndarray,
    scene_faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    sampled_points = sample_mesh_surface_points(
        verts=scene_verts_camera,
        faces=scene_faces,
        num_samples=int(num_samples),
        seed=int(seed),
    )
    return sampled_points.astype(np.float32), {
        "mode": "scene_surface",
        "num_scene_faces_total": int(scene_faces.shape[0]),
        "num_sampled_points": int(sampled_points.shape[0]),
    }


def save_static_snapshot_references(
    snapshots_dir: Path,
    interaction_edges: list[DynamicInteractionEdge],
    scene_verts_camera: np.ndarray,
    scene_faces_compact: np.ndarray,
    scene_vertex_source_ids: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> None:
    scene_verts_world = transform_camera_to_world(
        scene_verts_camera,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    scene_colors = np.tile(
        np.array([200, 200, 200], dtype=np.uint8),
        (scene_verts_world.shape[0], 1),
    )
    source_ids = np.asarray(scene_vertex_source_ids, dtype=np.int64)
    source_to_local = np.full(int(source_ids.max()) + 1, -1, dtype=np.int64)
    source_to_local[source_ids] = np.arange(source_ids.shape[0], dtype=np.int64)

    for edge in interaction_edges:
        if edge.fixed_vertex_ids is None:
            continue
        fixed_vertex_ids = np.asarray(edge.fixed_vertex_ids, dtype=np.int64)
        local_ids = source_to_local[
            fixed_vertex_ids[fixed_vertex_ids < source_to_local.shape[0]]
        ]
        local_ids = local_ids[local_ids >= 0]
        rgb = palette_color_for_edge(int(edge.palette_index))
        scene_colors[local_ids] = np.array(rgb, dtype=np.uint8)

    write_colored_ascii_ply(
        snapshots_dir / "scene.ply",
        scene_verts_world.astype(np.float32),
        scene_faces_compact,
        scene_colors,
    )


def save_human_iteration_snapshot(
    snapshots_dir: Path,
    iter_idx: int,
    verts_camera: torch.Tensor,
    faces_np: np.ndarray,
    interaction_edges: list[DynamicInteractionEdge],
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> None:
    verts_np = verts_camera.detach().cpu().numpy().astype(np.float32)
    verts_world = transform_camera_to_world(
        verts_np,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    colors = np.tile(
        np.array([230, 230, 230], dtype=np.uint8),
        (verts_world.shape[0], 1),
    )
    for edge in interaction_edges:
        rgb = palette_color_for_edge(int(edge.palette_index))
        colors[edge.moving_vertex_ids] = np.array(rgb, dtype=np.uint8)

    write_colored_ascii_ply(
        snapshots_dir / f"human_iter_{iter_idx:04d}.ply",
        verts_world,
        faces_np,
        colors,
    )


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
    palette_index: int = -1


class FullBodySMPLXParams(nn.Module):
    def __init__(
        self,
        transl_init: torch.Tensor,
        global_orient_init: torch.Tensor,
        body_pose_init: torch.Tensor,
        betas_init: torch.Tensor,
        canonical_height_m: float,
        height_prior_target_m: float,
        height_prior_min_m: float,
        height_prior_max_m: float,
        height_prior_sigma_m: float,
    ) -> None:
        super().__init__()
        if not height_prior_min_m < height_prior_target_m < height_prior_max_m:
            raise ValueError(
                "Expected height prior bounds to satisfy "
                "min < target < max, got "
                f"{height_prior_min_m}, {height_prior_target_m}, "
                f"{height_prior_max_m}."
            )
        if canonical_height_m <= 0.0:
            raise ValueError(f"Canonical SMPL-X height must be positive, got {canonical_height_m}.")
        if height_prior_sigma_m <= 0.0:
            raise ValueError(f"height_prior_sigma_m must be positive, got {height_prior_sigma_m}.")

        orient_matrix = axis_angle_to_matrix(global_orient_init.view(1, 3))[0]
        orient_6d = matrix_to_rotation_6d(orient_matrix.view(1, 3, 3))[0]
        self.transl = nn.Parameter(transl_init.clone())
        self.global_orient_6d = nn.Parameter(orient_6d.clone())
        self.body_pose = nn.Parameter(body_pose_init.clone())
        self.register_buffer("betas", betas_init.clone())
        dtype = transl_init.dtype
        device = transl_init.device
        log_scale_min = math.log(float(height_prior_min_m) / float(canonical_height_m))
        log_scale_max = math.log(float(height_prior_max_m) / float(canonical_height_m))
        log_scale_target = math.log(
            float(height_prior_target_m) / float(canonical_height_m)
        )
        ratio = (log_scale_target - log_scale_min) / (log_scale_max - log_scale_min)
        ratio = min(max(float(ratio), 1e-6), 1.0 - 1e-6)
        raw_init = math.log(ratio / (1.0 - ratio))
        self.log_scale_raw = nn.Parameter(
            torch.tensor(raw_init, dtype=dtype, device=device)
        )
        self.register_buffer(
            "canonical_height_m",
            torch.tensor(float(canonical_height_m), dtype=dtype, device=device),
        )
        self.register_buffer(
            "height_prior_target_m",
            torch.tensor(float(height_prior_target_m), dtype=dtype, device=device),
        )
        self.register_buffer(
            "height_prior_min_m",
            torch.tensor(float(height_prior_min_m), dtype=dtype, device=device),
        )
        self.register_buffer(
            "height_prior_max_m",
            torch.tensor(float(height_prior_max_m), dtype=dtype, device=device),
        )
        self.register_buffer(
            "height_prior_sigma_m",
            torch.tensor(float(height_prior_sigma_m), dtype=dtype, device=device),
        )
        self.register_buffer(
            "log_scale_min",
            torch.tensor(float(log_scale_min), dtype=dtype, device=device),
        )
        self.register_buffer(
            "log_scale_max",
            torch.tensor(float(log_scale_max), dtype=dtype, device=device),
        )

    def forward(self, smplx_layer: Any) -> dict[str, torch.Tensor]:
        orient_matrix = rotation_6d_to_matrix(self.global_orient_6d.view(1, 6))[0]
        global_orient = matrix_to_axis_angle(orient_matrix.view(1, 3, 3))[0]
        log_scale = self.log_scale_min + (
            self.log_scale_max - self.log_scale_min
        ) * torch.sigmoid(self.log_scale_raw)
        scale = torch.exp(log_scale)
        height_m = scale * self.canonical_height_m
        smplx_out = smplx_layer(
            transl=self.transl.view(1, 3),
            global_orient=global_orient.view(1, 3),
            body_pose=self.body_pose.view(1, -1),
            betas=self.betas.view(1, -1),
            return_full_pose=True,
        )
        verts_unscaled = smplx_out.vertices[0]
        joints_unscaled = smplx_out.joints[0]
        verts = self.transl[None] + scale * (verts_unscaled - self.transl[None])
        joints = self.transl[None] + scale * (joints_unscaled - self.transl[None])
        return {
            "verts": verts,
            "verts_unscaled": verts_unscaled,
            "joints": joints,
            "joints_unscaled": joints_unscaled,
            "smplx_output": smplx_out,
            "transl": self.transl,
            "global_orient_matrix": orient_matrix,
            "global_orient": global_orient,
            "body_pose": self.body_pose,
            "betas": self.betas,
            "log_scale": log_scale,
            "scale": scale,
            "height_m": height_m,
        }


class SelfIntersectionHelper:
    def __init__(self) -> None:
        self.bvh = BVH(max_collisions=8)
        self.dfp_loss = DistanceFieldPenetrationLoss(
            sigma=0.001,
            point2plane=False,
            vectorized=True,
            penalize_outside=True,
        )

    def __call__(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        triangles = vertices[faces].unsqueeze(0)
        with torch.no_grad():
            collision_idxs = self.bvh(triangles)
        if collision_idxs.ge(0).sum().item() == 0:
            return vertices.new_tensor(0.0)
        return torch.mean(self.dfp_loss(triangles, collision_idxs))


def build_default_paths(interaction_name: str) -> dict[str, Path]:
    defaults = build_shared_default_paths(interaction_name)
    defaults["output_root"] = SCRIPT_DIR / "output" / interaction_name
    defaults["smpl_folder"] = (
        SCRIPT_DIR.parent.parent / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    )
    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize first-frame full-body SMPL-X human grounding against a "
            "metric ScanNet scene using static SIG interaction semantics."
        )
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument("--generated_root", type=str, default=None)
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--human_pose_root", type=str, default=None)
    parser.add_argument("--sig-json", dest="sig_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--scannet_root", type=str, default=None)
    parser.add_argument("--smpl_folder", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--smpl_param_key", type=str, default="smpl_params_incam")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument("--adam_iters", type=int, default=2000)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument(
        "--rigid_stage_iters",
        type=int,
        default=400,
        help=(
            "Number of initial iterations with body_pose and global orientation frozen. "
            "Defaults to 40 percent of adam_iters, leaving at least one "
            "pose-enabled iteration when possible."
        ),
    )
    parser.add_argument("--orient_gvhmr_weight", type=float, default=100.0)
    parser.add_argument("--pose_gvhmr_weight", type=float, default=250.0)
    parser.add_argument("--height_prior_weight", type=float, default=1.0)
    parser.add_argument(
        "--height_prior_target_m",
        type=float,
        default=DEFAULT_HEIGHT_PRIOR_TARGET_M,
    )
    parser.add_argument(
        "--height_prior_min_m",
        type=float,
        default=DEFAULT_HEIGHT_PRIOR_MIN_M,
    )
    parser.add_argument(
        "--height_prior_max_m",
        type=float,
        default=DEFAULT_HEIGHT_PRIOR_MAX_M,
    )
    parser.add_argument("--height_prior_sigma_m", type=float, default=0.0508)
    parser.add_argument(
        "--scene_intersect_weight_start",
        type=float,
        default=60,
    )
    parser.add_argument(
        "--scene_intersect_weight_end",
        type=float,
        default=60,
    )
    parser.add_argument("--scene_intersect_margin_m", type=float, default=0.00)
    parser.add_argument("--scene_intersect_surface_samples", type=int, default=700000)
    parser.add_argument(
        "--scene_intersect_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--human_scene_depth_weight_start", type=float, default=0.0)
    parser.add_argument("--human_scene_depth_weight_end", type=float, default=0.0)
    parser.add_argument(
        "--human_scene_depth_penetration_tolerance_m",
        type=float,
        default=0.00,
    )
    parser.add_argument("--human_scene_depth_min_valid_weight", type=float, default=0.25)
    parser.add_argument("--nocontact_weight_start", type=float, default=800.0)
    parser.add_argument("--nocontact_weight_end", type=float, default=800.0)
    parser.add_argument("--self_intersect_weight_start", type=float, default=1e-3)
    parser.add_argument("--self_intersect_weight_end", type=float, default=1e-3)  # Original 1e-3
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=200)
    parser.add_argument("--contact_masks_dir", type=str, default=None)
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
    parser.add_argument("--snapshot_every_iters", type=int, default=100)
    return parser.parse_args()


def get_loss_weights(
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> dict[str, float]:
    return {
        "orient_gvhmr": float(args.orient_gvhmr_weight),
        "pose_gvhmr": float(args.pose_gvhmr_weight),
        "height_prior": float(args.height_prior_weight),
        "scene_intersect": linear_weight(
            args.scene_intersect_weight_start,
            args.scene_intersect_weight_end,
            iteration,
            total_iters,
        ),
        "human_scene_depth": linear_weight(
            args.human_scene_depth_weight_start,
            args.human_scene_depth_weight_end,
            iteration,
            total_iters,
        ),
        "nocontact": linear_weight(
            args.nocontact_weight_start,
            args.nocontact_weight_end,
            iteration,
            total_iters,
        ),
        "self_intersect": linear_weight(
            args.self_intersect_weight_start,
            args.self_intersect_weight_end,
            iteration,
            total_iters,
        ),
    }


def build_loss_row(
    iteration: int,
    losses: dict[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "iter": int(iteration),
        "stage": stage_name,
        "total": float(losses["total"].detach().cpu().item()),
    }
    for key in LOSS_TERM_KEYS:
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = float(weights[key])
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = float(weights[key]) * raw_value
    scene_stats = losses.get("scene_intersect_stats", {})
    if isinstance(scene_stats, dict):
        for key, value in scene_stats.items():
            row[f"scene_intersect_{key}"] = int(value)
    depth_stats = losses.get("human_scene_depth_stats", {})
    if isinstance(depth_stats, dict):
        for key, value in depth_stats.items():
            row[f"human_scene_depth_{key}"] = int(value)
    return row


def format_loss_log(
    iteration: int,
    total_iterations: int,
    losses: dict[str, Any],
) -> list[str]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    weights_fmt = "  ".join(f"{key}={weights[key]:.4g}" for key in LOSS_TERM_KEYS)
    raw_fmt = "  ".join(
        f"{key}={float(losses[key].detach().cpu().item()):.5f}" for key in LOSS_TERM_KEYS
    )
    scaled_fmt = "  ".join(
        f"{key}={weights[key] * float(losses[key].detach().cpu().item()):.5f}"
        for key in LOSS_TERM_KEYS
    )
    return [
        f"  [iter {iteration:4d}/{total_iterations}] "
        f"total={float(losses['total'].detach().cpu().item()):.5f}",
        f"      weights: {weights_fmt}",
        f"      raw:     {raw_fmt}",
        f"      scaled:  {scaled_fmt}",
    ]


def build_final_loss_summary_row(
    final_iter: int,
    losses: dict[str, Any],
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "final_iter": int(final_iter),
        "final_total_loss": float(losses["total"].detach().cpu().item()),
        "total_scaled": float(losses["total"].detach().cpu().item()),
    }
    for key in LOSS_TERM_KEYS:
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = float(weights[key])
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = float(weights[key]) * raw_value
    scene_stats = losses.get("scene_intersect_stats", {})
    if isinstance(scene_stats, dict):
        for key, value in scene_stats.items():
            row[f"scene_intersect_{key}"] = int(value)
    depth_stats = losses.get("human_scene_depth_stats", {})
    if isinstance(depth_stats, dict):
        for key, value in depth_stats.items():
            row[f"human_scene_depth_{key}"] = int(value)
    return row


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
        moving_segment_name = segment_catalog.get_display_name(
            moving_segment_id
        )

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
        contact_face_ids, projection_filter_stats = (
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
        if contact_face_ids.size == 0:
            raise RuntimeError(
                f"Contact mask for '{moving_part_name}' projects to no "
                f"visible scene faces (mask path under {contact_masks_dir}). "
                "Check camera/mesh alignment or mask coverage."
            )
        projected_face_count = int(projection_filter_stats["projected_faces"])
        fixed_vertex_ids = face_set_to_unique_vertex_ids(
            contact_face_ids,
            scene_faces_compact,
        )
        fixed_vertex_ids = scene_vertex_source_ids[fixed_vertex_ids]
        if fixed_vertex_ids.size == 0:
            raise RuntimeError(
                f"Empty scene vertex set for '{moving_part_name}' contact region."
            )
        fixed_face_ids = contact_face_ids
        fixed_points_part = sample_face_set_surface_points(
            contact_face_ids,
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
            f"contact_faces={contact_face_ids.size} "
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
                moving_vertex_ids=np.unique(
                    np.asarray(part_vert_ids, dtype=np.int64)
                ),
                fixed_points=fixed_points_part,
                reduction=_get_reduction((moving_node, fixed_node)),
                fixed_face_ids=fixed_face_ids,
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
    assign_interaction_palette_indices(interaction_edges)
    for edge in interaction_edges:
        rgb = palette_color_for_edge(int(edge.palette_index))
        print(
            f"  final correspondence '{edge.moving_part_name}' -> "
            f"'{edge.fixed_node.raw_node}': "
            f"human_vertices={edge.moving_vertex_ids.size} "
            f"scene_vertices={0 if edge.fixed_vertex_ids is None else int(edge.fixed_vertex_ids.size)} "
            f"scene_surface_points={edge.fixed_points.shape[0]} "
            f"color_rgb={rgb}"
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


def compute_canonical_smplx_height_m(
    smplx_layer: Any,
    betas: torch.Tensor,
) -> float:
    device = betas.device
    dtype = betas.dtype
    with torch.no_grad():
        out = smplx_layer(
            transl=torch.zeros((1, 3), device=device, dtype=dtype),
            global_orient=torch.zeros((1, 3), device=device, dtype=dtype),
            body_pose=torch.zeros((1, 63), device=device, dtype=dtype),
            betas=betas.view(1, -1),
            return_full_pose=True,
        )
        verts = out.vertices[0]
        height = verts[:, 1].max() - verts[:, 1].min()
    return float(height.detach().cpu().item())


def compute_orient_prior_loss(
    current_orient_matrix: torch.Tensor,
    init_orient_matrix: torch.Tensor,
) -> torch.Tensor:
    relative = init_orient_matrix.transpose(0, 1) @ current_orient_matrix
    relative_aa = matrix_to_axis_angle(relative.view(1, 3, 3))[0]
    return torch.mean(relative_aa.pow(2))


def compute_contact_distance_loss(
    current_vertices: torch.Tensor,
    edges: list[DynamicInteractionEdge],
) -> torch.Tensor:
    if not edges:
        raise RuntimeError("Contact distance loss requires at least one interaction edge.")
    values: list[torch.Tensor] = []
    for edge in edges:
        moving_points_seq = current_vertices[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points = torch.from_numpy(edge.fixed_points).to(
            device=current_vertices.device,
            dtype=current_vertices.dtype,
        )
        if fixed_points.shape[0] == 0:
            raise RuntimeError(
                f"Interaction edge '{edge.moving_part_name}' -> "
                f"'{edge.fixed_node.raw_node}' has no fixed scene points."
            )
        fixed_points_seq = fixed_points.unsqueeze(0)
        pdists = pcd_distance(
            moving_points_seq,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        values.append(pdists.mean())
    return torch.stack(values, dim=0).mean()


def query_human_sdf_for_scene_points(
    current: dict[str, torch.Tensor],
    smplx_layer: Any,
    scene_collision_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scene_collision_points.shape[0] == 0:
        raise RuntimeError("scene_intersect requires scene collision samples.")
    sdf = query_human_sdf_at_points(
        current=current,
        smplx_layer=smplx_layer,
        query_points=scene_collision_points,
    )
    return scene_collision_points, sdf


def clear_smplx_volume_cache(smplx_layer: Any) -> None:
    volume = getattr(smplx_layer, "volume", None)
    detach_cache = getattr(volume, "detach_cache", None)
    if callable(detach_cache):
        detach_cache()


def compute_scene_inside_human_loss(
    current: dict[str, torch.Tensor],
    smplx_layer: Any,
    scene_collision_points: torch.Tensor,
    clearance_margin_m: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    verts = current["verts"]
    zero = verts.new_tensor(0.0)
    stats = {
        "num_scene_collision_points": int(scene_collision_points.shape[0]),
        "num_sdf_query_points": int(scene_collision_points.shape[0]),
        "num_inside_or_margin_points": 0,
    }
    if scene_collision_points.shape[0] == 0:
        raise RuntimeError("scene_intersect requires scene collision samples.")

    _, sdf = query_human_sdf_for_scene_points(
        current=current,
        smplx_layer=smplx_layer,
        scene_collision_points=scene_collision_points,
    )

    violations = F.relu(float(clearance_margin_m) - sdf)
    stats["num_inside_or_margin_points"] = int((violations > 0).sum().detach().cpu().item())
    active = violations > 0
    if not torch.any(active):
        return zero, stats
    return violations[active].mean(), stats


def compute_human_scene_depth_loss(
    current_vertices: torch.Tensor,
    scene_depth: torch.Tensor,
    scene_depth_valid: torch.Tensor,
    intrinsics: torch.Tensor,
    penetration_tolerance_m: float,
    min_valid_weight: float,
) -> tuple[torch.Tensor, dict[str, int]]:
    if scene_depth.ndim != 4 or scene_depth.shape[0] != 1 or scene_depth.shape[1] != 1:
        raise ValueError("scene_depth must have shape [1, 1, H, W].")
    if scene_depth_valid.shape != scene_depth.shape:
        raise ValueError(
            "scene_depth_valid must have the same shape as scene_depth, got "
            f"{scene_depth_valid.shape} and {scene_depth.shape}."
        )
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape [3, 3], got {intrinsics.shape}.")

    height = int(scene_depth.shape[2])
    width = int(scene_depth.shape[3])
    z = current_vertices[:, 2]
    valid_z = z > 1e-6
    z_safe = torch.clamp(z, min=1e-6)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    u = fx * current_vertices[:, 0] / z_safe + cx
    v = fy * current_vertices[:, 1] / z_safe + cy
    valid_xy = (
        (u >= 0.0)
        & (u <= float(width - 1))
        & (v >= 0.0)
        & (v <= float(height - 1))
    )

    if width > 1:
        grid_x = (u / float(width - 1)) * 2.0 - 1.0
    else:
        grid_x = torch.zeros_like(u)
    if height > 1:
        grid_y = (v / float(height - 1)) * 2.0 - 1.0
    else:
        grid_y = torch.zeros_like(v)
    grid = torch.stack((grid_x, grid_y), dim=-1).view(1, -1, 1, 2)
    sampled_depth_sum = F.grid_sample(
        scene_depth,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    sampled_valid_weight = F.grid_sample(
        scene_depth_valid,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    sampled_depth = sampled_depth_sum / torch.clamp(sampled_valid_weight, min=1e-6)

    valid_scene = sampled_valid_weight >= float(min_valid_weight)
    valid = valid_z & valid_xy & valid_scene
    stats = {
        "num_human_vertices": int(current_vertices.shape[0]),
        "num_projected_vertices": int(valid.sum().detach().cpu().item()),
        "num_behind_scene_vertices": 0,
    }
    if not torch.any(valid):
        return current_vertices.new_tensor(0.0), stats

    violations = F.relu(
        z[valid] - sampled_depth[valid] - float(penetration_tolerance_m)
    )
    active = violations > 0
    stats["num_behind_scene_vertices"] = int(
        active.sum().detach().cpu().item()
    )
    if not torch.any(active):
        return current_vertices.new_tensor(0.0), stats
    return violations[active].mean(), stats


def color_scene_intersect_sdf(
    sdf: np.ndarray,
    clearance_margin_m: float,
) -> np.ndarray:
    colors = np.tile(np.array([150, 170, 210], dtype=np.uint8), (sdf.shape[0], 1))
    inside = sdf < 0.0
    in_margin = (sdf >= 0.0) & (sdf < float(clearance_margin_m))
    colors[inside] = np.array([255, 0, 0], dtype=np.uint8)
    colors[in_margin] = np.array([255, 170, 0], dtype=np.uint8)
    return colors


def color_sdf_grid_values(
    sdf: np.ndarray,
    clamp_m: float = SMPLX_SDF_DEBUG_CLAMP_M,
) -> np.ndarray:
    clamp = max(float(clamp_m), 1e-6)
    colors = np.zeros((sdf.shape[0], 3), dtype=np.float32)
    negative = sdf < 0.0
    positive = ~negative

    neg_t = np.clip(1.0 + sdf[negative] / clamp, 0.0, 1.0)[:, None]
    colors[negative] = (
        (1.0 - neg_t) * np.array([255, 0, 0], dtype=np.float32)
        + neg_t * np.array([255, 255, 255], dtype=np.float32)
    )

    pos_t = np.clip(sdf[positive] / clamp, 0.0, 1.0)[:, None]
    colors[positive] = (
        (1.0 - pos_t) * np.array([255, 255, 255], dtype=np.float32)
        + pos_t * np.array([40, 110, 255], dtype=np.float32)
    )
    return np.clip(np.round(colors), 0, 255).astype(np.uint8)


def build_sdf_grid_query_points(
    verts_camera: torch.Tensor,
    resolution: int = SMPLX_SDF_DEBUG_GRID_RESOLUTION,
    padding_m: float = 0.15,
) -> tuple[torch.Tensor, tuple[int, int, int], torch.Tensor, torch.Tensor]:
    verts = verts_camera.detach()
    vmin = verts.min(dim=0).values - float(padding_m)
    vmax = verts.max(dim=0).values + float(padding_m)
    res = int(resolution)
    xs = torch.linspace(vmin[0], vmax[0], res, device=verts.device, dtype=verts.dtype)
    ys = torch.linspace(vmin[1], vmax[1], res, device=verts.device, dtype=verts.dtype)
    zs = torch.linspace(vmin[2], vmax[2], res, device=verts.device, dtype=verts.dtype)

    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    points = torch.stack(
        [gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)],
        dim=1,
    )
    return points, (res, res, res), vmin, vmax


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


def save_scene_intersect_debug_artifacts(
    debug_dir: Path,
    stage_name: str,
    current: dict[str, torch.Tensor],
    smplx_layer: Any,
    scene_collision_points_t: torch.Tensor,
    clearance_margin_m: float,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> dict[str, Any]:
    ensure_dir(debug_dir)
    with torch.no_grad():
        scene_points, sdf = query_human_sdf_for_scene_points(
            current=current,
            smplx_layer=smplx_layer,
            scene_collision_points=scene_collision_points_t,
        )
        grid_points, grid_shape, grid_min, grid_max = build_sdf_grid_query_points(
            current["verts"]
        )
        grid_sdf = query_human_sdf_at_points(
            current=current,
            smplx_layer=smplx_layer,
            query_points=grid_points,
        )

    scene_points_np = scene_points.detach().cpu().numpy().astype(np.float32)
    scene_sdf_np = sdf.detach().cpu().numpy().astype(np.float32)
    scene_world = transform_camera_to_world(
        scene_points_np,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    scene_colors = color_scene_intersect_sdf(scene_sdf_np, clearance_margin_m)
    write_colored_point_cloud_ply(
        debug_dir / f"{stage_name}_scene_points_smplx_sdf.ply",
        scene_world.astype(np.float32),
        scene_colors,
    )

    grid_points_np = grid_points.detach().cpu().numpy().astype(np.float32)
    grid_sdf_np = grid_sdf.detach().cpu().numpy().astype(np.float32)
    grid_world = transform_camera_to_world(
        grid_points_np,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    write_colored_point_cloud_ply(
        debug_dir / f"{stage_name}_smplx_sdf_grid.ply",
        grid_world.astype(np.float32),
        color_sdf_grid_values(grid_sdf_np),
    )
    np.savez_compressed(
        debug_dir / f"{stage_name}_smplx_sdf_grid.npz",
        points_camera=grid_points_np,
        points_world=grid_world.astype(np.float32),
        sdf_m=grid_sdf_np,
        sdf_grid_m=grid_sdf_np.reshape(grid_shape),
        grid_shape=np.asarray(grid_shape, dtype=np.int32),
        bbox_min_camera=grid_min.detach().cpu().numpy().astype(np.float32),
        bbox_max_camera=grid_max.detach().cpu().numpy().astype(np.float32),
        color_clamp_m=np.asarray([SMPLX_SDF_DEBUG_CLAMP_M], dtype=np.float32),
    )

    scene_stats = {
        "num_scene_collision_points": int(scene_collision_points_t.shape[0]),
        "num_sdf_query_points": int(scene_points.shape[0]),
        "num_visualized_points": int(scene_points_np.shape[0]),
        "num_inside_points": int(np.count_nonzero(scene_sdf_np < 0.0)),
        "num_margin_points": int(
            np.count_nonzero(
                (scene_sdf_np >= 0.0)
                & (scene_sdf_np < float(clearance_margin_m))
            )
        ),
        "min_sdf_m": float(scene_sdf_np.min()),
        "max_sdf_m": float(scene_sdf_np.max()),
    }
    grid_stats = {
        "num_grid_points": int(grid_points.shape[0]),
        "grid_shape": list(grid_shape),
        "min_sdf_m": float(grid_sdf_np.min()),
        "max_sdf_m": float(grid_sdf_np.max()),
        "num_inside_grid_points": int(np.count_nonzero(grid_sdf_np < 0.0)),
        "num_near_surface_grid_points": int(
            np.count_nonzero(np.abs(grid_sdf_np) < float(clearance_margin_m))
        ),
        "color_clamp_m": float(SMPLX_SDF_DEBUG_CLAMP_M),
    }
    payload = {
        "stage": stage_name,
        "scene_point_colors": {
            "red": "inside SMPL-X human SDF; contributes to scene_intersect",
            "orange": "outside but within scene_intersect clearance margin",
            "blue_gray": "outside clearance margin",
        },
        "sdf_grid_colors": {
            "red_intensity": "inside SMPL-X; stronger red means more negative SDF",
            "white": "near zero SDF surface",
            "blue_intensity": "outside SMPL-X; stronger blue means larger positive SDF",
        },
        "clearance_margin_m": float(clearance_margin_m),
        "scene_points": scene_stats,
        "sdf_grid": grid_stats,
    }
    save_json(debug_dir / f"{stage_name}_scene_intersect_debug.json", payload)
    return payload


def compute_loss_dict(
    params_module: FullBodySMPLXParams,
    smplx_layer: Any,
    faces_t: torch.Tensor,
    interaction_edges: list[DynamicInteractionEdge],
    scene_collision_points_t: torch.Tensor,
    scene_intersect_margin_m: float,
    scene_depth_t: torch.Tensor | None,
    scene_depth_valid_t: torch.Tensor | None,
    scene_depth_intrinsics_t: torch.Tensor | None,
    human_scene_depth_penetration_tolerance_m: float,
    human_scene_depth_min_valid_weight: float,
    init_params: dict[str, torch.Tensor],
    self_intersection_helper: SelfIntersectionHelper,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    current = params_module(smplx_layer)
    verts_camera = current["verts"]

    zero = torch.zeros((), device=verts_camera.device, dtype=verts_camera.dtype)

    orient_gvhmr = compute_orient_prior_loss(
        current["global_orient_matrix"],
        init_params["global_orient_matrix"],
    )
    pose_gvhmr = torch.mean((current["body_pose"] - init_params["body_pose"]) ** 2)
    height_prior = (
        (current["height_m"] - params_module.height_prior_target_m)
        / params_module.height_prior_sigma_m
    ).pow(2)

    scene_intersect, scene_intersect_stats = compute_scene_inside_human_loss(
        current=current,
        smplx_layer=smplx_layer,
        scene_collision_points=scene_collision_points_t,
        clearance_margin_m=scene_intersect_margin_m,
    )
    if (
        scene_depth_t is not None
        and scene_depth_valid_t is not None
        and scene_depth_intrinsics_t is not None
    ):
        human_scene_depth, human_scene_depth_stats = compute_human_scene_depth_loss(
            current_vertices=verts_camera,
            scene_depth=scene_depth_t,
            scene_depth_valid=scene_depth_valid_t,
            intrinsics=scene_depth_intrinsics_t,
            penetration_tolerance_m=human_scene_depth_penetration_tolerance_m,
            min_valid_weight=human_scene_depth_min_valid_weight,
        )
    else:
        human_scene_depth = zero
        human_scene_depth_stats = {}
    nocontact = compute_contact_distance_loss(
        current_vertices=verts_camera,
        edges=interaction_edges,
    )

    if float(weights["self_intersect"]) > 0.0:
        self_intersect = self_intersection_helper(verts_camera, faces_t)
    else:
        self_intersect = zero

    total = (
        orient_gvhmr * float(weights["orient_gvhmr"])
        + pose_gvhmr * float(weights["pose_gvhmr"])
        + height_prior * float(weights["height_prior"])
        + scene_intersect * float(weights["scene_intersect"])
        + human_scene_depth * float(weights["human_scene_depth"])
        + nocontact * float(weights["nocontact"])
        + self_intersect * float(weights["self_intersect"])
    )
    return {
        "total": total,
        "orient_gvhmr": orient_gvhmr,
        "pose_gvhmr": pose_gvhmr,
        "height_prior": height_prior,
        "scene_intersect": scene_intersect,
        "human_scene_depth": human_scene_depth,
        "nocontact": nocontact,
        "self_intersect": self_intersect,
        "weights": weights,
        "current": current,
        "scene_intersect_stats": scene_intersect_stats,
        "human_scene_depth_stats": human_scene_depth_stats,
    }


def compute_interaction_metrics(
    current_vertices: np.ndarray,
    edges: list[DynamicInteractionEdge],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    device = torch.device("cpu")
    current_vertices_t = torch.from_numpy(current_vertices.astype(np.float32)).to(device)
    for edge in edges:
        fixed_points_t = torch.from_numpy(edge.fixed_points.astype(np.float32)).to(device)
        if fixed_points_t.shape[0] == 0:
            raise RuntimeError(
                f"Cannot compute interaction metrics for "
                f"'{edge.moving_part_name}' -> '{edge.fixed_node.raw_node}': "
                "edge has no fixed scene points."
            )
        moving_points_t = current_vertices_t[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points_seq = fixed_points_t.unsqueeze(0)
        pdists = pcd_distance(
            moving_points_t,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        nocontact_raw = pdists.mean().detach().cpu().item()
        metrics.append(
            {
                "node_a": edge.node_a.raw_node,
                "node_b": edge.node_b.raw_node,
                "moving_entity_name": edge.moving_node.entity_name,
                "moving_part_name": edge.moving_node.part_name,
                "moving_segment_id": edge.moving_segment_id,
                "moving_segment_name": edge.moving_segment_name,
                "moving_vertex_count": int(edge.moving_vertex_ids.size),
                "fixed_entity_name": edge.fixed_node.entity_name,
                "fixed_part_name": edge.fixed_node.part_name,
                "fixed_point_count": int(edge.fixed_points.shape[0]),
                "reduction": edge.reduction,
                "nocontact_raw": float(nocontact_raw),
                "nocontact_distance_m": float(math.sqrt(max(nocontact_raw, 0.0))),
            }
        )
    return metrics


def resolve_optimization_stage_iters(args: argparse.Namespace) -> tuple[int, int]:
    total_iters = int(args.adam_iters)
    if total_iters <= 0:
        raise RuntimeError("adam_iters must be > 0.")

    if args.rigid_stage_iters is None:
        if total_iters == 1:
            rigid_iters = 0
        else:
            rigid_iters = int(round(total_iters * 0.4))
            rigid_iters = min(max(rigid_iters, 1), total_iters - 1)
    else:
        rigid_iters = int(args.rigid_stage_iters)
        if rigid_iters < 0:
            raise RuntimeError("rigid_stage_iters must be >= 0.")
        if total_iters > 1 and rigid_iters >= total_iters:
            raise RuntimeError("rigid_stage_iters must be smaller than adam_iters.")
        if total_iters == 1 and rigid_iters > 0:
            raise RuntimeError("rigid_stage_iters must be 0 when adam_iters is 1.")

    pose_iters = total_iters - rigid_iters
    return rigid_iters, pose_iters


def set_stage_trainable_params(
    params_module: FullBodySMPLXParams,
    optimize_global_orient: bool,
    optimize_body_pose: bool,
) -> list[nn.Parameter]:
    params_module.transl.requires_grad_(True)
    params_module.global_orient_6d.requires_grad_(bool(optimize_global_orient))
    params_module.log_scale_raw.requires_grad_(True)
    params_module.body_pose.requires_grad_(bool(optimize_body_pose))
    return [param for param in params_module.parameters() if param.requires_grad]


def optimize_track(
    smplx_layer: Any,
    faces_t: torch.Tensor,
    init_params_np: dict[str, np.ndarray],
    interaction_edges: list[DynamicInteractionEdge],
    scene_collision_points: np.ndarray,
    scene_depth: np.ndarray | None,
    scene_depth_intrinsics: np.ndarray | None,
    args: argparse.Namespace,
    device: torch.device,
    snapshots_dir: Path,
    scene_intersect_debug_dir: Path,
    snapshot_every_iters: int,
    faces_np: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> dict[str, Any]:
    snapshot_active = int(snapshot_every_iters) > 0
    scene_intersect_debug_active = bool(args.scene_intersect_debug)
    init_params_t = {
        "transl": torch.from_numpy(init_params_np["transl"]).to(device=device, dtype=torch.float32),
        "global_orient": torch.from_numpy(init_params_np["global_orient"]).to(device=device, dtype=torch.float32),
        "body_pose": torch.from_numpy(init_params_np["body_pose"]).to(device=device, dtype=torch.float32),
        "betas": torch.from_numpy(init_params_np["betas"]).to(device=device, dtype=torch.float32),
    }
    init_params_t["global_orient_matrix"] = axis_angle_to_matrix(
        init_params_t["global_orient"].view(1, 3)
    )[0]
    canonical_height_m = compute_canonical_smplx_height_m(
        smplx_layer=smplx_layer,
        betas=init_params_t["betas"],
    )

    params_module = FullBodySMPLXParams(
        transl_init=init_params_t["transl"],
        global_orient_init=init_params_t["global_orient"],
        body_pose_init=init_params_t["body_pose"],
        betas_init=init_params_t["betas"],
        canonical_height_m=canonical_height_m,
        height_prior_target_m=float(args.height_prior_target_m),
        height_prior_min_m=float(args.height_prior_min_m),
        height_prior_max_m=float(args.height_prior_max_m),
        height_prior_sigma_m=float(args.height_prior_sigma_m),
    ).to(device)
    print(
        "  height prior: "
        f"canonical_unscaled={canonical_height_m:.4f}m "
        f"target={float(args.height_prior_target_m):.4f}m "
        f"range=[{float(args.height_prior_min_m):.4f}, "
        f"{float(args.height_prior_max_m):.4f}]m"
    )
    self_intersection_helper = SelfIntersectionHelper()

    scene_collision_points_t = torch.from_numpy(
        scene_collision_points.astype(np.float32)
    ).to(device)
    if human_scene_depth_loss_enabled(args):
        if scene_depth is None or scene_depth_intrinsics is None:
            raise RuntimeError(
                "human_scene_depth loss is enabled, but scene depth inputs are missing."
            )
        scene_depth_t = torch.from_numpy(
            scene_depth.astype(np.float32)
        ).to(device).view(1, 1, scene_depth.shape[0], scene_depth.shape[1])
        scene_depth_valid_t = (scene_depth_t > 1e-6).to(dtype=scene_depth_t.dtype)
        scene_depth_intrinsics_t = torch.from_numpy(
            scene_depth_intrinsics.astype(np.float32)
        ).to(device)
    else:
        scene_depth_t = None
        scene_depth_valid_t = None
        scene_depth_intrinsics_t = None

    iter_rows: list[dict[str, Any]] = []
    rigid_stage_iters, pose_stage_iters = resolve_optimization_stage_iters(args)
    print(
        "  optimization stages: "
        f"rigid={rigid_stage_iters} iters, pose={pose_stage_iters} iters"
    )

    scene_intersect_debug_payloads: dict[str, Any] = {}
    if scene_intersect_debug_active:
        with torch.no_grad():
            init_current = params_module(smplx_layer)
        scene_intersect_debug_payloads["init"] = save_scene_intersect_debug_artifacts(
            debug_dir=scene_intersect_debug_dir,
            stage_name="init",
            current=init_current,
            smplx_layer=smplx_layer,
            scene_collision_points_t=scene_collision_points_t,
            clearance_margin_m=float(args.scene_intersect_margin_m),
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )

    stages = [
        ("rigid", rigid_stage_iters, False, False),
        ("pose", pose_stage_iters, True, True),
    ]
    completed_iters = 0
    for stage_name, stage_iters, optimize_global_orient, optimize_body_pose in stages:
        if int(stage_iters) <= 0:
            continue
        active_params = set_stage_trainable_params(
            params_module,
            optimize_global_orient=optimize_global_orient,
            optimize_body_pose=optimize_body_pose,
        )
        optimizer = torch.optim.Adam(active_params, lr=float(args.adam_lr))
        print(f"  stage {stage_name}: {stage_iters} iterations")
        for local_iter_idx in range(1, int(stage_iters) + 1):
            iter_idx = completed_iters + local_iter_idx
            weights = get_loss_weights(args, iter_idx - 1, int(args.adam_iters) - 1)
            if not optimize_body_pose:
                weights = dict(weights)
                weights["self_intersect"] = 0.0
            optimizer.zero_grad(set_to_none=True)
            losses = compute_loss_dict(
                params_module=params_module,
                smplx_layer=smplx_layer,
                faces_t=faces_t,
                interaction_edges=interaction_edges,
                scene_collision_points_t=scene_collision_points_t,
                scene_intersect_margin_m=float(args.scene_intersect_margin_m),
                scene_depth_t=scene_depth_t,
                scene_depth_valid_t=scene_depth_valid_t,
                scene_depth_intrinsics_t=scene_depth_intrinsics_t,
                human_scene_depth_penetration_tolerance_m=float(
                    args.human_scene_depth_penetration_tolerance_m
                ),
                human_scene_depth_min_valid_weight=float(
                    args.human_scene_depth_min_valid_weight
                ),
                init_params=init_params_t,
                self_intersection_helper=self_intersection_helper,
                weights=weights,
            )
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
            optimizer.step()
            iter_rows.append(build_loss_row(iter_idx, losses, stage_name))
            if (
                iter_idx == 1
                or iter_idx % max(int(args.log_every), 1) == 0
                or iter_idx == int(args.adam_iters)
            ):
                print(f"  stage={stage_name}")
                for line in format_loss_log(iter_idx, int(args.adam_iters), losses):
                    print(line)

            if snapshot_active and (
                iter_idx == 1
                or iter_idx % max(int(snapshot_every_iters), 1) == 0
                or iter_idx == int(args.adam_iters)
            ):
                save_human_iteration_snapshot(
                    snapshots_dir=snapshots_dir,
                    iter_idx=iter_idx,
                    verts_camera=losses["current"]["verts"],
                    faces_np=faces_np,
                    interaction_edges=interaction_edges,
                    rotation_world_to_camera=rotation_world_to_camera,
                    translation_world_to_camera=translation_world_to_camera,
                )
                print(f"  snapshot iter={iter_idx}")
        completed_iters += int(stage_iters)

    final_weights = get_loss_weights(
        args,
        int(args.adam_iters) - 1,
        int(args.adam_iters) - 1,
    )
    with torch.no_grad():
        final_losses = compute_loss_dict(
            params_module=params_module,
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            interaction_edges=interaction_edges,
            scene_collision_points_t=scene_collision_points_t,
            scene_intersect_margin_m=float(args.scene_intersect_margin_m),
            scene_depth_t=scene_depth_t,
            scene_depth_valid_t=scene_depth_valid_t,
            scene_depth_intrinsics_t=scene_depth_intrinsics_t,
            human_scene_depth_penetration_tolerance_m=float(
                args.human_scene_depth_penetration_tolerance_m
            ),
            human_scene_depth_min_valid_weight=float(
                args.human_scene_depth_min_valid_weight
            ),
            init_params=init_params_t,
            self_intersection_helper=self_intersection_helper,
            weights=final_weights,
        )
        current = final_losses["current"]
        if scene_intersect_debug_active:
            scene_intersect_debug_payloads["final"] = save_scene_intersect_debug_artifacts(
                debug_dir=scene_intersect_debug_dir,
                stage_name="final",
                current=current,
                smplx_layer=smplx_layer,
                scene_collision_points_t=scene_collision_points_t,
                clearance_margin_m=float(args.scene_intersect_margin_m),
                rotation_world_to_camera=rotation_world_to_camera,
                translation_world_to_camera=translation_world_to_camera,
            )
    return {
        "iter_rows": iter_rows,
        "final_iter": int(args.adam_iters),
        "final_total_loss": float(final_losses["total"].detach().cpu().item()),
        "final_losses": final_losses,
        "verts_camera": current["verts"].detach().cpu().numpy().astype(np.float32),
        "joints_camera": current["joints"].detach().cpu().numpy().astype(np.float32),
        "transl": current["transl"].detach().cpu().numpy().astype(np.float32),
        "global_orient": current["global_orient"].detach().cpu().numpy().astype(np.float32),
        "body_pose": current["body_pose"].detach().cpu().numpy().astype(np.float32),
        "betas": current["betas"].detach().cpu().numpy().astype(np.float32),
        "scale": float(current["scale"].detach().cpu().item()),
        "log_scale": float(current["log_scale"].detach().cpu().item()),
        "height_m": float(current["height_m"].detach().cpu().item()),
        "canonical_height_unscaled_m": float(
            params_module.canonical_height_m.detach().cpu().item()
        ),
        "scene_intersect_stats": final_losses.get("scene_intersect_stats", {}),
        "human_scene_depth_stats": final_losses.get("human_scene_depth_stats", {}),
        "scene_intersect_debug": scene_intersect_debug_payloads,
        "stage_iters": {
            "rigid": int(rigid_stage_iters),
            "pose": int(pose_stage_iters),
        },
    }


def main() -> None:
    args = parse_args()
    defaults = build_default_paths(args.interaction_name)
    generated_root = resolve_path(args.generated_root, defaults["generated_root"])
    input_scene_json_path = resolve_path(args.input_scene_json, defaults["input_scene_json"])
    human_pose_root = resolve_path(args.human_pose_root, defaults["human_pose_root"])
    sig_json_path = resolve_path(args.sig_json, defaults["sig_json"])
    smpl_seg_json_path = resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    smpl_folder = resolve_path(args.smpl_folder, defaults["smpl_folder"])
    contact_canvas_path = defaults["contact_canvas_path"]
    contact_spec_path = defaults["contact_spec"]
    contact_masks_dir = resolve_path(
        args.contact_masks_dir, defaults["contact_masks_dir"]
    )
    if not contact_masks_dir.is_dir():
        raise FileNotFoundError(
            f"Contact masks directory not found: {contact_masks_dir}"
        )
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    scene_root = ensure_dir(output_root / "scene")
    human_scene_depth_active = human_scene_depth_loss_enabled(args)
    scene_depth_dir = scene_root / "depth"
    if human_scene_depth_active:
        ensure_dir(scene_depth_dir)
    debug_root = ensure_dir(output_root / "debug")
    summary_json_path = output_root / "alignment_summary.json"
    scannet_root = resolve_scannet_root(SCRIPT_DIR, args.scannet_root)
    device = parse_device(args.device)

    input_payload = load_json(input_scene_json_path)
    sig_payload = load_json(sig_json_path)
    scene_context = input_payload["scene_context"]
    scene_paths = resolve_scene_paths(scannet_root, scene_context)
    (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    ) = load_scannet_camera(scene_paths, scene_context)
    first_frame_path = generated_root / "inpainted_frame_resized.png"
    if not first_frame_path.exists():
        raise FileNotFoundError(
            f"Resized inpainted frame not found: {first_frame_path}"
        )
    first_frame_bgr = read_bgr(first_frame_path)
    if first_frame_bgr.shape[:2] != (height, width):
        raise ValueError(
            "Inpainted frame shape does not match ScanNet++ camera metadata: "
            f"image={first_frame_bgr.shape[1]}x{first_frame_bgr.shape[0]}, "
            f"metadata={width}x{height}. "
            "Run 02_Generate_Human_Frame/02_generate_human_frame.py first."
        )
    camera_ctx = build_identity_camera(
        intrinsics=intrinsics,
        width=width,
        height=height,
        device=device,
    )
    (
        target_intrinsics,
        target_width,
        target_height,
    ) = load_contact_camera(
        contact_spec_path,
        contact_canvas_path,
    )
    contact_camera_ctx = build_identity_camera(
        intrinsics=target_intrinsics,
        width=target_width,
        height=target_height,
        device=device,
    )

    target_object_name = resolve_sig_target_label(sig_payload)
    segment_catalog = load_smpl_segment_catalog(smpl_seg_json_path)

    print(f"Loading ScanNet scene mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces = load_mesh(scene_paths["mesh_path"])
    scene_verts_camera = transform_world_to_camera(
        scene_verts_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    if human_scene_depth_active:
        scene_faces_in_view = filter_faces_to_camera_view(
            verts_camera=scene_verts_camera,
            faces=scene_faces,
            intrinsics=intrinsics,
            width=width,
            height=height,
            max_depth_m=20.0,
            border_px=96.0,
        )
        (
            scene_verts_camera_render,
            scene_faces_render,
            _,
        ) = compact_mesh_with_vertex_ids(
            scene_verts_camera, scene_faces_in_view
        )
        if scene_faces_render.shape[0] == 0:
            raise RuntimeError("No scene faces remained after view-frustum filtering.")
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
    ) = compact_mesh_with_vertex_ids(
        scene_verts_camera,
        contact_scene_faces_in_view,
    )
    if contact_scene_faces_render.shape[0] == 0:
        raise RuntimeError(
            "No scene faces remained after contact crop camera filtering."
        )

    if human_scene_depth_active:
        scene_depth, _, _ = rasterize_depth_and_mask(
            scene_verts_camera_render,
            scene_faces_render,
            camera_ctx=camera_ctx,
            device=device,
        )
        np.save(scene_depth_dir / "scene_depth.npy", scene_depth.astype(np.float32))
        save_depth_visualization(scene_depth_dir / "scene_depth_vis.png", scene_depth)
    else:
        scene_depth = None

    human_result_dir = human_pose_root
    if not human_result_dir.is_dir():
        raise FileNotFoundError(
            f"Static GVHMR result directory not found: {human_result_dir}"
        )
    smplx_layer = build_smplx_layer(smpl_folder, device)
    faces_np = np.asarray(smplx_layer.faces, dtype=np.int64)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device)

    human_summary: dict[str, Any] | None = None

    for result_dir in [human_result_dir]:
        print("\nProcessing human")
        meshes_root = ensure_dir(output_root / "meshes")
        debug_track_root = debug_root
        overlay_dir = ensure_dir(debug_track_root / "overlays")
        csv_dir = ensure_dir(debug_track_root / "csv")
        plot_dir = ensure_dir(debug_track_root / "plots" / "iter")
        params_dir = ensure_dir(debug_track_root / "params")
        snapshots_dir = ensure_dir(debug_track_root / "snapshots")
        scene_intersect_debug_dir = ensure_dir(debug_track_root / "scene_intersect")

        init_params_torch = load_first_frame_smplx_params(
            result_dir,
            args.smpl_param_key,
        )
        with torch.no_grad():
            init_out = smplx_layer(
                transl=init_params_torch["transl"].view(1, 3).to(device),
                global_orient=init_params_torch["global_orient"].view(1, 3).to(device),
                body_pose=init_params_torch["body_pose"].view(1, -1).to(device),
                betas=init_params_torch["betas"].view(1, -1).to(device),
                return_full_pose=True,
            )
            init_verts_camera = init_out.vertices[0].detach().cpu().numpy().astype(np.float32)

        scene_collision_points, scene_collision_sampling_stats = (
            sample_scene_surface_points(
                scene_verts_camera=contact_scene_verts_camera,
                scene_faces=contact_scene_faces_render,
                num_samples=int(args.scene_intersect_surface_samples),
                seed=int(args.seed) + 4242,
            )
        )
        scene_collision_sampling_stats["source_camera"] = "contact"

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
            surface_sample_seed=int(args.seed),
            init_verts_camera=init_verts_camera,
            contact_projection_depth_jump_m=float(
                args.contact_projection_depth_jump_m
            ),
            contact_projection_nearby_depth_m=float(
                args.contact_projection_nearby_depth_m
            ),
            contact_projection_min_component_pixels=int(
                args.contact_projection_min_component_pixels
            ),
            contact_projection_max_component_gap_px=float(
                args.contact_projection_max_component_gap_px
            ),
        )
        save_static_snapshot_references(
            snapshots_dir=snapshots_dir,
            interaction_edges=interaction_edges,
            scene_verts_camera=contact_scene_verts_camera,
            scene_faces_compact=contact_scene_faces_render,
            scene_vertex_source_ids=contact_scene_vertex_source_ids,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        init_params_np = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in init_params_torch.items()
        }

        init_depth, _init_mask_rendered, _ = rasterize_depth_and_mask(
            init_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        save_depth_visualization(overlay_dir / "frame_0000_init_depth_vis.png", init_depth)
        init_interaction_metrics = compute_interaction_metrics(init_verts_camera, interaction_edges)

        optimization = optimize_track(
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            init_params_np=init_params_np,
            interaction_edges=interaction_edges,
            scene_collision_points=scene_collision_points,
            scene_depth=scene_depth,
            scene_depth_intrinsics=intrinsics if human_scene_depth_active else None,
            args=args,
            device=device,
            snapshots_dir=snapshots_dir,
            scene_intersect_debug_dir=scene_intersect_debug_dir,
            snapshot_every_iters=int(args.snapshot_every_iters),
            faces_np=faces_np,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )

        final_verts_camera = optimization["verts_camera"]
        final_depth, _final_mask_rendered, _ = rasterize_depth_and_mask(
            final_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        save_depth_visualization(overlay_dir / "frame_0000_final_depth_vis.png", final_depth)

        final_verts_world = transform_camera_to_world(
            final_verts_camera,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        write_ascii_ply(meshes_root / "frame_0000_camera.ply", final_verts_camera, faces_np)
        write_ascii_ply(meshes_root / "frame_0000_world.ply", final_verts_world, faces_np)

        optimized_params_payload = {
            "transl": optimization["transl"].tolist(),
            "global_orient": optimization["global_orient"].tolist(),
            "body_pose": optimization["body_pose"].tolist(),
            "betas": optimization["betas"].tolist(),
            "scale": float(optimization["scale"]),
            "log_scale": float(optimization["log_scale"]),
            "height_m": float(optimization["height_m"]),
            "canonical_height_unscaled_m": float(
                optimization["canonical_height_unscaled_m"]
            ),
        }
        torch.save(optimized_params_payload, params_dir / "optimized_frame_0000.pt")

        iter_metrics_csv = csv_dir / "iter_metrics.csv"
        final_loss_summary_csv = csv_dir / "final_loss_summary.csv"
        save_csv_rows(iter_metrics_csv, optimization["iter_rows"])
        save_csv_rows(
            final_loss_summary_csv,
            [build_final_loss_summary_row(optimization["final_iter"], optimization["final_losses"])],
        )
        save_loss_plot_tree(
            plot_dir,
            optimization["iter_rows"],
            x_key="iter",
            total_key="total",
            term_keys=LOSS_TERM_KEYS,
            x_label="Iteration",
            title_prefix="Iter",
        )

        final_interaction_metrics = compute_interaction_metrics(final_verts_camera, interaction_edges)

        human_summary = {
            "optimization": {
                "final_iter": int(optimization["final_iter"]),
                "final_total_loss": float(optimization["final_total_loss"]),
                "stage_iters": optimization["stage_iters"],
                "scene_intersect_sampling": scene_collision_sampling_stats,
                "scene_intersect_stats": optimization["scene_intersect_stats"],
                "human_scene_depth_stats": optimization["human_scene_depth_stats"],
                "scene_intersect_debug": optimization["scene_intersect_debug"],
            },
            "init_frame_0": {
                "interaction_edges": init_interaction_metrics,
            },
            "final_frame_0": {
                "interaction_edges": final_interaction_metrics,
            },
            "artifacts": {
                "camera_mesh": str(meshes_root / "frame_0000_camera.ply"),
                "world_mesh": str(meshes_root / "frame_0000_world.ply"),
                "init_depth_vis": str(overlay_dir / "frame_0000_init_depth_vis.png"),
                "final_depth_vis": str(overlay_dir / "frame_0000_final_depth_vis.png"),
                "optimized_params": str(params_dir / "optimized_frame_0000.pt"),
                "scene_intersect_debug": str(scene_intersect_debug_dir),
                "csv": {
                    "iter_metrics": str(iter_metrics_csv),
                    "final_loss_summary": str(final_loss_summary_csv),
                },
            },
        }

    save_json(
        summary_json_path,
        {
            "interaction_name": args.interaction_name,
            "scene_id": scene_context["scene_id"],
            "target_object": {
                "label": target_object_name,
            },
            "optimizer": {
                "adam_iters": int(args.adam_iters),
                "adam_lr": float(args.adam_lr),
                "stage_iters": {
                    "rigid": int(optimization["stage_iters"]["rigid"]),
                    "pose": int(optimization["stage_iters"]["pose"]),
                },
                "stage_trainable": {
                    "rigid": {
                        "transl": True,
                        "global_orient": False,
                        "scale": True,
                        "body_pose": False,
                    },
                    "pose": {
                        "transl": True,
                        "global_orient": True,
                        "scale": True,
                        "body_pose": True,
                    },
                },
                "loss_weights": {
                    "orient_gvhmr": float(args.orient_gvhmr_weight),
                    "pose_gvhmr": float(args.pose_gvhmr_weight),
                    "height_prior": float(args.height_prior_weight),
                    "scene_intersect": {
                        "start": float(args.scene_intersect_weight_start),
                        "end": float(args.scene_intersect_weight_end),
                        "clearance_margin_m": float(args.scene_intersect_margin_m),
                        "surface_samples": int(args.scene_intersect_surface_samples),
                        "debug": bool(args.scene_intersect_debug),
                    },
                    "human_scene_depth": {
                        "start": float(args.human_scene_depth_weight_start),
                        "end": float(args.human_scene_depth_weight_end),
                        "penetration_tolerance_m": float(
                            args.human_scene_depth_penetration_tolerance_m
                        ),
                        "min_valid_weight": float(
                            args.human_scene_depth_min_valid_weight
                        ),
                    },
                    "nocontact": {
                        "start": float(args.nocontact_weight_start),
                        "end": float(args.nocontact_weight_end),
                    },
                    "self_intersect": {
                        "start": float(args.self_intersect_weight_start),
                        "end": float(args.self_intersect_weight_end),
                    },
                },
                "height_prior": {
                    "target_m": float(args.height_prior_target_m),
                    "min_m": float(args.height_prior_min_m),
                    "max_m": float(args.height_prior_max_m),
                    "sigma_m": float(args.height_prior_sigma_m),
                    "target_ft_in": "6 ft 0 in",
                    "min_ft_in": "5 ft 10 in",
                    "max_ft_in": "6 ft 2 in",
                },
            },
            "scene": {
                "num_contact_crop_faces": int(contact_scene_faces_render.shape[0]),
                "num_contact_crop_vertices": int(contact_scene_verts_camera.shape[0]),
            },
            "human": human_summary,
        },
    )
    print(f"\nDone. Full-body outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
