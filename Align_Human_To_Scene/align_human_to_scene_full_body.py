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
    "mask",
    "root_orient_gvhmr",
    "pose_gvhmr",
    "height_prior",
    "scene_intersect",
    "nocontact",
    "floor_nocontact",
    "angle",
    "self_intersect",
)
CONTACT_SEGMENT_BY_BODY_SEGMENT = {
    "left_hand": "left_hand_inner",
    "right_hand": "right_hand_inner",
    "left_foot": "left_foot_bottom",
    "right_foot": "right_foot_bottom",
}

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INCH_TO_M = 0.0254
DEFAULT_HEIGHT_PRIOR_TARGET_M = 72.0 * INCH_TO_M
DEFAULT_HEIGHT_PRIOR_MIN_M = 70.0 * INCH_TO_M
DEFAULT_HEIGHT_PRIOR_MAX_M = 74.0 * INCH_TO_M
SMPLX_SDF_DEBUG_GRID_RESOLUTION = 64
SMPLX_SDF_DEBUG_CLAMP_M = 0.05


@dataclass
class HumanTrack:
    name: str
    mask_dir: Path
    result_dir: Path
    source_camera_mesh_dir: Path


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

    def get_body_segment_id(self, pag_part_name: str) -> str:
        segment_id = slugify_segment_name(pag_part_name)
        if segment_id not in self.body_segment_ids:
            raise KeyError(f"Missing body segment mapping for '{pag_part_name}'.")
        return segment_id

    def get_contact_segment_id(self, pag_part_name: str) -> str:
        body_segment_id = slugify_segment_name(pag_part_name)
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(body_segment_id)
        if segment_id is None or segment_id not in self.contact_segment_ids:
            raise KeyError(f"Missing contact segment mapping for '{pag_part_name}'.")
        return segment_id

    def get_contact_or_body_segment_id(self, pag_part_name: str) -> str:
        body_segment_id = slugify_segment_name(pag_part_name)
        segment_id = CONTACT_SEGMENT_BY_BODY_SEGMENT.get(body_segment_id)
        if segment_id is not None:
            if segment_id not in self.contact_segment_ids:
                raise KeyError(f"Missing contact segment mapping for '{pag_part_name}'.")
            return segment_id
        return self.get_body_segment_id(pag_part_name)


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


def slugify_segment_name(text: str) -> str:
    return normalize_label(text).replace(" ", "_")


def resolve_scannet_root(
    script_dir: Path,
    raw_scannet_root: str | None,
) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_paths(
    scannet_root: Path,
    scene_id: str,
) -> dict[str, Path]:
    scene_root = scannet_root / scene_id
    return {
        "scene_root": scene_root,
        "mesh_path": scene_root / "scans" / "mesh_aligned_0.05.ply",
        "segments_path": scene_root / "scans" / "segments.json",
        "segments_anno_path": scene_root / "scans" / "segments_anno.json",
    }


def build_candidate_instances(
    mesh_faces: np.ndarray,
    seg_indices: np.ndarray,
    seg_groups: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    face_batches: list[np.ndarray] = []
    face_instance_ids: list[np.ndarray] = []
    instance_meta: dict[int, dict[str, Any]] = {}

    for group in seg_groups:
        label = str(group["label"])
        if not normalize_label(label):
            continue

        object_id = int(group["objectId"])
        segments = np.asarray(group["segments"], dtype=np.int64)
        if segments.size == 0:
            continue

        vertex_mask = np.isin(seg_indices, segments)
        face_mask = np.all(vertex_mask[mesh_faces], axis=1)
        candidate_faces = mesh_faces[face_mask]
        if candidate_faces.size == 0:
            continue

        face_batches.append(candidate_faces.astype(np.int64))
        face_instance_ids.append(
            np.full((candidate_faces.shape[0],), object_id, dtype=np.int32)
        )
        instance_meta[object_id] = {
            "instance_id": object_id,
            "label": label,
        }

    if not face_batches:
        raise ValueError(
            "No valid instance annotations were found for the scene.")

    return (
        np.concatenate(face_batches, axis=0),
        np.concatenate(face_instance_ids, axis=0),
        instance_meta,
    )


def build_faces_for_labels(
    mesh_faces: np.ndarray,
    seg_indices: np.ndarray,
    seg_groups: list[dict[str, Any]],
    labels: set[str],
) -> np.ndarray:
    face_batches: list[np.ndarray] = []
    labels_norm = {normalize_label(label) for label in labels}
    for group in seg_groups:
        label = normalize_label(str(group["label"]))
        if label not in labels_norm:
            continue

        segments = np.asarray(group["segments"], dtype=np.int64)
        if segments.size == 0:
            continue

        vertex_mask = np.isin(seg_indices, segments)
        face_mask = np.all(vertex_mask[mesh_faces], axis=1)
        candidate_faces = mesh_faces[face_mask]
        if candidate_faces.size == 0:
            continue
        face_batches.append(candidate_faces.astype(np.int64))

    if not face_batches:
        raise RuntimeError(f"No scene faces found for labels: {sorted(labels_norm)}")
    return np.concatenate(face_batches, axis=0)


def remove_faces(
    faces: np.ndarray,
    faces_to_remove: np.ndarray,
) -> np.ndarray:
    if faces_to_remove.shape[0] == 0:
        return faces.copy()
    remove_set = {tuple(face.tolist()) for face in faces_to_remove}
    keep = [tuple(face.tolist()) not in remove_set for face in faces]
    return faces[np.asarray(keep, dtype=bool)]


def load_camera_payload(
    camera_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    payload = load_json(camera_path)
    intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    world_to_camera = np.asarray(
        payload["world_to_camera_4x4"],
        dtype=np.float32,
    )
    width = int(payload["width"])
    height = int(payload["height"])

    if intrinsics.shape != (3, 3):
        raise ValueError(
            f"Expected a 3x3 intrinsics matrix in {camera_path}, "
            f"got {intrinsics.shape}."
        )
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            "Expected a 4x4 world_to_camera_4x4 matrix in "
            f"{camera_path}, got {world_to_camera.shape}."
        )

    rotation_world_to_camera = world_to_camera[:3, :3].astype(np.float32)
    translation_world_to_camera = world_to_camera[:3, 3].astype(np.float32)
    return (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path if raw_path is None else Path(raw_path).resolve()


def build_shared_default_paths(video_name: str) -> dict[str, Path]:
    return {
        "generated_root": PROJECT_DIR /
        "Generate_Video" /
        "output" /
        video_name,
        "selection_json": PROJECT_DIR /
        "Select_Target_Instance" /
        "output" /
        video_name /
        "target_selection.json",
        "input_pag_json": PROJECT_DIR /
        "Select_Target_Instance" /
        "input_prompts" /
        video_name /
        "input_pag.json",
        "segment_root": PROJECT_DIR /
        "Segment_Video" /
        "output" /
        video_name,
        "human_motion_root": PROJECT_DIR /
        "Estimate_Human_Motion" /
        "output" /
        video_name /
        "humans",
        "pag_root": PROJECT_DIR /
        "Generate_PAG" /
        "output" /
        video_name,
        "smpl_seg_json": PROJECT_DIR /
        "Estimate_Human_Motion" /
        "assets" /
        "smplx_vert_segmentation.json",
        "output_root": SCRIPT_DIR /
        "output" /
        video_name,
        "contact_masks_dir": PROJECT_DIR /
        "Estimate_Contact_Masks" /
        "output" /
        video_name /
        "contact_masks",
    }


def find_pag_json(pag_dir: Path) -> Path:
    candidates = sorted(pag_dir.glob("*.json"))
    output_candidates = [
        candidate for candidate in candidates if candidate.name.startswith("output_pag")
    ]
    if len(output_candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one output_pag*.json in {pag_dir}, "
            f"found {len(output_candidates)}."
        )
    return output_candidates[0]


def parse_device(raw_device: str) -> torch.device:
    device = torch.device(raw_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def load_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {path}")
    return mask > 127


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


def parse_pag_node(node_str: str) -> InteractionNode:
    parts = node_str.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse PAG node: '{node_str}'")
    entity_name = parts[0].strip()
    part_name = parts[1].strip()
    return InteractionNode(
        raw_node=node_str,
        entity_name=entity_name,
        part_name=part_name,
        is_human=entity_name.lower().startswith("person"),
    )


def parse_pag_interaction_edges(
        pag_payload: dict[str, Any]) -> list[InteractionEdge]:
    edges: list[InteractionEdge] = []
    for edge_payload in pag_payload.get("interaction edges", []):
        node_values = edge_payload.get("nodes", [])
        if len(node_values) != 2:
            continue
        edges.append(
            InteractionEdge(
                node_a=parse_pag_node(str(node_values[0])),
                node_b=parse_pag_node(str(node_values[1])),
            )
        )
    return edges


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
    contact_segment_ids = [
        str(segment_id) for segment_id in contact_segment_ids
    ]
    for segment_id in body_segment_ids + contact_segment_ids:
        if segment_id not in segments:
            raise KeyError(
                f"Missing SMPL-X segment '{segment_id}' in {seg_path}."
            )
    for body_segment_id, contact_segment_id in (
        CONTACT_SEGMENT_BY_BODY_SEGMENT.items()
    ):
        if body_segment_id not in body_segment_ids:
            raise KeyError(
                f"Missing body segment '{body_segment_id}' in {seg_path}."
            )
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


def _get_reduction(nodes: tuple[InteractionNode, InteractionNode]) -> str:
    for node in nodes:
        if node.is_human and node.part_name.split(" ")[-1] in ("hand", "foot"):
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


def render_mask_overlay(
    background_bgr: np.ndarray,
    observed_mask: np.ndarray,
    rendered_mask: np.ndarray,
    title_lines: list[str],
) -> np.ndarray:
    overlay = background_bgr.astype(np.float32).copy()
    observed_only = np.logical_and(
        observed_mask, np.logical_not(rendered_mask))
    rendered_only = np.logical_and(
        rendered_mask, np.logical_not(observed_mask))
    overlap = np.logical_and(observed_mask, rendered_mask)

    overlay[observed_only] = 0.55 * overlay[observed_only] + \
        0.45 * np.array([0, 0, 255], dtype=np.float32)
    overlay[rendered_only] = 0.55 * overlay[rendered_only] + \
        0.45 * np.array([255, 200, 0], dtype=np.float32)
    overlay[overlap] = 0.45 * overlay[overlap] + \
        0.55 * np.array([0, 255, 0], dtype=np.float32)

    overlay_u8 = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    y = 30
    for line in title_lines:
        cv2.putText(
            overlay_u8,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 34
    return overlay_u8


def compute_binary_overlap(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    intersection = int(np.count_nonzero(np.logical_and(a_bool, b_bool)))
    union = int(np.count_nonzero(np.logical_or(a_bool, b_bool)))
    area_a = int(np.count_nonzero(a_bool))
    area_b = int(np.count_nonzero(b_bool))
    return {
        "intersection_px": intersection,
        "union_px": union,
        "area_a_px": area_a,
        "area_b_px": area_b,
        "iou": float(intersection / union) if union > 0 else 0.0,
        "dice": (
            float((2.0 * intersection) / (area_a + area_b))
            if (area_a + area_b) > 0
            else 0.0
        ),
    }


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


def project_points_np(
        points: np.ndarray,
        intrinsics: np.ndarray) -> np.ndarray:
    z = np.clip(points[:, 2], 1e-6, None)
    u = intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2] - 0.5
    return np.stack([u, v], axis=1).astype(np.float32)


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


def compact_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if faces.shape[0] == 0:
        raise RuntimeError("Cannot compact an empty mesh.")
    unique_vids, inverse = np.unique(faces.reshape(-1), return_inverse=True)
    compact_verts = verts[unique_vids].astype(np.float32)
    compact_faces = inverse.reshape(-1, 3).astype(np.int64)
    return compact_verts, compact_faces


def build_visible_subset(
    sampled_points: np.ndarray,
    mesh_verts: np.ndarray,
    mesh_faces: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
    visible_tol_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mesh_depth, _, _ = rasterize_depth_and_mask(
        mesh_verts, mesh_faces, camera_ctx, device)
    uv = project_points_np(sampled_points, camera_ctx.intrinsics)
    z = sampled_points[:, 2]
    ui = np.round(uv[:, 0]).astype(np.int64)
    vi = np.round(uv[:, 1]).astype(np.int64)
    in_frame = (
        (z > 1e-6)
        & (ui >= 0)
        & (ui < camera_ctx.width)
        & (vi >= 0)
        & (vi < camera_ctx.height)
    )

    visible = np.zeros(sampled_points.shape[0], dtype=bool)
    if np.any(in_frame):
        idx = np.nonzero(in_frame)[0]
        depth_at_pixel = mesh_depth[vi[idx], ui[idx]]
        visible[idx] = (depth_at_pixel > 0.0) & (
            np.abs(z[idx] - depth_at_pixel) <= float(visible_tol_m))

    num_visible = int(np.count_nonzero(visible))
    if num_visible < MIN_VISIBLE_SUBSET_POINTS:
        raise RuntimeError(
            "Too few visible mesh surface points after projection/depth filtering: "
            f"{num_visible} visible points, need at least "
            f"{MIN_VISIBLE_SUBSET_POINTS}. "
            "Check the mesh, camera, or visible_tol_m."
        )

    return sampled_points[visible].astype(np.float32), {
        "mode": "visible",
        "num_total_points": int(sampled_points.shape[0]),
        "num_in_frame_points": int(np.count_nonzero(in_frame)),
        "num_visible_points": num_visible,
        "num_kept_points": num_visible,
        "visible_tol_m": float(visible_tol_m),
    }


def sample_mask_pixels(
    mask: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise RuntimeError("Observed human mask is empty; cannot build mask loss.")
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    if max_points > 0 and coords.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(coords.shape[0], size=int(
            max_points), replace=False)
        coords = coords[chosen]
    return coords.astype(np.float32)


def pairwise_squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a2 = (a * a).sum(dim=1, keepdim=True)
    b2 = (b * b).sum(dim=1).unsqueeze(0)
    return torch.clamp(a2 + b2 - 2.0 * (a @ b.transpose(0, 1)), min=0.0)


def min_distances_chunked(
    obs: torch.Tensor,
    model: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    if obs.shape[0] == 0 or model.shape[0] == 0:
        raise RuntimeError("min_distances_chunked received an empty point set.")

    best_chunks: list[torch.Tensor] = []
    for start in range(0, obs.shape[0], chunk_size):
        obs_chunk = obs[start:start + chunk_size]
        best_d2 = None
        for model_start in range(0, model.shape[0], chunk_size):
            model_chunk = model[model_start:model_start + chunk_size]
            d2 = pairwise_squared_l2(obs_chunk, model_chunk)
            cur_best, _ = torch.min(d2, dim=1)
            best_d2 = cur_best if best_d2 is None else torch.minimum(
                best_d2, cur_best)
        assert best_d2 is not None
        best_chunks.append(best_d2)
    return torch.cat(best_chunks, dim=0)


def project_points_torch(points: torch.Tensor,
                         intrinsics: torch.Tensor) -> torch.Tensor:
    z = torch.clamp(points[:, 2], min=1e-6)
    u = intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2] - 0.5
    v = intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2] - 0.5
    return torch.stack([u, v], dim=1)


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
MIN_VISIBLE_SUBSET_POINTS = 64


def palette_color_for_edge(index: int) -> tuple[int, int, int]:
    return CONTACT_PALETTE_RGB[index % len(CONTACT_PALETTE_RGB)]


def assign_contact_palette_indices(contact_edges: list[DynamicContactEdge]) -> None:
    used_indices: set[int] = set()
    for edge in contact_edges:
        part_key = slugify_segment_name(edge.moving_part_name)
        palette_index = CONTACT_PART_PALETTE_INDEX[part_key]
        edge.palette_index = int(palette_index)
        used_indices.add(int(palette_index))


def _edge_centroid(points: np.ndarray) -> np.ndarray:
    return points.astype(np.float32).mean(axis=0)


def _swap_target_region_assignment(
    edge_a: DynamicContactEdge,
    edge_b: DynamicContactEdge,
) -> None:
    (
        edge_a.fixed_points,
        edge_b.fixed_points,
    ) = (
        edge_b.fixed_points,
        edge_a.fixed_points,
    )
    (
        edge_a.target_face_ids,
        edge_b.target_face_ids,
    ) = (
        edge_b.target_face_ids,
        edge_a.target_face_ids,
    )
    (
        edge_a.target_vertex_ids,
        edge_b.target_vertex_ids,
    ) = (
        edge_b.target_vertex_ids,
        edge_a.target_vertex_ids,
    )


def spatially_disambiguate_bilateral_contact_edges(
    contact_edges: list[DynamicContactEdge],
    init_verts_camera: np.ndarray,
) -> None:
    if len(contact_edges) < 2:
        return

    edge_by_key: dict[tuple[str, str, str], DynamicContactEdge] = {}
    for edge in contact_edges:
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

        _swap_target_region_assignment(left_edge, right_edge)
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


def project_mask_to_target_faces(
    mask_bool: np.ndarray,
    target_verts_camera: np.ndarray,
    target_faces_compact: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
) -> np.ndarray:
    _, _, pix_to_face = rasterize_depth_and_mask(
        target_verts_camera,
        target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
    )
    if pix_to_face.shape != mask_bool.shape:
        raise ValueError(
            "Rasterized pix_to_face and contact mask shapes disagree: "
            f"pix_to_face={pix_to_face.shape}, mask={mask_bool.shape}"
        )
    selected = pix_to_face[mask_bool & (pix_to_face >= 0)]
    if selected.size == 0:
        raise RuntimeError("Contact mask did not project onto any target mesh face.")
    return np.unique(selected.astype(np.int64))


def expand_face_set_along_surface(
    face_indices: np.ndarray,
    target_verts_camera: np.ndarray,
    target_faces_compact: np.ndarray,
    num_rings: int,
) -> np.ndarray:
    if face_indices.size == 0 or num_rings <= 0:
        return np.unique(face_indices.astype(np.int64))

    mesh = trimesh.Trimesh(
        vertices=target_verts_camera,
        faces=target_faces_compact,
        process=False,
    )
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return np.unique(face_indices.astype(np.int64))

    num_faces = int(target_faces_compact.shape[0])
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
    target_faces_compact: np.ndarray,
) -> np.ndarray:
    if face_indices.size == 0:
        raise RuntimeError("Cannot collect target vertices from an empty face set.")
    selected_faces = target_faces_compact[face_indices.astype(np.int64)]
    return np.unique(selected_faces.reshape(-1)).astype(np.int64)


def sample_face_set_surface_points(
    face_indices: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    if face_indices.size == 0:
        raise ValueError("Cannot sample target contact points from an empty face set.")
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
        raise ValueError("Cannot sample target contact points from zero-area faces.")

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


def sample_visible_scene_surface_points(
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
        "mode": "visible_scene_surface",
        "num_scene_faces_total": int(scene_faces.shape[0]),
        "num_sampled_points": int(sampled_points.shape[0]),
    }


def save_static_snapshot_references(
    snapshots_dir: Path,
    contact_edges: list[DynamicContactEdge],
    target_verts_camera: np.ndarray,
    target_faces_compact: np.ndarray,
    scene_no_target_verts_camera: np.ndarray,
    scene_no_target_faces_compact: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> None:
    target_verts_world = transform_camera_to_world(
        target_verts_camera,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    target_colors = np.tile(
        np.array([160, 160, 160], dtype=np.uint8),
        (target_verts_world.shape[0], 1),
    )
    for edge in contact_edges:
        if edge.target_vertex_ids is None:
            raise RuntimeError(
                f"Contact edge '{edge.moving_part_name}' has no target vertices."
            )
        rgb = palette_color_for_edge(int(edge.palette_index))
        target_colors[edge.target_vertex_ids] = np.array(rgb, dtype=np.uint8)
    write_colored_ascii_ply(
        snapshots_dir / "target.ply",
        target_verts_world.astype(np.float32),
        target_faces_compact,
        target_colors,
    )

    if scene_no_target_faces_compact.shape[0] > 0:
        scene_verts_world = transform_camera_to_world(
            scene_no_target_verts_camera,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        scene_colors = np.tile(
            np.array([200, 200, 200], dtype=np.uint8),
            (scene_verts_world.shape[0], 1),
        )
        write_colored_ascii_ply(
            snapshots_dir / "scene.ply",
            scene_verts_world.astype(np.float32),
            scene_no_target_faces_compact,
            scene_colors,
        )


def save_human_iteration_snapshot(
    snapshots_dir: Path,
    iter_idx: int,
    verts_camera: torch.Tensor,
    faces_np: np.ndarray,
    contact_edges: list[DynamicContactEdge],
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
    for edge in contact_edges:
        rgb = palette_color_for_edge(int(edge.palette_index))
        colors[edge.moving_vertex_ids] = np.array(rgb, dtype=np.uint8)

    write_colored_ascii_ply(
        snapshots_dir / f"human_iter_{iter_idx:04d}.ply",
        verts_world,
        faces_np,
        colors,
    )


def discover_human_tracks(segment_root: Path,
                          human_motion_root: Path) -> list[HumanTrack]:
    humans_dir = segment_root / "humans"
    if not humans_dir.is_dir():
        raise FileNotFoundError(
            f"Human segmentation directory not found: {humans_dir}")

    tracks: list[HumanTrack] = []
    for mask_track_dir in sorted(humans_dir.iterdir()):
        if not mask_track_dir.is_dir():
            continue
        mask_dir = mask_track_dir / "masks"
        result_dir = human_motion_root / mask_track_dir.name
        source_camera_mesh_dir = result_dir / "meshes" / "camera"
        if (
            mask_dir.is_dir()
            and result_dir.is_dir()
            and source_camera_mesh_dir.is_dir()
        ):
            tracks.append(
                HumanTrack(
                    name=mask_track_dir.name,
                    mask_dir=mask_dir,
                    result_dir=result_dir,
                    source_camera_mesh_dir=source_camera_mesh_dir,
                )
            )
    if not tracks:
        raise FileNotFoundError(
            "No matching human tracks with meshes/camera exports found between "
            f"{humans_dir} and {human_motion_root}"
        )
    return tracks


@dataclass
class DynamicContactEdge:
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
    target_face_ids: np.ndarray | None = None
    target_vertex_ids: np.ndarray | None = None
    palette_index: int = -1


class SMPLXAnglePrior(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip_idxs_signs = torch.as_tensor(
            [
                (1, 0, 1),
                (2, 0, 1),
                (3, 0, -1),
                (4, 0, -1),
                (5, 0, -1),
                (6, 0, -1),
                (7, 0, -1),
                (8, 0, -1),
                (9, 0, -1),
                (12, 0, -1),
                (13, 1, 1),
                (14, 1, -1),
                (16, 1, 1),
                (17, 1, -1),
                (18, 1, 1),
                (19, 1, -1),
            ]
        ).int()
        zero_idxs = torch.as_tensor(
            [
                (10, 0),
                (10, 1),
                (10, 2),
                (11, 0),
                (11, 1),
                (11, 2),
                (15, 0),
                (15, 1),
                (15, 2),
                (20, 1),
                (21, 1),
            ]
        ).int()
        self.register_buffer("clip_idxs_signs", clip_idxs_signs)
        self.register_buffer("zero_idxs", zero_idxs)

    def forward(self, pose: torch.Tensor, with_pelvis: bool = False) -> torch.Tensor:
        assert pose.ndim == 2
        cdata = self.clip_idxs_signs
        zdata = self.zero_idxs
        if not with_pelvis:
            assert pose.shape[1] == 21 * 3
            cdata = torch.clone(cdata)
            cdata[:, 0] -= 1
            zdata = torch.clone(zdata)
            zdata[:, 0] -= 1
        else:
            assert pose.shape[1] == 22 * 3

        cidxs = cdata[:, 0] * 3 + cdata[:, 1]
        csigns = cdata[:, 2]
        cres = F.relu(pose[:, cidxs] * torch.unsqueeze(csigns, 0))

        zidxs = zdata[:, 0] * 3 + zdata[:, 1]
        zres = torch.abs(pose[:, zidxs])
        return torch.mean(torch.cat((cres, zres), dim=1))


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


def build_default_paths(video_name: str) -> dict[str, Path]:
    defaults = build_shared_default_paths(video_name)
    defaults["smpl_folder"] = (
        SCRIPT_DIR.parent.parent / "GVHMR" / "inputs" / "checkpoints" / "body_models"
    )
    return defaults


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize first-frame full-body SMPL-X human grounding against a "
            "metric ScanNet scene using PAG contact semantics."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_01")
    parser.add_argument("--generated_root", type=str, default=None)
    parser.add_argument("--selection_json", type=str, default=None)
    parser.add_argument("--input_pag_json", type=str, default=None)
    parser.add_argument("--segment_root", type=str, default=None)
    parser.add_argument("--human_motion_root", type=str, default=None)
    parser.add_argument("--pag_json", type=str, default=None)
    parser.add_argument("--smpl_seg_json", type=str, default=None)
    parser.add_argument("--scannet_root", type=str, default=None)
    parser.add_argument("--smpl_folder", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--smpl_param_key", type=str, default="smpl_params_incam")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--target_surface_samples", type=int, default=6000)
    parser.add_argument("--mask_points", type=int, default=2500)
    parser.add_argument("--mask_vertex_samples", type=int, default=3000)
    parser.add_argument("--adam_iters", type=int, default=2000)
    parser.add_argument("--adam_lr", type=float, default=1e-3)
    parser.add_argument("--mask_weight", type=float, default=1)
    parser.add_argument("--root_orient_gvhmr_weight", type=float, default=20.0)
    parser.add_argument("--pose_gvhmr_weight", type=float, default=10.0)
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
        default=0.0,
    )
    parser.add_argument(
        "--scene_intersect_weight_end",
        type=float,
        default=10.0,
    )
    parser.add_argument("--scene_intersect_margin_m", type=float, default=0.02)
    parser.add_argument("--scene_intersect_surface_samples", type=int, default=700000)
    parser.add_argument(
        "--scene_intersect_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--nocontact_weight_start", type=float, default=500.0)
    parser.add_argument("--nocontact_weight_end", type=float, default=500.0)
    parser.add_argument("--floor_nocontact_weight_start", type=float, default=200.0)
    parser.add_argument("--floor_nocontact_weight_end", type=float, default=200.0)
    parser.add_argument("--angle_weight_start", type=float, default=0.0)
    parser.add_argument("--angle_weight_end", type=float, default=1.0)
    parser.add_argument("--self_intersect_weight_start", type=float, default=0.0)
    parser.add_argument("--self_intersect_weight_end", type=float, default=1e-3)
    parser.add_argument("--visible_tol_m", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--contact_masks_dir", type=str, default=None)
    parser.add_argument("--contact_region_expand_rings", type=int, default=0)
    parser.add_argument("--snapshot_every_iters", type=int, default=100)
    return parser.parse_args()


def get_loss_weights(
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> dict[str, float]:
    return {
        "mask": float(args.mask_weight),
        "root_orient_gvhmr": float(args.root_orient_gvhmr_weight),
        "pose_gvhmr": float(args.pose_gvhmr_weight),
        "height_prior": float(args.height_prior_weight),
        "scene_intersect": linear_weight(
            args.scene_intersect_weight_start,
            args.scene_intersect_weight_end,
            iteration,
            total_iters,
        ),
        "nocontact": linear_weight(
            args.nocontact_weight_start,
            args.nocontact_weight_end,
            iteration,
            total_iters,
        ),
        "floor_nocontact": linear_weight(
            args.floor_nocontact_weight_start,
            args.floor_nocontact_weight_end,
            iteration,
            total_iters,
        ),
        "angle": linear_weight(
            args.angle_weight_start,
            args.angle_weight_end,
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
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "iter": int(iteration),
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
    return row


def resolve_track_pag_name(track_name: str) -> str:
    return normalize_label(track_name)


def build_dynamic_contact_edges(
    pag_payload: dict[str, Any],
    track_name: str,
    target_object_name: str,
    segment_catalog: SmplxSegmentCatalog,
    contact_masks_dir: Path,
    target_verts_camera: np.ndarray,
    target_faces_compact: np.ndarray,
    camera_ctx: IdentityCameraContext,
    device: torch.device,
    expand_rings: int,
    surface_sample_seed: int,
    init_verts_camera: np.ndarray,
) -> list[DynamicContactEdge]:
    pag_edges = parse_pag_interaction_edges(pag_payload)
    target_object_norm = normalize_label(target_object_name)
    track_name_norm = resolve_track_pag_name(track_name)
    image_hw = (camera_ctx.height, camera_ctx.width)
    contact_edges: list[DynamicContactEdge] = []
    seen: set[tuple[str, str]] = set()

    for edge in pag_edges:
        nodes = [edge.node_a, edge.node_b]
        if sum(node.is_human for node in nodes) != 1:
            continue

        moving_node = nodes[0] if nodes[0].is_human else nodes[1]
        fixed_node = nodes[1] if nodes[0].is_human else nodes[0]
        if normalize_label(moving_node.entity_name) != track_name_norm:
            continue
        if normalize_label(fixed_node.entity_name) != target_object_norm:
            continue

        moving_part_name = normalize_label(moving_node.part_name)
        moving_segment_id = segment_catalog.get_contact_or_body_segment_id(
            moving_part_name
        )
        part_vert_ids = segment_catalog.get_indices(moving_segment_id)
        moving_segment_name = segment_catalog.get_display_name(
            moving_segment_id
        )

        dedup_key = (
            normalize_label(edge.node_a.raw_node),
            normalize_label(edge.node_b.raw_node),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        contact_mask = load_contact_mask_for_part(
            contact_masks_dir,
            moving_part_name,
            expected_hw=image_hw,
        )
        seed_face_ids = project_mask_to_target_faces(
            contact_mask,
            target_verts_camera,
            target_faces_compact,
            camera_ctx=camera_ctx,
            device=device,
        )
        if seed_face_ids.size == 0:
            raise RuntimeError(
                f"Contact mask for '{moving_part_name}' projects to no "
                f"target-mesh faces (mask path under {contact_masks_dir}). "
                "Check camera/mesh alignment or mask coverage."
            )
        expanded_face_ids = expand_face_set_along_surface(
            seed_face_ids,
            target_verts_camera,
            target_faces_compact,
            num_rings=int(expand_rings),
        )
        target_vertex_ids = face_set_to_unique_vertex_ids(
            expanded_face_ids,
            target_faces_compact,
        )
        if target_vertex_ids.size == 0:
            raise RuntimeError(
                f"Empty target vertex set for '{moving_part_name}' after "
                f"expansion ({expand_rings} rings)."
            )
        fixed_points_part = sample_face_set_surface_points(
            expanded_face_ids,
            verts=target_verts_camera,
            faces=target_faces_compact,
            num_samples=CONTACT_SURFACE_SAMPLES_PER_EDGE,
            seed=(
                CONTACT_SURFACE_SAMPLE_SEED
                + int(surface_sample_seed)
                + 97 * len(contact_edges)
            ),
        )
        print(
            f"  contact edge '{moving_part_name}': "
            f"seed_faces={seed_face_ids.size} -> "
            f"expanded_faces={expanded_face_ids.size} "
            f"target_vertices={target_vertex_ids.size} "
            f"target_surface_points={fixed_points_part.shape[0]}"
        )

        contact_edges.append(
            DynamicContactEdge(
                node_a=edge.node_a,
                node_b=edge.node_b,
                moving_node=moving_node,
                fixed_node=fixed_node,
                moving_part_name=moving_part_name,
                moving_segment_id=moving_segment_id,
                moving_segment_name=moving_segment_name,
                moving_vertex_ids=np.unique(
                    np.asarray(part_vert_ids, dtype=np.int64)
                ),
                fixed_points=fixed_points_part,
                reduction=_get_reduction((edge.node_a, edge.node_b)),
                target_face_ids=expanded_face_ids,
                target_vertex_ids=target_vertex_ids,
            )
        )

    if not contact_edges:
        raise RuntimeError(
            f"No PAG human-object contact edges found for {track_name} and "
            f"target object '{target_object_name}'."
        )
    spatially_disambiguate_bilateral_contact_edges(
        contact_edges,
        init_verts_camera=init_verts_camera,
    )
    assign_contact_palette_indices(contact_edges)
    for edge in contact_edges:
        rgb = palette_color_for_edge(int(edge.palette_index))
        print(
            f"  final correspondence '{edge.moving_part_name}' -> "
            f"'{edge.fixed_node.raw_node}': "
            f"human_vertices={edge.moving_vertex_ids.size} "
            f"target_vertices={int(edge.target_vertex_ids.size)} "
            f"target_surface_points={edge.fixed_points.shape[0]} "
            f"color_rgb={rgb}"
        )
    return contact_edges


def build_dynamic_floor_edges(
    track_name: str,
    segment_catalog: SmplxSegmentCatalog,
    floor_points_visible: np.ndarray,
) -> list[DynamicContactEdge]:
    if floor_points_visible.shape[0] == 0:
        raise RuntimeError("No visible floor points were available for floor contact.")
    floor_node = InteractionNode(
        raw_node="floor",
        entity_name="floor",
        part_name="floor",
        is_human=False,
    )
    contact_edges: list[DynamicContactEdge] = []
    for part_name in ("left foot", "right foot"):
        moving_segment_id = segment_catalog.get_contact_or_body_segment_id(part_name)
        part_vert_ids = segment_catalog.get_indices(moving_segment_id)
        moving_node = InteractionNode(
            raw_node=f"{track_name}.{part_name.replace(' ', '_')}",
            entity_name=track_name,
            part_name=part_name,
            is_human=True,
        )
        contact_edges.append(
            DynamicContactEdge(
                node_a=moving_node,
                node_b=floor_node,
                moving_node=moving_node,
                fixed_node=floor_node,
                moving_part_name=part_name,
                moving_segment_id=moving_segment_id,
                moving_segment_name=segment_catalog.get_display_name(
                    moving_segment_id
                ),
                moving_vertex_ids=np.unique(
                    np.asarray(part_vert_ids, dtype=np.int64)
                ),
                fixed_points=floor_points_visible.astype(np.float32),
                reduction=_get_reduction((moving_node, floor_node)),
            )
        )
    if len(contact_edges) != 2:
        raise RuntimeError(
            f"Floor contact edges were not built for both feet ({len(contact_edges)} edges)."
        )
    return contact_edges


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


def compute_root_orient_loss(
    current_orient_matrix: torch.Tensor,
    init_orient_matrix: torch.Tensor,
) -> torch.Tensor:
    relative = init_orient_matrix.transpose(0, 1) @ current_orient_matrix
    relative_aa = matrix_to_axis_angle(relative.view(1, 3, 3))[0]
    return torch.mean(relative_aa.pow(2))


def compute_contact_distance_loss(
    current_vertices: torch.Tensor,
    edges: list[DynamicContactEdge],
) -> torch.Tensor:
    if not edges:
        raise RuntimeError("Contact distance loss requires at least one contact edge.")
    values: list[torch.Tensor] = []
    for edge in edges:
        moving_points_seq = current_vertices[edge.moving_vertex_ids].unsqueeze(0)
        fixed_points = torch.from_numpy(edge.fixed_points).to(
            device=current_vertices.device,
            dtype=current_vertices.dtype,
        )
        if fixed_points.shape[0] == 0:
            raise RuntimeError(
                f"Contact edge '{edge.moving_part_name}' -> "
                f"'{edge.fixed_node.raw_node}' has no fixed target points."
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
    mask_vertex_ids_t: torch.Tensor,
    obs_mask_pixels_norm_t: torch.Tensor,
    contact_edges: list[DynamicContactEdge],
    floor_edges: list[DynamicContactEdge],
    scene_collision_points_t: torch.Tensor,
    scene_intersect_margin_m: float,
    intrinsics_t: torch.Tensor,
    width: int,
    height: int,
    init_params: dict[str, torch.Tensor],
    angle_prior: SMPLXAnglePrior,
    self_intersection_helper: SelfIntersectionHelper,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    current = params_module(smplx_layer)
    verts_camera = current["verts"]
    sampled_vertices = verts_camera[mask_vertex_ids_t]
    uv_pixels = project_points_torch(sampled_vertices, intrinsics_t)
    model_pixels_norm = torch.stack(
        [
            uv_pixels[:, 0] / max(float(width - 1), 1.0),
            uv_pixels[:, 1] / max(float(height - 1), 1.0),
        ],
        dim=1,
    )

    zero = torch.zeros((), device=verts_camera.device, dtype=verts_camera.dtype)

    d2_obs_to_model = min_distances_chunked(
        obs_mask_pixels_norm_t, model_pixels_norm
    )
    d2_model_to_obs = min_distances_chunked(
        model_pixels_norm, obs_mask_pixels_norm_t
    )
    mask_loss = 0.5 * (d2_obs_to_model.mean() + d2_model_to_obs.mean())

    root_orient_gvhmr = compute_root_orient_loss(
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
    nocontact = compute_contact_distance_loss(
        current_vertices=verts_camera,
        edges=contact_edges,
    )
    floor_nocontact = compute_contact_distance_loss(
        current_vertices=verts_camera,
        edges=floor_edges,
    )

    angle = angle_prior(current["body_pose"].view(1, -1), with_pelvis=False)
    if float(weights["self_intersect"]) > 0.0:
        self_intersect = self_intersection_helper(verts_camera, faces_t)
    else:
        self_intersect = zero

    total = (
        mask_loss * float(weights["mask"])
        + root_orient_gvhmr * float(weights["root_orient_gvhmr"])
        + pose_gvhmr * float(weights["pose_gvhmr"])
        + height_prior * float(weights["height_prior"])
        + scene_intersect * float(weights["scene_intersect"])
        + nocontact * float(weights["nocontact"])
        + floor_nocontact * float(weights["floor_nocontact"])
        + angle * float(weights["angle"])
        + self_intersect * float(weights["self_intersect"])
    )
    return {
        "total": total,
        "mask": mask_loss,
        "root_orient_gvhmr": root_orient_gvhmr,
        "pose_gvhmr": pose_gvhmr,
        "height_prior": height_prior,
        "scene_intersect": scene_intersect,
        "nocontact": nocontact,
        "floor_nocontact": floor_nocontact,
        "angle": angle,
        "self_intersect": self_intersect,
        "weights": weights,
        "current": current,
        "scene_intersect_stats": scene_intersect_stats,
    }


def compute_contact_metrics(
    current_vertices: np.ndarray,
    edges: list[DynamicContactEdge],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    device = torch.device("cpu")
    current_vertices_t = torch.from_numpy(current_vertices.astype(np.float32)).to(device)
    for edge in edges:
        fixed_points_t = torch.from_numpy(edge.fixed_points.astype(np.float32)).to(device)
        if fixed_points_t.shape[0] == 0:
            raise RuntimeError(
                f"Cannot compute contact metrics for "
                f"'{edge.moving_part_name}' -> '{edge.fixed_node.raw_node}': "
                "edge has no fixed target points."
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


def optimize_track(
    smplx_layer: Any,
    faces_t: torch.Tensor,
    init_params_np: dict[str, np.ndarray],
    obs_mask_pixels: np.ndarray,
    contact_edges: list[DynamicContactEdge],
    floor_edges: list[DynamicContactEdge],
    scene_collision_points: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    mask_vertex_ids: np.ndarray,
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
    angle_prior = SMPLXAnglePrior().to(device)
    self_intersection_helper = SelfIntersectionHelper()

    obs_mask_pixels_norm_t = torch.from_numpy(obs_mask_pixels.astype(np.float32)).to(device)
    obs_mask_pixels_norm_t[:, 0] /= max(float(width - 1), 1.0)
    obs_mask_pixels_norm_t[:, 1] /= max(float(height - 1), 1.0)
    intrinsics_t = torch.from_numpy(intrinsics.astype(np.float32)).to(device)
    mask_vertex_ids_t = torch.from_numpy(mask_vertex_ids.astype(np.int64)).to(device)
    scene_collision_points_t = torch.from_numpy(
        scene_collision_points.astype(np.float32)
    ).to(device)

    optimizer = torch.optim.Adam(params_module.parameters(), lr=float(args.adam_lr))
    iter_rows: list[dict[str, Any]] = []

    if int(args.adam_iters) <= 0:
        raise RuntimeError("adam_iters must be > 0.")

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

    for iter_idx in range(1, int(args.adam_iters) + 1):
        weights = get_loss_weights(args, iter_idx - 1, int(args.adam_iters) - 1)
        optimizer.zero_grad(set_to_none=True)
        losses = compute_loss_dict(
            params_module=params_module,
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            mask_vertex_ids_t=mask_vertex_ids_t,
            obs_mask_pixels_norm_t=obs_mask_pixels_norm_t,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            scene_collision_points_t=scene_collision_points_t,
            scene_intersect_margin_m=float(args.scene_intersect_margin_m),
            intrinsics_t=intrinsics_t,
            width=width,
            height=height,
            init_params=init_params_t,
            angle_prior=angle_prior,
            self_intersection_helper=self_intersection_helper,
            weights=weights,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params_module.parameters(), max_norm=5.0)
        optimizer.step()
        iter_rows.append(build_loss_row(iter_idx, losses))
        if (
            iter_idx == 1
            or iter_idx % max(int(args.log_every), 1) == 0
            or iter_idx == int(args.adam_iters)
        ):
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
                contact_edges=contact_edges,
                rotation_world_to_camera=rotation_world_to_camera,
                translation_world_to_camera=translation_world_to_camera,
            )
            print(f"  snapshot iter={iter_idx}")

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
            mask_vertex_ids_t=mask_vertex_ids_t,
            obs_mask_pixels_norm_t=obs_mask_pixels_norm_t,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            scene_collision_points_t=scene_collision_points_t,
            scene_intersect_margin_m=float(args.scene_intersect_margin_m),
            intrinsics_t=intrinsics_t,
            width=width,
            height=height,
            init_params=init_params_t,
            angle_prior=angle_prior,
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
        "scene_intersect_debug": scene_intersect_debug_payloads,
    }


def main() -> None:
    args = parse_args()
    defaults = build_default_paths(args.video_name)
    generated_root = resolve_path(args.generated_root, defaults["generated_root"])
    selection_json_path = resolve_path(args.selection_json, defaults["selection_json"])
    input_pag_json_path = resolve_path(args.input_pag_json, defaults["input_pag_json"])
    segment_root = resolve_path(args.segment_root, defaults["segment_root"])
    human_motion_root = resolve_path(args.human_motion_root, defaults["human_motion_root"])
    pag_json_path = (
        Path(args.pag_json).resolve()
        if args.pag_json is not None
        else find_pag_json(defaults["pag_root"])
    )
    smpl_seg_json_path = resolve_path(args.smpl_seg_json, defaults["smpl_seg_json"])
    smpl_folder = resolve_path(args.smpl_folder, defaults["smpl_folder"])
    contact_masks_dir = resolve_path(
        args.contact_masks_dir, defaults["contact_masks_dir"]
    )
    if not contact_masks_dir.is_dir():
        raise FileNotFoundError(
            f"Contact masks directory not found: {contact_masks_dir}"
        )
    output_root = ensure_dir(resolve_path(args.output_root, defaults["output_root"]))
    scene_root = ensure_dir(output_root / "scene")
    scene_depth_dir = ensure_dir(scene_root / "depth")
    debug_root = ensure_dir(output_root / "debug")
    summary_json_path = output_root / "alignment_summary.json"
    scannet_root = resolve_scannet_root(SCRIPT_DIR, args.scannet_root)
    device = parse_device(args.device)

    (
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    ) = load_camera_payload(generated_root / "resized_camera.json")
    first_frame_path = generated_root / "first_frames_resized" / "frame_00.png"
    if not first_frame_path.exists():
        raise FileNotFoundError(f"Generated first frame not found: {first_frame_path}")
    camera_ctx = build_identity_camera(
        intrinsics=intrinsics,
        width=width,
        height=height,
        device=device,
    )

    input_payload = load_json(input_pag_json_path)
    selection_payload = load_json(selection_json_path)
    pag_payload = load_json(pag_json_path)
    segment_catalog = load_smpl_segment_catalog(smpl_seg_json_path)
    scene_id = input_payload["scene_context"]["scene_id"]
    scene_paths = resolve_scene_paths(scannet_root, scene_id)

    print(f"Loading ScanNet scene mesh from: {scene_paths['mesh_path']}")
    scene_verts_world, scene_faces = load_mesh(scene_paths["mesh_path"])
    scene_verts_camera = transform_world_to_camera(
        scene_verts_world,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    scene_faces_in_view = filter_faces_to_camera_view(
        verts_camera=scene_verts_camera,
        faces=scene_faces,
        intrinsics=intrinsics,
        width=width,
        height=height,
        max_depth_m=20.0,
        border_px=96.0,
    )
    scene_verts_camera_render, scene_faces_render = compact_mesh(
        scene_verts_camera, scene_faces_in_view
    )
    if scene_faces_render.shape[0] == 0:
        raise RuntimeError("No scene faces remained after view-frustum filtering.")

    scene_depth, _, _ = rasterize_depth_and_mask(
        scene_verts_camera_render,
        scene_faces_render,
        camera_ctx=camera_ctx,
        device=device,
    )
    np.save(scene_depth_dir / "scene_depth.npy", scene_depth.astype(np.float32))
    save_depth_visualization(scene_depth_dir / "scene_depth_vis.png", scene_depth)

    segments_payload = load_json(scene_paths["segments_path"])
    anno_payload = load_json(scene_paths["segments_anno_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    candidate_faces, face_instance_ids, instance_meta = build_candidate_instances(
        mesh_faces=scene_faces,
        seg_indices=seg_indices,
        seg_groups=anno_payload["segGroups"],
    )

    target_instance_id = int(selection_payload["target_selection"]["instance_id"])
    target_meta = instance_meta.get(target_instance_id)
    if target_meta is None:
        raise KeyError(
            "Target instance_id "
            f"{target_instance_id} is not present in the scene annotations."
        )
    target_faces = candidate_faces[face_instance_ids == target_instance_id]
    if target_faces.shape[0] == 0:
        raise RuntimeError(f"No faces found for target instance {target_instance_id}.")
    target_verts_camera, target_faces_compact = compact_mesh(
        scene_verts_camera, target_faces
    )
    scene_faces_without_target = remove_faces(scene_faces_in_view, target_faces)
    scene_no_target_verts_camera, scene_no_target_faces_compact = compact_mesh(
        scene_verts_camera,
        scene_faces_without_target,
    )
    if scene_no_target_faces_compact.shape[0] == 0:
        raise RuntimeError("No non-target scene faces remained for snapshots.")
    target_surface_samples = sample_mesh_surface_points(
        verts=target_verts_camera,
        faces=target_faces_compact,
        num_samples=int(args.target_surface_samples),
        seed=int(args.seed),
    )
    _, target_visible_stats = build_visible_subset(
        sampled_points=target_surface_samples,
        mesh_verts=target_verts_camera,
        mesh_faces=target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
        visible_tol_m=float(args.visible_tol_m),
    )

    floor_faces = build_faces_for_labels(
        mesh_faces=scene_faces,
        seg_indices=seg_indices,
        seg_groups=anno_payload["segGroups"],
        labels={"floor"},
    )
    floor_verts_camera, floor_faces_compact = compact_mesh(scene_verts_camera, floor_faces)
    floor_surface_samples = sample_mesh_surface_points(
        verts=floor_verts_camera,
        faces=floor_faces_compact,
        num_samples=int(args.target_surface_samples),
        seed=int(args.seed),
    )
    floor_points_visible, _ = build_visible_subset(
        sampled_points=floor_surface_samples,
        mesh_verts=floor_verts_camera,
        mesh_faces=floor_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
        visible_tol_m=float(args.visible_tol_m),
    )

    tracks = discover_human_tracks(
        segment_root=segment_root,
        human_motion_root=human_motion_root,
    )
    smplx_layer = build_smplx_layer(smpl_folder, device)
    faces_np = np.asarray(smplx_layer.faces, dtype=np.int64)
    faces_t = torch.from_numpy(faces_np.astype(np.int64)).to(device)

    summary_tracks: list[dict[str, Any]] = []

    for track in tracks:
        print(f"\nProcessing human track: {track.name}")
        track_output_root = ensure_dir(output_root / track.name)
        meshes_root = ensure_dir(track_output_root / "meshes")
        debug_track_root = ensure_dir(debug_root / track.name)
        overlay_dir = ensure_dir(debug_track_root / "overlays")
        csv_dir = ensure_dir(debug_track_root / "csv")
        plot_dir = ensure_dir(debug_track_root / "plots" / "iter")
        params_dir = ensure_dir(debug_track_root / "params")
        snapshots_dir = ensure_dir(debug_track_root / "snapshots")
        scene_intersect_debug_dir = ensure_dir(debug_track_root / "scene_intersect")

        human_mask = load_mask(track.mask_dir / "frame_0000.png")
        if human_mask.shape != (height, width):
            raise ValueError(
                f"Human mask shape mismatch for {track.name}: "
                f"got {human_mask.shape[::-1]}, expected {(width, height)}"
            )
        obs_mask_pixels = sample_mask_pixels(
            mask=human_mask,
            max_points=int(args.mask_points),
            seed=int(args.seed),
        )
        init_params_torch = load_first_frame_smplx_params(track.result_dir, args.smpl_param_key)
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
            sample_visible_scene_surface_points(
                scene_verts_camera=scene_verts_camera,
                scene_faces=scene_faces_in_view,
                num_samples=int(args.scene_intersect_surface_samples),
                seed=int(args.seed) + 4242,
            )
        )

        rng = np.random.default_rng(int(args.seed))
        num_vertices = init_verts_camera.shape[0]
        mask_vertex_count = min(int(args.mask_vertex_samples), num_vertices)
        mask_vertex_ids = rng.choice(num_vertices, size=mask_vertex_count, replace=False)
        contact_edges = build_dynamic_contact_edges(
            pag_payload=pag_payload,
            track_name=track.name,
            target_object_name=selection_payload["target_selection"]["object"],
            segment_catalog=segment_catalog,
            contact_masks_dir=contact_masks_dir,
            target_verts_camera=target_verts_camera,
            target_faces_compact=target_faces_compact,
            camera_ctx=camera_ctx,
            device=device,
            expand_rings=int(args.contact_region_expand_rings),
            surface_sample_seed=int(args.seed),
            init_verts_camera=init_verts_camera,
        )
        save_static_snapshot_references(
            snapshots_dir=snapshots_dir,
            contact_edges=contact_edges,
            target_verts_camera=target_verts_camera,
            target_faces_compact=target_faces_compact,
            scene_no_target_verts_camera=scene_no_target_verts_camera,
            scene_no_target_faces_compact=scene_no_target_faces_compact,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        floor_edges = build_dynamic_floor_edges(
            track_name=track.name,
            segment_catalog=segment_catalog,
            floor_points_visible=floor_points_visible,
        )
        init_params_np = {
            key: value.detach().cpu().numpy().astype(np.float32)
            for key, value in init_params_torch.items()
        }

        init_depth, init_mask_rendered, _ = rasterize_depth_and_mask(
            init_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        init_mask_overlap = compute_binary_overlap(human_mask, init_mask_rendered)
        init_overlay = render_mask_overlay(
            background_bgr=read_bgr(first_frame_path),
            observed_mask=human_mask,
            rendered_mask=init_mask_rendered,
            title_lines=[
                f"{track.name}: init full-body",
                f"Human IoU: {init_mask_overlap['iou']:.3f}",
                f"Human Dice: {init_mask_overlap['dice']:.3f}",
            ],
        )
        cv2.imwrite(str(overlay_dir / "frame_0000_init_mask_overlay.png"), init_overlay)
        save_depth_visualization(overlay_dir / "frame_0000_init_depth_vis.png", init_depth)
        init_contact_metrics = compute_contact_metrics(init_verts_camera, contact_edges + floor_edges)

        optimization = optimize_track(
            smplx_layer=smplx_layer,
            faces_t=faces_t,
            init_params_np=init_params_np,
            obs_mask_pixels=obs_mask_pixels,
            contact_edges=contact_edges,
            floor_edges=floor_edges,
            scene_collision_points=scene_collision_points,
            intrinsics=intrinsics,
            width=width,
            height=height,
            mask_vertex_ids=mask_vertex_ids,
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
        final_depth, final_mask_rendered, _ = rasterize_depth_and_mask(
            final_verts_camera,
            faces_np,
            camera_ctx=camera_ctx,
            device=device,
        )
        final_mask_overlap = compute_binary_overlap(human_mask, final_mask_rendered)
        final_overlay = render_mask_overlay(
            background_bgr=read_bgr(first_frame_path),
            observed_mask=human_mask,
            rendered_mask=final_mask_rendered,
            title_lines=[
                f"{track.name}: final full-body",
                f"Human IoU: {final_mask_overlap['iou']:.3f}",
                f"Human Dice: {final_mask_overlap['dice']:.3f}",
            ],
        )
        cv2.imwrite(str(overlay_dir / "frame_0000_final_mask_overlay.png"), final_overlay)
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
            title_prefix=f"{track.name} Iter",
        )

        final_contact_metrics = compute_contact_metrics(final_verts_camera, contact_edges + floor_edges)

        summary_tracks.append(
            {
                "name": track.name,
                "optimization": {
                    "final_iter": int(optimization["final_iter"]),
                    "final_total_loss": float(optimization["final_total_loss"]),
                    "scene_intersect_sampling": scene_collision_sampling_stats,
                    "scene_intersect_stats": optimization["scene_intersect_stats"],
                    "scene_intersect_debug": optimization["scene_intersect_debug"],
                },
                "init_frame_0": {
                    "mask_overlap": init_mask_overlap,
                    "contact_edges": init_contact_metrics,
                },
                "final_frame_0": {
                    "mask_overlap": final_mask_overlap,
                    "contact_edges": final_contact_metrics,
                },
                "artifacts": {
                    "camera_mesh": str(meshes_root / "frame_0000_camera.ply"),
                    "world_mesh": str(meshes_root / "frame_0000_world.ply"),
                    "init_overlay": str(overlay_dir / "frame_0000_init_mask_overlay.png"),
                    "final_overlay": str(overlay_dir / "frame_0000_final_mask_overlay.png"),
                    "optimized_params": str(params_dir / "optimized_frame_0000.pt"),
                    "scene_intersect_debug": str(scene_intersect_debug_dir),
                    "csv": {
                        "iter_metrics": str(iter_metrics_csv),
                        "final_loss_summary": str(final_loss_summary_csv),
                    },
                },
            }
        )

    save_json(
        summary_json_path,
        {
            "video_name": args.video_name,
            "scene_id": scene_id,
            "target_selection": selection_payload["target_selection"],
            "optimizer": {
                "adam_iters": int(args.adam_iters),
                "adam_lr": float(args.adam_lr),
                "loss_weights": {
                    "mask": float(args.mask_weight),
                    "root_orient_gvhmr": float(args.root_orient_gvhmr_weight),
                    "pose_gvhmr": float(args.pose_gvhmr_weight),
                    "height_prior": float(args.height_prior_weight),
                    "scene_intersect": {
                        "start": float(args.scene_intersect_weight_start),
                        "end": float(args.scene_intersect_weight_end),
                        "clearance_margin_m": float(args.scene_intersect_margin_m),
                        "surface_samples": int(args.scene_intersect_surface_samples),
                        "debug": bool(args.scene_intersect_debug),
                    },
                    "nocontact": {
                        "start": float(args.nocontact_weight_start),
                        "end": float(args.nocontact_weight_end),
                    },
                    "floor_nocontact": {
                        "start": float(args.floor_nocontact_weight_start),
                        "end": float(args.floor_nocontact_weight_end),
                    },
                    "angle": {
                        "start": float(args.angle_weight_start),
                        "end": float(args.angle_weight_end),
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
                "target_instance_id": target_instance_id,
                "target_label": target_meta["label"],
                "target_visible_surface": target_visible_stats,
                "num_floor_visible_points": int(floor_points_visible.shape[0]),
            },
            "tracks": summary_tracks,
        },
    )
    print(f"\nDone. Full-body outputs saved to: {output_root}")


if __name__ == "__main__":
    main()
