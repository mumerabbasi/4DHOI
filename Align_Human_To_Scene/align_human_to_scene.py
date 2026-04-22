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
import torch
import torch.nn.functional as F
import trimesh

from pytorch3d.ops import knn_points
from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

ALIGN_LOSS_TERM_KEYS = (
    "mask",
    "front",
    "scale_reg",
    "translation_reg",
    "nocontact",
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


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
    is_continuous: bool
    is_rel_static: bool


@dataclass
class ContactEdgeData:
    node_a: InteractionNode
    node_b: InteractionNode
    moving_node: InteractionNode
    fixed_node: InteractionNode
    moving_segment_name: str
    is_continuous: bool
    reduction: str
    moving_points_base: np.ndarray
    fixed_points_base: np.ndarray


@dataclass
class ContactEdgeTorch:
    node_a: InteractionNode
    node_b: InteractionNode
    moving_node: InteractionNode
    fixed_node: InteractionNode
    moving_segment_name: str
    is_continuous: bool
    reduction: str
    moving_points_base: torch.Tensor
    fixed_points_base: torch.Tensor


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


def snake_to_pag_name(segment_name: str) -> str:
    return normalize_label(segment_name.replace("_inner", "").replace("_", " "))


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


def resize_and_center_crop(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    interpolation: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    src_height, src_width = image.shape[:2]
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"Invalid source image shape: {image.shape}")

    scale = target_width / float(src_width)
    scaled_height = int(round(src_height * scale))
    if scaled_height < target_height:
        raise ValueError(
            "Scaled height is smaller than the requested crop height: "
            f"src={(src_width, src_height)}, scaled_height={scaled_height}, "
            f"target_height={target_height}"
        )

    resized = cv2.resize(
        image,
        (target_width, scaled_height),
        interpolation=interpolation,
    )
    crop_top = (scaled_height - target_height) // 2
    cropped = resized[crop_top:crop_top + target_height, :]
    transform = {
        "source_width": int(src_width),
        "source_height": int(src_height),
        "target_width": int(target_width),
        "target_height": int(target_height),
        "scale": float(scale),
        "scaled_height": int(scaled_height),
        "crop_top": int(crop_top),
    }
    return cropped, transform


def load_camera_payload(
    camera_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, int, int]:
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
        payload,
        intrinsics,
        rotation_world_to_camera,
        translation_world_to_camera,
        width,
        height,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align the first-frame GVHMR human to a metric ScanNet scene, "
            "then apply the same similarity transform to the full motion "
            "sequence."
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
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device for rasterization and optimization.",
    )
    parser.add_argument(
        "--surface_samples",
        type=int,
        default=6000,
        help="Number of human surface samples used in optimization.",
    )
    parser.add_argument(
        "--target_surface_samples",
        type=int,
        default=6000,
        help=(
            "Number of target-object surface samples cached for contact "
            "alignment."
        ),
    )
    parser.add_argument(
        "--mask_points",
        type=int,
        default=2500,
        help=(
            "Maximum number of human-mask pixels sampled for the 2D "
            "silhouette loss."
        ),
    )
    parser.add_argument(
        "--stage1_iters",
        type=int,
        default=4000,
        help="Optimization iterations for the stage-1 mask+front pass.",
    )
    parser.add_argument(
        "--stage2_iters",
        type=int,
        default=2500,
        help="Optimization iterations for the stage-2 contact-enabled pass.",
    )
    parser.add_argument(
        "--stage1_lr",
        type=float,
        default=0.005,
        help="Adam learning rate for stage 1.",
    )
    parser.add_argument(
        "--stage2_lr",
        type=float,
        default=0.005,
        help="Adam learning rate for stage 2.",
    )
    parser.add_argument(
        "--front_margin_m",
        type=float,
        default=0.03,
        help=(
            "Human points should stay at least this much in front of scene "
            "depth."
        ),
    )
    parser.add_argument(
        "--mask_weight",
        type=float,
        default=1.0,
        help="Fixed weight for the 2D silhouette chamfer term.",
    )
    parser.add_argument(
        "--front_weight",
        type=float,
        default=20.0,
        help="Fixed weight for the front-of-scene depth term.",
    )
    parser.add_argument(
        "--scale_reg_weight",
        type=float,
        default=0.01,
        help="Fixed weight for the scale regularizer.",
    )
    parser.add_argument(
        "--translation_reg_weight",
        type=float,
        default=0.5,
        help="Fixed weight for the translation regularizer.",
    )
    parser.add_argument(
        "--nocontact_weight_start",
        type=float,
        default=100.0,
        help="Stage-2 start weight for the nocontact term.",
    )
    parser.add_argument(
        "--nocontact_weight_end",
        type=float,
        default=100.0,
        help="Stage-2 end weight for the nocontact term.",
    )
    parser.add_argument(
        "--init_depth_offset_m",
        type=float,
        default=0.25,
        help=(
            "Depth offset that places the human slightly in front of the "
            "local scene depth."
        ),
    )
    parser.add_argument(
        "--visible_tol_m",
        type=float,
        default=0.02,
        help=(
            "Depth tolerance used when keeping only rasterized-visible "
            "surface samples."
        ),
    )
    parser.add_argument(
        "--human_dilate_kernel",
        type=int,
        default=25,
        help=(
            "Odd kernel size used to dilate the human mask for the "
            "depth-based init."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for surface and mask subsampling.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Iteration logging frequency for both optimization stages.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path if raw_path is None else Path(raw_path).resolve()


def build_default_paths(video_name: str) -> dict[str, Path]:
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
    }


def find_pag_json(pag_dir: Path) -> Path:
    candidates = sorted(pag_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No PAG JSON found in: {pag_dir}")
    for candidate in candidates:
        if candidate.name.startswith("output_pag"):
            return candidate
    return candidates[0]


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
    if not rows:
        return
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


def get_stage2_loss_weights(
    args: argparse.Namespace,
    iteration: int,
    total_iters: int,
) -> dict[str, float]:
    return {
        "mask": float(args.mask_weight),
        "front": float(args.front_weight),
        "scale_reg": float(args.scale_reg_weight),
        "translation_reg": float(args.translation_reg_weight),
        "nocontact": linear_weight(
            args.nocontact_weight_start,
            args.nocontact_weight_end,
            iteration,
            total_iters,
        ),
    }


def get_stage1_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "mask": float(args.mask_weight),
        "front": float(args.front_weight),
        "scale_reg": float(args.scale_reg_weight),
        "translation_reg": float(args.translation_reg_weight),
        "nocontact": 0.0,
    }


def build_loss_row(
    iteration: int,
    stage: str,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "iter": int(iteration),
        "stage": stage,
        "total": float(losses["total"].detach().cpu().item()),
    }
    for key in ALIGN_LOSS_TERM_KEYS:
        weight = float(weights[key])
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = weight
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = weight * raw_value
    return row


def format_loss_log(
    iteration: int,
    total_iterations: int,
    stage: str,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> list[str]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    weights_fmt = "  ".join(
        f"{key}={weights[key]:.4g}" for key in ALIGN_LOSS_TERM_KEYS)
    raw_fmt = "  ".join(
        f"{key}={float(losses[key].detach().cpu().item()):.5f}"
        for key in ALIGN_LOSS_TERM_KEYS
    )
    scaled_fmt = "  ".join(
        f"{key}={weights[key] * float(losses[key].detach().cpu().item()):.5f}"
        for key in ALIGN_LOSS_TERM_KEYS
    )
    return [
        f"  [{stage}:{iteration:4d}/{total_iterations}] "
        f"total={float(losses['total'].detach().cpu().item()):.5f}",
        f"      weights: {weights_fmt}",
        f"      raw:     {raw_fmt}",
        f"      scaled:  {scaled_fmt}",
    ]


def build_final_loss_summary_row(
    best_iter: int,
    stage: str,
    losses: dict[str, torch.Tensor | dict[str, float]],
) -> dict[str, Any]:
    weights = losses["weights"]
    assert isinstance(weights, dict)
    row: dict[str, Any] = {
        "best_iter": int(best_iter),
        "stage": stage,
        "best_total_loss": float(losses["total"].detach().cpu().item()),
        "total_scaled": float(losses["total"].detach().cpu().item()),
    }
    for key in ALIGN_LOSS_TERM_KEYS:
        raw_value = float(losses[key].detach().cpu().item())
        row[f"{key}_weight"] = float(weights[key])
        row[f"{key}_raw"] = raw_value
        row[f"{key}_scaled"] = float(weights[key]) * raw_value
    return row


def save_loss_plot_tree(
    plot_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    total_key: str,
    term_keys: tuple[str, ...] | list[str],
    x_label: str,
    title_prefix: str,
) -> None:
    if not rows:
        return

    ensure_dir(plot_dir)
    raw_dir = ensure_dir(plot_dir / "raw")
    scaled_dir = ensure_dir(plot_dir / "scaled")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

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
                is_continuous=bool(edge_payload.get("is_continuous", True)),
                is_rel_static=bool(edge_payload.get("is_rel_static", False)),
            )
        )
    return edges


def load_smpl_segments(
    seg_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw = load_json(seg_path)
    raw_segments = raw.get("segments")
    if not isinstance(raw_segments, dict):
        raise KeyError(
            f"Expected a 'segments' mapping in {seg_path}, but it was not found."
        )

    body_segment_keys = raw.get("project_body_part_order")
    body_part_nodes = raw.get("project_body_part_nodes")
    if not isinstance(body_segment_keys, list) or not isinstance(body_part_nodes, list):
        raise KeyError(
            "Expected 'project_body_part_order' and 'project_body_part_nodes' in "
            f"{seg_path}."
        )
    if len(body_segment_keys) != len(body_part_nodes):
        raise ValueError(
            "The SMPL-X segmentation asset has mismatched body-part key/name lists: "
            f"{len(body_segment_keys)} vs {len(body_part_nodes)}."
        )

    body_segments: dict[str, np.ndarray] = {}
    for segment_key, part_name in zip(body_segment_keys, body_part_nodes):
        indices = raw_segments.get(segment_key)
        if indices is None:
            raise KeyError(
                f"Missing body segment '{segment_key}' in the SMPL-X segmentation asset."
            )
        body_segments[normalize_label(str(part_name))] = np.unique(
            np.asarray(indices, dtype=np.int64)
        )

    contact_segment_keys = raw.get("contact_segment_names")
    if not isinstance(contact_segment_keys, list):
        raise KeyError(
            f"Expected 'contact_segment_names' in {seg_path}, but it was not found."
        )

    contact_segments: dict[str, np.ndarray] = {}
    for segment_key in contact_segment_keys:
        indices = raw_segments.get(segment_key)
        if indices is None:
            raise KeyError(
                f"Missing contact segment '{segment_key}' in the SMPL-X segmentation asset."
            )
        contact_segments[snake_to_pag_name(str(segment_key))] = np.unique(
            np.asarray(indices, dtype=np.int64)
        )

    return body_segments, contact_segments


def resolve_contact_reduction(part_name: str) -> str:
    if part_name.split(" ")[-1] in ("hand", "foot"):
        return "mean"
    return "min"


def pcd_distance(
    p1: torch.Tensor | None,
    p2: torch.Tensor | None,
    reduction: str = "min",
) -> torch.Tensor | None:
    if p1 is None or p2 is None:
        return None
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
        return np.zeros((0, 3), dtype=np.float32)

    triangles = verts[faces]
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_sum = float(np.sum(areas))
    rng = np.random.default_rng(seed)

    if not np.isfinite(area_sum) or area_sum <= 1e-8:
        chosen = rng.choice(verts.shape[0], size=int(
            num_samples), replace=True)
        return verts[chosen].astype(np.float32)

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
        return np.zeros(
            (0, 3), dtype=np.float32), np.zeros(
            (0, 3), dtype=np.int64)
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

    if np.count_nonzero(visible) >= 64:
        mode = "strict_visible"
        keep = visible
    elif np.count_nonzero(in_frame) >= 64:
        mode = "fallback_in_frame"
        keep = in_frame
    else:
        mode = "fallback_all"
        keep = z > 1e-6

    return sampled_points[keep].astype(np.float32), {
        "mode": mode,
        "num_total_points": int(sampled_points.shape[0]),
        "num_in_frame_points": int(np.count_nonzero(in_frame)),
        "num_visible_points": int(np.count_nonzero(visible)),
        "num_kept_points": int(np.count_nonzero(keep)),
        "visible_tol_m": float(visible_tol_m),
    }


def sample_mask_pixels(
    mask: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
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
        return torch.zeros((0,), device=obs.device, dtype=obs.dtype)

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


def sample_image_bilinear(
    image_hw: torch.Tensor,
    uv_pixels: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    if uv_pixels.shape[0] == 0:
        return torch.zeros((0,), device=image_hw.device, dtype=image_hw.dtype)
    x_norm = (2.0 * uv_pixels[:, 0] / max(width - 1, 1)) - 1.0
    y_norm = (2.0 * uv_pixels[:, 1] / max(height - 1, 1)) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        image_hw[None, None],
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled.view(-1)


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


def write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


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


def list_human_mesh_frames(human_mesh_dir: Path) -> list[Path]:
    frames = sorted(human_mesh_dir.glob("frame_*.ply"))
    if not frames:
        raise FileNotFoundError(
            f"No frame_*.ply files found in: {human_mesh_dir}")
    return frames


def resolve_track_pag_name(track_name: str) -> str:
    return normalize_label(track_name)


def build_contact_edges(
    pag_payload: dict[str, Any],
    track_name: str,
    target_object_name: str,
    body_segments: dict[str, np.ndarray],
    contact_segments: dict[str, np.ndarray],
    human_verts: np.ndarray,
    target_points_visible: np.ndarray,
) -> list[ContactEdgeData]:
    pag_edges = parse_pag_interaction_edges(pag_payload)
    target_object_norm = normalize_label(target_object_name)
    track_name_norm = resolve_track_pag_name(track_name)
    contact_edges: list[ContactEdgeData] = []
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
        part_vert_ids = contact_segments.get(
            moving_part_name,
            body_segments.get(moving_part_name),
        )
        moving_segment_name = (
            f"{moving_part_name} contact"
            if moving_part_name in contact_segments
            else moving_part_name
        )
        if part_vert_ids is None:
            raise KeyError(
                "Unsupported human contact part "
                f"'{moving_node.part_name}' for {track_name}. Missing "
                "SMPL-X segmentation mapping."
            )

        dedup_key = (
            normalize_label(edge.node_a.raw_node),
            normalize_label(edge.node_b.raw_node),
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        moving_points_base = human_verts[part_vert_ids].astype(np.float32)
        if moving_points_base.shape[0] == 0:
            raise RuntimeError(
                f"No vertices found for human part '{moving_part_name}' in "
                f"track {track_name}."
            )
        if target_points_visible.shape[0] == 0:
            raise RuntimeError(
                "Target visible surface samples are required for nocontact "
                "loss computation."
            )

        contact_edges.append(
            ContactEdgeData(
                node_a=edge.node_a,
                node_b=edge.node_b,
                moving_node=moving_node,
                fixed_node=fixed_node,
                moving_segment_name=moving_segment_name,
                is_continuous=bool(edge.is_continuous),
                reduction=resolve_contact_reduction(moving_part_name),
                moving_points_base=moving_points_base,
                fixed_points_base=target_points_visible.astype(np.float32),
            )
        )

    if not contact_edges:
        raise RuntimeError(
            f"No PAG human-object contact edges found for {track_name} and "
            f"target object '{target_object_name}'."
        )
    return contact_edges


def contact_edges_to_torch(
    contact_edges: list[ContactEdgeData],
    device: torch.device,
) -> list[ContactEdgeTorch]:
    edges_torch: list[ContactEdgeTorch] = []
    for edge in contact_edges:
        edges_torch.append(
            ContactEdgeTorch(
                node_a=edge.node_a,
                node_b=edge.node_b,
                moving_node=edge.moving_node,
                fixed_node=edge.fixed_node,
                moving_segment_name=edge.moving_segment_name,
                is_continuous=edge.is_continuous,
                reduction=edge.reduction,
                moving_points_base=torch.from_numpy(
                    edge.moving_points_base
                ).to(
                    device=device,
                    dtype=torch.float32,
                ),
                fixed_points_base=torch.from_numpy(edge.fixed_points_base).to(
                    device=device,
                    dtype=torch.float32,
                ),
            )
        )
    return edges_torch


def compute_contact_edge_metrics(
    contact_edges: list[ContactEdgeData],
    scale: float,
    tx: float,
    ty: float,
    tz: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    translation_t = torch.tensor(
        [tx, ty, tz],
        dtype=torch.float32,
        device=device,
    )
    metrics: list[dict[str, Any]] = []

    for edge in contact_edges:
        moving_points_t = torch.from_numpy(edge.moving_points_base).to(
            device=device,
            dtype=torch.float32,
        )
        fixed_points_t = torch.from_numpy(edge.fixed_points_base).to(
            device=device,
            dtype=torch.float32,
        )
        if fixed_points_t.shape[0] == 0:
            continue
        moving_points_seq = (
            float(scale) * moving_points_t + translation_t[None]
        ).unsqueeze(0)
        fixed_points_seq = fixed_points_t.unsqueeze(0)
        pdists = pcd_distance(
            moving_points_seq,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        if pdists is None:
            continue
        nocontact_raw = (
            pdists.mean() if edge.is_continuous else pdists.min()
        ).detach().cpu().item()
        metrics.append(
            {
                "node_a": edge.node_a.raw_node,
                "node_b": edge.node_b.raw_node,
                "moving_entity_name": edge.moving_node.entity_name,
                "moving_part_name": edge.moving_node.part_name,
                "moving_segment_name": edge.moving_segment_name,
                "fixed_entity_name": edge.fixed_node.entity_name,
                "fixed_part_name": edge.fixed_node.part_name,
                "reduction": edge.reduction,
                "is_continuous": bool(edge.is_continuous),
                "nocontact_raw": float(nocontact_raw),
                "nocontact_distance_m": float(
                    math.sqrt(max(nocontact_raw, 0.0))
                ),
            }
        )

    return metrics


def compute_init_depth_translation(
    scene_depth: np.ndarray,
    human_mask: np.ndarray,
    human_verts: np.ndarray,
    kernel_size: int,
    init_depth_offset_m: float,
    target_points_visible: np.ndarray,
) -> tuple[float, dict[str, float]]:
    if kernel_size <= 0:
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(human_mask.astype(np.uint8), kernel, iterations=1) > 0
    region = scene_depth[np.logical_and(dilated, scene_depth > 0.0)]
    region_source = "dilated_human_mask"
    if region.size == 0:
        region = scene_depth[scene_depth > 0.0]
        region_source = "global_scene_depth"
    if region.size == 0 and target_points_visible.shape[0] > 0:
        region = target_points_visible[:, 2]
        region_source = "target_surface_depth"
    if region.size == 0:
        raise RuntimeError(
            "Could not determine a valid scene depth region for "
            "initialization."
        )

    scene_depth_median = float(np.median(region))
    target_depth = scene_depth_median - float(init_depth_offset_m)
    human_depth_median = float(np.median(human_verts[:, 2]))
    tz_init = target_depth - human_depth_median
    return tz_init, {
        "scene_depth_region_source": region_source,
        "scene_depth_region_count": int(region.size),
        "scene_depth_region_median_m": scene_depth_median,
        "target_depth_m": target_depth,
        "human_median_depth_m": human_depth_median,
        "tz_init_m": tz_init,
    }


def build_observed_projected_target_mask(
    selection_json_path: Path,
    width: int,
    height: int,
) -> np.ndarray:
    projected_mask_path = selection_json_path.parent / \
        "3d" / "projected_target_mask.png"
    if not projected_mask_path.exists():
        raise FileNotFoundError(
            f"Projected target mask not found: {projected_mask_path}")
    projected_mask = load_mask(projected_mask_path)
    resized_mask_u8, _ = resize_and_center_crop(
        projected_mask.astype(np.uint8) * 255,
        width,
        height,
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_mask_u8 > 127


def render_target_overlay(
    background_bgr: np.ndarray,
    mask: np.ndarray,
    lines: list[str],
) -> np.ndarray:
    overlay = background_bgr.astype(np.float32).copy()
    overlay[mask] = 0.6 * overlay[mask] + 0.4 * \
        np.array([0, 255, 255], dtype=np.float32)
    overlay_u8 = np.clip(overlay, 0.0, 255.0).astype(np.uint8)
    y = 30
    for line in lines:
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


def compute_alignment_loss(
    human_points_visible_base: torch.Tensor,
    obs_mask_pixels_norm: torch.Tensor,
    human_mask_t: torch.Tensor,
    scene_depth_t: torch.Tensor,
    contact_edges_t: list[ContactEdgeTorch],
    intrinsics_t: torch.Tensor,
    width: int,
    height: int,
    log_scale: torch.Tensor,
    delta_tx: torch.Tensor,
    delta_ty: torch.Tensor,
    delta_tz: torch.Tensor,
    tx_init_t: torch.Tensor,
    ty_init_t: torch.Tensor,
    tz_init_t: torch.Tensor,
    front_margin_m: float,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    scale = torch.exp(log_scale)
    tx = tx_init_t + delta_tx
    ty = ty_init_t + delta_ty
    tz = tz_init_t + delta_tz
    translation = torch.stack([tx, ty, tz])

    model_points = scale * human_points_visible_base + translation[None]
    uv_pixels = project_points_torch(model_points, intrinsics_t)
    model_pixels_norm = torch.stack(
        [
            uv_pixels[:, 0] / max(float(width - 1), 1.0),
            uv_pixels[:, 1] / max(float(height - 1), 1.0),
        ],
        dim=1,
    )

    zero = torch.zeros(
        (),
        device=human_points_visible_base.device,
        dtype=human_points_visible_base.dtype,
    )

    if obs_mask_pixels_norm.shape[0] > 0 and model_pixels_norm.shape[0] > 0:
        d2_obs_to_model = min_distances_chunked(
            obs_mask_pixels_norm, model_pixels_norm)
        d2_model_to_obs = min_distances_chunked(
            model_pixels_norm, obs_mask_pixels_norm)
        mask_loss = 0.5 * (d2_obs_to_model.mean() + d2_model_to_obs.mean())
    else:
        mask_loss = zero

    depth_at_uv = sample_image_bilinear(
        scene_depth_t, uv_pixels, width=width, height=height)
    mask_at_uv = sample_image_bilinear(
        human_mask_t, uv_pixels, width=width, height=height)
    in_frame = (
        (uv_pixels[:, 0] >= 0.0)
        & (uv_pixels[:, 0] <= float(width - 1))
        & (uv_pixels[:, 1] >= 0.0)
        & (uv_pixels[:, 1] <= float(height - 1))
        & (model_points[:, 2] > 1e-6)
        & (depth_at_uv > 0.0)
        & (mask_at_uv > 0.1)
    )
    if torch.any(in_frame):
        penetration = model_points[in_frame, 2] - \
            (depth_at_uv[in_frame] - float(front_margin_m))
        front_loss = torch.relu(penetration).pow(2).mean()
    else:
        front_loss = zero

    nocontact_values: list[torch.Tensor] = []
    for edge in contact_edges_t:
        if edge.fixed_points_base.shape[0] == 0:
            continue
        moving_points_seq = (
            scale * edge.moving_points_base + translation[None]
        ).unsqueeze(0)
        fixed_points_seq = edge.fixed_points_base.unsqueeze(0)
        pdists = pcd_distance(
            moving_points_seq,
            fixed_points_seq,
            reduction=edge.reduction,
        )
        if pdists is None:
            continue
        if edge.is_continuous:
            nocontact_values.append(pdists.mean())
        else:
            nocontact_values.append(pdists.min())

    if nocontact_values:
        loss_nocontact = torch.stack(nocontact_values, dim=0).mean()
    else:
        loss_nocontact = zero

    scale_reg = log_scale.pow(2)
    translation_reg = delta_tx.pow(2) + delta_ty.pow(2) + delta_tz.pow(2)
    total = (
        mask_loss * float(weights["mask"])
        + front_loss * float(weights["front"])
        + scale_reg * float(weights["scale_reg"])
        + translation_reg * float(weights["translation_reg"])
        + loss_nocontact * float(weights["nocontact"])
    )
    return {
        "total": total,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "translation": translation,
        "mask": mask_loss,
        "front": front_loss,
        "nocontact": loss_nocontact,
        "scale_reg": scale_reg,
        "translation_reg": translation_reg,
        "weights": weights,
    }


def optimize_alignment(
    human_points_visible_base_np: np.ndarray,
    obs_mask_pixels_np: np.ndarray,
    human_mask: np.ndarray,
    scene_depth: np.ndarray,
    contact_edges: list[ContactEdgeData],
    intrinsics: np.ndarray,
    width: int,
    height: int,
    tx_init: float,
    ty_init: float,
    tz_init: float,
    stage1_iters: int,
    stage2_iters: int,
    stage1_lr: float,
    stage2_lr: float,
    front_margin_m: float,
    log_every: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    human_points_visible_base_t = torch.from_numpy(
        human_points_visible_base_np).to(device=device, dtype=torch.float32, )
    obs_mask_pixels_norm_t = torch.from_numpy(obs_mask_pixels_np).to(
        device=device,
        dtype=torch.float32,
    )
    if obs_mask_pixels_norm_t.numel() > 0:
        obs_mask_pixels_norm_t[:, 0] /= max(float(width - 1), 1.0)
        obs_mask_pixels_norm_t[:, 1] /= max(float(height - 1), 1.0)
    human_mask_t = torch.from_numpy(
        human_mask.astype(np.float32)).to(device=device)
    scene_depth_t = torch.from_numpy(
        scene_depth.astype(np.float32)).to(device=device)
    contact_edges_t = contact_edges_to_torch(contact_edges, device)
    intrinsics_t = torch.from_numpy(
        intrinsics.astype(np.float32)).to(device=device)

    tx_init_t = torch.tensor(
        float(tx_init), device=device, dtype=torch.float32)
    ty_init_t = torch.tensor(
        float(ty_init), device=device, dtype=torch.float32)
    tz_init_t = torch.tensor(
        float(tz_init), device=device, dtype=torch.float32)

    log_scale = torch.nn.Parameter(torch.tensor(
        0.0, device=device, dtype=torch.float32))
    delta_tx = torch.nn.Parameter(torch.tensor(
        0.0, device=device, dtype=torch.float32))
    delta_ty = torch.nn.Parameter(torch.tensor(
        0.0, device=device, dtype=torch.float32))
    delta_tz = torch.nn.Parameter(torch.tensor(
        0.0, device=device, dtype=torch.float32))

    iter_rows: list[dict[str, Any]] = []
    best_total = float("inf")
    best_iter = -1
    best_stage = ""
    best_weights: dict[str, float] | None = None
    best_state: dict[str, float] | None = None
    global_iter = 0

    def run_stage(
        stage_name: str,
        num_iters: int,
        lr: float,
        track_best: bool,
    ) -> None:
        nonlocal best_total
        nonlocal best_iter
        nonlocal best_stage
        nonlocal best_weights
        nonlocal best_state
        nonlocal global_iter
        if num_iters <= 0:
            return
        optimizer = torch.optim.Adam(
            [log_scale, delta_tx, delta_ty, delta_tz],
            lr=float(lr),
        )
        for iter_idx in range(1, num_iters + 1):
            global_iter += 1
            if stage_name == "stage1":
                weights = get_stage1_loss_weights(args)
            else:
                weights = get_stage2_loss_weights(
                    args, iter_idx - 1, num_iters - 1)
            optimizer.zero_grad(set_to_none=True)
            losses = compute_alignment_loss(
                human_points_visible_base=human_points_visible_base_t,
                obs_mask_pixels_norm=obs_mask_pixels_norm_t,
                human_mask_t=human_mask_t,
                scene_depth_t=scene_depth_t,
                contact_edges_t=contact_edges_t,
                intrinsics_t=intrinsics_t,
                width=width,
                height=height,
                log_scale=log_scale,
                delta_tx=delta_tx,
                delta_ty=delta_ty,
                delta_tz=delta_tz,
                tx_init_t=tx_init_t,
                ty_init_t=ty_init_t,
                tz_init_t=tz_init_t,
                front_margin_m=front_margin_m,
                weights=weights,
            )
            losses["total"].backward()
            optimizer.step()
            with torch.no_grad():
                log_scale.clamp_(math.log(0.65), math.log(1.6))
                delta_tx.clamp_(-2.0, 2.0)
                delta_ty.clamp_(-2.0, 2.0)
                delta_tz.clamp_(-3.0, 3.0)

            loss_val = float(losses["total"].detach().cpu().item())
            if track_best and loss_val < best_total:
                best_total = loss_val
                best_iter = global_iter
                best_stage = stage_name
                best_weights = dict(weights)
                best_state = {
                    "log_scale": float(log_scale.detach().cpu().item()),
                    "delta_tx": float(delta_tx.detach().cpu().item()),
                    "delta_ty": float(delta_ty.detach().cpu().item()),
                    "delta_tz": float(delta_tz.detach().cpu().item()),
                }

            row = build_loss_row(global_iter, stage_name, losses)
            iter_rows.append(row)
            if (
                global_iter == 1
                or global_iter % max(int(log_every), 1) == 0
                or iter_idx == num_iters
            ):
                for line in format_loss_log(
                    global_iter,
                    stage1_iters + stage2_iters,
                    stage_name,
                    losses,
                ):
                    print(line)

    track_best_in_stage1 = stage2_iters <= 0
    run_stage("stage1", stage1_iters, stage1_lr, track_best_in_stage1)
    run_stage("stage2", stage2_iters, stage2_lr, True)

    if best_state is None or best_weights is None:
        raise RuntimeError(
            "Alignment optimization did not produce a valid state.")

    with torch.no_grad():
        log_scale.copy_(
            torch.tensor(
                best_state["log_scale"],
                device=device,
                dtype=torch.float32,
            )
        )
        delta_tx.copy_(
            torch.tensor(best_state["delta_tx"],
                         device=device, dtype=torch.float32)
        )
        delta_ty.copy_(
            torch.tensor(best_state["delta_ty"],
                         device=device, dtype=torch.float32)
        )
        delta_tz.copy_(
            torch.tensor(best_state["delta_tz"],
                         device=device, dtype=torch.float32)
        )

    final_losses = compute_alignment_loss(
        human_points_visible_base=human_points_visible_base_t,
        obs_mask_pixels_norm=obs_mask_pixels_norm_t,
        human_mask_t=human_mask_t,
        scene_depth_t=scene_depth_t,
        contact_edges_t=contact_edges_t,
        intrinsics_t=intrinsics_t,
        width=width,
        height=height,
        log_scale=log_scale,
        delta_tx=delta_tx,
        delta_ty=delta_ty,
        delta_tz=delta_tz,
        tx_init_t=tx_init_t,
        ty_init_t=ty_init_t,
        tz_init_t=tz_init_t,
        front_margin_m=front_margin_m,
        weights=best_weights,
    )

    return {
        "scale": float(final_losses["scale"].detach().cpu().item()),
        "tx": float(final_losses["tx"].detach().cpu().item()),
        "ty": float(final_losses["ty"].detach().cpu().item()),
        "tz": float(final_losses["tz"].detach().cpu().item()),
        "log_scale": float(log_scale.detach().cpu().item()),
        "delta_tx": float(delta_tx.detach().cpu().item()),
        "delta_ty": float(delta_ty.detach().cpu().item()),
        "delta_tz": float(delta_tz.detach().cpu().item()),
        "tx_init": float(tx_init),
        "ty_init": float(ty_init),
        "tz_init": float(tz_init),
        "best_iter": int(best_iter),
        "best_stage": best_stage,
        "best_total_loss": float(final_losses["total"].detach().cpu().item()),
        "iter_rows": iter_rows,
        "final_losses": final_losses,
        "final_weights": best_weights,
    }


def apply_similarity_transform(
    verts: np.ndarray,
    scale: float,
    tx: float,
    ty: float,
    tz: float,
) -> np.ndarray:
    out = float(scale) * verts.astype(np.float32).copy()
    out[:, 0] += float(tx)
    out[:, 1] += float(ty)
    out[:, 2] += float(tz)
    return out.astype(np.float32)


def build_similarity_matrix_4x4(
    scale: float,
    tx: float,
    ty: float,
    tz: float,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 0] = float(scale)
    matrix[1, 1] = float(scale)
    matrix[2, 2] = float(scale)
    matrix[0, 3] = float(tx)
    matrix[1, 3] = float(ty)
    matrix[2, 3] = float(tz)
    return matrix


def compute_visible_behind_fraction(
    verts: np.ndarray,
    faces: np.ndarray,
    sampled_points_base: np.ndarray,
    camera_ctx: IdentityCameraContext,
    scene_depth: np.ndarray,
    device: torch.device,
    visible_tol_m: float,
    front_margin_m: float,
) -> dict[str, float]:
    sampled_visible, visible_stats = build_visible_subset(
        sampled_points=sampled_points_base,
        mesh_verts=verts,
        mesh_faces=faces,
        camera_ctx=camera_ctx,
        device=device,
        visible_tol_m=visible_tol_m,
    )
    if sampled_visible.shape[0] == 0:
        return {
            "num_visible_points": 0,
            "num_behind_points": 0,
            "behind_fraction": 0.0,
            **visible_stats,
        }
    uv = project_points_np(sampled_visible, camera_ctx.intrinsics)
    z = sampled_visible[:, 2]
    ui = np.round(uv[:, 0]).astype(np.int64)
    vi = np.round(uv[:, 1]).astype(np.int64)
    valid = (
        (ui >= 0)
        & (ui < camera_ctx.width)
        & (vi >= 0)
        & (vi < camera_ctx.height)
        & (z > 1e-6)
        & (scene_depth[vi, ui] > 0.0)
    )
    behind = np.zeros(sampled_visible.shape[0], dtype=bool)
    if np.any(valid):
        idx = np.nonzero(valid)[0]
        behind[idx] = z[idx] > (
            scene_depth[vi[idx], ui[idx]] - float(front_margin_m))
    num_valid = int(np.count_nonzero(valid))
    num_behind = int(np.count_nonzero(behind))
    return {
        "num_visible_points": num_valid,
        "num_behind_points": num_behind,
        "behind_fraction": float(
            num_behind /
            num_valid) if num_valid > 0 else 0.0,
        **visible_stats,
    }


def main() -> None:
    args = parse_args()
    defaults = build_default_paths(args.video_name)
    generated_root = resolve_path(
        args.generated_root, defaults["generated_root"])
    selection_json_path = resolve_path(
        args.selection_json, defaults["selection_json"])
    input_pag_json_path = resolve_path(
        args.input_pag_json, defaults["input_pag_json"])
    segment_root = resolve_path(args.segment_root, defaults["segment_root"])
    human_motion_root = resolve_path(
        args.human_motion_root, defaults["human_motion_root"])
    pag_json_path = (
        Path(args.pag_json).resolve()
        if args.pag_json is not None
        else find_pag_json(defaults["pag_root"])
    )
    smpl_seg_json_path = resolve_path(
        args.smpl_seg_json, defaults["smpl_seg_json"])
    output_root = ensure_dir(resolve_path(
        args.output_root, defaults["output_root"]))
    scene_root = ensure_dir(output_root / "scene")
    scene_depth_dir = ensure_dir(scene_root / "depth")
    scene_target_dir = ensure_dir(scene_root / "target")
    meshes_root = ensure_dir(output_root / "meshes")
    debug_root = ensure_dir(output_root / "debug")
    summary_json_path = output_root / "alignment_summary.json"
    transforms_json_path = meshes_root / "transforms.json"
    scannet_root = resolve_scannet_root(SCRIPT_DIR, args.scannet_root)
    device = parse_device(args.device)

    (
        camera_payload,
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
    body_segments, contact_segments = load_smpl_segments(smpl_seg_json_path)
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
        scene_verts_camera, scene_faces_in_view)
    if scene_faces_render.shape[0] == 0:
        raise RuntimeError(
            "No scene faces remained after view-frustum filtering.")
    print(
        "Scene face culling:",
        {
            "total_faces": int(scene_faces.shape[0]),
            "faces_in_view": int(scene_faces_in_view.shape[0]),
            "render_verts": int(scene_verts_camera_render.shape[0]),
            "render_faces": int(scene_faces_render.shape[0]),
        },
    )

    scene_depth, scene_mask, _ = rasterize_depth_and_mask(
        scene_verts_camera_render,
        scene_faces_render,
        camera_ctx=camera_ctx,
        device=device,
    )
    scene_depth_path = scene_depth_dir / "scene_depth.npy"
    scene_depth_vis_path = scene_depth_dir / "scene_depth_vis.png"
    np.save(scene_depth_path, scene_depth.astype(np.float32))
    save_depth_visualization(scene_depth_vis_path, scene_depth)

    segments_payload = load_json(scene_paths["segments_path"])
    anno_payload = load_json(scene_paths["segments_anno_path"])
    seg_indices = np.asarray(segments_payload["segIndices"], dtype=np.int64)
    candidate_faces, face_instance_ids, instance_meta = (
        build_candidate_instances(
            mesh_faces=scene_faces,
            seg_indices=seg_indices,
            seg_groups=anno_payload["segGroups"],
        )
    )

    target_instance_id = int(
        selection_payload["target_selection"]["instance_id"])
    target_meta = instance_meta.get(target_instance_id)
    if target_meta is None:
        raise KeyError(
            "Target instance_id "
            f"{target_instance_id} is not present in the scene annotations."
        )
    target_faces = candidate_faces[face_instance_ids == target_instance_id]
    if target_faces.shape[0] == 0:
        raise RuntimeError(
            f"No faces found for target instance {target_instance_id}.")
    target_verts_camera, target_faces_compact = compact_mesh(
        scene_verts_camera, target_faces)

    target_depth, target_mask_rendered, _ = rasterize_depth_and_mask(
        target_verts_camera,
        target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
    )
    target_surface_samples = sample_mesh_surface_points(
        verts=target_verts_camera,
        faces=target_faces_compact,
        num_samples=int(args.target_surface_samples),
        seed=int(args.seed),
    )
    target_points_visible, target_visible_stats = build_visible_subset(
        sampled_points=target_surface_samples,
        mesh_verts=target_verts_camera,
        mesh_faces=target_faces_compact,
        camera_ctx=camera_ctx,
        device=device,
        visible_tol_m=float(args.visible_tol_m),
    )
    target_surface_samples_path = (
        scene_target_dir / "target_surface_samples.npy"
    )
    target_visible_samples_path = (
        scene_target_dir / "target_surface_visible_samples.npy"
    )
    target_visible_ply_path = (
        scene_target_dir / "target_surface_visible_samples.ply"
    )
    np.save(
        target_surface_samples_path,
        target_surface_samples.astype(np.float32),
    )
    np.save(
        target_visible_samples_path,
        target_points_visible.astype(np.float32),
    )
    write_point_cloud_ply(
        target_visible_ply_path,
        target_points_visible.astype(np.float32),
    )

    stored_target_mask = build_observed_projected_target_mask(
        selection_json_path=selection_json_path,
        width=width,
        height=height,
    )
    target_mask_overlap = compute_binary_overlap(
        stored_target_mask, target_mask_rendered)
    target_overlay = render_target_overlay(
        background_bgr=read_bgr(first_frame_path),
        mask=target_mask_rendered,
        lines=[
            f"Target instance {target_instance_id}: {target_meta['label']}",
            f"Rendered vs stored IoU: {target_mask_overlap['iou']:.3f}",
            f"Rendered vs stored Dice: {target_mask_overlap['dice']:.3f}",
        ],
    )
    target_overlay_path = scene_target_dir / "target_projection_overlay.png"
    cv2.imwrite(str(target_overlay_path), target_overlay)
    target_compare_overlay = render_mask_overlay(
        background_bgr=read_bgr(first_frame_path),
        observed_mask=stored_target_mask,
        rendered_mask=target_mask_rendered,
        title_lines=[
            f"Target instance {target_instance_id}: {target_meta['label']}",
            f"Target IoU: {target_mask_overlap['iou']:.3f}",
            f"Target Dice: {target_mask_overlap['dice']:.3f}",
        ],
    )
    target_compare_overlay_path = (
        scene_target_dir / "target_projection_comparison_overlay.png"
    )
    cv2.imwrite(
        str(target_compare_overlay_path),
        target_compare_overlay,
    )

    tracks = discover_human_tracks(
        segment_root=segment_root,
        human_motion_root=human_motion_root,
    )
    camera_to_world = np.linalg.inv(
        np.asarray(camera_payload["world_to_camera_4x4"], dtype=np.float32)
    ).astype(np.float32)
    transforms_out: list[dict[str, Any]] = []
    summary_tracks: list[dict[str, Any]] = []

    for track in tracks:
        print(f"\nProcessing human track: {track.name}")
        track_mesh_root = ensure_dir(meshes_root / track.name)
        camera_mesh_dir = ensure_dir(track_mesh_root / "camera")
        world_mesh_dir = ensure_dir(track_mesh_root / "world")
        track_diag_root = ensure_dir(debug_root / track.name)
        debug_csv_dir = ensure_dir(track_diag_root / "csv")
        plot_iter_dir = ensure_dir(track_diag_root / "plots" / "iter")
        overlay_dir = ensure_dir(track_diag_root / "overlays")
        depth_vis_dir = ensure_dir(track_diag_root / "depth")

        first_mask_path = track.mask_dir / "frame_0000.png"
        human_mask = load_mask(first_mask_path)
        if human_mask.shape != (height, width):
            raise ValueError(
                f"Human mask shape mismatch for {track.name}: "
                f"got {human_mask.shape[::-1]}, expected {(width, height)}"
            )

        human_frame_paths = list_human_mesh_frames(track.source_camera_mesh_dir)
        first_human_verts, human_faces = load_mesh(human_frame_paths[0])
        human_surface_samples = sample_mesh_surface_points(
            verts=first_human_verts,
            faces=human_faces,
            num_samples=int(args.surface_samples),
            seed=int(args.seed),
        )
        contact_edges = build_contact_edges(
            pag_payload=pag_payload,
            track_name=track.name,
            target_object_name=selection_payload["target_selection"]["object"],
            body_segments=body_segments,
            contact_segments=contact_segments,
            human_verts=first_human_verts,
            target_points_visible=target_points_visible,
        )

        tz_init, init_info = compute_init_depth_translation(
            scene_depth=scene_depth,
            human_mask=human_mask,
            human_verts=first_human_verts,
            kernel_size=int(args.human_dilate_kernel),
            init_depth_offset_m=float(args.init_depth_offset_m),
            target_points_visible=target_points_visible,
        )
        init_human_verts = apply_similarity_transform(
            first_human_verts, 1.0, 0.0, 0.0, tz_init)
        init_human_samples = apply_similarity_transform(
            human_surface_samples, 1.0, 0.0, 0.0, tz_init)
        human_points_visible_init, human_visible_stats = build_visible_subset(
            sampled_points=init_human_samples,
            mesh_verts=init_human_verts,
            mesh_faces=human_faces,
            camera_ctx=camera_ctx,
            device=device,
            visible_tol_m=float(args.visible_tol_m),
        )
        human_points_visible_base = human_points_visible_init.copy()
        human_points_visible_base[:, 2] -= float(tz_init)

        obs_mask_pixels = sample_mask_pixels(
            mask=human_mask,
            max_points=int(args.mask_points),
            seed=int(args.seed),
        )

        optimization = optimize_alignment(
            human_points_visible_base_np=human_points_visible_base,
            obs_mask_pixels_np=obs_mask_pixels,
            human_mask=human_mask,
            scene_depth=scene_depth,
            contact_edges=contact_edges,
            intrinsics=intrinsics,
            width=width,
            height=height,
            tx_init=0.0,
            ty_init=0.0,
            tz_init=tz_init,
            stage1_iters=int(args.stage1_iters),
            stage2_iters=int(args.stage2_iters),
            stage1_lr=float(args.stage1_lr),
            stage2_lr=float(args.stage2_lr),
            front_margin_m=float(args.front_margin_m),
            log_every=int(args.log_every),
            args=args,
            device=device,
        )

        scale = float(optimization["scale"])
        tx = float(optimization["tx"])
        ty = float(optimization["ty"])
        tz = float(optimization["tz"])

        aligned_first_verts = apply_similarity_transform(
            first_human_verts, scale, tx, ty, tz)
        aligned_first_world = transform_camera_to_world(
            aligned_first_verts,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        write_ascii_ply(
            camera_mesh_dir /
            human_frame_paths[0].name,
            aligned_first_verts,
            human_faces)
        write_ascii_ply(
            world_mesh_dir /
            human_frame_paths[0].name,
            aligned_first_world,
            human_faces)

        rendered_human_depth, rendered_human_mask, _ = (
            rasterize_depth_and_mask(
                aligned_first_verts,
                human_faces,
                camera_ctx=camera_ctx,
                device=device,
            )
        )
        human_mask_overlap = compute_binary_overlap(
            human_mask, rendered_human_mask)
        contact_edge_metrics = compute_contact_edge_metrics(
            contact_edges=contact_edges,
            scale=scale,
            tx=tx,
            ty=ty,
            tz=tz,
            device=device,
        )

        first_frame_overlay = render_mask_overlay(
            background_bgr=read_bgr(first_frame_path),
            observed_mask=human_mask,
            rendered_mask=rendered_human_mask,
            title_lines=[
                f"{track.name}: first-frame alignment",
                f"Human IoU: {human_mask_overlap['iou']:.3f}",
                f"Human Dice: {human_mask_overlap['dice']:.3f}",
            ],
        )
        frame_overlay_path = overlay_dir / "frame_0000_mask_overlay.png"
        frame_depth_vis_path = depth_vis_dir / "frame_0000_depth_vis.png"
        cv2.imwrite(str(frame_overlay_path), first_frame_overlay)
        save_depth_visualization(frame_depth_vis_path, rendered_human_depth)
        iter_metrics_csv = debug_csv_dir / "iter_metrics.csv"
        final_loss_summary_csv = debug_csv_dir / "final_loss_summary.csv"
        final_loss_summary_row = build_final_loss_summary_row(
            optimization["best_iter"],
            optimization["best_stage"],
            optimization["final_losses"],
        )
        save_csv_rows(iter_metrics_csv, optimization["iter_rows"])
        save_csv_rows(final_loss_summary_csv, [final_loss_summary_row])
        save_loss_plot_tree(
            plot_iter_dir,
            optimization["iter_rows"],
            x_key="iter",
            total_key="total",
            term_keys=ALIGN_LOSS_TERM_KEYS,
            x_label="Iteration",
            title_prefix=f"{track.name} Iter",
        )

        behind_fraction_first = compute_visible_behind_fraction(
            verts=aligned_first_verts,
            faces=human_faces,
            sampled_points_base=apply_similarity_transform(
                human_surface_samples, scale, tx, ty, tz),
            camera_ctx=camera_ctx,
            scene_depth=scene_depth,
            device=device,
            visible_tol_m=float(args.visible_tol_m),
            front_margin_m=float(args.front_margin_m),
        )

        print("Exporting aligned human sequence...")
        for human_frame_path in human_frame_paths[1:]:
            frame_verts, _ = load_mesh(human_frame_path)
            aligned_frame_verts = apply_similarity_transform(
                frame_verts, scale, tx, ty, tz)
            aligned_world_verts = transform_camera_to_world(
                aligned_frame_verts,
                rotation_world_to_camera=rotation_world_to_camera,
                translation_world_to_camera=translation_world_to_camera,
            )
            write_ascii_ply(camera_mesh_dir / human_frame_path.name,
                            aligned_frame_verts, human_faces)
            write_ascii_ply(world_mesh_dir / human_frame_path.name,
                            aligned_world_verts, human_faces)

        source_to_aligned_camera = build_similarity_matrix_4x4(
            scale,
            tx,
            ty,
            tz,
        )
        transforms_out.append(
            {
                "name": track.name,
                "kind": "human",
                "source_mesh_dir": str(track.source_camera_mesh_dir),
                "aligned_camera_mesh_dir": str(camera_mesh_dir),
                "aligned_world_mesh_dir": str(world_mesh_dir),
                "source_to_aligned_camera_4x4": (
                    source_to_aligned_camera.tolist()
                ),
                "aligned_camera_to_world_4x4": camera_to_world.tolist(),
                "optimized_similarity": {
                    "scale": float(scale),
                    "tx_m": float(tx),
                    "ty_m": float(ty),
                    "tz_m": float(tz),
                },
            }
        )
        summary_tracks.append(
            {
                "name": track.name,
                "initialization": {
                    "scene_depth_region_source": init_info[
                        "scene_depth_region_source"
                    ],
                    "scene_depth_region_median_m": init_info[
                        "scene_depth_region_median_m"
                    ],
                    "tz_init_m": init_info["tz_init_m"],
                },
                "optimization": {
                    "best_iter": int(optimization["best_iter"]),
                    "best_stage": optimization["best_stage"],
                    "best_total_loss": float(optimization["best_total_loss"]),
                },
                "frame_0": {
                    "mask_overlap": human_mask_overlap,
                    "behind_fraction": behind_fraction_first,
                    "contact_edges": contact_edge_metrics,
                },
                "artifacts": {
                    "camera_mesh_dir": str(camera_mesh_dir),
                    "world_mesh_dir": str(world_mesh_dir),
                    "frame_overlay": str(frame_overlay_path),
                    "frame_depth_vis": str(frame_depth_vis_path),
                    "csv": {
                        "iter_metrics": str(iter_metrics_csv),
                        "final_loss_summary": str(final_loss_summary_csv),
                    },
                    "plots": {
                        "iter_dir": str(plot_iter_dir),
                    },
                },
            }
        )

    summary_out = {
        "video_name": args.video_name,
        "inputs": {
            "scene_mesh_path": str(scene_paths["mesh_path"]),
            "target_selection_json": str(selection_json_path),
            "input_pag_json": str(input_pag_json_path),
            "pag_json": str(pag_json_path),
            "resized_camera_json": str(generated_root / "resized_camera.json"),
            "human_motion_root": str(human_motion_root),
        },
        "camera": {
            "camera_name": camera_payload["camera_name"],
            "width": int(width),
            "height": int(height),
            "intrinsics_3x3": intrinsics.tolist(),
            "world_to_camera_4x4": camera_payload["world_to_camera_4x4"],
        },
        "optimization": {
            "transform": "X' = exp(alpha) * X + [tx, ty, tz]^T",
            "stage1": {
                "iters": int(args.stage1_iters),
                "lr": float(args.stage1_lr),
            },
            "stage2": {
                "iters": int(args.stage2_iters),
                "lr": float(args.stage2_lr),
            },
            "weights": {
                "mask": float(args.mask_weight),
                "front": float(args.front_weight),
                "scale_reg": float(args.scale_reg_weight),
                "translation_reg": float(args.translation_reg_weight),
                "nocontact": {
                    "start": float(args.nocontact_weight_start),
                    "end": float(args.nocontact_weight_end),
                },
            },
            "sampling": {
                "human_surface_samples": int(args.surface_samples),
                "target_surface_samples": int(args.target_surface_samples),
                "mask_points": int(args.mask_points),
            },
            "geometry": {
                "front_margin_m": float(args.front_margin_m),
                "init_depth_offset_m": float(args.init_depth_offset_m),
                "visible_tol_m": float(args.visible_tol_m),
                "human_dilate_kernel": int(args.human_dilate_kernel),
            },
        },
        "scene": {
            "depth": {
                "path": str(scene_depth_path),
                "vis_path": str(scene_depth_vis_path),
                "valid_pixels": int(np.count_nonzero(scene_mask)),
                "depth_min_m": (
                    float(scene_depth[scene_depth > 0.0].min())
                    if np.any(scene_depth > 0.0)
                    else 0.0
                ),
                "depth_max_m": (
                    float(scene_depth.max()) if scene_depth.size > 0 else 0.0
                ),
            },
            "target_object": {
                "instance_id": target_instance_id,
                "label": target_meta["label"],
                "projection_overlap": target_mask_overlap,
                "visible_surface_stats": target_visible_stats,
                "artifacts": {
                    "surface_samples_npy": str(target_surface_samples_path),
                    "visible_surface_samples_npy": str(
                        target_visible_samples_path
                    ),
                    "visible_surface_samples_ply": str(
                        target_visible_ply_path
                    ),
                    "projection_overlay": str(target_overlay_path),
                    "projection_comparison_overlay": str(
                        target_compare_overlay_path
                    ),
                },
            },
        },
        "tracks": summary_tracks,
        "outputs": {
            "meshes_dir": str(meshes_root),
            "debug_dir": str(debug_root),
            "transforms_json": str(transforms_json_path),
        },
    }

    save_json(summary_json_path, summary_out)
    save_json(transforms_json_path, {"transforms": transforms_out})
    print(f"\nSaved alignment outputs to: {output_root}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Transforms JSON: {transforms_json_path}")


if __name__ == "__main__":
    main()
