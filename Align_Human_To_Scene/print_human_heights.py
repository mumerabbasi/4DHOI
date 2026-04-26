#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import smplx
import torch
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parent.parent


@dataclass
class HumanTrack:
    name: str
    result_dir: Path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def build_default_paths(video_name: str) -> dict[str, Path]:
    return {
        "generated_root": PROJECT_DIR / "Generate_Video" / "output" / video_name,
        "segment_root": PROJECT_DIR / "Segment_Video" / "output" / video_name,
        "human_motion_root": (
            PROJECT_DIR / "Estimate_Human_Motion" / "output" / video_name / "humans"
        ),
        "output_root": SCRIPT_DIR / "output" / video_name,
        "smpl_folder": REPO_ROOT / "GVHMR" / "inputs" / "checkpoints" / "body_models",
    }


def load_camera_transform(camera_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = load_json(camera_path)
    world_to_camera = np.asarray(payload["world_to_camera_4x4"], dtype=np.float32)
    if world_to_camera.shape != (4, 4):
        raise ValueError(
            f"Expected a 4x4 world_to_camera_4x4 in {camera_path}, "
            f"got {world_to_camera.shape}."
        )
    rotation_world_to_camera = world_to_camera[:3, :3].astype(np.float32)
    translation_world_to_camera = world_to_camera[:3, 3].astype(np.float32)
    return rotation_world_to_camera, translation_world_to_camera


def transform_camera_to_world(
    points_camera: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (points_camera - translation_world_to_camera[None]) @ rotation_world_to_camera


def world_z_height(vertices_world: np.ndarray) -> tuple[float, float, float]:
    z_min = float(vertices_world[:, 2].min())
    z_max = float(vertices_world[:, 2].max())
    return z_max - z_min, z_min, z_max


def canonical_y_height(vertices: np.ndarray) -> tuple[float, float, float]:
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    return y_max - y_min, y_min, y_max


def discover_human_tracks(segment_root: Path, human_motion_root: Path) -> list[HumanTrack]:
    humans_dir = segment_root / "humans"
    if not humans_dir.is_dir():
        raise FileNotFoundError(f"Human segmentation directory not found: {humans_dir}")

    tracks: list[HumanTrack] = []
    for mask_track_dir in sorted(humans_dir.iterdir()):
        if not mask_track_dir.is_dir():
            continue
        result_dir = human_motion_root / mask_track_dir.name
        hmr_path = result_dir / "hmr4d_results.pt"
        if hmr_path.is_file():
            tracks.append(HumanTrack(name=mask_track_dir.name, result_dir=result_dir))

    if not tracks:
        raise FileNotFoundError(
            "No matching human tracks with hmr4d_results.pt found between "
            f"{humans_dir} and {human_motion_root}"
        )
    return tracks


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
    )
    layer = layer.to(device)
    layer.requires_grad_(False)
    return layer


def load_initial_params(
    result_dir: Path,
    param_key: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    result_path = result_dir / "hmr4d_results.pt"
    payload = torch.load(result_path, map_location=device, weights_only=True)
    if param_key not in payload:
        raise KeyError(
            f"Could not find '{param_key}' in {result_path}. "
            f"Available keys: {sorted(payload.keys())}"
        )
    params = payload[param_key]
    return {
        "transl": params["transl"][0].detach().clone().to(device).float(),
        "global_orient": params["global_orient"][0].detach().clone().to(device).float(),
        "body_pose": params["body_pose"][0].detach().clone().to(device).float(),
        "betas": params["betas"][0].detach().clone().to(device).float(),
        "scale": torch.ones((), device=device, dtype=torch.float32),
    }


def load_optimized_params(
    output_root: Path,
    track_name: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    params_path = output_root / "debug" / track_name / "params" / "optimized_frame_0000.pt"
    if not params_path.is_file():
        raise FileNotFoundError(f"Optimized params not found: {params_path}")
    payload = torch.load(params_path, map_location=device, weights_only=True)
    required_keys = {"transl", "global_orient", "body_pose", "betas", "scale"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise KeyError(f"Missing keys in {params_path}: {missing}")
    return {
        "transl": torch.as_tensor(payload["transl"], device=device).float(),
        "global_orient": torch.as_tensor(payload["global_orient"], device=device).float(),
        "body_pose": torch.as_tensor(payload["body_pose"], device=device).float(),
        "betas": torch.as_tensor(payload["betas"], device=device).float(),
        "scale": torch.as_tensor(payload["scale"], device=device).float(),
    }


def posed_vertices_camera(
    params: dict[str, torch.Tensor],
    smplx_layer: Any,
) -> np.ndarray:
    with torch.no_grad():
        out = smplx_layer(
            transl=params["transl"].view(1, 3),
            global_orient=params["global_orient"].view(1, 3),
            body_pose=params["body_pose"].view(1, -1),
            betas=params["betas"].view(1, -1),
        )
        verts = out.vertices[0]
        scale = params["scale"].reshape(())
        if float(scale.detach().cpu().item()) != 1.0:
            verts = params["transl"].view(1, 3) + scale * (
                verts - params["transl"].view(1, 3)
            )
    return verts.detach().cpu().numpy().astype(np.float32)


def canonical_vertices(
    params: dict[str, torch.Tensor],
    smplx_layer: Any,
) -> np.ndarray:
    betas = params["betas"]
    device = betas.device
    dtype = betas.dtype
    with torch.no_grad():
        out = smplx_layer(
            transl=torch.zeros((1, 3), device=device, dtype=dtype),
            global_orient=torch.zeros((1, 3), device=device, dtype=dtype),
            body_pose=torch.zeros((1, 63), device=device, dtype=dtype),
            betas=betas.view(1, -1),
        )
        verts = out.vertices[0] * params["scale"].reshape(())
    return verts.detach().cpu().numpy().astype(np.float32)


def load_optimized_vertices_world(output_root: Path, track_name: str) -> np.ndarray:
    mesh_path = output_root / track_name / "meshes" / "frame_0000_world.ply"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Optimized world mesh not found: {mesh_path}")
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected optimized mesh vertices shape: {vertices.shape}")
    return vertices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print world-coordinate SMPL-X human heights before and after first-frame "
            "alignment. This script does not save any files."
        )
    )
    parser.add_argument("video", nargs="?", default=None)
    parser.add_argument("--video_name", type=str, default=None)
    parser.add_argument("--generated_root", type=str, default=None)
    parser.add_argument("--segment_root", type=str, default=None)
    parser.add_argument("--human_motion_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--smpl_folder", type=str, default=None)
    parser.add_argument("--smpl_param_key", type=str, default="smpl_params_incam")
    parser.add_argument("--track_name", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_name = args.video_name or args.video or "video_01"
    defaults = build_default_paths(video_name)

    generated_root = resolve_path(args.generated_root, defaults["generated_root"])
    segment_root = resolve_path(args.segment_root, defaults["segment_root"])
    human_motion_root = resolve_path(
        args.human_motion_root,
        defaults["human_motion_root"],
    )
    output_root = resolve_path(args.output_root, defaults["output_root"])
    smpl_folder = resolve_path(args.smpl_folder, defaults["smpl_folder"])

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    rotation_world_to_camera, translation_world_to_camera = load_camera_transform(
        generated_root / "resized_camera.json"
    )
    tracks = discover_human_tracks(segment_root, human_motion_root)
    if args.track_name is not None:
        tracks = [track for track in tracks if track.name == args.track_name]
        if not tracks:
            raise FileNotFoundError(f"Track not found: {args.track_name}")

    smplx_layer = build_smplx_layer(smpl_folder, device)

    print(f"video: {video_name}")
    print(f"output_root: {output_root}")
    for track in tracks:
        init_params = load_initial_params(
            result_dir=track.result_dir,
            param_key=args.smpl_param_key,
            device=device,
        )
        opt_params = load_optimized_params(
            output_root=output_root,
            track_name=track.name,
            device=device,
        )

        init_canonical_verts = canonical_vertices(init_params, smplx_layer)
        opt_canonical_verts = canonical_vertices(opt_params, smplx_layer)
        init_actual_height, init_body_y_min, init_body_y_max = canonical_y_height(
            init_canonical_verts
        )
        opt_actual_height, opt_body_y_min, opt_body_y_max = canonical_y_height(
            opt_canonical_verts
        )

        init_verts_camera = posed_vertices_camera(init_params, smplx_layer)
        init_verts_world = transform_camera_to_world(
            init_verts_camera,
            rotation_world_to_camera=rotation_world_to_camera,
            translation_world_to_camera=translation_world_to_camera,
        )
        opt_verts_world = load_optimized_vertices_world(output_root, track.name)

        init_height, init_z_min, init_z_max = world_z_height(init_verts_world)
        opt_height, opt_z_min, opt_z_max = world_z_height(opt_verts_world)

        print(f"\ntrack: {track.name}")
        print(
            f"  initial_actual_height_m:   {init_actual_height:.4f} "
            f"(canonical_y_min={init_body_y_min:.4f}, "
            f"canonical_y_max={init_body_y_max:.4f}, scale=1.0000)"
        )
        print(
            f"  optimized_actual_height_m: {opt_actual_height:.4f} "
            f"(canonical_y_min={opt_body_y_min:.4f}, "
            f"canonical_y_max={opt_body_y_max:.4f}, "
            f"scale={float(opt_params['scale'].detach().cpu().item()):.4f})"
        )
        print(f"  actual_height_delta_m:     {opt_actual_height - init_actual_height:+.4f}")
        print(
            f"  initial_posed_world_z_m:   {init_height:.4f} "
            f"(z_min={init_z_min:.4f}, z_max={init_z_max:.4f})"
        )
        print(
            f"  optimized_posed_world_z_m: {opt_height:.4f} "
            f"(z_min={opt_z_min:.4f}, z_max={opt_z_max:.4f})"
        )


if __name__ == "__main__":
    main()
