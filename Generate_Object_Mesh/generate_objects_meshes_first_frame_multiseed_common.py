"""Common implementation for first-frame multi-seed mesh generation variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import trimesh
from mesh_generation_utils import (
    compute_overlay_focal_scale,
    create_posed_mesh,
    discover_frames,
    discover_objects,
    estimate_camera_intrinsics,
    extract_pose_data,
    find_frame_image_path,
    generate_mesh,
    load_sam3d,
    sam3d_mesh_to_trimesh,
    save_pose_json,
    scale_camera_intrinsics,
)
from rendering_utils import (
    build_object_color_map,
    camera_k_from_info,
    ensure_quality_backend_available,
    render_multi_object_overlay_quality,
    render_single_object_overlay_quality,
)

def generate_mesh_with_seed(inference: Any, image: np.ndarray, mask: np.ndarray, seed: int) -> Dict[str, Any]:
    """Generate SAM3D output for a specific seed."""
    return generate_mesh(inference, image, mask, seed=seed)


def _save_single_overlay(
    image_rgb: np.ndarray,
    posed_mesh: trimesh.Trimesh,
    overlay_path: Path,
    color_bgr: Tuple[int, int, int],
    camera_k: np.ndarray,
) -> None:
    overlay = render_single_object_overlay_quality(
        image_rgb=image_rgb,
        posed_mesh=posed_mesh,
        camera_k=camera_k,
        color_bgr=color_bgr,
        fill_alpha=0.35,
        contour_thickness=2,
    )
    cv2.imwrite(str(overlay_path), overlay)


def _write_camera_intrinsics_json(path: Path, camera_info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(camera_info, f, indent=2)


def run_first_frame_multiseed(
    mode: str,
    input_dir: Path,
    sam3d: Any,
    mesh_output_root: Path,
    num_seeds: int,
    seed_start: int,
    seed_stride: int,
    focal_length_mm: Optional[float],
    f_scale: float,
) -> None:
    """Run first-frame multi-seed generation in one of two modes.

    mode='shared_canonical':
        Canonical mesh is generated from first seed only and reused for all seed poses.
    mode='per_seed_canonical':
        Canonical mesh is generated independently for each seed.
    """
    if mode not in {"shared_canonical", "per_seed_canonical"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if num_seeds <= 0:
        raise ValueError(f"--num_seeds must be > 0, got {num_seeds}")
    if seed_stride <= 0:
        raise ValueError(f"--seed_stride must be > 0, got {seed_stride}")
    ensure_quality_backend_available()

    objects = discover_objects(input_dir)
    frame_stems = discover_frames(input_dir)
    first_frame_stem = frame_stems[0]
    seeds = [int(seed_start + i * seed_stride) for i in range(num_seeds)]
    first_seed = seeds[0]

    video_name = input_dir.name
    output_root = (mesh_output_root / video_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    frames_dir = input_dir / "_frames"
    first_frame_path = find_frame_image_path(frames_dir, first_frame_stem)
    if first_frame_path is None:
        raise FileNotFoundError(f"Frame image not found for first frame: {first_frame_stem}")

    image_bgr = cv2.imread(str(first_frame_path))
    if image_bgr is None:
        raise RuntimeError(f"Could not load frame image: {first_frame_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"Video: {video_name}")
    print(f"Frame: {first_frame_stem}")
    print(f"Objects: {[name for name, _ in objects]}")
    print(f"Mode: {mode}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Output: {output_root}")

    camera_info = estimate_camera_intrinsics(sam3d, image_rgb)
    if focal_length_mm is None:
        camera_info = scale_camera_intrinsics(camera_info, f_scale)
        focal_overlay_mm = float(camera_info["blender_recommendation"]["lens_mm"])
        overlay_focal_scale = 1.0
        print(
            "Focal mode: auto+f_scale "
            f"(f_scale={f_scale}, focal_for_projection={focal_overlay_mm:.3f}mm)"
        )
    else:
        focal_overlay_mm = float(focal_length_mm)
        overlay_focal_scale = compute_overlay_focal_scale(camera_info, focal_overlay_mm)
        print(
            "Focal mode: explicit --focal_length "
            f"(focal_for_projection={focal_overlay_mm:.3f}mm, focal_scale={overlay_focal_scale:.4f})"
        )

    camera_k_overlay = camera_k_from_info(camera_info, focal_scale=overlay_focal_scale)
    seed_output_dirs: Dict[int, Path] = {seed: output_root / f"seed_{seed:04d}" for seed in seeds}
    camera_json_paths: Dict[int, Path] = {}
    for seed, seed_dir in seed_output_dirs.items():
        seed_dir.mkdir(parents=True, exist_ok=True)
        cam_json = seed_dir / "camera_intrinsics.json"
        _write_camera_intrinsics_json(cam_json, camera_info)
        camera_json_paths[seed] = cam_json

    seed_overlay_meshes: Dict[int, List[trimesh.Trimesh]] = {seed: [] for seed in seeds}
    seed_overlay_colors: Dict[int, List[Tuple[int, int, int]]] = {seed: [] for seed in seeds}
    object_color_map = build_object_color_map([name for name, _ in objects])

    for obj_name, mask_dir in objects:
        print(f"\n{'=' * 60}")
        print(f"Object: {obj_name}")
        mask_path = mask_dir / f"{first_frame_stem}.png"
        if not mask_path.exists():
            print(f"  Warning: mask not found - {mask_path}; skipping object.")
            continue

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            print(f"  Warning: failed to read mask - {mask_path}; skipping object.")
            continue
        mask = (mask_gray > 127).astype(np.uint8)

        per_seed_results: Dict[int, Dict[str, Any]] = {}
        for seed in seeds:
            try:
                output = generate_mesh_with_seed(sam3d, image_rgb, mask, seed)
                mesh_data = output["mesh"]
                canonical_mesh_seed = sam3d_mesh_to_trimesh(mesh_data[0])
                pose_data = extract_pose_data(output)
                per_seed_results[seed] = {
                    "output": output,
                    "canonical_mesh": canonical_mesh_seed,
                    "pose_data": pose_data,
                }
                print(
                    f"  Seed {seed}: ok | "
                    f"t={pose_data['translation'].tolist()} "
                    f"s={pose_data['scale'].tolist()}"
                )
            except Exception as exc:
                print(f"  Seed {seed}: failed ({exc})")

        if not per_seed_results:
            print("  No successful seeds for this object; skipping.")
            continue

        if mode == "shared_canonical":
            if first_seed not in per_seed_results:
                # If the nominal first seed failed, use first successful seed as canonical.
                canonical_seed = sorted(per_seed_results.keys())[0]
                print(
                    f"  Warning: first seed {first_seed} failed; using seed {canonical_seed} for canonical mesh."
                )
            else:
                canonical_seed = first_seed

            canonical_mesh_shared = per_seed_results[canonical_seed]["canonical_mesh"]

            for seed in seeds:
                if seed not in per_seed_results:
                    continue
                pose_data = per_seed_results[seed]["pose_data"]
                output = per_seed_results[seed]["output"]
                posed_mesh = create_posed_mesh(
                    canonical_mesh_shared,
                    pose_data["rotation_quat"],
                    pose_data["translation"],
                    pose_data["scale"],
                )

                seed_dir = seed_output_dirs[seed] / obj_name
                seed_dir.mkdir(parents=True, exist_ok=True)
                _write_camera_intrinsics_json(seed_dir / "camera_intrinsics.json", camera_info)
                if seed == canonical_seed:
                    # Shared-canonical mode stores canonical mesh once, under the canonical seed only.
                    canonical_mesh_shared.export(str(seed_dir / "mesh.ply"))
                save_pose_json(
                    output=output,
                    pose_data=pose_data,
                    output_path=seed_dir / "pose.json",
                    focal_length_mm=focal_overlay_mm,
                    camera_intrinsics_json=camera_json_paths[seed],
                    extra_fields={"seed": int(seed)},
                )
                posed_mesh.export(str(seed_dir / "mesh_posed.ply"))
                _save_single_overlay(
                    image_rgb=image_rgb,
                    posed_mesh=posed_mesh,
                    overlay_path=seed_dir / "mesh_posed_overlay.png",
                    color_bgr=object_color_map[obj_name],
                    camera_k=camera_k_overlay,
                )
                seed_overlay_meshes[seed].append(posed_mesh)
                seed_overlay_colors[seed].append(object_color_map[obj_name])

        else:
            for seed in seeds:
                if seed not in per_seed_results:
                    continue
                output = per_seed_results[seed]["output"]
                canonical_mesh_seed = per_seed_results[seed]["canonical_mesh"]
                pose_data = per_seed_results[seed]["pose_data"]
                posed_mesh = create_posed_mesh(
                    canonical_mesh_seed,
                    pose_data["rotation_quat"],
                    pose_data["translation"],
                    pose_data["scale"],
                )

                seed_dir = seed_output_dirs[seed] / obj_name
                seed_dir.mkdir(parents=True, exist_ok=True)
                _write_camera_intrinsics_json(seed_dir / "camera_intrinsics.json", camera_info)
                canonical_mesh_seed.export(str(seed_dir / "mesh.ply"))
                save_pose_json(
                    output=output,
                    pose_data=pose_data,
                    output_path=seed_dir / "pose.json",
                    focal_length_mm=focal_overlay_mm,
                    camera_intrinsics_json=camera_json_paths[seed],
                    extra_fields={"seed": int(seed)},
                )
                posed_mesh.export(str(seed_dir / "mesh_posed.ply"))
                _save_single_overlay(
                    image_rgb=image_rgb,
                    posed_mesh=posed_mesh,
                    overlay_path=seed_dir / "mesh_posed_overlay.png",
                    color_bgr=object_color_map[obj_name],
                    camera_k=camera_k_overlay,
                )
                seed_overlay_meshes[seed].append(posed_mesh)
                seed_overlay_colors[seed].append(object_color_map[obj_name])

    for seed in seeds:
        posed_meshes = seed_overlay_meshes.get(seed, [])
        if not posed_meshes:
            continue

        overlay = render_multi_object_overlay_quality(
            image_rgb=image_rgb,
            posed_meshes=posed_meshes,
            camera_k=camera_k_overlay,
            colors_bgr=seed_overlay_colors[seed],
            fill_alpha=0.35,
            contour_thickness=2,
        )
        cv2.imwrite(str(seed_output_dirs[seed] / "all_objects_overlay.png"), overlay)

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path.resolve()


def load_sam3d_pipeline() -> Any:
    print("Loading SAM 3D Objects...")
    sam3d = load_sam3d()
    print("SAM 3D Objects loaded successfully\n")
    return sam3d
