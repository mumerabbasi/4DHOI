#!/usr/bin/env python3
"""Run PROX with a Module-03 image-crop TSDF and coverage-selected views.

This is an independent experimental variant of ``00_run_prox.py``.  It reuses
that wrapper's original-PROX integration, but the scene volume no longer depends
only on PROX's monocular 3D initialization:

1. Reconstruct the visible ScanNet++ surfaces inside Module 03's padded human
   image crop using the ground-truth source camera.
2. Union those surfaces with the initialized body and form a padded 3D volume.
3. Collapse near-duplicate registered DSLR poses before expensive depth tests.
4. Select the remaining cameras greedily by newly visible crop-surface area.
5. Render the full scene from selected cameras and fuse only the crop volume.

Module 03 contact masks, contact-part names, and inferred contact facts are not
used.  Only its target-scene image crop and adjusted crop intrinsics are used.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
BASE_SCRIPT = MODULE_DIR / "00_run_prox.py"
OUTPUT_ROOT = MODULE_DIR / "output_scene_crop"
MODULE03_OUTPUT = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"

ROI_PADDING_M = 0.25
COVERAGE_RENDER_WIDTH = 320
COVERAGE_SURFACE_SAMPLES = 12_000
COVERAGE_DEPTH_TOLERANCE_M = 0.04
COVERAGE_MIN_VISIBLE_SAMPLES = 12
POSE_DEDUP_TRANSLATION_M = 0.08
POSE_DEDUP_ROTATION_DEG = 4.0
MAX_TSDF_VIEWS = 64
RANDOM_SEED = 24017


def load_base_wrapper() -> Any:
    spec = importlib.util.spec_from_file_location("prox_module09_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base PROX wrapper: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_wrapper()


def load_trimesh(path: Path) -> Any:
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError(f"Scene mesh is empty: {path}")
    return mesh


def load_module03_crop(interaction_name: str, source_intrinsics: Any) -> dict[str, Any]:
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
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    crop_intrinsics = np.asarray(
        payload["camera"]["intrinsics_3x3"], dtype=np.float32
    )
    crop_width, crop_height = Image.open(image_path).size
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
    render_camera: Any,
    crop_xyxy_source: list[float],
    source_camera: Any,
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
    points_world = BASE.camera_to_world(points_camera, render_camera)
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
    used_vertices, inverse = np.unique(faces[keep_faces].reshape(-1), return_inverse=True)
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
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
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


def camera_center(camera: Any) -> Any:
    return -camera.rotation_world_to_camera.T @ camera.translation_world_to_camera


def projected_crop_samples(camera: Any, samples: Any) -> tuple[Any, Any, Any, Any]:
    """Project sampled crop surfaces without doing an occlusion render."""
    import numpy as np

    points_camera = BASE.world_to_camera(samples, camera)
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
    cameras: list[Any],
    projected_counts: list[int],
    source_camera: Any,
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

    centers = np.stack([camera_center(camera) for camera in cameras]).astype(np.float32)
    rotations = np.stack(
        [np.asarray(camera.rotation_world_to_camera, dtype=np.float32) for camera in cameras]
    )
    source_index = next(
        (index for index, camera in enumerate(cameras) if camera.name == source_camera.name),
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
        # trace(R_a R_b^T) equals the elementwise dot product of both matrices.
        traces = np.einsum("nij,ij->n", rotations[retained_array], rotations[index])
        cos_angles = np.clip((traces - 1.0) * 0.5, -1.0, 1.0)
        rotation_angles = np.arccos(cos_angles)
        is_duplicate = np.any(
            (translations <= POSE_DEDUP_TRANSLATION_M)
            & (rotation_angles <= rotation_threshold_rad)
        )
        if not is_duplicate:
            retained.append(index)

    # Restore acquisition/COLMAP order for deterministic progress and manifests.
    retained.sort()
    metadata = {
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
    return retained, metadata


def select_cameras_by_surface_coverage(
    full_scene_mesh: Any,
    crop_mesh: Any,
    cameras: list[Any],
    source_camera: Any,
    roi_center: Any,
    debug_dir: Path,
) -> tuple[list[Any], dict[str, Any]]:
    import numpy as np
    import pyrender

    samples = sample_mesh_surface(crop_mesh, COVERAGE_SURFACE_SAMPLES, RANDOM_SEED)
    scaled = [BASE.scaled_camera(camera, COVERAGE_RENDER_WIDTH) for camera in cameras]
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
    BASE.save_json(debug_dir / "camera_pose_dedup.json", deduplication)
    BASE.log(
        "    pose dedup kept "
        f"{len(prepass_indices)}/{len(projection_candidate_indices)} projected camera(s) "
        f"(removed {deduplication['removed_camera_count']})"
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
    eligible_cameras: list[Any] = []
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
            node = scene.add(projection, pose=BASE.pyrender_camera_pose(camera))
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
            if prepass_number % 100 == 0 or prepass_number == len(prepass_indices):
                BASE.log(
                    f"    coverage prepass {prepass_number}/{len(prepass_indices)} "
                    f"eligible={len(eligible_cameras)}"
                )
    finally:
        renderer.delete()

    if not visibility_rows:
        raise RuntimeError("No registered camera visibly covers the Module 03 scene crop.")
    visibility = np.stack(visibility_rows, axis=0)
    source_index = next(
        (i for i, camera in enumerate(eligible_cameras) if camera.name == source_camera.name),
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
    BASE.save_json(debug_dir / "camera_coverage.json", manifest)
    return selected, manifest


def derive_crop_volume(
    interaction_name: str,
    full_scene_mesh: Any,
    source_camera: Any,
    source_intrinsics: Any,
    initial_vertices_world: Any,
    debug_dir: Path,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import numpy as np

    crop = load_module03_crop(interaction_name, source_intrinsics)
    rendered_cameras, depths = BASE.render_scene_depths(
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
    combined_min = np.minimum(crop_points.min(axis=0), initial_vertices_world.min(axis=0))
    combined_max = np.maximum(crop_points.max(axis=0), initial_vertices_world.max(axis=0))
    bbox_min = combined_min - ROI_PADDING_M
    bbox_max = combined_max + ROI_PADDING_M
    crop_mesh = crop_mesh_to_bounds(full_scene_mesh, bbox_min, bbox_max)
    roi_center = (bbox_min + bbox_max) * 0.5

    preview_count = min(100_000, crop_points.shape[0])
    if crop_points.shape[0] > preview_count:
        indices = np.linspace(0, crop_points.shape[0] - 1, preview_count).astype(np.int64)
        preview = crop_points[indices]
    else:
        preview = crop_points
    BASE.write_binary_ply_points(
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
    BASE.save_json(debug_dir / "scene_crop_volume.json", metadata)
    return bbox_min, bbox_max, crop_mesh, metadata


def run_interaction(name: str, torch: Any, config: dict[str, Any]) -> None:
    import numpy as np

    started = time.time()
    inputs = BASE.resolve_interaction_inputs(name)
    final_root = OUTPUT_ROOT / name
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True)
    BASE.log(f"\n[*] Running crop-volume PROX for {name}")
    data = BASE.load_interaction_data(inputs)
    openpose = BASE.run_openpose(inputs.human_image, final_root / "openpose")
    device = torch.device("cuda:0")
    body_model = BASE.create_body_model(device, torch, config)
    vposer, pose_embedding = BASE.load_vposer(device)
    initialization = BASE.initialize_original_prox(
        body_model,
        vposer,
        pose_embedding,
        openpose["keypoints"],
        data["intrinsics"],
        config,
        torch,
        device,
    )
    initial_pose = BASE.decoded_body_pose(vposer, pose_embedding)
    initial_vertices_camera, _ = BASE.body_vertices_and_joints(
        body_model, initial_pose, torch
    )
    source_camera = data["source_camera"]
    initial_vertices_world = BASE.camera_to_world(initial_vertices_camera, source_camera)
    BASE.export_mesh(
        final_root / "initial_smplx_camera.ply",
        initial_vertices_camera,
        body_model.faces,
    )
    if openpose["overlay"] is not None:
        shutil.copy2(openpose["overlay"], final_root / "keypoints_overlay.png")
    else:
        BASE.save_keypoint_overlay(
            inputs.human_image,
            openpose["body25"],
            final_root / "keypoints_overlay.png",
        )

    BASE.log("  deriving Module 03 crop volume from source-visible scene surfaces")
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
    BASE.log(
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
    BASE.log(
        f"  selected {len(observing)} TSDF view(s), "
        f"coverage={coverage_metadata['covered_fraction']:.3f}"
    )
    render_cameras, depths = BASE.render_scene_depths(
        full_scene_mesh,
        observing,
        final_root / "debug" / "tsdf_views",
    )
    BASE.log(
        f"  building crop-volume TSDF dim={BASE.SDF_GRID_DIM} "
        f"trunc={BASE.SDF_TRUNCATION_M:.3f}m"
    )
    sdf_meta_path, sdf_metadata = BASE.build_visibility_tsdf(
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
            "method": BASE.TSDF_METHOD,
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
    BASE.save_json(sdf_meta_path, sdf_metadata)
    del depths
    BASE.log("  exporting complete TSDF debug artifacts")
    tsdf_debug = BASE.write_tsdf_debug(
        sdf_meta_path,
        final_root / "debug" / "tsdf",
        roi_center,
    )

    scene_dir = final_root / "scene"
    cam2world_dir = final_root / "cam2world"
    scene_dir.mkdir()
    cam2world_dir.mkdir()
    scene_mesh.export(scene_dir / f"{inputs.scene_id}.ply")
    BASE.save_json(
        cam2world_dir / f"{inputs.scene_id}.json",
        BASE.camera_to_world_transform(source_camera),
    )
    final_camera_path = final_root / "final_smplx_camera.ply"
    result_path = final_root / "result.pkl"
    BASE.log("  calling upstream PROX fit_single_frame()")
    BASE.run_upstream_prox(
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
    result, final_vertices_camera, final_faces = BASE.load_selected_prox_mesh(
        result_path, body_model, torch
    )
    BASE.export_mesh(final_camera_path, final_vertices_camera, final_faces)
    final_vertices_world = BASE.camera_to_world(final_vertices_camera, source_camera)
    BASE.export_mesh(
        final_root / "final_smplx_world.ply",
        final_vertices_world,
        final_faces,
    )
    BASE.render_body_overlay(
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
        "camera_to_world": BASE.camera_to_world_transform(source_camera),
        "initialization": "original PROX zero-VPoser initialization",
        "contact_body_parts": BASE.CONTACT_BODY_PARTS,
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
    BASE.save_json(final_root / "metadata.json", metadata)
    BASE.log(f"  wrote {final_root}")
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


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if str(BASE.PROX_ROOT / "prox") not in sys.path:
        sys.path.insert(0, str(BASE.PROX_ROOT / "prox"))
    import fit_single_frame as _upstream_fit_single_frame  # noqa: F401

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PROX requires CUDA, but CUDA is unavailable.")
    names = BASE.resolve_interactions(args)
    config = BASE.load_original_prox_config()
    BASE.validate_prox_assets(config)
    BASE.validate_openpose()
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    BASE.TSDF_METHOD = (
        "module03_scene_crop_coverage_visibility_tsdf_384_depth2048_pose_dedup_v2"
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    BASE.log(f"[*] Selected {len(names)} interaction(s): " + ", ".join(names))
    for name in names:
        run_interaction(name, torch, config)
    BASE.log(f"\n[*] Crop-volume PROX finished for {len(names)} interaction(s).")


if __name__ == "__main__":
    main()
