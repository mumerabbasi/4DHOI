from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"
DEFAULT_SDF_DIM = 128
DEFAULT_SDF_PADDING_M = 0.5

if str(GENZI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZI_ROOT))


def configure_headless_rendering(opengl_platform: str | None) -> None:
    if opengl_platform is None:
        return
    platform = str(opengl_platform).strip()
    if not platform:
        return
    os.environ.setdefault("PYOPENGL_PLATFORM", platform)


def log(message: str) -> None:
    print(message, flush=True)


def load_singleview_helpers() -> ModuleType:
    script_path = MODULE_DIR / "01_run_singleview.py"
    spec = importlib.util.spec_from_file_location("dhsi_genzi_singleview", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def discover_interactions(output_mode: str) -> list[str]:
    del output_mode
    generated_root = PROJECT_DIR / "02_Generate_Human_Frame" / "output"
    names = sorted(path.name for path in generated_root.glob("interaction_*") if path.is_dir())
    if not names:
        raise RuntimeError(f"No interaction directories found under {generated_root}")
    return names


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


def build_external_inpaint_layout(
    interaction_name: str,
    output_root: Path,
    num_views: int,
    inpaints_per_view: int,
) -> dict[str, Any]:
    external_root = output_root / "external_inpaints"
    expected_dirs = []
    for inpaint_id in range(int(inpaints_per_view)):
        rel_dir = Path(interaction_name) / interaction_name / f"stage000_inpaint{inpaint_id:03d}"
        dst_dir = external_root / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        expected_dirs.append(
            {
                "inpaint_id": inpaint_id,
                "directory": dst_dir,
                "expected_files": [str(dst_dir / f"view{view_id:03d}.png") for view_id in range(num_views)],
            }
        )
    return {
        "external_inpaint_root": external_root,
        "stage000_expected_dirs": expected_dirs,
        "note": (
            "GenZI's external inpaint_dir expects paths shaped like "
            "<inpaint_dir>/<scene_name>/<prompt_id>/stage000_inpaintNNN/viewXXX.png."
        ),
    }


def maybe_save_manifest(path: Path | None, payload: dict[str, Any]) -> None:
    if path is not None:
        save_json(path, payload)


HAND_CONTACT_MASKS = ("left_hand", "right_hand")


def derive_hand_contact_look_at(
    contact_masks_dir: Path,
    contact_camera: Any,
    scene_vertices_world: Any,
    helpers: ModuleType,
    hand_contact_mask: str,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    import cv2

    vertices = np.asarray(scene_vertices_world, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("scene_vertices_world must have shape (N, 3)")

    mask_names = list(HAND_CONTACT_MASKS)
    if hand_contact_mask != "auto":
        mask_names = [hand_contact_mask]

    vertices_camera = helpers.transform_world_to_camera(vertices, contact_camera)
    u, v = helpers.project_points(vertices_camera, contact_camera.intrinsics)
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    in_frame = (
        (vertices_camera[:, 2] > 1e-6)
        & (ui >= 0)
        & (ui < contact_camera.width)
        & (vi >= 0)
        & (vi < contact_camera.height)
    )

    candidates: list[dict[str, Any]] = []
    for mask_name in mask_names:
        mask_path = contact_masks_dir / f"{mask_name}.png"
        if not mask_path.exists():
            candidates.append(
                {
                    "mask_name": mask_name,
                    "mask_path": mask_path,
                    "hit_vertices": 0,
                    "status": "missing",
                }
            )
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            candidates.append(
                {
                    "mask_name": mask_name,
                    "mask_path": mask_path,
                    "hit_vertices": 0,
                    "status": "unreadable",
                }
            )
            continue
        if mask.shape != (contact_camera.height, contact_camera.width):
            candidates.append(
                {
                    "mask_name": mask_name,
                    "mask_path": mask_path,
                    "hit_vertices": 0,
                    "status": f"wrong_shape:{mask.shape}",
                }
            )
            continue

        hits = in_frame.copy()
        hits[hits] &= mask[vi[hits], ui[hits]] > 127
        hit_indices = np.flatnonzero(hits)
        candidates.append(
            {
                "mask_name": mask_name,
                "mask_path": mask_path,
                "hit_vertices": int(hit_indices.shape[0]),
                "hit_indices": hit_indices,
                "status": "ok",
            }
        )

    valid_candidates = [item for item in candidates if item.get("hit_vertices", 0) > 0]
    if not valid_candidates:
        report = [
            {
                key: value
                for key, value in item.items()
                if key not in {"hit_indices"}
            }
            for item in candidates
        ]
        raise RuntimeError(f"No scene vertices projected into hand contact masks: {report}")

    selected = max(valid_candidates, key=lambda item: item["hit_vertices"])
    hit_indices = selected["hit_indices"]
    hit_vertices = vertices[hit_indices]
    region_centroid = np.mean(hit_vertices, axis=0)
    nearest_hit = int(np.argmin(np.linalg.norm(hit_vertices - region_centroid[None], axis=1)))
    vertex_index = int(hit_indices[nearest_hit])
    look_at = vertices[vertex_index]
    metadata = {
        "source": "hand_contact_scene_vertex",
        "selected_mask": selected["mask_name"],
        "selected_mask_path": selected["mask_path"],
        "vertex_index": vertex_index,
        "region_centroid": region_centroid.astype(np.float32),
        "distance_from_region_centroid": float(
            np.linalg.norm(look_at - region_centroid)
        ),
        "hit_vertices": int(selected["hit_vertices"]),
        "candidates": [
            {
                key: value
                for key, value in item.items()
                if key not in {"hit_indices"}
            }
            for item in candidates
        ],
    }
    return look_at.astype(np.float32), metadata


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
    np.save(sdf_path, sdf)
    metadata = {
        "dim": dim,
        "min": bbox_min.astype(float).tolist(),
        "max": bbox_max.astype(float).tolist(),
        "mesh_path": "",
        "method": "selected_view_depth_tsdf",
        "trunc_m": trunc_m,
        "num_views": int(num_views),
        "depth_image_size": [int(height), int(width)],
        "unknown_value": trunc_m,
        "observed_voxels": int(observed.sum()),
        "total_voxels": int(observed.shape[0]),
        "observed_fraction": float(observed.mean()),
        "negative_voxels": int((sdf < 0.0).sum()),
        "positive_voxels": int((sdf >= 0.0).sum()),
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


def write_sdf_visualizations(
    sdf_meta_path: Path,
    output_dir: Path,
    max_points: int,
    surface_band_m: float | None,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    metadata = json.loads(sdf_meta_path.read_text(encoding="utf-8"))
    sdf_path = sdf_meta_path.with_name(sdf_meta_path.stem + "_sdf.npy")
    sdf = np.load(sdf_path).astype(np.float32)
    max_points = max(0, int(max_points))
    dim = int(metadata["dim"])
    bbox_min = np.asarray(metadata["min"], dtype=np.float32)
    bbox_max = np.asarray(metadata["max"], dtype=np.float32)
    trunc_m = float(metadata.get("trunc_m", np.nanmax(np.abs(sdf))))
    voxel_size = (bbox_max - bbox_min) / float(max(dim - 1, 1))
    band = (
        float(surface_band_m)
        if surface_band_m is not None
        else max(float(np.max(voxel_size)), trunc_m * 0.08)
    )

    flat_sdf = sdf.reshape(-1)
    rng = np.random.default_rng(int(seed))

    def sample_indices(mask: Any, limit: int) -> Any:
        indices = np.flatnonzero(mask)
        if indices.shape[0] > limit:
            indices = rng.choice(indices, size=limit, replace=False)
        return np.sort(indices)

    def indices_to_points(indices: Any) -> Any:
        ijk = np.column_stack(np.unravel_index(indices, sdf.shape)).astype(np.float32)
        return bbox_min[None] + ijk / float(max(dim - 1, 1)) * (bbox_max - bbox_min)[None]

    def constant_colors(count: int, rgb: tuple[int, int, int]) -> Any:
        colors = np.empty((count, 3), dtype=np.uint8)
        colors[:, :] = np.asarray(rgb, dtype=np.uint8)
        return colors

    categories = {
        "negative_occupied": {
            "mask": flat_sdf < -band,
            "color": (220, 40, 40),
            "path": output_dir / "sdf_negative_occupied.ply",
        },
        "zero_surface_band": {
            "mask": np.abs(flat_sdf) <= band,
            "color": (255, 220, 0),
            "path": output_dir / "sdf_zero_surface_band.ply",
        },
        "positive_free_observed": {
            "mask": (flat_sdf > band) & (flat_sdf < trunc_m - 1e-6),
            "color": (45, 125, 255),
            "path": output_dir / "sdf_positive_free_observed.ply",
        },
        "unknown_positive": {
            "mask": flat_sdf >= trunc_m - 1e-6,
            "color": (170, 170, 170),
            "path": output_dir / "sdf_unknown_positive_sample.ply",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "sdf_meta_path": sdf_meta_path,
        "sdf_npy_path": sdf_path,
        "surface_band_m": band,
        "max_points_per_category": int(max_points),
        "color_legend": {
            "negative_occupied": "red: behind observed surfaces, treated as collision/inside",
            "zero_surface_band": "yellow: near the inferred zero surface",
            "positive_free_observed": "blue: observed free space in front of surfaces",
            "unknown_positive": "gray: unobserved/unknown, kept positive",
        },
        "files": {},
    }
    combined_points = []
    combined_colors = []
    for name, spec in categories.items():
        full_count = int(np.count_nonzero(spec["mask"]))
        indices = sample_indices(spec["mask"], int(max_points))
        points = indices_to_points(indices)
        colors = constant_colors(points.shape[0], spec["color"])
        write_binary_ply_points(spec["path"], points, colors)
        combined_points.append(points)
        combined_colors.append(colors)
        manifest["files"][name] = {
            "path": spec["path"],
            "full_voxel_count": full_count,
            "written_points": int(points.shape[0]),
            "rgb": list(spec["color"]),
        }

    if combined_points:
        points = np.concatenate(combined_points, axis=0)
        colors = np.concatenate(combined_colors, axis=0)
        if points.shape[0] > int(max_points):
            indices = rng.choice(points.shape[0], size=int(max_points), replace=False)
            points = points[indices]
            colors = colors[indices]
        combined_path = output_dir / "sdf_colored_samples.ply"
        write_binary_ply_points(combined_path, points, colors)
        manifest["files"]["combined_colored_samples"] = {
            "path": combined_path,
            "written_points": int(points.shape[0]),
        }

    manifest_path = output_dir / "sdf_visualization_manifest.json"
    save_json(manifest_path, manifest)
    manifest["manifest_path"] = manifest_path
    return manifest


def render_interaction(
    interaction_name: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    helpers: ModuleType,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    from genzi.render import Renderer
    from genzi.scene import Scene

    output_root = Path(args.output_base).resolve() / interaction_name
    log(f"  output_root={output_root} (debug artifacts only)")

    paths = helpers.build_paths(interaction_name, output_base=Path(args.output_base).resolve())
    required = {
        "input_scene_json": paths.input_scene_json,
        "sig_json": paths.sig_json,
        "contact_masks_dir": paths.contact_masks_dir,
        "contact_canvas_path": paths.contact_canvas_path,
        "contact_spec": paths.contact_spec,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + "; ".join(missing))

    log("  loading 4DHSI metadata")
    input_payload = helpers.load_json(paths.input_scene_json)
    sig_payload = helpers.load_json(paths.sig_json)
    scene_context = input_payload["scene_context"]
    scene_paths = helpers.resolve_scene_paths(
        helpers.resolve_scannet_root(args.scannet_root),
        scene_context,
    )
    for name in ("transforms_path", "colmap_images_path", "mesh_path"):
        if not scene_paths[name].exists():
            raise FileNotFoundError(f"Missing scene {name}: {scene_paths[name]}")

    scene_vertices_world, _scene_faces = helpers.load_mesh(scene_paths["mesh_path"])
    camera = helpers.load_scannet_camera(scene_paths, scene_context)
    contact_camera = helpers.load_contact_camera(
        paths.contact_spec,
        paths.contact_canvas_path,
        camera,
    )
    log(
        "  deriving look-at point from hand contact mask(s): "
        f"{paths.contact_masks_dir}"
    )
    look_at, look_at_stats = derive_hand_contact_look_at(
        contact_masks_dir=paths.contact_masks_dir,
        contact_camera=contact_camera,
        scene_vertices_world=scene_vertices_world,
        helpers=helpers,
        hand_contact_mask=args.hand_contact_mask,
    )

    scene_id = str(scene_context["scene_id"])
    log(
        f"  interaction point={np.asarray(look_at).round(4).tolist()} "
        f"source={look_at_stats.get('source')} "
        f"mask={look_at_stats.get('selected_mask')}"
    )
    sdf_meta_path = output_root / "sdf" / f"{scene_id}.json"
    sdf_npy_path = output_root / "sdf" / f"{scene_id}_sdf.npy"
    sdf_metadata = {}
    if sdf_meta_path.exists() and sdf_npy_path.exists():
        try:
            sdf_metadata = json.loads(sdf_meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            sdf_metadata = {}

    sdf_dim = int(args.sdf_dim or cfg.get("data.sdf_dim", DEFAULT_SDF_DIM))
    sdf_padding_m = float(
        args.sdf_padding_m
        if args.sdf_padding_m is not None
        else cfg.get("data.sdf_padding_m", DEFAULT_SDF_PADDING_M)
    )
    sdf_needs_build = (
        bool(args.force_sdf)
        or not sdf_meta_path.exists()
        or not sdf_npy_path.exists()
        or sdf_metadata.get("method") != "selected_view_depth_tsdf"
    )
    bootstrap_tmpdir = None
    if sdf_meta_path.exists() and sdf_npy_path.exists():
        view_selection_sdf = sdf_meta_path
        view_selection_sdf_source = "existing_sdf"
        log(f"  using existing SDF for view selection: {sdf_meta_path}")
    else:
        bootstrap_tmpdir = tempfile.TemporaryDirectory(prefix="genzi_sdf_bootstrap_")
        view_selection_sdf = write_neutral_sdf(
            scene_id=scene_id,
            scene_vertices_world=scene_vertices_world,
            sdf_dir=Path(bootstrap_tmpdir.name),
            dim=int(args.bootstrap_sdf_dim),
            padding_m=sdf_padding_m,
        )
        view_selection_sdf_source = "temporary_neutral_bootstrap"
        log("  using temporary neutral SDF for first-time view selection")

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
    debug_views_dir = None
    debug_view_paths = []
    renderer.set_cameras(
        eyes=viewpoints,
        at=selected_look_at,
        up=up_dir,
        fov=float(cfg["data.fov"]),
    )
    used_views = list(range(renderer.num_cameras()))
    scene_images = None
    scene_depths = None
    if sdf_needs_build or args.save_debug_renders:
        log("  rendering selected scene views/depths with pyrender")
        scene_images, _scene_masks, scene_depths = renderer.render(
            tri_meshes=[scene3d.get_trimesh()],
            camera_ids=used_views,
            **render_args,
        )

    if sdf_needs_build:
        assert scene_depths is not None
        log(
            "  building selected-view depth TSDF "
            f"dim={sdf_dim} trunc={float(args.depth_sdf_trunc_m):.3f}m"
        )
        sdf_meta_path, sdf_metadata = build_depth_tsdf_sdf(
            scene_id=scene_id,
            scene_vertices_world=scene_vertices_world,
            renderer=renderer,
            scene_depths=scene_depths,
            sdf_dir=output_root / "sdf",
            dim=sdf_dim,
            padding_m=sdf_padding_m,
            trunc_m=float(args.depth_sdf_trunc_m),
        )
        save_json(
            output_root / "sdf" / "sdf_summary.json",
            {
                "interaction_name": interaction_name,
                "scene_id": scene_id,
                "mesh_path": scene_paths["mesh_path"],
                "sdf_meta": sdf_meta_path,
                "view_selection_sdf": view_selection_sdf,
                "view_selection_sdf_source": view_selection_sdf_source,
                "sdf_stats": sdf_metadata,
                "look_at": selected_look_at,
                "look_at_stats": look_at_stats,
                "viewpoints": viewpoints,
                "up_dir": up_dir,
                "fov": float(cfg["data.fov"]),
                "view_distance": view_distance_m,
                "selected_views": int(viewpoints.shape[0]),
            },
        )
        log(
            "  wrote depth-based SDF "
            f"observed={sdf_metadata.get('observed_fraction', 0.0):.3f} "
            f"path={sdf_meta_path}"
        )
    else:
        log(f"  reusing depth-based SDF: {sdf_meta_path}")

    sdf_visualization = None
    if args.save_sdf_visualization:
        log("  writing SDF visualization PLYs")
        sdf_visualization = write_sdf_visualizations(
            sdf_meta_path=sdf_meta_path,
            output_dir=output_root / "sdf_visualization",
            max_points=int(args.sdf_vis_max_points),
            surface_band_m=args.sdf_vis_surface_band_m,
            seed=int(args.seed),
        )

    if args.save_debug_renders:
        assert scene_images is not None
        debug_views_dir = output_root / "views_stage000"
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
        "token_indices": [],
        "lookats": [selected_look_at.astype(float).tolist()],
        "viewpoints": [[viewpoints.astype(float).tolist() for _ in cfg["optim.steps"]]],
        "interactions": [interaction_label],
    }
    genzi_scene_root = (
        Path(args.genzi_scene_root).resolve()
        if args.genzi_scene_root
        else output_root
    )
    genzi_scene_config_path = genzi_scene_root / f"{interaction_name}_v1.yml"
    write_yaml(genzi_scene_config_path, scene_yaml_payload)
    log(f"  wrote GenZI scene config: {genzi_scene_config_path}")

    inpaint_layout = None
    if args.prepare_external_inpaints:
        inpaint_layout = build_external_inpaint_layout(
            interaction_name=interaction_name,
            output_root=output_root,
            num_views=int(viewpoints.shape[0]),
            inpaints_per_view=int(cfg["loss.inpaints_per_view"][0]),
        )
    manifest = {
        "interaction_name": interaction_name,
        "scene_id": scene_id,
        "artifact_mode": "genzi_scene_config",
        "genzi_scene_config": genzi_scene_config_path,
        "debug_views_dir": debug_views_dir,
        "debug_view_paths": debug_view_paths,
        "mesh_path": scene_paths["mesh_path"],
        "sdf_meta": sdf_meta_path,
        "sdf_stats": sdf_metadata,
        "sdf_visualization": sdf_visualization,
        "view_selection_sdf": view_selection_sdf,
        "view_selection_sdf_source": view_selection_sdf_source,
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
        "external_inpaint_layout": inpaint_layout,
    }
    manifest_path = output_root / "multiview_scene_summary.json" if args.write_manifest else None
    maybe_save_manifest(manifest_path, manifest)
    if bootstrap_tmpdir is not None:
        bootstrap_tmpdir.cleanup()
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render GenZI-style multiview scene inputs for 4DHSI. This uses "
            "GenZI's own Renderer and Scene.get_viewpoints, not module-06 cameras."
        )
    )
    parser.add_argument("--interaction_name", default="interaction_10")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--run_cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output_base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument(
        "--genzi_scene_root",
        default=None,
        help="Optional output directory for the GenZI scene YAML. Default is output_base/<interaction>.",
    )
    parser.add_argument("--scannet_root", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sdf_dim", type=int, default=None)
    parser.add_argument("--sdf_padding_m", type=float, default=None)
    parser.add_argument(
        "--depth_sdf_trunc_m",
        type=float,
        default=0.25,
        help="Truncation distance for the selected-view depth TSDF used by GenZI.",
    )
    parser.add_argument(
        "--bootstrap_sdf_dim",
        type=int,
        default=16,
        help="Small temporary neutral SDF grid used only if no real SDF exists before view selection.",
    )
    parser.add_argument("--force_sdf", action="store_true")
    parser.add_argument(
        "--save_sdf_visualization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write MeshLab-friendly PLY point clouds showing the generated SDF.",
    )
    parser.add_argument(
        "--sdf_vis_max_points",
        type=int,
        default=250000,
        help="Maximum sampled points per SDF visualization category.",
    )
    parser.add_argument(
        "--sdf_vis_surface_band_m",
        type=float,
        default=None,
        help="Optional absolute-value SDF band, in meters, used for the yellow near-surface PLY.",
    )
    parser.add_argument("--dir_light_intensity", type=float, default=5.0)
    parser.add_argument("--pt_light_intensity", type=float, default=5.0)
    parser.add_argument(
        "--num_viewpoints",
        type=int,
        default=None,
        help="Override GenZI candidate camera count. Default uses run_cfg data.num_viewpoints.",
    )
    parser.add_argument(
        "--max_views",
        type=int,
        default=None,
        help="Override GenZI selected view cap. Default uses run_cfg data.max_views.",
    )
    parser.add_argument(
        "--view_distance_m",
        type=float,
        default=None,
        help=(
            "Camera distance from the interaction point. Default uses run_cfg "
            "data.view_distances[0]."
        ),
    )
    parser.add_argument("--up_dir", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    parser.add_argument(
        "--opengl_platform",
        default="egl",
        help=(
            "Headless PyOpenGL backend for pyrender. Use 'egl' on CUDA machines; "
            "try 'osmesa' only if EGL is unavailable and OSMesa is installed."
        ),
    )
    parser.add_argument("--prompt_prefix", default="")
    parser.add_argument("--prompt_suffix", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--hand_contact_mask",
        choices=("auto", *HAND_CONTACT_MASKS),
        default="auto",
        help=(
            "Hand contact mask used to choose the GenZI interaction point. "
            "Default auto selects the available hand mask with the most projected scene vertices."
        ),
    )
    parser.add_argument(
        "--save_debug_renders",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also render selected scene views to output_base/<interaction>/views_stage000 for inspection.",
    )
    parser.add_argument(
        "--prepare_external_inpaints",
        action="store_true",
        help="Create GenZI-shaped external inpaint directories under output_base/<interaction>.",
    )
    parser.add_argument(
        "--write_manifest",
        action="store_true",
        help="Write a small summary JSON under output_base. Default only writes the GenZI scene YAML.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_headless_rendering(args.opengl_platform)
    cfg = load_cfg(Path(args.run_cfg).resolve())
    cfg["gpus"] = [int(str(args.device).split(":")[-1])] if str(args.device).startswith("cuda:") else cfg["gpus"]
    helpers = load_singleview_helpers()

    interaction_names = (
        discover_interactions("output")
        if args.all_interactions or args.interaction_name == "all"
        else [args.interaction_name]
    )

    summaries = []
    failures = []
    for interaction_name in interaction_names:
        started = time.time()
        try:
            print(f"\n[*] Rendering GenZI multiview scene inputs for {interaction_name}")
            summary = render_interaction(interaction_name, args, cfg, helpers)
            summary["elapsed_s"] = time.time() - started
            summaries.append(summary)
            print(
                f"  selected_views={summary['selected_views']} "
                f"scene_config={summary['genzi_scene_config']}"
            )
        except Exception as exc:
            failure = {
                "interaction_name": interaction_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            print(f"[!] Failed {interaction_name}: {failure['error']}")
            if not args.all_interactions and args.interaction_name != "all":
                raise

    output_base = Path(args.output_base).resolve()
    maybe_save_manifest(
        output_base / "genzi_multiview_scene_batch_summary.json" if args.write_manifest else None,
        {
            "summaries": summaries,
            "failures": failures,
            "run_cfg": args.run_cfg,
        },
    )
    print(
        "\n[*] Finished GenZI multiview scene rendering. "
        f"successes={len(summaries)} failures={len(failures)}"
    )


if __name__ == "__main__":
    main()
