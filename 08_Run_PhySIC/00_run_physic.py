#!/usr/bin/env python3
"""Run the ScanNet++ GT-scene PhySIC baseline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from physic_eval_utils import write_evaluation_artifacts
from scannet_gt_scene import build_scannet_gt_observation


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_DIR = PROJECT_DIR.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PhySIC against visible ScanNet++ GT geometry."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--all_interactions", action="store_true")
    parser.add_argument("--output_root", type=Path, default=SCRIPT_DIR / "output_scannet")
    parser.add_argument("--scannet_root", type=Path, default=REPO_DIR / "Scannet++" / "data")
    parser.add_argument("--physic_root", type=Path, default=REPO_DIR / "Phy-SIC")
    return parser.parse_args(argv)


def interaction_names(run_all: bool, requested: str) -> list[str]:
    if not run_all:
        return [requested]
    root = PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"
    names = [path.name for path in root.glob("interaction_*") if path.is_dir()]
    if not names:
        raise FileNotFoundError(f"No interactions found under {root}.")
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def interaction_inputs(interaction_name: str) -> tuple[Path, Path]:
    assets = (
        PROJECT_DIR
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
    )
    human_image = assets / "reference_inpainted_crop.png"
    scene_image = assets / "target_scene_crop.png"
    missing = [str(path) for path in (human_image, scene_image) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing interaction input(s): " + "; ".join(missing))
    return human_image, scene_image


def replace_path_prefix(value, old_prefix: str, new_prefix: str):
    if isinstance(value, dict):
        return {
            key: replace_path_prefix(item, old_prefix, new_prefix)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_path_prefix(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix) :]
    return value


def publish(staging_root: Path, final_root: Path) -> None:
    manifest_path = staging_root / "metadata" / "artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = replace_path_prefix(
        manifest,
        str(staging_root.resolve()),
        str(final_root.resolve()),
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    backup = final_root.with_name(f".{final_root.name}.backup-{uuid.uuid4().hex}")
    if final_root.exists():
        os.replace(final_root, backup)
    try:
        os.replace(staging_root, final_root)
    except Exception:
        if backup.exists() and not final_root.exists():
            os.replace(backup, final_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def gpu_metadata(torch) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_torch_device": "cuda:0",
        "name": properties.name,
        "memory_total_mb": int(properties.total_memory / 1024**2),
    }


def serialize_result(result, observation, torch, physic_root: Path, started: float):
    diagnostics = result.scannet_gt_diagnostics
    return {
        "depth": result.depth.cpu().numpy(),
        "K": result.K.cpu().numpy(),
        "pts3d": result.pts3d.cpu().numpy(),
        "inlier_mask": result.inlier_mask.cpu().numpy(),
        "scale": result.scale.detach().cpu().numpy(),
        "normals": result.normals.cpu().numpy(),
        "plane_points": result.plane_points.cpu().numpy(),
        "plane_normal": result.plane_normal.cpu().numpy(),
        "body_params": {
            key: value.detach().cpu().numpy()
            for key, value in result.body_params.items()
        },
        "cam_trans": result.cam_trans.detach().cpu().numpy(),
        "scannet_gt": {
            "protocol": observation["protocol"],
            "gt_depth": observation["depth"],
            "gt_intrinsics": observation["K"],
            "raw_point_map": observation["points"],
            "raw_valid_mask": observation["valid_mask"],
            "raster_face_ids": observation["raster_face_ids"],
            "visible_vertices_camera": observation["visible_vertices_camera"],
            "visible_vertices_world": observation["visible_vertices_world"],
            "visible_faces": observation["visible_faces"],
            "visible_colors": observation["visible_colors"],
            "rotation_world_to_camera": observation["rotation_world_to_camera"],
            "translation_world_to_camera": observation["translation_world_to_camera"],
            "validation": observation["validation"],
            "metadata": observation["metadata"],
            "raw_moge_depth": diagnostics["raw_moge_depth"],
            "aligned_moge_depth": diagnostics["aligned_moge_depth"],
            "moge_valid_mask": diagnostics["moge_valid_mask"],
            "human_mask_undilated": diagnostics["human_mask_undilated"],
            "per_human_masks_undilated": diagnostics["per_human_masks_undilated"],
            "alignment_fit_mask": diagnostics["alignment_fit_mask"],
            "aligned_valid_mask": diagnostics["aligned_valid_mask"],
            "alignment": diagnostics["alignment"],
            "filtered_scene_points": result.pts3d.detach().cpu().numpy(),
            "filtered_scene_normals": result.normals.detach().cpu().numpy(),
            "point_inlier_mask": result.inlier_mask_pts.detach().cpu().numpy(),
            "normal_inlier_mask": result.inlier_mask_normals.detach().cpu().numpy(),
            "human_target_points": result.pts_humans.detach().cpu().numpy(),
            "human_target_lengths": result.pts_human_lengths.detach().cpu().numpy(),
            "gpu": {
                **gpu_metadata(torch),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "python_executable": sys.executable,
                "physic_root": str(physic_root),
                "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
                "peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
                "runtime_seconds_to_serialization": float(time.time() - started),
            },
        },
    }


def run_interaction(
    interaction_name: str,
    staging_root: Path,
    scannet_root: Path,
    physic_root: Path,
    cfg,
    torch,
    HumanScene,
) -> None:
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    human_image, scene_image = interaction_inputs(interaction_name)
    original_dir = staging_root / "original"
    original_dir.mkdir(parents=True)

    observation = build_scannet_gt_observation(
        project_dir=PROJECT_DIR,
        scannet_root=scannet_root,
        interaction_name=interaction_name,
        human_image_path=human_image,
        scene_image_path=scene_image,
        max_img_size=int(cfg.max_img_size),
        device=torch.device("cuda:0"),
    )
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        result = HumanScene(
            cfg,
            image_path=str(human_image),
            output_path=original_dir,
            scene_observation=observation,
        )
    if result.scale.requires_grad or float(result.scale.item()) != 1.0:
        raise RuntimeError("ScanNet++ scene scale must remain fixed at 1.0.")

    data = serialize_result(result, observation, torch, physic_root, started)
    with (original_dir / "scene_data_final.pkl").open("wb") as handle:
        pickle.dump(data, handle)
    write_evaluation_artifacts(original_dir, staging_root, physic_root)

    manifest_path = staging_root / "metadata" / "artifacts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"] = {
        **data["scannet_gt"]["gpu"],
        "total_runtime_seconds": float(time.time() - started),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    scannet_root = args.scannet_root.resolve()
    physic_root = args.physic_root.resolve()
    names = interaction_names(args.all_interactions, args.interaction_name)
    for name in names:
        interaction_inputs(name)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir(physic_root)
    sys.path[:0] = [str(physic_root), str(physic_root / "external" / "CameraHMR")]

    import torch
    from omegaconf import OmegaConf
    from optimizer import (
        HumanScene,
        load_chmr,
        load_deco,
        load_gsam,
        load_moge,
        load_vitpose,
        load_wilor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("PhySIC requires CUDA, but CUDA is unavailable.")

    cfg = OmegaConf.load(physic_root / "cfg" / "v1.yaml")
    cfg.compute_floor_points = False
    for stage in ("opt_1", "opt_2", "opt_3"):
        cfg[stage].train_params = [
            name for name in cfg[stage].train_params if name != "scale"
        ]

    for loader in (load_gsam, load_vitpose, load_chmr, load_deco, load_wilor, load_moge):
        loader()
    torch.set_default_dtype(torch.float32)

    output_root.mkdir(parents=True, exist_ok=True)
    for interaction_name in names:
        final_root = output_root / interaction_name
        staging_root = Path(
            tempfile.mkdtemp(prefix=f".{interaction_name}.staging-", dir=output_root)
        )
        try:
            run_interaction(
                interaction_name,
                staging_root,
                scannet_root,
                physic_root,
                cfg,
                torch,
                HumanScene,
            )
            publish(staging_root, final_root)
        except Exception:
            shutil.rmtree(staging_root, ignore_errors=True)
            raise
        print(f"{interaction_name}: {final_root}")
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
