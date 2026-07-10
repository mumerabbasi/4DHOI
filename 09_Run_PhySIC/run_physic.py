#!/usr/bin/env python3
"""Run PhySIC on generated human frames for selected interactions."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physic-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Phy-SIC",
    )
    parser.add_argument(
        "--human-frame-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "02_Generate_Human_Frame" / "output",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--interactions",
        nargs="+",
        default=["02"],
        help="Interaction ids, e.g. 02 03, interaction_02, or all.",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="Optional image path to use for a single interaction run.",
    )
    parser.add_argument(
        "--scene-image-mode",
        choices=("omni", "target-scene-crop"),
        default="omni",
        help="Use default PhySIC OmniEraser, or use precomputed target_scene_crop.png scene images.",
    )
    parser.add_argument(
        "--target-scene-root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "03_Estimate_Contact_Agentic"
        / "output",
        help="Root containing interaction_XX/assets/target_scene_crop.png files.",
    )
    parser.add_argument(
        "--target-scene-image",
        type=Path,
        default=None,
        help="Optional precomputed scene image path to use with one interaction.",
    )
    return parser.parse_args()


def get_gpu_memory_gb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().splitlines()[0]) / 1024
    except Exception:
        pass
    return 0


def normalize_interaction(value: str) -> str:
    if value.startswith("interaction_"):
        value = value.removeprefix("interaction_")
    return f"interaction_{int(value):02d}"


def resolve_interactions(values: list[str], human_frame_root: Path) -> list[str]:
    if len(values) == 1 and values[0].lower() == "all":
        return sorted(path.name for path in human_frame_root.glob("interaction_*") if path.is_dir())
    return [normalize_interaction(value) for value in values]


def timed_load(label: str, fn) -> None:
    start = time.time()
    fn()
    print(f"Time taken to load {label}: {time.time() - start:.4f} seconds", flush=True)


def main() -> None:
    args = parse_args()
    physic_root = args.physic_root.resolve()
    human_frame_root = args.human_frame_root.resolve()
    if args.output_root is None:
        output_dir_name = (
            "output"
            if args.scene_image_mode == "omni"
            else f"output_{args.scene_image_mode.replace('-', '_')}"
        )
        output_root = (Path(__file__).resolve().parent / output_dir_name).resolve()
    else:
        output_root = args.output_root.resolve()

    if get_gpu_memory_gb() > 60:
        os.environ["DISABLE_OFFLOAD"] = "1"
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    os.chdir(physic_root)
    sys.path.insert(0, str(physic_root))
    sys.path.insert(0, str(physic_root / "external" / "CameraHMR"))

    import torch
    from omegaconf import OmegaConf
    import optimizer as optimizer_module
    from optimizer import (
        HumanScene,
        load_chmr,
        load_deco,
        load_dpro,
        load_gsam,
        load_moge,
        load_omni,
        load_vitpose,
        load_wilor,
    )
    from utils.vis import get_scene

    cfg = OmegaConf.load(str(physic_root / "cfg" / "v1.yaml"))
    print(OmegaConf.to_yaml(cfg), flush=True)

    timed_load("gsam", load_gsam)
    timed_load("vitpose", load_vitpose)
    if args.scene_image_mode == "omni":
        timed_load("omni", load_omni)
    else:
        print(
            "Using precomputed target_scene_crop.png as PhySIC scene_image; OmniEraser not loaded.",
            flush=True,
        )
    if cfg.smpl_model != "chmr":
        raise ValueError(f"Unknown SMPL model: {cfg.smpl_model}")
    timed_load("chmr", load_chmr)
    timed_load("deco", load_deco)
    timed_load("wilor", load_wilor)
    timed_load("moge", load_moge)
    timed_load("dpro", load_dpro)

    failed: list[tuple[str, str]] = []
    interactions = resolve_interactions(args.interactions, human_frame_root)
    if args.input_image is not None and len(interactions) != 1:
        raise ValueError("--input-image can only be used with one interaction.")
    if args.target_scene_image is not None and len(interactions) != 1:
        raise ValueError("--target-scene-image can only be used with one interaction.")
    if args.target_scene_image is not None and args.scene_image_mode != "target-scene-crop":
        raise ValueError("--target-scene-image requires --scene-image-mode target-scene-crop.")

    for interaction in interactions:
        image_path = (
            args.input_image.resolve()
            if args.input_image is not None
            else human_frame_root / interaction / "inpainted_frame_resized.png"
        )
        target_scene_path = (
            args.target_scene_image.resolve()
            if args.target_scene_image is not None
            else args.target_scene_root.resolve()
            / interaction
            / "assets"
            / "target_scene_crop.png"
        )
        out_dir = output_root / interaction
        if not image_path.exists():
            failed.append((interaction, f"missing input {image_path}"))
            continue
        if args.scene_image_mode == "target-scene-crop" and not target_scene_path.exists():
            failed.append((interaction, f"missing target scene image {target_scene_path}"))
            continue

        print(f"Processing {interaction}: {image_path}", flush=True)
        if args.scene_image_mode == "target-scene-crop":
            print(f"Using target scene image: {target_scene_path}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            if args.scene_image_mode == "target-scene-crop":
                from PIL import Image

                def use_target_scene_crop(image, mask, scene_path=target_scene_path):
                    return Image.open(scene_path).convert("RGB").resize(image.size, Image.LANCZOS)

                optimizer_module.get_inpainted_image_omni = use_target_scene_crop

            with torch.amp.autocast(enabled=False, device_type="cuda"):
                result = HumanScene(cfg, image_path=str(image_path), output_path=out_dir)

            data = {
                "depth": result.depth.cpu().numpy(),
                "K": result.K.cpu().numpy(),
                "pts3d": result.pts3d.cpu().numpy(),
                "inlier_mask": result.inlier_mask.cpu().numpy(),
                "scale": result.scale.detach().cpu().numpy(),
                "normals": result.normals.cpu().numpy(),
                "plane_points": result.plane_points.cpu().numpy()
                if hasattr(result.plane_points, "cpu")
                else result.plane_points,
                "plane_normal": result.plane_normal.cpu().numpy()
                if hasattr(result.plane_normal, "cpu")
                else result.plane_normal,
                "body_params": {
                    key: value.detach().cpu().numpy()
                    for key, value in result.body_params.items()
                },
                "cam_trans": result.cam_trans.detach().cpu().numpy(),
            }
            with open(out_dir / "scene_data_final.pkl", "wb") as handle:
                pickle.dump(data, handle)

            with torch.amp.autocast(enabled=False, device_type="cuda"):
                scene = get_scene(out_dir, max_faces=int(1e18))
            scene.export(out_dir / "humanscene.ply")
        except Exception as exc:
            failed.append((interaction, repr(exc)))
            print(f"Failed {interaction}: {exc!r}", flush=True)

    print(f"Failed interactions: {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
