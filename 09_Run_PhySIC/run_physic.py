#!/usr/bin/env python3
"""Run PhySIC for one 4DHSI interaction."""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Run PhySIC for one interaction.")
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--mode", choices=("default", "scannet"), default="default")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--human-image", default=None)
    parser.add_argument("--scene-image", default=None)
    parser.add_argument("--physic-root", default=None)
    parser.set_defaults(script_dir=script_dir, project_dir=project_dir)
    return parser.parse_args(argv)


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


def timed_load(label: str, fn) -> None:
    start = time.time()
    fn()
    print(f"Time taken to load {label}: {time.time() - start:.4f} seconds", flush=True)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    output_dir_name = "output" if args.mode == "default" else "output_scannet"
    out_dir = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / output_dir_name / args.interaction_name
    )
    human_image_path = (
        Path(args.human_image).resolve()
        if args.human_image
        else project_dir
        / "03_Estimate_Contact_Agentic"
        / "output"
        / args.interaction_name
        / "assets"
        / "reference_inpainted_crop.png"
    )
    scene_image_path = (
        Path(args.scene_image).resolve()
        if args.scene_image
        else project_dir
        / "03_Estimate_Contact_Agentic"
        / "output"
        / args.interaction_name
        / "assets"
        / "target_scene_crop.png"
    )
    physic_root = (
        Path(args.physic_root).resolve()
        if args.physic_root
        else project_dir.parent / "Phy-SIC"
    )

    if not human_image_path.exists():
        raise FileNotFoundError(f"Human image not found: {human_image_path}")
    if args.mode == "scannet" and not scene_image_path.exists():
        raise FileNotFoundError(f"Scene image not found: {scene_image_path}")

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
    if args.mode == "default":
        timed_load("omni", load_omni)
    else:
        print(
            "Using precomputed scene image as PhySIC scene_image; OmniEraser not loaded.",
            flush=True,
        )
    if cfg.smpl_model != "chmr":
        raise ValueError(f"Unknown SMPL model: {cfg.smpl_model}")
    timed_load("chmr", load_chmr)
    timed_load("deco", load_deco)
    timed_load("wilor", load_wilor)
    timed_load("moge", load_moge)
    timed_load("dpro", load_dpro)

    print(f"Processing {args.interaction_name}: {human_image_path}", flush=True)
    if args.mode == "scannet":
        print(f"Using scene image: {scene_image_path}", flush=True)

        from PIL import Image

        def use_scene_image(image, mask):
            return Image.open(scene_image_path).convert("RGB").resize(image.size, Image.LANCZOS)

        optimizer_module.get_inpainted_image_omni = use_scene_image

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        result = HumanScene(cfg, image_path=str(human_image_path), output_path=out_dir)

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

    print(f"Wrote PhySIC outputs for {args.interaction_name}: {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    main()
