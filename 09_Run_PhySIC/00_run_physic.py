#!/usr/bin/env python3
"""Run PhySIC for 4DHSI Agentic Contact interactions."""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path

from physic_eval_utils import write_evaluation_artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(description="Run PhySIC for 4DHSI interactions.")
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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


def interaction_sort_key(interaction_name: str) -> tuple[int, str]:
    match = re.fullmatch(r"interaction_(\d+)", interaction_name)
    return (int(match.group(1)), interaction_name) if match else (sys.maxsize, interaction_name)


def discover_agentic_interactions(project_dir: Path) -> list[str]:
    output_root = project_dir / "03_Estimate_Contact_Agentic" / "output"
    interaction_names = sorted(
        [path.name for path in output_root.glob("interaction_*") if path.is_dir()],
        key=interaction_sort_key,
    )
    if not interaction_names:
        raise FileNotFoundError(
            f"No Agentic Contact interactions found under {output_root}."
        )
    return interaction_names


def resolve_interaction_paths(
    args: argparse.Namespace,
    interaction_name: str,
) -> tuple[Path, Path, Path, Path]:
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    if args.outdir:
        out_dir = Path(args.outdir).resolve()
        interaction_root = out_dir.parent if args.mode == "scannet" else out_dir
    elif args.mode == "scannet":
        interaction_root = script_dir / "output_scannet" / interaction_name
        out_dir = interaction_root / "original"
    else:
        interaction_root = script_dir / "output" / interaction_name
        out_dir = interaction_root
    human_image_path = (
        Path(args.human_image).resolve()
        if args.human_image
        else project_dir
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
        / "reference_inpainted_crop.png"
    )
    scene_image_path = (
        Path(args.scene_image).resolve()
        if args.scene_image
        else project_dir
        / "03_Estimate_Contact_Agentic"
        / "output"
        / interaction_name
        / "assets"
        / "target_scene_crop.png"
    )
    if not human_image_path.exists():
        raise FileNotFoundError(f"Human image not found: {human_image_path}")
    if args.mode == "scannet" and not scene_image_path.exists():
        raise FileNotFoundError(f"Scene image not found: {scene_image_path}")
    return interaction_root, out_dir, human_image_path, scene_image_path


def run_interaction(
    interaction_name: str,
    args: argparse.Namespace,
    cfg,
    torch,
    optimizer_module,
    HumanScene,
    get_scene,
    physic_root: Path,
) -> Path:
    interaction_root, out_dir, human_image_path, scene_image_path = (
        resolve_interaction_paths(args, interaction_name)
    )

    print(f"Processing {interaction_name}: {human_image_path}", flush=True)
    if args.mode == "scannet":
        print(f"Using scene image: {scene_image_path}", flush=True)

        from PIL import Image

        def use_scene_image(image, mask):
            return Image.open(scene_image_path).convert("RGB").resize(
                image.size, Image.LANCZOS
            )

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

    if args.mode == "scannet":
        print(
            f"Writing PhySIC-native evaluation artifacts: {interaction_root}",
            flush=True,
        )
        write_evaluation_artifacts(
            original_dir=out_dir,
            interaction_root=interaction_root,
            physic_root=physic_root,
        )

    print(f"Wrote PhySIC outputs for {interaction_name}: {out_dir}", flush=True)
    return out_dir


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    project_dir: Path = args.project_dir
    physic_root = (
        Path(args.physic_root).resolve()
        if args.physic_root
        else project_dir.parent / "Phy-SIC"
    )
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        custom_paths = [
            option
            for option, value in (
                ("--outdir", args.outdir),
                ("--human-image", args.human_image),
                ("--scene-image", args.scene_image),
            )
            if value is not None
        ]
        if custom_paths:
            raise ValueError(
                "All-interactions mode cannot use per-interaction overrides: "
                + ", ".join(custom_paths)
            )
        interaction_names = discover_agentic_interactions(project_dir)
    else:
        interaction_names = [args.interaction_name]

    # Validate every input before loading the expensive PhySIC models.
    for interaction_name in interaction_names:
        resolve_interaction_paths(args, interaction_name)

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

    outputs = []
    for index, interaction_name in enumerate(interaction_names, start=1):
        print(
            f"[{index}/{len(interaction_names)}] Starting {interaction_name}",
            flush=True,
        )
        outputs.append(
            run_interaction(
                interaction_name=interaction_name,
                args=args,
                cfg=cfg,
                torch=torch,
                optimizer_module=optimizer_module,
                HumanScene=HumanScene,
                get_scene=get_scene,
                physic_root=physic_root,
            )
        )
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"Completed {len(outputs)} interaction(s) in {args.mode} mode.",
        flush=True,
    )
    return outputs


if __name__ == "__main__":
    main()
