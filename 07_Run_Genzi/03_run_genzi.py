from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"

if str(GENZI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZI_ROOT))


def configure_runtime(opengl_platform: str, wandb_mode: str) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", opengl_platform)
    os.environ.setdefault("WANDB_MODE", wandb_mode)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


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
    flat["run_cfg"] = str(path.resolve())
    return flat


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(to_jsonable(payload), sort_keys=False), encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def ensure_smplx_symlink(model_path: str) -> None:
    model_root = Path(model_path)
    expected_dir = model_root / "smplx"
    expected_npz = expected_dir / "SMPLX_NEUTRAL.npz"
    if expected_npz.exists():
        return
    source_dir = WORKSPACE_ROOT / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx"
    source_npz = source_dir / "SMPLX_NEUTRAL.npz"
    if not source_npz.exists():
        return
    model_root.mkdir(parents=True, exist_ok=True)
    if expected_dir.exists() or expected_dir.is_symlink():
        return
    expected_dir.symlink_to(source_dir, target_is_directory=True)


def ensure_alphapose_import_path() -> None:
    try:
        alphapose_spec = importlib.util.find_spec("alphapose")
    except Exception:
        return
    if alphapose_spec is None or alphapose_spec.origin is None:
        return
    alphapose_root = Path(alphapose_spec.origin).resolve().parent.parent
    if str(alphapose_root) not in sys.path:
        sys.path.insert(0, str(alphapose_root))


def install_diffusers_compat_shims() -> None:
    try:
        import huggingface_hub
    except Exception:
        return
    if hasattr(huggingface_hub, "cached_download"):
        return

    def cached_download(*args: Any, **kwargs: Any) -> str:
        from huggingface_hub import hf_hub_download

        if args:
            first = str(args[0])
            if "/" in first and not first.startswith(("http://", "https://")):
                repo_id = first
                filename = kwargs.pop("filename", None)
                if filename is None and len(args) > 1:
                    filename = args[1]
                if filename is not None:
                    return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
        raise RuntimeError(
            "diffusers requested huggingface_hub.cached_download, but this "
            "environment has a newer huggingface_hub without that API. Pin "
            "huggingface_hub<0.20 for full legacy compatibility."
        )

    huggingface_hub.cached_download = cached_download


def install_textureless_smplx_obj_export(generation_module: Any) -> None:
    original_save_smplx_mesh = generation_module.save_smplx_mesh

    def save_smplx_mesh_without_required_uv(
        filepath: str,
        template_path: str,
        texture_path: str,
        vertices: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            return original_save_smplx_mesh(
                filepath,
                template_path,
                texture_path,
                vertices,
                *args,
                **kwargs,
            )
        except FileNotFoundError as exc:
            ply_path = Path(filepath).with_suffix(".ply")
            if not ply_path.exists():
                raise

            import trimesh

            mesh = trimesh.load(str(ply_path), force="mesh", process=False)
            mesh.export(str(filepath))

            mtl_path = Path(filepath).with_suffix(".mtl")
            if mtl_path.exists():
                mtl_path.unlink()

            print(
                "[!] Missing SMPL-X UV/texture asset during OBJ export; "
                f"wrote textureless OBJ from {ply_path.name} instead. "
                f"Original error: {exc}",
                flush=True,
            )
            return None

    generation_module.save_smplx_mesh = save_smplx_mesh_without_required_uv


def limit_cfg_to_stage0(cfg: dict[str, Any]) -> None:
    stage_list_keys = [
        "data.view_distances",
        "optim.transl_lrs",
        "optim.orient_lrs",
        "optim.pose_lrs",
        "optim.shape_lrs",
        "optim.is_lrs",
        "optim.steps",
        "loss.inpaints_per_view",
        "loss.inpaint_score_weights",
        "loss.joint2d_torso_weights",
        "loss.joint2d_limb_weights",
        "loss.vposer_weights",
        "loss.beta_weights",
        "loss.scene_intersect_weights",
        "loss.scene_nocontact_weights",
        "loss.self_intersect_weights",
        "loss.angle_weights",
        "loss.floating_weights",
        "loss.joint3d_weights",
        "vlm.dynamic_mask_starts",
        "vlm.dynamic_mask_stops",
        "vlm.dilate_size",
        "vlm.dilate_iterations",
    ]
    for key in stage_list_keys:
        value = cfg.get(key)
        if isinstance(value, list) and len(value) > 1:
            cfg[key] = value[:1]


def prepare_runtime_scene_config(
    interaction_name: str,
    output_root: Path,
    source_scene_config: Path,
    stages: str,
) -> Path:
    if stages != "stage0":
        return source_scene_config

    scene_cfg = read_yaml(source_scene_config)
    prompt_viewpoints = scene_cfg.get("viewpoints", [])
    trimmed = []
    for viewpoints in prompt_viewpoints:
        if isinstance(viewpoints, list):
            trimmed.append(viewpoints[:1])
        else:
            trimmed.append(viewpoints)
    scene_cfg["viewpoints"] = trimmed

    runtime_scene_root = output_root / "stage0_scene_config"
    runtime_scene_config = runtime_scene_root / f"{interaction_name}_v1.yml"
    write_yaml(runtime_scene_config, scene_cfg)
    return runtime_scene_config


def discover_scene_config(interaction_name: str, output_root: Path, raw_path: str | None) -> Path:
    candidates = []
    if raw_path:
        candidates.append(Path(raw_path).resolve())
    candidates.extend(
        [
            output_root / f"{interaction_name}_v1.yml",
        ]
    )
    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing GenZI scene config. Run 02_render_multiview.py first, or pass "
        "--scene_config. Checked: " + ", ".join(str(path) for path in unique_candidates)
    )


def validate_external_inpaints(
    interaction_name: str,
    external_root: Path,
    cfg: dict[str, Any],
    stages: str,
    allow_partial: bool,
) -> dict[str, Any]:
    stage_count = 1 if stages == "stage0" else len(cfg["optim.steps"])
    records = []
    missing = []
    for stage_idx in range(stage_count):
        if stage_idx > 0:
            records.append(
                {
                    "stage_idx": stage_idx,
                    "note": (
                        "Stage > 0 external inpaint paths depend on selected "
                        "stage-0 priors, e.g. stage000_inpaint000_stage001_inpaint000. "
                        "GenZI will validate those at runtime."
                    ),
                }
            )
            continue
        inpaint_count = int(cfg["loss.inpaints_per_view"][stage_idx])
        for inpaint_id in range(inpaint_count):
            rel_dir = (
                Path(interaction_name)
                / interaction_name
                / f"stage{stage_idx:03d}_inpaint{inpaint_id:03d}"
            )
            directory = external_root / rel_dir
            files = sorted(directory.glob("view*.png")) if directory.exists() else []
            record = {
                "stage_idx": stage_idx,
                "inpaint_id": inpaint_id,
                "directory": directory,
                "num_files": len(files),
            }
            records.append(record)
            if not files:
                missing.append(record)
    if missing and not allow_partial:
        message = "\n".join(f"  - {item['directory']}" for item in missing)
        raise FileNotFoundError(
            "Missing external inpaint images. Expected view*.png in:\n" + message
        )
    return {"external_root": external_root, "records": records, "missing": missing}


def build_cfg(args: argparse.Namespace) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    output_root = Path(args.output_base).resolve() / args.interaction_name
    scene_config = discover_scene_config(args.interaction_name, output_root, args.scene_config)
    runtime_scene_config = prepare_runtime_scene_config(
        interaction_name=args.interaction_name,
        output_root=output_root,
        source_scene_config=scene_config,
        stages=args.stages,
    )

    cfg = load_cfg(Path(args.run_cfg).resolve())
    if args.stages == "stage0":
        limit_cfg_to_stage0(cfg)

    cfg["data.root_dir"] = str(runtime_scene_config.parent.resolve())
    cfg["data.scenes"] = [args.interaction_name]
    cfg["data.cfg_suffix"] = "_v1.yml"
    cfg["log_dir"] = str(
        Path(args.log_dir).resolve()
        if args.log_dir
        else (output_root / "genzi_runs").resolve()
    )
    if args.exp_name:
        cfg["exp_time"] = args.exp_name
    cfg["gpus"] = [int(str(args.device).split(":")[-1])] if str(args.device).startswith("cuda:") else [0]
    cfg["seed"] = int(args.seed)
    if args.ldm_inpaint_path:
        cfg["vlm.ldm_inpaint_path"] = args.ldm_inpaint_path

    if args.max_steps is not None:
        target = int(args.max_steps)
        limited_steps = []
        for stage_steps in cfg["optim.steps"]:
            raw = [int(step) for step in stage_steps]
            if sum(raw) <= target:
                limited_steps.append(raw)
                continue
            scale = max(target / max(sum(raw), 1), 0.0)
            scaled = [max(1, int(step * scale)) for step in raw]
            while sum(scaled) > target:
                idx = max(range(len(scaled)), key=lambda i: scaled[i])
                if scaled[idx] <= 1:
                    break
                scaled[idx] -= 1
            while sum(scaled) < target:
                idx = max(range(len(raw)), key=lambda i: raw[i])
                scaled[idx] += 1
            limited_steps.append(scaled)
        cfg["optim.steps"] = limited_steps

    external_info: dict[str, Any] = {}
    if args.inpaint_mode == "external":
        external_root = (
            Path(args.external_inpaint_root).resolve()
            if args.external_inpaint_root
            else output_root / "external_inpaints"
        )
        external_info = validate_external_inpaints(
            interaction_name=args.interaction_name,
            external_root=external_root,
            cfg=cfg,
            stages=args.stages,
            allow_partial=bool(args.allow_partial_external),
        )
        cfg["vlm.inpaint_dir"] = str(external_root.resolve())
    else:
        cfg["vlm.inpaint_dir"] = ""

    ensure_smplx_symlink(cfg["smplx.model_path"])
    return cfg, runtime_scene_config, external_info


def run_genzi(args: argparse.Namespace) -> dict[str, Any]:
    configure_runtime(args.opengl_platform, args.wandb_mode)
    ensure_alphapose_import_path()
    install_diffusers_compat_shims()
    cfg, runtime_scene_config, external_info = build_cfg(args)

    from genzi.misc import seeding
    import genzi.generation as generation

    install_textureless_smplx_obj_export(generation)

    if args.inpaint_mode == "external" and bool(args.skip_ldm_load_in_external):
        generation.get_ldm_inpaint = lambda _path, _device: None

    seeding(int(cfg["seed"]))
    generation.cfg = cfg
    app = generation.GenZI(cfg)
    app.run_scenes()

    summary = {
        "interaction_name": args.interaction_name,
        "inpaint_mode": args.inpaint_mode,
        "stages": args.stages,
        "scene_config": runtime_scene_config,
        "log_dir": cfg["log_dir"],
        "run_cfg": cfg["run_cfg"],
        "vlm.inpaint_dir": cfg["vlm.inpaint_dir"],
        "external": external_info,
    }
    if args.write_summary:
        output_root = Path(args.output_base).resolve() / args.interaction_name
        save_json(output_root / "genzi_run_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GenZI on 4DHSI multiview scene configs from 02_render_multiview.py. "
            "Default mode uses GenZI's original Stable Diffusion inpainting."
        )
    )
    parser.add_argument("--interaction_name", required=True)
    parser.add_argument("--run_cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output_base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--scene_config", default=None)
    parser.add_argument(
        "--inpaint_mode",
        choices=("ldm", "external"),
        default="ldm",
        help="ldm uses GenZI's Stable Diffusion path; external reads pre-inpainted view*.png files.",
    )
    parser.add_argument("--external_inpaint_root", default=None)
    parser.add_argument(
        "--ldm_inpaint_path",
        default=None,
        help=(
            "Override the Hugging Face repo id or local path for GenZI LDM "
            "inpainting. Default comes from run_cfg."
        ),
    )
    parser.add_argument(
        "--stages",
        choices=("all", "stage0"),
        default="all",
        help=(
            "Run all GenZI stages, or only stage0. External full two-stage mode "
            "requires staged external inpaints for every prior/stage directory."
        ),
    )
    parser.add_argument("--allow_partial_external", action="store_true")
    parser.add_argument("--skip_ldm_load_in_external", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--exp_name", default=None)
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Override GenZI log_dir. Default is output_base/<interaction>/genzi_runs.",
    )
    parser.add_argument(
        "--write_summary",
        action="store_true",
        help="Write a wrapper summary JSON under output_base/<interaction>. Default leaves only GenZI outputs.",
    )
    parser.add_argument("--opengl_platform", default="egl")
    parser.add_argument("--wandb_mode", default="offline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    started = time.time()
    print(
        f"[*] Running GenZI for {args.interaction_name} "
        f"mode={args.inpaint_mode} stages={args.stages}",
        flush=True,
    )
    summary = run_genzi(args)
    elapsed_s = time.time() - started
    print(f"[*] Finished GenZI run in {elapsed_s:.1f}s", flush=True)
    print(f"[*] Log dir: {summary['log_dir']}", flush=True)


if __name__ == "__main__":
    main()
