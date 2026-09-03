"""Run native GenZI on scene configurations prepared from ScanNet++."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent
WORKSPACE_ROOT = PROJECT_DIR.parent
GENZI_ROOT = WORKSPACE_ROOT / "GenZI"
DEFAULT_OUTPUT_BASE = MODULE_DIR / "output"
MODULE_05_OUTPUT = PROJECT_DIR / "05_Optimize_Static_Scene" / "output"
DEFAULT_RUN_CFG = GENZI_ROOT / "config" / "proxs_gen.yml"
DEFAULT_GENZI_PYTHON = Path("/root/miniconda3/envs/genzi/bin/python")
EXPECTED_TSDF_METHOD = (
    "module03_scene_crop_coverage_visibility_tsdf_384_depth2048_pose_dedup_v2"
)
EXPECTED_SDF_DIM = 384
EXPECTED_SCENE_SCOPE = "module03_crop"

if str(GENZI_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZI_ROOT))


def log(message: str) -> None:
    print(message, flush=True)


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
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cfg(path: Path) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from genzi.misc import omegaconf_to_dotdict

    payload = OmegaConf.load(path)
    payload = OmegaConf.to_container(payload, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    def absolutize(value: Any) -> Any:
        if isinstance(value, str) and value.startswith("./"):
            return str((GENZI_ROOT / value[2:]).resolve())
        return value

    cfg = omegaconf_to_dotdict(OmegaConf.create(payload))
    for key, value in list(cfg.items()):
        if key.endswith("_path") or key in {"path_prefix", "log_dir"}:
            cfg[key] = absolutize(value)
    cfg["run_cfg"] = str(path.resolve())
    return cfg


def interaction_sort_key(name: str) -> tuple[int, str]:
    prefix = "interaction_"
    suffix = name[len(prefix):] if name.startswith(prefix) else ""
    return (int(suffix) if suffix.isdigit() else 10**9, name)


def discover_module05_interactions() -> list[str]:
    """Return interactions with module 05's final completion artifact."""
    names = sorted(
        (
            path.parent.name
            for path in MODULE_05_OUTPUT.glob("interaction_*/alignment_summary.json")
        ),
        key=interaction_sort_key,
    )
    if not names:
        raise RuntimeError(
            "No processed interactions were found under module 05 output: "
            f"{MODULE_05_OUTPUT}"
        )
    return names


def select_prepared_module05_interactions(
    scene_config_root: Path,
) -> tuple[list[str], list[str]]:
    """Split module-05 interactions into prepared and not-yet-prepared sets."""
    processed = discover_module05_interactions()
    prepared = []
    missing = []
    for interaction_name in processed:
        config_path = scene_config_root / f"{interaction_name}_v1.yml"
        (prepared if config_path.is_file() else missing).append(interaction_name)
    if not prepared:
        raise FileNotFoundError(
            "No module-05 interactions have prepared GenZI scene configs under "
            f"{scene_config_root}. Run 00_prepare_genzi.py first."
        )
    return prepared, missing


def validate_scene_configs(
    scene_config_root: Path,
    interaction_names: list[str],
) -> list[dict[str, Any]]:
    import numpy as np
    import yaml

    reports: list[dict[str, Any]] = []
    output_base = scene_config_root.parent
    for interaction_name in interaction_names:
        config_path = scene_config_root / f"{interaction_name}_v1.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing prepared scene config: {config_path}. Run 00_prepare_genzi.py first."
            )
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("scene"), dict):
            raise ValueError(f"Invalid prepared scene config: {config_path}")
        for key in ("mesh_path", "sdf_path"):
            asset_path = Path(str(payload["scene"].get(key, "")))
            if not asset_path.exists():
                raise FileNotFoundError(
                    f"Prepared config {config_path} references missing scene.{key}: {asset_path}"
                )
        mesh_path = Path(str(payload["scene"]["mesh_path"])).resolve()
        sdf_meta_path = Path(str(payload["scene"]["sdf_path"])).resolve()
        summary_path = output_base / interaction_name / "preparation_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing preparation summary: {summary_path}")
        summary = load_json(summary_path)
        if summary.get("scene_scope") != EXPECTED_SCENE_SCOPE:
            raise ValueError(
                f"{interaction_name} is not a module-03 crop preparation; "
                f"scene_scope={summary.get('scene_scope')!r}"
            )
        summary_mesh = Path(str(summary.get("mesh_path", ""))).resolve()
        summary_sdf = Path(str(summary.get("sdf_meta", ""))).resolve()
        if mesh_path != summary_mesh or sdf_meta_path != summary_sdf:
            raise ValueError(
                f"{interaction_name} config assets disagree with its preparation summary"
            )
        full_mesh_raw = str(summary.get("full_scene_mesh_path", "")).strip()
        if full_mesh_raw and mesh_path == Path(full_mesh_raw).resolve():
            raise ValueError(f"{interaction_name} still references the full ScanNet++ mesh")

        sdf_metadata = load_json(sdf_meta_path)
        if sdf_metadata.get("method") != EXPECTED_TSDF_METHOD:
            raise ValueError(
                f"Unexpected TSDF method for {interaction_name}: "
                f"{sdf_metadata.get('method')!r}"
            )
        if int(sdf_metadata.get("dim", -1)) != EXPECTED_SDF_DIM:
            raise ValueError(f"{interaction_name} TSDF dim must be {EXPECTED_SDF_DIM}")
        if sdf_metadata.get("scene_scope") != EXPECTED_SCENE_SCOPE:
            raise ValueError(f"{interaction_name} SDF metadata is not crop-scoped")
        if not np.isclose(float(sdf_metadata.get("trunc_m", -1.0)), 0.20):
            raise ValueError(f"{interaction_name} TSDF truncation must be 0.20m")
        if not np.isclose(float(sdf_metadata.get("negative_band_m", -1.0)), 0.20):
            raise ValueError(f"{interaction_name} TSDF negative band must be 0.20m")
        sdf_path = sdf_meta_path.with_name(sdf_meta_path.stem + "_sdf.npy")
        if not sdf_path.is_file():
            raise FileNotFoundError(f"Missing TSDF array: {sdf_path}")
        sdf = np.load(sdf_path, mmap_mode="r")
        expected_shape = (EXPECTED_SDF_DIM,) * 3
        if sdf.shape != expected_shape or sdf.dtype != np.float32:
            raise ValueError(
                f"Invalid TSDF array for {interaction_name}: shape={sdf.shape}, dtype={sdf.dtype}"
            )
        observed_path = Path(str(sdf_metadata.get("observed_mask_path", "")))
        if not observed_path.is_file():
            raise FileNotFoundError(f"Missing TSDF observation mask: {observed_path}")
        observed = np.load(observed_path, mmap_mode="r")
        if observed.shape != expected_shape or observed.dtype != np.bool_:
            raise ValueError(
                f"Invalid observation mask for {interaction_name}: "
                f"shape={observed.shape}, dtype={observed.dtype}"
            )
        bounds_min = np.asarray(sdf_metadata.get("min"), dtype=np.float32)
        bounds_max = np.asarray(sdf_metadata.get("max"), dtype=np.float32)
        anchor = np.asarray(summary.get("look_at"), dtype=np.float32)
        if bounds_min.shape != (3,) or bounds_max.shape != (3,) or not np.all(bounds_min < bounds_max):
            raise ValueError(f"Invalid TSDF bounds for {interaction_name}")
        if anchor.shape != (3,) or not np.all((anchor >= bounds_min) & (anchor <= bounds_max)):
            raise ValueError(
                f"GenZI look-at for {interaction_name} lies outside its crop TSDF bounds"
            )
        crop = summary.get("crop")
        if not isinstance(crop, dict) or not np.allclose(
            np.asarray(crop.get("bbox_min"), dtype=np.float32), bounds_min, atol=1e-6
        ) or not np.allclose(
            np.asarray(crop.get("bbox_max"), dtype=np.float32), bounds_max, atol=1e-6
        ):
            raise ValueError(f"{interaction_name} crop and TSDF bounds disagree")
        reports.append(
            {
                "interaction_name": interaction_name,
                "scene_scope": EXPECTED_SCENE_SCOPE,
                "mesh_path": mesh_path,
                "sdf_meta_path": sdf_meta_path,
                "sdf_path": sdf_path,
                "sdf_dim": EXPECTED_SDF_DIM,
                "tsdf_method": EXPECTED_TSDF_METHOD,
            }
        )
    return reports


def ensure_alphapose_import_path() -> None:
    try:
        spec = importlib.util.find_spec("alphapose")
    except Exception:
        return
    if spec is None or spec.origin is None:
        return
    root = Path(spec.origin).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def ensure_smplx_symlink(model_path: str) -> None:
    model_root = Path(model_path)
    expected = model_root / "smplx" / "SMPLX_NEUTRAL.npz"
    if expected.exists():
        return
    source = WORKSPACE_ROOT / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx"
    if not (source / "SMPLX_NEUTRAL.npz").exists():
        return
    model_root.mkdir(parents=True, exist_ok=True)
    target_dir = model_root / "smplx"
    if not target_dir.exists() and not target_dir.is_symlink():
        target_dir.symlink_to(source, target_is_directory=True)


def install_diffusers_compat_shims() -> None:
    try:
        import huggingface_hub
    except Exception:
        return
    if hasattr(huggingface_hub, "cached_download"):
        return

    def cached_download(*call_args: Any, **call_kwargs: Any) -> str:
        from huggingface_hub import hf_hub_download

        if call_args:
            repo_id = str(call_args[0])
            filename = call_kwargs.pop("filename", None)
            if filename is None and len(call_args) > 1:
                filename = call_args[1]
            if (
                "/" in repo_id
                and filename is not None
                and not repo_id.startswith(("http://", "https://"))
            ):
                return hf_hub_download(repo_id=repo_id, filename=filename, **call_kwargs)
        raise RuntimeError("Legacy diffusers called cached_download with unsupported arguments.")

    huggingface_hub.cached_download = cached_download


def install_textureless_smplx_obj_export(generation_module: Any) -> None:
    original = generation_module.save_smplx_mesh

    def save_without_required_uv(
        filepath: str,
        template_path: str,
        texture_path: str,
        vertices: Any,
        *call_args: Any,
        **call_kwargs: Any,
    ) -> Any:
        try:
            return original(
                filepath,
                template_path,
                texture_path,
                vertices,
                *call_args,
                **call_kwargs,
            )
        except FileNotFoundError as exc:
            import trimesh

            ply_path = Path(filepath).with_suffix(".ply")
            if not ply_path.exists():
                raise
            trimesh.load(str(ply_path), force="mesh", process=False).export(filepath)
            mtl_path = Path(filepath).with_suffix(".mtl")
            if mtl_path.exists():
                mtl_path.unlink()
            log(f"[!] Missing SMPL-X texture asset; wrote textureless OBJ instead: {exc}")
            return None

    generation_module.save_smplx_mesh = save_without_required_uv


def limit_cfg_to_stage0(cfg: dict[str, Any]) -> None:
    stage_keys = [
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
    for key in stage_keys:
        value = cfg.get(key)
        if isinstance(value, list) and len(value) > 1:
            cfg[key] = value[:1]


def limit_optimization_steps(cfg: dict[str, Any], max_steps: int | None) -> None:
    if max_steps is None:
        return
    target = int(max_steps)
    if target < 4:
        raise ValueError("--max-steps must be at least 4 because GenZI has four phases.")
    for stage_index, stage_steps in enumerate(cfg["optim.steps"]):
        raw = [int(step) for step in stage_steps]
        if sum(raw) <= target:
            continue
        scaled = [max(1, int(step * target / sum(raw))) for step in raw]
        while sum(scaled) > target:
            index = max(range(len(scaled)), key=lambda item: scaled[item])
            if scaled[index] <= 1:
                break
            scaled[index] -= 1
        while sum(scaled) < target:
            index = max(range(len(raw)), key=lambda item: raw[item])
            scaled[index] += 1
        cfg["optim.steps"][stage_index] = scaled


def build_runtime_cfg(
    args: argparse.Namespace,
    interaction_names: list[str],
) -> dict[str, Any]:
    cfg = load_cfg(Path(args.run_cfg).resolve())
    if args.stages == "stage0":
        limit_cfg_to_stage0(cfg)
    limit_optimization_steps(cfg, args.max_steps)

    output_base = Path(args.output_base).resolve()
    cfg["data.root_dir"] = str((output_base / "_scene_configs").resolve())
    cfg["data.scenes"] = interaction_names
    cfg["data.cfg_suffix"] = "_v1.yml"
    cfg["log_dir"] = str((output_base / "genzi_runs").resolve())
    cfg["gpus"] = (
        [int(str(args.device).split(":")[-1])]
        if str(args.device).startswith("cuda:")
        else [0]
    )
    cfg["seed"] = int(args.seed)
    cfg["vlm.inpaint_dir"] = ""
    if args.exp_name:
        cfg["exp_time"] = str(args.exp_name)
    if args.ldm_inpaint_path:
        cfg["vlm.ldm_inpaint_path"] = str(args.ldm_inpaint_path)
    ensure_smplx_symlink(cfg["smplx.model_path"])
    return cfg


def run_native_genzi(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", str(args.opengl_platform))
    os.environ.setdefault("WANDB_MODE", str(args.wandb_mode))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ensure_alphapose_import_path()
    install_diffusers_compat_shims()

    # Import native GenZI before any renderer is initialized in this process.
    from genzi.misc import seeding
    import genzi.generation as generation

    install_textureless_smplx_obj_export(generation)
    seeding(int(cfg["seed"]))
    generation.cfg = cfg
    generation.GenZI(cfg).run_scenes()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native GenZI on ScanNet++ scenes prepared by 00_prepare_genzi.py."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--interaction-name",
        "--interaction_name",
        dest="interaction_name",
        default="interaction_01",
    )
    selection.add_argument(
        "--all-interactions",
        "--all_interactions",
        dest="all_interactions",
        action="store_true",
        help=(
            "Run interactions that are both processed by module 05 and have a "
            "prepared GenZI scene config; report and skip unprepared interactions."
        ),
    )
    parser.add_argument("--run-cfg", default=str(DEFAULT_RUN_CFG))
    parser.add_argument("--output-base", default=str(DEFAULT_OUTPUT_BASE))
    parser.add_argument("--genzi-python", default=str(DEFAULT_GENZI_PYTHON))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stages", choices=("all", "stage0"), default="all")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument(
        "--no-root-summary",
        action="store_true",
        help="Do not update the shared run summary (used by parallel launchers).",
    )
    parser.add_argument("--ldm-inpaint-path", default=None)
    parser.add_argument("--opengl-platform", default="egl")
    parser.add_argument("--wandb-mode", default="offline")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    requested_python = Path(args.genzi_python).resolve()
    if Path(sys.executable).resolve() != requested_python:
        if not requested_python.exists():
            raise FileNotFoundError(f"GenZI Python does not exist: {requested_python}")
        command = [requested_python, Path(__file__).resolve(), *(argv or sys.argv[1:])]
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=str(WORKSPACE_ROOT),
        )
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        return

    output_base = Path(args.output_base).resolve()
    scene_config_root = output_base / "_scene_configs"
    skipped_unprepared: list[str] = []
    if args.all_interactions:
        interaction_names, skipped_unprepared = select_prepared_module05_interactions(
            scene_config_root
        )
        log(
            f"[*] Selected {len(interaction_names)} prepared module-05 interaction(s): "
            + ", ".join(interaction_names)
        )
        if skipped_unprepared:
            log(
                f"[!] Skipping {len(skipped_unprepared)} module-05 interaction(s) "
                "without prepared GenZI configs: "
                + ", ".join(skipped_unprepared)
            )
    else:
        interaction_names = [args.interaction_name]
    scene_validations = validate_scene_configs(scene_config_root, interaction_names)
    cfg = build_runtime_cfg(args, interaction_names)

    log(f"[*] Running native GenZI for {len(interaction_names)} prepared interaction(s)")
    started = time.time()
    run_native_genzi(args, cfg)
    elapsed_s = float(time.time() - started)

    summary = {
        "interactions": interaction_names,
        "skipped_unprepared_interactions": skipped_unprepared,
        "run_cfg": str(Path(args.run_cfg).resolve()),
        "scene_config_root": scene_config_root,
        "genzi_log_dir": cfg["log_dir"],
        "stages": args.stages,
        "elapsed_s": elapsed_s,
        "scene_scope": EXPECTED_SCENE_SCOPE,
        "scene_validations": scene_validations,
    }
    if not args.no_root_summary:
        save_json(output_base / "genzi_run_summary.json", summary)
    for interaction_name in interaction_names:
        save_json(output_base / interaction_name / "genzi_run_summary.json", summary)
    log(f"[*] Native GenZI finished in {elapsed_s:.1f}s")
    log(f"[*] Log dir: {cfg['log_dir']}")


if __name__ == "__main__":
    main()
