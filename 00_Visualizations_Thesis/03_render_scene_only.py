from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_ROOT = PROJECT_DIR / "06_Evaluate_Interaction" / "output"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output_scenes"
DEFAULT_BLENDER_BIN = Path("/my_workspace/blender-4.2.17-linux-x64/blender")
HUMAN_OBJECT_NAME = "optimized_human"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalize_interaction_name(raw_name: str) -> str:
    name = raw_name.strip()
    if name.startswith("interaction_"):
        suffix = name.removeprefix("interaction_")
    else:
        suffix = name
    if not suffix.isdigit():
        raise ValueError(
            f"Invalid interaction name '{raw_name}'; expected interaction_XX or XX."
        )
    return f"interaction_{int(suffix):02d}"


def discover_interactions(source_root: Path) -> list[str]:
    return [
        path.parents[2].name
        for path in sorted(
            source_root.glob("interaction_*/semantics/assets/render_config.json")
        )
    ]


def build_blender_env(gpu_index: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if gpu_index is not None and gpu_index.strip():
        env["CUDA_VISIBLE_DEVICES"] = gpu_index.strip()
    return env


def validate_source_config(
    interaction_name: str,
    config_path: Path,
    source_config: dict[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    source_blend = Path(str(source_config.get("blend_path", ""))).resolve()
    if not source_blend.is_file():
        raise FileNotFoundError(
            f"{interaction_name}: source blend does not exist: {source_blend}"
        )

    raw_views = source_config.get("views")
    if not isinstance(raw_views, list) or not raw_views:
        raise ValueError(f"{interaction_name}: no views in {config_path}")

    views: list[dict[str, Any]] = []
    required = ("name", "camera_matrix_world", "intrinsics", "width", "height")
    for index, raw_view in enumerate(raw_views):
        if not isinstance(raw_view, dict):
            raise ValueError(f"{interaction_name}: view {index} is not an object")
        missing = [key for key in required if key not in raw_view]
        if missing:
            raise ValueError(
                f"{interaction_name}: view {index} is missing {', '.join(missing)}"
            )
        views.append(dict(raw_view))
    return source_blend, views


def build_job_config(
    interaction_name: str,
    source_config_path: Path,
    source_blend: Path,
    source_config: dict[str, Any],
    views: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    render_views = []
    for view in views:
        render_path = output_dir / f"{view['name']}.png"
        render_view = dict(view)
        render_view["source_render_path"] = view.get("render_path")
        render_view["render_path"] = str(render_path.resolve())
        render_views.append(render_view)

    return {
        "interaction_name": interaction_name,
        "source_render_config": str(source_config_path.resolve()),
        "source_blend": str(source_blend),
        "human_object_name": HUMAN_OBJECT_NAME,
        "cycles_samples": int(source_config.get("cycles_samples", 64)),
        "resolution_percentage": int(source_config.get("resolution_percentage", 100)),
        "overwrite": bool(overwrite),
        "views": render_views,
    }


def run_interaction(
    interaction_name: str,
    source_root: Path,
    output_root: Path,
    blender_bin: Path,
    gpu_index: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    source_config_path = (
        source_root
        / interaction_name
        / "semantics"
        / "assets"
        / "render_config.json"
    )
    if not source_config_path.is_file():
        raise FileNotFoundError(
            f"{interaction_name}: missing evaluation render config: {source_config_path}"
        )

    source_config = load_json(source_config_path)
    source_blend, views = validate_source_config(
        interaction_name, source_config_path, source_config
    )
    output_dir = output_root / interaction_name
    output_dir.mkdir(parents=True, exist_ok=True)
    job_config = build_job_config(
        interaction_name=interaction_name,
        source_config_path=source_config_path,
        source_blend=source_blend,
        source_config=source_config,
        views=views,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    job_config_path = output_dir / "render_config.json"
    save_json(job_config_path, job_config)

    expected_renders = [Path(view["render_path"]) for view in job_config["views"]]
    if not overwrite and all(path.is_file() for path in expected_renders):
        print(f"Skipping {interaction_name}: all {len(expected_renders)} renders exist")
        return {
            "interaction_name": interaction_name,
            "status": "skipped_existing",
            "source_render_config": str(source_config_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "render_paths": [str(path.resolve()) for path in expected_renders],
        }

    command = [
        str(blender_bin),
        "--background",
        str(source_blend),
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--blender-driver",
        str(job_config_path.resolve()),
    ]
    print(
        f"Rendering {interaction_name}: {len(expected_renders)} scene-only views -> "
        f"{output_dir}"
    )
    subprocess.run(
        command,
        check=True,
        env=build_blender_env(gpu_index),
    )

    missing = [str(path) for path in expected_renders if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"{interaction_name}: Blender completed but outputs are missing: {missing}"
        )
    return {
        "interaction_name": interaction_name,
        "status": "rendered",
        "source_render_config": str(source_config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "render_paths": [str(path.resolve()) for path in expected_renders],
    }


def configure_cycles_gpu(samples: int) -> None:
    import bpy

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = True

    preferences = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            gpu_devices = [
                device for device in preferences.devices if device.type != "CPU"
            ]
            if gpu_devices:
                selected_backend = backend
                break
        except Exception as exc:
            print(f"Cycles GPU backend {backend} unavailable: {exc}")

    if selected_backend is None:
        scene.cycles.device = "CPU"
        print("Cycles GPU unavailable; falling back to CPU")
        return

    for device in preferences.devices:
        device.use = device.type != "CPU"
    enabled = [
        f"{device.name} ({device.type})"
        for device in preferences.devices
        if device.use
    ]
    print(f"Cycles GPU backend: {selected_backend}; devices: {enabled}")


def run_blender_driver(job_config_path: Path) -> None:
    import bpy
    from mathutils import Matrix

    job = load_json(job_config_path)
    human_name = str(job["human_object_name"])
    human_obj = bpy.data.objects.get(human_name)
    if human_obj is None:
        available_meshes = [obj.name for obj in bpy.data.objects if obj.type == "MESH"]
        raise RuntimeError(
            f"Human object '{human_name}' is absent from source blend; "
            f"mesh objects are {available_meshes}"
        )
    human_obj.hide_render = True
    human_obj.hide_viewport = True
    print(f"Scene-only mode: disabled render visibility for '{human_name}'")

    configure_cycles_gpu(int(job["cycles_samples"]))
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    default_percentage = int(job["resolution_percentage"])
    overwrite = bool(job.get("overwrite", False))
    sensor_width = 36.0

    for view in job["views"]:
        render_path = Path(view["render_path"])
        if render_path.is_file() and not overwrite:
            print(f"Skipping existing view: {render_path}")
            continue

        camera_obj = bpy.data.objects.get(str(view["name"]))
        if camera_obj is None or camera_obj.type != "CAMERA":
            raise RuntimeError(f"Camera '{view['name']}' is absent from source blend")

        width = int(view["width"])
        height = int(view["height"])
        intrinsics = view["intrinsics"]
        fx = float(intrinsics[0][0])
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])
        camera_obj.matrix_world = Matrix(view["camera_matrix_world"])
        camera_obj.data.sensor_fit = "HORIZONTAL"
        camera_obj.data.sensor_width = sensor_width
        camera_obj.data.lens = fx * sensor_width / float(width)
        camera_obj.data.shift_x = (float(width) * 0.5 - cx) / float(width)
        camera_obj.data.shift_y = (cy - float(height) * 0.5) / float(width)

        scene.camera = camera_obj
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = int(
            view.get("resolution_percentage", default_percentage)
        )
        scene.render.filepath = str(render_path)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Rendering scene-only view: {view['name']} -> {render_path}")
        bpy.ops.render.render(write_still=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-render the cameras saved by 06_Evaluate_Interaction/output with "
            "the human hidden, producing matched scene-only thesis images."
        )
    )
    parser.add_argument(
        "interactions",
        nargs="*",
        help="Interaction names or numbers (default: every evaluation output).",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--blender-bin", type=Path, default=DEFAULT_BLENDER_BIN)
    parser.add_argument(
        "--gpu-index",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value passed to Blender.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render PNGs that already exist (default is resumable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    blender_bin = args.blender_bin.resolve()
    if not blender_bin.is_file():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")

    if args.interactions:
        interactions = [normalize_interaction_name(name) for name in args.interactions]
    else:
        interactions = discover_interactions(source_root)
    interactions = list(dict.fromkeys(interactions))
    if not interactions:
        raise RuntimeError(f"No evaluation render configs found below {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for interaction_name in interactions:
        try:
            record = run_interaction(
                interaction_name=interaction_name,
                source_root=source_root,
                output_root=output_root,
                blender_bin=blender_bin,
                gpu_index=args.gpu_index,
                overwrite=bool(args.overwrite),
            )
        except Exception as exc:
            print(f"FAILED {interaction_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            record = {
                "interaction_name": interaction_name,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(interaction_name)
        records.append(record)
        save_json(output_root / "scene_only_renders.json", records)

    rendered = sum(record["status"] == "rendered" for record in records)
    skipped = sum(record["status"] == "skipped_existing" for record in records)
    print(
        f"Finished: rendered={rendered}, skipped_existing={skipped}, "
        f"failed={len(failures)}; manifest={output_root / 'scene_only_renders.json'}"
    )
    if failures:
        raise SystemExit(f"Failed interactions: {', '.join(failures)}")


if __name__ == "__main__":
    blender_args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if blender_args[:1] == ["--blender-driver"]:
        if len(blender_args) != 2:
            raise SystemExit("Expected --blender-driver JOB_CONFIG.json")
        run_blender_driver(Path(blender_args[1]).resolve())
    else:
        main()
