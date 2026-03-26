#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen


STAGES = [
    "pag",
    "first_frame",
    "select_first_frame",
    "video",
    "segment_video",
    "depth",
    "object_mesh",
    "human_motion",
    "human_ply",
    "align_meshes",
    "align_human",
    "optical_flow",
    "render_mesh_views",
    "segment_renders",
    "segment_meshes",
    "track_object_mesh",
    "track_human_object_mesh",
    "track_human_object_joint",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def glob_sorted(base: Path, pattern: str) -> list[Path]:
    return sorted(base.glob(pattern))


def matches_for_stage(repo_root: Path, video_name: str, stage: str) -> list[Path]:
    patterns = {
        "pag": [f"Generate_PAG/output/{video_name}/output_pag_*.json"],
        "first_frame": [
            f"Generate_Video/output/{video_name}/first_frames/*.png",
        ],
        "select_first_frame": [
            f"Generate_Video/output/{video_name}/selected_first_frame.json",
        ],
        "video": [f"Generate_Video/output/{video_name}/*.mp4"],
        "segment_video": [
            f"Segment_Video/output/{video_name}/_frames/frame_0000.jpg",
            f"Segment_Video/output/{video_name}/_frames/0000.jpg",
        ],
        "depth": [f"Estimate_Depth/output/{video_name}/run_summary.json"],
        "object_mesh": [
            f"Generate_Object_Mesh/output/{video_name}/camera_intrinsics.json",
            f"Generate_Object_Mesh/output/{video_name}/meshes/*.ply",
        ],
        "human_motion": [
            (
                f"Estimate_Human_Motion/output/{video_name}"
                "/humans/*/hmr4d_results.pt"
            ),
        ],
        "human_ply": [
            (
                f"Estimate_Human_Motion/output/{video_name}"
                "/humans/*/human_plys/frame_0000.ply"
            ),
        ],
        "align_meshes": [f"Align_Meshes/output/{video_name}/alignment_summary.json"],
        "align_human": [
            (
                f"Align_Meshes/output/{video_name}"
                "/human_motion_aligned/*/frame_0000.ply"
            ),
        ],
        "optical_flow": [
            f"Estimate_Optical_Flow/output_cotracker/{video_name}/run_summary.json",
        ],
        "render_mesh_views": [
            f"Segment_Object_Mesh/output/{video_name}/*/renders/cameras.json",
        ],
        "segment_renders": [
            f"Segment_Object_Mesh/output/{video_name}/*/bboxes/part_bboxes.json",
        ],
        "segment_meshes": [
            (
                f"Segment_Object_Mesh/output/{video_name}"
                "/*/segmented_meshes/*_triangle_labels.json"
            ),
        ],
        "track_object_mesh": [f"Track_Object_Mesh/output/{video_name}/*/poses.json"],
        "track_human_object_mesh": [
            f"Track_Human_Object_Mesh/output/{video_name}/run_summary.json",
        ],
        "track_human_object_joint": [
            f"Track_Human_Object_Joint/output/{video_name}/run_summary.json",
        ],
    }[stage]

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(glob_sorted(repo_root, pattern))
    return sorted(set(matches))


def stage_done(repo_root: Path, video_name: str, stage: str) -> bool:
    return len(matches_for_stage(repo_root, video_name, stage)) > 0


def host_is_local(host: str) -> bool:
    parts = urlsplit(host)
    hostname = (parts.hostname or "").lower()
    return hostname in {"127.0.0.1", "localhost"}


def ollama_tags_url(host: str) -> str:
    parts = urlsplit(host)
    scheme = parts.scheme or "http"
    hostname = parts.hostname or "127.0.0.1"
    port = parts.port or 11434
    return f"{scheme}://{hostname}:{port}/api/tags"


def ollama_is_ready(host: str, timeout_seconds: float = 2.0) -> bool:
    request_timeout = max(1.0, timeout_seconds)
    try:
        with urlopen(ollama_tags_url(host), timeout=request_timeout) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


@dataclass
class Logger:
    log_path: Path

    def log(self, message: str) -> None:
        stamped = (
            f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] "
            f"{message}"
        )
        print(stamped, flush=True)
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(stamped + "\n")


@dataclass
class PipelineContext:
    args: argparse.Namespace
    repo_root: Path
    workspace_root: Path
    run_dir: Path
    logger: Logger
    summary_path: Path
    stage_results: dict[str, dict[str, str]] = field(default_factory=dict)
    ollama_process: subprocess.Popen[str] | None = None

    @property
    def pag_script(self) -> Path:
        return self.repo_root / "Generate_PAG" / "generate_pag.py"

    @property
    def first_frame_script(self) -> Path:
        return self.repo_root / "Generate_Video" / "generate_first_frame.py"

    @property
    def select_first_frame_script(self) -> Path:
        return self.repo_root / "Generate_Video" / "select_first_frame.py"

    @property
    def generate_video_script(self) -> Path:
        return self.repo_root / "Generate_Video" / "generate_video.py"

    @property
    def segment_video_script(self) -> Path:
        return self.repo_root / "Segment_Video" / "segment_video.py"

    @property
    def depth_script(self) -> Path:
        return self.repo_root / "Estimate_Depth" / "estimate_depth.py"

    @property
    def object_mesh_script(self) -> Path:
        return (
            self.repo_root
            / "Generate_Object_Mesh"
            / "generate_objects_meshes.py"
        )

    @property
    def human_motion_script(self) -> Path:
        return (
            self.repo_root
            / "Estimate_Human_Motion"
            / "estimate_human_motion.py"
        )

    @property
    def human_ply_script(self) -> Path:
        return (
            self.repo_root
            / "Estimate_Human_Motion"
            / "export_human_motion_to_ply.py"
        )

    @property
    def align_meshes_script(self) -> Path:
        return self.repo_root / "Align_Meshes" / "align_meshes.py"

    @property
    def align_human_script(self) -> Path:
        return self.repo_root / "Align_Meshes" / "align_human_motion_sequence.py"

    @property
    def optical_flow_script(self) -> Path:
        return (
            self.repo_root
            / "Estimate_Optical_Flow"
            / "estimate_optical_flow_cotracker.py"
        )

    @property
    def render_mesh_views_script(self) -> Path:
        return self.repo_root / "Segment_Object_Mesh" / "render_mesh_views.py"

    @property
    def segment_renders_script(self) -> Path:
        return self.repo_root / "Segment_Object_Mesh" / "segment_renders.py"

    @property
    def segment_meshes_script(self) -> Path:
        return self.repo_root / "Segment_Object_Mesh" / "segment_meshes.py"

    @property
    def track_object_script(self) -> Path:
        return self.repo_root / "Track_Object_Mesh" / "track_object_mesh.py"

    @property
    def track_human_object_script(self) -> Path:
        return (
            self.repo_root
            / "Track_Human_Object_Mesh"
            / "track_human_object_mesh.py"
        )

    @property
    def track_human_object_joint_script(self) -> Path:
        return (
            self.repo_root
            / "Track_Human_Object_Joint"
            / "track_human_object_joint.py"
        )

    @property
    def gvhmr_root(self) -> Path:
        return Path(self.args.gvhmr_root).resolve()

    @property
    def smpl_folder(self) -> Path:
        return Path(self.args.smpl_folder).resolve()

    @property
    def smplx2smpl_path(self) -> Path:
        return Path(self.args.smplx2smpl_path).resolve()


def update_summary(ctx: PipelineContext) -> None:
    payload = {
        "video_name": ctx.args.video_name,
        "run_dir": str(ctx.run_dir),
        "start_stage": ctx.args.start_stage,
        "end_stage": ctx.args.end_stage,
        "worker_gpu": ctx.args.worker_gpu,
        "ollama_gpu": ctx.args.ollama_gpu,
        "ollama_host": ctx.args.ollama_host,
        "stages": ctx.stage_results,
    }
    ctx.summary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def build_conda_python_command(
    env_name: str,
    script_path: Path,
    extra_args: Iterable[str],
) -> list[str]:
    return [
        "conda",
        "run",
        "-n",
        env_name,
        "python",
        str(script_path),
        *list(extra_args),
    ]


def stage_log_path(ctx: PipelineContext, stage_name: str) -> Path:
    index = STAGES.index(stage_name) + 1
    return ctx.run_dir / f"{index:02d}_{stage_name}.log"


def masked_env(base_env: dict[str, str], gpu: int | None) -> dict[str, str]:
    env = dict(base_env)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def run_command(
    ctx: PipelineContext,
    stage_name: str,
    command: list[str],
    *,
    env_overrides: dict[str, str] | None = None,
) -> None:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    log_path = stage_log_path(ctx, stage_name)
    ctx.logger.log(f"{stage_name}: {shlex.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=str(ctx.workspace_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def maybe_start_ollama(ctx: PipelineContext) -> None:
    if not ctx.args.start_ollama:
        ctx.logger.log("ollama: start skipped by flag")
        return
    if not host_is_local(ctx.args.ollama_host):
        raise ValueError("--start-ollama only supports a local --ollama-host")
    if ollama_is_ready(ctx.args.ollama_host):
        ctx.logger.log(f"ollama: already reachable at {ctx.args.ollama_host}")
        return

    env = masked_env(os.environ.copy(), ctx.args.ollama_gpu)
    env["OLLAMA_CONTEXT_LENGTH"] = str(ctx.args.ollama_context_length)
    log_path = ctx.run_dir / "ollama.log"
    ctx.logger.log(
        f"ollama: starting local service on gpu={ctx.args.ollama_gpu} "
        f"with OLLAMA_CONTEXT_LENGTH={ctx.args.ollama_context_length}"
    )
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            ["ollama", "serve"],
            cwd=str(ctx.workspace_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    ctx.ollama_process = process
    deadline = time.time() + ctx.args.ollama_wait_seconds
    while time.time() < deadline:
        if ollama_is_ready(ctx.args.ollama_host):
            ctx.logger.log(f"ollama: ready at {ctx.args.ollama_host}")
            return
        time.sleep(1.0)
    raise RuntimeError(
        "Ollama did not become ready within "
        f"{ctx.args.ollama_wait_seconds} seconds."
    )


def stage_extra_args(ctx: PipelineContext, stage_name: str) -> list[str]:
    video_name = ctx.args.video_name
    qwen_model = ctx.args.qwen_model
    ollama_host = ctx.args.ollama_host
    reasoning = ctx.args.qwen_reasoning_effort

    if stage_name == "pag":
        args = ["--video_name", video_name, "--host", ollama_host]
        if ctx.args.pag_model:
            args += ["--model", ctx.args.pag_model]
        if ctx.args.pag_input_dir:
            args += ["--input-dir", ctx.args.pag_input_dir]
        if reasoning:
            args += ["--reasoning-effort", reasoning]
        return args

    if stage_name == "first_frame":
        args = [
            "--video_name",
            video_name,
            "--device",
            "cuda:0",
            "--n",
            str(ctx.args.first_frame_count),
        ]
        if ctx.args.first_frame_seed is not None:
            args += ["--seed", str(ctx.args.first_frame_seed)]
        return args

    if stage_name == "select_first_frame":
        args = [
            "--video_name",
            video_name,
            "--ollama_host",
            ollama_host,
            "--qwen_model",
            qwen_model,
        ]
        if reasoning:
            args += ["--reasoning-effort", reasoning]
        return args

    if stage_name == "video":
        args = ["--video_name", video_name, "--device", "cuda:0"]
        if ctx.args.video_model:
            args += ["--model", ctx.args.video_model]
        if ctx.args.video_seed is not None:
            args += ["--seed", str(ctx.args.video_seed)]
        return args

    if stage_name == "segment_video":
        return [
            "--video_name",
            video_name,
            "--ollama_host",
            ollama_host,
            "--qwen_model",
            qwen_model,
        ]

    if stage_name == "depth":
        return ["--video_name", video_name, "--device", "cuda:0"]

    if stage_name == "object_mesh":
        return ["--video_name", video_name]

    if stage_name == "human_motion":
        return [
            "--video_name",
            video_name,
            "--gvhmr_path",
            str(ctx.gvhmr_root),
            "--device",
            "cuda:0",
        ]

    if stage_name == "human_ply":
        return [
            "--video_name",
            video_name,
            "--smpl_folder",
            str(ctx.smpl_folder),
            "--smplx2smpl_path",
            str(ctx.smplx2smpl_path),
            "--gvhmr_path",
            str(ctx.gvhmr_root),
        ]

    if stage_name == "align_meshes":
        return ["--video_name", video_name, "--device", "cuda:0"]

    if stage_name == "align_human":
        return ["--video_name", video_name]

    if stage_name == "optical_flow":
        return ["--video_name", video_name, "--device", "cuda:0"]

    if stage_name == "render_mesh_views":
        return ["--video_name", video_name]

    if stage_name == "segment_renders":
        args = [
            "--video_name",
            video_name,
            "--device",
            "cuda:0",
            "--ollama_host",
            ollama_host,
            "--qwen_model",
            qwen_model,
        ]
        if reasoning:
            args += ["--reasoning-effort", reasoning]
        return args

    if stage_name == "segment_meshes":
        return ["--video_name", video_name]

    if stage_name == "track_object_mesh":
        return ["--video_name", video_name, "--device", "cuda:0"]

    if stage_name == "track_human_object_mesh":
        return ["--video_name", video_name, "--device", "cuda:0"]

    if stage_name == "track_human_object_joint":
        return ["--video_name", video_name, "--device", "cuda:0"]

    raise KeyError(stage_name)


def stage_command(
    ctx: PipelineContext,
    stage_name: str,
) -> tuple[list[str], dict[str, str] | None]:
    gpu_env = masked_env({}, ctx.args.worker_gpu)
    if stage_name == "pag":
        command = build_conda_python_command(
            ctx.args.pag_env,
            ctx.pag_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, None
    if stage_name == "first_frame":
        command = build_conda_python_command(
            ctx.args.video_env,
            ctx.first_frame_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "select_first_frame":
        command = build_conda_python_command(
            ctx.args.video_env,
            ctx.select_first_frame_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, None
    if stage_name == "video":
        command = build_conda_python_command(
            ctx.args.video_env,
            ctx.generate_video_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "segment_video":
        command = build_conda_python_command(
            ctx.args.segment_env,
            ctx.segment_video_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "depth":
        command = build_conda_python_command(
            ctx.args.depth_env,
            ctx.depth_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "object_mesh":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.object_mesh_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "human_motion":
        command = build_conda_python_command(
            ctx.args.human_env,
            ctx.human_motion_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "human_ply":
        command = build_conda_python_command(
            ctx.args.human_env,
            ctx.human_ply_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, None
    if stage_name == "align_meshes":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.align_meshes_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "align_human":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.align_human_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, None
    if stage_name == "optical_flow":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.optical_flow_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "render_mesh_views":
        return (
            [
                ctx.args.blender_cmd,
                "--background",
                "--python",
                str(ctx.render_mesh_views_script),
                "--",
                *stage_extra_args(ctx, stage_name),
            ],
            gpu_env,
        )
    if stage_name == "segment_renders":
        command = build_conda_python_command(
            ctx.args.segment_env,
            ctx.segment_renders_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "segment_meshes":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.segment_meshes_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, None
    if stage_name == "track_object_mesh":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.track_object_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "track_human_object_mesh":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.track_human_object_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    if stage_name == "track_human_object_joint":
        command = build_conda_python_command(
            ctx.args.sam3d_env,
            ctx.track_human_object_joint_script,
            stage_extra_args(ctx, stage_name),
        )
        return command, gpu_env
    raise KeyError(stage_name)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    gvhmr_root = repo_root.parent / "GVHMR"

    parser = argparse.ArgumentParser(
        description=(
            "Run one video through the full 4DHOI pipeline "
            "with resumable stage checks."
        ),
    )
    parser.add_argument("--video_name", required=True)
    parser.add_argument("--start-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--end-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun stages even when their output markers already exist.",
    )

    parser.add_argument(
        "--worker-gpu",
        type=int,
        default=0,
        help="GPU used for non-Ollama model stages.",
    )
    parser.add_argument(
        "--ollama-gpu",
        type=int,
        default=7,
        help="GPU used when starting a local Ollama server.",
    )
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama-context-length", type=int, default=200000)
    parser.add_argument(
        "--start-ollama",
        dest="start_ollama",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-start-ollama", dest="start_ollama", action="store_false")
    parser.add_argument("--ollama-wait-seconds", type=int, default=60)
    parser.add_argument("--qwen-model", default="qwen3-vl:32b-thinking")
    parser.add_argument(
        "--qwen-reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="high",
    )

    parser.add_argument("--pag-env", default="4dhoi")
    parser.add_argument("--video-env", default="4dhoi")
    parser.add_argument("--segment-env", default="sam3")
    parser.add_argument("--sam3d-env", default="sam3d-objects")
    parser.add_argument("--depth-env", default="depth-anything3")
    parser.add_argument("--human-env", default="gvhmr")

    parser.add_argument("--pag-model", default=None)
    parser.add_argument(
        "--pag-input-dir",
        default=None,
        help="Override Generate_PAG/input_prompts/<video_name>.",
    )
    parser.add_argument("--first-frame-count", type=int, default=5)
    parser.add_argument("--first-frame-seed", type=int, default=None)
    parser.add_argument("--video-model", default=None)
    parser.add_argument("--video-seed", type=int, default=None)
    parser.add_argument("--blender-cmd", default="blender")

    parser.add_argument("--gvhmr-root", default=str(gvhmr_root))
    parser.add_argument(
        "--smpl-folder",
        default=str(gvhmr_root / "inputs" / "checkpoints" / "body_models"),
    )
    parser.add_argument(
        "--smplx2smpl-path",
        default=str(
            gvhmr_root
            / "hmr4d"
            / "utils"
            / "body_model"
            / "smplx2smpl_sparse.pt"
        ),
    )

    return parser.parse_args()


def build_context(args: argparse.Namespace) -> PipelineContext:
    repo_root = Path(__file__).resolve().parent
    workspace_root = repo_root.parent
    run_dir = (
        workspace_root
        / "run_logs"
        / "pipeline_runner"
        / args.video_name
        / utc_timestamp()
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_link = (
        workspace_root
        / "run_logs"
        / "pipeline_runner"
        / args.video_name
        / "latest"
    )
    latest_link.parent.mkdir(parents=True, exist_ok=True)
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)

    logger = Logger(run_dir / "runner.log")
    ctx = PipelineContext(
        args=args,
        repo_root=repo_root,
        workspace_root=workspace_root,
        run_dir=run_dir,
        logger=logger,
        summary_path=run_dir / "summary.json",
    )
    return ctx


def selected_stages(start_stage: str, end_stage: str) -> list[str]:
    start_index = STAGES.index(start_stage)
    end_index = STAGES.index(end_stage)
    if end_index < start_index:
        raise ValueError("--end-stage must not come before --start-stage")
    return STAGES[start_index: end_index + 1]


def main() -> None:
    args = parse_args()
    ctx = build_context(args)
    ctx.logger.log(f"run_dir: {ctx.run_dir}")
    ctx.logger.log(f"video_name: {args.video_name}")
    ctx.logger.log(f"stage_range: {args.start_stage} -> {args.end_stage}")
    ctx.logger.log(f"worker_gpu: {args.worker_gpu}")
    ctx.logger.log(f"ollama_gpu: {args.ollama_gpu}")

    maybe_start_ollama(ctx)

    for stage_name in selected_stages(args.start_stage, args.end_stage):
        done_before = stage_done(ctx.repo_root, args.video_name, stage_name)
        marker_before = matches_for_stage(ctx.repo_root, args.video_name, stage_name)
        if done_before and not args.force:
            marker_str = str(marker_before[0]) if marker_before else "existing output"
            ctx.logger.log(f"{stage_name}: skipping existing output -> {marker_str}")
            ctx.stage_results[stage_name] = {"status": "skipped", "marker": marker_str}
            update_summary(ctx)
            continue

        command, env_overrides = stage_command(ctx, stage_name)
        run_command(ctx, stage_name, command, env_overrides=env_overrides)

        marker_after = matches_for_stage(ctx.repo_root, args.video_name, stage_name)
        marker_str = str(marker_after[0]) if marker_after else ""
        ctx.stage_results[stage_name] = {"status": "completed", "marker": marker_str}
        update_summary(ctx)

    final_markers = matches_for_stage(
        ctx.repo_root,
        args.video_name,
        args.end_stage,
    )
    if args.end_stage in {"track_human_object_mesh", "track_human_object_joint"} and not final_markers:
        raise RuntimeError(
            f"Final stage finished but {args.end_stage} "
            "run_summary.json was not found."
        )

    ctx.logger.log("pipeline: completed requested stage range")
    if final_markers:
        ctx.logger.log(f"final_run_summary: {final_markers[0]}")


if __name__ == "__main__":
    main()
