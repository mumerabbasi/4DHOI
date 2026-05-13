from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


F_MM_AUTO = "auto"
FULL_FRAME_SENSOR_WIDTH_MM = 36.0
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_f_mm_arg(raw: str) -> int | None | str:
    value = raw.strip().lower()
    if value in {"none", "null"}:
        return None
    if value == F_MM_AUTO:
        return F_MM_AUTO
    return int(value)


def resolve_scannet_root(
    script_dir: Path,
    raw_scannet_root: str | None,
) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_paths(
    scannet_root: Path,
    scene_context: dict[str, Any],
) -> dict[str, Path]:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )

    image_rel, transforms_rel = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    scene_root = scannet_root / scene_id
    return {
        "image_path": scene_root / image_rel / camera_name,
        "transforms_path": scene_root / transforms_rel,
        "colmap_images_path": scene_root / "dslr" / "colmap" / "images.txt",
    }


def resolve_f_mm(
    transforms_path: Path,
    cli_f_mm: int | None | str,
) -> int | None:
    if cli_f_mm != F_MM_AUTO:
        return cli_f_mm
    camera = load_json(transforms_path)
    width_px = int(camera["w"])
    fx_px = float(camera["fl_x"])
    return int(round(fx_px * (FULL_FRAME_SENSOR_WIDTH_MM / width_px)))


def colmap_qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.astype(np.float64)
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float32,
    )


def load_colmap_pose(
    colmap_images_path: Path,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    for line in colmap_images_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[-1] != camera_name:
            continue
        qvec = np.asarray(list(map(float, parts[1:5])), dtype=np.float32)
        tvec = np.asarray(list(map(float, parts[5:8])), dtype=np.float32)
        return colmap_qvec_to_rotmat(qvec), tvec
    raise ValueError(
        f"Could not find camera '{camera_name}' in {colmap_images_path}"
    )


def camera_to_world_points(
    points_camera: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    translation_world_to_camera: np.ndarray,
) -> np.ndarray:
    return (
        points_camera - translation_world_to_camera.reshape(1, 3)
    ) @ rotation_world_to_camera


def create_repeat_frame_clip(
    image_path: Path,
    output_video_path: Path,
    num_frames: int,
    fps: float,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    height, width = image_bgr.shape[:2]
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_video_path}")
    try:
        for _ in range(int(num_frames)):
            writer.write(image_bgr)
    finally:
        writer.release()


def write_ascii_ply(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def ensure_gvhmr_on_path(gvhmr_path: Path) -> None:
    gvhmr_root = str(gvhmr_path.resolve())
    if gvhmr_root not in sys.path:
        sys.path.insert(0, gvhmr_root)


def build_first_frame_smplx_vertices(
    result_path: Path,
    gvhmr_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    ensure_gvhmr_on_path(gvhmr_path)

    from hmr4d.utils.smplx_utils import make_smplx

    result = torch.load(result_path, map_location="cpu")
    params = {
        key: value[:1]
        for key, value in result["smpl_params_incam"].items()
        if isinstance(value, torch.Tensor)
    }
    model = make_smplx("supermotion")
    model.eval()
    with torch.no_grad():
        output = model(**params)

    vertices = output.vertices[0].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, faces


def export_first_frame_world_mesh(
    result_path: Path,
    mesh_path: Path,
    gvhmr_path: Path,
    scene_paths: dict[str, Path],
    scene_context: dict[str, Any],
) -> Path:
    vertices_camera, faces = build_first_frame_smplx_vertices(
        result_path=result_path,
        gvhmr_path=gvhmr_path,
    )
    rotation_world_to_camera, translation_world_to_camera = load_colmap_pose(
        scene_paths["colmap_images_path"],
        scene_context["camera"]["name"],
    )
    vertices_world = camera_to_world_points(
        points_camera=vertices_camera,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
    )
    write_ascii_ply(mesh_path, vertices_world, faces)
    return mesh_path


def build_subprocess_env(gvhmr_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{gvhmr_path}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(gvhmr_path)
    return env


def build_gvhmr_demo_cmd(
    demo_script: Path,
    video_path: Path,
    output_root: Path,
    resolved_f_mm: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(demo_script),
        "--video",
        str(video_path),
        "--output_root",
        str(output_root),
        "-s",
    ]
    if resolved_f_mm is not None:
        cmd.append(f"--f_mm={resolved_f_mm}")
    return cmd


def clear_existing_gvhmr_files(output_root: Path) -> None:
    for path in output_root.iterdir():
        if path.name in {
            "0_input_video.mp4",
            "1_incam.mp4",
            "2_global.mp4",
            "3_incam_global_horiz.mp4",
            "hmr4d_results.pt",
            "humans",
            "metadata.json",
            "person_1_first_frame_smplx_world.json",
            "person_1_first_frame_smplx_world.ply",
            "preprocess",
            "static_clip.mp4",
        } or path.name.endswith("_3_incam_global_horiz.mp4"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def finalize_gvhmr_output(src_dir: Path, dst_dir: Path) -> Path:
    if not src_dir.exists():
        raise FileNotFoundError(f"Expected GVHMR output directory not found: {src_dir}")
    result_path = src_dir / "hmr4d_results.pt"
    if not result_path.exists():
        raise FileNotFoundError(f"Expected GVHMR result file not found: {result_path}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    clear_existing_gvhmr_files(dst_dir)
    for path in src_dir.iterdir():
        name = path.name
        if name.endswith("_3_incam_global_horiz.mp4"):
            name = "3_incam_global_horiz.mp4"
        shutil.move(str(path), str(dst_dir / name))
    return dst_dir


def run_gvhmr(
    video_path: Path,
    temp_output_root: Path,
    final_output_root: Path,
    gvhmr_path: Path,
    resolved_f_mm: int | None,
) -> Path:
    demo_script = gvhmr_path / "tools" / "demo" / "demo.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"GVHMR demo script not found: {demo_script}")
    temp_output_root.mkdir(parents=True, exist_ok=True)
    cmd = build_gvhmr_demo_cmd(
        demo_script=demo_script,
        video_path=video_path,
        output_root=temp_output_root,
        resolved_f_mm=resolved_f_mm,
    )
    subprocess.run(
        cmd,
        cwd=str(gvhmr_path),
        check=True,
        env=build_subprocess_env(gvhmr_path),
    )
    return finalize_gvhmr_output(
        temp_output_root / video_path.stem,
        final_output_root,
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Estimate a static one-frame human pose with GVHMR."
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--human-frame-root", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--gvhmr_path", default=str(project_dir.parent / "GVHMR"))
    parser.add_argument("--num-repeat-frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--f_mm", type=parse_f_mm_arg, default=F_MM_AUTO)
    args = parser.parse_args()

    human_frame_root = (
        Path(args.human_frame_root).resolve()
        if args.human_frame_root
        else project_dir / "03_Generate_Human_Frame" / "output" / args.interaction_name
    )
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else project_dir / "01_Generate_SIG" / "input_prompts" / args.interaction_name
    )
    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name
    )
    gvhmr_path = Path(args.gvhmr_path).resolve()
    inpainted_frame = human_frame_root / "inpainted_frame.png"
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)
    input_scene_path = input_dir / "input_scene.json"
    input_payload = load_json(input_scene_path)
    scene_paths = resolve_scene_paths(
        scannet_root,
        input_payload["scene_context"],
    )
    scene_context = input_payload["scene_context"]
    resolved_f_mm = resolve_f_mm(scene_paths["transforms_path"], args.f_mm)

    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="4dhsi_static_pose_") as tmp_dir:
        temp_root = Path(tmp_dir)
        static_clip_path = temp_root / "input.mp4"
        temp_gvhmr_output = temp_root / "gvhmr_output"
        create_repeat_frame_clip(
            image_path=inpainted_frame,
            output_video_path=static_clip_path,
            num_frames=int(args.num_repeat_frames),
            fps=float(args.fps),
        )
        run_gvhmr(
            video_path=static_clip_path,
            temp_output_root=temp_gvhmr_output,
            final_output_root=output_root,
            gvhmr_path=gvhmr_path,
            resolved_f_mm=resolved_f_mm,
        )
    mesh_path = export_first_frame_world_mesh(
        result_path=output_root / "hmr4d_results.pt",
        mesh_path=output_root / "first_frame_smplx_world.ply",
        gvhmr_path=gvhmr_path,
        scene_paths=scene_paths,
        scene_context=scene_context,
    )

    print(f"Wrote GVHMR result: {output_root}")
    print(f"Wrote first-frame SMPL-X world mesh: {mesh_path}")


if __name__ == "__main__":
    main()
