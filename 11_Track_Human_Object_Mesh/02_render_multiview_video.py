"""Headless Blender multiview renderer for tracked human/object mesh videos.

Run this script with Blender in background mode:

    blender --background --python render_multiview_video.py -- \
        --video_name video_01

It imports every `<sequence>/meshes/*.ply` under
`Track_Human_Object_Mesh/output/<video_name>`, renders the four standard
views, and stitches them into one MP4.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


SEQUENCE_COLORS = [
    (0.95, 0.08, 0.08, 1.0),
    (0.00, 0.28, 1.00, 1.0),
    (0.00, 0.78, 0.16, 1.0),
    (1.00, 0.58, 0.00, 1.0),
    (0.82, 0.00, 0.82, 1.0),
    (0.00, 0.88, 0.88, 1.0),
]

DEFAULT_SENSOR_WIDTH_MM = 36.0
DEFAULT_SENSOR_HEIGHT_MM = 24.0
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_FPS = 24.0
DEFAULT_OUTPUT_NAME = "multiview_render.mp4"


def _default_executable(name: str) -> str:
    preferred = Path("/usr/bin") / name
    if preferred.exists():
        return str(preferred)
    return name


def _extract_cli_args(argv: list[str] | None = None) -> list[str]:
    argv = sys.argv if argv is None else argv
    if "--" in argv:
        return argv[argv.index("--") + 1:]
    return argv[1:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render Track_Human_Object_Mesh output/<video_name> into a stitched "
            "four-view MP4 using Blender in background mode."
        )
    )
    parser.add_argument("--video_name", type=str, default="video_01")
    parser.add_argument(
        "--track_output_dir",
        type=str,
        default=None,
        help=(
            "Video directory containing per-sequence meshes "
            "(default: ./output/<video_name>, relative to this script)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help=(
            "Output root used when --track_output_dir is not provided "
            "(default: ./output, relative to this script)."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Explicit output MP4 path. Default: "
            "<track_output_dir>/multiview_render.mp4."
        ),
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output filename when --output_path is omitted (default: {DEFAULT_OUTPUT_NAME}).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24,
        help=(
            "Output FPS. When omitted, auto-detect from nearby MP4s if enabled; "
            f"otherwise fallback to {DEFAULT_FPS}."
        ),
    )
    parser.add_argument(
        "--auto_detect_fps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect FPS from overlay/source videos when --fps is omitted.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Render samples per frame (default: 32).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="AUTO",
        help=(
            "Blender render engine. Default AUTO picks the best available "
            "engine in this order: BLENDER_EEVEE_NEXT, BLENDER_EEVEE, CYCLES. "
            "You can also force a specific engine."
        ),
    )
    parser.add_argument(
        "--front_backoff",
        type=float,
        default=2.5,
        help="Front camera distance along negative Z (default: 2.5).",
    )
    parser.add_argument(
        "--top_distance_scale",
        type=float,
        default=3.2,
        help="Top view distance scale relative to scene radius (default: 3.2).",
    )
    parser.add_argument(
        "--perspective_distance_scale",
        type=float,
        default=3.0,
        help=(
            "Perspective view distance scale relative to scene radius "
            "(default: 3.0)."
        ),
    )
    parser.add_argument(
        "--resolution_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to the inferred render resolution. "
            "Use <1.0 for faster test renders."
        ),
    )
    parser.add_argument(
        "--root_collection_name",
        type=str,
        default="",
        help="Optional custom root collection name inside Blender.",
    )
    parser.add_argument(
        "--keep_temp_frames",
        action="store_true",
        help="Keep the temporary per-view PNG sequences after stitching.",
    )
    parser.add_argument(
        "--ffmpeg_executable",
        type=str,
        default=_default_executable("ffmpeg"),
        help="ffmpeg executable used for stitching the final MP4.",
    )
    parser.add_argument(
        "--ffprobe_executable",
        type=str,
        default=_default_executable("ffprobe"),
        help="ffprobe executable used for FPS auto-detection.",
    )
    return parser.parse_args(argv)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_video_dir(args: argparse.Namespace, script_dir: Path) -> Path:
    if args.track_output_dir:
        video_dir = resolve_path(args.track_output_dir, script_dir)
    else:
        output_root = resolve_path(args.output_dir, script_dir)
        video_dir = (output_root / args.video_name).resolve()

    if not video_dir.is_dir():
        raise NotADirectoryError(f"Track output directory not found: {video_dir}")
    return video_dir


def resolve_output_path(
    args: argparse.Namespace, video_dir: Path, script_dir: Path
) -> Path:
    if args.output_path:
        return resolve_path(args.output_path, script_dir)
    return (video_dir / args.output_name).resolve()


def _resolve_executable(executable: str, label: str) -> str:
    candidate = str(executable).strip()
    if not candidate:
        raise ValueError(f"{label} executable is empty.")

    expanded = Path(candidate).expanduser()
    if expanded.exists():
        return str(expanded.resolve())

    found = shutil.which(candidate)
    if found:
        return found

    raise FileNotFoundError(f"Could not find {label} executable: {candidate}")


def _extract_index(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 10**18


def _natural_key(name: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _load_intrinsics_payload(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _is_project_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "Generate_Object_Mesh").is_dir()
        and (path / "Estimate_Depth").is_dir()
        and (path / "Blender_Scripts").is_dir()
    )


def _discover_project_dir(video_dir: Path) -> Path | None:
    for candidate in [video_dir.resolve(), *video_dir.resolve().parents]:
        if _is_project_root(candidate):
            return candidate
    return None


def _discover_intrinsics_path(video_dir: Path) -> Path | None:
    project_dir = _discover_project_dir(video_dir)
    video_name = video_dir.name

    candidates = [video_dir / "camera_intrinsics.json"]
    if project_dir is not None:
        candidates.extend(
            [
                project_dir
                / "Generate_Object_Mesh"
                / "output"
                / video_name
                / "camera_intrinsics.json",
                project_dir
                / "Estimate_Depth"
                / "output"
                / video_name
                / "camera_intrinsics.json",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _round_positive_int(value: float, fallback: int) -> int:
    try:
        resolved = int(round(float(value)))
    except Exception:
        return fallback
    return resolved if resolved > 0 else fallback


def _extract_render_resolution(payload: dict | None) -> tuple[int, int]:
    if not payload:
        return DEFAULT_RESOLUTION

    intrinsics = payload.get("intrinsics_pixels_3x3")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3:
        return DEFAULT_RESOLUTION

    try:
        cx = float(intrinsics[0][2])
        cy = float(intrinsics[1][2])
    except Exception:
        return DEFAULT_RESOLUTION

    width = _round_positive_int(cx * 2.0, DEFAULT_RESOLUTION[0])
    height = _round_positive_int(cy * 2.0, DEFAULT_RESOLUTION[1])
    return width, height


def _apply_resolution_scale(
    resolution: tuple[int, int], scale: float
) -> tuple[int, int]:
    if scale <= 0:
        raise ValueError("--resolution_scale must be > 0")
    width, height = resolution
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def _probe_video_fps(video_path: Path, ffprobe_exe: str) -> float:
    cmd = [
        ffprobe_exe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    raw_fps = result.stdout.strip()
    if not raw_fps:
        raise RuntimeError(f"Could not determine FPS for video: {video_path}")
    return float(Fraction(raw_fps))


def _discover_fps(video_dir: Path, ffprobe_exe: str) -> float:
    project_dir = _discover_project_dir(video_dir)
    video_name = video_dir.name

    candidates = [video_dir / "overlay.mp4"]
    candidates.extend(sorted(video_dir.glob("*.mp4")))

    if project_dir is not None:
        candidates.extend(
            sorted(
                (project_dir / "Generate_Video" / "output" / video_name).glob(
                    "*.mp4"
                )
            )
        )
        candidates.extend(
            sorted(
                (
                    project_dir
                    / "Estimate_Human_Motion"
                    / "output"
                    / video_name
                ).glob("humans/*/0_input_video.mp4")
            )
        )

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            fps = _probe_video_fps(resolved, ffprobe_exe)
            print(f"Detected FPS {fps:.6f} from: {resolved}")
            return fps
        except Exception as exc:
            print(f"Warning: failed to probe FPS from {resolved}: {exc}")

    print(f"Falling back to default FPS: {DEFAULT_FPS}")
    return DEFAULT_FPS


def _find_sequences(video_dir: Path) -> list[tuple[str, list[Path]]]:
    sequences = []

    for child in sorted(video_dir.iterdir(), key=lambda path: _natural_key(path.name)):
        if not child.is_dir():
            continue

        meshes_dir = child / "meshes"
        if not meshes_dir.is_dir():
            continue

        ply_files = sorted(
            [
                path
                for path in meshes_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".ply"
            ],
            key=lambda path: _extract_index(path.name),
        )
        if ply_files:
            sequences.append((child.name, ply_files))

    return sequences


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)

    for scene in bpy.data.scenes:
        scene.world = None

    datablock_names = [
        "meshes",
        "materials",
        "cameras",
        "images",
        "lights",
        "actions",
        "armatures",
        "curves",
        "grease_pencils",
        "node_groups",
        "worlds",
    ]
    for datablock_name in datablock_names:
        datablock_iter = getattr(bpy.data, datablock_name, None)
        if datablock_iter is None:
            continue
        for datablock in list(datablock_iter):
            if datablock.users == 0:
                datablock_iter.remove(datablock)

    if hasattr(bpy.ops.outliner, "orphans_purge"):
        for _ in range(3):
            result = bpy.ops.outliner.orphans_purge(
                do_local_ids=True,
                do_linked_ids=True,
                do_recursive=True,
            )
            if result != {"FINISHED"}:
                break


def _delete_collection_tree(collection_name: str) -> None:
    root = bpy.data.collections.get(collection_name)
    if root is None:
        return

    collections = []
    objects = []
    seen_objects = set()

    def _walk(collection: bpy.types.Collection) -> None:
        collections.append(collection)
        for obj in collection.objects:
            if obj.name in seen_objects:
                continue
            seen_objects.add(obj.name)
            objects.append(obj)
        for child in collection.children:
            _walk(child)

    _walk(root)

    for obj in objects:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    for collection in reversed(collections):
        if collection.name in bpy.data.collections:
            bpy.data.collections.remove(collection)


def _gather_mesh_objects(
    imported_objects: list[bpy.types.Object],
) -> list[bpy.types.Object]:
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    if meshes:
        return meshes

    meshes = []
    for obj in imported_objects:
        for child in obj.children_recursive:
            if child.type == "MESH":
                meshes.append(child)
    return meshes


def _import_ply(filepath: str) -> None:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=filepath)
        return
    if hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=filepath)
        return
    raise RuntimeError("No PLY importer found (wm.ply_import or import_mesh.ply).")


def _make_empty(name: str, parent=None) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.25
    empty.parent = parent
    if parent is not None:
        empty.matrix_parent_inverse = parent.matrix_world.inverted()
    return empty


def _get_or_create_material(name: str, rgba) -> bpy.types.Material:
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = rgba
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.55
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.35
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.35

    return mat


def _apply_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return

    if obj.data.materials:
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = mat
    else:
        obj.data.materials.append(mat)


def _import_ply_video_hierarchies(
    video_dir: Path, root_collection_name: str
) -> list[bpy.types.Object]:
    sequences = _find_sequences(video_dir)
    if not sequences:
        raise FileNotFoundError(
            "No sequences found under "
            f"{video_dir}. Expected <sequence>/meshes/*.ply."
        )

    root_name = root_collection_name.strip() or video_dir.name
    root_empty_name = f"{root_name}::root"

    print(f"Video directory: {video_dir}")
    print(f"Found {len(sequences)} sequences:")
    for sequence_name, ply_files in sequences:
        print(f"  - {sequence_name}: {len(ply_files)} frames")

    _delete_collection_tree(root_name)

    root_collection = bpy.data.collections.new(root_name)
    bpy.context.scene.collection.children.link(root_collection)

    root_empty = _make_empty(root_empty_name)
    root_collection.objects.link(root_empty)

    max_frame_count = 1
    imported_object_count = 0
    imported_meshes = []

    for sequence_index, (sequence_name, ply_files) in enumerate(sequences):
        sequence_collection_name = f"{root_name}::{sequence_name}"
        sequence_empty_name = f"{root_name}::{sequence_name}::root"
        material_name = f"{root_name}_{sequence_name}_mat"
        material_color = SEQUENCE_COLORS[sequence_index % len(SEQUENCE_COLORS)]
        material = _get_or_create_material(material_name, material_color)

        sequence_collection = bpy.data.collections.new(sequence_collection_name)
        root_collection.children.link(sequence_collection)

        sequence_empty = _make_empty(sequence_empty_name, parent=root_empty)
        sequence_collection.objects.link(sequence_empty)

        for frame_index, ply_path in enumerate(ply_files):
            bpy.ops.object.select_all(action="DESELECT")
            _import_ply(str(ply_path))

            imported = list(bpy.context.selected_objects)
            mesh_objects = _gather_mesh_objects(imported)
            if not mesh_objects:
                print(f"Skipping {ply_path}: no mesh objects imported.")
                continue

            frame_num = frame_index + 1
            max_frame_count = max(max_frame_count, frame_num)

            for mesh_index, obj in enumerate(mesh_objects):
                if len(mesh_objects) == 1:
                    obj.name = f"{sequence_name}_frame_{frame_index:04d}"
                else:
                    obj.name = (
                        f"{sequence_name}_frame_{frame_index:04d}_{mesh_index:02d}"
                    )

                for collection in list(obj.users_collection):
                    collection.objects.unlink(obj)
                sequence_collection.objects.link(obj)

                obj.parent = sequence_empty
                obj.matrix_parent_inverse = sequence_empty.matrix_world.inverted()
                _apply_material(obj, material)

                if hasattr(obj.data, "use_auto_smooth"):
                    obj.data.use_auto_smooth = True

                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=0)
                obj.keyframe_insert(data_path="hide_render", frame=0)

                obj.hide_viewport = False
                obj.hide_render = False
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num)

                obj.hide_viewport = True
                obj.hide_render = True
                obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
                obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)

                imported_object_count += 1
                imported_meshes.append(obj)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max_frame_count
    scene.frame_current = 1

    print(f"Imported {imported_object_count} mesh objects into '{root_name}'.")
    print(f"Timeline set to frames 1..{max_frame_count}.")
    return imported_meshes


def _get_scene_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    min_vec = Vector((float("inf"), float("inf"), float("inf")))
    max_vec = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)
            min_vec.x = min(min_vec.x, world_corner.x)
            min_vec.y = min(min_vec.y, world_corner.y)
            min_vec.z = min(min_vec.z, world_corner.z)
            max_vec.x = max(max_vec.x, world_corner.x)
            max_vec.y = max(max_vec.y, world_corner.y)
            max_vec.z = max(max_vec.z, world_corner.z)

    if math.isinf(min_vec.x):
        zero = Vector((0.0, 0.0, 0.0))
        return zero, zero

    return min_vec, max_vec


def _scene_stats(mesh_objects: list[bpy.types.Object]) -> dict:
    min_corner, max_corner = _get_scene_bounds(mesh_objects)
    center = (min_corner + max_corner) * 0.5
    extents = max_corner - min_corner
    radius = max(extents.length * 0.5, 0.5)

    return {
        "min_corner": min_corner,
        "max_corner": max_corner,
        "center": center,
        "extents": extents,
        "radius": radius,
    }


def _apply_camera_intrinsics(
    camera_obj: bpy.types.Object,
    payload: dict | None,
) -> None:
    camera = camera_obj.data
    recommendation = payload.get("blender_recommendation", {}) if payload else {}

    camera.sensor_fit = str(recommendation.get("sensor_fit", "HORIZONTAL"))
    camera.sensor_width = float(
        recommendation.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM)
    )
    camera.sensor_height = float(
        recommendation.get("sensor_height_mm", DEFAULT_SENSOR_HEIGHT_MM)
    )
    camera.lens = float(recommendation.get("lens_mm", 35.0))


def _make_camera(name: str, payload: dict | None) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(name=name)
    camera_obj = bpy.data.objects.new(name, camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    _apply_camera_intrinsics(camera_obj, payload)
    return camera_obj


def _create_look_at_matrix(
    position: Vector,
    target: Vector,
    up_hint: Vector,
) -> Matrix:
    forward = (target - position).normalized()
    if forward.length < 1e-8:
        raise ValueError("Camera position and target cannot be identical.")

    right = forward.cross(up_hint)
    if right.length < 1e-8:
        fallback_up = Vector((0.0, 0.0, 1.0))
        if abs(forward.dot(fallback_up)) > 0.999:
            fallback_up = Vector((1.0, 0.0, 0.0))
        right = forward.cross(fallback_up)
    right.normalize()

    up = right.cross(forward)
    up.normalize()

    return Matrix(
        (
            (right.x, up.x, -forward.x, position.x),
            (right.y, up.y, -forward.y, position.y),
            (right.z, up.z, -forward.z, position.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def _configure_front_camera(
    camera_obj: bpy.types.Object,
    scene_stats: dict,
    backoff: float,
) -> None:
    center = scene_stats["center"]
    camera_obj.location = (0.0, 0.0, -float(backoff))
    camera_obj.rotation_euler = (math.radians(180.0), 0.0, 0.0)
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = max(
        1000.0,
        float(scene_stats["max_corner"].z + backoff + 100.0),
    )

    print(
        "Front view: camera=(%.4f, %.4f, %.4f), target axis=+Z, "
        "scene center=(%.4f, %.4f, %.4f)"
        % (
            camera_obj.location.x,
            camera_obj.location.y,
            camera_obj.location.z,
            center.x,
            center.y,
            center.z,
        )
    )


def _configure_top_camera(
    camera_obj: bpy.types.Object,
    scene_stats: dict,
    distance_scale: float,
) -> None:
    center = scene_stats["center"]
    radius = scene_stats["radius"]
    distance = max(radius * float(distance_scale), 1.0)
    position = Vector((center.x, center.y - distance, center.z))

    camera_obj.matrix_world = _create_look_at_matrix(
        position,
        center,
        Vector((0.0, 0.0, 1.0)),
    )
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = max(1000.0, distance * 4.0)


def _configure_perspective_camera(
    camera_obj: bpy.types.Object,
    scene_stats: dict,
    distance_scale: float,
) -> None:
    center = scene_stats["center"]
    radius = scene_stats["radius"]
    distance = max(radius * float(distance_scale), 1.0)
    position = center + Vector(
        (-0.85 * distance, -0.65 * distance, -0.95 * distance)
    )

    camera_obj.matrix_world = _create_look_at_matrix(
        position,
        center,
        Vector((0.0, -1.0, 0.0)),
    )
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = max(1000.0, distance * 4.0)


def _configure_opposite_perspective_camera(
    camera_obj: bpy.types.Object,
    scene_stats: dict,
    distance_scale: float,
) -> None:
    center = scene_stats["center"]
    radius = scene_stats["radius"]
    distance = max(radius * float(distance_scale), 1.0)
    position = center + Vector(
        (0.85 * distance, -0.65 * distance, 0.95 * distance)
    )

    camera_obj.matrix_world = _create_look_at_matrix(
        position,
        center,
        Vector((0.0, -1.0, 0.0)),
    )
    camera_obj.data.clip_start = 0.01
    camera_obj.data.clip_end = max(1000.0, distance * 4.0)


def _setup_background() -> None:
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")

    bpy.context.scene.world = world
    world.use_nodes = True

    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.96, 0.97, 0.99, 1.0)
        if "Strength" in background.inputs:
            background.inputs["Strength"].default_value = 0.9


def _setup_lighting() -> None:
    lights = [
        ("KeyLight", "SUN", 3.2, (0.0, -6.0, 5.0), (55.0, 0.0, 0.0)),
        ("FillLight", "SUN", 1.8, (5.0, -4.0, 4.0), (42.0, 18.0, 35.0)),
        ("RimLight", "SUN", 1.4, (-5.0, 5.0, 7.0), (50.0, 0.0, -135.0)),
    ]

    for name, light_type, energy, location, rotation_deg in lights:
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        if hasattr(light_data, "angle"):
            light_data.angle = math.radians(7.0)

        light_obj = bpy.data.objects.new(name, light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(math.radians(value) for value in rotation_deg)


def _setup_render_settings(
    width: int,
    height: int,
    samples: int,
    engine: str,
) -> None:
    scene = bpy.context.scene

    available_engines = {
        item.identifier
        for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    }
    requested_engine = (engine or "AUTO").strip().upper()

    if requested_engine in {"", "AUTO"}:
        for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
            if candidate in available_engines:
                requested_engine = candidate
                break
        else:
            raise ValueError(
                "No supported render engine is available. "
                f"Available: {sorted(available_engines)}"
            )
        print(f"Using auto-selected render engine: {requested_engine}")
    elif requested_engine not in available_engines:
        raise ValueError(
            f"Render engine '{requested_engine}' is unavailable. "
            f"Available: {sorted(available_engines)}"
        )

    scene.render.engine = requested_engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True

    if requested_engine == "CYCLES":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for compute_device_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                try:
                    prefs.compute_device_type = compute_device_type
                    break
                except Exception:
                    continue
            if hasattr(prefs, "get_devices"):
                prefs.get_devices()
            if hasattr(prefs, "devices"):
                for device in prefs.devices:
                    device.use = True
            scene.cycles.device = "GPU"
        except Exception:
            scene.cycles.device = "CPU"

        scene.cycles.samples = max(1, int(samples))
        scene.cycles.use_denoising = True
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = max(1, int(samples))
        if hasattr(scene.eevee, "use_gtao"):
            scene.eevee.use_gtao = True
        if hasattr(scene.eevee, "use_bloom"):
            scene.eevee.use_bloom = False

    scene.view_settings.exposure = 0.25

    _setup_background()
    _setup_lighting()


def _render_animation_from_camera(
    camera_obj: bpy.types.Object,
    output_dir: Path,
) -> None:
    scene = bpy.context.scene
    scene.camera = camera_obj
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_dir / "frame_")
    bpy.ops.render.render(animation=True)


def _stitch_videos_from_png_sequences(
    front_dir: Path,
    top_dir: Path,
    perspective_dir: Path,
    opposite_perspective_dir: Path,
    fps: float,
    output_path: Path,
    ffmpeg_exe: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        "1",
        "-i",
        str(front_dir / "frame_%04d.png"),
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        "1",
        "-i",
        str(top_dir / "frame_%04d.png"),
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        "1",
        "-i",
        str(perspective_dir / "frame_%04d.png"),
        "-framerate",
        f"{fps:.6f}",
        "-start_number",
        "1",
        "-i",
        str(opposite_perspective_dir / "frame_%04d.png"),
        "-filter_complex",
        (
            "[0:v][1:v]hstack=inputs=2[top];"
            "[2:v][3:v]hstack=inputs=2[bottom];"
            "[top][bottom]vstack=inputs=2[v]"
        ),
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _render_in_blender(args: argparse.Namespace, script_dir: Path) -> None:
    video_dir = resolve_video_dir(args, script_dir)
    output_path = resolve_output_path(args, video_dir, script_dir)
    temp_root = video_dir / "_multiview_render_tmp"

    ffmpeg_exe = _resolve_executable(args.ffmpeg_executable, "ffmpeg")
    ffprobe_exe = _resolve_executable(args.ffprobe_executable, "ffprobe")

    _clear_scene()
    if temp_root.exists():
        shutil.rmtree(temp_root)

    intrinsics_path = _discover_intrinsics_path(video_dir)
    intrinsics_payload = (
        _load_intrinsics_payload(intrinsics_path) if intrinsics_path else None
    )
    render_width, render_height = _apply_resolution_scale(
        _extract_render_resolution(intrinsics_payload),
        args.resolution_scale,
    )

    if args.fps is not None:
        fps = float(args.fps)
    elif args.auto_detect_fps:
        fps = _discover_fps(video_dir, ffprobe_exe)
    else:
        fps = DEFAULT_FPS

    print(f"Output video: {output_path}")
    print(f"Render resolution: {render_width}x{render_height}")
    print(f"FPS: {fps:.6f}")
    if intrinsics_path is not None:
        print(f"Camera intrinsics: {intrinsics_path}")
    else:
        print("Camera intrinsics: not found, using default 35mm full-frame camera")

    imported_meshes = _import_ply_video_hierarchies(
        video_dir, args.root_collection_name
    )
    if not imported_meshes:
        raise RuntimeError("No mesh objects were imported.")

    scene_stats = _scene_stats(imported_meshes)
    front_backoff = max(
        0.0,
        float(args.front_backoff),
    )

    print(
        "Scene bounds: min=(%.4f, %.4f, %.4f), max=(%.4f, %.4f, %.4f), radius=%.4f"
        % (
            scene_stats["min_corner"].x,
            scene_stats["min_corner"].y,
            scene_stats["min_corner"].z,
            scene_stats["max_corner"].x,
            scene_stats["max_corner"].y,
            scene_stats["max_corner"].z,
            scene_stats["radius"],
        )
    )
    print(f"Front camera backoff: {front_backoff:.4f}")

    _setup_render_settings(render_width, render_height, args.samples, args.engine)

    front_camera = _make_camera("FrontCamera", intrinsics_payload)
    _configure_front_camera(front_camera, scene_stats, front_backoff)

    top_camera = _make_camera("TopCamera", intrinsics_payload)
    _configure_top_camera(top_camera, scene_stats, args.top_distance_scale)

    perspective_camera = _make_camera("PerspectiveCamera", intrinsics_payload)
    _configure_perspective_camera(
        perspective_camera,
        scene_stats,
        args.perspective_distance_scale,
    )

    opposite_perspective_camera = _make_camera(
        "OppositePerspectiveCamera",
        intrinsics_payload,
    )
    _configure_opposite_perspective_camera(
        opposite_perspective_camera,
        scene_stats,
        args.perspective_distance_scale,
    )

    front_dir = temp_root / "front"
    top_dir = temp_root / "top"
    perspective_dir = temp_root / "perspective"
    opposite_perspective_dir = temp_root / "opposite_perspective"

    print("Rendering front view...")
    _render_animation_from_camera(front_camera, front_dir)
    print("Rendering top view...")
    _render_animation_from_camera(top_camera, top_dir)
    print("Rendering three-quarter view...")
    _render_animation_from_camera(perspective_camera, perspective_dir)
    print("Rendering opposite three-quarter view...")
    _render_animation_from_camera(
        opposite_perspective_camera,
        opposite_perspective_dir,
    )

    print("Stitching four views with ffmpeg...")
    _stitch_videos_from_png_sequences(
        front_dir,
        top_dir,
        perspective_dir,
        opposite_perspective_dir,
        fps,
        output_path,
        ffmpeg_exe,
    )

    if not args.keep_temp_frames and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Saved stitched multiview video to: {output_path}")


def main() -> None:
    cli_args = _extract_cli_args()
    args = parse_args(cli_args)
    script_dir = Path(__file__).resolve().parent
    _render_in_blender(args, script_dir)


if __name__ == "__main__":
    main()
