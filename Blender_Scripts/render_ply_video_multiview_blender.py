"""Render a stitched three-view video from animated PLY hierarchies.

Run this file from Blender's Text Editor with "Run Script".

Before running, set `VIDEO_DIR` to a directory like:
    video_xx/<sequence>/meshes/*.ply
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


# ---------------------------------------------------------------------------
# Blender-side configuration
# ---------------------------------------------------------------------------

VIDEO_DIR = r"D:\MSCE\Thesis\4DHOI\current\output_final\video_01"
ROOT_COLLECTION_NAME = ""
OUTPUT_VIDEO_NAME = "multiview_render.mp4"

FPS = None
AUTO_DETECT_FPS = False
SAMPLES = 32
ENGINE = None
FRONT_BACKOFF = None
TOP_DISTANCE_SCALE = 2.2
PERSPECTIVE_DISTANCE_SCALE = 2.0
KEEP_TEMP_FRAMES = False

# On Windows, set these to your ffmpeg binaries if they are not on PATH.
# Example:
# FFMPEG_EXE = r"C:\ffmpeg\bin\ffmpeg.exe"
# FFPROBE_EXE = r"C:\ffmpeg\bin\ffprobe.exe"
FFMPEG_EXE = r"C:\Users\umerh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
FFPROBE_EXE = r"C:\Users\umerh\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1-full_build\bin\ffprobe.exe"

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


def _resolve_executable(executable: str, label: str) -> str:
    candidate = str(executable).strip()
    if not candidate:
        raise ValueError(f"{label} executable is empty.")

    expanded = Path(bpy.path.abspath(candidate)).expanduser()
    if expanded.exists():
        return str(expanded.resolve())

    found = shutil.which(candidate)
    if found:
        return found

    raise FileNotFoundError(
        f"Could not find {label} executable: {candidate}. "
        f"Set {label.upper()}_EXE at the top of this script."
    )


def _extract_index(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 10**18


def _natural_key(name: str) -> list[int | str]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _resolve_video_dir() -> Path:
    if not VIDEO_DIR:
        raise ValueError("Set VIDEO_DIR at the top of the script.")

    resolved_video_dir = (
        Path(bpy.path.abspath(VIDEO_DIR)).expanduser().resolve()
    )
    if not resolved_video_dir.is_dir():
        raise NotADirectoryError(
            f"Video directory not found: {resolved_video_dir}"
        )
    return resolved_video_dir


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    for datablock_iter in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.images,
        bpy.data.lights,
        bpy.data.actions,
    ):
        for datablock in list(datablock_iter):
            if datablock.users == 0:
                datablock_iter.remove(datablock)


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
    raise RuntimeError(
        "No PLY importer found (wm.ply_import or import_mesh.ply)."
    )


def _find_sequences(video_dir: Path) -> list[tuple[str, list[Path]]]:
    sequences = []

    for child in sorted(
        video_dir.iterdir(),
        key=lambda path: _natural_key(path.name),
    ):
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


def _apply_material(
    obj: bpy.types.Object,
    mat: bpy.types.Material,
) -> None:
    if obj.type != "MESH":
        return

    if obj.data.materials:
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = mat
    else:
        obj.data.materials.append(mat)


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


def _apply_camera_intrinsics(
    camera_obj: bpy.types.Object,
    payload: dict | None,
) -> None:
    camera = camera_obj.data
    recommendation = (
        payload.get("blender_recommendation", {})
        if payload
        else {}
    )

    camera.sensor_fit = str(recommendation.get("sensor_fit", "HORIZONTAL"))
    camera.sensor_width = float(
        recommendation.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM)
    )
    camera.sensor_height = float(
        recommendation.get("sensor_height_mm", DEFAULT_SENSOR_HEIGHT_MM)
    )
    camera.lens = float(recommendation.get("lens_mm", 35.0))


def _probe_video_fps(video_path: Path) -> float:
    ffprobe_exe = _resolve_executable(FFPROBE_EXE, "ffprobe")
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


def _discover_fps(video_dir: Path) -> float:
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
            fps = _probe_video_fps(resolved)
            print(f"Detected FPS {fps:.6f} from: {resolved}")
            return fps
        except Exception as exc:
            print(f"Warning: failed to probe FPS from {resolved}: {exc}")

    print(f"Falling back to default FPS: {DEFAULT_FPS}")
    return DEFAULT_FPS


def _setup_render_settings(
    width: int,
    height: int,
    samples: int,
    engine: str | None,
) -> None:
    scene = bpy.context.scene

    if engine is None:
        engine_items = (
            bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
        )
        available_engines = {item.identifier for item in engine_items}
        if "BLENDER_EEVEE_NEXT" in available_engines:
            engine = "BLENDER_EEVEE_NEXT"
        elif "BLENDER_EEVEE" in available_engines:
            engine = "BLENDER_EEVEE"
        else:
            engine = "CYCLES"

    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False

    if engine == "CYCLES":
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            if hasattr(prefs, "compute_device_type"):
                prefs.compute_device_type = "CUDA"
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
    scene.render.use_file_extension = True

    _setup_background()
    _setup_lighting()


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
        light_obj.rotation_euler = tuple(
            math.radians(value) for value in rotation_deg
        )


def _get_scene_bounds(
    objects: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
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
        "Front view: camera=(%.4f, %.4f, %.4f), "
        "target axis=+Z, scene center=(%.4f, %.4f, %.4f)"
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
    fps: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = _resolve_executable(FFMPEG_EXE, "ffmpeg")

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
        "-filter_complex",
        "[0:v][1:v][2:v]hstack=inputs=3[v]",
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


def _import_ply_video_hierarchies(video_dir: Path) -> list[bpy.types.Object]:
    sequences = _find_sequences(video_dir)
    if not sequences:
        raise FileNotFoundError(
            "No sequences found under "
            f"{video_dir}. Expected video_xx/<sequence>/meshes/*.ply"
        )

    root_name = ROOT_COLLECTION_NAME.strip() or video_dir.name
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

        sequence_collection = bpy.data.collections.new(
            sequence_collection_name
        )
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
                        f"{sequence_name}_frame_{frame_index:04d}_"
                        f"{mesh_index:02d}"
                    )

                for collection in list(obj.users_collection):
                    collection.objects.unlink(obj)
                sequence_collection.objects.link(obj)

                obj.parent = sequence_empty
                obj.matrix_parent_inverse = (
                    sequence_empty.matrix_world.inverted()
                )
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
                obj.keyframe_insert(
                    data_path="hide_viewport",
                    frame=frame_num + 1,
                )
                obj.keyframe_insert(
                    data_path="hide_render",
                    frame=frame_num + 1,
                )

                imported_object_count += 1
                imported_meshes.append(obj)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max_frame_count
    scene.frame_current = 1

    print(f"Imported {imported_object_count} mesh objects into '{root_name}'.")
    print(f"Timeline set to frames 1..{max_frame_count}.")
    return imported_meshes


def main() -> None:
    video_dir = _resolve_video_dir()
    output_path = (video_dir / OUTPUT_VIDEO_NAME).resolve()
    temp_root = video_dir / "_multiview_render_tmp"

    _clear_scene()

    intrinsics_path = _discover_intrinsics_path(video_dir)
    intrinsics_payload = (
        _load_intrinsics_payload(intrinsics_path)
        if intrinsics_path
        else None
    )
    render_width, render_height = _extract_render_resolution(
        intrinsics_payload
    )
    if FPS is not None:
        fps = float(FPS)
    elif AUTO_DETECT_FPS:
        fps = _discover_fps(video_dir)
    else:
        fps = DEFAULT_FPS

    print(f"Output video: {output_path}")
    print(f"Render resolution: {render_width}x{render_height}")
    print(f"FPS: {fps:.6f}")
    if intrinsics_path is not None:
        print(f"Camera intrinsics: {intrinsics_path}")
    else:
        print(
            "Camera intrinsics: not found, using default "
            "35mm full-frame camera"
        )

    imported_meshes = _import_ply_video_hierarchies(video_dir)
    if not imported_meshes:
        raise RuntimeError("No mesh objects were imported.")

    scene_stats = _scene_stats(imported_meshes)
    front_backoff = (
        float(FRONT_BACKOFF)
        if FRONT_BACKOFF is not None
        else max(
            0.5,
            float(scene_stats["radius"]) * 0.30,
            abs(float(scene_stats["min_corner"].z)) * 0.05,
        )
    )

    print(
        "Scene bounds: min=(%.4f, %.4f, %.4f), "
        "max=(%.4f, %.4f, %.4f), radius=%.4f"
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

    _setup_render_settings(render_width, render_height, SAMPLES, ENGINE)

    front_camera = _make_camera("FrontCamera", intrinsics_payload)
    _configure_front_camera(front_camera, scene_stats, front_backoff)

    top_camera = _make_camera("TopCamera", intrinsics_payload)
    _configure_top_camera(top_camera, scene_stats, TOP_DISTANCE_SCALE)

    perspective_camera = _make_camera("PerspectiveCamera", intrinsics_payload)
    _configure_perspective_camera(
        perspective_camera,
        scene_stats,
        PERSPECTIVE_DISTANCE_SCALE,
    )

    front_dir = temp_root / "front"
    top_dir = temp_root / "top"
    perspective_dir = temp_root / "perspective"

    print("Rendering front view...")
    _render_animation_from_camera(front_camera, front_dir)
    print("Rendering top view...")
    _render_animation_from_camera(top_camera, top_dir)
    print("Rendering three-quarter view...")
    _render_animation_from_camera(perspective_camera, perspective_dir)

    print("Stitching three views with ffmpeg...")
    _stitch_videos_from_png_sequences(
        front_dir,
        top_dir,
        perspective_dir,
        fps,
        output_path,
    )

    if not KEEP_TEMP_FRAMES and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Saved stitched multiview video to: {output_path}")


if __name__ == "__main__":
    main()
