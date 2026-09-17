"""
Blender 4.2 script to render multiple views for PAG objects from generated meshes.

Renders RGB images and per-pixel face ID maps from multiple viewpoints.
Face IDs are stored as raw float values in EXR format for exact integer recovery.

Usage:
    blender --background --python 01_render_mesh_views.py -- --interaction_name interaction_01
"""

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Camera configuration
ORBIT_ANGLES_DEG = [10, 55, 100, 155, 190, 235, 280]  # horizontal orbit around object (Z-up)
HEIGHT_ANGLES_DEG = [10, 30, -15]  # level orbit, then up orbit, then down orbit
DEFAULT_CAMERA_DISTANCE = 0.05
DEFAULT_RESOLUTION = 1024
CAMERA_FRAME_MARGIN = 1.15

# Face ID attribute/AOV names
FACE_ID_ATTR = "face_id"
FACE_ID_AOV = "face_id"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments after '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description=(
            "Render all PAG objects from 04_Generate_Object_Mesh meshes into "
            "08_Segment_Object_Mesh objects/"
        )
    )
    parser.add_argument(
        "--interaction_name",
        type=str,
        default="interaction_01",
        help="Interaction name used to resolve default input paths.",
    )
    parser.add_argument(
        "--object_mesh_video_dir",
        type=str,
        default=None,
        help=(
            "Generated object-mesh interaction dir "
            "(default: ../04_Generate_Object_Mesh/output/<interaction_name>)."
        ),
    )
    parser.add_argument(
        "--pag_file",
        type=str,
        default=None,
        help=(
            "PAG JSON path (default: first output_pag_*.json in "
            "../01_Generate_PAG/output/<interaction_name>)."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./output",
        help="Output root (default: ./output, relative to this script).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Resolution of output images (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--camera_distance",
        type=float,
        default=None,
        help="Camera distance from center (default: auto-calculated).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Render with transparent background.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of render samples (default: 1 for accurate face IDs).",
    )

    return parser.parse_args(argv)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve path against base_dir when relative."""
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_default_dirs(
    args: argparse.Namespace, script_dir: Path
) -> tuple[Path, Path]:
    """Resolve generated object-mesh input dir and output interaction dir."""
    if args.object_mesh_video_dir is None:
        object_mesh_video_dir = (
            script_dir.parent / "04_Generate_Object_Mesh" / "output" / args.interaction_name
        ).resolve()
    else:
        object_mesh_video_dir = resolve_path(args.object_mesh_video_dir, script_dir)

    output_root = resolve_path(args.output_root, script_dir)
    output_video_dir = (output_root / args.interaction_name).resolve()
    return object_mesh_video_dir, output_video_dir


def resolve_pag_path(args: argparse.Namespace, script_dir: Path) -> Path:
    """Resolve PAG JSON path."""
    if args.pag_file is not None:
        pag_path = resolve_path(args.pag_file, script_dir)
        if not pag_path.exists():
            raise FileNotFoundError(f"PAG file not found: {pag_path}")
        return pag_path

    pag_dir = (script_dir.parent / "01_Generate_PAG" / "output" / args.interaction_name).resolve()
    if not pag_dir.exists():
        raise FileNotFoundError(f"PAG directory not found: {pag_dir}")

    pag_candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not pag_candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return pag_candidates[0]


def _sanitize_object_name(name: str) -> str:
    """Match the repo slug convention used across object-processing scripts."""
    return name.strip().replace(" ", "_").replace("-", "_")


def load_pag_objects_from_states_only(pag_path: Path) -> list[tuple[str, str]]:
    """Load unique objects from PAG['object states'] as (name, slug)."""
    with pag_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    object_states = payload.get("object states")
    if not isinstance(object_states, list):
        raise RuntimeError(
            "PAG must contain a list in 'object states'. "
            f"Got: {type(object_states).__name__}"
        )

    objects: list[tuple[str, str]] = []
    seen = set()
    for item in object_states:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        objects.append((name, _sanitize_object_name(name)))

    if not objects:
        raise RuntimeError(
            "No valid object names found in PAG 'object states'. "
            f"File: {pag_path}"
        )
    return objects


def resolve_object_mesh_path(
    object_slug: str,
    meshes_dir: Path,
) -> Path | None:
    """Resolve generated object mesh path for an object."""
    mesh_path = (meshes_dir / f"{object_slug}.ply").resolve()
    if mesh_path.exists():
        return mesh_path
    return None


def clear_scene() -> None:
    """Remove all objects from the scene and purge orphan data."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def import_mesh(filepath: str) -> list[bpy.types.Object]:
    """
    Import a mesh file preserving original transforms.

    Args:
        filepath: Path to the mesh file (.glb/.gltf/.ply).

    Returns:
        List of imported objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Mesh file not found: {filepath}")

    existing = set(bpy.data.objects)
    ext = Path(filepath).suffix.lower()
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".ply":
        imported = False
        exceptions: list[Exception] = []
        for importer in (
            lambda p: bpy.ops.import_mesh.ply(filepath=p),
            lambda p: bpy.ops.wm.ply_import(filepath=p),
        ):
            try:
                importer(filepath)
                imported = True
                break
            except Exception as exc:
                exceptions.append(exc)
        if not imported:
            raise RuntimeError(
                "Failed to import .ply mesh. Ensure PLY import operator is available."
            ) from exceptions[-1]
    else:
        raise ValueError(f"Unsupported mesh format: {ext}")

    new_objects = [obj for obj in bpy.data.objects if obj not in existing]

    print("\n=== Imported Objects ===")
    for obj in new_objects:
        if obj.type == "MESH":
            loc = obj.location
            print(f"  {obj.name}: loc=({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})")

    return new_objects


def apply_opencv_to_blender_transform(objects: list[bpy.types.Object]) -> None:
    """
    Convert imported objects from OpenCV to Blender coordinates.

    OpenCV camera coords: +X right, +Y down, +Z forward
    Blender world coords: +X right, +Y forward, +Z up
    """
    cv_to_blender = Matrix.Rotation(math.radians(-90.0), 4, "X")
    for obj in objects:
        obj.matrix_world = cv_to_blender @ obj.matrix_world

    bpy.context.view_layer.update()


def setup_vertex_color_materials(objects: list[bpy.types.Object]) -> None:
    """
    Configure materials to display vertex colors for SAM3D-exported meshes.

    SAM3D exports meshes with vertex colors instead of texture maps.
    This function creates appropriate material node setups.

    Args:
        objects: List of imported Blender objects.
    """
    for obj in objects:
        if obj.type != "MESH":
            continue

        mesh = obj.data
        color_attr_name = _get_color_attribute_name(mesh)

        if not color_attr_name:
            print(f"  {obj.name}: No vertex colors, skipping")
            continue

        if _has_texture_material(obj):
            print(f"  {obj.name}: Has texture material, keeping")
            continue

        _create_vertex_color_material(obj, color_attr_name)


def _get_color_attribute_name(mesh: bpy.types.Mesh) -> str | None:
    """Get the name of the first color attribute on a mesh."""
    if hasattr(mesh, "color_attributes") and mesh.color_attributes:
        return mesh.color_attributes[0].name
    if hasattr(mesh, "vertex_colors") and mesh.vertex_colors:
        return mesh.vertex_colors[0].name
    return None


def _has_texture_material(obj: bpy.types.Object) -> bool:
    """Check if object has a material with a texture image."""
    for mat in obj.data.materials or []:
        if mat and mat.use_nodes:
            for node in mat.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    return True
    return False


def _create_vertex_color_material(
    obj: bpy.types.Object, color_attr_name: str
) -> None:
    """Create a material that displays vertex colors."""
    mat = bpy.data.materials.new(name=f"{obj.name}_VertexColorMat")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create shader nodes
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (400, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)

    vertex_color = nodes.new(type="ShaderNodeVertexColor")
    vertex_color.location = (-200, 0)
    vertex_color.layer_name = color_attr_name

    # Connect nodes
    links.new(vertex_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Assign material
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    print(f"  {obj.name}: Created vertex color material")


def add_face_id_attribute(objects: list[bpy.types.Object]) -> None:
    """
    Add per-corner color attribute encoding polygon indices.

    Face IDs are stored as raw float values in the R channel of a FLOAT_COLOR
    attribute. This allows exact integer recovery up to 2^24 (~16M faces)
    when saved to 32-bit EXR.

    Args:
        objects: List of mesh objects to process.
    """
    print("\n=== Creating face_id attributes ===")

    for obj in objects:
        if obj.type != "MESH":
            continue

        mesh = obj.data
        _remove_existing_color_attr(mesh, FACE_ID_ATTR)

        # Create FLOAT_COLOR attribute (CORNER domain for per-loop storage)
        color_attr = mesh.color_attributes.new(
            name=FACE_ID_ATTR,
            type="FLOAT_COLOR",
            domain="CORNER",
        )

        # Fill each corner with its polygon's index
        for poly in mesh.polygons:
            face_id = float(poly.index)
            loop_end = poly.loop_start + poly.loop_total
            for loop_idx in range(poly.loop_start, loop_end):
                # Store face_id in R channel; G,B unused, A=1
                color_attr.data[loop_idx].color = (face_id, 0.0, 0.0, 1.0)

        poly_count = len(mesh.polygons)
        loop_count = len(mesh.loops)
        print(f"  {obj.name}: {poly_count} faces, {loop_count} corners")


def _remove_existing_color_attr(mesh: bpy.types.Mesh, name: str) -> None:
    """Remove a color attribute if it exists."""
    if hasattr(mesh, "color_attributes"):
        for attr in list(mesh.color_attributes):
            if attr.name == name:
                mesh.color_attributes.remove(attr)
                break


def setup_face_id_aov(objects: list[bpy.types.Object]) -> None:
    """
    Configure AOV output for face IDs.

    Creates a ViewLayer AOV and connects each material to output face IDs.

    Args:
        objects: List of mesh objects with face_id attributes.
    """
    # Add AOV to ViewLayer
    view_layer = bpy.context.scene.view_layers["ViewLayer"]
    if not any(aov.name == FACE_ID_AOV for aov in view_layer.aovs):
        aov = view_layer.aovs.add()
        aov.name = FACE_ID_AOV
        print(f"\n=== Added AOV: '{FACE_ID_AOV}' ===")

    # Attach AOV output to each material
    print("\n=== Configuring material AOV outputs ===")
    processed = set()

    for obj in objects:
        if obj.type != "MESH":
            continue

        for mat in obj.data.materials or []:
            if mat and mat.name not in processed:
                _attach_face_id_aov(mat)
                processed.add(mat.name)
                print(f"  Material '{mat.name}': AOV configured")


def _attach_face_id_aov(mat: bpy.types.Material) -> None:
    """Add AOV output nodes to a material for face_id output."""
    if not mat:
        return

    if not mat.use_nodes:
        mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Skip if already configured
    for node in nodes:
        if node.type == "OUTPUT_AOV":
            if getattr(node, "aov_name", "") == FACE_ID_AOV:
                return

    # Read face_id from vertex color attribute
    vc_node = nodes.new(type="ShaderNodeVertexColor")
    vc_node.location = (200, -200)
    vc_node.layer_name = FACE_ID_ATTR

    # Output to AOV
    aov_node = nodes.new(type="ShaderNodeOutputAOV")
    aov_node.location = (500, -200)
    aov_node.aov_name = FACE_ID_AOV

    # Connect Color output (face_id is in R channel)
    if "Color" in aov_node.inputs:
        links.new(vc_node.outputs["Color"], aov_node.inputs["Color"])


def get_scene_bounds(
    objects: list[bpy.types.Object],
) -> tuple[Vector, Vector]:
    """
    Calculate world-space bounding box of all mesh objects.

    Args:
        objects: List of Blender objects.

    Returns:
        Tuple of (min_corner, max_corner) vectors.
    """
    min_vec = Vector((float("inf"),) * 3)
    max_vec = Vector((float("-inf"),) * 3)

    for obj in objects:
        if obj.type != "MESH":
            continue

        for corner in obj.bound_box:
            world_coord = obj.matrix_world @ Vector(corner)
            for i in range(3):
                min_vec[i] = min(min_vec[i], world_coord[i])
                max_vec[i] = max(max_vec[i], world_coord[i])

    if min_vec[0] == float("inf"):
        return Vector((0, 0, 0)), Vector((0, 0, 0))

    return min_vec, max_vec


def calculate_camera_params(
    objects: list[bpy.types.Object],
) -> tuple[float, tuple[float, float, float]]:
    """
    Calculate optimal camera distance and scene center.

    Args:
        objects: List of Blender objects.

    Returns:
        Tuple of (camera_distance, scene_center).
    """
    min_corner, max_corner = get_scene_bounds(objects)
    center = (min_corner + max_corner) / 2
    diagonal = (max_corner - min_corner).length
    radius = max(diagonal * 0.5, DEFAULT_CAMERA_DISTANCE)
    sensor_width_mm = 36.0
    lens_mm = 70.0
    fov = 2.0 * math.atan(sensor_width_mm / (2.0 * lens_mm))
    distance = max(radius * CAMERA_FRAME_MARGIN / math.tan(fov * 0.5), DEFAULT_CAMERA_DISTANCE)
    return distance, tuple(center)


def orbit_angles_to_cartesian(
    orbit_angle_deg: float, height_angle_deg: float, radius: float
) -> tuple[float, float, float]:
    """
    Convert orbit+height angles to Cartesian coordinates.

    orbit_angle_deg: rotation around world Z axis (horizontal orbit)
    height_angle_deg: vertical camera angle above/below horizon
    """
    az = math.radians(orbit_angle_deg)
    el = math.radians(height_angle_deg)
    radius_xy = radius * math.cos(el)
    return (
        radius_xy * math.cos(az),
        radius_xy * math.sin(az),
        radius * math.sin(el),
    )


def create_camera() -> bpy.types.Object:
    """Create and register the render camera."""
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera_data.lens = 70.0
    camera_data.clip_start = 0.001
    camera_data.clip_end = 1000.0
    camera_obj = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    return camera_obj


def position_camera(
    camera: bpy.types.Object,
    orbit_angle: float,
    height_angle: float,
    distance: float,
    target: tuple[float, float, float],
) -> None:
    """Position camera on a sphere looking at target."""
    offset = Vector(orbit_angles_to_cartesian(orbit_angle, height_angle, distance))
    cam_pos = offset + Vector(target)

    # Compute look-at matrix
    direction = (Vector(target) - cam_pos).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.matrix_world = Matrix.Translation(cam_pos) @ rot_quat.to_matrix().to_4x4()


def get_camera_intrinsics(
    camera: bpy.types.Object, resolution: int
) -> list[list[float]]:
    """
    Compute 3x3 camera intrinsic matrix K.

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    """
    cam = camera.data
    sensor_fit = cam.sensor_fit

    if sensor_fit == "AUTO":
        sensor_fit = "HORIZONTAL"  # Square images

    if sensor_fit == "HORIZONTAL":
        focal_px = cam.lens * resolution / cam.sensor_width
    else:
        focal_px = cam.lens * resolution / cam.sensor_height

    cx = resolution / 2.0 + cam.shift_x * resolution
    cy = resolution / 2.0 + cam.shift_y * resolution

    return [
        [focal_px, 0.0, cx],
        [0.0, focal_px, cy],
        [0.0, 0.0, 1.0],
    ]


def get_camera_matrices(camera: bpy.types.Object, resolution: int) -> dict:
    """Extract only essential camera transforms/projection metadata."""
    c2w = camera.matrix_world.copy()
    w2c = c2w.inverted()

    def to_list(mat):
        return [list(row) for row in mat]

    return {
        "camera_to_world": to_list(c2w),
        "world_to_camera": to_list(w2c),
        "intrinsic_matrix": get_camera_intrinsics(camera, resolution),
    }


def setup_lighting() -> None:
    """Configure balanced studio-like lighting."""
    lights = [
        # name, energy, location, rotation_deg, RGB color
        ("KeyLight", 3.0, (5, -5, 8), (45, 10, 40), (1.00, 0.96, 0.90)),
        ("FillLight", 1.8, (-6, 4, 5), (55, -20, -35), (0.90, 0.95, 1.00)),
        ("RimLight", 2.2, (0, 6, 7), (35, 0, 180), (1.00, 1.00, 1.00)),
    ]

    for name, energy, location, rotation_deg, color in lights:
        light_data = bpy.data.lights.new(name=name, type="SUN")
        light_data.energy = energy
        light_data.color = color
        light_data.angle = math.radians(6.0)  # soften shadows for less harsh contrast
        light_obj = bpy.data.objects.new(name, light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = location
        light_obj.rotation_euler = tuple(math.radians(r) for r in rotation_deg)


def setup_render_settings(
    resolution: int, samples: int, transparent: bool
) -> None:
    """Configure Cycles render settings."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    # Try GPU rendering
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = True
        scene.cycles.device = "GPU"
    except Exception:
        print("GPU not available, using CPU")
        scene.cycles.device = "CPU"

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True

    # Use BOX filter with minimal width to prevent face ID interpolation
    # This prevents anti-aliasing from blending face IDs at triangle edges
    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.filter_width = 0.01  # Minimal filter width

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.image_settings.color_depth = "8"

    scene.render.film_transparent = transparent

    # Keep world lighting enabled even with transparent film, so RGB remains well-lit.
    _setup_background()

    # Slightly brighter photographic look without overexposure.
    scene.view_settings.exposure = 0.35

    scene.use_nodes = True


def _setup_background() -> None:
    """Set up soft sky-like world lighting."""
    scene = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.62, 0.66, 0.72, 1.0)
        if "Strength" in bg_node.inputs:
            bg_node.inputs["Strength"].default_value = 0.65


def _find_aov_socket(
    render_layers: bpy.types.Node, aov_name: str
) -> bpy.types.NodeSocket:
    """Find the compositor socket for an AOV output."""
    for name in [aov_name, f"AOV {aov_name}"]:
        if name in render_layers.outputs:
            return render_layers.outputs[name]

    # Fallback: substring search
    for socket in render_layers.outputs:
        if aov_name in socket.name:
            return socket

    available = [s.name for s in render_layers.outputs]
    raise KeyError(f"AOV '{aov_name}' not found. Available: {available}")


def setup_compositor(output_dir: str, view_name: str) -> None:
    """
    Configure compositor for RGB and face_id output.

    Args:
        output_dir: Base output directory.
        view_name: Name for output files.
    """
    scene = bpy.context.scene
    tree = scene.node_tree

    # Clear existing nodes
    tree.nodes.clear()

    # Render Layers
    rl = tree.nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    # Color correction
    cc = tree.nodes.new(type="CompositorNodeColorCorrection")
    cc.location = (200, 0)
    cc.master_saturation = 1.0
    cc.master_contrast = 1.02
    cc.master_gain = 1.0
    tree.links.new(rl.outputs["Image"], cc.inputs["Image"])

    # Composite output (RGB)
    comp = tree.nodes.new(type="CompositorNodeComposite")
    comp.location = (500, 0)
    tree.links.new(cc.outputs["Image"], comp.inputs["Image"])

    # Face ID file output (EXR)
    face_id_dir = os.path.join(output_dir, "face_id")
    os.makedirs(face_id_dir, exist_ok=True)

    file_out = tree.nodes.new(type="CompositorNodeOutputFile")
    file_out.location = (400, -200)
    file_out.base_path = face_id_dir
    file_out.format.file_format = "OPEN_EXR"
    file_out.format.color_depth = "32"
    file_out.format.color_mode = "RGB"
    file_out.file_slots[0].path = view_name

    aov_socket = _find_aov_socket(rl, FACE_ID_AOV)
    tree.links.new(aov_socket, file_out.inputs[0])


def render_view(
    camera: bpy.types.Object,
    orbit_angle: float,
    height_angle: float,
    distance: float,
    target: tuple[float, float, float],
    output_dir: str,
    view_name: str,
    resolution: int,
) -> dict:
    """
    Render a single view and return camera metadata.

    Returns:
        Dictionary with camera matrices and file paths.
    """
    position_camera(camera, orbit_angle, height_angle, distance, target)
    bpy.context.view_layer.update()

    # Setup output paths
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    rgb_path = os.path.join(rgb_dir, f"{view_name}.png")

    # Configure compositor
    setup_compositor(output_dir, view_name)

    # Collect camera info
    camera_info = {
        "view_name": view_name,
        "image_path": f"rgb/{view_name}.png",
        "face_id_path": f"face_id/{view_name}0001.exr",
        **get_camera_matrices(camera, resolution),
    }

    # Render
    bpy.context.scene.render.filepath = rgb_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {view_name}")

    return camera_info


def render_all_views(
    camera: bpy.types.Object,
    distance: float,
    target: tuple[float, float, float],
    output_dir: str,
    resolution: int,
) -> dict:
    """Render all configured views and collect camera metadata."""
    result = {"views": []}

    for height_angle in HEIGHT_ANGLES_DEG:
        for orbit_angle in ORBIT_ANGLES_DEG:
            view_name = f"az{orbit_angle:03d}_el{height_angle:+03d}"
            camera_info = render_view(
                camera, orbit_angle, height_angle, distance, target,
                output_dir, view_name, resolution
            )
            result["views"].append(camera_info)

    return result


def render_single_object(
    *,
    mesh_path: Path,
    object_name: str,
    object_slug: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Render all configured views for one object mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup scene
    clear_scene()
    imported = import_mesh(str(mesh_path))
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"No mesh objects imported from: {mesh_path}")

    if mesh_path.suffix.lower() == ".ply":
        # Generated PLY meshes are in OpenCV camera coordinates; convert once to Blender axes.
        apply_opencv_to_blender_transform(mesh_objects)

    print(f"Imported {len(imported)} objects ({len(mesh_objects)} meshes)")

    print("\n=== Setting up materials ===")
    setup_vertex_color_materials(imported)

    # Add face IDs
    add_face_id_attribute(imported)
    setup_face_id_aov(imported)

    # Calculate camera parameters
    auto_dist, center = calculate_camera_params(imported)
    cam_dist = args.camera_distance if args.camera_distance is not None else auto_dist

    print("\n=== Scene Info ===")
    print(f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"Camera distance: {cam_dist:.4f}")

    # Setup rendering
    setup_lighting()
    setup_render_settings(args.resolution, args.samples, args.transparent)
    camera = create_camera()

    # Render
    n_views = len(ORBIT_ANGLES_DEG) * len(HEIGHT_ANGLES_DEG)
    print(f"\n=== Rendering {n_views} views ===")
    camera_data = render_all_views(
        camera,
        cam_dist,
        center,
        str(output_dir),
        args.resolution,
    )
    camera_data["object_name"] = object_name
    camera_data["object_slug"] = object_slug
    camera_data["source_mesh_path"] = str(mesh_path)
    camera_data["resolution"] = [args.resolution, args.resolution]

    # Save per-object metadata
    cameras_path = output_dir / "cameras.json"
    with cameras_path.open("w", encoding="utf-8") as f:
        json.dump(camera_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Completed object: {object_name} ({object_slug})")
    print(f"  RGB: {output_dir}/rgb/")
    print(f"  Face IDs: {output_dir}/face_id/")
    print(f"  Cameras: {cameras_path}")
    print(f"{'=' * 60}\n")

    return None


def main() -> None:
    """Main entry point."""
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent

    object_mesh_dir, output_video_dir = resolve_default_dirs(args, script_dir)
    pag_path = resolve_pag_path(args, script_dir)
    output_video_dir.mkdir(parents=True, exist_ok=True)

    if not object_mesh_dir.exists():
        raise NotADirectoryError(f"Generated object mesh dir not found: {object_mesh_dir}")
    meshes_dir = (object_mesh_dir / "meshes").resolve()
    if not meshes_dir.exists():
        raise NotADirectoryError(f"Generated object meshes dir not found: {meshes_dir}")

    pag_objects = load_pag_objects_from_states_only(pag_path)

    print(f"\n{'=' * 60}")
    print("render_object_views.py — PAG object batch renderer")
    print(f"  video:     {args.interaction_name}")
    print(f"  meshes:    {object_mesh_dir}")
    print(f"  pag:       {pag_path.name} ({len(pag_objects)} objects)")
    print(f"  out_root:  {output_video_dir}")
    print(f"  resolution:{args.resolution}x{args.resolution}")
    print(f"  samples:   {args.samples}")
    print(f"{'=' * 60}")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for object_name, object_slug in pag_objects:
        mesh_path = resolve_object_mesh_path(
            object_slug=object_slug,
            meshes_dir=meshes_dir,
        )

        if mesh_path is None:
            reason = f"No generated object mesh found in {meshes_dir} for object '{object_name}' ({object_slug})"
            print(f"\n[SKIP] {object_slug}: {reason}")
            skipped_count += 1
            continue

        object_out_dir = (output_video_dir / object_slug).resolve()
        render_out_dir = (object_out_dir / "renders").resolve()

        print(f"\n{'─' * 50}")
        print(f"[OBJECT] {object_name} ({object_slug})")
        print(f"  mesh: {mesh_path}")
        print(f"  out:  {render_out_dir}")
        print(f"{'─' * 50}")

        try:
            render_single_object(
                mesh_path=mesh_path,
                object_name=object_name,
                object_slug=object_slug,
                output_dir=render_out_dir,
                args=args,
            )
            processed_count += 1
            print(f"[OK] {object_slug}")
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {object_slug}: {reason}")
            traceback.print_exc()
            failed_count += 1

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"  processed: {processed_count}")
    print(f"  skipped:   {skipped_count}")
    print(f"  failed:    {failed_count}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
