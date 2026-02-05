"""
Blender 4.2 script to render multiple views of a GLB mesh with camera matrices.

Renders RGB images and per-pixel face ID maps from multiple viewpoints.
Face IDs are stored as raw float values in EXR format for exact integer recovery.

Usage:
    blender --background --python render_views_blender.py -- \\
        --input objects/iron/mesh.glb --resolution 1024
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

# Camera configuration
AZIMUTHS = [10, 100, 190, 280]  # degrees
ELEVATIONS = [-15, 25, 45]  # degrees
DEFAULT_CAMERA_DISTANCE = 0.5
DEFAULT_RESOLUTION = 1024

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
        description="Render multiple views of a GLB mesh from different angles."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./objects/video_01/iron/mesh.glb",
        help="Path to input GLB mesh file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save renders (default: <input_dir>/renders/).",
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


def clear_scene() -> None:
    """Remove all objects from the scene and purge orphan data."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def import_glb(filepath: str) -> list[bpy.types.Object]:
    """
    Import a GLB file preserving original transforms.

    Args:
        filepath: Path to the GLB file.

    Returns:
        List of imported objects.

    Raises:
        FileNotFoundError: If the GLB file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GLB file not found: {filepath}")

    existing = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=filepath)
    new_objects = [obj for obj in bpy.data.objects if obj not in existing]

    print("\n=== Imported Objects ===")
    for obj in new_objects:
        if obj.type == "MESH":
            loc = obj.location
            print(f"  {obj.name}: loc=({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})")

    return new_objects


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
    distance = max(diagonal * 1.5, DEFAULT_CAMERA_DISTANCE)
    return distance, tuple(center)


def spherical_to_cartesian(
    azimuth_deg: float, elevation_deg: float, radius: float
) -> tuple[float, float, float]:
    """
    Convert spherical to Cartesian coordinates.

    Convention: azimuth 0° = +X, 90° = +Y; elevation 0° = XY plane, 90° = +Z.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return (
        radius * math.cos(el) * math.cos(az),
        radius * math.cos(el) * math.sin(az),
        radius * math.sin(el),
    )


def create_camera() -> bpy.types.Object:
    """Create and register the render camera."""
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera_obj = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    return camera_obj


def position_camera(
    camera: bpy.types.Object,
    azimuth: float,
    elevation: float,
    distance: float,
    target: tuple[float, float, float],
) -> None:
    """Position camera on a sphere looking at target."""
    offset = Vector(spherical_to_cartesian(azimuth, elevation, distance))
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
    """Extract camera matrices for 3D-to-2D projection."""
    c2w = camera.matrix_world.copy()
    w2c = c2w.inverted()

    def to_list(mat):
        return [list(row) for row in mat]

    return {
        "camera_to_world": to_list(c2w),
        "world_to_camera": to_list(w2c),
        "intrinsic_matrix": get_camera_intrinsics(camera, resolution),
        "camera_position": list(camera.matrix_world.translation),
        "resolution": [resolution, resolution],
        "focal_length_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
    }


def setup_lighting() -> None:
    """Configure three-point lighting."""
    lights = [
        ("KeyLight", "SUN", 2.0, (5, -5, 8), (45, 15, 45)),
        ("FillLight", "SUN", 1.5, (-5, 5, 4), (60, -15, -45)),
        ("RimLight", "SUN", 2.0, (0, 5, 6), (30, 0, 180)),
    ]

    for name, light_type, energy, location, rotation_deg in lights:
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
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

    if not transparent:
        _setup_background()

    scene.use_nodes = True


def _setup_background() -> None:
    """Set up gray world background."""
    scene = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)


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
    cc.master_saturation = 1.05
    cc.master_contrast = 1.05
    cc.master_gain = 0.95
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
    azimuth: float,
    elevation: float,
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
    position_camera(camera, azimuth, elevation, distance, target)
    bpy.context.view_layer.update()

    # Setup output paths
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    rgb_path = os.path.join(rgb_dir, f"{view_name}.png")

    # Configure compositor
    setup_compositor(output_dir, view_name)

    # Collect camera info
    camera_info = get_camera_matrices(camera, resolution)
    camera_info.update({
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "target": list(target),
        "distance": distance,
        "image_path": f"rgb/{view_name}.png",
        "face_id_path": f"face_id/{view_name}0001.exr",
    })

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
    mesh_name: str,
    resolution: int,
) -> dict:
    """Render all configured views and collect camera metadata."""
    result = {
        "mesh_name": mesh_name,
        "scene_center": list(target),
        "camera_distance": distance,
        "resolution": [resolution, resolution],
        "views": [],
    }

    for elevation in ELEVATIONS:
        for azimuth in AZIMUTHS:
            view_name = f"az{azimuth:03d}_el{elevation:+03d}"
            camera_info = render_view(
                camera, azimuth, elevation, distance, target,
                output_dir, view_name, resolution
            )
            result["views"].append(camera_info)

    return result


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    object_dir = input_path.parent
    mesh_name = object_dir.name

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else object_dir / "renders"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"{'=' * 60}")

    # Setup scene
    clear_scene()
    imported = import_glb(str(input_path))
    print(f"Imported {len(imported)} objects")

    print("\n=== Setting up materials ===")
    setup_vertex_color_materials(imported)

    # Add face IDs
    add_face_id_attribute(imported)
    setup_face_id_aov(imported)

    # Calculate camera parameters
    auto_dist, center = calculate_camera_params(imported)
    cam_dist = args.camera_distance or auto_dist

    print("\n=== Scene Info ===")
    print(f"Center: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"Camera distance: {cam_dist:.4f}")

    # Setup rendering
    setup_lighting()
    setup_render_settings(args.resolution, args.samples, args.transparent)
    camera = create_camera()

    # Render
    print(f"\n=== Rendering {len(AZIMUTHS) * len(ELEVATIONS)} views ===")
    camera_data = render_all_views(
        camera, cam_dist, center, str(output_dir), mesh_name, args.resolution
    )

    # Save metadata
    cameras_path = output_dir / "cameras.json"
    with open(cameras_path, "w") as f:
        json.dump(camera_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Completed!")
    print(f"  RGB: {output_dir}/rgb/")
    print(f"  Face IDs: {output_dir}/face_id/")
    print(f"  Cameras: {cameras_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
