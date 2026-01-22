"""
Blender 4.2 script to render 8 views of a GLB mesh with camera matrices.

Camera positions on a sphere with:
- Azimuth: 0°, 90°, 180°, 270°
- Elevation: 20° and 60°

IMPORTANT: This script does NOT modify any object positions or transforms.
Objects remain exactly as they are in the GLB file. This is critical for
point cloud projection workflows.

Expected directory structure:
    objects/<object_name>/
        ├── mesh.glb          (input)
        └── renders/          (output)
            ├── rgb_az000_el20.png ... rgb_az270_el60.png
            └── cameras.json

Usage:
    blender --background --python render_8views_blender.py -- \
        --input objects/iron/mesh.glb --resolution 1024

Requirements:
    Blender 4.2+
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
AZIMUTHS = [0, 90, 180, 270]  # degrees
ELEVATIONS = [20, 60]  # degrees
DEFAULT_CAMERA_DISTANCE = 3.0
DEFAULT_RESOLUTION = 1024


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments after '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render 8 views of a GLB mesh from different angles."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
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
        default=128,
        help="Number of render samples (default: 128).",
    )

    return parser.parse_args(argv)


def clear_scene() -> None:
    """Remove all default objects from the scene (not imported meshes)."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data blocks using Blender's built-in purge
    bpy.ops.outliner.orphans_purge(do_recursive=True)


def import_glb(filepath: str) -> list[bpy.types.Object]:
    """
    Import a GLB file WITHOUT modifying any transforms.

    Args:
        filepath: Path to the GLB file.

    Returns:
        List of imported objects.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GLB file not found: {filepath}")

    existing_objects = set(bpy.data.objects)

    # Import GLB - this preserves original transforms
    bpy.ops.import_scene.gltf(filepath=filepath)

    new_objects = [obj for obj in bpy.data.objects if obj not in existing_objects]

    # VERIFICATION: Print object transforms to confirm they're unchanged
    print("\n=== Object Transforms (UNCHANGED from GLB) ===")
    for obj in new_objects:
        if obj.type == "MESH":
            loc = obj.location
            print(f"  {obj.name}: location=({loc.x:.4f}, {loc.y:.4f}, {loc.z:.4f})")

    return new_objects


def get_scene_bounding_box(
    objects: list[bpy.types.Object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Calculate bounding box of all mesh objects in WORLD coordinates.

    Does NOT modify any object transforms.

    Args:
        objects: List of Blender objects.

    Returns:
        Tuple of (min_corner, max_corner) as 3D coordinates.
    """
    min_vec = Vector((float("inf"),) * 3)
    max_vec = Vector((float("-inf"),) * 3)
    found_mesh = False

    for obj in objects:
        if obj.type == "MESH":
            found_mesh = True
            # Transform bound_box corners from local to world coordinates
            for corner in obj.bound_box:
                world_coord = obj.matrix_world @ Vector(corner)
                # Element-wise min/max using Vector operations
                min_vec = Vector((min(min_vec[i], world_coord[i]) for i in range(3)))
                max_vec = Vector((max(max_vec[i], world_coord[i]) for i in range(3)))

    if not found_mesh:
        return (0, 0, 0), (0, 0, 0)

    return tuple(min_vec), tuple(max_vec)


def calculate_scene_info(
    objects: list[bpy.types.Object],
) -> tuple[float, tuple[float, float, float]]:
    """
    Calculate scene center and recommended camera distance.

    Does NOT modify any object transforms.

    Args:
        objects: List of Blender objects.

    Returns:
        Tuple of (recommended_distance, scene_center).
    """
    min_corner, max_corner = get_scene_bounding_box(objects)
    min_vec, max_vec = Vector(min_corner), Vector(max_corner)

    center = (min_vec + max_vec) / 2
    diagonal = (max_vec - min_vec).length

    # 1.5x diagonal ensures object fits well in frame
    distance = max(diagonal * 1.5, DEFAULT_CAMERA_DISTANCE)
    return distance, tuple(center)


def spherical_to_cartesian(
    azimuth_deg: float,
    elevation_deg: float,
    radius: float,
) -> tuple[float, float, float]:
    """
    Convert spherical coordinates to Cartesian.

    Convention:
    - Azimuth 0° = +X axis, 90° = +Y axis
    - Elevation 0° = XY plane, 90° = +Z axis

    Args:
        azimuth_deg: Azimuth angle in degrees.
        elevation_deg: Elevation angle in degrees.
        radius: Distance from origin.

    Returns:
        Tuple of (x, y, z) Cartesian coordinates.
    """
    azimuth_rad = math.radians(azimuth_deg)
    elevation_rad = math.radians(elevation_deg)

    x = radius * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = radius * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = radius * math.sin(elevation_rad)

    return (x, y, z)


def create_camera() -> bpy.types.Object:
    """Create a camera object."""
    camera_data = bpy.data.cameras.new(name="RenderCamera")
    camera_obj = bpy.data.objects.new("RenderCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    return camera_obj


def look_at(
    camera_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> Matrix:
    """
    Compute a camera-to-world matrix that looks at target from camera_pos.

    Uses Blender's built-in Vector.to_track_quat() for robust orientation.
    Camera convention: looks down -Z axis, Y is up.

    Args:
        camera_pos: Camera position in world coordinates.
        target_pos: Target position to look at.

    Returns:
        4x4 camera-to-world transformation matrix.
    """
    cam_pos = Vector(camera_pos)
    direction = (Vector(target_pos) - cam_pos).normalized()

    # Blender camera looks down -Z, Y is up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    return Matrix.Translation(cam_pos) @ rot_quat.to_matrix().to_4x4()


def position_camera(
    camera: bpy.types.Object,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    target: tuple[float, float, float],
) -> None:
    """
    Position camera on a sphere looking at target.

    Uses direct matrix assignment (no constraints) for reliable matrix extraction.

    Args:
        camera: Camera object.
        azimuth_deg: Azimuth angle in degrees.
        elevation_deg: Elevation angle in degrees.
        distance: Distance from target.
        target: Point to look at.
    """
    # Calculate camera position on sphere centered at target
    offset = Vector(spherical_to_cartesian(azimuth_deg, elevation_deg, distance))
    camera_pos = offset + Vector(target)

    # Set camera transform directly (no constraints needed)
    camera.matrix_world = look_at(tuple(camera_pos), target)


def get_camera_intrinsic_matrix(
    camera: bpy.types.Object,
    resolution_x: int,
    resolution_y: int,
) -> list[list[float]]:
    """
    Compute the 3x3 camera intrinsic matrix K.

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Args:
        camera: Camera object.
        resolution_x: Image width in pixels.
        resolution_y: Image height in pixels.

    Returns:
        3x3 intrinsic matrix as nested list.
    """
    cam_data = camera.data

    # Focal length in mm
    focal_length_mm = cam_data.lens

    # Sensor size in mm
    sensor_width_mm = cam_data.sensor_width
    sensor_height_mm = cam_data.sensor_height

    # Handle sensor fit mode
    if cam_data.sensor_fit == "AUTO":
        if resolution_x >= resolution_y:
            sensor_fit = "HORIZONTAL"
        else:
            sensor_fit = "VERTICAL"
    else:
        sensor_fit = cam_data.sensor_fit

    # Compute focal length in pixels
    if sensor_fit == "HORIZONTAL":
        fx = focal_length_mm * resolution_x / sensor_width_mm
        fy = fx  # Square pixels
    else:
        fy = focal_length_mm * resolution_y / sensor_height_mm
        fx = fy  # Square pixels

    # Principal point (image center)
    cx = resolution_x / 2.0
    cy = resolution_y / 2.0

    # Handle principal point shift if set
    shift_x = cam_data.shift_x
    shift_y = cam_data.shift_y
    cx += shift_x * resolution_x
    cy += shift_y * resolution_y

    return [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]


def get_camera_matrices(
    camera: bpy.types.Object,
    resolution_x: int,
    resolution_y: int,
) -> dict:
    """
    Extract all camera matrices needed for 3D-to-2D projection.

    For projecting a 3D world point p_world to 2D image coordinates:

    1. Transform to camera space:
       p_cam = world_to_camera @ [p_world, 1]  (homogeneous)

    2. Project to 2D (using intrinsic matrix K):
       p_2d_hom = K @ p_cam[:3]
       u = p_2d_hom[0] / p_2d_hom[2]
       v = p_2d_hom[1] / p_2d_hom[2]

    Note: v is from top of image. For bottom-up, use: v = resolution_y - v

    Args:
        camera: Camera object.
        resolution_x: Image width.
        resolution_y: Image height.

    Returns:
        Dictionary with camera matrices.
    """
    # Camera-to-world (4x4) - where camera is in world
    camera_to_world = camera.matrix_world.copy()

    # World-to-camera (4x4) - extrinsic matrix
    world_to_camera = camera_to_world.inverted()

    # Intrinsic matrix (3x3)
    intrinsic = get_camera_intrinsic_matrix(camera, resolution_x, resolution_y)

    # Camera position in world coordinates
    camera_position = list(camera.matrix_world.translation)

    # Convert Blender matrices to nested lists for JSON
    def matrix_to_list(mat):
        return [list(row) for row in mat]

    return {
        "camera_to_world": matrix_to_list(camera_to_world),
        "world_to_camera": matrix_to_list(world_to_camera),
        "intrinsic_matrix": intrinsic,
        "camera_position": camera_position,
        "resolution": [resolution_x, resolution_y],
        "focal_length_mm": camera.data.lens,
        "sensor_width_mm": camera.data.sensor_width,
    }


def setup_lighting() -> None:
    """Set up three-point lighting."""
    # Key light
    key_light_data = bpy.data.lights.new(name="KeyLight", type="SUN")
    key_light_data.energy = 3.0
    key_light = bpy.data.objects.new("KeyLight", key_light_data)
    bpy.context.scene.collection.objects.link(key_light)
    key_light.location = (5, -5, 8)
    key_light.rotation_euler = (math.radians(45), math.radians(15), math.radians(45))

    # Fill light
    fill_light_data = bpy.data.lights.new(name="FillLight", type="SUN")
    fill_light_data.energy = 1.5
    fill_light = bpy.data.objects.new("FillLight", fill_light_data)
    bpy.context.scene.collection.objects.link(fill_light)
    fill_light.location = (-5, 5, 4)
    fill_light.rotation_euler = (math.radians(60), math.radians(-15), math.radians(-45))

    # Rim light
    rim_light_data = bpy.data.lights.new(name="RimLight", type="SUN")
    rim_light_data.energy = 2.0
    rim_light = bpy.data.objects.new("RimLight", rim_light_data)
    bpy.context.scene.collection.objects.link(rim_light)
    rim_light.location = (0, 5, 6)
    rim_light.rotation_euler = (math.radians(30), 0, math.radians(180))


def setup_render_settings(
    resolution: int,
    samples: int,
    transparent: bool,
) -> None:
    """Configure render settings."""
    scene = bpy.context.scene

    scene.render.engine = "CYCLES"

    # GPU rendering
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

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    scene.render.image_settings.color_depth = "8"

    if transparent:
        scene.render.film_transparent = True
    else:
        scene.render.film_transparent = False
        world = bpy.data.worlds.get("World")
        if world is None:
            world = bpy.data.worlds.new("World")
        scene.world = world
        world.use_nodes = True
        bg_node = world.node_tree.nodes.get("Background")
        if bg_node:
            bg_node.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)


def render_view(
    camera: bpy.types.Object,
    azimuth: float,
    elevation: float,
    distance: float,
    target: tuple[float, float, float],
    output_path: str,
    resolution: int,
) -> dict:
    """
    Render a single view and return camera matrices.

    Args:
        camera: Camera object.
        azimuth: Azimuth angle in degrees.
        elevation: Elevation angle in degrees.
        distance: Camera distance from target.
        target: Point to look at.
        output_path: Path to save rendered image.
        resolution: Image resolution.

    Returns:
        Camera matrices dictionary.
    """
    position_camera(camera, azimuth, elevation, distance, target)

    # Update scene to ensure matrix is computed
    bpy.context.view_layer.update()

    # Extract camera matrices BEFORE rendering
    camera_info = get_camera_matrices(camera, resolution, resolution)
    camera_info["azimuth_deg"] = azimuth
    camera_info["elevation_deg"] = elevation
    camera_info["target"] = list(target)
    camera_info["distance"] = distance
    camera_info["image_path"] = os.path.basename(output_path)

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered: {output_path}")

    return camera_info


def render_all_views(
    camera: bpy.types.Object,
    distance: float,
    target: tuple[float, float, float],
    output_dir: str,
    mesh_name: str,
    resolution: int,
) -> dict:
    """
    Render all 8 views and save camera matrices.

    Args:
        camera: Camera object.
        distance: Camera distance from target.
        target: Scene center (what camera looks at).
        output_dir: Directory for outputs.
        mesh_name: Base name for files.
        resolution: Image resolution.

    Returns:
        Dictionary with all camera information.
    """
    all_cameras = {
        "mesh_name": mesh_name,
        "scene_center": list(target),
        "camera_distance": distance,
        "resolution": [resolution, resolution],
        "views": [],
    }

    for elevation in ELEVATIONS:
        for azimuth in AZIMUTHS:
            filename = f"rgb_az{azimuth:03d}_el{elevation:02d}.png"
            output_path = os.path.join(output_dir, filename)

            camera_info = render_view(
                camera, azimuth, elevation, distance, target, output_path, resolution
            )
            all_cameras["views"].append(camera_info)

    return all_cameras


def save_camera_matrices(camera_data: dict, output_path: str) -> None:
    """Save camera matrices to JSON file."""
    with open(output_path, "w") as f:
        json.dump(camera_data, f, indent=2)
    print(f"Camera matrices saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Object name is the parent directory name (e.g., objects/iron/mesh.glb -> iron)
    object_dir = input_path.parent
    mesh_name = object_dir.name

    # Output to <object_dir>/renders/ by default
    if args.output_dir is None:
        output_dir = object_dir / "renders"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Samples: {args.samples}")
    print(f"Transparent: {args.transparent}")
    print(f"{'='*60}\n")

    # Clear default scene objects
    clear_scene()

    # Import mesh - NO TRANSFORMS MODIFIED
    print(f"Importing: {input_path}")
    imported_objects = import_glb(str(input_path))
    print(f"Imported {len(imported_objects)} objects")

    # Calculate scene info without modifying objects
    auto_distance, scene_center = calculate_scene_info(imported_objects)
    camera_distance = args.camera_distance if args.camera_distance else auto_distance

    print("\n=== Scene Info ===")
    print(f"Scene center: ({scene_center[0]:.4f}, {scene_center[1]:.4f}, {scene_center[2]:.4f})")
    print(f"Camera distance: {camera_distance:.4f}")

    # Setup scene
    setup_lighting()
    setup_render_settings(args.resolution, args.samples, args.transparent)

    # Create camera
    camera = create_camera()

    # Render all views and collect camera matrices
    print("\n=== Rendering 8 views ===")
    camera_data = render_all_views(
        camera, camera_distance, scene_center, str(output_dir), mesh_name, args.resolution
    )

    # Save camera matrices
    cameras_json_path = os.path.join(str(output_dir), "cameras.json")
    save_camera_matrices(camera_data, cameras_json_path)

    print(f"\n{'='*60}")
    print("Completed!")
    print(f"  - 8 images saved to: {output_dir}")
    print(f"  - Camera matrices saved to: {cameras_json_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
