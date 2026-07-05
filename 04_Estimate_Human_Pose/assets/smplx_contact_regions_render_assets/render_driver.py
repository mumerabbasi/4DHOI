import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def import_ply(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    after = set(bpy.context.scene.objects)
    new_objects = list(after - before)
    if not new_objects:
        raise RuntimeError(f"Failed to import PLY: {path}")
    return new_objects[0]


def assign_vertex_color_material(obj):
    mesh = obj.data
    mat = bpy.data.materials.new(name=f"{obj.name}_vertex_color")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    attr = nodes.new(type="ShaderNodeAttribute")
    if getattr(mesh, "color_attributes", None) and len(mesh.color_attributes) > 0:
        attr.attribute_name = mesh.color_attributes[0].name
    else:
        attr.attribute_name = "Col"
    mat.node_tree.links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.65
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def configure_cycles_gpu(samples):
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.render.use_persistent_data = True
    bpy.context.scene.cycles.device = "GPU"
    bpy.context.scene.cycles.samples = int(samples)
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.max_bounces = 6
    bpy.context.scene.cycles.diffuse_bounces = 3
    bpy.context.scene.cycles.glossy_bounces = 3
    bpy.context.scene.cycles.transparent_max_bounces = 4

    prefs = bpy.context.preferences.addons["cycles"].preferences
    selected_backend = None
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            gpu_devices = [device for device in prefs.devices if device.type != "CPU"]
            if gpu_devices:
                selected_backend = backend
                break
        except Exception as exc:
            print(f"Cycles GPU backend {backend} unavailable: {exc}")

    if selected_backend is None:
        bpy.context.scene.cycles.device = "CPU"
        print("Cycles GPU device unavailable; falling back to CPU")
        return

    for device in prefs.devices:
        device.use = device.type != "CPU"
    enabled = [
        f"{device.name} ({device.type})"
        for device in prefs.devices
        if device.use
    ]
    print(f"Cycles GPU backend: {selected_backend}")
    print(f"Cycles GPU devices: {enabled}")


def object_bounds_corners_world(obj):
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def object_bounds_center_world(obj):
    corners = object_bounds_corners_world(obj)
    center = Vector((0.0, 0.0, 0.0))
    for corner in corners:
        center += corner
    return center / max(len(corners), 1)


def object_bounds_diagonal_world(obj):
    corners = object_bounds_corners_world(obj)
    if not corners:
        return 1.0
    mins = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    maxs = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return max(float((maxs - mins).length), 1.0)


def aim_object_at(obj, target, up_axis="Y"):
    direction = Vector(target) - obj.location
    if direction.length < 1e-6:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", up_axis).to_euler()


def add_shadowless_area_light(name, location, target, energy, size):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(energy)
    light_data.size = float(size)
    if hasattr(light_data, "use_shadow"):
        light_data.use_shadow = False
    light_obj = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = Vector(location)
    aim_object_at(light_obj, target)
    return light_obj


def configure_soft_room_lighting(human_obj):
    focus = object_bounds_center_world(human_obj)
    focus.z += 0.65

    world = bpy.context.scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
        background.inputs["Strength"].default_value = 0.25

    add_shadowless_area_light(
        "room_overhead_softbox",
        (focus.x, focus.y, focus.z + 2.4),
        focus,
        energy=100.0,
        size=4.0,
    )
    print("Lighting: low world fill + shadowless room area light")


def camera_matrix_world(location, target, desired_up):
    location = Vector(location)
    forward = (Vector(target) - location).normalized()
    z_axis = -forward
    x_axis = Vector(desired_up).cross(z_axis)
    if x_axis.length < 1e-6:
        x_axis = Vector((1.0, 0.0, 0.0))
    x_axis.normalize()
    y_axis = z_axis.cross(x_axis).normalized()
    matrix = Matrix(
        (
            (x_axis.x, y_axis.x, z_axis.x, location.x),
            (x_axis.y, y_axis.y, z_axis.y, location.y),
            (x_axis.z, y_axis.z, z_axis.z, location.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    return matrix


def create_camera_for_view(name, direction, target, distance, width, height, margin, obj):
    camera_data = bpy.data.cameras.new(name)
    camera_obj = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera_obj)

    direction = Vector(direction).normalized()
    location = Vector(target) + direction * float(distance)
    desired_up = (0.0, 1.0, 0.0) if abs(direction.z) > 0.95 else (0.0, 0.0, 1.0)
    camera_obj.matrix_world = camera_matrix_world(location, target, desired_up)

    camera_data.type = "ORTHO"
    camera_data.clip_start = 0.01
    camera_data.clip_end = max(100.0, float(distance) * 4.0)
    aspect = max(float(width) / max(float(height), 1.0), 1e-6)

    corners = object_bounds_corners_world(obj)
    local = [camera_obj.matrix_world.inverted() @ corner for corner in corners]
    x_extent = max(c.x for c in local) - min(c.x for c in local)
    y_extent = max(c.y for c in local) - min(c.y for c in local)
    camera_data.ortho_scale = max(y_extent, x_extent / aspect) * float(margin)
    return camera_obj


argv = sys.argv
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

mesh_obj = import_ply(config["input_ply"])
mesh_obj.name = "smplx_contact_regions"
assign_vertex_color_material(mesh_obj)

# SMPL-X canonical vertices are Y-up. Rotate to Blender Z-up while preserving
# canonical front/back semantics: Blender front view sees canonical +Z.
mesh_obj.rotation_euler[0] = math.radians(90.0)
bpy.context.view_layer.objects.active = mesh_obj
bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

for polygon in mesh_obj.data.polygons:
    polygon.use_smooth = True

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = bpy.data.worlds.new("world") if bpy.context.scene.world is None else bpy.context.scene.world
configure_soft_room_lighting(mesh_obj)

bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0

target = object_bounds_center_world(mesh_obj)
distance = object_bounds_diagonal_world(mesh_obj) * float(config["camera_distance_scale"])
width = int(config["width"])
height = int(config["height"])
views = [
    ("front", (0.0, -1.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("bottom", (0.0, 0.0, -1.0)),
]

bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])
for name, direction in views:
    camera_obj = create_camera_for_view(
        name=name,
        direction=direction,
        target=target,
        distance=distance,
        width=width,
        height=height,
        margin=float(config["ortho_margin"]),
        obj=mesh_obj,
    )
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.filepath = config["render_paths"][name]
    bpy.ops.render.render(write_still=True)
