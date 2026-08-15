import json
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


def assign_vertex_color_material(obj, roughness=0.62):
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
    bsdf.inputs["Roughness"].default_value = float(roughness)
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def srgb_channel_to_linear(value):
    value = float(value)
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def assign_human_contact_material(obj):
    mesh = obj.data
    mat = bpy.data.materials.new(name=f"{obj.name}_module06_blue_with_contacts")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")

    attr = nodes.new(type="ShaderNodeAttribute")
    if getattr(mesh, "color_attributes", None) and len(mesh.color_attributes) > 0:
        attr.attribute_name = mesh.color_attributes[0].name
    else:
        attr.attribute_name = "Col"

    encoded_base_linear = tuple(
        srgb_channel_to_linear(channel / 255.0)
        for channel in (179.0, 206.0, 249.0)
    )
    distance = nodes.new(type="ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    distance.inputs[1].default_value = encoded_base_linear
    mat.node_tree.links.new(attr.outputs["Color"], distance.inputs[0])

    is_contact = nodes.new(type="ShaderNodeMath")
    is_contact.operation = "GREATER_THAN"
    is_contact.inputs[1].default_value = 1e-4
    mat.node_tree.links.new(distance.outputs["Value"], is_contact.inputs[0])

    mix = nodes.new(type="ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[1].default_value = (0.45, 0.62, 0.95, 1.0)
    mat.node_tree.links.new(is_contact.outputs["Value"], mix.inputs[0])
    mat.node_tree.links.new(attr.outputs["Color"], mix.inputs[2])
    mat.node_tree.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.55
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


def object_bounds_center_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = Vector((0.0, 0.0, 0.0))
    for corner in corners:
        center += corner
    return center / max(len(corners), 1)


def aim_object_at(obj, target):
    direction = Vector(target) - obj.location
    if direction.length < 1e-6:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


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


def configure_soft_room_lighting(human_obj, camera_objects):
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


argv = sys.argv
config_path = Path(argv[argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene_obj = import_ply(config["scene_contact_mesh"])
scene_obj.name = "scene_contact_regions"
assign_vertex_color_material(scene_obj, roughness=0.65)

human_objects = {}
for state in config["states"]:
    obj = import_ply(state["mesh_path"])
    obj.name = f"{state['name']}_human_contact_regions"
    assign_human_contact_material(obj)
    obj.hide_render = True
    obj.hide_viewport = True
    human_objects[state["name"]] = obj

default_width = int(config["width"])
default_height = int(config["height"])
default_intrinsics = config["intrinsics"]
sensor_width = 36.0
camera_objects = {}
for view in config["views"]:
    width = int(view.get("width", default_width))
    height = int(view.get("height", default_height))
    resolution_percentage = int(
        view.get("resolution_percentage", config["resolution_percentage"])
    )
    intrinsics = view.get("intrinsics", default_intrinsics)
    fx = float(intrinsics[0][0])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    camera_data = bpy.data.cameras.new(view["name"])
    camera_obj = bpy.data.objects.new(view["name"], camera_data)
    bpy.context.collection.objects.link(camera_obj)
    camera_obj.matrix_world = Matrix(view["camera_matrix_world"])
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = sensor_width
    camera_data.lens = fx * sensor_width / float(width)
    camera_data.shift_x = (float(width) * 0.5 - cx) / float(width)
    camera_data.shift_y = (cy - float(height) * 0.5) / float(width)
    camera_data.clip_start = 0.01
    camera_data.clip_end = 100.0
    camera_objects[view["name"]] = (
        camera_obj,
        width,
        height,
        resolution_percentage,
    )

configure_cycles_gpu(config["cycles_samples"])
bpy.context.scene.world = (
    bpy.data.worlds.new("world")
    if bpy.context.scene.world is None
    else bpy.context.scene.world
)
configure_soft_room_lighting(human_objects["optimized"], camera_objects)
bpy.context.scene.render.resolution_percentage = int(config["resolution_percentage"])
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.view_settings.view_transform = "Filmic"
bpy.context.scene.view_settings.look = "Medium High Contrast"
bpy.context.scene.view_settings.exposure = 0.0
bpy.context.scene.view_settings.gamma = 1.0
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])

for job in config["render_jobs"]:
    for obj in human_objects.values():
        obj.hide_render = True
        obj.hide_viewport = True
    human_obj = human_objects[job["state"]]
    human_obj.hide_render = False
    human_obj.hide_viewport = False
    camera_obj, width, height, resolution_percentage = camera_objects[job["view"]]
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.render.resolution_x = int(width)
    bpy.context.scene.render.resolution_y = int(height)
    bpy.context.scene.render.resolution_percentage = int(resolution_percentage)
    bpy.context.scene.render.filepath = job["render_path"]
    bpy.ops.render.render(write_still=True)
