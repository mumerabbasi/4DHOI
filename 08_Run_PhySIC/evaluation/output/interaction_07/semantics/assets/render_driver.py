import json
import sys
from pathlib import Path

import bpy


def import_ply(path):
    before = set(bpy.context.scene.objects)
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    imported = list(set(bpy.context.scene.objects) - before)
    if len(imported) != 1:
        raise RuntimeError(f"Expected one imported PLY object, got {len(imported)}")
    return imported[0]


def enable_cycles_gpu():
    scene = bpy.context.scene
    if scene.render.engine != "CYCLES":
        return
    preferences = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = backend
            preferences.get_devices()
            if any(device.type != "CPU" for device in preferences.devices):
                for device in preferences.devices:
                    device.use = device.type != "CPU"
                scene.cycles.device = "GPU"
                return
        except Exception:
            pass


config_path = Path(sys.argv[sys.argv.index("--") + 1])
config = json.loads(config_path.read_text(encoding="utf-8"))

human = bpy.data.objects.get("optimized_human")
if human is None or human.type != "MESH":
    raise RuntimeError("Module 06 blend does not contain the optimized_human mesh")

old_mesh = human.data
materials = list(old_mesh.materials)
replacement = import_ply(config["human_mesh_world"])
human.data = replacement.data
bpy.data.objects.remove(replacement, do_unlink=True)
human.data.materials.clear()
for material in materials:
    human.data.materials.append(material)
if old_mesh.users == 0:
    bpy.data.meshes.remove(old_mesh)

enable_cycles_gpu()
bpy.ops.wm.save_as_mainfile(filepath=config["blend_path"])

default_width = int(config["width"])
default_height = int(config["height"])
default_percentage = int(config["resolution_percentage"])
for view in config["views"]:
    camera = bpy.data.objects.get(view["name"])
    if camera is None or camera.type != "CAMERA":
        raise RuntimeError(f"Missing Module 06 camera: {view['name']}")
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = int(view.get("width", default_width))
    bpy.context.scene.render.resolution_y = int(view.get("height", default_height))
    bpy.context.scene.render.resolution_percentage = int(
        view.get("resolution_percentage", default_percentage)
    )
    bpy.context.scene.render.filepath = view["render_path"]
    bpy.ops.render.render(write_still=True)
