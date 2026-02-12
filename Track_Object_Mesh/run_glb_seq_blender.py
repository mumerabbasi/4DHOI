import bpy
import os
import re
import math

GLB_FOLDER = r"C:\Users\umerh\Desktop\Thesis Temp\track_out\video_01\meshes"
COLLECTION_NAME = "TrackedMeshes"


def _extract_index(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 10**18


def _delete_collection_and_objects(collection_name: str):
    if collection_name not in bpy.data.collections:
        return
    col = bpy.data.collections[collection_name]
    objs = list(col.objects)

    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    if objs:
        bpy.ops.object.delete(use_global=False)

    bpy.data.collections.remove(col)


def _gather_mesh_objects(imported_objects):
    meshes = [o for o in imported_objects if o.type == "MESH"]
    if meshes:
        return meshes

    # If selected objects are empties, find mesh children
    meshes = []
    for o in imported_objects:
        for c in o.children_recursive:
            if c.type == "MESH":
                meshes.append(c)
    return meshes


def import_glb_sequence():
    if not os.path.exists(GLB_FOLDER):
        print("Error: Folder not found!")
        return

    files = sorted(
        [f for f in os.listdir(GLB_FOLDER) if f.lower().endswith(".glb")],
        key=_extract_index,
    )
    if not files:
        print("No .glb files found.")
        return

    _delete_collection_and_objects(COLLECTION_NAME)

    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)

    for i, filename in enumerate(files):
        filepath = os.path.join(GLB_FOLDER, filename)

        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.import_scene.gltf(filepath=filepath)

        imported = list(bpy.context.selected_objects)
        mesh_objs = _gather_mesh_objects(imported)
        if not mesh_objs:
            continue

        frame_num = i + 1

        for obj in mesh_objs:
            obj.name = f"Frame_{i:04d}"

            # Rotate 180 deg around Z (OpenCV camera convention in your Blender setup)
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (0.0, 0.0, math.radians(180.0))

            # Link to our collection
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)

            # --- Stop-motion visibility ---
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

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(files)


if __name__ == "__main__":
    import_glb_sequence()
