import bpy
import os
import re
import math

# -----------------------------
# User settings
# -----------------------------
GLB_FOLDER = r"C:\Users\umerh\Desktop\Thesis Temp\video_01_waft\iron\meshes"
COLLECTION_NAME = "TrackedMeshes_waft_iron"

# Pick your color here (R, G, B, A) in 0..1
COLLECTION_COLOR = (0.0, 0.0, 1.0, 1.0)  # red

# Material settings
MATERIAL_NAME = f"{COLLECTION_NAME}_mat"

# Toggle: overwrite imported materials with a single colored material
OVERWRITE_MATERIAL_COLOR = True
# -----------------------------


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


def _get_or_create_material(name: str, rgba):
    """Create a simple Principled material with Base Color = rgba."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")

    bsdf.inputs["Base Color"].default_value = rgba
    return mat


def _apply_material(obj: bpy.types.Object, mat: bpy.types.Material, overwrite: bool):
    """
    Assign material.
    - overwrite=True: replace existing slots (all) with mat
    - overwrite=False: keep existing; only add mat if object has none
    """
    if obj.type != "MESH":
        return

    if obj.data.materials:
        if overwrite:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
        # else: leave imported materials untouched
    else:
        obj.data.materials.append(mat)


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

    mat = None
    if OVERWRITE_MATERIAL_COLOR:
        mat = _get_or_create_material(MATERIAL_NAME, COLLECTION_COLOR)

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

            # Rotate 180 deg around Z (to convert to OpenCV camera convention)
            obj.rotation_mode = "XYZ"
            obj.rotation_euler = (0.0, 0.0, math.radians(180.0))

            # Link to our collection
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)

            # Optional: overwrite imported materials with single colored material
            if OVERWRITE_MATERIAL_COLOR and mat is not None:
                _apply_material(obj, mat, overwrite=True)

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

    print(f"Imported {len(files)} frames into '{COLLECTION_NAME}'.")
    if OVERWRITE_MATERIAL_COLOR:
        print(f"Material '{MATERIAL_NAME}' applied (overwriting imported materials).")
    else:
        print("Imported materials kept (no overwrite).")


if __name__ == "__main__":
    import_glb_sequence()
