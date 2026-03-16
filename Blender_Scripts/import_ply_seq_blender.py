import bpy
import os
import re

PLY_FOLDER = ""
COLLECTION_NAME = ""

# Pick your color here (R, G, B, A) in 0..1
COLLECTION_COLOR = (1.0, 0.2, 0.2, 1.0)
MATERIAL_NAME = f"{COLLECTION_NAME}_mat"

# Toggle: overwrite imported materials with a single colored material
OVERWRITE_MATERIAL_COLOR = True


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
    if obj.type != "MESH":
        return

    if obj.data.materials:
        if overwrite:
            for i in range(len(obj.data.materials)):
                obj.data.materials[i] = mat
    else:
        obj.data.materials.append(mat)


def _import_ply(filepath: str):
    """Import a PLY mesh with Blender-version fallback."""
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=filepath)
        return
    if hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=filepath)
        return
    raise RuntimeError("No PLY importer found (wm.ply_import or import_mesh.ply).")


def import_ply_sequence():
    # 1) Get file list
    if not os.path.exists(PLY_FOLDER):
        print("Error: Folder not found!")
        return

    files = sorted(
        [f for f in os.listdir(PLY_FOLDER) if f.lower().endswith(".ply")],
        key=_extract_index,
    )
    if not files:
        print("No .ply files found.")
        return

    print(f"Found {len(files)} frames. Importing...")

    # 2) Setup collection
    _delete_collection_and_objects(COLLECTION_NAME)
    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)

    mat = None
    if OVERWRITE_MATERIAL_COLOR:
        mat = _get_or_create_material(MATERIAL_NAME, COLLECTION_COLOR)

    # 3) Import and animate
    for i, filename in enumerate(files):
        filepath = os.path.join(PLY_FOLDER, filename)

        bpy.ops.object.select_all(action="DESELECT")
        _import_ply(filepath)

        imported = list(bpy.context.selected_objects)
        mesh_objs = _gather_mesh_objects(imported)
        if not mesh_objs:
            continue

        frame_num = i + 1
        for obj in mesh_objs:
            obj.name = f"Frame_{i:04d}"

            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)

            if OVERWRITE_MATERIAL_COLOR and mat is not None:
                _apply_material(obj, mat, overwrite=True)

            # --- Animate visibility (stop-motion) ---
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

    # Set timeline length
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(files)
    print(f"Imported {len(files)} frames into '{COLLECTION_NAME}'.")
    if OVERWRITE_MATERIAL_COLOR:
        print(f"Material '{MATERIAL_NAME}' applied (overwriting imported materials).")
    else:
        print("Imported materials kept (no overwrite).")


if __name__ == "__main__":
    import_ply_sequence()
