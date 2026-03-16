"""Import all PLY frame sequences under a video directory into Blender."""

import argparse
import bpy
import re
import sys
from pathlib import Path


VIDEO_DIR = ""

# Leave blank to derive the root collection name from VIDEO_DIR.
ROOT_COLLECTION_NAME = ""

# Pick your color here (R, G, B, A) in 0..1.
COLLECTION_COLOR = (1.0, 0.2, 0.2, 1.0)
OVERWRITE_MATERIAL_COLOR = True


def _extract_index(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 10**18


def _natural_key(name: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _delete_collection_tree(collection_name: str):
    root = bpy.data.collections.get(collection_name)
    if root is None:
        return

    collections = []
    objects = []
    seen_objects = set()

    def _walk(collection: bpy.types.Collection):
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


def _gather_mesh_objects(imported_objects):
    meshes = [obj for obj in imported_objects if obj.type == "MESH"]
    if meshes:
        return meshes

    meshes = []
    for obj in imported_objects:
        for child in obj.children_recursive:
            if child.type == "MESH":
                meshes.append(child)
    return meshes


def _get_or_create_material(name: str, rgba):
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
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=filepath)
        return
    if hasattr(bpy.ops.import_mesh, "ply"):
        bpy.ops.import_mesh.ply(filepath=filepath)
        return
    raise RuntimeError("No PLY importer found (wm.ply_import or import_mesh.ply).")


def _find_sequences(video_dir: Path):
    sequences = []
    for child in sorted(video_dir.iterdir(), key=lambda path: _natural_key(path.name)):
        if not child.is_dir():
            continue

        meshes_dir = child / "meshes"
        if not meshes_dir.is_dir():
            continue

        ply_files = sorted(
            [path for path in meshes_dir.iterdir() if path.is_file() and path.suffix.lower() == ".ply"],
            key=lambda path: _extract_index(path.name),
        )
        if ply_files:
            sequences.append((child.name, ply_files))

    return sequences


def _make_empty(name: str, parent=None):
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.25
    empty.parent = parent
    if parent is not None:
        empty.matrix_parent_inverse = parent.matrix_world.inverted()
    return empty


def _parse_video_dir_arg():
    if "--" not in sys.argv:
        return None

    parser = argparse.ArgumentParser(description="Import all PLY sequences inside a video directory.")
    parser.add_argument("video_dir", nargs="?", help="Path to a video_xx directory.")
    parser.add_argument("--video-dir", dest="video_dir_option", help="Path to a video_xx directory.")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    return args.video_dir_option or args.video_dir


def import_ply_video_hierarchies(video_dir=None):
    video_dir_arg = video_dir or _parse_video_dir_arg() or VIDEO_DIR
    if not video_dir_arg:
        raise ValueError("Set VIDEO_DIR or pass the video directory after '--'.")

    resolved_video_dir = Path(bpy.path.abspath(video_dir_arg)).expanduser().resolve()
    if not resolved_video_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {resolved_video_dir}")

    sequences = _find_sequences(resolved_video_dir)
    if not sequences:
        raise FileNotFoundError(
            f"No sequences found under {resolved_video_dir}. Expected video_xx/<sequence>/meshes/*.ply"
        )

    root_name = ROOT_COLLECTION_NAME.strip() or resolved_video_dir.name
    root_empty_name = f"{root_name}::root"
    material_name = f"{root_name}_mat"

    print(f"Video directory: {resolved_video_dir}")
    print(f"Found {len(sequences)} sequences:")
    for sequence_name, ply_files in sequences:
        print(f"  - {sequence_name}: {len(ply_files)} frames")

    _delete_collection_tree(root_name)

    root_collection = bpy.data.collections.new(root_name)
    bpy.context.scene.collection.children.link(root_collection)

    root_empty = _make_empty(root_empty_name)
    root_collection.objects.link(root_empty)

    material = None
    if OVERWRITE_MATERIAL_COLOR:
        material = _get_or_create_material(material_name, COLLECTION_COLOR)

    max_frame_count = 1
    imported_object_count = 0

    for sequence_name, ply_files in sequences:
        sequence_collection_name = f"{root_name}::{sequence_name}"
        sequence_empty_name = f"{root_name}::{sequence_name}::root"

        sequence_collection = bpy.data.collections.new(sequence_collection_name)
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
                    obj.name = f"{sequence_name}_frame_{frame_index:04d}_{mesh_index:02d}"

                for collection in list(obj.users_collection):
                    collection.objects.unlink(obj)
                sequence_collection.objects.link(obj)

                obj.parent = sequence_empty
                obj.matrix_parent_inverse = sequence_empty.matrix_world.inverted()

                if OVERWRITE_MATERIAL_COLOR and material is not None:
                    _apply_material(obj, material, overwrite=True)

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

                imported_object_count += 1

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max_frame_count

    print(f"Imported {imported_object_count} mesh objects into '{root_name}'.")
    print(f"Timeline set to frames 1..{max_frame_count}.")
    if OVERWRITE_MATERIAL_COLOR:
        print(f"Material '{material_name}' applied (overwriting imported materials).")
    else:
        print("Imported materials kept (no overwrite).")


if __name__ == "__main__":
    import_ply_video_hierarchies()
