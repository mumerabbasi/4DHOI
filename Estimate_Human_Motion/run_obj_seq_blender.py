import bpy
import os

OBJ_FOLDER = r"C:\Users\umerh\Desktop\Thesis Temp\video_01_hm\output_objs"


def import_obj_sequence():
    # 1. Get file list
    if not os.path.exists(OBJ_FOLDER):
        print("Error: Folder not found!")
        return

    files = sorted([f for f in os.listdir(OBJ_FOLDER) if f.endswith(".obj")])
    if not files:
        print("No .obj files found.")
        return

    print(f"Found {len(files)} frames. Importing...")

    # 2. Setup Collection
    collection_name = "Human_Motion"
    if collection_name in bpy.data.collections:
        # Remove old collection if it exists to avoid duplicates
        bpy.data.collections.remove(bpy.data.collections[collection_name])

    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)

    # 3. Import and Animate
    for i, filename in enumerate(files):
        filepath = os.path.join(OBJ_FOLDER, filename)

        # We force the axes here
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='Y', up_axis='Z')

        # Get the imported object
        obj = bpy.context.selected_objects[0]
        obj.name = f"Frame_{i:04d}"

        # Rotate 180 degrees on Z-axis to convert it from OpenCV camera to PyTorch 3D camera.
        # To match with SAM3D-Objects meshes.
        # Turning this off, as I'm converting sam3d objects into OpenCV camera coords for now.
        # obj.rotation_euler = (0, 0, math.radians(180))

        # Link to collection
        for col in obj.users_collection:
            col.objects.unlink(obj)
        collection.objects.link(obj)

        # --- ANIMATE VISIBILITY (Stop Motion) ---
        frame_num = i + 1

        # Hide on frame 0
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=0)
        obj.keyframe_insert(data_path="hide_render", frame=0)

        # Show on CURRENT frame
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num)
        obj.keyframe_insert(data_path="hide_render", frame=frame_num)

        # Hide on NEXT frame
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame_num + 1)
        obj.keyframe_insert(data_path="hide_render", frame=frame_num + 1)

    # Set timeline length
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(files)
    print("Done! Press Play.")


if __name__ == "__main__":
    import_obj_sequence()
