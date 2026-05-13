Scene Preservation Instructions:
Use the provided scene image as a fixed camera reference.
Insert exactly one person performing the described interaction with the target object.
This frame must represent the pre-motion start state: the person is already in natural contact
with the target object of the scene, but no visible object motion has begun yet.
Target object is already present in the scene. Do not add it.
Preserve the camera pose, field of view, perspective, room geometry, lighting, shadows, textures,
and all existing object identities.
Do not add new furniture or remove existing items.
Keep the target object in exactly the same pose, position, orientation, scale, material,
and appearance as in the original scene.
Do not move, rotate, deform, lift, translate, or reposition the target object in any way.
Only add the human in plausible contact, as if about to start the action in the next frame.

Input Images:
The first image is the original scene. The second image is the binary target-object mask.
Use the mask only to identify the existing target object that must remain fixed and contacted.

Output:
Return only the edited scene image.

Interaction:
A person stands facing the wooden door, reaching out with their right hand to grasp the silver handle. The hand is positioned to turn the handle or push the door open. The left arm hangs naturally by the side. Both feet are planted on the dark floor in a balanced stance. The person has short dark hair, a neutral focused expression, and wears a light shirt, dark pants, and dark shoes. The room is a clean indoor space with white walls, a whiteboard, and a table.
