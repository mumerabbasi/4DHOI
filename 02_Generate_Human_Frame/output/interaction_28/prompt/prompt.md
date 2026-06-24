Input Image:

- The image is the original scene. Use the interaction description to identify the existing
target object that must remain fixed and contacted.

Scene Preservation Instructions:

- Use the provided scene image as a fixed camera reference.
- Insert exactly one person performing the described interaction with the target object.
- This frame must represent the pre-motion start state: the person is already in natural contact
with the target object of the scene, but no visible object motion has begun yet.
- Target object is already present in the scene. Do not add it.
- Preserve the camera pose, field of view, perspective, room geometry, lighting, shadows, textures,
and all existing object identities.
- Do not add new furniture or remove existing items.
- Keep the target object in exactly the same pose, position, orientation, scale, material,
and appearance as in the original scene.
- Do not move, rotate, deform, lift, translate, or reposition the target object in any way.
- Only add the human in plausible contact, as if about to start the action in the next frame.

Output:

- Return only the edited scene image.

Interaction:
A person runs on the treadmill in a forward-moving exercise pose. The left hand lightly holds the left side handle of the treadmill, the right hand lightly holds the right side handle of the treadmill, and the right foot lands on the moving belt of the treadmill while the left leg lifts behind. The person has tied-back dark hair, a focused facial expression, an athletic top, leggings, and running shoes.
