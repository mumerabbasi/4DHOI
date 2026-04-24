## Prompt 1

Use the provided scene image as a fixed camera reference.
Insert exactly one person performing the described interaction with the target object.
This frame must represent the pre-motion start state: the person is already in natural contact with the target object, but no visible object motion has begun yet.
Preserve the camera pose, field of view, perspective, room geometry, lighting, shadows, textures, and all existing object identities.
Do not add new furniture or remove existing items.
Keep the target object in exactly the same pose, position, orientation, scale, material, and appearance as in the original scene.
Do not move, rotate, deform, lift, translate, or reposition the target object in any way.
Only add the human in plausible contact, as if about to start the action in the next frame.

## Prompt 2

Use the provided scene image as a fixed camera reference.

Goal

- Insert exactly one person performing the described interaction with the target object.
- This frame must represent the pre-motion start state.
- The person must already be in natural, physically plausible contact with the target object.
- No visible object motion should have begun yet.

Scene preservation requirements

- Preserve the camera pose exactly.
- Preserve the original field of view, perspective, and framing.
- Preserve the room geometry, lighting, shadows, textures, and all existing object identities.
- Do not add any new furniture or objects.
- Do not remove any existing items.

Target object constraints

- Keep the target object in exactly the same pose, position, orientation, scale, material, and appearance as in the original scene.
- Do not move, rotate, deform, lift, translate, or reposition the target object in any way.
- Do not alter the target object’s geometry or visual identity.

Human insertion requirements

- Add only one person.
- The person must be placed in plausible contact with the target object.
- The pose should clearly suggest they are about to begin the action in the next frame.
- The contact should look natural, stable, and physically believable.
- The person should be the only new element introduced into the scene.

Output requirements

- The final image should look like the same scene captured by the same fixed camera.
- The only change from the original image should be the addition of the person in the correct pre-motion contact pose.