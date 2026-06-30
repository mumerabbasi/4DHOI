Attach the images from this folder in this exact order:
1. 01_reference_image.png - Reference Image: Human interacting with the target object; use only to infer contact locations.
2. 02_canvas_image.png - Canvas Image: Authoritative base image to preserve exactly.
3. 03_previous_composite.png - Previous Composite: Correction context only; do not use as the base image.
Provided Images:

Reference Image: An image showing a human interacting with the target object. Use this image only to infer which human body parts are physically contacting the target object and which object-surface regions are being touched.

Canvas Image: An image of the same scene without the human. Use this image as the base image and as the authoritative reference for the target object's final position, pose, geometry, and boundaries.

Task Description:

Generate an edited copy of the Canvas Image with solid-color segmentation masks overlaid only on the target object surface regions that are contacted by the human in the Reference Image.

Infer the contacted region relative to the target object in the Reference Image, then place the corresponding contact mask on the matching object region in the Canvas Image.

Critical Base-Image Rule:
Use the Canvas Image as the only valid base image. Preserve the Canvas Image exactly and only add the specified solid-color contact masks. Do not use the Reference Image as the base. Do not redraw, regenerate, stylize, relight, crop, pad, resize, move, or otherwise alter the Canvas Image.

Analysis Requirements:

1. Contact Inference:
   Analyze the Reference Image to determine which specified human body parts are physically touching the target object.

2. Object-Region Identification:
   For each contacting body part, identify the specific region of the target object being touched, such as the tabletop front edge, tabletop side edge, tabletop upper surface, tabletop underside, table leg, chair seat, chair arm, cabinet handle, ladder rung, rail, shelf, handle, edge, or other relevant object part.

3. Object-Relative Localization:
   Map each inferred contact region onto the corresponding part of the target object in the Canvas Image. Use the Canvas Image object pose and boundaries as ground truth. Prioritize correct placement on the object in the Canvas Image over exact pixel alignment with the Reference Image.

4. Contact Footprint Generation:
   Create a localized contact mask only on the object surface where contact occurs. The mask should represent the apparent contact footprint on the object, not the full visible human body part.

Execution Requirements:

Base Image:
Return the Canvas Image with the contact masks overlaid on top of it.

Canvas Size:
Match the Canvas Image dimensions and coordinate system.

Mask Placement:
Place each mask on the corresponding target-object region in the Canvas Image, using object-relative correspondence rather than direct coordinate transfer.

Mask Shape:
Use precise, dense, solid-color segmentation masks shaped like the localized contact footprint on the object surface. The mask should reflect the approximate size, orientation, and extent of the contact implied by the touching body part.

Object-Surface Clipping:
All mask pixels must be clipped strictly to the target object boundary in the Canvas Image. No mask may extend onto the human body, floor, wall, background, or any non-target-object region.

Scene Preservation:
Use the Canvas Image as a fixed camera reference. Preserve the camera pose, field of view, perspective, resolution, aspect ratio, room geometry, lighting, shadows, textures, and all existing object identities.
Keep the target object in exactly the same pose, position, orientation, scale, material, and appearance as in the Canvas Image. Do not move, rotate, deform, lift, translate, resize, crop, pad, or reposition the target object or any other scene element in any way.
Only add the specified solid-color contact mask overlays. Leave every non-overlay pixel visually unchanged from the Canvas Image.

Colors:
Use only the exact specified solid colors for the contacting body-part masks. Do not use transparency, antialiasing gradients, shadows, blended colors, or pastel variants.

Restrictions:
Do not add the generated human body.
Do not copy or paste the full visible hand, foot, hip, arm, leg, or other body-part silhouette.
Do not transfer masks by raw Reference Image pixel coordinates.
Do not add text labels, arrows, legends, boxes, keypoints, outlines, or any other annotations.
Show only the original Canvas Image with solid localized contact masks overlaid on the target object.

Example Guidance:
If the Reference Image shows a person gripping a ladder, infer which rungs or rails the hands and feet contact. Then place solid masks on those same ladder regions in the Canvas Image, aligned to the ladder as it appears in the Canvas Image.

Color Mapping for ContactMasks:
Left Foot: The left foot is in contact with the target object. Use solid #FF0000 / RGB(255, 0, 0) (pure red).
Right Foot: The right foot is in contact with the target object. Use solid #00FF00 / RGB(0, 255, 0) (pure green).
Right Hand: The right hand is in contact with the target object. Use solid #0000FF / RGB(0, 0, 255) (pure blue).

Required target-object contacts from SIG:
- Left Foot: required contact with target object 'shower stall'. Use #FF0000 / RGB(255, 0, 0) (pure red). SIG note: the left foot stands on the shower floor
- Right Foot: required contact with target object 'shower stall'. Use #00FF00 / RGB(0, 255, 0) (pure green). SIG note: the right foot stands on the shower floor
- Right Hand: required contact with target object 'shower stall'. Use #0000FF / RGB(0, 0, 255) (pure blue). SIG note: the right hand touches the shower control or glass door
- Interaction description: A person stands inside the shower stall taking a shower. Both feet are on the shower floor within the glass enclosure, and the right hand reaches toward the shower control or glass door while the body faces the shower wall. The torso is upright and relaxed, with the head angled forward as if under the shower area. The person has short dark hair and a calm neutral expression. The bathroom is bright with white walls, gray tile flooring, a window, a toilet, a sink, open shelving, a mirror, and a potted plant near the vanity.

Correction instructions from the VLM evaluator:
The blue right-hand mask is placed on the glass door frame, but it should be moved to the shower control handle where the hand is actually touching in the reference image. The red and green foot masks are correctly placed on the shower floor.

Use the original Canvas image as the base again.
If a previous composite is provided, use it only as correction context.
Do not use the previous composite or generated image as the base.
Keep the canvas unchanged.
Only fix the specified colored contact masks.
