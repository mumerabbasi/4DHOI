Attached images are provided in this order:
Image 1: 01_reference_image.png - Reference Image
Image 2: 02_canvas_image.png - Canvas Image
Image 3: 03_previous_composite.png - Previous Composite

Provided Images:

Image 1: Reference Image
An image showing a human interacting with the target object. Use this image only to infer which human body parts are physically contacting the target object and which object-surface regions are being touched.

Image 2: Canvas Image
An image of the same scene without the human. Use this image as the base image and as the authoritative reference for the target object's final position, pose, geometry, and boundaries.

Optional Image 3: Previous Composite
If provided, this image shows the previous round's Canvas Image with generated colored contact masks. Use it only as correction context for the VLM evaluator's feedback. If the evaluator says its masks are correct, copy the correct masks from it to the corresponding locations onto the Canvas Image. Also, draw the new corrected masks, according to the evaluator instructions, onto the Canvas Image.

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

Color Mapping for ContactMasks:
Left Hand: The left hand is in contact with the target object. Use solid #FFFF00 / RGB(255, 255, 0) (pure yellow).
Right Hand: The right hand is in contact with the target object. Use solid #FF00FF / RGB(255, 0, 255) (pure magenta).
Left Foot: The left foot is in contact with the target object. Use solid #FF0000 / RGB(255, 0, 0) (pure red).
Right Foot: The right foot is in contact with the target object. Use solid #00FF00 / RGB(0, 255, 0) (pure green).

Required target-object contacts from SIG:
- Left Hand: required contact with target object 'ladder'. Use #FFFF00 / RGB(255, 255, 0) (pure yellow). SIG note: the left hand grips the ladder frame
- Right Hand: required contact with target object 'ladder'. Use #FF00FF / RGB(255, 0, 255) (pure magenta). SIG note: the right hand grips the ladder frame
- Left Foot: required contact with target object 'ladder'. Use #FF0000 / RGB(255, 0, 0) (pure red). SIG note: the left foot remains on a ladder step while stepping down
- Right Foot: required contact with target object 'ladder'. Use #00FF00 / RGB(0, 255, 0) (pure green). SIG note: the right foot remains on another ladder step while stepping down
- Interaction description: A person steps down from the ladder in a cautious descending pose. The left hand grips the ladder frame, the right hand grips the ladder frame, the left foot remains on a ladder step, and the right foot remains on another ladder step. The person has short black hair, a careful facial expression, a casual shirt, pants, and socks or soft shoes.

Restrictions:
Do not add the generated human body.
Do not copy or paste the full visible hand, foot, hip, arm, leg, or other body-part silhouette.
Do not transfer masks by raw Reference Image pixel coordinates.
Do not add text labels, arrows, legends, boxes, keypoints, outlines, or any other annotations.
Show only the original Canvas Image with solid localized contact masks overlaid on the target object.

Example Guidance:
If the Reference Image shows a person gripping a ladder, infer which rungs or rails the hands and feet contact. Then place solid masks on those same ladder regions in the Canvas Image, aligned to the ladder as it appears in the Canvas Image.

Correction Instructions from an Evaluator:
Keep the yellow left-hand mask on the left vertical ladder rail. Move the magenta right-hand mask down onto the upper horizontal rung/ladder frame where the right hand is actually gripping, not on the wall/background. Move the red left-foot mask onto the upper visible foot-contact rung at the left-center of the ladder, aligned with the rung surface. Move the green right-foot mask onto the lower step/rung where the lower foot rests, slightly more to the center/right and on the horizontal rung surface. Keep all masks small and focused on the contact footprints only.
