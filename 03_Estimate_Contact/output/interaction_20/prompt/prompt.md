Provided Images:

Reference Image: An image showing a human interacting with the target object. Use this image only to infer which human body parts are physically contacting the target object and which object-surface regions are being touched.

Canvas Image: An image of the same scene without the human. Use this image as the base image and as the authoritative reference for the target object’s final position, pose, geometry, and boundaries.

Task Description:

Generate an edited copy of the Canvas Image with solid-color segmentation masks overlaid only on the target object surface regions that are contacted by the human in the Reference Image.

Infer the contacted region relative to the target object in the Reference Image, then place the corresponding contact mask on the matching object region in the Canvas Image.

Analysis Requirements:

1. Contact Inference:
   Analyze the Reference Image to determine which specified human body parts are physically touching the target object.

2. Object-Region Identification:
   For each contacting body part, identify the specific region of the target object being touched, such as the tabletop front edge, tabletop side edge, tabletop upper surface, tabletop underside, table leg, chair seat, chair arm, cabinet handle, or other relevant object part.

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
Use the Canvas Image as a fixed camera reference. Preserve the camera pose,
field of view, perspective, resolution, aspect ratio, room geometry, lighting,
shadows, textures, and all existing object identities.
Keep the target object in exactly the same pose, position, orientation, scale,
material, and appearance as in the Canvas Image. Do not move, rotate, deform,
lift, translate, resize, crop, pad, or reposition the target object or any
other scene element in any way.
Only add the specified solid-color contact mask overlays. Leave every
non-overlay pixel visually unchanged from the Canvas Image.

Colors:
Use only the exact specified solid colors for the contacting body-part masks. Do not use transparency, antialiasing gradients, shadows, blended colors, or pastel variants.

Restrictions:
Do not add the generated human body.
Do not copy or paste the full visible hand, foot, hip, arm, leg, or other body-part silhouette.
Do not transfer masks by raw Reference Image pixel coordinates.
Do not add text labels, arrows, legends, boxes, keypoints, outlines, or any other annotations.
Show only the original Canvas Image with solid localized contact masks overlaid on the target object.

Example Guidance:
If the Reference Image shows a person lifting a table with both hands, infer which parts of the table the hands are contacting, such as the front edge or underside of the tabletop. Then place solid masks on those same table regions in the Canvas Image, aligned to the table as it appears in the Canvas Image.

Color Mapping for ContactMasks:
Hips: The hips is in contact with the target object. Use solid #0000FF / RGB(0, 0, 255) (pure blue).
Left Leg: The left leg is in contact with the target object. Use solid #FF00FF / RGB(255, 0, 255) (pure magenta).
Right Leg: The right leg is in contact with the target object. Use solid #00FF00 / RGB(0, 255, 0) (pure green).
Right Hand: The right hand is in contact with the target object. Use solid #00FFFF / RGB(0, 255, 255) (pure cyan).
