Provided Images:
Reference Image (image_0.png): A cropped inpainted frame showing the generated human interacting with the scene. Use this image only to infer which human body parts touch the target object and where those contacts occur.
Canvas Image (image_1.png): A cropped original ScanNet scene image. This is the image to edit and return.

Task Description:
Generate an edit of the Canvas Image by applying precise, segmentation-style colored overlays that mark the inferred contact regions on the visible target object surface, based on the human-object interaction visible in the Reference Image.

Analysis:
Contextualize: Analyze the Reference Image to determine which specified human body parts are physically touching the target object.
Localize: Transfer those contact locations onto the matching visible target object surface in the Canvas Image.

Execution Requirements:
Base Image: Use the Canvas Image, structurally unchanged, as the base output.
Marker Style: Apply small, precise, dense colored segmentation masks directly on the visible target object surface where contact occurs.
Preservation: Preserve the Canvas Image exactly except for the colored contact masks. Do not change object geometry, object pose, background, lighting, camera framing, texture, or any scene content.
Constraints: Do not place colored marks on the human, floor, wall, or non-target objects. Do not add text labels, arrows, legends, boxes, keypoints, outlines, or other annotations. Show only the contact region markers on the target object surface.

Color Mapping for Segmentation Overlays:
Hips: Mark this contact region on the visible target object surface using a precise bright violet overlay.
