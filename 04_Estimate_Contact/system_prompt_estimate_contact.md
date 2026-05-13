Provided Images:
Reference Image (image_0.png): A cropped inpainted frame showing the generated human interacting with the scene. Use this image only to infer which human body parts touch the target object and where those contacts occur.
Canvas Image (image_1.png): A cropped original ScanNet scene image. This is the image to edit and return.

Task Description:
Generate an edit of the Canvas Image by applying precise, segmentation-style colored overlays for the specified human body parts that are in contact with the target object in the Reference Image. For each contacting body part, use the visible body-part mask from the Reference Image as the overlay shape, then place that same mask at the matching image location on the Canvas Image.

Analysis:
Contextualize: Analyze the Reference Image to determine which specified human body parts are physically touching the target object.
Localize: Transfer the visible masks of those contacting body parts from the Reference Image onto the matching image locations in the Canvas Image.

Execution Requirements:
Base Image: Use the Canvas Image, structurally unchanged, as the base output.
Marker Style: Apply precise, dense colored segmentation masks shaped like the visible contacting body parts in the Reference Image. Do not invent a small point, blob, footprint, outline, or approximate patch; copy the visible mask shape of the contacting hand, foot, hips, or other specified part.
Preservation: Preserve the Canvas Image exactly except for the colored contact masks. Do not change object geometry, object pose, background, lighting, camera framing, texture, or any scene content.
Constraints: Do not add a human or any scene content. Do not add text labels, arrows, legends, boxes, keypoints, outlines, or other annotations. Show only the colored body-part masks for the specified contacting parts.
