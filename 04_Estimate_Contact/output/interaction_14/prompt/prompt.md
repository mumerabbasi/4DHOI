Provided Images:
Reference Image (image_0.png): A cropped inpainted frame showing the generated human interacting with the scene. Use this image only to infer which human body parts touch the target object and where those contacts occur.
Canvas Image (image_1.png): A cropped original ScanNet scene image. Use this image only as the spatial coordinate reference for where masks should be placed.

Task Description:
Generate an edited copy of the Canvas Image with solid-color segmentation masks overlaid for the specified human body parts that are in contact with the target object in the Reference Image. For each contacting body part, use the visible body-part mask from the Reference Image as the mask shape, then place that same mask at the matching image location in Canvas Image coordinates.

Analysis:
Contextualize: Analyze the Reference Image to determine which specified human body parts are physically touching the target object.
Localize: Transfer the visible masks of those contacting body parts from the Reference Image onto the matching image locations in Canvas Image coordinates.

Execution Requirements:
Base Image: Return the Canvas Image with contact masks overlaid on top of it.
Canvas Size: Match the Canvas Image dimensions and coordinate system as closely as possible.
Marker Style: Apply precise, dense, solid-color segmentation masks shaped like the visible contacting body parts in the Reference Image. Do not invent a small point, blob, footprint, outline, or approximate patch; copy the visible mask shape of the contacting hand, foot, hips, or other specified part.
Scene Preservation: Leave all non-contact pixels visually unchanged from the Canvas Image.
Colors: Use only the exact solid colors listed below for contacting body-part masks. Do not use transparency, antialiasing gradients, shadows, blended colors, or pastel variants.
Constraints: Do not add the generated human body, text labels, arrows, legends, boxes, keypoints, outlines, or any other annotations. Show only the original Canvas Image with the colored body-part masks overlaid for the specified contacting parts.

Color Mapping for Segmentation Masks:
Left Foot: The left foot is in contact with the target object. Copy the visible left foot mask shape from the Reference Image onto the matching location in the returned overlay image using solid #FF0000 / RGB(255, 0, 0) (pure red).
Right Foot: The right foot is in contact with the target object. Copy the visible right foot mask shape from the Reference Image onto the matching location in the returned overlay image using solid #00FF00 / RGB(0, 255, 0) (pure green).
Right Hand: The right hand is in contact with the target object. Copy the visible right hand mask shape from the Reference Image onto the matching location in the returned overlay image using solid #0000FF / RGB(0, 0, 255) (pure blue).
