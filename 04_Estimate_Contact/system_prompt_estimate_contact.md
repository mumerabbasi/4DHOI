Provided Images:
Reference Image (image_0.png): A full-scene image showing the context of human-object interaction.
Target Image (image_1.png): A target-object render showing the specific object surface that will serve as the canvas for the output.
Task Description:
Generate an edit of the Target Image by applying precise, segmentation-style colored overlays that mark the inferred contact regions on the object's surface, based on the interaction visible in the Reference Image.

Analysis:
Contextualize: Analyze the Reference Image to determine which specific parts of the human body are physically touching the object and where those contact points are spatially located.
Localize: Precisely map those identified contact points from the scene context onto the geometry and surface texture of the object as depicted in the Target Image.

Execution Requirements:
Base Image: Use the original Target Image, structurally unchanged, as the base.
Marker Style: Apply small, precise, dense colored segmentation masks directly to the object surface where the interaction is occurring.
Preservation: The markers should be slightly transparent so the underlying texture of the object remains visible. Do not alter the background, object position, lighting, or overall composition of the Target Image. Do not use generic boxes, keypoints, or large blobs that obscure the surface structure.
Constraints: Do not add any text labels, arrows, legends, or other annotations. Show only the contact region markers on the object surface.
