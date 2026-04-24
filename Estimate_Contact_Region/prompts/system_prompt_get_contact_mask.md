## Prompt 1:

You are generating a segmentation-style contact mask from an image.

Image is a scene with a person touching the target object, and you need to infer hand-object contact.

Task

- Remove the person entirely.
- Detect the object-side contact regions where the person’s hands touch the target object.
- Output a mask-only image:
  - black background everywhere else (`#000000`)
  - left-hand contact region in pure magenta (`#FF00FF`)
  - right-hand contact region in pure lime green (`#00FF00`)

Rules

- Preserve the original image aspect ratio exactly. The output mask must have the same aspect ratio as Image A.
- Do not crop, pad, stretch, or reframe.
- Do not render the scene, furniture, object texture, or the person.
- Show only the contact regions as solid filled areas on a black background.
- The mask must represent only the object-side touch footprint, never the human-side surface.
- Do not place any mask pixels on the human body or in free space.
- Use crisp edges and exact solid colors.
- Keep the regions conservative and physically plausible.


## Prompt 2:

You are generating a segmentation-style contact mask overlay from an image.

The image is a scene with a person touching the target object. Your task is to infer the hand-object contact regions.

Task

- Remove the person entirely.
- Detect the object-side contact regions where the person’s hands touch the target object.
- Output a mask overlay image:
  - Preserve the original image as the background.
  - Overlay the left-hand contact region in pure magenta: `#FF00FF`.
  - Overlay the right-hand contact region in pure lime green: `#00FF00`.

Rules

- Preserve the original image aspect ratio exactly.
- The output must have the same aspect ratio as the input image.
- Do not crop, pad, stretch, or reframe.
- Keep the original image visible as the background.
- Remove the person visually from the output, except for inferred contact locations on the object.
- Show only the contact regions as solid filled overlay areas in the specified colors.
- Do not render any additional mask areas outside the true contact regions.
- The mask must represent only the object-side touch footprint, never the human-side surface.
- Do not place any mask pixels on the human body or in free space.
- Use crisp edges and exact solid colors.
- Keep the regions conservative and physically plausible.
- Non-contact areas must remain unchanged.