You are evaluating colored contact-region masks for a human-object interaction task.

You are given three images:

Image 1: Reference Image.
This image shows a human physically interacting with the target object. Use it to infer which body parts are touching which object surfaces.

Image 2: Original Canvas Image.
This image shows the same scene without the human. This is the authoritative geometry and base image.

Image 3: Current Composite Image.
This is the original Canvas image with colored contact masks pasted onto it. Evaluate only the colored masks.

Target object:
{target_object}

Color mapping:
{color_mapping}

Task:
Decide whether the colored masks in the Current Composite Image correctly mark the localized contact regions implied by the Reference Image.

Focus especially on location. For each mask, check whether it is on the correct object part and correct local surface.

Evaluate these possible errors:
1. Missing contact: a body part touches the target object in the Reference Image but its mask is absent.
2. Extra contact: a mask exists for a body part that does not touch the target object.
3. Wrong color: a contact uses the wrong color for the body part.
4. Wrong object surface: the mask is on the wrong object part, such as wrong rung, wrong rail, wrong edge, wrong handle, wrong shelf, wrong tabletop region, etc.
5. Shifted mask: the mask is too far left, right, up, or down from the correct contact location.
6. Size error: the mask is too large or too small for the local contact footprint.

For every problem, write a concrete correction instruction for the next ChatGPT image-generation prompt.

Good correction examples:
- "Keep the original canvas unchanged and move only the green right-foot mask upward onto the lower ladder rung."
- "The yellow left-hand mask is missing. Add a small yellow mask on the vertical side rail where the left hand contacts the ladder."
- "The blue foot mask is on the wrong rung. Move it down to the rung directly under the visible foot contact."
- "The red right-hand mask is too large. Keep its center but make it smaller and more focused."
- "This mask is correct. Keep it unchanged."

Bad correction examples:
- "The result is bad."
- "Fix the mask."
- "Try again."

Return strictly valid JSON only:

```json
{
  "done": false,
  "confidence": 0.0,
  "correction_instruction": "text"
}
```

Set "done": true only if all required contacts are present, there are no extra masks, each mask uses the correct color, each mask is on the correct target-object surface, and each mask has acceptable location and size.
