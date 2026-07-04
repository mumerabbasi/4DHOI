Attached images are provided in this order:
{candidate_listing}

Provided Images:

Each image is a candidate generated human-frame result for the same fixed scene and the same human-object interaction.

Task Description:

Select the single candidate image that best matches the intended interaction and is most usable as the human frame for downstream contact estimation and static-scene optimization.

Interaction:
{interaction}

Evaluation Requirements:

1. Human Visibility:
   Prefer images where the main human is fully visible from head to feet, clearly inside the frame, and not cropped by image borders.

2. Human Anatomy:
   Prefer normal human body proportions and body parts. Reject candidates with distorted, duplicated, missing, or anatomically implausible limbs, hands, feet, torso, or head.

3. Interaction Match:
   Prefer the image where the human-object interaction most directly matches the interaction text.

4. Contact Correctness:
   Prefer physically correct human contact with the target object. The contacting body parts should touch the intended object surface or object part without gaps, floating limbs, penetration artifacts, or impossible support.

5. Plausible Pose and Support:
   Prefer a natural, stable pose that could occur immediately before the interaction motion begins. Avoid floating humans, unsupported body weight, impossible leaning, and unnatural body-object relationships.

6. Scene and Object Preservation:
   The scene geometry must remain unchanged. The target object of interaction must keep the same shape, texture, appearance, pose, position, orientation, scale, material, and visible structure. Reject candidates where the target object is moved, deformed, duplicated, removed, retextured, or replaced.

7. Image Quality:
   Prefer clear, sharp images. Avoid motion blur, smeared body parts, severe generation artifacts, or visual clutter that makes the contact ambiguous.

8. Realistic Style:
   Prefer photographic or realistic images over cartoons, drawings, illustrations, or highly stylized images.

9. Tie Break:
   After detailed inspection, if all candidate images are effectively the same quality and none is clearly better, choose frame_00.png.

Return strictly valid JSON only:

```json
{
  "selected_frame": "frame_XX.png",
  "reason": "short reason"
}
```

The selected_frame value must be exactly one filename from the provided candidate list.
