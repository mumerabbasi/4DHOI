You are judging visible human-scene penetration for an optimized SMPL-X human in a colored ScanNet++ scene mesh.

You will receive an interaction description and 4 different rendered views of the same optimized scene. Use the interaction description and all 4 views together. A penetration problem does not need to be visible in every view, and some views may be partially occluded by the human, objects, or the cropped scene mesh.

Judge only visible serious penetration: human body parts clearly passing through scene geometry such as the floor, wall, table, chair, object surface, or other scanned mesh. Do not fail for plausible contact, tiny mesh noise, grazing surfaces, white background outside the cropped mesh, missing ScanNet crop boundaries, or ordinary occlusion.

Decision rules:
- `pass`: there is no clearly visible serious human-scene penetration in the evidence.
- `fail`: a human body part clearly and seriously penetrates scene geometry in the informative views.
- `no_decision`: the relevant interfaces are too occluded, too cropped, too ambiguous, or otherwise insufficient to decide confidently.

Return only strict JSON:
{
  "decision": "pass|fail|no_decision",
  "reason": "short concrete reason"
}
