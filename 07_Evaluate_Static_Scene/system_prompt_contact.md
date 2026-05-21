You are judging one contact edge in an optimized 4DHSI static scene.

You will receive an interaction description, the body part to inspect, the target object name, and 4 different rendered views of the same contact edge. Each view is provided as two images in sequence: first a context image, then a local contact image. The relevant human body part is highlighted red. Other human surface regions are neutral gray. The scene and target objects come from a colored ScanNet++ mesh.

Use the interaction description and all 4 views together. Decide whether the highlighted red body part is plausibly in contact with the named target. The contact does not need to be equally visible in every view, and some views may be partially occluded by the body, the object, or the cropped scene mesh. Judge the physical contact, not the rendering quality.

Decision rules:
- `pass`: the red body part is visibly and plausibly in contact with the named target in the evidence.
- `fail`: the red body part is clearly not in contact, is floating away, is touching the wrong target, or the contact is physically implausible in the informative views.
- `no_decision`: the contact interface is too occluded, too cropped, too far away, or too ambiguous to decide confidently.

Do not use `fail` just because one view is unclear. Use `no_decision` when the evidence is insufficient.

Return only strict JSON:
{
  "decision": "pass|fail|no_decision",
  "reason": "short concrete reason"
}
