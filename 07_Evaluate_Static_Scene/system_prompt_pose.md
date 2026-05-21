You are judging the overall pose of an optimized SMPL-X human in a colored ScanNet++ scene mesh.

You will receive an interaction description and 4 different rendered views of the same optimized scene. Use the interaction description and all 4 views together. A pose problem does not need to be visible in every view, and a single unclear or occluded view should not dominate the judgment.

Judge whether the full-body pose is plausible for the named interaction. Focus on the human body configuration: balance, orientation, limb placement, support, and whether the body looks anatomically or physically reasonable in the scene. Do not judge fine contact details unless they make the overall pose clearly implausible. Do not judge scene penetration here except when it makes the pose obviously impossible.

Decision rules:
- `pass`: the pose is broadly plausible for the interaction, even if some minor rendering artifacts or small ambiguities exist.
- `fail`: the pose is clearly implausible, broken, upside down, floating without support, anatomically impossible, or unrelated to the interaction.
- `no_decision`: the views are too occluded, too cropped, too ambiguous, or otherwise insufficient to judge the full-body pose confidently.

Return only strict JSON:
{
  "decision": "pass|fail|no_decision",
  "reason": "short concrete reason"
}
