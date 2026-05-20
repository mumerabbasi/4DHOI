You are evaluating the overall pose of an optimized SMPL-X human placed in a colored ScanNet++ scene mesh.

The render is from the original ScanNet++ camera view used by Module 1. Judge whether the human pose is plausible for the stated interaction and whether the body looks grossly impossible, broken, upside down, or unrelated to the interaction.

Do not judge fine contact here unless it makes the whole pose implausible.

Return only strict JSON:
{
  "score_0_to_5": int,
  "pass": bool,
  "failure_tags": [str],
  "reason": "short concrete reason"
}

Use failure tags from: implausible_pose, wrong_interaction, floating, occluded_unclear, needs_review.
