You are evaluating scene penetration for an optimized SMPL-X human placed in a colored ScanNet++ scene mesh.

The render is from the original ScanNet++ camera view used by Module 1. Use the image and the provided SDF/inside-point metrics to judge whether the human visibly or metrically penetrates scene geometry in a serious way.

Minor ambiguity from occlusion is acceptable. Fail only for clear severe penetration or metrics that indicate severe penetration.

Return only strict JSON:
{
  "score_0_to_5": int,
  "pass": bool,
  "failure_tags": [str],
  "reason": "short concrete reason"
}

Use failure tags from: severe_penetration, visible_penetration, occluded_unclear, needs_review.
