You are judging visible scene penetration for an optimized SMPL-X human in a colored ScanNet++ scene mesh.

You receive one or more rendered views of the same optimized scene. Use all images together. Decide whether the human visibly penetrates scene geometry in a serious way. Minor visual ambiguity from occlusion is acceptable. Fail only for clear visible penetration.

Return only strict JSON:
{
  "pass": bool,
  "reason": "short concrete reason"
}
