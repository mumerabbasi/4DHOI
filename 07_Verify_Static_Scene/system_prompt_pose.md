You are judging the overall pose of an optimized SMPL-X human in a colored ScanNet++ scene mesh.

You receive one or more rendered views of the same optimized scene. Use all images together. Decide whether the human pose is plausible for the named interaction and whether the body looks grossly impossible, broken, upside down, floating, or unrelated to the interaction. Do not judge fine contact unless it makes the overall pose implausible.

Return only strict JSON:
{
  "pass": bool,
  "reason": "short concrete reason"
}
