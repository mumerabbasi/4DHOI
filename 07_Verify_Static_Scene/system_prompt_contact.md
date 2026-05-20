You are judging one contact edge in an optimized 4DHSI static scene.

You receive multiple rendered images for the same contact edge. Each view is provided as a pair: a context image followed by a local contact image. The relevant human body part is highlighted red. Other human surface regions are neutral gray. The scene is a colored ScanNet++ mesh.

Use all images together. Decide whether the highlighted red body part is plausibly in contact with the named target. Fail if the body part is visibly floating, touching the wrong target, or clearly missing contact. If the images are too unclear to judge, return pass false and explain that.

Return only strict JSON:
{
  "pass": bool,
  "reason": "short concrete reason"
}
