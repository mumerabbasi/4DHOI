You are evaluating one contact edge in a 4DHSI optimized static scene.

You receive two images for one camera turn of a multi-view contact check.

Image 1 is a 1280x720 context render where the full optimized SMPL-X human should be visible with the interaction object. Image 2 is a 1280x720 optical-zoom render centered on the full visual body segment for this edge, for example `left_hand` rather than the smaller optimizer contact segment `left_hand_inner`. The body segment being judged is colored red. Other human surface regions are neutral gray. The colored background geometry is the ScanNet++ scene mesh. Camera turns may include the original camera and synthetic orbit cameras.

Judge only whether the red human body part is plausibly in contact with the specified target object or scene part. Use the context render to understand the target object and the optical-zoom render to judge contact. Use the provided contact distance as supporting evidence, but rely on the images for visible mistakes such as floating, wrong target, or clearly missing contact.

Return only strict JSON:
{
  "score_0_to_5": int,
  "pass": bool,
  "failure_tags": [str],
  "reason": "short concrete reason"
}

Use failure tags from: missing_contact, floating, wrong_target, occluded_unclear, needs_review.
