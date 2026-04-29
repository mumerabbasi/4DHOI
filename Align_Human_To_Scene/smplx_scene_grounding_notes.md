# SMPL-X Scene Grounding Notes

## Goal

This stage optimizes the first-frame SMPL-X human so the generated human-object interaction is grounded in the metric scene. The main target is not just a good 2D overlay, but correct 3D contact: the intended human parts should reach the intended target-object regions while the full body stays plausible.

The current script is:

- `align_human_to_scene_full_body.py`

The current experiments focus on frame 0 for `video_01` and `video_02`.

## Pipeline Context

The input scene image contains a target object. We segment that target object, inpaint a human into the image, and use the generated interaction plus PAG contact semantics to decide which SMPL-X body parts should contact which object regions.

The inpainted image can slightly change the target object's appearance or pose. Because of that, the target mask and generated image are useful diagnostics, but the grounding objective should still be driven by explicit 3D semantic contacts wherever possible.

## Optimized SMPL-X Variables

The optimizer updates:

- `transl`
- `global_orient`
- `body_pose`
- `global_scale`

These variables are enough to move the whole body, rotate the body, articulate the torso/limbs, and absorb small global scale mismatch.

The optimizer keeps fixed:

- `betas`
- left and right detailed hand pose
- face, jaw, and expression parameters
- per-vertex offsets

`betas` are fixed to the GVHMR estimate because they describe body shape and identity, not the interaction pose. Letting `betas` change made the optimization less clean conceptually: the solver could change the person's proportions instead of solving the actual grounding problem with translation, orientation, pose, and scale. Since `betas` are fixed, there is no `betas_gvhmr` loss in the current objective.

Detailed hand pose is also fixed for now. The current contact regions are hand/body-part level, not finger-level. Unlocking detailed hands before the body grounding is stable would add many degrees of freedom that are hard to supervise.

## Current Data Terms

After the first ablation stage, the selected data terms are:

- `nocontact`
- `scene_intersect`
- `floor_nocontact`
- `mask`

### `nocontact`

This is the main semantic contact attraction term. For each PAG contact edge, it pulls the selected SMPL-X contact part toward the corresponding target-object contact region.

Even though the name is `nocontact`, in this script it is the term that encourages contact. We keep the existing name because it matches the original code convention and the ablation output names.

This is the most important term for the core research question: whether the correct human part moves to the correct object region.

### `scene_intersect`

This penalizes scene surface points that lie inside the SMPL-X human SDF, or inside a small clearance margin around the human.

Instead of building an SDF from the scanned target mesh, it uses the human body as the signed volume and checks visible nearby scene/target surface points against that body volume. This is cleaner for ScanNet++ because the human is much closer to watertight than the raw scene/object mesh.

### `floor_nocontact`

This pulls support regions of the body, especially the feet, toward visible floor/support points.

It is kept because object contact alone can produce plausible hand placement while leaving the lower body poorly grounded. Floor contact helps the whole pose sit in the scene rather than only satisfying the hand-object interaction.

### `mask`

This aligns the projected SMPL-X human with the generated human mask in the first frame.

The useful setting from the ablation was `mask_weight = 1`. A much smaller value, such as `1e-3`, was effectively inactive because the scaled mask contribution was tiny compared with the contact and regularization terms.

The mask term is a 2D observation anchor, not the main source of contact correctness. It helps keep the optimized body consistent with the inpainted image while the 3D contact terms determine where the body should touch the object.

## Scene Intersection Status

The old global scene SDF formulation was not reliable enough.

The full-room ScanNet mesh is not a clean watertight collision object. In practice, the global scene SDF marked large parts of the human, including the head and neck, as inside the scene even when the visual mesh did not show meaningful collision there. That made the optimizer rotate or distort upper-body pose to satisfy a bad collision signal.

The current `scene_intersect` term instead follows the open-vocabulary/VolumetricSMPL-style direction: sample visible scene surface points near the body, query those points against the current SMPL-X body SDF, and penalize only scene points that are inside the body or within a small clearance margin.

This keeps the sign source on the human, which is much closer to a clean closed volume than a raw ScanNet++ scene mesh.

The default is intentionally moderate: it ramps from `0` to `10`, like the original GenZI-style intersection schedule, and the loss averages only over active collision/clearance-violation points so the raw value is not diluted by safe scene points.

## Current Regularization Terms

After the second ablation stage, the selected regularization terms are:

- `root_orient_gvhmr`
- `pose_gvhmr`
- `height_prior`
- `angle`

These terms keep the optimized human plausible while still allowing the root position to move enough for natural contact.

### `root_orient_gvhmr`

This keeps the global body orientation close to the GVHMR initialization.

It is useful because the initial body facing direction is usually reasonable, while the global position may need more freedom to move the human closer to the target object.

### `pose_gvhmr`

This keeps the articulated SMPL-X `body_pose` close to the GVHMR initialization.

It is useful because the generated/GVHMR body pose is already a strong estimate of the interaction. The contact losses should refine that pose, not invent a new body configuration from scratch.

### `height_prior`

This keeps the canonical SMPL-X body height close to a physically plausible adult height.

The target height is 6 ft, exactly `1.8288 m`. The allowed scale range corresponds to 5 ft 10 in through 6 ft 2 in, exactly `1.7780 m` to `1.8796 m`.

This replaces the old initial-scale prior. The old prior trusted GVHMR's metric scale; the height prior instead anchors the optimized body to a real-world human-height range while still allowing small scale adjustment.

### `angle`

This anatomical prior discourages implausible joint bends.

It is kept because contact objectives can otherwise solve the problem by using unnatural elbows, knees, shoulders, spine, or neck rotations. The angle prior makes the pose changes more human-like.

## Why Root Freedom Matters

The interaction often looks better when the whole human moves toward the object before the arms deform heavily. If the root is held too tightly to the initial GVHMR placement, the optimizer may stretch an arm to satisfy hand-object contact instead of translating the body closer.

The selected configuration therefore emphasizes contact, scene/body collision handling, floor support, mask alignment, body-pose anchoring, physical height, and anatomical plausibility while leaving enough freedom for the global body placement to adapt to the scene.

## Current Selected Objective

The current preferred objective is:

```text
data:
  nocontact
  scene_intersect
  floor_nocontact
  mask, weight = 1

regularization:
  root_orient_gvhmr
  pose_gvhmr
  height_prior
  angle
```

For reproducing the current selected runs, the corresponding weights are:

```text
--mask_weight 1
--nocontact_weight_start 500
--nocontact_weight_end 500
--floor_nocontact_weight_start 200
--floor_nocontact_weight_end 200
--root_orient_gvhmr_weight 20
--pose_gvhmr_weight 10
--height_prior_weight 1
--height_prior_target_m 1.8288
--height_prior_min_m 1.778
--height_prior_max_m 1.8796
--height_prior_sigma_m 0.0508
--angle_weight_start 0
--angle_weight_end 1
--scene_intersect_weight_start 0
--scene_intersect_weight_end 10
--self_intersect_weight_start 0
--self_intersect_weight_end 0
```

## Ablation Summary

The first ablation stage tested which data terms should be used. The best configuration kept:

- `nocontact`
- `scene_intersect`
- `floor_nocontact`
- `mask` with weight `1`

The second ablation stage tested regularization choices on top of that data-term setup. The chosen configuration keeps the terms that stabilize the body pose, global scale, and joint realism while preserving enough global placement freedom for contact to look natural.

The main qualitative checks for these ablations were:

- the right hand reaches the red target contact region,
- the left hand reaches the green target contact region,
- the body translates naturally toward the object instead of only stretching one arm,
- the feet stay reasonably supported,
- the head and neck do not rotate strangely,
- the final rendered mask remains consistent with the generated human.

## Next Work

A useful direction is a staged optimization schedule:

1. first optimize root placement and scale so the whole body moves toward the object,
2. then unlock full `body_pose` refinement for local contact and posture correction.

This would directly address the case where the hand reaches the object only by overextending the arm.
