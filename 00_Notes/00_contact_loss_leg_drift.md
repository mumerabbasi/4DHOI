# Why Hand Contact Loss Can Make Legs Drift

This note explains the bad deformation seen in the `A0_nocontact` ablation, where only the no-contact/contact-distance objective is active and the human pose becomes implausible, including a leg floating in the air.

## Short Version

The optimizer is not directly trying to lift the leg. The active loss only asks the hand vertices to move closer to the target table contact region. However, because SMPL-X is not a pure kinematic skeleton, hand-contact loss can still produce small gradients on lower-body pose parameters.

Those small gradients mostly come through SMPL-X pose-corrective blend shapes (`posedirs`). Lower-body joint rotations can slightly change the final mesh surface outside the lower body. So even a hand-only vertex loss can have nonzero derivatives with respect to leg joints.

With no pose prior, no floor contact loss, and no lower-body constraint, those small gradients are free to accumulate. Adam can keep applying updates along these weak directions for many iterations, so the legs drift even though the hand loss is the main objective.

## Optimization Landscape Intuition

In the `A0_nocontact` objective, the loss landscape only cares about this:

```text
make the hand contact vertices close to the target table contact points
```

It does not care about:

```text
keep the pose close to GVHMR
keep the feet on the floor
keep the body human-plausible
preserve the lower-body pose
```

So the landscape has a broad valley: many different body poses give nearly the same hand-table contact loss. Some of those poses are plausible, and some are physically absurd. Without another term to select the plausible solution, the optimizer can slide sideways in that valley.

## Why Leg Parameters Receive Gradients

If SMPL-X were only a skeleton with local skinning, changing a knee angle should not affect the hand vertices. But SMPL-X also applies pose-dependent corrective deformations. These corrective bases are learned mesh offsets driven by joint rotations.

That means a lower-body joint can slightly affect the global mesh shape. The effect is tiny compared with spine or arm joints, but it is not zero. Autograd therefore sees a weak slope in leg-pose directions, even though the measured contact points are on the hands.

In the A0 debug probe, the hand/table loss produced large gradients on spine and arms, but also small nonzero gradients on lower-body joints. When `smplx_layer.posedirs` was zeroed, those lower-body gradients almost disappeared. That identifies SMPL-X pose correctives as the main gradient leakage path.

## Why Tiny Gradients Become a Large Deformation

The optimizer uses Adam over all body-pose parameters. Adam rescales parameter updates based on running gradient statistics. A tiny but consistent gradient can still move a parameter noticeably if no loss term pushes it back.

So the process is:

1. Hand-table loss gives strong gradients to arms/spine.
2. The same loss gives tiny gradients to leg joints through SMPL-X pose correctives.
3. No pose prior, angle prior, floor term, or lower-body lock opposes those leg updates.
4. Adam keeps stepping for many iterations.
5. The lower body drifts into a weird but low-loss configuration.

This is why the final pose can be far from GVHMR even though the initial pose was already plausible.

## Practical Takeaway

The floating leg is not a sign that the floor/contact target is explicitly pulling the leg upward. It is a sign that the objective is underconstrained.

For a hand-object contact optimization, use at least one of these guards:

- keep a GVHMR pose prior on non-contact joints
- freeze lower-body pose when optimizing hand contact
- keep a floor contact or foot grounding term active
- add an angle or plausibility prior
- use different optimizer parameter groups or smaller learning rates for unrelated joints

The cleanest interpretation is:

```text
The hand-contact loss creates a shallow downhill direction in leg-pose parameters through SMPL-X pose correctives. Since A0 has no term that says "do not move the legs," Adam follows that direction until the lower body drifts.
```
