# Comparison: Our Static Scene Interaction vs. Open-Vocabulary HSI

## Shared Goal

Both methods aim to generate a plausible 3D human interaction in a scene from a text prompt.

In simple terms:

- input: a scene and a text instruction
- output: a posed SMPL-X human that appears to interact naturally with the scene
- main challenge: the human should make contact at the right places, avoid obvious penetration, and keep a plausible body pose

This note compares the interaction-generation/refinement idea.

## Open-Vocabulary HSI Method

The open-vocabulary method starts from a scene and a text prompt, then uses vision-language reasoning to identify functional scene elements and likely body-part contacts.

For example, for a prompt like:

```text
a person opens the door
```

it tries to infer:

- which scene element matters, such as the door handle
- which human body part should contact it, such as the right hand
- how the body should be placed near that element

Then it fits/refines an SMPL-X human using contact, collision, and pose-prior losses.

The important physical idea is that collision is handled by querying scene points against a human-body SDF. This is useful because the human body is much closer to a clean closed shape than a raw scanned scene mesh.

## Our Current Method

Our method also starts from a scene and a text prompt, then generates a human in the scene image and reconstructs/refines the SMPL-X human in 3D.

The current refinement focuses on:

- matching the human mask
- pulling selected body parts toward estimated contact regions
- using the same contact loss for target-object and floor interaction edges
- keeping pose and height plausible
- discouraging visible scene surface points from intersecting the human body
- preventing SMPL-X self-intersection

The current method is more direct and practical: estimate the contact regions, then optimize the first-frame SMPL-X human so the mesh agrees with those regions.

## Main Similarities

Both methods use the same broad recipe:

```text
text/scene understanding
-> initial human placement
-> SMPL-X refinement
-> contact + collision + pose plausibility
```

Both methods also treat SMPL-X as the human representation and rely on optimization after the initial generation/estimation step.

## Main Differences

### Contact Reasoning

Open-vocabulary HSI uses a more general semantic reasoning module to infer functional scene contacts from text.

Our method currently uses a more explicit contact setup: body-part contact masks and target contact regions are estimated, then used directly in optimization.

### Collision Handling

Open-vocabulary HSI uses a human-body SDF and asks:

```text
are scene points inside the human?
```

Our current script now mostly asks:

```text
are visible scene surface points inside the human?
```

This is useful because scanned scene meshes are often not watertight. A scene SDF can give unreliable inside/outside signs when the mesh has holes or inconsistent normals.

Using a human-body SDF is cleaner for scanned scenes because SMPL-X is a much more reliable closed body than the ScanNet++ scene geometry.

### Scope

Open-vocabulary HSI is designed to handle many interaction types and functional scene elements.

Our current method is narrower and more controlled. It is focused on getting one generated human interaction to align well with a chosen scene/object setup.

### Practicality

Our current losses are easier to implement and debug.

The open-vocabulary approach is more principled, but depends on stronger components:

- reliable functional scene understanding
- good interaction graph prediction
- differentiable human-body SDF
- stronger pose priors

## What We Should Borrow

The most useful idea to borrow is the collision direction:

```text
query scene surface points against the current SMPL-X human body
```

instead of relying only on:

```text
query SMPL-X vertices against a scanned target/scene SDF
```

For our case, a good next loss would be:

```text
visible scene surface points should not be inside the human body
```

This should stay local and semantic. We should avoid applying a noisy full-room collision loss blindly.

## Simple Takeaway

Our method is a practical first-frame grounding pipeline.

The open-vocabulary HSI method is a more general interaction synthesis framework.

The biggest technical lesson for us is collision handling: for scanned, non-watertight scenes, it is usually better to use the SMPL-X body as the signed volume and test nearby scene points against it.
