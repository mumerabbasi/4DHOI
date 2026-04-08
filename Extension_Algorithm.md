# Algorithm: Scene-Conditioned 4DHOI in ScanNet++

## Goal

Input:

- a text prompt
- a ScanNet++ scene
- a chosen camera view
- a chosen target object instance

Output:

- a human interacting with the chosen object in the 3D scene
- recovered human motion
- recovered object motion if the object moves
- both aligned and refined in a shared scene frame

Assume:

- single human
- single target object
- static camera
- manually chosen scene, view, and target object

## Main Flow

### 1. Load scene and target object

Load the chosen ScanNet++ scene and camera view, then extract:

- scene RGB image
- scene depth source for that view
- camera intrinsics and extrinsics
- target object mesh
- projected target mask and bbox in the chosen image

This gives a fixed scene anchor before generation begins.

### 2. Build a target-aware first-frame editing input

Instead of asking the model to guess where the interaction should happen, explicitly mark the chosen object in the scene image.

For now:

- highlight the selected target object in the image
- define a local edit mask around the target object
- keep the rest of the room unchanged

The purpose is only to tell the image-editing model:

- which object is the target
- where a human may appear

The model can decide the exact human pose, scale, occlusion, and local contact layout.

### 3. Generate the first frame by inpainting a human near the target object

Replace the old generic first-frame generation with local scene editing.

Recommended approach:

- use a FLUX inpainting or editing model
- input the real scene image
- input the local edit mask around the chosen object
- prompt for a human interacting with the highlighted object

Example interaction prompt:

- `insert one full-body person lifting the highlighted chair with both hands; preserve the exact room layout, preserve all other furniture, remove the highlight marker`

Generate multiple candidates and rerank them by:

- full human visibility
- clear interaction with the chosen object
- low background drift
- no extra people

This first frame becomes the animation anchor.

### 4. Animate the interaction video

Use the selected edited first frame as input to the existing animation stage.

Recommended setup:

- keep the current `Wan2.2 I2V 14B` setup
- use a static-camera prompt
- ask for slow motion and minimal scene changes

Example high-level prompt constraints:

- same room
- same object
- static camera
- other furniture unchanged
- only the intended interaction occurs

This produces a scene-conditioned interaction video rather than a free-floating one.

### 5. Segment the human and the target object

After video generation:

- segment the human as usual
- segment the target object using the known target-object projection as guidance

Important:

- do not treat this as generic `segment all chairs`
- the target object is already known
- use its projected region to isolate that specific object in frame 0
- then propagate or track that object mask through the video

### 6. Recover human motion

Run the existing human-motion recovery stage:

- segment human
- run `GVHMR`
- export human mesh sequence

This reuses the strongest part of the current human side of the pipeline.

### 7. Recover object motion

If the object is static support geometry, it can stay fixed in the scene.

If the object is manipulated, reuse the current optical-flow-based object-motion pipeline:

- use the selected ScanNet++ object mesh as the object template
- use target-object masks and optical-flow cues from the generated video
- recover the object transform sequence over time

This is especially important for interactions like:

- lifting a chair
- carrying a chair
- pulling a chair

### 8. Align human and object into a shared scene frame

Use the current depth-based alignment logic as the conceptual base:

- align first-frame human mesh to scene depth
- align the target object mesh to the same scene depth
- initialize both in the same camera and scene frame

Depth source:

- rendered DSLR depth for the chosen DSLR view, or
- native iPhone depth if using the iPhone stream

This keeps the extension compatible with the current mesh-depth alignment setup.

### 9. Refine with scene-aware losses

After initialization in a shared frame, refine the motion with:

- human-object contact losses
- object smoothness and tracking losses
- human-object intersection losses
- human-scene penetration losses
- object-scene penetration losses
- floor and support consistency losses

This stage extends the current joint refinement rather than replacing it.

### 10. Export the final 3D interaction

Final outputs should include:

- human mesh sequence
- object transform sequence
- aligned scene frame
- renders or overlays for verification

## Pipeline Changes

The main flow above still maps cleanly onto the current repo with a few targeted changes.

### Reuse strongly

- `Generate_Video/` for animation
- `Segment_Video/` for human and object masks
- `Estimate_Human_Motion/` for `GVHMR`
- `Estimate_Optical_Flow/` and `Track_Object_Mesh/` for object motion
- `Align_Meshes/` for first-frame shared-frame alignment
- `Track_Human_Object_Mesh/` for final refinement

### Add new scene-aware pieces

- scene loading and instance extraction from ScanNet++
- target-object projection in the selected view
- target-marked first-frame editing
- object-to-scene-instance bookkeeping
- scene-aware refinement losses

### Replace for scene mode

- replace generic first-frame generation with scene-conditioned inpainting or editing
- replace estimated depth with scene depth from ScanNet++
- replace generic object template generation with the chosen ScanNet++ object mesh

## Current Scope

Prioritize interactions with simple or nearly constant contact structure:

- lifting a chair
- carrying a chair a short distance
- leaning on a table
- holding or moving a simple object

Harder interactions like `pull chair and then sit` should be treated as later work because they need time-varying contact schedules.

## Future Extensions

- automate scene/view/object selection with a VLM
- add stronger pose priors for first-frame insertion if prompt-only editing is unstable
- add time-varying or phase-wise PAG/contact schedules
- support harder multi-stage interactions such as pulling then sitting
