# 4DHOI

## What This Repo Is

4DHOI is a modular text-to-4D human-object interaction system. The current repo is not a thin wrapper around `Original_Code/`; it is a new pipeline that combines language-driven interaction specification, image/video generation, monocular reconstruction, alignment, tracking, and interaction-aware geometric refinement.

At a high level, the method takes a short interaction prompt and produces a temporally aligned 4D HOI representation:

- a generated interaction video with a locked camera,
- an aligned human mesh sequence,
- aligned object meshes,
- object-part labels on object meshes,
- per-frame object trajectories,
- and a final refined human-object scene sequence in a shared coordinate frame.

In this repo, "4D" means time-varying geometry in a consistent scene frame, not only a video.

## Problem Formulation

The method targets the following problem:

> Given a text description of a human interacting with one or more objects, generate a plausible video of the interaction, recover human and object geometry from that video, and refine the resulting scene so that object motion, contact, part semantics, and geometry become mutually consistent over time.

The current code is best understood as a full-stack HOI bootstrapping pipeline:

1. Convert text into an explicit part-affordance interaction structure.
2. Use that structure to generate a suitable first frame and video.
3. Recover human motion, object geometry, depth, masks, tracks, and mesh parts.
4. Refine object motion with PAG-guided contact and geometric constraints.

## Inputs and Outputs

### Main input

- `Generate_PAG/input_prompts/<video_name>/input_pag.json`

This input provides:

- object names,
- a short interaction description,
- optional prompt-specific metadata depending on the script.

### Main outputs

- `Generate_PAG/output/<video_name>/output_pag_*.json`
- `Generate_Video/output/<video_name>/*.mp4`
- `Align_Meshes/output/<video_name>/human_motion_aligned/`
- `Align_Meshes/output/<video_name>/meshes/*.ply`
- `Segment_Object_Mesh/output/<video_name>/*/segmented_meshes/*_triangle_labels.json`
- `Track_Object_Mesh/output/<video_name>/*/poses.json`
- `Track_Human_Object_Mesh/output/<video_name>/`

## End-to-End Method

### Stage 1: PAG generation from text

Script:

- `Generate_PAG/generate_pag.py`

Purpose:

- Turn a short interaction description into an explicit Part Affordance Graph (PAG).

What it produces:

- object part nodes,
- human body-part nodes,
- interaction edges,
- object motion-state flags,
- a longer interaction prompt for video generation.

Important implementation details:

- PAG generation is not single-shot.
- The script queries an LLM multiple times and aggregates the outputs with majority voting.
- Voting is applied to object-part nodes, body-part nodes, interaction edges, and object motion-state flags.
- The graph distinguishes:
  - `is_continuous`: continuous contact vs intermittent contact,
  - `is_rel_static`: relatively static contact vs sliding/drifting contact.

Why this matters:

- The PAG is the semantic backbone of the whole system.
- It is used both upstream for generation and downstream for geometric refinement.

### Stage 2: first-frame generation

Script:

- `Generate_Video/generate_first_frame.py`

Purpose:

- Generate several candidate first frames that satisfy the interaction prompt.

Important implementation details:

- The script appends an always-on framing suffix.
- That suffix enforces a wide shot with both human and object fully visible.
- The framing is designed so the whole interaction can remain visible without camera movement.
- The current implementation uses `FLUX.1-dev` through diffusers.

Why this matters:

- The whole downstream pipeline assumes a visible human, visible objects, and stable framing.

### Stage 3: first-frame selection

Script:

- `Generate_Video/select_first_frame.py`

Purpose:

- Choose the best first frame among multiple generated candidates.

Important implementation details:

- The script performs a VLM tournament with pairwise image comparisons.
- It compares candidate images side-by-side and advances the winner of each round.
- This reduces reliance on a single noisy ranking call.

Why this matters:

- First-frame quality strongly affects video generation, object reconstruction, and segmentation reliability.

### Stage 4: video generation

Script:

- `Generate_Video/generate_video.py`

Purpose:

- Generate the interaction video from the selected first frame and the PAG interaction text.

Important implementation details:

- The script appends a strong always-on camera-lock suffix.
- It also adds a negative prompt that explicitly suppresses pan, tilt, zoom, dolly, orbit, roll, handheld motion, and perspective drift.
- The current implementation uses an image-to-video diffusion model through diffusers, with `Wan2.2-I2V` as the default.

Why this matters:

- A static camera simplifies monocular depth estimation, alignment, segmentation, tracking, and 2D-3D consistency losses.

### Stage 5: video segmentation

Script:

- `Segment_Video/segment_video.py`

Purpose:

- Recover masks for humans, objects, and object parts across the generated video.

Important implementation details:

- The first frame is parsed with a VLM to obtain open-vocabulary bounding boxes for humans, objects, and object parts.
- SAM3 then propagates masks across time.
- Human part segmentation is not performed here; human body parts are handled later using SMPL vertex groups.

Why this matters:

- These masks are reused by object mesh generation, optical flow seeding, object tracking, and joint refinement.

### Stage 6: first-frame object mesh reconstruction

Script:

- `Generate_Object_Mesh/generate_objects_meshes.py`

Purpose:

- Reconstruct a 3D mesh for each visible object from the first frame and its mask.

Important implementation details:

- The method is first-frame only at this stage.
- Each object is reconstructed independently.
- The script estimates camera intrinsics and saves them for later stages.
- The generated mesh is converted into OpenCV camera coordinates and exported as a posed mesh.

Why this matters:

- This gives the system explicit object geometry early, which later stages can align, segment into parts, and track over time.

### Stage 7: monocular depth estimation

Script:

- `Estimate_Depth/estimate_depth.py`

Purpose:

- Estimate metric depth for the generated video.

Important implementation details:

- The code is built around Depth Anything 3.
- It saves raw depth, normalized depth masks, visualizations, and summaries.
- It also computes camera-related metadata that is reused downstream.

Why this matters:

- Depth anchors the scene geometry in camera space and is the reference for the alignment stage.

### Stage 8: human motion estimation

Scripts:

- `Estimate_Human_Motion/estimate_human_motion.py`
- `Estimate_Human_Motion/export_human_motion_to_ply.py`

Purpose:

- Recover the human motion sequence from the generated video.

Important implementation details:

- The current pipeline uses GVHMR for human motion estimation.
- GVHMR outputs SMPL-X parameters.
- `export_human_motion_to_ply.py` converts them to fixed-topology SMPL meshes with 6890 vertices.
- Because topology is fixed across time, the same human part vertex groups can be reused for the entire sequence.

Why this matters:

- The human sequence supplies the human geometry used in contact reasoning and final visualization.

### Stage 9: mesh alignment into a shared scene frame

Scripts:

- `Align_Meshes/align_meshes.py`
- `Align_Meshes/align_human_motion_sequence.py`

Purpose:

- Move object meshes and the human sequence into the same aligned camera-space scene frame.

Important implementation details:

- Alignment is performed against first-frame depth and masks.
- The method uses 3D and 2D bidirectional chamfer losses.
- A fixed visible subset of sampled mesh points is computed at initialization and reused during optimization.
- This reduces instability from changing visibility assignments during optimization.

Why this matters:

- Alignment is the bridge between monocular reconstruction and later time-consistent tracking/refinement.

### Stage 10: object mesh part segmentation

Scripts:

- `Segment_Object_Mesh/render_mesh_views.py`
- `Segment_Object_Mesh/segment_renders.py`
- `Segment_Object_Mesh/segment_meshes.py`

Purpose:

- Convert aligned whole-object meshes into semantically segmented object-part meshes.

Pipeline:

1. Render each aligned object from multiple views.
2. Segment parts in those rendered images with a VLM plus SAM3.
3. Map 2D masks back to triangles using rendered face IDs.
4. Aggregate multi-view triangle votes and optionally smooth labels.

Why this matters:

- The final refinement stage needs part labels, not just whole-object geometry.
- This is how language-defined PAG parts become 3D mesh parts.

### Stage 11: optical flow / point tracking

Scripts:

- `Estimate_Optical_Flow/estimate_optical_flow_cotracker.py`
- `Estimate_Optical_Flow/estimate_optical_flow_waft.py`

Purpose:

- Recover time-consistent 2D point tracks for each object.

Important implementation details:

- The default branch uses CoTracker3.
- Tracking points are sampled from frame-0 object masks.
- The branch is object-specific rather than scene-wide.
- WAFT is supported as an optional alternative branch.

Why this matters:

- These tracked points serve as the main temporal anchor for object pose estimation.

### Stage 12: object pose tracking

Script:

- `Track_Object_Mesh/track_object_mesh.py`

Purpose:

- Estimate a per-frame `SE(3)` trajectory for each object mesh.

Important implementation details:

- Frame-0 2D seed points are mapped onto the object mesh by rasterization and barycentric interpolation.
- The optimizer estimates an `SE(3)` transform sequence under:
  - robust reprojection loss,
  - velocity smoothness,
  - acceleration smoothness,
  - visibility and mask gating.
- The code supports sequential PnP-RANSAC initialization and track outlier rejection.

Why this matters:

- This turns sparse 2D tracks into explicit 3D object motion.

### Stage 13: PAG-guided human-object refinement

Script:

- `Track_Human_Object_Mesh/track_human_object_mesh.py`

Purpose:

- Refine object motion using human geometry, object geometry, part labels, masks, and PAG contact semantics.

This is the current default refinement stage in the repo.

Important implementation details from the current code:

- The human mesh sequence is fixed at this stage.
- The optimizer refines objects, not human articulation or human global pose.
- This is more precise than older high-level summaries in the repo that described the stage as fully joint human-object refinement.
- Each object gets:
  - per-frame rotation delta,
  - per-frame translation delta,
  - one bounded global uniform scale correction.
- The final object pose is:
  - tracked object pose from `Track_Object_Mesh`,
  - followed by an optimized correction transform,
  - plus an optimized global scale.

Losses in the current code:

- tracking anchor loss on object corrections,
- whole-object 2D chamfer loss against video masks,
- object-part 2D chamfer loss against part masks,
- translational smoothness or staticness depending on PAG object-state flags,
- rotational smoothness or staticness depending on PAG object-state flags,
- bounded object-scale regularization,
- SDF-based human-object intersection penalty,
- contact loss for PAG edges,
- contact-drift loss in canonical object space.

How contact is modeled:

- For continuous contact edges, the loss uses mean nearest-neighbor distance.
- For non-continuous contact edges, it uses a minimum-distance formulation.
- For relatively static contact edges, canonical contact points are encouraged to stay static.
- For dynamic contact edges, canonical contact points are encouraged to evolve smoothly instead.

Why this matters:

- This is where the repo most clearly injects HOI semantics into geometry refinement.

### Stage 14: experimental CoTracker-driven joint branch

Script:

- `Track_Human_Object_Joint/track_human_object_joint.py`

Purpose:

- Provide an alternative refinement branch that starts directly from CoTracker correspondences instead of consuming `Track_Object_Mesh` outputs.

Current status:

- This branch is implemented and runnable.
- It should be treated as experimental relative to `Track_Human_Object_Mesh`.
- Like the default branch, it still keeps the human sequence fixed in the current code.
- Its distinguishing feature is that it uses CoTracker reprojection as the main object-motion data term inside the final refinement stage.

## Key Modeling Choices Encoded in the Repo

### 1. Explicit semantic intermediate: PAG

4DHOI does not go directly from text to geometry. It first builds an explicit interaction graph over object parts and human body parts. This is one of the most important design decisions in the repo because it makes interaction semantics reusable across generation and refinement.

### 2. Strongly constrained camera behavior

The generation stage deliberately suppresses camera motion. This is not only a prompt-engineering trick; it is a systems choice that reduces ambiguity for downstream geometry recovery.

### 3. First-frame-first geometry bootstrapping

Instead of directly reconstructing full 4D object geometry from the video, the repo first reconstructs first-frame object meshes and then tracks/refines them over time.

### 4. Fixed-visible-subset alignment

The alignment stage avoids repeatedly recomputing visibility during optimization. It fixes a visible subset at initialization, then optimizes against that subset. This is a practical stabilizing choice.

### 5. Part-aware object reasoning

Object-part reasoning is not only textual. The repo renders aligned meshes, segments them into parts in 2D, and transfers those labels back onto triangles. This gives the PAG a true 3D counterpart.

### 6. Explicit rigid-object motion parameterization

Object motion is modeled with per-frame `SE(3)` transforms, followed by refinement deltas and an optional global scale correction. This is a rigid-object assumption, but it makes the optimization well-posed and interpretable.

### 7. Contact reasoning in canonical object coordinates

The refinement stage measures contact drift in canonical object space, not only in the moving world frame. This is important because it distinguishes stable grasp/contact from accidental geometric proximity.

### 8. Human-aware but not human-optimized final refinement

This is important to state clearly:

- the default current refinement stage is interaction-aware,
- but it is not yet a full human-and-object joint motion optimizer.

The human sequence is loaded as fixed aligned geometry and used inside losses and overlays, while object motion is what gets optimized.

## Repo-Specific Contributions vs Borrowed Foundations

### Repo-specific contributions

The current repo contributes the following system-level ideas:

- a modular text-to-4D HOI pipeline built from explicit intermediate artifacts,
- a reusable PAG representation that connects language, segmentation, part semantics, and refinement,
- a first-frame object-mesh reconstruction plus temporal object tracking strategy,
- a rendered-view object-part labeling pipeline that turns PAG part names into triangle labels,
- an HOI-aware refinement objective that uses masks, part masks, SDF penetration, and PAG contact semantics,
- practical tooling for running the full stack over multiple videos and preserving stage outputs.

### Borrowed or external foundations

The repo also depends on strong external building blocks:

- HOI-PAGE / `Original_Code/` for the original affordance-guided HOI idea and shared-loss ancestry,
- GVHMR for human motion estimation,
- Depth Anything 3 for monocular depth,
- SAM3 for video and rendered-view segmentation,
- SAM-3D-style mesh generation utilities for object reconstruction,
- CoTracker or WAFT for point tracking,
- diffusers-based image and video generation backbones.

The thesis value of 4DHOI is therefore not "we invented every component." It is that the repo composes these components into an explicit 4D HOI system with part-level semantics and geometry-aware refinement.

## What the Method Currently Does Well

- It produces an interpretable chain of intermediate representations instead of a black-box final output.
- It keeps semantic interaction structure explicit through the PAG.
- It can reason about object parts instead of only whole-object proximity.
- It moves from language to video to geometry in one reproducible repo.
- It already supports multiple example interactions and saves rich diagnostics.
- It can bootstrap a thesis quickly because most stages are already modular and inspectable.

## Current Limitations

- Errors accumulate across many stages: generation, segmentation, depth, human motion, alignment, tracking, and refinement.
- The final default refinement stage keeps the human sequence fixed, so human motion is not yet jointly corrected with object motion.
- Objects are treated as rigid, with only one global scale correction per object.
- Contact is semantic and geometric, but not yet fully physics-based.
- The system depends heavily on the quality of the generated video and its segmentations.
- The pipeline is not end-to-end trainable.
- Evaluation is still tool-oriented and qualitative; it is not yet framed as a benchmark-ready research protocol.

## Thesis Interpretation

The strongest current thesis interpretation of 4DHOI is:

> An explicit, part-aware, language-conditioned 4D HOI pipeline that bridges text generation and monocular 4D reconstruction through a Part Affordance Graph and interaction-aware object refinement.

If this repo is pushed toward publication, the likely center of gravity should be:

- explicit semantic interaction structure,
- part-aware object reasoning,
- geometry-consistent HOI refinement,
- and a stronger learned or benchmarked interaction prior on top of the current modular stack.
