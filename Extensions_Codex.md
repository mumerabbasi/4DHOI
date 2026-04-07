# Plan: Extend 4DHOI to Scene-Conditioned Human-Object Interaction Synthesis

## Context

`4DHOI` currently generates zero-shot 4D human-object interactions from text prompts, but the outputs are standalone or floating interactions rather than interactions grounded in a real 3D scene. The thesis extension is to place the generated human and interaction inside scanned indoor environments from ScanNet++, while reusing as much of the current video-based 4DHOI pipeline as possible.

The recommended direction is no longer "skip the video pipeline and generate motion directly in 3D." Instead, the primary v1 should be:

- choose a real ScanNet++ scene and camera view
- generate a scene-conditioned interaction video from that view
- recover human motion with `GVHMR`
- recover manipulated-object motion with the current optical-flow-based tracking stack
- align both human and object to scene depth in a shared frame
- refine with scene-aware losses

This keeps the extension much closer to the existing codebase and preserves the current strengths of `4DHOI`.

## Recommended 5 ScanNet++ Scenes

These are the current recommended scenes for a furniture-affordance starter pack:

1. `1b75758486` - Conference Room  
   Best initial scene for chair and table interactions such as sitting, pulling a chair, lifting a chair, and leaning on a table.

2. `4ba22fa7e4` - Office Day  
   Strong office scene for desk and office-chair interactions, with cleaner lighting than harder night scenes.

3. `8d563fc2cc` - Office Night  
   Useful second office geometry with harder appearance conditions and similar furniture affordances.

4. `bb87c292ad` - Kitchen  
   Adds counter, table, and support-surface interactions, with potential for standing support and object handling near furniture.

5. `e8ea9b4da8` - Bedroom  
   Adds bed and side-table interactions, broadening support-contact behavior beyond chairs and desks.

Backup scenes:

- `0a5c013435` - Utility Room
- `d415cc449b` - Tool Room

Note: these scene IDs were inferred from public ScanNet++ references and room-name mappings. Once local metadata is available, verify actual object richness and replace any weak scene if needed.

## Download Policy

For the first implementation pass, download the official default assets for the 5 selected scenes:

- meshes
- semantics
- iPhone data
- low-resolution DSLR data

For practical use in this project, prioritize these folders and files:

- `scans/mesh_aligned_0.05.ply`
- `scans/mesh_aligned_0.05_semantic.ply`
- `scans/segments.json`
- `scans/segments_anno.json`
- `dslr/resized_undistorted_images` or `dslr/resized_images`
- `dslr/colmap`
- `iphone/rgb.mp4` or decoded RGB frames
- `iphone/depth` or decoded depth frames
- camera transforms and intrinsics for the chosen image stream

Defer these unless they become necessary:

- point clouds
- hi-res DSLR
- panocam

## Camera and Depth Strategy

One important correction to the earlier plan:

- ScanNet++ does provide iPhone depth directly.
- DSLR images are fisheye in their raw camera model.
- ScanNet++ also supports undistorted DSLR images for pinhole-camera workflows.
- Depth for DSLR and iPhone views can be rendered from the aligned mesh using the official toolkit.

### Recommended choice for v1

Use **undistorted DSLR images as the main appearance-conditioning path** and **rendered scene depth as the alignment depth source**.

Why this is a good default:

- DSLR images have better appearance quality for first-frame and video generation.
- Undistorted DSLR images avoid the fisheye problem.
- The rendered DSLR depth is scene-consistent and already aligned to the known mesh and camera pose.
- This fits well with the current alignment logic, which already expects image + depth + mesh alignment.

### iPhone as fallback or secondary path

Use iPhone RGB + depth if the DSLR path becomes too cumbersome at the start.

Pros:

- perspective-style imagery
- native depth available
- direct RGB-depth coupling

Cons:

- lower visual quality than DSLR
- lower depth resolution than rendered mesh depth

Practical recommendation:

- v1 default: undistorted DSLR image + rendered mesh depth
- fallback: iPhone RGB + native depth if it makes early prototyping faster

## Architecture Direction

The primary strategy is now:

- full scene-aware generation from the start
- reuse the current video-based 4DHOI pipeline wherever possible
- use mesh + semantics as the core scene representation
- use optical-flow-based object tracking for manipulated objects
- use `GVHMR` for human motion recovery
- keep v1 to a single human, a single target scene instance, and a static scene background

High-level goal:

- choose a real ScanNet++ scene and a real camera view
- resolve the prompt to a real scene object instance
- generate a video of the interaction in that scene
- recover human and object motion from the video
- align both into the scene frame
- refine using object contact and scene-aware losses

## Main Pipeline Changes

### 1. Add a scene ingestion and cache layer

Create a new scene-loading subsystem that reads ScanNet++ scene geometry, semantics, instance annotations, and camera metadata, then builds a cached scene manifest with:

- scene ID
- camera list
- instance list
- semantic labels
- per-instance bounding boxes
- per-instance extracted submeshes
- support-surface candidates
- per-camera visibility information for candidate objects

This layer should answer:

- which views show a given chair or table clearly
- which objects are good candidates for a given prompt
- which scene assets are needed for later alignment and tracking

### 2. Add target-object extraction from ScanNet++

For scene-conditioned HOI, the manipulated or contacted object should come from the scan, not from `Generate_Object_Mesh/`.

For each selected target instance:

- extract the object submesh from `segments_anno.json` and `segments.json`
- optionally run the existing part-segmentation pipeline on the extracted mesh
- build a canonical object template in scene coordinates
- render projected masks of the object into candidate camera views

This canonical object mesh becomes the tracked object model for dynamic-object interactions.

### 3. Add scene-conditioned first-frame and video generation

Replace the current generic indoor first-frame setup with a scene-conditioned version:

- choose a real ScanNet++ view
- use the scene image as the background condition
- mention the target object and its location or role in the prompt
- generate an interaction video that preserves the chosen scene geometry and camera framing

The first-frame generation prompt should explicitly preserve:

- static background
- camera viewpoint
- target object identity
- enough free space for the human body to stay fully visible

### 4. Recover human motion with the existing human-motion pipeline

After scene-conditioned video generation:

- segment the human in the generated video
- run `GVHMR` to estimate human motion
- export the human mesh sequence as usual

This keeps the human side of the new pipeline close to the current repo structure.

### 5. Recover object motion with the existing optical-flow and tracking pipeline

Dynamic object interactions still need explicit object-motion recovery. For cases like pulling out a chair and then sitting on it, the object does not stay static with respect to the scene.

Recommended object-motion path:

- segment the target object in the generated video
- use optical-flow cues as in the current approach
- run `Track_Object_Mesh/`-style tracking on the extracted ScanNet++ object mesh

Important distinction:

- **static support-object interactions**: object can stay fixed in the scene
- **dynamic manipulated-object interactions**: object motion must be tracked explicitly

For manipulated objects, the current optical-flow-based approach remains highly relevant and should be reused rather than replaced.

### 6. Align human and object into a shared frame using depth

The shared-frame alignment should remain conceptually similar to the current `Align_Meshes/` stage:

- recover first-frame human mesh
- use the extracted ScanNet++ object mesh as the object template
- align both to the chosen scene depth map in camera coordinates

Depth source options:

- rendered mesh depth for DSLR view
- native iPhone depth for iPhone view

This gives:

- a known shared camera frame
- a direct way to initialize human and object inside the scene
- compatibility with the current depth-based chamfer alignment code

### 7. Extend final refinement with scene-aware losses

Build on the existing optimization framework and add losses for:

- human-scene penetration
- manipulated-object vs static-scene penetration
- foot-floor stability
- seat or support contact consistency
- contact consistency with the selected scene object

Keep existing losses where they still apply:

- object tracking anchor
- object smoothness
- human-object contact
- contact drift
- human-object intersection

## DSLR vs iPhone Tradeoff Summary

### DSLR path

Pros:

- higher visual quality
- better appearance conditioning for video generation
- undistorted pinhole images are available
- rendered depth from the aligned mesh can be very clean

Cons:

- raw DSLR is fisheye
- depth is rendered rather than directly captured
- pipeline setup is slightly more involved

### iPhone path

Pros:

- no fisheye headache
- native depth available
- simpler RGB-depth pairing

Cons:

- lower image quality
- lower depth resolution
- weaker appearance anchor for video generation

Decision for now:

- start with undistorted DSLR as the preferred path
- keep iPhone as the simpler backup path for fast validation

## How to Identify the Target Object Among Many Similar Objects

This is a core problem and needs to be explicit in the roadmap.

For prompts like "a person pulls out a chair and sits," there may be many chairs in the scene. The system needs to decide which chair in the generated video corresponds to which real chair instance in the ScanNet++ scene.

Recommended first-frame matching strategy:

1. From the prompt, collect candidate scene instances of the desired category.
2. Project each candidate instance into the chosen camera view.
3. Generate the first frame and segment the interacted object.
4. Compare the generated object mask against projected masks of all candidate instances.
5. Select the instance with the best score using:
   - mask IoU
   - centroid distance
   - visible area
   - depth ordering consistency
   - prompt compatibility with nearby furniture context

Useful tie-breakers:

- nearest chair to the table if the prompt says "sit at the table"
- chair with enough free space for the human body
- object whose expected motion direction is physically plausible

This selected instance ID must be stored and propagated through the whole pipeline.

## Contact Schedule and PAG Simplification

Another important correction: contact edges will not always stay active for the whole video.

Example:

- move toward chair
- hand contacts chair
- chair is pulled
- hand contact changes
- hips contact chair during sitting

So a single static PAG is not fully sufficient for all interactions.

### v1 simplification

For the first implementation, prioritize interactions with approximately constant contact topology:

- a person lifting a chair
- a person carrying a chair
- a person leaning both hands on a table
- a person holding and moving a simple object

In these cases, the active contact set changes little over time, so a fixed PAG is still reasonable.

### Future extension

Explicitly plan for a **phase-wise or time-varying PAG** later.

This should allow:

- different active edges in different temporal phases
- contact activation and deactivation
- transitions such as hand-contact to hip-contact

Future ways to estimate this schedule:

- video-language parsing
- temporal mask-distance heuristics
- object-velocity changes
- human-object proximity events
- manually defined contact phases for benchmark prompts

This is not required for v1, but it must be tracked as a known limitation and a future work item.

## Roadmap

### Phase 1: Scene Prep and Sensor Decision

- download the 5 recommended scenes
- verify available assets for DSLR and iPhone paths
- build a reusable `SceneManifest` cache
- choose the default camera path for implementation, with DSLR first and iPhone fallback

### Phase 2: Object and View Grounding

- identify candidate objects per prompt
- compute per-view visibility and projected masks
- select a stable scene view for each target prompt
- extract the chosen object instance mesh from the scan

### Phase 3: Scene-Conditioned Generation

- implement scene-conditioned first-frame generation
- implement scene-conditioned video generation
- ensure the generated interaction stays spatially consistent with the chosen scene view

### Phase 4: Video Understanding and Motion Recovery

- segment the human and target object in the generated video
- recover human motion with `GVHMR`
- recover object motion with the current optical-flow-based object tracking approach

### Phase 5: Shared-Frame Alignment and Refinement

- align first-frame human and object meshes to scene depth
- initialize both in a shared scene frame
- add scene-aware refinement losses
- validate penetration, support, and contact quality

### Phase 6: Harder Interactions and Dynamic Contacts

- add multi-phase contact schedules
- support interactions like "pull chair and sit"
- support object switching or contact-role changes over time

### Phase 7: Thesis Evaluation

- compare standalone 4DHOI versus scene-aware 4DHOI
- run the selected prompt set across the 5 scenes
- document success and failure cases with visualizations and metrics

## Acceptance Criteria

The scene-aware extension should be considered working when:

- the human is not visibly floating
- there is no major human-scene or object-scene penetration
- the target object is a real annotated ScanNet++ instance
- the correct instance is selected among similar nearby objects
- dynamic-object interactions recover plausible object motion
- support and contact behavior match the intended prompt

For v1 specifically:

- at least one constant-contact dynamic-object interaction should work, such as lifting a chair
- at least one static-support interaction should work, such as leaning on a table
- time-varying contact interactions like "pull chair then sit" may remain a documented future extension

## References

- ScanNet++ documentation: <https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation>
- ScanNet++ changelog: <https://scannetpp.mlsg.cit.tum.de/scannetpp/changelog>
- ScanNet++ GitHub toolkit documentation: <https://github.com/scannetpp/scannetpp>
- Public room-type mapping used to infer scene IDs: <https://www.mdpi.com/2313-433X/10/12/330/xml>
- Public conference-room cue for `1b75758486`: <https://nianticlabs.github.io/morpheus/resources/Morpheus.pdf>
