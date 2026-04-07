# 4DHOI Project Memory for Codex

## Project Identity

`4DHOI` is a zero-shot text-to-4D human-object interaction pipeline. It currently generates dynamic 3D human-object interactions from prompts, but the outputs are standalone or floating rather than grounded in a real scene.

## Current Main Pipeline

1. `Generate_PAG/`  
   Prompt to Part Affordance Graph.

2. `Generate_Video/`  
   PAG to first frame and full video.

3. `Segment_Video/`  
   Video to object and human masks.

4. `Estimate_Depth/`  
   Video to depth.

5. `Estimate_Human_Motion/`  
   Video to human motion and meshes.

6. `Generate_Object_Mesh/`  
   First frame and masks to object meshes.

7. `Align_Meshes/`  
   Human and object geometry into one shared frame.

8. `Segment_Object_Mesh/`  
   Object-part segmentation.

9. `Track_Object_Mesh/`  
   Per-object pose tracking.

10. `Track_Human_Object_Mesh/`  
    Final joint refinement using contact, smoothness, and SDF-based constraints.

## Current Extension Goal

Extend `4DHOI` to scene-conditioned human-object interaction synthesis in ScanNet++.

Target examples:

- sitting on a chair
- lifting a chair
- leaning on a table
- sitting on a bed edge

Preferred v1 path:

- generate a scene-conditioned interaction video from a real ScanNet++ view
- recover human motion with `GVHMR`
- recover manipulated-object motion with the current optical-flow-based tracking stack
- align human and object to scene depth in a shared frame
- add scene-aware refinement losses on top of the current optimization setup

## Current Strategic Decisions

- Use ScanNet++ as the scene dataset.
- Optimize first for full scene-aware generation, not just post-hoc placement.
- Use mesh + semantics first as the core data path.
- Prefer scene-conditioned video generation over direct retrieval or diffusion-only 3D motion generation for v1.
- Prefer reusing `GVHMR`, optical-flow object tracking, and depth-based alignment from the current pipeline.
- Prioritize furniture-affordance scenes.
- Current 5-scene starter pack:
  - `1b75758486` — Conference Room
  - `4ba22fa7e4` — Office Day
  - `8d563fc2cc` — Office Night
  - `bb87c292ad` — Kitchen
  - `e8ea9b4da8` — Bedroom
- Backups:
  - `0a5c013435`
  - `d415cc449b`

## Key Code Areas to Reuse

- `Track_Human_Object_Mesh/losses.py`
- `Track_Human_Object_Mesh/data_loading.py`
- `Track_Human_Object_Mesh/models.py`
- `Track_Human_Object_Mesh/geometry.py`
- `Generate_PAG/`
- `Original_Code/hms/scene.py`
- `Original_Code/hms/scene_init.py`

## Important Constraints

- Preserve existing working pipeline outputs.
- Prefer adding new modules over rewriting stable stages.
- Keep ScanNet++ camera/world coordinates explicit and well documented.
- v1 should stay single-human, single-target-instance, static-background.
- Existing standalone 4DHOI flow should remain usable.
- Expect constant-contact interactions to be easier than time-varying-contact interactions in v1.

## Immediate Next Implementation Tasks

1. Build a ScanNet++ scene manifest and cache layer.
2. Implement scene-instance extraction.
3. Add target-instance disambiguation among multiple similar objects in one scene.
4. Add a scene-aware PAG mode.
5. Add scene-conditioned first-frame and video generation.
6. Reuse `GVHMR` for human recovery and optical-flow tracking for object motion.
7. Add scene-aware refinement losses.

## File Relationship Note

- `CLAUDE.md` is the existing Claude-oriented project memory file.
- `CODEX.md` is the Codex counterpart.
- `Extension_Claude.md` is the earlier extension roadmap.
- `Extensions_Codex.md` is the Codex extension roadmap.
