# 4DHOI

Modular 4D human-object interaction pipeline built around:
- PAG generation from text prompts,
- video generation,
- video and mesh segmentation,
- object mesh generation and alignment,
- human motion estimation with GVHMR,
- optional object tracking,
- optional joint human-object refinement.

This README reflects the current code in this repo, not the original paper
layout in `Original_Code/`.

## Repo Layout

- `Generate_PAG/`: generate a Part Affordance Graph (PAG) JSON from prompts.
- `Generate_Video/`: generate a first frame and then a full video from the PAG.
- `Segment_Video/`: segment humans, objects, and object parts in the video.
- `Generate_Object_Mesh/`: reconstruct first-frame object meshes from masks.
- `Estimate_Depth/`: estimate depth for the generated video.
- `Estimate_Human_Motion/`: run GVHMR and export human meshes.
- `Align_Meshes/`: align first-frame meshes to scene depth and move the full
  human sequence into the aligned frame.
- `Segment_Object_Mesh/`: render aligned object meshes from multiple views and
  segment them into semantic parts.
- `Estimate_Optical_Flow/`: estimate per-object tracks with CoTracker or WAFT.
- `Track_Object_Mesh/`: optimize per-frame object `SE(3)` trajectories.
- `Track_Human_Object_Mesh/`: jointly refine human and object motion with PAG
  contact constraints, mask losses, and penetration losses.
- `Original_Code/`: reference implementation from the original framework.
- `Conda_Environments/`: environment YAMLs used by the different stages.

## Current Code Flow

```mermaid
flowchart TD
    A[Generate_PAG/generate_pag.py] --> B[Generate_Video/generate_first_frame.py]
    A --> C[Generate_Video/generate_video.py]
    B --> C

    C --> D[Segment_Video/segment_video.py]
    D --> E[Generate_Object_Mesh/generate_objects_meshes.py]

    C --> F[Estimate_Depth/estimate_depth.py]
    C --> G[Estimate_Human_Motion/estimate_human_motion.py]
    G --> H[Estimate_Human_Motion/export_human_motion_to_ply.py]

    E --> I[Align_Meshes/align_meshes.py]
    F --> I
    H --> I
    I --> J[Align_Meshes/align_human_motion_sequence.py]

    A --> K[Segment_Object_Mesh/render_mesh_views.py]
    I --> K
    K --> L[Segment_Object_Mesh/segment_parts.py]
    L --> M[Segment_Object_Mesh/segment_meshes.py]

    C --> N[Estimate_Optical_Flow/estimate_optical_flow_cotracker.py]
    D --> N
    I --> O[Track_Object_Mesh/track_object_mesh.py]
    N --> O
    D --> O

    J --> P[Track_Human_Object_Mesh/track_human_object_mesh.py]
    O --> P
    M --> P
    D --> P
    A --> P
    I --> P
```

## Main Stages

| Stage | Main script(s) | Main inputs | Main outputs |
| --- | --- | --- | --- |
| PAG generation | `Generate_PAG/generate_pag.py` | `Generate_PAG/input_prompts/<video>/input_pag.json` | `Generate_PAG/output/<video>/output_pag_*.json` |
| First frame generation | `Generate_Video/generate_first_frame.py` | PAG JSON | `Generate_Video/output/<video>/first_frames/*.png` |
| Video generation | `Generate_Video/generate_video.py` | first frame, PAG JSON | `Generate_Video/output/<video>/*.mp4` |
| Video segmentation | `Segment_Video/segment_video.py` | generated video, PAG JSON | masks for humans, objects, and object parts under `Segment_Video/output/<video>/` |
| Object mesh generation | `Generate_Object_Mesh/generate_objects_meshes.py` | first-frame object masks, first-frame image | per-object meshes, poses, overlays, intrinsics under `Generate_Object_Mesh/output/<video>/` |
| Depth estimation | `Estimate_Depth/estimate_depth.py` | generated video | frame extraction, metric depth, optional relative depth, run summary under `Estimate_Depth/output/<video>/` |
| Human motion estimation | `Estimate_Human_Motion/estimate_human_motion.py` | generated video | GVHMR outputs under `Estimate_Human_Motion/output/<video>/` |
| Human mesh export | `Estimate_Human_Motion/export_human_motion_to_ply.py` | `hmr4d_results.pt` | `output_plys/frame_*.ply` |
| GVHMR outputs, PAG, SMPL segmentation | colored per-frame human meshes and part mappings |
| Mesh alignment | `Align_Meshes/align_meshes.py` | object meshes, first-frame human mesh, depth, masks | aligned meshes, `transforms.json`, overlays, summaries under `Align_Meshes/output/<video>/` |
| Full human sequence alignment | `Align_Meshes/align_human_motion_sequence.py` | `output_plys`, human entry in `transforms.json` | `Align_Meshes/output/<video>/human_motion_aligned/` |
| Object mesh part segmentation | `Segment_Object_Mesh/render_mesh_views.py`, `segment_parts.py`, `segment_meshes.py` | aligned object meshes, PAG, video part names | rendered multi-view RGB/face IDs, part masks, triangle labels, segmented meshes |
| Object track estimation | `Estimate_Optical_Flow/estimate_optical_flow_cotracker.py` or `estimate_optical_flow_waft.py` | video, frame-0 object masks | per-object tracks under `Estimate_Optical_Flow/output_*` |
| Object pose tracking | `Track_Object_Mesh/track_object_mesh.py` | aligned meshes, tracks, video masks, intrinsics | per-object `poses.json`, mesh sequence, overlays, debug metrics |
| Joint human-object refinement | `Track_Human_Object_Mesh/track_human_object_mesh.py` | aligned human sequence, tracked object poses, PAG, object-part labels, video masks | refined human meshes, refined object transforms, debug loss CSV/plots, overlay video |

## Recommended Run Orders

### 1. Minimal aligned-scene pipeline

Use this when the goal is to get aligned human and object geometry into a
shared coordinate frame.

1. `Generate_PAG/generate_pag.py`
2. `Generate_Video/generate_first_frame.py`
3. `Generate_Video/generate_video.py`
4. `Segment_Video/segment_video.py`
5. `Generate_Object_Mesh/generate_objects_meshes.py`
6. `Estimate_Depth/estimate_depth.py`
7. `Estimate_Human_Motion/estimate_human_motion.py`
8. `Estimate_Human_Motion/export_human_motion_to_ply.py`
9. `Align_Meshes/align_meshes.py`
10. `Align_Meshes/align_human_motion_sequence.py`

### 2. Full refinement pipeline

Use this when the goal is final 4D human-object motion with object tracking and
joint refinement.

1. Run the minimal aligned-scene pipeline above.
2. `Segment_Object_Mesh/render_mesh_views.py`
3. `Segment_Object_Mesh/segment_parts.py`
4. `Segment_Object_Mesh/segment_meshes.py`
5. `Estimate_Optical_Flow/estimate_optical_flow_cotracker.py`
6. `Track_Object_Mesh/track_object_mesh.py`
7. `Track_Human_Object_Mesh/track_human_object_mesh.py`

### 3. Optional branches

- Replace CoTracker with WAFT by using
  `Estimate_Optical_Flow/estimate_optical_flow_waft.py`.

## Important Data Conventions

### Human geometry

- GVHMR predicts SMPL-X parameters.
- `export_human_motion_to_ply.py` converts those to SMPL-space meshes with
  `6890` vertices using a fixed `smplx2smpl` matrix.
- The topology is fixed across frames, so the same SMPL vertex indices can be
  reused for part segmentation across the whole sequence.
- `Track_Human_Object_Mesh/track_human_object_mesh.py` uses:
  - aligned human meshes from
    `Align_Meshes/output/<video>/human_motion_aligned/`,
  - SMPL vertex groups from `smpl_vert_segmentation.json`.

### Why `human_motion_aligned` exists

- GVHMR outputs the human sequence in its own camera-frame geometry.
- `Align_Meshes/align_meshes.py` estimates `source_to_output_matrix_4x4`
  transforms that move assets into the aligned scene frame.
- `align_human_motion_sequence.py` applies the human transform to every human
  frame so that the whole human motion sequence lives in the same frame as the
  aligned object meshes.

### Object mesh part segmentation

`Segment_Object_Mesh/` is a three-step pipeline:

1. `render_mesh_views.py`
   - Blender renders each aligned object from multiple views.
   - Saves RGB renders and face-ID EXRs.
2. `segment_parts.py`
   - Uses Qwen-VL and SAM3 to segment part masks in those rendered views.
3. `segment_meshes.py`
   - Maps 2D part masks back to mesh triangles via face IDs and exports part
     labels and part meshes.

This stage is required for `Track_Human_Object_Mesh/track_human_object_mesh.py`
because the joint optimizer uses per-object part labels from
`*_triangle_labels.json`.

## Current Joint Refinement Stage

`Track_Human_Object_Mesh/track_human_object_mesh.py` currently performs joint
human-object refinement with:

- per-frame global human `SE(3)` corrections,
- per-frame object `SE(3)` deltas on top of tracked poses,
- one global uniform scale per object,
- PAG contact and contact-dynamics losses,
- SDF-based penetration losses,
- temporal smoothness,
- 2D bidirectional chamfer losses against human/object/part masks from
  `Segment_Video`.

It uses:

- aligned human meshes from `Align_Meshes/output/<video>/human_motion_aligned/`,
- object trajectories from `Track_Object_Mesh/output_cotracker/<video>/`,
- object part labels from `Segment_Object_Mesh/output/<video>/`,
- PAG JSON from `Generate_PAG/output/<video>/`,
- video masks from `Segment_Video/output/<video>/`.

## Environment Notes

The repo is split across multiple Conda environments:

- `Conda_Environments/4dhoi.yml`
  - general pipeline utilities,
  - video generation,
  - depth estimation,
  - alignment,
  - tracking/refinement utilities that do not depend on external repos.
- `Conda_Environments/gvhmr.yml`
  - GVHMR human motion estimation,
  - SMPL-X / SMPL export scripts.
- `Conda_Environments/sam3.yml`
  - SAM3-based video and rendered-view segmentation.
- `Conda_Environments/sam3d-objects.yml`
  - object mesh generation and related mesh tools.
- `Conda_Environments/waft.yml`
  - WAFT optical-flow tracking branch.

Exact package compatibility still depends on the sibling repos checked out next
to `4DHOI`, especially:

- `GVHMR/`
- `sam3/`
- `sam-3d-objects/`
- `Depth-Anything-3/`
- `WAFT/`

## Notes on `Original_Code/`

`Original_Code/` contains the original `HumanMotionSynthesis` framework. It is
not the same pipeline as the modular scripts in this repo.

- The current `4DHOI/` scripts are a custom staged pipeline.
- `Original_Code/` is best treated as a reference implementation for the
  original framework and loss design.
- The current joint refinement stage borrows PAG logic and some optimization
  ideas from there, but it is not a drop-in wrapper around `Original_Code/`.
