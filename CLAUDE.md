# 4DHOI Project Instructions

## Project Overview

4DHOI is a zero-shot pipeline for 4D human-object interaction synthesis from text prompts. It generates dynamic 3D interactions (SMPL human + object meshes) via a multi-stage pipeline: text -> PAG -> video -> 3D extraction -> optimization.

### Current Pipeline (10 stages)
1. `Generate_PAG/` - Text -> Part Affordance Graph (LLM-based, deepseek-r1 or Claude)
2. `Generate_Video/` - PAG -> First frame (FLUX.1-dev) -> Video (Wan2.2 I2V, 81 frames)
3. `Segment_Video/` - Video -> masks (Qwen-VL + SAM3)
4. `Estimate_Depth/` - Video -> depth maps (Depth-Anything-3)
5. `Estimate_Human_Motion/` - Video -> SMPL-X params (GVHMR)
6. `Generate_Object_Mesh/` - First frame + masks -> 3D meshes (SAM3D)
7. `Align_Meshes/` - Align human + object meshes (Chamfer loss optimization)
8. `Segment_Object_Mesh/` - Object part segmentation from rendered views
9. `Track_Object_Mesh/` - SE(3) object pose tracking via optical flow
10. `Track_Human_Object_Mesh/` - Joint refinement (SDF intersection, PAG contact, smoothness)

### Key Technical Details
- **Human representation**: SMPL-X converted to SMPL (6890 vertices), PLY per frame
- **Coordinate system**: OpenCV camera coordinates (X-right, Y-down, Z-forward)
- **Object tracking**: SE(3) + global scale per frame
- **Contact system**: PAG-driven, SDF-based intersection prevention
- **Output format**: Per-frame PLY meshes + 4x4 transform JSON + overlay MP4

### Extension: Scene-Conditioned HOI (in progress)
Extending to generate interactions inside ScanNet++ scanned scenes. See `Extension_Claude.md` for full plan.
- **Key approach: Video-based (reuses most of existing pipeline)**
  - Replace FLUX first frame with scene image (undistorted DSLR from ScanNet++)
  - Replace estimated depth with GT depth (rendered from scene mesh)
  - Add object-to-scene instance matching (new stage)
  - Extend joint refinement with scene collision constraints (new losses)
  - Keep: video generation, segmentation, GVHMR, SAM3D, optical flow, alignment, tracking
- New modules: `Scene_Loading/`, `Evaluation/`, new files in `Track_Human_Object_Mesh/`
- Dataset: ScanNet++ (Z-up, metric, PLY format, undistorted DSLR images are pinhole)
- **Temporal contact simplification**: Initially target interactions with constant contact pattern (e.g., lifting, pushing). Temporal PAG for changing contacts is future work.
- **DSLR images are fisheye** (Sony Alpha 7 IV + fisheye lens, 180° FOV). Use `resized_undistorted_images/` (pinhole model) or iPhone frames (standard perspective).
- Objects are NOT static -- their motion is tracked via optical flow (same as current pipeline)

## Critical Rules

### File Safety
- **NEVER modify, delete, or overwrite files outside of new directories you create** (e.g., `Scene_Loading/`, `Generate_Motion/`, `Evaluation/`)
- Existing pipeline stage outputs are carefully generated and must be preserved
- Read-only access to existing directories for reference is OK
- When extending `Track_Human_Object_Mesh/`, create NEW files (e.g., `optimizer_scene.py`) rather than modifying existing ones

### Code Style
- Single conda environment: `Conda_Environments/4dhoi.yml`
- PyTorch + PyTorch3D for 3D operations
- Open3D and trimesh for mesh I/O
- Roma for rotation utilities
- All meshes in PLY format
- Camera intrinsics stored as 3x3 matrices in JSON

### Key Dependencies (from existing codebase)
- torch==2.10.0, torchvision==0.25.0
- pytorch3d, trimesh, open3d==0.19.0, roma
- diffusers==0.35.0, transformers==4.48.0
- numpy==2.2.6, scipy==1.15.3

### Important Files for Scene Extension
- `Track_Human_Object_Mesh/losses.py` - Core loss functions (contact, intersection, smoothness) to reuse
- `Track_Human_Object_Mesh/data_loading.py` - `_build_sdf_grid()` at line ~401, body part segmentation mapping
- `Track_Human_Object_Mesh/models.py` - Data models (SDFGrid, HumanData, ObjectData, LossResult)
- `Track_Human_Object_Mesh/geometry.py` - SDF query, similarity transforms
- `Generate_PAG/system_prompt_pag.md` - PAG system prompt to fork for scene-aware version
- `Original_Code/hms/scene.py` - MeshScenes class (reference pattern for scene loading)
- `Original_Code/hms/scene_init.py` - Canonicalization, coordinate transforms (line 136-138)
