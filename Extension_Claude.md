# Plan: Extend 4DHOI to Scene-Conditioned Human-Object Interaction Synthesis

## Context

The current 4DHOI pipeline generates **standalone/floating** 4D human-object interactions from text prompts. The thesis extension places these interactions **inside real scanned 3D scenes** from ScanNet++. Given a scene and a text prompt like "a person pulling out a chair and sitting on it", the system should generate a physically plausible SMPL human interacting with a specific object of the scene while respecting all scene geometry.

**Core insight: The pipeline stays video-based.** We render the scene, generate a video of the interaction in that scene, extract human motion (GVHMR) and object motion (optical flow + SE(3) tracking), align them to the scene using GT depth, and refine with scene-aware constraints. Most of the current pipeline is reused as-is.

---

## Part 1: ScanNet++ Scene Selection (5 Scenes)

### Why ScanNet++
- Sub-millimeter laser scans (vs commodity depth in ScanNet v1)
- 1500+ semantic classes (vs 20 in ScanNet v1)
- **No prior HSI benchmark on ScanNet++ exists** -- opportunity to establish the first
- Professor's recommendation

### Selection Protocol

Scene IDs in ScanNet++ are opaque hex hashes (e.g., `56a0ec536c`), so you cannot browse before downloading. Follow this protocol:

**Step 1: Get access** at https://kaldir.vc.in.tum.de/scannetpp/ (takes a few days)

**Step 2: Download metadata only** (~tiny):
- `metadata/scene_types.json` (maps scene_id -> room type)
- `splits/nvs_sem_train.txt` and `nvs_sem_val.txt`
- `metadata/semantic_benchmark/` (label mappings)

**Step 3: Download meshes for ~20 candidate scenes** (~2.6 GB total, ~130 MB each):
- Per scene you need: `mesh_aligned_0.05.ply`, `mesh_aligned_0.05_semantic.ply`, `segments_anno.json`
- Pick ~4 candidates per room type from the 5 target types below

**Step 4: Score and select top 5** by:
- Count of interactable objects (chairs, tables, sofas, cabinets, etc.)
- Available floor area (from floor-labeled vertices)
- Scan quality (check for holes/artifacts)
- Pick the best scene per room type

### Target 5 Room Types and Interactions

| # | Room Type | Target Interactions | Key Objects Needed |
|---|-----------|--------------------|--------------------|
| 1 | **Living room** | Sit on sofa, pick up book/remote from coffee table, lean on armrest | Sofa, coffee table, cushion, shelf |
| 2 | **Office/study** | Sit on office chair, reach for desk objects, open drawer | Office chair, desk, monitor, shelf |
| 3 | **Kitchen** | Open cabinet, lean on counter, pick up items | Kitchen counter, cabinet, refrigerator, stool |
| 4 | **Bedroom** | Sit on bed, open wardrobe/drawer, pick up pillow | Bed, nightstand, wardrobe, pillow |
| 5 | **Conference/meeting room** | Pull/push chair, sit at table, stand and gesture | Multiple chairs, large table, open floor |

### What to Download Per Scene (Full Data)
For these 5 final scenes, download everything:
```
data/<scene_id>/
  scans/
    mesh_aligned_0.05.ply              # Decimated mesh (~50-150 MB)
    mesh_aligned_0.05_semantic.ply     # Semantic vertex labels
    pc_aligned.ply                     # Full point cloud (~500-700 MB)
    segments.json                      # Over-segmentation
    segments_anno.json                 # Instance annotations
  dslr/
    resized_images/                    # ~2MP JPGs (1752x1168, fisheye)
    resized_undistorted_images/        # ~2MP JPGs (1752x1168, pinhole) ← USE THESE
    colmap/                            # Camera poses (OPENCV_FISHEYE model)
    nerfstudio/
      transforms.json                 # Fisheye model, OpenGL convention
      transforms_undistorted.json     # Pinhole model, OpenGL convention ← USE THIS
  iphone/
    rgb.mkv                           # 60 FPS video (1920x1440)
    depth.bin                         # Packed 16-bit depth (256x192, mm, LiDAR)
    pose_intrinsic_imu.json           # 4x4 cam-to-world + intrinsics
    nerfstudio/transforms.json
```
Estimated total: ~7-10 GB for 5 scenes with full data.

### ScanNet++ Technical Details
- **Coordinate system**: Z-up, metric (meters) -- must convert to OpenCV (X-right, Y-down, Z-forward) for pipeline compatibility
- **Mesh format**: PLY exclusively -- loads directly with `trimesh.load()` or Open3D
- **Scale**: metric (meters), matches SMPL scale
- **Semantic labels**: 1500+ free-text, with top-100 benchmark mapping in `metadata/semantic_benchmark/map_benchmark.csv`

---

## Part 2: DSLR vs iPhone Images

ScanNet++ captures scenes with **three devices**: Faro laser scanner (for mesh), Sony Alpha 7 IV DSLR (33MP images), and iPhone 13 Pro (RGB-D video).

### DSLR Details
- **Camera**: Sony Alpha 7 IV + **fisheye lens** (180° FOV)
- **Raw resolution**: ~33 MP (7000x5000), resized to **1752x1168**
- **Distortion**: OPENCV_FISHEYE model (k1, k2, k3, k4 radial coefficients)
- **Undistorted images available**: `dslr/resized_undistorted_images/` (pinhole model, same 1752x1168)
  - Downloads after April 30, 2025 include them automatically
  - Or generate with: `python -m dslr.undistort dslr/configs/undistort.yml`
- **Poses**: COLMAP (metric scale, aligned to laser scan)
- **~200 images per scene** (train) + 15-25 test views
- **No native depth** -- but can render depth from mesh at any DSLR camera pose

### iPhone Details
- **Camera**: iPhone 13 Pro (standard perspective, no fisheye)
- **RGB resolution**: **1920x1440** at 60 FPS (stored as `rgb.mkv`)
- **Depth**: Apple LiDAR at **256x192**, 16-bit PNG in millimeters
  - ~5-10 cm accuracy, noisy at edges, limited range (~5m)
- **Poses**: ARKit 4x4 cam-to-world matrices (aligned to laser scan)
- **~7200 frames per scene** (2 min at 60 FPS)

### Comparison for Our Use Case

| Aspect | DSLR (undistorted) | iPhone |
|--------|-------------------|--------|
| Image quality | Higher (full-frame sensor) | Lower (phone camera) |
| Projection model | Pinhole (after undistortion) | Standard perspective |
| Resolution | 1752x1168 | 1920x1440 |
| Native depth | No | Yes (LiDAR, 256x192, ~5-10cm accuracy) |
| GT depth from mesh | Yes (render at any resolution) | Yes (render at any resolution) |
| Poses aligned to scan | Yes (COLMAP) | Yes (ARKit) |
| # viewpoints | ~200 | ~7200 |
| NeRF/3DGS support | Yes (nerfstudio transforms) | Yes |
| Video gen input | Good (higher quality → better I2V) | OK |

### Recommendation

**Use undistorted DSLR images as first frame for video generation:**
- Higher image quality → better video generation from Wan2.2 I2V
- Already undistorted to pinhole model (no fisheye correction needed)
- Pinhole intrinsics are standard (fx, fy, cx, cy) -- compatible with the pipeline

**Use mesh-rendered depth for alignment (both DSLR and iPhone):**
- Render the scene mesh depth at the chosen camera pose using `renderpy` or PyTorch3D
- Output: 16-bit uint16 PNG in millimeters (same as pipeline expects)
- Much more precise than iPhone LiDAR (sub-mm mesh vs ~5-10cm LiDAR)
- Works for any camera viewpoint

**iPhone as fallback/alternative:**
- If undistorted DSLR images aren't available (older downloads), use iPhone frames
- iPhone gives more viewpoint options (7200 vs 200)
- iPhone depth can serve as rough sanity check

**Future option -- 3D Gaussian Splatting:**
- ScanNet++ provides a [3DGS demo pipeline](https://github.com/scannetpp/3DGS-demo)
- Trains on undistorted DSLR images using `transforms_undistorted.json`
- Gives photorealistic renders from ANY viewpoint (not limited to captured views)
- Best quality but requires training time per scene

---

## Part 3: Revised Pipeline Architecture

### Key Insight: Reuse Most of the Current Pipeline

The current pipeline already does: video → extract human + object → align → track → refine. For the scene extension, we:
- **Replace** first frame generation (FLUX) with a scene render/image
- **Replace** depth estimation with GT depth from mesh rendering
- **Add** object-to-scene instance matching
- **Add** scene collision constraints in refinement
- **Keep everything else** (video gen, segmentation, GVHMR, SAM3D, optical flow, alignment, tracking, refinement)

### Full Pipeline (14 stages)

```
Text + Scene
    │
    ├─[1] Load Scene ─────────────────────── ScanNet++ mesh + semantics + annotations
    │
    ├─[2] Select Viewpoint ──────────────── Pick undistorted DSLR image showing target area
    │
    ├─[3] Render GT Depth ───────────────── Render scene mesh depth at chosen camera pose
    │
    ├─[4] Generate Scene-Aware PAG ──────── LLM generates PAG with scene context
    │
    ├─[5] Generate Video ────────────────── Wan2.2 I2V: scene image + text → 81-frame video
    │    (scene image replaces FLUX first frame)
    │
    ├─[6] Segment Video ─────────────────── Qwen-VL + SAM3 (REUSE as-is)
    │
    ├─[7] Estimate Human Motion ─────────── GVHMR → SMPL-X (REUSE as-is)
    │
    ├─[8] Generate Object Mesh ──────────── SAM3D from first frame (REUSE as-is)
    │
    ├─[9] Estimate Optical Flow ─────────── CoTracker (REUSE as-is)
    │
    ├─[10] Align Meshes ─────────────────── Chamfer loss with GT depth (REUSE, better depth input)
    │
    ├─[11] Match Object to Scene Instance ─ NEW: identify which scene object was interacted with
    │
    ├─[12] Track Object Mesh ────────────── SE(3) tracking via optical flow (REUSE as-is)
    │
    ├─[13] Segment Object Parts ─────────── Multi-view part segmentation (REUSE as-is)
    │
    ├─[14] Joint Refinement ─────────────── EXTENDED: add scene collision constraints
    │
    └─[15] Export ───────────────────────── PLY sequences in scene coordinates
```

### What Changes vs Current Pipeline

| Current Stage | Scene Version | Change Type |
|---------------|---------------|-------------|
| Generate first frame (FLUX) | Use scene image (undistorted DSLR) | **REPLACE** |
| Generate video (Wan2.2) | Same, but scene image as first frame | **MINOR MODIFY** |
| Segment video | Same | REUSE |
| Estimate depth | Render from scene mesh | **REPLACE** |
| Estimate human motion (GVHMR) | Same | REUSE |
| Generate object mesh (SAM3D) | Same | REUSE |
| Estimate optical flow (CoTracker) | Same | REUSE |
| Align meshes | Same code, GT depth input | **INPUT CHANGE** |
| Segment object parts | Same | REUSE |
| Track object mesh | Same | REUSE |
| Joint refinement | Extended with scene constraints | **EXTEND** |
| -- | Load scene | **NEW** |
| -- | Select viewpoint | **NEW** |
| -- | Render GT depth | **NEW** |
| -- | Match object to scene instance | **NEW** |
| -- | Scene-aware PAG | **NEW** |

---

## Part 4: New Stages in Detail

### Stage 1: Load Scene (`Scene_Loading/scannetpp_loader.py` -- NEW)

- Load `mesh_aligned_0.05.ply` (scene mesh) and `mesh_aligned_0.05_semantic.ply` (semantic labels)
- Load `segments_anno.json` (instance-level object annotations)
- Convert coordinate system: ScanNet++ is Z-up → OpenCV is Y-down, Z-forward
  - Rotation: 180° around X-axis (similar to `Original_Code/hms/scene_init.py` line 136-138)
- Extract floor plane via RANSAC on floor-labeled vertices
- Build per-category object instance index (e.g., all chairs, all tables)

### Stage 2: Select Viewpoint (`Scene_Loading/viewpoint_selector.py` -- NEW)

Given the target object category from the text prompt, select the best captured viewpoint:
1. For each undistorted DSLR image, project the target object instances into the camera frame
2. Score by: object visibility (% of object pixels visible), distance from camera, object centering, floor space visible around object
3. Pick the highest-scoring image
4. Load its camera intrinsics (pinhole, from `transforms_undistorted.json`) and pose (cam-to-world)

**Output**: The undistorted DSLR image + camera intrinsics + camera pose

### Stage 3: Render GT Depth (`Scene_Loading/render_depth.py` -- NEW)

Render the scene mesh from the selected camera viewpoint:
- Use PyTorch3D, Open3D, or ScanNet++'s `renderpy` tool
- Output: depth map as float32 NPY [H, W] in meters (same format as `metric_depth.npy` from `Estimate_Depth/`)
- Also output camera intrinsics JSON in pipeline format: `{"intrinsics_pixels_3x3": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}`
- Resolution: match video resolution (1280x720 or the scene image resolution)

**Key advantage**: GT depth from mesh rendering is sub-millimeter precise (vs estimated depth which has ~5-10% error, or iPhone LiDAR at ~5-10cm).

### Stage 4: Scene-Aware PAG (`Generate_PAG/generate_pag_scene.py` -- NEW)

Fork `generate_pag.py` to add scene context to the LLM input:
- Object identity and category (from semantic annotations)
- Object bounding box dimensions in meters
- Object height from floor
- Nearby obstacles and their positions
- Available approach directions

The PAG output format stays identical -- same `object part nodes`, `body part nodes`, `interaction edges`, `object states`. This ensures downstream compatibility with all existing loss functions.

### Stage 11: Match Object to Scene Instance (`Scene_Loading/object_matcher.py` -- NEW)

After aligning the SAM3D-generated object mesh to the scene frame (using GT depth in Stage 10), we need to identify which specific scene object instance it corresponds to.

**Problem**: The text says "pull out a chair" but the scene has 6 chairs. The video generation model picks one (based on scene image), and we need to figure out which one.

**Approach**:
1. From `segments_anno.json`, get all instances of the target category (e.g., all chairs)
2. For each scene instance:
   - Compute **3D chamfer distance** between aligned SAM3D mesh and scene instance mesh
   - Compute **2D mask overlap** (IoU): project both into the camera frame and compare silhouettes
3. Pick the instance with lowest combined distance / highest IoU
4. **Optionally replace** the SAM3D mesh with the higher-quality scene mesh for downstream stages
   - Transfer the tracked SE(3) trajectory to the scene mesh
   - Scene mesh has better geometry (laser-scan quality vs SAM3D reconstruction)

**Fallback**: If matching is ambiguous (e.g., two chairs very close), use the 2D mask from video segmentation (frame 0) projected into the scene to disambiguate.

### Stage 14: Extended Joint Refinement

Extend `Track_Human_Object_Mesh/` with scene-aware constraints. Create new files, don't modify existing ones.

**New file: `Track_Human_Object_Mesh/losses_scene.py`**

Additional losses on top of existing contact/intersection/smoothness:

1. **Scene penetration loss** (prevent human from going through walls/floor/furniture):
   - Build SDF for scene geometry (excluding the target object) using existing `_build_sdf_grid()` from `data_loading.py:401-433`
   - Query SDF with all human vertices per frame
   - Penalize negative values (same structure as existing `_compute_intersect_diagnostic()`)
   - Use local SDF: 2-3m bounding box around interaction area at resolution 256 (~67MB)

2. **Floor contact loss** (keep feet on ground when walking/standing):
   - Get foot vertex IDs from `BODY_PART_TO_SEG_KEYS` in `data_loading.py:48-64`
   - Penalize distance from floor plane when feet should be grounded
   - Floor plane from RANSAC in Stage 1

3. **Balance loss** (optional, for standing interactions):
   - Penalize center-of-mass displacement from support polygon
   - Projected convex hull of grounded feet

**New file: `Track_Human_Object_Mesh/optimizer_scene.py`**

Modified optimizer that includes scene losses alongside existing losses.

---

## Part 5: Object Motion -- Why We Need It

Objects are NOT static during interactions. Examples:
- **Pulling a chair**: chair translates and rotates
- **Opening a drawer**: drawer slides out
- **Lifting a cup**: cup follows hand trajectory
- **Pushing a table**: table translates

The current pipeline already handles this via optical flow (CoTracker) → SE(3) tracking (`Track_Object_Mesh/`). This is **reused as-is** for the scene extension. The object mesh starts at its scene position and its motion is tracked from the generated video.

After tracking, the object's SE(3) trajectory tells us exactly how it moved during the interaction. This is combined with the GVHMR human motion for joint refinement.

---

## Part 6: Temporal Contact Dynamics (Simplified for Now)

### The Problem

Many interactions have **changing contact patterns** over time:
- **"Pull chair and sit"**: First hands contact chair back → then hands release → then hips contact seat
- **"Pick up cup from table"**: First hands approach cup → then hands grasp cup → cup lifts off table
- **"Open door and walk through"**: Hand contacts handle → door rotates → hand releases → human walks through

The current PAG specifies contact edges with `is_continuous` and `is_rel_static` flags, but these are **static for the entire video**. Contact losses are applied uniformly across all frames.

### Current Simplification

**For the initial implementation, target interactions where the PAG remains constant throughout the video:**

Good initial interactions (constant contact pattern):
- **Lifting a chair** (hands on chair throughout)
- **Carrying a box** (hands on box throughout)
- **Pushing a table** (hands on table throughout)
- **Leaning on a counter** (hands on counter throughout)
- **Moving an object across a surface** (hands on object throughout)

Avoid for now:
- Pull and sit (contact switches from hands to hips)
- Pick up and place (contact starts/ends)
- Open door and walk through (sequential contacts)

### Future Work: Temporal PAG

Extend the PAG format to include temporal annotations:

```json
{
  "interaction_edges": [
    {
      "nodes": ["chair_back", "right_hand"],
      "is_continuous": true,
      "temporal_range": [0.0, 0.6],     // ← NEW: active during first 60% of video
      "contact_type": "grasp"
    },
    {
      "nodes": ["chair_seat", "hips"],
      "is_continuous": true,
      "temporal_range": [0.7, 1.0],     // ← NEW: active during last 30%
      "contact_type": "support"
    }
  ]
}
```

This requires:
- Modified LLM system prompt to generate temporal ranges
- Modified loss computation to only apply contact losses during active temporal ranges
- A way to detect contact transitions from the video (could use segmentation masks proximity over time)

**This is a significant extension and should be a separate milestone after the core pipeline works.**

---

## Part 7: Module Reuse Map

| Existing Module | Status | Notes |
|---|---|---|
| `Generate_PAG/` | **FORK** | New scene-aware version alongside existing |
| `Generate_Video/generate_video.py` | **REUSE** | Scene image as first frame (already accepts any image) |
| `Generate_Video/generate_first_frame.py` | **SKIP** | Replaced by scene image selection |
| `Segment_Video/` | **REUSE as-is** | Works on any video, handles multiple same-category objects |
| `Estimate_Depth/` | **SKIP** | Replaced by GT depth from mesh rendering |
| `Estimate_Human_Motion/` | **REUSE as-is** | GVHMR works on any video |
| `Generate_Object_Mesh/` | **REUSE as-is** | SAM3D from first frame |
| `Estimate_Optical_Flow/` | **REUSE as-is** | CoTracker works on any video |
| `Align_Meshes/` | **REUSE** | Same code, just feed GT depth instead of estimated depth |
| `Segment_Object_Mesh/` | **REUSE as-is** | Part segmentation from rendered views |
| `Track_Object_Mesh/` | **REUSE as-is** | SE(3) tracking via optical flow |
| `Track_Human_Object_Mesh/losses.py` | **HEAVY REUSE** | Contact, intersection, smoothness -- core of refinement |
| `Track_Human_Object_Mesh/data_loading.py` | **REUSE** | `_build_sdf_grid()`, body part segmentation mapping |
| `Track_Human_Object_Mesh/models.py` | **EXTEND** (new file) | Add `SceneData`, new loss fields |
| `Track_Human_Object_Mesh/geometry.py` | **DIRECT REUSE** | SDF query, similarity transforms |
| `Original_Code/hms/scene.py` | **REFERENCE** | `MeshScenes` pattern for scene loading |
| `Blender_Scripts/` | **REUSE** | Visualization of results in scene |

**New modules to create:**
- `Scene_Loading/scannetpp_loader.py` -- scene mesh + annotations loading
- `Scene_Loading/viewpoint_selector.py` -- pick best camera viewpoint
- `Scene_Loading/render_depth.py` -- render GT depth from mesh
- `Scene_Loading/object_matcher.py` -- match extracted object to scene instance
- `Scene_Loading/scene_context.py` -- build scene context for PAG
- `Generate_PAG/generate_pag_scene.py` -- scene-aware PAG generation
- `Generate_PAG/system_prompt_pag_scene.md` -- scene-aware PAG prompt
- `Track_Human_Object_Mesh/losses_scene.py` -- scene penetration + floor contact losses
- `Track_Human_Object_Mesh/optimizer_scene.py` -- refinement with scene losses
- `Evaluation/` -- metrics and benchmark scripts

---

## Part 8: Timeline (3 months remaining, ~April-July 2026)

| Week | Focus | Deliverable |
|---|---|---|
| 1-2 | Scene Loading + Rendering | ScanNet++ loader, viewpoint selector, GT depth rendering, scene SDF |
| 3 | Scene-aware PAG | Extended system prompt, scene context builder, PAG generation working |
| 4 | Video generation + existing pipeline | End-to-end: scene render → video → segmentation → GVHMR → SAM3D → optical flow |
| 5 | Alignment with GT depth | Align meshes using rendered GT depth, verify improvement over estimated depth |
| 6 | Object-to-scene matching | Match extracted objects to scene instances, validate on multiple scenes |
| 7 | Scene-aware refinement | Scene penetration + floor contact losses, joint refinement with scene constraints |
| 8 | End-to-end integration | Full pipeline working: text + scene → 4D HOI in scene → PLY export |
| 9 | Quality tuning | Loss weight tuning, edge cases, diverse interaction types |
| 10-11 | Evaluation | Metrics, baselines (standalone 4DHOI, HUMANISE, TRUMANS), benchmark |
| 11-13 | Paper writing | Draft, figures, supplementary video, submission prep |

---

## Part 9: Key Technical Challenges

1. **Video generation quality from scene renders**: Wan2.2 I2V was not trained specifically on indoor scene images. The generated video may not preserve scene geometry perfectly. Mitigate by: using high-quality scene images (undistorted DSLR), keeping static camera prompts, potentially using 3DGS renders for more photorealistic input.

2. **Object-to-scene matching ambiguity**: When multiple instances of the same category exist (6 chairs in a conference room), matching can be hard. Mitigate by: combining 3D chamfer distance + 2D mask overlap, using the first-frame segmentation mask projected into the scene.

3. **Scene SDF memory**: Large scenes need local SDF around the interaction area. Use a 2-3m bounding box around the target object at resolution 256 (~67MB). Reuse `_build_sdf_grid()` on cropped scene geometry.

4. **Coordinate system mismatch**: ScanNet++ is Z-up, pipeline is OpenCV (Y-down, Z-forward). Single rotation transform at scene load time. Camera poses from COLMAP/ARKit need conversion to pipeline convention.

5. **Object motion tracking accuracy**: Optical flow tracking may be noisy for objects with few texture features (e.g., plain white chair). The existing Huber-robustified SE(3) tracking handles this, but scene objects with uniform appearance may need denser tracking points.

6. **Temporal contact dynamics**: Deferred to future work. For now, only handle interactions with constant contact patterns.

---

## Part 10: Verification Plan

1. **Scene loading**: Load ScanNet++ scene, convert coordinates, visualize in Blender -- verify orientation and scale match SMPL
2. **GT depth**: Render depth from scene mesh, compare with iPhone LiDAR depth -- verify alignment
3. **Video generation**: Feed undistorted DSLR image + interaction text to Wan2.2 → verify video shows interaction in the scene
4. **Existing pipeline stages**: Run segmentation, GVHMR, SAM3D, optical flow on the generated video -- verify they work on scene-conditioned videos
5. **Alignment**: Align meshes with GT depth, compare with alignment using estimated depth -- verify improvement
6. **Object matching**: Extract object, align, match to scene instance -- verify correct instance identified
7. **Scene refinement**: Run joint refinement with scene losses -- verify no penetration with walls/floor
8. **End-to-end**: Full pipeline → render result in Blender with scene mesh as background
9. **Quantitative**: Metrics on 5 scenes x 10 constant-contact prompts each

---

## Part 11: Publication Angle

**Novel contributions for a top-venue paper:**
1. First framework for **text-driven 4D HOI synthesis in real scanned scenes** (zero-shot, no paired training data)
2. **Scene-aware PAG**: LLM-driven interaction structure that respects spatial constraints
3. **Video-based scene-conditioned generation**: leverages I2V models for physically grounded interaction synthesis
4. **Automatic object-to-scene instance matching**: resolves which scene object participates in the interaction
5. **Scene-constrained optimization**: extends contact/intersection losses with floor contact + scene penetration
6. **ScanNet++ HOI benchmark**: first standardized evaluation for human-object interaction on ScanNet++

**Positioning**: Combines video-based HOI generation (4DHOI) with scene-aware constraints, bridging the gap between text-to-motion (MDM, MotionDiffuse), scene-aware HSI (HUMANISE, TRUMANS), and structured interaction reasoning (PAG framework).

---

## Part 12: Related Work

### Scene-Aware Human Motion Generation
- **HUMANISE** (NeurIPS 2022): Text + scene → SMPL motion. Uses ScanNet v1. Open source.
- **TRUMANS** (CVPR 2024): Scene-aware motion generation, demonstrated zero-shot on ScanNet++.
- **LINGO** (2024): Language-grounded interaction generation in VR-captured scenes.
- **InteractMove** (ACM MM 2025): Aligns HOI sequences with ScanNet scenes.
- **CHOIS** (ECCV 2024 Oral): Contact-guided 3D HOI generation.
- **SceneDiffuser** (CVPR 2023): Diffusion-based scene-conditioned generation.

### Motion Datasets
- **AMASS**: Unified motion capture database (for potential retrieval baselines).
- **BABEL**: Text annotations for AMASS motions.

### Text-to-Motion
- **MDM** (ICLR 2023): Motion Diffusion Model. Text-conditioned but not scene-aware.
- **MotionDiffuse** (2023): Text-driven motion generation with fine-grained control.
- **TMR** (2023): Text-motion retrieval model.
