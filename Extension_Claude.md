# Plan: Extend 4DHOI to Scene-Conditioned Human-Object Interaction Synthesis

## Context

The current 4DHOI pipeline generates **standalone/floating** 4D human-object interactions from text prompts. The thesis extension places these interactions **inside real scanned 3D scenes** from ScanNet++. Given a scene and a text prompt like "a person sitting on the chair", the system should generate a physically plausible SMPL human interacting with a specific scanned object while respecting all scene geometry (no penetration with walls, floor, furniture).

This is a significant extension that skips most of the video-based pipeline (video generation, segmentation, depth estimation, optical flow, object mesh reconstruction) since the scene already provides ground-truth 3D. The core reuse is the **loss/optimization infrastructure** from `Track_Human_Object_Mesh/`.

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
    resized_images/                    # 2MP DSLR images (for rendering/viz)
    colmap/                            # Camera poses
    nerfstudio/                        # NeRF transforms
  iphone/
    rgb.mkv, depth.bin                 # RGB-D (useful for baseline comparisons)
```
Estimated total: ~7-10 GB for 5 scenes with full data.

### Important ScanNet++ Technical Details
- **Coordinate system**: Z-up, metric (meters) -- must convert to OpenCV (X-right, Y-down, Z-forward) for pipeline compatibility
- **Mesh format**: PLY exclusively -- loads directly with `trimesh.load()` or Open3D
- **Scale**: metric (meters), matches SMPL scale
- **Semantic labels**: 1500+ free-text, with top-100 benchmark mapping in `metadata/semantic_benchmark/map_benchmark.csv`

---

## Part 2: Architecture for Scene-Conditioned HOI Synthesis

### Key Insight: Skip Video, Go Direct to 3D

The current pipeline: Text -> Video -> Extract 3D from video
The scene pipeline: Text + Scene -> **Generate motion directly in 3D**

Since the scene provides ground-truth geometry, we skip video generation, segmentation, depth estimation, optical flow, and object mesh reconstruction. Instead, we generate SMPL motion parameters directly in scene coordinates and refine with the existing loss infrastructure.

### Recommended Approach: Retrieval/Diffusion + Test-Time Optimization

1. **Initialize** human motion via retrieval from AMASS/BABEL or a pre-trained motion diffusion model
2. **Refine** using test-time optimization with extended losses (reusing `Track_Human_Object_Mesh/losses.py`)
3. **Scene constraints** added as new loss terms (floor contact, scene penetration, balance)

### Modified Pipeline (7 stages vs current 10)

```
Text + Scene ──> [1] Load Scene ──> [2] Extract Target Object ──> [3] Generate Scene-Aware PAG
                                                                           │
     ┌─────────────────────────────────────────────────────────────────────┘
     v
[4] Generate Initial Motion ──> [5] Scene-Aware Refinement ──> [6] Export ──> [7] Evaluate
     (retrieval or diffusion)    (extended loss optimization)    (PLY seq)
```

### Stage Details

#### Stage 1: Scene Loading (`Scene_Loading/scannetpp_loader.py` -- NEW)
- Load ScanNet++ PLY mesh + semantic annotations
- Convert Z-up to OpenCV coordinates (180-deg rotation around X, similar to `Original_Code/hms/scene_init.py` line 136-138)
- Extract floor plane via RANSAC on floor-labeled vertices
- Build scene bounding box and spatial index

#### Stage 2: Target Object Extraction (`Scene_Loading/object_extractor.py` -- NEW)
- Given object category + instance ID from `segments_anno.json`, extract sub-mesh
- Compute object SDF using existing `_build_sdf_grid()` from `Track_Human_Object_Mesh/data_loading.py:401-433`
- Run part segmentation using existing `Segment_Object_Mesh/` pipeline (render multi-view, segment with SAM)
- Compute object bounding box, centroid, approach directions

#### Stage 3: Scene-Aware PAG (`Generate_PAG/generate_pag_scene.py` -- NEW)
- Fork `Generate_PAG/system_prompt_pag.md` into `system_prompt_pag_scene.md`
- Add scene context to LLM input: object dimensions, height from floor, nearby obstacles, approach directions
- PAG output format stays identical (same downstream compatibility)

#### Stage 4: Motion Generation (`Generate_Motion/` -- NEW directory)

**4A. Retrieval approach** (implement first, weeks 4-6):
- `motion_retrieval.py`: Query AMASS/BABEL with text using TMR or CLIP-based text-motion similarity
- `motion_retarget.py`: Place retrieved motion in scene -- face object, feet on floor, within arm's reach

**4B. Diffusion approach** (implement second, weeks 6-8):
- `motion_diffusion.py`: Wrap a pre-trained model (HUMANISE, MDM, or TRUMANS)
- Generated motion serves as better initialization for refinement

#### Stage 5: Scene-Aware Refinement (`Track_Human_Object_Mesh/optimizer_scene.py` -- NEW)

**Optimization variables**: SMPL-X `transl` (F,3), `global_orient` (F,3), `body_pose` (F,63)

**Reused losses** (from `Track_Human_Object_Mesh/losses.py`):
- Contact loss (PAG-driven body-part to object-part distance, lines 582-731)
- Intersection loss (SDF query for human-object penetration, lines 206-232)
- Smoothness losses (translation + rotation temporal smoothing, lines 38-66)

**New losses**:
- **Floor contact**: penalize foot-vertex distance from floor plane when feet should be grounded
- **Scene penetration**: query scene SDF with all human vertices, penalize negative values (structurally identical to existing intersect loss)
- **Balance/stability**: penalize center-of-mass displacement from support polygon
- **VPoser prior**: body pose regularization (pattern in `Original_Code/hms/human.py`)

**Key difference from current optimizer**: Current optimizes *object* poses with human fixed. Scene optimizer does the **inverse** -- optimizes *human* poses with object (from scan) fixed.

#### Stage 6: Export (`Generate_Motion/export_motion.py` -- NEW)
- Convert optimized SMPL-X parameters to per-frame PLY meshes
- Match existing output format: `human_motion_aligned/<person>/frame_XXXX.ply`

#### Stage 7: Evaluation (`Evaluation/` -- NEW directory)
- **Physical plausibility**: penetration depth (human-scene, human-object), ground contact %, foot skating
- **Contact accuracy**: body-part to object-part distance (from PAG)
- **Motion quality**: FID against AMASS, diversity, naturalness
- **Baselines**: standalone 4DHOI placed post-hoc, HUMANISE, TRUMANS, SceneDiffuser

### Module Reuse Map

| Existing Module | Status | Notes |
|---|---|---|
| `Track_Human_Object_Mesh/losses.py` | **HEAVY REUSE** | Contact, intersection, smoothness -- core of refinement |
| `Track_Human_Object_Mesh/data_loading.py` | **REUSE** | `_build_sdf_grid()`, body part segmentation mapping |
| `Track_Human_Object_Mesh/models.py` | **EXTEND** | Add `SceneData`, new loss fields |
| `Track_Human_Object_Mesh/geometry.py` | **DIRECT REUSE** | SDF query, similarity transforms |
| `Segment_Object_Mesh/` | **REUSE** | Part-segment the extracted scanned object |
| `Generate_PAG/` | **FORK** | Scene-aware version of PAG generation |
| `Original_Code/hms/scene.py` | **REFERENCE** | `MeshScenes` pattern for scene SDF loading |
| `Blender_Scripts/` | **REUSE** | Visualization of results in scene |
| `Generate_Video/` | SKIP | No video generation needed |
| `Segment_Video/` | SKIP | No video to segment |
| `Estimate_Depth/` | SKIP | Scene has ground-truth geometry |
| `Estimate_Human_Motion/` | SKIP | Motion is generated, not extracted from video |
| `Generate_Object_Mesh/` | SKIP | Object comes from scan |
| `Align_Meshes/` | SKIP | Everything is already in scene coordinates |
| `Estimate_Optical_Flow/` | SKIP | No video-based tracking |
| `Track_Object_Mesh/` | SKIP | Object is static in scene |

---

## Part 3: Timeline (3 months remaining, ~April-July 2026)

| Week | Focus | Deliverable |
|---|---|---|
| 1-2 | Scene Loading + SDF | ScanNet++ loader, object extractor, scene SDF, floor plane |
| 3 | Scene-aware PAG | Extended system prompt, scene context builder, PAG generation working |
| 4-5 | Motion retrieval + retargeting | AMASS integration, text-based retrieval, placement in scene |
| 5-6 | Core optimization loop | Scene-aware losses (floor, penetration, balance), test-time optimizer |
| 6-7 | Motion diffusion integration | Pre-trained model as initialization, end-to-end pipeline working |
| 7-8 | Quality tuning | Loss weight tuning, edge cases, diverse interaction types |
| 8-9 | Evaluation framework | Metrics implementation, baseline comparisons, benchmark setup |
| 9-11 | Experiments + ablations | Full quantitative eval on 5+ scenes, 50+ prompts |
| 11-13 | Paper writing | Draft, figures, supplementary video, submission prep |

---

## Part 4: Key Technical Challenges

1. **Scene SDF memory**: Large scenes need local SDF (2-3m box around target object, resolution 256 = ~67MB). Use existing `_build_sdf_grid()` on cropped scene geometry.

2. **Motion naturalness**: Retrieval + optimization can produce unnatural poses. Mitigate with VPoser prior and smoothness losses. Diffusion model (Phase 4B) provides better initialization.

3. **Noisy scan meshes**: ScanNet++ scans have some holes/noise. The `pysdf` library in existing code handles non-watertight meshes. Use the `padding` parameter in `_build_sdf_grid()`.

4. **Diverse interactions**: The PAG abstraction provides a unified interface -- same loss framework handles sitting, reaching, lifting, etc. via different contact edges.

5. **Coordinate system mismatch**: ScanNet++ is Z-up, pipeline is OpenCV (Y-down, Z-forward). Single rotation transform at load time.

---

## Part 5: Verification Plan

1. **Scene loading**: Load a ScanNet++ scene, visualize in Blender with coordinate axes -- verify orientation and scale
2. **Object extraction**: Extract a chair from scene, compute SDF, verify SDF values at known inside/outside points
3. **PAG generation**: Generate scene-aware PAG for "sit on chair in office" -- verify it includes spatial constraints
4. **Motion retrieval**: Retrieve a sitting motion from AMASS, place in scene -- verify feet on floor, facing object
5. **Optimization**: Run refinement on placed motion -- verify losses decrease, no penetration, contact achieved
6. **End-to-end**: Full pipeline from text + scene -> animated PLY sequence rendered in Blender inside the scene
7. **Quantitative**: Run metrics on 5 scenes x 10 prompts each, compare against baselines

---

## Part 6: Publication Angle

**Novel contributions for a top-venue paper:**
1. First framework for **text-driven 4D HOI synthesis in real scanned scenes** (zero-shot, no paired training data)
2. **Scene-aware PAG**: LLM-driven interaction structure that respects spatial constraints
3. **Test-time optimization with scene constraints**: Extends contact/intersection losses to full scene geometry
4. **ScanNet++ HOI benchmark**: First standardized evaluation for human-scene interaction on ScanNet++

**Positioning**: Combines the strengths of text-to-motion (MDM, MotionDiffuse), scene-aware HSI (HUMANISE, TRUMANS), and structured interaction reasoning (your PAG framework) into a unified zero-shot system.

---

## Part 7: Related Work to Study

### Scene-Aware Human Motion Generation
- **HUMANISE** (NeurIPS 2022): Text + scene -> SMPL motion. Uses ScanNet v1. Open source.
- **TRUMANS** (CVPR 2024): Scene-aware motion generation, demonstrated zero-shot on ScanNet++.
- **LINGO** (2024): Language-grounded interaction generation in VR-captured scenes.
- **InteractMove** (ACM MM 2025): Aligns HOI sequences with ScanNet scenes.
- **CHOIS** (ECCV 2024 Oral): Contact-guided 3D HOI generation.
- **SceneDiffuser** (CVPR 2023): Diffusion-based scene-conditioned generation.

### Motion Datasets
- **AMASS**: Unified motion capture database. Required for retrieval approach.
- **BABEL**: Text annotations for AMASS motions. Required for text-based retrieval.

### Text-to-Motion
- **MDM** (ICLR 2023): Motion Diffusion Model. Text-conditioned but not scene-aware.
- **MotionDiffuse** (2023): Text-driven motion generation with fine-grained control.
- **TMR** (2023): Text-motion retrieval model. Useful for the retrieval approach.
