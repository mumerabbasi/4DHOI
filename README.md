<div align="center">

# 4DHOI

### From Text Prompt to 4D Human-Object Interaction

A modular research pipeline that turns a text prompt into a temporally coherent 4D human-object interaction: it builds a structured interaction graph, generates a video, reconstructs human and object geometry, aligns everything in a shared 3D scene, labels semantic object parts, tracks object motion, and jointly refines the final sequence.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-LLM_%2B_VLM-000000?logo=ollama&logoColor=white)
![FLUX](https://img.shields.io/badge/FLUX.1--dev-First_Frame-F97316)
![Wan 2.2](https://img.shields.io/badge/Wan_2.2-Image_to_Video-2563EB)
![Qwen-VL](https://img.shields.io/badge/Qwen_VL-Detection_%26_Selection-7C3AED)
![SAM3](https://img.shields.io/badge/SAM3-Video_%26_Part_Segmentation-0F766E)
![SAM3D](https://img.shields.io/badge/SAM3D-Object_Meshes-475569)
![Depth Anything 3](https://img.shields.io/badge/Depth_Anything_3-Monocular_Depth-16A34A)
![GVHMR](https://img.shields.io/badge/GVHMR-Human_Motion-1D4ED8)
![CoTracker / WAFT](https://img.shields.io/badge/CoTracker_%2F_WAFT-Object_Tracking-4B5563)

</div>

---

## Highlights

| Capability | Detail |
|:-----------|:-------|
| Structured interaction planning | A text prompt is first converted into a **Part Affordance Graph (PAG)** that describes the interaction, relevant objects, object parts, and state changes |
| Video synthesis | **FLUX.1-dev** samples candidate first frames, a **Qwen-VL** tournament picks the strongest one, and **Wan 2.2** expands it into a locked-camera video |
| 4D scene recovery | **Qwen-VL + SAM3** segment humans, objects, and parts, **Depth Anything 3** estimates monocular depth, and **GVHMR** recovers human motion |
| Object reasoning | **SAM3D** reconstructs first-frame object meshes, rendered-view segmentation labels semantic parts, and **CoTracker / WAFT** provides motion cues |
| Final refinement | Human and object trajectories are jointly optimized with tracking, mask, part, smoothness, contact, and intersection-aware losses |

---

## Demo

https://github.com/user-attachments/assets/5880ff57-1e0d-4a90-93dc-c89f6ff8a941

**Text prompt:** `a person moving an iron on an ironing board while standing`

---

## What It Does

4DHOI starts from language rather than captured motion. The pipeline first converts a prompt into a Part Affordance Graph (PAG), then uses that structured interaction description to drive both video generation and downstream reconstruction.

From there, the repo:

1. samples multiple first-frame candidates with FLUX and selects the best one with a VLM tournament
2. generates a fixed-camera interaction video with Wan image-to-video
3. segments humans, objects, and object parts with Qwen-VL + SAM3
4. reconstructs object meshes, estimates monocular depth, and recovers human motion
5. aligns human and object assets into a shared 3D camera frame
6. labels semantic object parts, tracks object motion over time, and jointly refines the final 4D interaction sequence

This makes the repository more than a text-to-video demo: it is a prompt-to-geometry pipeline for recovering a structured, editable human-object interaction in 4D.

---

## Architecture

```text
Text Prompt
    |
    v
+-----------------------------+
|  PAG Generation             |  DeepSeek / Ollama -> interaction, parts, states
+------------+----------------+
             | structured PAG
             v
+-----------------------------+
|  First Frame Generation     |  FLUX.1-dev sampling + VLM tournament selection
+------------+----------------+
             | selected frame
             v
+-----------------------------+
|  Video Generation           |  Wan 2.2 image-to-video with locked camera
+----+---------------+--------+
     |               |
     |               +---------------------> Qwen-VL + SAM3 video masks
     |               +---------------------> Depth Anything 3
     |               +---------------------> GVHMR human motion
     v
+-----------------------------+
|  Object Mesh Reconstruction |  SAM3D from first-frame object masks
+------------+----------------+
             |
             v
+-----------------------------+
|  Alignment + Part Labeling  |  Depth/mask chamfer alignment + rendered-view part segmentation
+------------+----------------+
             |
             v
+-----------------------------+
|  Object Tracking            |  CoTracker / WAFT cues + per-object SE(3) optimization
+------------+----------------+
             |
             v
+-----------------------------+
|  Joint HOI Refinement       |  Contact, part, tracking, smoothness, and intersection losses
+-----------------------------+
```

---

## Key Pipeline Stages

| Stage | Role | Main folders |
|:------|:-----|:-------------|
| PAG generation | Convert free-form language into a structured Part Affordance Graph (PAG) that describes the interaction, objects, parts, and state changes | `Generate_PAG/` |
| Video generation | Sample candidate first frames, select the best one, and expand it into a fixed-camera interaction video | `Generate_Video/` |
| Scene understanding | Segment humans/objects/parts, estimate monocular depth, and recover human motion | `Segment_Video/`, `Estimate_Depth/`, `Estimate_Human_Motion/` |
| Object reconstruction | Reconstruct object meshes from the selected first frame and its masks | `Generate_Object_Mesh/` |
| Alignment | Register human and object assets into one shared camera-centric 3D frame | `Align_Meshes/` |
| Part labeling and motion cues | Render aligned meshes, label semantic parts, and estimate tracking cues from video | `Segment_Object_Mesh/`, `Estimate_Optical_Flow/` |
| Final 4D refinement | Track object pose over time and jointly optimize the full human-object sequence | `Track_Object_Mesh/`, `Track_Human_Object_Mesh/` |

---

## Repository Structure

```text
4DHOI/
├── Generate_PAG/              # Prompt inputs + Part Affordance Graph generation
├── Generate_Video/            # First-frame sampling, selection, and video generation
├── Segment_Video/             # Human/object/part segmentation across the generated video
├── Generate_Object_Mesh/      # First-frame object mesh reconstruction
├── Estimate_Depth/            # Monocular depth estimation and point cloud export
├── Estimate_Human_Motion/     # GVHMR-based human motion recovery and export
├── Align_Meshes/              # Human/object alignment in a shared 3D frame
├── Segment_Object_Mesh/       # Render-time semantic object-part segmentation
├── Estimate_Optical_Flow/     # CoTracker / WAFT motion cues
├── Track_Object_Mesh/         # Per-object SE(3) tracking
├── Track_Human_Object_Mesh/   # Final joint human-object refinement
├── Blender_Scripts/           # Visualization and import helpers
└── Conda_Environments/        # Environment definitions
```

---

## Getting Started

### Prerequisites

- Conda / Miniconda
- NVIDIA GPU recommended for the generation and reconstruction stages
- `ffmpeg` and Blender for visualization and mesh-processing steps
- An OpenAI-compatible endpoint for the LLM/VLM stages; the repo is currently wired for Ollama
- External dependencies cloned alongside this repo: `GVHMR/`, `Depth-Anything-3/`, `sam3/`, `sam-3d-objects/`, and `WAFT/`

### Setup

```bash
git clone https://github.com/mumerabbasi/4DHOI.git
cd 4DHOI
conda env create -f Conda_Environments/4dhoi.yml
conda activate 4dhoi
```

Expected workspace layout:

```text
workspace/
├── 4DHOI/
├── GVHMR/
├── Depth-Anything-3/
├── sam3/
├── sam-3d-objects/
└── WAFT/
```

### Example Run Order

```bash
python Generate_PAG/generate_pag.py --video_name video_01
python Generate_Video/generate_first_frame.py --video_name video_01
python Generate_Video/select_first_frame.py --video_name video_01
python Generate_Video/generate_video.py --video_name video_01
python Segment_Video/segment_video.py --video_name video_01
python Generate_Object_Mesh/generate_objects_meshes.py --video_name video_01
python Estimate_Depth/estimate_depth.py --video_name video_01
python Estimate_Human_Motion/estimate_human_motion.py --video_name video_01
python Align_Meshes/align_meshes.py --video_name video_01
python Segment_Object_Mesh/render_mesh_views.py --video_name video_01
python Segment_Object_Mesh/segment_renders_sam3.py --video_name video_01
python Estimate_Optical_Flow/estimate_optical_flow_cotracker.py --video_name video_01
python Track_Object_Mesh/track_object_mesh.py --video_name video_01
python Track_Human_Object_Mesh/track_human_object_mesh.py --video_name video_01
```

Prompt inputs live in `Generate_PAG/input_prompts/<video_name>/`. Most stages write outputs to `<stage>/output/<video_name>/`, and the final refined sequence is saved under `Track_Human_Object_Mesh/output/<video_name>/`.

The repository is intentionally modular, so individual stages can be swapped, rerun, or debugged without rebuilding the entire pipeline from scratch.

---

## Acknowledgements

This project integrates components and tooling around GVHMR, SAM3, SAM3D, Depth Anything 3, WAFT, and CoTracker.
