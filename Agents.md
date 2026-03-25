# Agents.md for 4DHOI

## First Rule

Every newly spawned agent working inside this repo must read `Agents.md` before:

- describing the method,
- proposing architectural changes,
- editing stage interfaces,
- changing losses,
- touching saved outputs,
- or comparing the thesis against other papers.

If there is any conflict between a quick assumption and `Agents.md`, follow `Agents.md` and then verify against code.

## Repo Identity

Treat this repo as a full-stack 4D HOI system:

- text -> PAG,
- PAG -> first frame,
- first frame -> video,
- video -> masks / depth / human motion / object mesh,
- alignment -> object tracking -> PAG-guided refinement.

Do not treat `Original_Code/` as the current method. It is background and ancestry, not the main implementation.

## Default Method to Assume

Unless the task explicitly says otherwise, assume the main claimed method is the pipeline that ends in:

- `Track_Human_Object_Mesh/track_human_object_mesh.py`

Interpret the current branches as:

- `Track_Human_Object_Mesh/`: default main refinement branch,
- `Track_Human_Object_Joint/`: experimental alternative branch using CoTracker reprojection inside final refinement,
- `Estimate_Optical_Flow/estimate_optical_flow_waft.py`: optional tracking alternative,
- `Original_Code/`: reference only, not the default executable method.

## Important Current Truths

Agents should preserve these implementation truths unless code changes prove otherwise:

- the repo is modular rather than end-to-end trained,
- the PAG is central to both generation and refinement,
- the final default refinement stage currently keeps the human sequence fixed,
- object motion is what gets optimized in the default refinement branch,
- object-part labels are required for the main refinement method,
- static camera prompting is intentional and method-relevant, not cosmetic.

## Navigation Shortcuts

Primary high-level docs and entry points:

- `Agents.md`
- `README.md`
- `run_video_pipeline.py`

Main stage entry points:

- `Generate_PAG/generate_pag.py`
- `Generate_Video/generate_first_frame.py`
- `Generate_Video/select_first_frame.py`
- `Generate_Video/generate_video.py`
- `Segment_Video/segment_video.py`
- `Generate_Object_Mesh/generate_objects_meshes.py`
- `Estimate_Depth/estimate_depth.py`
- `Estimate_Human_Motion/estimate_human_motion.py`
- `Align_Meshes/align_meshes.py`
- `Segment_Object_Mesh/segment_meshes.py`
- `Track_Object_Mesh/track_object_mesh.py`
- `Track_Human_Object_Mesh/track_human_object_mesh.py`

Main output roots:

- `Generate_PAG/output/`
- `Generate_Video/output/`
- `Segment_Video/output/`
- `Generate_Object_Mesh/output/`
- `Estimate_Depth/output/`
- `Estimate_Human_Motion/output/`
- `Align_Meshes/output/`
- `Segment_Object_Mesh/output/`
- `Estimate_Optical_Flow/output_cotracker/`
- `Track_Object_Mesh/output/`
- `Track_Human_Object_Mesh/output/`

## Working Norms

- Ground claims in current code, current saved outputs, and `Agents.md`.
- Do not rewrite the thesis narrative around `Original_Code/`.
- Do not silently assume the README is perfectly up to date; verify against code.
- Preserve generated outputs, run summaries, overlays, and debug CSVs unless the user explicitly asks to remove them.
- When describing the method, say whether you are referring to:
  - the intended full pipeline,
  - the current default implemented branch,
  - or the experimental branch.
- If you touch refinement code, check both:
  - `Track_Human_Object_Mesh/`
  - `Track_Human_Object_Joint/`
  so you do not incorrectly generalize from one branch to the other.

## External Dependencies to Remember

This repo expects several sibling repos or external backends:

- `GVHMR/`
- `sam3/`
- `sam-3d-objects/`
- `Depth-Anything-3/`
- optional `WAFT/`
- Ollama or OpenAI-compatible endpoints for some language / VLM stages

Agents should not assume the entire pipeline runs in one environment. Check `Conda_Environments/` and stage scripts before changing dependency assumptions.

## If You Spawn Again

When a future agent is spawned from this repo, the first actions should be:

1. Read `Agents.md`.
2. Skim `README.md` and `run_video_pipeline.py`.
3. Confirm whether the task concerns:
   - generation,
   - reconstruction/alignment,
   - object tracking,
   - or final refinement.
4. Only then inspect the relevant stage code.
