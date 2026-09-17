# Module 12: Joint Human-Object Tracking

This module replaces the old `10_Track_Object_Mesh -> 11_Track_Human_Object_Mesh`
sequence with one optimizer state.

The entrypoint is:

```bash
python 01_track_human_object_joint.py --interaction_name interaction_01
```

It reads object point tracks directly from module 07, frame-0 aligned meshes and
transforms from module 09, frozen GVHMR SMPL-X humans from module 06, masks from
module 03, object part/SDF data from module 08, and PAG constraints from module
01.

Stage 1 optimizes object SE(3) trajectories and bounded global object scale from
2D point tracks.  Stage 2 continues the same variables with module-11-style
contact, non-penetration, drift, object-mask, and object-part-mask losses.  Human
SMPL-X motion is fixed in v1.

Outputs are written to:

```text
output/<interaction_name>/
```

Each object gets `poses.json`, `transform_refined.json`, `delta_stats.json`, and
`meshes/frame_XXXX.ply`.  Each human gets fixed SMPL-X mesh frames and
`fixed_input_stats.json`.
