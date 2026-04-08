# ScanNet++ Notes for 4DHOI

## Purpose

This file collects the ScanNet++-specific decisions for the scene-conditioned 4DHOI extension:

- which scenes to start with
- what to download
- which image stream to use
- which depth source to use
- what scene metadata matters for the pipeline

## Recommended 5 Starter Scenes

Current furniture-affordance starter pack:

1. `1b75758486` - Conference Room
2. `4ba22fa7e4` - Office Day
3. `8d563fc2cc` - Office Night
4. `bb87c292ad` - Kitchen
5. `e8ea9b4da8` - Bedroom

Backup scenes:

- `0a5c013435` - Utility Room
- `d415cc449b` - Tool Room

Why these scenes:

- chairs, tables, desks, counters, and beds are good affordance targets
- they support simple first interactions such as lifting a chair or leaning on a table
- they are a good fit for a first scene-conditioned HOI benchmark

## What to Download First

Download the default assets for the selected scenes and prioritize:

- `scans/mesh_aligned_0.05.ply`
- `scans/mesh_aligned_0.05_semantic.ply`
- `scans/segments.json`
- `scans/segments_anno.json`
- `dslr/resized_undistorted_images` or `dslr/resized_images`
- `dslr/colmap`
- `iphone/rgb` or decoded RGB frames
- `iphone/depth` or decoded depth frames

Defer unless needed:

- point clouds
- hi-res DSLR
- panocam

## Camera Choice

### Preferred path

Use:

- **undistorted DSLR images** for scene appearance and first-frame conditioning
- **rendered scene depth** for the same DSLR view during alignment

Why:

- DSLR gives stronger visual quality for first-frame editing and video generation
- undistorted DSLR avoids fisheye issues
- rendered depth from the aligned scene mesh is clean and scene-consistent

### Backup path

Use:

- **iPhone RGB + native iPhone depth**

Why:

- simpler RGB-depth pairing
- perspective images without fisheye handling

Tradeoff:

- lower visual quality than DSLR

## Important ScanNet++ Facts for This Project

- ScanNet++ scenes are metric and mesh-aligned.
- DSLR raw imagery is fisheye, but undistorted DSLR images are available for pinhole workflows.
- iPhone depth is available directly.
- Depth for DSLR and iPhone views can also be rendered from the aligned scene mesh using the official toolkit.
- `segments.json` and `segments_anno.json` are central for instance extraction.

## What the Scene Layer Must Provide

The scene-processing code should be able to produce:

- scene ID
- chosen camera/view ID
- camera intrinsics and extrinsics
- candidate object instances for a category such as `chair`
- projected masks and bboxes for each candidate instance in the chosen view
- the selected target instance mesh in scene coordinates
- nearby context such as table proximity or free floor space

## Current Manual Assumption

For now, the following can be selected manually:

- scene
- camera view
- target object instance

Automation can come later with a VLM or scene-ranking module.

## References

- ScanNet++ documentation: <https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation>
- ScanNet++ changelog: <https://scannetpp.mlsg.cit.tum.de/scannetpp/changelog>
- ScanNet++ toolkit: <https://github.com/scannetpp/scannetpp>
- Public room-type mapping used for scene shortlist: <https://www.mdpi.com/2313-433X/10/12/330/xml>
