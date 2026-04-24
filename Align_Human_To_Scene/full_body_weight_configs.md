# Full-Body Weight Configs

Reference weight presets for `align_human_to_scene_full_body.py`.

## Baseline

Original defaults before retuning.

- `mask_weight = 0.1`
- `front_weight = 20`
- `root_trans_gvhmr_weight = 10`
- `root_orient_gvhmr_weight = 10`
- `pose_gvhmr_weight = 10`
- `betas_gvhmr_weight = 10`
- `scale_prior_weight = 10`
- `intersect_weight_start = 0`
- `intersect_weight_end = 10`
- `floor_intersect_weight_start = 0`
- `floor_intersect_weight_end = 10`
- `nocontact_weight_start = 1000`
- `nocontact_weight_end = 1000`
- `floor_nocontact_weight_start = 1000`
- `floor_nocontact_weight_end = 1000`
- `angle_weight_start = 0`
- `angle_weight_end = 1`
- `self_intersect_weight_start = 0`
- `self_intersect_weight_end = 1e-5`

## Retune 1

First rebalance toward silhouette preservation.

- `mask_weight = 500`
- `front_weight = 500`
- `root_trans_gvhmr_weight = 20`
- `root_orient_gvhmr_weight = 20`
- `pose_gvhmr_weight = 10`
- `betas_gvhmr_weight = 10`
- `scale_prior_weight = 25`
- `intersect_weight_start = 0`
- `intersect_weight_end = 15`
- `floor_intersect_weight_start = 0`
- `floor_intersect_weight_end = 20`
- `nocontact_weight_start = 500`
- `nocontact_weight_end = 500`
- `floor_nocontact_weight_start = 200`
- `floor_nocontact_weight_end = 200`
- `angle_weight_start = 0`
- `angle_weight_end = 1`
- `self_intersect_weight_start = 0`
- `self_intersect_weight_end = 1e-5`

## Retune 2

Retune 1 plus a stronger front term. This is the current script default.

- Same as Retune 1, except:
- `front_weight = 1500`

## Baseline Commands

Video 1:

```bash
/root/miniconda3/envs/sam3d-objects/bin/python 4DHSI/Align_Human_To_Scene/align_human_to_scene_full_body.py \
  --video_name video_01 \
  --device cuda:0 \
  --output_root 4DHSI/Align_Human_To_Scene/output_full_body_baseline/video_01 \
  --mask_weight 0.1 \
  --front_weight 20 \
  --root_trans_gvhmr_weight 10 \
  --root_orient_gvhmr_weight 10 \
  --pose_gvhmr_weight 10 \
  --betas_gvhmr_weight 10 \
  --scale_prior_weight 10 \
  --intersect_weight_start 0 \
  --intersect_weight_end 10 \
  --floor_intersect_weight_start 0 \
  --floor_intersect_weight_end 10 \
  --nocontact_weight_start 1000 \
  --nocontact_weight_end 1000 \
  --floor_nocontact_weight_start 1000 \
  --floor_nocontact_weight_end 1000 \
  --angle_weight_start 0 \
  --angle_weight_end 1 \
  --self_intersect_weight_start 0 \
  --self_intersect_weight_end 1e-5
```

Video 2:

```bash
/root/miniconda3/envs/sam3d-objects/bin/python 4DHSI/Align_Human_To_Scene/align_human_to_scene_full_body.py \
  --video_name video_02 \
  --device cuda:0 \
  --output_root 4DHSI/Align_Human_To_Scene/output_full_body_baseline/video_02 \
  --mask_weight 0.1 \
  --front_weight 20 \
  --root_trans_gvhmr_weight 10 \
  --root_orient_gvhmr_weight 10 \
  --pose_gvhmr_weight 10 \
  --betas_gvhmr_weight 10 \
  --scale_prior_weight 10 \
  --intersect_weight_start 0 \
  --intersect_weight_end 10 \
  --floor_intersect_weight_start 0 \
  --floor_intersect_weight_end 10 \
  --nocontact_weight_start 1000 \
  --nocontact_weight_end 1000 \
  --floor_nocontact_weight_start 1000 \
  --floor_nocontact_weight_end 1000 \
  --angle_weight_start 0 \
  --angle_weight_end 1 \
  --self_intersect_weight_start 0 \
  --self_intersect_weight_end 1e-5
```
