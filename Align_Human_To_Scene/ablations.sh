cd /my_workspace/4DHHOI

SCRIPT=/my_workspace/4DHHOI/4DHSI/Align_Human_To_Scene/align_human_to_scene_full_body.py
OUT=/my_workspace/4DHHOI/4DHSI/Align_Human_To_Scene/output_ablation_stage2_regularizers_maskweight1
LOGS="$OUT/logs"
mkdir -p "$LOGS"

# Edit this if you want to use fewer/more GPUs.
GPUS=(0 1 2 3)

DATA_ARGS="
  --adam_iters 2000
  --adam_lr 1e-3
  --mask_weight 1
  --intersect_weight_start 0
  --intersect_weight_end 15
  --nocontact_weight_start 500
  --nocontact_weight_end 500
  --floor_nocontact_weight_start 200
  --floor_nocontact_weight_end 200
  --scene_intersect_weight_start 0
  --scene_intersect_weight_end 0
  --self_intersect_weight_start 0
  --self_intersect_weight_end 0
"

run_ablation () {
  JOB_ID=$1
  VIDEO=$2
  NAME=$3
  REG_ARGS=$4

  GPU=${GPUS[$((JOB_ID % ${#GPUS[@]}))]}

  CUDA_VISIBLE_DEVICES=$GPU conda run -n sam3d-objects python "$SCRIPT" \
    --video_name "$VIDEO" \
    --device cuda:0 \
    --output_root "$OUT/$NAME/$VIDEO" \
    $DATA_ARGS \
    $REG_ARGS \
    > "$LOGS/${NAME}_${VIDEO}.log" 2>&1 &
}

JOB=0

for VIDEO in video_01 video_02; do
  run_ablation $JOB "$VIDEO" B00_baseline_all_regularizers "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 10
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B01_no_root_trans_gvhmr "
    --root_trans_gvhmr_weight 0
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 10
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B02_weak_root_trans_gvhmr_weight2 "
    --root_trans_gvhmr_weight 2
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 10
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B03_no_root_orient_gvhmr "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 0
    --pose_gvhmr_weight 10
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B04_no_pose_gvhmr "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 0
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B05_weak_pose_gvhmr_weight1 "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 1
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B06_no_scale_prior "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 10
    --scale_prior_weight 0
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B07_no_angle_prior "
    --root_trans_gvhmr_weight 20
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 10
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 0
  "
  JOB=$((JOB+1))

  run_ablation $JOB "$VIDEO" B08_weak_root_trans_and_pose "
    --root_trans_gvhmr_weight 2
    --root_orient_gvhmr_weight 20
    --pose_gvhmr_weight 1
    --scale_prior_weight 25
    --angle_weight_start 0
    --angle_weight_end 1
  "
  JOB=$((JOB+1))
done

wait
echo "Finished stage-2 regularizer ablations: $OUT"
