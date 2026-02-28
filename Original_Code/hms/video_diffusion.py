import os
import os.path as osp
import sys
import math
import ast
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from diffusers import FluxPipeline, CogVideoXImageToVideoPipeline
from diffusers.hooks import apply_group_offloading
from transformers import AutoProcessor, VitPoseForPoseEstimation
from PIL import Image, ImageDraw
from loguru import logger

ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), osp.pardir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from hms.config import VideoDiffusionConfig, parse_config
from hms.common import get_device, to_torch, valid_str


def center_crop(image, target_width, target_height):
    img_width, img_height = image.size
    img_aspect_ratio = img_width / img_height
    target_aspect_ratio = target_width / target_height
    if img_aspect_ratio > target_aspect_ratio:
        new_height = target_height
        new_width = int(new_height * img_aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / img_aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image


def parse_det_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i + 1 :])
            json_output = json_output.split("```")[0]
            break
    try:
        json_output = ast.literal_eval(json_output)
    except Exception as e:
        end_idx = json_output.rfind('"}') + len('"}')
        truncated_text = json_output[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)
    return json_output


class VideoDiffusion(nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__()

        self.cfg = parse_config(VideoDiffusionConfig, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)

    def configure(self, mllm, *args, **kwargs):
        self.mllm = mllm
        self.weights_dtype = torch.bfloat16
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def open_detect(self, image_path, text_prompt):
        cfg = self.cfg

        logger.info(f"Detection prompt:\n{text_prompt}")

        det_res = self.mllm.chat_images(
            provider=cfg.det_provider,
            model_name=cfg.det_model,
            system_prompt=cfg.det_system_prompt,
            user_prompt=cfg.det_user_prompt.format(text_prompt),
            images=[image_path],
            task_name=cfg.det_task,
            temperature=cfg.det_temperature,
        )
        # time.sleep(4)

        det_response = det_res["response"].strip()
        logger.info(f"Detection response:\n{det_response}")

        ow, oh = det_res["original_width"], det_res["original_height"]
        nw, nh = det_res["new_width"], det_res["new_height"]
        if not det_response.startswith("```json"):
            return list()
        json_output = parse_det_json(det_response)

        res = list()
        for idx, bounding_box in enumerate(json_output):
            abs_x1 = int(bounding_box["bbox_2d"][0] / nw * ow)
            abs_y1 = int(bounding_box["bbox_2d"][1] / nh * oh)
            abs_x2 = int(bounding_box["bbox_2d"][2] / nw * ow)
            abs_y2 = int(bounding_box["bbox_2d"][3] / nh * oh)
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            res.append((abs_x1, abs_y1, abs_x2, abs_y2))

        return res

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def segment_object(self, sam2_model, sam2_predictor, image, bboxes):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        sam2_predictor.set_image(image)
        seg_masks, _, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(bboxes),  # [(x_min, y_min, x_max, y_max), ...]
            multimask_output=False,
        )
        if seg_masks.ndim == 3:
            seg_masks = np.expand_dims(seg_masks, axis=0)
        seg_mask = np.any(seg_masks > 0, axis=(0, 1))
        return seg_mask

    @torch.no_grad()
    def detect_pose(self, model, image_processor, image, mask, bboxes, threshold=0.3):
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")

        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        bboxes = [[bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] for bbox in bboxes]

        inputs = image_processor(image, boxes=[bboxes], return_tensors="pt").to(self.device)
        # This is MOE architecture, we should specify dataset indexes for each image in range 0..5
        inputs["dataset_index"] = torch.tensor([0], device=self.device)

        outputs = model(**inputs)

        pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[bboxes], threshold=threshold)
        image_pose_result = pose_results[0]  # results for first image

        width, height = image.width, image.height
        mask = mask.astype(np.int32)

        joints = np.ones((len(image_pose_result), 17, 3), dtype=np.float32) * -1
        joints[..., 2] = 0.0
        for pid, person_pose in enumerate(image_pose_result):
            for keypoint, label, score in zip(person_pose["keypoints"], person_pose["labels"], person_pose["scores"]):
                label = label.item()
                # joint_name = model.config.id2label[label]
                x, y = keypoint
                x = x.item()
                y = y.item()
                score = score.item()
                if x < 0 or y < 0 or x >= width or y >= height:
                    score = 0
                elif mask[int(y), int(x)].item() == 0:
                    score = 0
                joints[pid, label] = [x, y, score]
        return joints

    def query_mllm(self, user_prompt, images):
        response = self.mllm.chat_images(
            provider=self.cfg.icomp_provider,
            model_name=self.cfg.icomp_model,
            system_prompt=self.cfg.icomp_system_prompt,
            user_prompt=self.cfg.icomp_user_prompt.format(user_prompt),
            images=images,
            task_name=self.cfg.icomp_task,
            temperature=self.cfg.icomp_temperature,
        )["response"]
        if not valid_str(response):
            raise ValueError("Invalid response from LLM")
        return response

    def compare_images(self, prompt, images):
        assert len(images) == 2
        images = [np.array(img) for img in images]
        gap = np.ones_like(images[0][:, :20, :]) * 255
        images = np.concatenate([images[0], gap, images[1]], axis=1)
        images = Image.fromarray(images)

        max_tries = 3
        for attempt in range(max_tries):
            try:
                response = self.query_mllm(prompt, images)
                break
            except Exception as e:
                logger.warning(f"LLM query failed:\n{e}")
                if attempt == max_tries - 1:
                    raise RuntimeError(f"LLM query failed after {max_tries} attempts")

        logger.info(f"Image comparison result: {response}")
        if response == "right":
            return 1
        else:
            return 0

    def text_to_image(self, prompt, prompt_enhanced, prompt_negative, out_dir):
        logger.info("Generating image from prompt")
        frame_width = self.cfg.frame_width
        frame_height = self.cfg.frame_height

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        segment_model = build_sam2(self.cfg.segmenter_model_cfg, self.cfg.segmenter_model_path, device=self.device)
        segment_predictor = SAM2ImagePredictor(segment_model)

        pose2d_processor = AutoProcessor.from_pretrained(self.cfg.pose2d_model_path)
        pose2d_model = VitPoseForPoseEstimation.from_pretrained(self.cfg.pose2d_model_path, device_map=self.device)

        pipe = FluxPipeline.from_pretrained(self.cfg.t2i_model_path, torch_dtype=self.weights_dtype)
        # pipe.to(self.device)
        # pipe.enable_model_cpu_offload()
        # pipe.vae.enable_tiling()
        # pipe.vae.enable_slicing()

        goargs = {
            "offload_device": torch.device("cpu"),
            "onload_device": self.device,
            "offload_type": "leaf_level",
            # "num_blocks_per_group": 2,
            "use_stream": True,
        }
        apply_group_offloading(pipe.transformer, **goargs)
        apply_group_offloading(pipe.text_encoder, **goargs)
        apply_group_offloading(pipe.text_encoder_2, **goargs)
        apply_group_offloading(pipe.vae, **goargs)

        score_thresh = 10
        images = list()
        for idx in range(self.cfg.t2i_trials):
            img = pipe(
                prompt=prompt_enhanced,
                negative_prompt=prompt_negative,
                guidance_scale=3.5,
                height=frame_height,
                width=frame_width,
                num_inference_steps=50,
                output_type="pil",
            ).images[0]
            img.save(osp.join(out_dir, f"diffusion_t2i_temp_{idx}.png"))

            # Human detection
            bboxes = self.open_detect(img, "humans")
            if len(bboxes) == 0:
                continue
            # Human mask
            img_mask = self.segment_object(segment_model, segment_predictor, img, bboxes)
            # Image.fromarray(img_mask.astype(np.uint8) * 255).save(osp.join(out_dir, f"diffusion_t2i_temp_{idx}_mask.png"))
            # Pose estimation
            joints2d = self.detect_pose(pose2d_model, pose2d_processor, img, img_mask, bboxes)

            # # Visualize
            # img_clone = img.copy()
            # img_draw = ImageDraw.Draw(img_clone)
            # # Draw bounding boxes onto the image
            # for bbox in bboxes:
            #     x1, y1, x2, y2 = bbox
            #     img_draw.rectangle((x1, y1, x2, y2), outline="green", width=2)
            # # Draw joints2d onto the image
            # for person_joints in joints2d:
            #     for joint in person_joints:
            #         x, y, score = joint
            #         radius = 3
            #         img_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red", outline="red")
            # img_clone.save(osp.join(out_dir, f"diffusion_t2i_temp_{idx}_joints.png"))

            head_scores = joints2d[:, :5, 2]  # (number of persons, 5 head joints)
            head_scores = np.any(head_scores >= 0.5, axis=1, keepdims=True).astype(joints2d.dtype)
            joint_scores = np.concatenate([head_scores, joints2d[:, 5:, 2]], axis=1)  # 1 + 12
            img_score = np.mean(np.sum(joint_scores, axis=1)).item()
            logger.info(f"{idx} image score: {img_score}")

            if idx == self.cfg.t2i_trials // 2 and len(images) < 2:
                score_thresh = 8
            if img_score < score_thresh:
                continue
            images.append(img)
            if len(images) == 5:
                break
        for idx in range(len(images)):
            images[idx].save(osp.join(out_dir, f"diffusion_t2i_{idx}.png"))

        best = 0
        for i in range(1, len(images)):
            comp_res = self.compare_images(prompt, [images[best], images[i]])
            time.sleep(1)
            if comp_res == 1:
                best = i
        out = images[best]
        os.rename(
            osp.join(out_dir, f"diffusion_t2i_{best}.png"),
            osp.join(out_dir, "diffusion_t2i_best.png"),
        )

        logger.info("Finished image generation from prompt")

        del pipe
        del pose2d_model
        del pose2d_processor
        del segment_predictor
        del segment_model
        torch.cuda.empty_cache()
        gc.collect()
        return out

    def image_to_video(self, image, prompt, prompt_enhanced, prompt_negative, out_dir):
        logger.info("Generating video from image")
        assert image.width == self.cfg.frame_width
        assert image.height == self.cfg.frame_height

        pipe = CogVideoXImageToVideoPipeline.from_pretrained(self.cfg.i2v_model_path, torch_dtype=self.weights_dtype)
        pipe.to(self.device)
        pipe.enable_model_cpu_offload()
        # pipe.vae.enable_tiling()
        # pipe.vae.enable_slicing()

        out = pipe(
            prompt=prompt_enhanced,
            image=image,
            negative_prompt=prompt_negative,
            guidance_scale=6,
            num_inference_steps=50,
            output_type="pil",
        ).frames[0]

        logger.info("Finished video generation from image")

        del pipe
        torch.cuda.empty_cache()
        gc.collect()
        return out

    def forward(self, prompt, prompt_enhanced, out_dir):
        prompt_negative = self.cfg.negative_prompt
        res = {"prompt": prompt, "prompt_negative": prompt_negative}

        prompt_t2i = prompt_enhanced + f" {self.cfg.prompt_suffix_image}"
        image = self.text_to_image(prompt, prompt_t2i, prompt_negative, out_dir)
        res["image_t2i"] = image
        res["prompt_t2i"] = prompt_t2i

        prompt_i2v = prompt_enhanced + f" {self.cfg.prompt_suffix_video}"
        video = self.image_to_video(image, prompt, prompt_i2v, prompt_negative, out_dir)
        res["video"] = video
        res["video_i2v"] = video
        res["prompt_i2v"] = prompt_i2v

        return res
