from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract_first_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    ok, frame_bgr = capture.read()
    capture.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return frame_bgr


def select_best_mask(output: dict) -> np.ndarray:
    masks = output.get("masks")
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM3 returned no masks for prompt 'person'.")

    scores = output.get("scores")
    best_idx = 0
    if scores is not None and len(scores) == len(masks):
        score_vals = []
        for s in scores:
            if hasattr(s, "item"):
                score_vals.append(float(s.item()))
            else:
                score_vals.append(float(s))
        best_idx = int(np.argmax(score_vals))

    mask = masks[best_idx]
    if hasattr(mask, "detach"):
        mask = mask.detach()
    if hasattr(mask, "cpu"):
        mask = mask.cpu().numpy()
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return (mask > 0.5).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract first frame and segment human with SAM3."
    )
    parser.add_argument(
        "--video",
        default="../Generate_Video/output/video_03",
        help="Directory containing exactly one video file.",
    )
    parser.add_argument("--outdir", default="./output")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="SAM3 confidence threshold.",
    )
    args = parser.parse_args()

    video_dir = Path(args.video).resolve()
    outdir = Path(args.outdir).resolve()

    video_files = [
        p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if len(video_files) != 1:
        raise ValueError(
            f"--video must contain exactly one video file, found {len(video_files)} in {video_dir}"
        )
    video_path = video_files[0]

    output_dir = outdir / video_dir.name / "first_frame"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "frame_0000.png"
    mask_path = output_dir / "mask_0000.png"

    frame_bgr = extract_first_frame(video_path)
    cv2.imwrite(str(frame_path), frame_bgr)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    model = build_sam3_image_model().to(args.device)
    processor = Sam3Processor(model, confidence_threshold=args.confidence)
    state = processor.set_image(Image.fromarray(frame_rgb))
    output = processor.set_text_prompt(state=state, prompt="person")
    mask = select_best_mask(output)
    processor.reset_all_prompts(state)

    cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    print(f"Saved frame: {frame_path}")
    print(f"Saved mask:  {mask_path}")


if __name__ == "__main__":
    main()
