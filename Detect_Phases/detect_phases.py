"""
Detect interaction phases (approach, grab, release) from a generated video
using Qwen3-VL via Ollama.

Sends uniformly sampled frames to the VLM with the interaction description
from the PAG file, and asks it to identify frame ranges for each phase.
Outputs a JSON with phase boundaries that can be used for T-PAG loss scheduling.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI


OLLAMA_HOST = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen3-vl:32b-thinking"

PHASES = ["approach", "grab", "release"]


def find_single_mp4(video_dir: Path) -> Path:
    """Find exactly one MP4 file in a directory."""
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if len(mp4_files) == 0:
        raise RuntimeError(f"No .mp4 files found in {video_dir}")
    if len(mp4_files) > 1:
        print(f"Warning: multiple .mp4 files in {video_dir}, using {mp4_files[0].name}")
    return mp4_files[0]


def extract_frames(video_path: Path) -> tuple[list[np.ndarray], int]:
    """Extract all frames from a video. Returns (frames_bgr, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {video_path}")
    return frames, fps


def sample_frame_indices(total_frames: int, num_samples: int) -> list[int]:
    """Return uniformly spaced frame indices."""
    if total_frames <= num_samples:
        return list(range(total_frames))
    return [int(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)]


def encode_frame_base64(frame_bgr: np.ndarray) -> str:
    """Encode a BGR frame to a JPEG base64 string."""
    _, buffer = cv2.imencode(".jpg", frame_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def build_messages(
    frames_bgr: list[np.ndarray],
    sampled_indices: list[int],
    interaction_desc: str,
    system_prompt: str,
) -> list[dict]:
    """Build the chat messages with interleaved frame images and labels."""
    user_content: list[dict] = []

    user_content.append({
        "type": "text",
        "text": (
            f"Interaction description: {interaction_desc}\n\n"
            f"Below are {len(sampled_indices)} frames sampled from the video "
            f"({len(frames_bgr)} total frames). The label before each image "
            f"shows its original frame index."
        ),
    })

    for idx in sampled_indices:
        user_content.append({
            "type": "text",
            "text": f"[Frame {idx}]",
        })
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_frame_base64(frames_bgr[idx])}",
            },
        })

    user_content.append({
        "type": "text",
        "text": (
            f"\nThe video has {len(frames_bgr)} frames total (indices 0 to "
            f"{len(frames_bgr) - 1}). You saw a sampled subset above. Based on "
            f"these frames, identify the start and end frame indices for each "
            f"phase across the FULL frame range. Interpolate between sampled "
            f"frames as needed."
        ),
    })

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen responses."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_phase_response(text: str, total_frames: int) -> dict:
    """Parse the JSON response from the VLM into phase boundaries."""
    text = strip_thinking(text)

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not find JSON in VLM response:\n{text}")

    raw = json.loads(json_match.group())

    phases: dict[str, dict] = {}
    for phase in PHASES:
        entry = raw.get(phase, {})
        start = entry.get("start")
        end = entry.get("end")

        if start is not None:
            start = max(0, min(int(start), total_frames - 1))
        if end is not None:
            end = max(0, min(int(end), total_frames - 1))

        phases[phase] = {"start": start, "end": end}

    return phases


def load_interaction_description(pag_path: Path) -> str:
    """Load the interaction description from a PAG JSON file."""
    with open(pag_path) as f:
        pag = json.load(f)
    return pag.get("interaction", "A person interacting with an object.")


def resolve_pag_path(script_dir: Path, video_name: str) -> Path | None:
    """Find the PAG JSON file for a given video."""
    pag_dir = script_dir.parent / "Generate_PAG" / "output" / video_name
    if not pag_dir.is_dir():
        return None
    pag_files = sorted(pag_dir.glob("output_pag_*.json"))
    if pag_files:
        return pag_files[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect interaction phases (approach, grab, release) from video."
    )
    parser.add_argument(
        "--video_name", default="video_01",
        help="Name of the video subdirectory (default: video_01)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Number of frames to sample and send to the VLM (default: 16)",
    )
    parser.add_argument(
        "--pag_file", default=None,
        help="Path to PAG JSON file. If not given, auto-resolved from Generate_PAG/output/.",
    )
    parser.add_argument(
        "--video_dir", default=None,
        help="Path to directory containing the .mp4 file. If not given, auto-resolved.",
    )
    parser.add_argument(
        "--system_prompt", default=None,
        help="Path to system prompt markdown file. If not given, uses system_prompt_phases.md.",
    )
    parser.add_argument(
        "--ollama_host", default=OLLAMA_HOST,
        help=f"Ollama API host (default: {OLLAMA_HOST})",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output" / args.video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load system prompt
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_phases.md"
    )
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    print(f"System prompt: {system_prompt_path}")

    # Resolve video path
    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        video_dir = script_dir.parent / "Generate_Video" / "output" / args.video_name
    video_path = find_single_mp4(video_dir)
    print(f"Video: {video_path}")

    # Resolve PAG path for interaction description
    if args.pag_file:
        pag_path = Path(args.pag_file)
    else:
        pag_path = resolve_pag_path(script_dir, args.video_name)

    if pag_path and pag_path.exists():
        interaction_desc = load_interaction_description(pag_path)
        print(f"PAG: {pag_path}")
    else:
        interaction_desc = "A person interacting with an object."
        print("No PAG file found, using generic interaction description.")
    print(f"Interaction: {interaction_desc[:100]}...")

    # Extract and sample frames
    frames, fps = extract_frames(video_path)
    total_frames = len(frames)
    sampled_indices = sample_frame_indices(total_frames, args.num_frames)
    print(f"Total frames: {total_frames} ({fps} fps)")
    print(f"Sampled {len(sampled_indices)} frames: {sampled_indices}")

    # Query VLM
    client = OpenAI(base_url=args.ollama_host, api_key=OLLAMA_API_KEY)
    messages = build_messages(frames, sampled_indices, interaction_desc, system_prompt)

    print(f"Querying {args.model} via Ollama...")
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content or ""
    print(f"\nRaw VLM response:\n{strip_thinking(raw_text)}\n")

    # Parse phases
    phases = parse_phase_response(raw_text, total_frames)

    # Build output
    result = {
        "video": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "sampled_frame_indices": sampled_indices,
        "interaction": interaction_desc,
        "model": args.model,
        "phases": phases,
    }

    out_path = output_dir / "detected_phases.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Summary
    for phase_name, bounds in phases.items():
        s, e = bounds["start"], bounds["end"]
        if s is not None and e is not None:
            print(f"  {phase_name:10s}: frames {s} - {e}")
        else:
            print(f"  {phase_name:10s}: not detected")


if __name__ == "__main__":
    main()
