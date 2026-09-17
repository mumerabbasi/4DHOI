from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

from openai import APITimeoutError, OpenAI
from PIL import Image


def extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                continue

            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts).strip()

    return ""


def parse_selected_frame(text: str, frame_paths: list[Path]) -> Path:
    raw_text = text.strip()
    frame_by_name = {path.name: path for path in frame_paths}
    if raw_text in frame_by_name:
        return frame_by_name[raw_text]

    normalized = raw_text.strip("`'\" \n\t")
    if normalized in frame_by_name:
        return frame_by_name[normalized]

    lower_to_path = {path.name.lower(): path for path in frame_paths}
    if normalized.lower() in lower_to_path:
        return lower_to_path[normalized.lower()]

    tokens = [
        token.strip("`'\".,:;!?()[]{}")
        for token in raw_text.replace("\n", " ").split()
    ]
    matches = [frame_by_name[token] for token in tokens if token in frame_by_name]
    if len(set(matches)) == 1:
        return matches[0]

    lower_matches = [
        lower_to_path[token.lower()]
        for token in tokens
        if token.lower() in lower_to_path
    ]
    if len(set(lower_matches)) == 1:
        return lower_matches[0]

    raise ValueError(
        "Expected exactly one selected frame filename, got: "
        f"{raw_text!r}. Valid choices: {', '.join(frame_by_name)}"
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        payload = json.loads(cleaned[start: end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_selection_response(text: str, frame_paths: list[Path]) -> tuple[Path, str]:
    payload = extract_json_object(text)
    if payload is None:
        return parse_selected_frame(text, frame_paths), ""

    selected_text = payload.get("selected_frame") or payload.get("winner")
    if not isinstance(selected_text, str):
        raise ValueError(
            "Expected JSON field 'selected_frame' with one candidate filename, "
            f"got: {text!r}"
        )

    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = ""
    return parse_selected_frame(selected_text, frame_paths), reason.strip()


def fallback_frame(frame_paths: list[Path]) -> Path:
    for frame_path in frame_paths:
        if frame_path.name == "frame_00.png":
            return frame_path
    return frame_paths[0]


def load_pag_prompt(path: Path) -> str:
    pag = json.loads(path.read_text(encoding="utf-8"))
    return pag["interaction"]


def encode_image_base64(image: Image.Image, jpeg_quality: int) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_file_base64(
    image_path: Path,
    max_image_side: int,
    jpeg_quality: int,
) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        image.thumbnail((max_image_side, max_image_side), resample)
        return encode_image_base64(image, jpeg_quality)


def select_best_frame(
    client: OpenAI,
    model: str,
    system_prompt: str,
    interaction: str,
    frame_paths: list[Path],
    reasoning_effort: str | None,
    max_image_side: int,
    jpeg_quality: int,
    retry_index: int = 0,
) -> dict[str, Any]:
    reminder = ""
    if retry_index > 0:
        reminder = (
            "\n\nYour previous reply was empty or invalid. "
            "Reply with a valid JSON object containing selected_frame and reason."
        )

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Human-object interaction description:\n"
                f"{interaction}\n\n"
                "Candidate first frames, in order:\n"
                + "\n".join(
                    f"{index + 1}. {frame_path.name}"
                    for index, frame_path in enumerate(frame_paths)
                )
                + "\n\nChoose the single best first frame for this interaction. "
                "Prefer the image that best shows the human, object, intended "
                "interaction, usable composition, and enough space for the later "
                "fixed-camera video. If several frames are good, choose the one "
                "with the clearest full-body composition and most plausible pose. "
                "After detailed inspection, if all images are effectively the same "
                "quality and none is clearly better, choose frame_00.png. "
                "Reply with only a JSON object in this exact "
                'shape: {"selected_frame": "frame_XX.png", "reason": "short reason"}. '
                "The selected_frame value must be one filename from the candidate "
                f"list.{reminder}"
            ),
        }
    ]
    for index, frame_path in enumerate(frame_paths):
        content.append(
            {
                "type": "text",
                "text": f"Candidate {index + 1}: {frame_path.name}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        f"{encode_image_file_base64(frame_path, max_image_side, jpeg_quality)}"
                    )
                },
            }
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    raw_text = extract_text_content(response.choices[0].message.content)
    if not raw_text.strip():
        finish_reason = response.choices[0].finish_reason
        reasoning = getattr(response.choices[0].message, "reasoning", None)
        reasoning_preview = ""
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning_preview = reasoning.strip()[:200]
        raise ValueError(
            "Model returned empty content while selecting a frame. "
            f"finish_reason={finish_reason!r}, reasoning_preview={reasoning_preview!r}"
        )
    selected_path, reason = parse_selection_response(raw_text, frame_paths)
    return {
        "selected_frame": selected_path.name,
        "reason": reason,
        "raw_response": raw_text.strip(),
    }


def resolve_pag_path(script_dir: Path, interaction_name: str, raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).resolve()
    return next((script_dir.parent / "01_Generate_PAG" / "output" / interaction_name).glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best generated first frame using one VLM request.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--pag", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--ollama_host", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--ollama_api_key", default="ollama")
    parser.add_argument("--qwen_model", default="qwen3.6:27b")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="medium",
    )
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-image-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    args = parser.parse_args()
    args.jpeg_quality = max(1, min(args.jpeg_quality, 95))
    if args.max_image_side <= 0:
        raise ValueError("--max-image-side must be a positive integer")

    script_dir = Path(__file__).resolve().parent
    video_dir = script_dir / "output" / args.interaction_name
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else video_dir / "first_frames"
    pag_path = resolve_pag_path(script_dir, args.interaction_name, args.pag)
    prompt_path = (
        Path(args.prompt_file).resolve()
        if args.prompt_file
        else script_dir / "system_prompt_select_first_frame.md"
    )
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else video_dir / "selected_first_frame.json"
    )

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No generated first frames found in: {frames_dir}")

    system_prompt = prompt_path.read_text(encoding="utf-8")
    interaction = load_pag_prompt(pag_path)
    client = OpenAI(
        base_url=args.ollama_host,
        api_key=args.ollama_api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    reasoning_effort = args.reasoning_effort
    if reasoning_effort == "none":
        reasoning_effort = None

    print(f"Interaction name: {args.interaction_name}")
    print(f"Frames directory: {frames_dir}")
    print(f"PAG file: {pag_path}")
    print(f"Prompt file: {prompt_path}")
    print(f"Model: {args.qwen_model}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Retries: {args.retries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Image payload: max side {args.max_image_side}px, JPEG quality {args.jpeg_quality}")
    print(f"Found {len(frame_paths)} candidate frames")

    if len(frame_paths) == 1:
        selected_frame = frame_paths[0]
        selection: dict[str, Any] = {
            "selected_frame": selected_frame.name,
            "reason": "Only one candidate frame was available.",
            "raw_response": "only candidate",
        }
        print(f"Only one frame found, selecting {selected_frame.name}")
    else:
        last_error: Exception | None = None
        selection = {}
        for retry_index in range(args.retries + 1):
            if retry_index == 0:
                print(f"Asking model to choose from {len(frame_paths)} frames")
            else:
                print(f"Retry {retry_index}/{args.retries}")
            try:
                selection = select_best_frame(
                    client,
                    args.qwen_model,
                    system_prompt,
                    interaction,
                    frame_paths,
                    reasoning_effort,
                    args.max_image_side,
                    args.jpeg_quality,
                    retry_index,
                )
                break
            except (APITimeoutError, ValueError) as error:
                last_error = error
                print(f"Timed out or got an invalid reply from Ollama: {error}")

        if not selection:
            selected_frame = fallback_frame(frame_paths)
            selection = {
                "selected_frame": selected_frame.name,
                "reason": (
                    "Selection model timed out or returned invalid output after "
                    f"{args.retries + 1} attempt(s); fell back to {selected_frame.name}."
                ),
            }
            print(f"Falling back to {selected_frame.name}: {last_error}")
        else:
            selected_frame_name = selection["selected_frame"]
            selected_frame = next(path for path in frame_paths if path.name == selected_frame_name)
            print(f"Model selected: {selected_frame.name}")
            if selection.get("reason"):
                print(f"Reason: {selection['reason']}")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "interaction_name": args.interaction_name,
        "selected_frame": selected_frame.name,
        "selected_frame_path": str(Path("first_frames") / selected_frame.name),
        "reason": selection.get("reason", ""),
        "model": args.qwen_model,
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected frame: {selected_frame}")
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()
