from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

OLLAMA_HOST = "http://127.0.0.1:11434/v1"
OLLAMA_API_KEY = "ollama"
QWEN_MODEL = "qwen3-vl:32b-thinking"


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


def parse_choice(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in {"left", "right"}:
        return normalized

    tokens = [
        token.strip(".,:;!?()[]{}\"'")
        for token in normalized.split()
    ]
    choices = [token for token in tokens if token in {"left", "right"}]
    if len(choices) == 1:
        return choices[0]

    raise ValueError(f"Expected 'left' or 'right', got: {text!r}")


def load_pag_prompt(path: Path) -> str:
    pag = json.loads(path.read_text(encoding="utf-8"))
    return pag["interaction"]


def encode_image_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_side_by_side(left_path: Path, right_path: Path) -> Image.Image:
    with Image.open(left_path) as left_source:
        left_img = left_source.convert("RGB")
    with Image.open(right_path) as right_source:
        right_img = right_source.convert("RGB")

    canvas = Image.new(
        "RGB",
        (left_img.width + right_img.width, max(left_img.height, right_img.height)),
        color=(255, 255, 255),
    )
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width, 0))
    return canvas


def compare_frames(
    client: OpenAI,
    model: str,
    system_prompt: str,
    interaction: str,
    left_path: Path,
    right_path: Path,
    reasoning_effort: str | None,
    retry_index: int = 0,
) -> dict[str, str]:
    comparison_image = make_side_by_side(left_path, right_path)
    image_b64 = encode_image_base64(comparison_image)
    comparison_image.close()

    reminder = ""
    if retry_index > 0:
        reminder = (
            "\n\nYour previous reply was empty or invalid. "
            "Reply with exactly one word only: left or right."
        )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Human-object interaction description:\n"
                            f"{interaction}\n\n"
                            "Choose the better image. Reply with exactly one word: "
                            "left or right."
                            f"{reminder}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
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
    choice = parse_choice(raw_text)
    winner_path = left_path if choice == "left" else right_path
    return {
        "left": left_path.name,
        "right": right_path.name,
        "winner": winner_path.name,
        "choice": choice,
        "raw_response": raw_text.strip(),
    }


def run_tournament(
    client: OpenAI,
    model: str,
    system_prompt: str,
    interaction: str,
    frames: list[Path],
    reasoning_effort: str | None,
    pair_retries: int,
) -> tuple[Path, list[dict[str, Any]]]:
    current_round = list(frames)
    rounds: list[dict[str, Any]] = []
    round_index = 1

    while len(current_round) > 1:
        next_round: list[Path] = []
        pairs: list[dict[str, Any]] = []
        total_pairs = (len(current_round) + 1) // 2

        print(f"Round {round_index}: {len(current_round)} candidate frames")

        for pair_start in range(0, len(current_round), 2):
            left_path = current_round[pair_start]
            right_index = pair_start + 1
            pair_number = pair_start // 2 + 1

            if right_index >= len(current_round):
                print(
                    f"  Pair {pair_number}/{total_pairs}: {left_path.name} advances automatically"
                )
                next_round.append(left_path)
                pairs.append(
                    {
                        "left": left_path.name,
                        "right": None,
                        "winner": left_path.name,
                        "choice": "bye",
                        "raw_response": "bye",
                    }
                )
                continue

            right_path = current_round[right_index]
            last_error: Exception | None = None
            result: dict[str, str] | None = None
            for retry_index in range(pair_retries + 1):
                if retry_index == 0:
                    print(
                        f"  Pair {pair_number}/{total_pairs}: "
                        f"{left_path.name} vs {right_path.name}"
                    )
                else:
                    print(
                        f"    Retry {retry_index}/{pair_retries}: "
                        f"{left_path.name} vs {right_path.name}"
                    )
                try:
                    result = compare_frames(
                        client,
                        model,
                        system_prompt,
                        interaction,
                        left_path,
                        right_path,
                        reasoning_effort,
                        retry_index,
                    )
                    break
                except ValueError as error:
                    last_error = error
                    print(f"    Empty/invalid reply from Ollama: {error}")
                    if retry_index == pair_retries:
                        raise

            if result is None:
                raise RuntimeError(
                    f"Failed to compare {left_path.name} vs {right_path.name}: {last_error}"
                )
            winner_path = left_path if result["choice"] == "left" else right_path
            print(
                f"    Winner: {winner_path.name} "
                f"(model chose {result['choice']})"
            )
            next_round.append(winner_path)
            pairs.append(result)

        rounds.append({"round": round_index, "pairs": pairs})
        current_round = next_round
        round_index += 1

    return current_round[0], rounds


def resolve_pag_path(script_dir: Path, video_name: str, raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).resolve()
    return next((script_dir.parent / "Generate_PAG" / "output" / video_name).glob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best generated first frame using a VLM tournament.",
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--pag", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--ollama_host", default=OLLAMA_HOST)
    parser.add_argument("--ollama_api_key", default=OLLAMA_API_KEY)
    parser.add_argument("--qwen_model", default=QWEN_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="high",
    )
    parser.add_argument("--pair-retries", type=int, default=3)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    video_dir = script_dir / "output" / args.video_name
    frames_dir = Path(args.frames_dir).resolve() if args.frames_dir else video_dir / "first_frames"
    pag_path = resolve_pag_path(script_dir, args.video_name, args.pag)
    prompt_path = (
        Path(args.prompt_file).resolve()
        if args.prompt_file
        else script_dir / "prompt_select_first_frame.md"
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
    client = OpenAI(base_url=args.ollama_host, api_key=args.ollama_api_key)
    reasoning_effort = args.reasoning_effort

    print(f"Video name: {args.video_name}")
    print(f"Frames directory: {frames_dir}")
    print(f"PAG file: {pag_path}")
    print(f"Prompt file: {prompt_path}")
    print(f"Model: {args.qwen_model}")
    print(f"Reasoning effort: {reasoning_effort}")
    print(f"Pair retries: {args.pair_retries}")
    print(f"Found {len(frame_paths)} candidate frames")

    if len(frame_paths) == 1:
        selected_frame = frame_paths[0]
        rounds: list[dict[str, Any]] = []
        print(f"Only one frame found, selecting {selected_frame.name}")
    else:
        selected_frame, rounds = run_tournament(
            client,
            args.qwen_model,
            system_prompt,
            interaction,
            frame_paths,
            reasoning_effort,
            args.pair_retries,
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "video_name": args.video_name,
        "frames_dir": "first_frames",
        "pag_path": str(pag_path),
        "prompt_file": str(prompt_path),
        "model": args.qwen_model,
        "selected_frame": selected_frame.name,
        "selected_frame_path": str(Path("first_frames") / selected_frame.name),
        "rounds": rounds,
    }
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Selected frame: {selected_frame}")
    print(f"Wrote: {output_json}")


if __name__ == "__main__":
    main()
