from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from common import load_json, read_api_key, resize_cover_center_crop


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Select the best generated human frame with Gemini.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--selector-prompt", default=None)
    parser.add_argument(
        "--api-key-file",
        default=str(project_dir / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--expected-candidates", type=int, default=5)
    parser.add_argument("--gemini-retries", type=int, default=3)
    parser.add_argument("--gemini-retry-sleep-s", type=float, default=8.0)
    parser.add_argument("--overwrite-inpainted", action="store_true")
    parser.set_defaults(script_dir=script_dir, project_dir=project_dir)
    return parser.parse_args(argv)


def response_chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    parts: list[str] = []
    for candidate in getattr(chunk, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "".join(parts)


def save_gemini_artifact(
    artifact_path: Path,
    prompt: str,
    raw_response: str,
) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        "PROMPT\n"
        "======\n"
        f"{prompt.rstrip()}\n\n"
        "RAW RESPONSE\n"
        "============\n"
        f"{raw_response.rstrip()}\n",
        encoding="utf-8",
    )


def open_rgb_image(path: Path) -> Any:
    from PIL import Image

    return Image.open(path).convert("RGB")


def gemini_generate_json(
    api_key: str,
    model: str,
    user_prompt: str,
    image_paths: list[Path],
    temperature: float,
    seed: int,
    max_output_tokens: int,
    artifact_path: Path,
) -> str:
    from google import genai
    from google.genai import types

    content_chunks: list[str] = []

    def write_accumulated() -> None:
        save_gemini_artifact(
            artifact_path=artifact_path,
            prompt=user_prompt,
            raw_response="".join(content_chunks),
        )

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        seed=int(seed),
        maxOutputTokens=int(max_output_tokens),
        responseMimeType="application/json",
    )
    contents = [user_prompt] + [open_rgb_image(path) for path in image_paths]
    write_accumulated()
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        text = response_chunk_text(chunk)
        if text:
            content_chunks.append(text)
            write_accumulated()

    content = "".join(content_chunks).strip()
    if not content:
        raise RuntimeError(
            "Gemini response did not contain text content. Artifact: "
            f"{artifact_path}"
        )
    return content


def parse_json_response(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Gemini selection response must be a JSON object.")
    return parsed


def find_candidate_frames(frames_dir: Path, expected_count: int) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Candidate human_frames directory not found: {frames_dir}"
        )
    candidates = sorted(frames_dir.glob("frame_*.png"))
    if len(candidates) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} candidate frames in {frames_dir}, "
            f"found {len(candidates)}."
        )
    return candidates


def build_selection_prompt(
    template: str,
    interaction: str,
    candidate_paths: list[Path],
) -> str:
    candidate_listing = "\n".join(
        f"Image {index}: {path.name}" for index, path in enumerate(candidate_paths, 1)
    )
    return (
        template.replace("{candidate_listing}", candidate_listing)
        .replace("{interaction}", interaction.strip())
        .strip()
    )


def select_frame_with_retries(
    api_key: str,
    model: str,
    prompt: str,
    candidate_paths: list[Path],
    args: argparse.Namespace,
    artifact_path: Path,
) -> dict[str, Any]:
    max_attempts = max(1, int(args.gemini_retries))
    retry_sleep_s = max(0.0, float(args.gemini_retry_sleep_s))
    last_error: Exception | None = None
    for attempt_index in range(1, max_attempts + 1):
        try:
            raw_response = gemini_generate_json(
                api_key=api_key,
                model=model,
                user_prompt=prompt,
                image_paths=candidate_paths,
                temperature=args.temperature,
                seed=args.seed,
                max_output_tokens=args.max_output_tokens,
                artifact_path=artifact_path,
            )
            return parse_json_response(raw_response)
        except Exception as exc:
            last_error = exc
            if attempt_index >= max_attempts:
                break
            print(
                f"Gemini selection attempt {attempt_index}/{max_attempts} failed: {exc}"
            )
            print(f"Retrying Gemini selection in {retry_sleep_s:.1f}s...")
            time.sleep(retry_sleep_s)
    assert last_error is not None
    raise last_error


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    script_dir: Path = args.script_dir
    project_dir: Path = args.project_dir

    output_root = (
        Path(args.outdir).resolve()
        if args.outdir
        else script_dir / "output" / args.interaction_name
    )
    sig_json_path = (
        Path(args.sig_json).resolve()
        if args.sig_json
        else project_dir
        / "01_Generate_SIG"
        / "output"
        / args.interaction_name
        / "sig.json"
    )
    frames_dir = (
        Path(args.frames_dir).resolve()
        if args.frames_dir
        else output_root / "human_frames"
    )
    selector_prompt_path = (
        Path(args.selector_prompt).resolve()
        if args.selector_prompt
        else script_dir / "prompt_select_best_human_frame.md"
    )
    inpainted_path = output_root / "inpainted_frame.png"
    inpainted_resized_path = output_root / "inpainted_frame_resized.png"
    selection_json_path = output_root / "selected_human_frame.json"
    artifact_path = output_root / "gemini_select_best_human_frame.txt"

    output_root.mkdir(parents=True, exist_ok=True)
    if inpainted_path.exists() and not args.overwrite_inpainted:
        print(
            "Skipping Gemini selection; found existing inpainted frame: "
            f"{inpainted_path}"
        )
        return inpainted_path

    sig_payload = load_json(sig_json_path)
    interaction = str(sig_payload["interaction"])
    prompt_template = selector_prompt_path.read_text(encoding="utf-8")
    candidate_paths = find_candidate_frames(frames_dir, args.expected_candidates)
    prompt = build_selection_prompt(prompt_template, interaction, candidate_paths)

    api_key = read_api_key(Path(args.api_key_file).resolve())
    selection = select_frame_with_retries(
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        candidate_paths=candidate_paths,
        args=args,
        artifact_path=artifact_path,
    )

    candidate_by_name = {path.name: path for path in candidate_paths}
    selected_frame = selection.get("selected_frame")
    if selected_frame not in candidate_by_name:
        valid_names = ", ".join(candidate_by_name)
        raise ValueError(
            f"Gemini selected invalid frame {selected_frame!r}. "
            f"Expected one of: {valid_names}"
        )

    selected_path = candidate_by_name[str(selected_frame)]
    shutil.copy2(selected_path, inpainted_path)

    scene_image_path = output_root / "prompt" / "scene_image.png"
    if scene_image_path.exists():
        scene_image = open_rgb_image(scene_image_path)
        selected_image = open_rgb_image(selected_path)
        resize_cover_center_crop(selected_image, scene_image.size).save(
            inpainted_resized_path
        )

    selection_json_path.write_text(
        json.dumps(
            {
                "selected_frame": selected_frame,
                "reason": str(selection.get("reason", "")).strip(),
                "selected_path": str(selected_path),
                "model": args.model,
                "candidates": [path.name for path in candidate_paths],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Selected human frame: {selected_frame}")
    print(f"Wrote selection artifact: {selection_json_path}")
    print(f"Wrote inpainted frame: {inpainted_path}")
    if inpainted_resized_path.exists():
        print(f"Wrote resized inpainted frame: {inpainted_resized_path}")
    return inpainted_path


if __name__ == "__main__":
    main()
