from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def request_sample(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    text = strip_json_fence(response.choices[0].message.content)
    return json.loads(text)


def build_model_input(user_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "objects": user_payload["interaction_context"]["objects"],
        "interaction": user_payload["interaction_context"]["interaction"],
    }


def build_output_pag(
    generated_pag: dict[str, Any],
    user_payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(generated_pag, dict):
        raise TypeError("Expected generated PAG to be a JSON object.")

    return {
        "scene_context": user_payload["scene_context"],
        "pag": generated_pag,
    }


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a Part Affordance Graph (PAG) using Ollama.",
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--host", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:27b")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="high",
        help=(
            "Reasoning control for Ollama's OpenAI-compatible endpoint. "
            "Use 'none' to omit the field."
        ),
    )
    args = parser.parse_args()

    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_pag.md"
    )
    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else script_dir / "input_prompts" / args.video_name
    )
    output_dir = script_dir / "output" / args.video_name

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    user_payload = json.loads(
        (input_dir / "input_pag.json").read_text(encoding="utf-8")
    )
    model_input = build_model_input(user_payload)

    reasoning_effort = None
    if args.reasoning_effort != "none":
        reasoning_effort = args.reasoning_effort

    client = OpenAI(base_url=args.host, api_key="ollama")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video name: {args.video_name}")
    print(f"System prompt: {system_prompt_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    generated_pag = request_sample(
        client,
        args.model,
        system_prompt,
        model_input,
        args.temperature,
        reasoning_effort,
    )
    final_pag = build_output_pag(generated_pag, user_payload)
    model_tag = args.model.replace(":", "_").replace("-", "_").replace(".", "_")
    out_path = output_dir / f"output_pag_{model_tag}.json"
    out_path.write_text(
        json.dumps(final_pag, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Generated one PAG from a single model response.")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
