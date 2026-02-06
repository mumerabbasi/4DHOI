from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def model_suffix(model: str) -> str:
    """Convert a model tag into a safe filename suffix."""
    return model.replace(":", "_").replace("-", "_").replace(".", "_")


def strip_json_fence(text: str) -> str:
    """Remove ``` / ```json fences if the model wraps JSON in a code block."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # drop closing fence
        text = "\n".join(lines).strip()
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Part Affordance Graph (PAG) using Ollama + DeepSeek-R1.",
    )
    parser.add_argument("--host", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="deepseek-r1:32b")
    parser.add_argument("--system-prompt", default="./system_prompt_pag.md")
    parser.add_argument("--input-dir", default="./pags/video_02")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    system_prompt = load_text(Path(args.system_prompt))

    input_dir = Path(args.input_dir)
    user_payload = load_json(input_dir / "input_pag.json")

    client = OpenAI(base_url=args.host, api_key="ollama")

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=args.temperature,
    )

    output_text = strip_json_fence(response.choices[0].message.content)
    output_obj = json.loads(output_text)

    out_path = input_dir / f"output_pag_{model_suffix(args.model)}.json"
    out_path.write_text(
        json.dumps(output_obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
