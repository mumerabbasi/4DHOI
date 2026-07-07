from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_PROMPT_TEMPLATE_PATH = SCRIPT_DIR / "prompt_eval_interations.md"
OUTPUT_MODES = ("output", "output_round1", "output_init")
VLM_PROVIDERS = ("qwen", "gemini")
DEFAULT_QWEN_MODEL = "qwen3-vl:32b"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

CSV_FIELDNAMES = [
    "target_object_score",
    "human_action_score",
    "contact_score",
    "physical_plausibility_score",
    "mean_score",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt template is empty: {path}")
    return text


def render_prompt_template(template: str, interaction_prompt: str) -> str:
    placeholder = "{interaction}"
    if placeholder not in template:
        raise ValueError(f"Prompt template is missing placeholder: {placeholder}")
    return template.replace(placeholder, interaction_prompt.strip())


def read_provider_api_key(path: Path, provider_name: str) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"{provider_name} API key file not found: {path}. "
            "Create it with a single API key line."
        )
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"{provider_name} API key file is empty: {path}")
    return key


def build_default_paths(
    interaction_name: str,
    output_mode: str = "output",
) -> dict[str, Path]:
    interaction_output = SCRIPT_DIR / output_mode / interaction_name
    return {
        "input_scene_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        "render_root": interaction_output / "semantics",
        "output_root": interaction_output / "vlm",
    }


def discover_interactions(output_mode: str) -> list[str]:
    output_root = (
        PROJECT_DIR / "04_Estimate_Human_Pose" / "output"
        if output_mode == "output_init"
        else PROJECT_DIR / "05_Optimize_Static_Scene" / output_mode
    )
    names = [
        path.name
        for path in sorted(output_root.glob("interaction_*"))
        if (
            (path / "first_frame_smplx_world.ply").exists()
            if output_mode == "output_init"
            else (path / "meshes" / "frame_0000_world.ply").exists()
        )
    ]
    if not names:
        raise RuntimeError(f"No interactions found under {output_root}.")
    return names


def resolve_interaction_prompt(input_scene_json_path: Path) -> str:
    payload = load_json(input_scene_json_path)
    interaction_context = payload.get("interaction_context", {})
    if not isinstance(interaction_context, dict):
        raise ValueError(f"Missing interaction_context in {input_scene_json_path}.")
    prompt = str(interaction_context.get("interaction", "")).strip()
    if not prompt:
        raise ValueError(
            f"Missing interaction_context.interaction in {input_scene_json_path}."
        )
    return prompt


def collect_render_paths(render_root: Path) -> list[Path]:
    render_dir = render_root / "renders"
    paths = sorted(
        path
        for path in render_dir.glob("view_*.png")
        if path.stem.removeprefix("view_").isdigit()
    )
    if not paths:
        raise FileNotFoundError(f"No renders found under {render_dir}.")
    return paths


def encode_image_base64(path: Path, max_image_side: int) -> str:
    image = Image.open(path).convert("RGB")
    if max_image_side > 0:
        width, height = image.size
        scale = min(1.0, float(max_image_side) / float(max(width, height)))
        if scale < 1.0:
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def open_rgb_image(path: Path, max_image_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_image_side > 0:
        width, height = image.size
        scale = min(1.0, float(max_image_side) / float(max(width, height)))
        if scale < 1.0:
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def prepare_prompt_package(
    prompt_dir: Path,
    prompt: str,
    render_paths: list[Path],
    max_image_side: int,
) -> list[Path]:
    if prompt_dir.exists():
        shutil.rmtree(prompt_dir)
    render_prompt_dir = prompt_dir / "renders"
    render_prompt_dir.mkdir(parents=True, exist_ok=True)

    save_text(
        prompt_dir / "prompt.md",
        prompt,
    )

    packaged_render_paths = []
    for render_path in render_paths:
        image = open_rgb_image(render_path, max_image_side=max_image_side)
        packaged_path = render_prompt_dir / f"{render_path.stem}.png"
        image.save(packaged_path, format="PNG")
        packaged_render_paths.append(packaged_path)
    return packaged_render_paths

def ollama_chat(
    host: str,
    model: str,
    prompt: str,
    images_base64: list[str],
    timeout_s: int,
    temperature: float,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_base64,
            },
        ],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": float(temperature),
            "seed": int(seed),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
    content = response_payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Malformed Ollama response: {response_payload}")
    return content.strip()


def response_chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text
    parts = []
    for candidate in getattr(chunk, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "".join(parts)

def gemini_generate_json(
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    max_image_side: int,
    temperature: float,
    seed: int,
    max_output_tokens: int,
) -> str:
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The Google GenAI Python package is not installed. Install it in "
            "this environment, for example with: pip install google-genai"
        ) from exc

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=float(temperature),
        seed=int(seed),
        maxOutputTokens=int(max_output_tokens),
        responseMimeType="application/json",
    )
    contents = [prompt] + [
        open_rgb_image(path, max_image_side=max_image_side) for path in image_paths
    ]
    content_chunks: list[str] = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        text = response_chunk_text(chunk)
        if text:
            content_chunks.append(text)

    content = "".join(content_chunks)
    if not content.strip():
        raise RuntimeError("Gemini response did not contain text content.")
    return content.strip()


def effective_vlm_model(args: argparse.Namespace) -> str:
    if args.vlm_provider == "gemini":
        return str(args.gemini_model)
    return str(args.qwen_model)


def vlm_generate_json(
    args: argparse.Namespace,
    prompt: str,
    render_paths: list[Path],
) -> str:
    if args.vlm_provider == "qwen":
        images_base64 = [
            encode_image_base64(path, max_image_side=int(args.max_image_side))
            for path in render_paths
        ]
        return ollama_chat(
            host=args.ollama_host,
            model=str(args.qwen_model),
            prompt=prompt,
            images_base64=images_base64,
            timeout_s=int(args.timeout_s),
            temperature=float(args.temperature),
            seed=int(args.seed),
        )

    gemini_api_key = read_provider_api_key(
        Path(args.gemini_api_key_file).resolve(),
        "Gemini",
    )
    max_attempts = max(1, int(args.gemini_retries))
    retry_sleep_s = max(0.0, float(args.gemini_retry_sleep_s))
    last_error: Exception | None = None
    for attempt_index in range(1, max_attempts + 1):
        try:
            return gemini_generate_json(
                api_key=gemini_api_key,
                model=str(args.gemini_model),
                prompt=prompt,
                image_paths=render_paths,
                max_image_side=int(args.max_image_side),
                temperature=float(args.temperature),
                seed=int(args.seed),
                max_output_tokens=int(args.gemini_max_output_tokens),
            )
        except Exception as exc:
            last_error = exc
            if attempt_index >= max_attempts:
                break
            print(
                "Gemini VLM attempt failed: "
                f"{exc}. Retrying in {retry_sleep_s:.1f}s..."
            )
            time.sleep(retry_sleep_s)
    if last_error is None:
        raise RuntimeError("Gemini VLM evaluation did not run.")
    raise RuntimeError("Gemini VLM evaluation failed.") from last_error


def parse_json_response(raw_response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
        if match is None:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON must be an object.")
    return parsed


def score_from(result: dict[str, Any], *keys: str) -> int | None:
    value: Any = result
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, int | float):
        return int(value)
    return None


def flatten_result_row(
    result: dict[str, Any],
) -> dict[str, Any]:
    criteria = result.get("criteria", {})
    target_object_score = score_from(
        criteria, "target_object_correctness", "score_1_to_5"
    )
    human_action_score = score_from(
        criteria, "human_action_correctness", "score_1_to_5"
    )
    contact_score = score_from(
        criteria, "contact_and_spatial_relation", "score_1_to_5"
    )
    physical_plausibility_score = score_from(
        criteria, "physical_plausibility", "score_1_to_5"
    )
    criterion_scores = [
        score
        for score in (
            target_object_score,
            human_action_score,
            contact_score,
            physical_plausibility_score,
        )
        if isinstance(score, int | float)
    ]
    return {
        "target_object_score": target_object_score,
        "human_action_score": human_action_score,
        "contact_score": contact_score,
        "physical_plausibility_score": physical_plausibility_score,
        "mean_score": (
            float(sum(criterion_scores) / len(criterion_scores))
            if criterion_scores
            else None
        ),
    }


def evaluate_interaction_vlm(
    interaction_name: str,
    args: argparse.Namespace,
    prompt_template: str,
    prompt_template_path: Path,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name, args.output_mode)
    input_scene_json_path = resolve_path(
        args.input_scene_json,
        defaults["input_scene_json"],
    )
    render_root = resolve_path(args.render_root, defaults["render_root"])
    output_root = resolve_path(args.output_root, defaults["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    interaction_prompt = resolve_interaction_prompt(input_scene_json_path)
    render_paths = collect_render_paths(render_root)
    model = effective_vlm_model(args)
    prompt = render_prompt_template(prompt_template, interaction_prompt)
    prompt_dir = output_root / "prompt"
    prompt_render_paths = prepare_prompt_package(
        prompt_dir=prompt_dir,
        prompt=prompt,
        render_paths=render_paths,
        max_image_side=int(args.max_image_side),
    )
    raw_response = vlm_generate_json(
        args=args,
        prompt=prompt,
        render_paths=prompt_render_paths,
    )
    parsed_response = parse_json_response(raw_response)
    row = flatten_result_row(
        result=parsed_response,
    )
    save_json(
        output_root / "metrics.json",
        {
            "interaction_name": interaction_name,
            "interaction_instruction": interaction_prompt,
            "provider": args.vlm_provider,
            "model": model,
            "prompt_template_path": str(prompt_template_path),
            "prompt_dir": str(prompt_dir),
            "render_paths": [str(path) for path in render_paths],
            "prompt_render_paths": [str(path) for path in prompt_render_paths],
            "vlm_result": parsed_response,
            "raw_response": raw_response,
        },
    )
    save_csv_rows(output_root / "metrics.csv", [row], CSV_FIELDNAMES)
    print(
        f"{interaction_name}: {args.vlm_provider}_vlm_mean_score="
        f"{row['mean_score']} renders={len(render_paths)} model={model}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rendered interactions with a Qwen or Gemini VLM verifier."
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--output_mode",
        choices=OUTPUT_MODES,
        default="output",
        help=(
            "Choose the matching optimization/evaluation output set. "
            "'output' uses 05_Optimize_Static_Scene/output and reads/writes "
            "06_Evaluate_Interaction/output by default; 'output_round1' "
            "uses/reads/writes the output_round1 ablation folders; 'output_init' "
            "reads/writes the module-04 first-frame evaluation folder."
        ),
    )
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--vlm_provider",
        choices=VLM_PROVIDERS,
        default="gemini",
        help="VLM backend to use for scoring rendered views.",
    )
    parser.add_argument(
        "--qwen_model",
        type=str,
        default=DEFAULT_QWEN_MODEL,
        help="Ollama Qwen model used when --vlm_provider=qwen.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model used when --vlm_provider=gemini.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Legacy alias for --qwen_model.",
    )
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--gemini_api_key_file",
        type=str,
        default=str(PROJECT_DIR / ".secrets" / "gemini_api_key"),
    )
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Legacy alias for --prompt_template.",
    )
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--render_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--gemini_max_output_tokens", type=int, default=4096)
    parser.add_argument("--gemini_retries", type=int, default=3)
    parser.add_argument("--gemini_retry_sleep_s", type=float, default=8.0)
    args = parser.parse_args()
    if args.model is not None:
        args.qwen_model = args.model
    return args


def main() -> None:
    args = parse_args()
    prompt_template_override = (
        args.prompt_template
        if args.prompt_template is not None
        else args.system_prompt
    )
    prompt_template_path = resolve_path(
        prompt_template_override,
        DEFAULT_PROMPT_TEMPLATE_PATH,
    )
    prompt_template = load_text(prompt_template_path)
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if any(
            value is not None
            for value in (
                args.input_scene_json,
                args.render_root,
                args.output_root,
            )
        ):
            raise ValueError(
                "--all_interactions cannot be combined with per-interaction "
                "input/render/output overrides."
            )
        interaction_names = discover_interactions(args.output_mode)
    else:
        interaction_names = [args.interaction_name]

    rows = [
        evaluate_interaction_vlm(
            interaction_name=interaction_name,
            args=args,
            prompt_template=prompt_template,
            prompt_template_path=prompt_template_path,
        )
        for interaction_name in interaction_names
    ]

    if all_mode:
        output_root = SCRIPT_DIR / args.output_mode
        save_csv_rows(output_root / "vlm.csv", rows, CSV_FIELDNAMES)
        model = effective_vlm_model(args)
        mean_scores = [
            float(row["mean_score"])
            for row in rows
            if isinstance(row.get("mean_score"), int | float)
        ]
        save_json(
            output_root / "vlm.json",
            {
                "provider": args.vlm_provider,
                "model": model,
                "num_interactions": len(rows),
                "mean_score": (
                    float(sum(mean_scores) / len(mean_scores))
                    if mean_scores
                    else None
                ),
                "interactions": rows,
            },
        )


if __name__ == "__main__":
    main()
