from __future__ import annotations

import argparse
import importlib.util
import mimetypes
import re
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "02_evaluate_static_scene.py"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"


def load_eval_base() -> Any:
    spec = importlib.util.spec_from_file_location("static_scene_eval", EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator helpers: {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_base = load_eval_base()
DEFAULT_API_KEY_FILE = eval_base.PROJECT_DIR / ".secrets" / "gemini_api_key"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a rendered static scene with deterministic metrics and Gemini VLM.",
    )
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--optimizer-output-root", default=None)
    parser.add_argument("--sig-json", default=None)
    parser.add_argument("--input-scene-json", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--contact-threshold-m", type=float, default=0.05)
    parser.add_argument("--severe-penetration-min-sdf-m", type=float, default=-0.03)
    parser.add_argument("--severe-penetration-inside-points", type=int, default=1000)
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--gemini-rate-limit-max-waits", type=int, default=5)
    parser.add_argument("--gemini-rate-limit-fallback-delay-sec", type=float, default=20.0)
    return parser.parse_args()


def read_gemini_api_key() -> str:
    if not DEFAULT_API_KEY_FILE.exists():
        raise FileNotFoundError(
            f"Gemini API key file not found: {DEFAULT_API_KEY_FILE}. "
            "Create it with a single API key line."
        )
    key = DEFAULT_API_KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"Gemini API key file is empty: {DEFAULT_API_KEY_FILE}")
    return key


def image_part(path: Path, types: Any) -> Any:
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    parts = getattr(response, "parts", None)
    if parts is None and getattr(response, "candidates", None):
        parts = response.candidates[0].content.parts
    if parts:
        chunks = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(str(part_text))
        if chunks:
            return "\n".join(chunks)
    return ""


def gemini_request(
    client: Any,
    types: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
) -> str:
    contents: list[Any] = [task_text]
    for path in image_paths:
        if path.exists():
            contents.append(image_part(path, types))
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    return extract_response_text(response)


def is_rate_limit_error(error: Exception) -> bool:
    text = str(error)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def retry_delay_seconds(error: Exception, fallback_delay_sec: float) -> float:
    text = str(error)
    match = re.search(r"'retryDelay': '([0-9.]+)s'", text)
    if not match:
        match = re.search(r'"retryDelay":\s*"([0-9.]+)s"', text)
    if not match:
        match = re.search(r"Please retry in ([0-9.]+)s", text)
    if match:
        return max(float(match.group(1)) + 1.0, 1.0)
    return max(float(fallback_delay_sec), 1.0)


def judge_with_gemini(
    judge_name: str,
    client: Any,
    types: Any,
    model: str,
    system_prompt: str,
    task_text: str,
    image_paths: list[Path],
    args: argparse.Namespace,
    edge_index: int | None = None,
) -> dict[str, Any]:
    attempt = 0
    rate_limit_waits = 0
    max_attempts = int(args.retries) + 1
    max_rate_limit_waits = int(args.gemini_rate_limit_max_waits)
    while attempt < max_attempts:
        try:
            suffix = f" edge={edge_index}" if edge_index is not None else ""
            eval_base.log("vlm", f"start Gemini judge={judge_name}{suffix} attempt={attempt + 1}")
            text = gemini_request(client, types, model, system_prompt, task_text, image_paths)
            payload = eval_base.extract_json_object(text)
            judgment = eval_base.normalize_judgment(payload, edge_index=edge_index)
            eval_base.log(
                "vlm",
                f"done Gemini judge={judge_name}{suffix} decision={judgment['decision']}",
            )
            return judgment
        except Exception as error:
            if is_rate_limit_error(error) and rate_limit_waits < max_rate_limit_waits:
                delay = retry_delay_seconds(
                    error,
                    float(args.gemini_rate_limit_fallback_delay_sec),
                )
                rate_limit_waits += 1
                eval_base.log(
                    "warn",
                    f"Gemini rate limited judge={judge_name}; waiting {delay:.1f}s "
                    f"before retrying same attempt "
                    f"({rate_limit_waits}/{max_rate_limit_waits})",
                )
                time.sleep(delay)
                continue
            eval_base.log("warn", f"Gemini judge={judge_name} failed attempt={attempt + 1}: {error}")
            attempt += 1
    return eval_base.normalize_judgment(
        {
            "decision": "no_decision",
            "reason": f"Gemini judge failed after retries: {judge_name}",
        },
        edge_index=edge_index,
    )


def run_gemini_vlm_judgments(rendered_evidence: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.skip_vlm:
        eval_base.log("vlm", "skipped because --skip-vlm was set")
        return {
            "enabled": False,
            "provider": "gemini",
            "model": None,
            "contact_edges": [],
            "pose": {},
            "penetration": {},
        }

    try:
        from google import genai
        from google.genai import types
    except Exception as error:
        raise RuntimeError("Could not import the Gemini SDK. Install it with: pip install google-genai") from error

    client = genai.Client(api_key=read_gemini_api_key())
    eval_base.log("vlm", f"enabled provider=gemini model={args.model} retries={args.retries}")
    contact_prompt = eval_base.read_prompt(eval_base.CONTACT_PROMPT, "Judge contact and return strict JSON.")
    pose_prompt = eval_base.read_prompt(eval_base.POSE_PROMPT, "Judge pose and return strict JSON.")
    penetration_prompt = eval_base.read_prompt(
        eval_base.PENETRATION_PROMPT,
        "Judge penetration and return strict JSON.",
    )

    pose = rendered_evidence.get("pose", {})
    interaction = str(pose.get("interaction", ""))
    contact_judgments = []
    for edge in rendered_evidence.get("contact_edges", []):
        edge_index = int(edge.get("edge_index", len(contact_judgments)))
        image_paths = [
            Path(view["images"][key])
            for view in edge.get("views", [])
            for key in ["context", "local_contact"]
            if view.get("images", {}).get(key)
        ]
        if not image_paths:
            judgment = {
                "decision": "no_decision",
                "pass": None,
                "reason": "No selected contact images were available for Gemini judging.",
            }
        else:
            judgment = judge_with_gemini(
                "contact",
                client,
                types,
                args.model,
                contact_prompt,
                eval_base.contact_task_text(edge, interaction),
                image_paths,
                args,
                edge_index=edge_index,
            )
        contact_judgments.append(
            {
                "edge_index": edge_index,
                "body_part": edge.get("body_part"),
                "target": edge.get("target"),
                **eval_base.slim_judgment(judgment),
            }
        )

    pose_images = [Path(path) for path in pose.get("images", {}).get("views", []) if path]
    pose_judgment = judge_with_gemini(
        "pose",
        client,
        types,
        args.model,
        pose_prompt,
        eval_base.pose_task_text(pose, interaction),
        pose_images,
        args,
    )
    penetration = rendered_evidence.get("penetration", {})
    penetration_images = [Path(path) for path in penetration.get("images", {}).get("views", []) if path]
    penetration_judgment = judge_with_gemini(
        "penetration",
        client,
        types,
        args.model,
        penetration_prompt,
        eval_base.penetration_task_text(penetration, interaction),
        penetration_images,
        args,
    )
    return {
        "enabled": True,
        "provider": "gemini",
        "model": args.model,
        "contact_edges": contact_judgments,
        "pose": eval_base.slim_judgment(pose_judgment),
        "penetration": eval_base.slim_judgment(penetration_judgment),
    }


def main() -> None:
    args = parse_args()
    metrics, _input_scene, _optimizer_root, outdir = eval_base.load_metrics_context(args)
    outdir.mkdir(parents=True, exist_ok=True)
    rendered_evidence = eval_base.require_rendered_evidence(metrics, outdir)
    eval_base.save_json(outdir / "metrics.json", eval_base.slim_metrics(metrics))
    eval_base.log(
        "metrics",
        f"contact_pass={metrics['contact']['pass']} penetration_pass={metrics['penetration']['pass']} "
        f"edges={metrics['contact']['edge_count']} deterministic_pass={metrics['deterministic']['pass']}",
    )
    vlm_judgments = run_gemini_vlm_judgments(rendered_evidence, args)
    eval_base.save_json(outdir / "vlm_judgments.json", vlm_judgments)
    summary = eval_base.build_human_verification_summary(metrics, vlm_judgments)
    eval_base.save_json(outdir / "verification_summary.json", summary)
    eval_base.log("summary", f"status={summary['status']} failure_tags={summary['failure_tags']}")
    eval_base.log("summary", f"wrote metrics: {outdir / 'metrics.json'}")
    eval_base.log("summary", f"wrote VLM judgments: {outdir / 'vlm_judgments.json'}")
    eval_base.log("summary", f"wrote verification summary: {outdir / 'verification_summary.json'}")


def run_cli() -> None:
    try:
        main()
    except Exception as error:
        raise SystemExit(f"[error] {error}") from None


if __name__ == "__main__":
    run_cli()
