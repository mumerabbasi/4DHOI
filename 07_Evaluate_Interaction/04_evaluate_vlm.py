from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

SYSTEM_PROMPT = """### ROLE
You are a 3D Human-Scene Interaction quality assurance agent.

### INPUT DATA
You will receive:

1. Interaction Instruction:
A text description of the intended human-scene interaction.

2. Rendered Views:
Multiple rendered images of the same static 3D human-scene interaction from different viewpoints.

3. Quantitative Metrics:
Contact, collision, semantic, and render-visibility metrics computed by separate evaluation scripts.

### TASK
Evaluate whether the rendered 3D human-scene interaction satisfies the interaction instruction.

Use the rendered views as the primary evidence.
Use the quantitative metrics as supporting evidence.
Judge the interaction itself, not general render aesthetics.

### EVALUATION LOGIC
1. Identify the target object or scene element described in the interaction instruction.
2. Check whether the target object or scene element is visible and matches the instruction.
3. Check whether the human pose expresses the requested action.
4. Check whether the relevant human body parts are spatially close to or contacting the correct object or floor region.
5. Check whether the human-scene relation is physically plausible.
6. Check whether the rendered views provide enough evidence to judge the interaction.
7. Compare the visual evidence with the quantitative metrics.
8. Assign a 1 to 5 score for each criterion and for the overall interaction.

### CRITERIA
Use this 1 to 5 scale for every score:

1 = Incorrect, missing, or impossible to judge.
2 = Mostly incorrect or very unclear.
3 = Partially correct, but ambiguous or incomplete.
4 = Mostly correct with minor issues.
5 = Clearly correct and physically plausible.

Score these criteria:

1. Target Object Correctness:
Is the correct object or scene element visible and identifiable?

2. Human Action Correctness:
Does the human pose match the requested action?

3. Contact and Spatial Relation:
Are the relevant body parts plausibly interacting with the correct scene element?

4. Physical Plausibility:
Is the pose and body-object relation plausible, without severe floating, penetration, impossible support, or nonsensical placement?

5. Visibility and Evidence:
Do the views provide enough visual evidence to judge the interaction?

6. Metric Consistency:
Are the quantitative metrics consistent with the visual evidence?

7. Overall:
How well does the final interaction satisfy the instruction?

### OUTPUT FORMAT
Return ONLY a valid JSON object. Do not include markdown or extra text.

Use this exact schema:

{
  "overall": {
    "score_1_to_5": 0,
    "reason": ""
  },
  "criteria": {
    "target_object_correctness": {
      "score_1_to_5": 0,
      "reason": ""
    },
    "human_action_correctness": {
      "score_1_to_5": 0,
      "reason": ""
    },
    "contact_and_spatial_relation": {
      "score_1_to_5": 0,
      "reason": ""
    },
    "physical_plausibility": {
      "score_1_to_5": 0,
      "reason": ""
    },
    "visibility_and_evidence": {
      "score_1_to_5": 0,
      "reason": ""
    },
    "metric_consistency": {
      "score_1_to_5": 0,
      "reason": ""
    }
  },
  "best_view_ids": [],
  "failure_modes": [],
  "brief_summary": ""
}
"""

CSV_FIELDNAMES = [
    "interaction_name",
    "overall_score",
    "target_object_score",
    "human_action_score",
    "contact_score",
    "physical_plausibility_score",
    "visibility_score",
    "metric_consistency_score",
    "num_renders",
    "best_view_ids",
    "failure_modes",
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


def save_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def build_default_paths(interaction_name: str) -> dict[str, Path]:
    interaction_output = SCRIPT_DIR / "output" / interaction_name
    return {
        "input_scene_json": PROJECT_DIR
        / "01_Generate_SIG"
        / "input_prompts"
        / interaction_name
        / "input_scene.json",
        "render_root": interaction_output / "semantics",
        "physical_metrics_json": interaction_output
        / "physical_plausibility"
        / "metrics.json",
        "semantic_metrics_json": interaction_output / "semantics" / "metrics.json",
        "selected_views_json": interaction_output
        / "semantics"
        / "assets"
        / "selected_views.json",
        "output_root": interaction_output / "vlm",
    }


def discover_interactions() -> list[str]:
    output_root = PROJECT_DIR / "06_Optimize_Static_Scene" / "output"
    names = [
        path.name
        for path in sorted(output_root.glob("interaction_*"))
        if (path / "meshes" / "frame_0000_world.ply").exists()
    ]
    if not names:
        raise RuntimeError(f"No optimized interactions found under {output_root}.")
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


def view_id_from_path(path: Path) -> str:
    return path.stem


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


def load_optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    return load_json(path)


def summarize_physical_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, list):
        return {"available": False, "reason": "missing or malformed physical metrics"}
    contact_edges = []
    ncs_values = []
    mean_pen_values = []
    max_pen_values = []
    for row in metrics:
        if not isinstance(row, dict):
            continue
        contact = row.get("contact", {})
        collision = row.get("collision", {})
        if isinstance(collision, dict):
            for key, target in (
                ("ncs", ncs_values),
                ("mean_penetration_m", mean_pen_values),
                ("max_penetration_m", max_pen_values),
            ):
                value = collision.get(key)
                if isinstance(value, int | float):
                    target.append(float(value))
        contact_edges.append(
            {
                "node_a": row.get("node_a"),
                "node_b": row.get("node_b"),
                "min_distance_m": contact.get("min_distance_m")
                if isinstance(contact, dict)
                else None,
                "max_distance_m": contact.get("max_distance_m")
                if isinstance(contact, dict)
                else None,
                "mean_distance_m": contact.get("mean_distance_m")
                if isinstance(contact, dict)
                else None,
            }
        )

    def mean_or_none(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    return {
        "available": True,
        "contact_edges": contact_edges,
        "mean_contact_distance_m": mean_or_none(
            [
                float(edge["mean_distance_m"])
                for edge in contact_edges
                if isinstance(edge.get("mean_distance_m"), int | float)
            ]
        ),
        "ncs": mean_or_none(ncs_values),
        "mean_penetration_m": mean_or_none(mean_pen_values),
        "max_penetration_m": max(max_pen_values) if max_pen_values else None,
    }


def summarize_semantic_metrics(metrics: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"available": False, "reason": "missing or malformed semantic metrics"}
    renders = []
    for item in metrics.get("renders", []):
        if not isinstance(item, dict):
            continue
        render_path = item.get("render_path", "")
        renders.append(
            {
                "view_id": Path(str(render_path)).stem,
                "clip_score": item.get("clip_score"),
            }
        )
    return {
        "available": True,
        "mean_clip_score": metrics.get("clip_score"),
        "per_view_clip_scores": renders,
    }


def summarize_selected_views(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"available": False, "reason": "missing or malformed selected views"}
    views = []
    for view in payload.get("views", []):
        if not isinstance(view, dict):
            continue
        views.append(
            {
                "view_id": view.get("name"),
                "human_visible_fraction": view.get("human_visible_fraction"),
                "interaction_part_visible_fraction": view.get(
                    "interaction_part_visible_fraction"
                ),
                "valid": view.get("valid"),
            }
        )
    return {
        "available": True,
        "interaction_parts": payload.get("interaction_parts"),
        "views": views,
    }


def build_user_prompt(
    interaction_name: str,
    interaction_prompt: str,
    render_paths: list[Path],
    quantitative_metrics: dict[str, Any],
) -> str:
    view_ids = [view_id_from_path(path) for path in render_paths]
    payload = {
        "interaction_name": interaction_name,
        "interaction_instruction": interaction_prompt,
        "rendered_view_ids_in_image_order": view_ids,
        "quantitative_metrics": quantitative_metrics,
    }
    return (
        "Evaluate the following 3D human-scene interaction.\n\n"
        "The attached images are the rendered views in the exact order listed "
        "in rendered_view_ids_in_image_order.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def ollama_chat(
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images_base64: list[str],
    timeout_s: int,
    temperature: float,
    seed: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
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
    interaction_name: str,
    result: dict[str, Any],
    num_renders: int,
) -> dict[str, Any]:
    criteria = result.get("criteria", {})
    best_view_ids = result.get("best_view_ids", [])
    failure_modes = result.get("failure_modes", [])
    return {
        "interaction_name": interaction_name,
        "overall_score": score_from(result, "overall", "score_1_to_5"),
        "target_object_score": score_from(
            criteria, "target_object_correctness", "score_1_to_5"
        ),
        "human_action_score": score_from(
            criteria, "human_action_correctness", "score_1_to_5"
        ),
        "contact_score": score_from(
            criteria, "contact_and_spatial_relation", "score_1_to_5"
        ),
        "physical_plausibility_score": score_from(
            criteria, "physical_plausibility", "score_1_to_5"
        ),
        "visibility_score": score_from(
            criteria, "visibility_and_evidence", "score_1_to_5"
        ),
        "metric_consistency_score": score_from(
            criteria, "metric_consistency", "score_1_to_5"
        ),
        "num_renders": int(num_renders),
        "best_view_ids": ",".join(map(str, best_view_ids))
        if isinstance(best_view_ids, list)
        else str(best_view_ids),
        "failure_modes": "; ".join(map(str, failure_modes))
        if isinstance(failure_modes, list)
        else str(failure_modes),
    }


def evaluate_interaction_vlm(
    interaction_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    defaults = build_default_paths(interaction_name)
    input_scene_json_path = resolve_path(
        args.input_scene_json,
        defaults["input_scene_json"],
    )
    render_root = resolve_path(args.render_root, defaults["render_root"])
    physical_metrics_json = resolve_path(
        args.physical_metrics_json,
        defaults["physical_metrics_json"],
    )
    semantic_metrics_json = resolve_path(
        args.semantic_metrics_json,
        defaults["semantic_metrics_json"],
    )
    selected_views_json = resolve_path(
        args.selected_views_json,
        defaults["selected_views_json"],
    )
    output_root = resolve_path(args.output_root, defaults["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    interaction_prompt = resolve_interaction_prompt(input_scene_json_path)
    render_paths = collect_render_paths(render_root)
    quantitative_metrics = {
        "physical_plausibility": summarize_physical_metrics(
            load_optional_json(physical_metrics_json)
        ),
        "clip_semantics": summarize_semantic_metrics(
            load_optional_json(semantic_metrics_json)
        ),
        "render_visibility": summarize_selected_views(
            load_optional_json(selected_views_json)
        ),
    }
    user_prompt = build_user_prompt(
        interaction_name=interaction_name,
        interaction_prompt=interaction_prompt,
        render_paths=render_paths,
        quantitative_metrics=quantitative_metrics,
    )
    images_base64 = [
        encode_image_base64(path, max_image_side=int(args.max_image_side))
        for path in render_paths
    ]
    raw_response = ollama_chat(
        host=args.ollama_host,
        model=args.model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        images_base64=images_base64,
        timeout_s=int(args.timeout_s),
        temperature=float(args.temperature),
        seed=int(args.seed),
    )
    parsed_response = parse_json_response(raw_response)
    row = flatten_result_row(
        interaction_name=interaction_name,
        result=parsed_response,
        num_renders=len(render_paths),
    )
    save_json(
        output_root / "metrics.json",
        {
            "interaction_name": interaction_name,
            "interaction_instruction": interaction_prompt,
            "model": args.model,
            "render_paths": [str(path) for path in render_paths],
            "quantitative_metrics": quantitative_metrics,
            "vlm_result": parsed_response,
            "raw_response": raw_response,
        },
    )
    save_csv_rows(output_root / "metrics.csv", [row], CSV_FIELDNAMES)
    print(
        f"{interaction_name}: vlm_overall_score="
        f"{row['overall_score']} renders={len(render_paths)}"
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rendered interactions with an Ollama VLM verifier."
    )
    parser.add_argument("--interaction_name", type=str, default="interaction_01")
    parser.add_argument(
        "--all_interactions",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--model", type=str, default="qwen3.6:27b")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434")
    parser.add_argument("--input_scene_json", type=str, default=None)
    parser.add_argument("--render_root", type=str, default=None)
    parser.add_argument("--physical_metrics_json", type=str, default=None)
    parser.add_argument("--semantic_metrics_json", type=str, default=None)
    parser.add_argument("--selected_views_json", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--max_image_side", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout_s", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_mode = bool(args.all_interactions) or args.interaction_name == "all"
    if all_mode:
        if any(
            value is not None
            for value in (
                args.input_scene_json,
                args.render_root,
                args.physical_metrics_json,
                args.semantic_metrics_json,
                args.selected_views_json,
                args.output_root,
            )
        ):
            raise ValueError(
                "--all_interactions cannot be combined with per-interaction "
                "input/render/output overrides."
            )
        interaction_names = discover_interactions()
    else:
        interaction_names = [args.interaction_name]

    rows = [
        evaluate_interaction_vlm(interaction_name=interaction_name, args=args)
        for interaction_name in interaction_names
    ]

    if all_mode:
        output_root = SCRIPT_DIR / "output"
        save_csv_rows(output_root / "vlm.csv", rows, CSV_FIELDNAMES)
        overall_scores = [
            float(row["overall_score"])
            for row in rows
            if isinstance(row.get("overall_score"), int | float)
        ]
        save_json(
            output_root / "vlm.json",
            {
                "model": args.model,
                "num_interactions": len(rows),
                "mean_overall_score": (
                    float(sum(overall_scores) / len(overall_scores))
                    if overall_scores
                    else None
                ),
                "interactions": rows,
            },
        )


if __name__ == "__main__":
    main()
