from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from openai import OpenAI


HUMAN_PARTS = (
    "left hand",
    "right hand",
    "left arm",
    "right arm",
    "left shoulder",
    "right shoulder",
    "left leg",
    "right leg",
    "left foot",
    "right foot",
    "head",
    "hips",
)
HUMAN_PART_VOCAB = set(HUMAN_PARTS)
SCENE_NODE_ORDER = ("target_object", "floor")
SCENE_NODES = set(SCENE_NODE_ORDER)
MAX_INTERACTION_WORDS = 150
IMAGE_SOURCE_TO_REL_PATHS: dict[str, tuple[str, str]] = {
    "dslr_resized_undistorted": (
        "dslr/resized_undistorted_images",
        "dslr/nerfstudio/transforms_undistorted.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def normalize_label(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").replace("-", " ").split())


def normalize_scene_element(text: str, target_labels: set[str] | None = None) -> str:
    raw = str(text).strip().lower()
    normalized = normalize_label(text)
    labels = target_labels or set()
    if raw == "target_object" or normalized in {"target object", "object"} or normalized in labels:
        return "target_object"
    return normalized


def count_words(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def parse_human_part_node(node: str) -> str:
    parts = str(node).split(",", 1)
    if len(parts) == 2:
        return normalize_label(parts[1])
    return normalize_label(node)


def format_human_part_node(human_part: str) -> str:
    return f"person 1, {human_part}"


def build_model_input(input_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "interaction": input_payload["interaction_context"]["interaction"],
    }


def resolve_scannet_root(script_dir: Path, raw_scannet_root: str | None) -> Path:
    if raw_scannet_root:
        return Path(raw_scannet_root).resolve()
    return (script_dir.parent.parent / "Scannet++" / "data").resolve()


def resolve_scene_image_path(scannet_root: Path, scene_context: dict[str, Any]) -> Path:
    scene_id = scene_context["scene_id"]
    camera = scene_context["camera"]
    camera_source = camera["source"]
    camera_name = camera["name"]
    if camera_source not in IMAGE_SOURCE_TO_REL_PATHS:
        raise ValueError(
            f"Unsupported camera.source '{camera_source}'. "
            f"Supported values: {sorted(IMAGE_SOURCE_TO_REL_PATHS)}"
        )
    image_rel, _ = IMAGE_SOURCE_TO_REL_PATHS[camera_source]
    return scannet_root / scene_id / image_rel / camera_name


def encode_image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_user_message_content(
    user_payload: dict[str, Any],
    scene_image_path: Path,
) -> list[dict[str, Any]]:
    text = (
        "Use the provided scene image and JSON request to generate the SIG.\n\n"
        f"JSON request:\n{json.dumps(user_payload, ensure_ascii=False, indent=2)}"
    )
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": encode_image_data_url(scene_image_path)},
        },
    ]


def request_sig(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    scene_image_path: Path,
    temperature: float,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_message_content(user_payload, scene_image_path),
            },
        ],
        "temperature": temperature,
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    text = strip_json_fence(response.choices[0].message.content)
    return json.loads(text)


def validate_sig(sig: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(sig, dict):
        raise TypeError("SIG must be a JSON object.")

    target_object = sig.get("target_object")
    if not isinstance(target_object, dict):
        raise ValueError("SIG must contain target_object.")

    label = str(target_object.get("label", "")).strip()
    if not label:
        raise ValueError("target_object.label must be non-empty.")

    raw_edges = sig.get("interaction_edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError("SIG must contain at least one interaction edge in interaction_edges.")

    clean_edges: list[dict[str, Any]] = []
    edge_human_parts: set[str] = set()
    scene_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        human_part = normalize_label(str(edge.get("human_part", "")))
        target_labels = {normalize_label(label)}
        scene_element = normalize_scene_element(str(edge.get("scene_element", "")), target_labels)
        if not human_part or not scene_element:
            continue
        if human_part not in HUMAN_PART_VOCAB:
            raise ValueError(
                f"Unsupported SIG human_part '{human_part}'. "
                f"Allowed parts: {sorted(HUMAN_PART_VOCAB)}"
            )
        if scene_element not in SCENE_NODES:
            raise ValueError(
                f"Unsupported SIG scene_element '{scene_element}'. "
                f"Allowed scene nodes: {sorted(SCENE_NODES)}"
            )
        dedup_key = (human_part, scene_element)
        if dedup_key in seen_edges:
            continue
        seen_edges.add(dedup_key)
        edge_human_parts.add(human_part)
        scene_nodes.add(scene_element)
        clean_edges.append(
            {
                "human_part": human_part,
                "scene_element": scene_element,
                "notes": str(edge.get("notes", "")).strip(),
            }
        )

    if not clean_edges:
        raise ValueError("SIG did not contain any usable interaction_edges.")

    raw_human_nodes = sig.get("human_part_nodes", [])
    if not isinstance(raw_human_nodes, list):
        raise ValueError("SIG human_part_nodes must be a list.")
    node_human_parts: set[str] = set()
    for node in raw_human_nodes:
        human_part = parse_human_part_node(str(node))
        if not human_part:
            continue
        if human_part not in HUMAN_PART_VOCAB:
            raise ValueError(
                f"Unsupported human_part_nodes entry '{node}'. "
                f"Allowed parts: {sorted(HUMAN_PART_VOCAB)}"
            )
        node_human_parts.add(human_part)
    node_human_parts.update(edge_human_parts)

    raw_scene_nodes = sig.get("scene_nodes", [])
    if not isinstance(raw_scene_nodes, list):
        raise ValueError("SIG scene_nodes must be a list.")
    for node in raw_scene_nodes:
        scene_node = normalize_scene_element(
            str(node),
            {normalize_label(label)},
        )
        if not scene_node:
            continue
        if scene_node not in SCENE_NODES:
            raise ValueError(
                f"Unsupported scene_nodes entry '{node}'. "
                f"Allowed nodes: {sorted(SCENE_NODES)}"
            )
        scene_nodes.add(scene_node)

    interaction = str(sig.get("interaction", "")).strip()
    if not interaction:
        raise ValueError("SIG must contain a non-empty interaction field.")
    if count_words(interaction) > MAX_INTERACTION_WORDS:
        raise ValueError(
            f"SIG interaction must be at most {MAX_INTERACTION_WORDS} words; "
            f"got {count_words(interaction)}."
        )

    sig["target_object"] = {"label": label}
    sig["human_part_nodes"] = [
        format_human_part_node(part_name)
        for part_name in HUMAN_PARTS
        if part_name in node_human_parts
    ]
    sig["scene_nodes"] = [
        node for node in SCENE_NODE_ORDER if node in scene_nodes
    ]
    sig["interaction_edges"] = clean_edges
    sig["interaction"] = interaction
    return sig


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate a Scene Interaction Graph for a static scene.")
    parser.add_argument("--interaction_name", default="interaction_01")
    parser.add_argument("--host", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.6:27b")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--scannet-root", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="none",
    )
    args = parser.parse_args()

    input_dir = (
        Path(args.input_dir).resolve()
        if args.input_dir
        else script_dir / "input_prompts" / args.interaction_name
    )
    output_root = Path(args.outdir).resolve() if args.outdir else script_dir / "output" / args.interaction_name
    system_prompt_path = (
        Path(args.system_prompt).resolve()
        if args.system_prompt
        else script_dir / "system_prompt_sig.md"
    )

    input_payload = load_json(input_dir / "input_scene.json")
    scannet_root = resolve_scannet_root(script_dir, args.scannet_root)
    scene_image_path = resolve_scene_image_path(
        scannet_root=scannet_root,
        scene_context=input_payload["scene_context"],
    )
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    model_input = build_model_input(input_payload)
    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    client = OpenAI(base_url=args.host, api_key="ollama")
    sig = request_sig(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        user_payload=model_input,
        scene_image_path=scene_image_path,
        temperature=args.temperature,
        reasoning_effort=reasoning_effort,
    )
    sig = validate_sig(sig)
    out_path = output_root / "scene_interaction_graph.json"
    save_json(out_path, sig)

    print(f"Input scene: {input_dir / 'input_scene.json'}")
    print(f"Scene image: {scene_image_path}")
    print(f"System prompt: {system_prompt_path}")
    print(f"Wrote SIG: {out_path}")


if __name__ == "__main__":
    main()
