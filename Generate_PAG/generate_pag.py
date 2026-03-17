from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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


def clean_text(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_name(value: Any) -> str:
    return clean_text(value).lower()


def split_node(node: Any) -> tuple[str, str]:
    entity, part = str(node).split(",", 1)
    return clean_text(entity), clean_text(part)


def normalize_node(node: Any) -> str:
    entity, part = split_node(node)
    return f"{entity.lower()}, {part.lower()}"


def display_node(node: Any) -> str:
    entity, part = split_node(node)
    return f"{entity}, {part}"


def normalize_edge(nodes: list[Any]) -> tuple[str, str]:
    return tuple(sorted((normalize_node(nodes[0]), normalize_node(nodes[1]))))


def vote_bool(values: list[bool], fallback: bool) -> bool:
    if not values:
        return fallback

    counts = Counter(values)
    top_count = max(counts.values())
    winners = {value for value, count in counts.items() if count == top_count}
    if fallback in winners:
        return fallback
    return counts.most_common(1)[0][0]


def save_sample(index: int, sample: dict[str, Any], samples_dir: Path) -> None:
    sample_path = samples_dir / f"sample_{index:02d}.json"
    sample_path.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


def vote_node_list(
    samples: list[dict[str, Any]],
    field_name: str,
    majority_threshold: int,
    required_nodes: dict[str, str] | None = None,
) -> list[str]:
    counts: Counter[str] = Counter()
    displays: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for sample in samples:
        seen: set[str] = set()
        for node in sample.get(field_name, []):
            normalized = normalize_node(node)
            if normalized in seen:
                continue
            seen.add(normalized)
            counts[normalized] += 1
            displays[normalized][display_node(node)] += 1

    selected: dict[str, str] = {}
    for normalized, count in counts.items():
        if count >= majority_threshold:
            selected[normalized] = displays[normalized].most_common(1)[0][0]

    if required_nodes:
        selected.update(required_nodes)

    ordered: list[str] = []
    seen_ordered: set[str] = set()
    for node in samples[0].get(field_name, []):
        normalized = normalize_node(node)
        if normalized in selected and normalized not in seen_ordered:
            ordered.append(selected[normalized])
            seen_ordered.add(normalized)

    for normalized in sorted(selected, key=lambda key: selected[key].lower()):
        if normalized not in seen_ordered:
            ordered.append(selected[normalized])
            seen_ordered.add(normalized)

    return ordered


def vote_edges(
    samples: list[dict[str, Any]],
    majority_threshold: int,
) -> list[dict[str, Any]]:
    base_edges = samples[0].get("interaction edges", [])
    base_order: dict[tuple[str, str], int] = {}
    base_display: dict[tuple[str, str], tuple[str, str]] = {}

    for index, edge in enumerate(base_edges):
        key = normalize_edge(edge["nodes"])
        base_order[key] = index
        base_display[key] = (
            display_node(edge["nodes"][0]),
            display_node(edge["nodes"][1]),
        )

    counts: Counter[tuple[str, str]] = Counter()
    displays = defaultdict(Counter)
    continuous_votes = defaultdict(list)
    static_votes = defaultdict(list)

    for sample in samples:
        seen: set[tuple[str, str]] = set()
        for edge in sample.get("interaction edges", []):
            key = normalize_edge(edge["nodes"])
            if key in seen:
                continue
            seen.add(key)
            counts[key] += 1
            displays[key][
                (display_node(edge["nodes"][0]), display_node(edge["nodes"][1]))
            ] += 1
            continuous_votes[key].append(bool(edge.get("is_continuous", True)))
            static_votes[key].append(bool(edge.get("is_rel_static", False)))

    voted_edges: list[dict[str, Any]] = []
    for key, count in counts.items():
        if count < majority_threshold:
            continue

        base_edge = next(
            (
                edge
                for edge in base_edges
                if normalize_edge(edge["nodes"]) == key
            ),
            None,
        )
        fallback_continuous = (
            bool(base_edge.get("is_continuous", True)) if base_edge else True
        )
        fallback_static = (
            bool(base_edge.get("is_rel_static", False)) if base_edge else False
        )
        node_pair = base_display.get(key, displays[key].most_common(1)[0][0])
        voted_edges.append(
            {
                "nodes": [node_pair[0], node_pair[1]],
                "is_continuous": vote_bool(
                    continuous_votes[key],
                    fallback_continuous,
                ),
                "is_rel_static": vote_bool(static_votes[key], fallback_static),
            }
        )

    voted_edges.sort(
        key=lambda edge: base_order.get(
            normalize_edge(edge["nodes"]),
            len(base_order),
        )
    )
    return voted_edges


def vote_object_states(
    samples: list[dict[str, Any]],
    expected_objects: list[str],
) -> list[dict[str, Any]]:
    base_states = {
        normalize_name(state["name"]): dict(state)
        for state in samples[0].get("object states", [])
        if state.get("name")
    }
    sample_states = [
        {
            normalize_name(state["name"]): state
            for state in sample.get("object states", [])
            if state.get("name")
        }
        for sample in samples
    ]

    voted_states: list[dict[str, Any]] = []
    for name in expected_objects:
        key = normalize_name(name)
        state = dict(base_states.get(key, {"name": name, "description": ""}))
        matches = [sample_state[key] for sample_state in sample_states if key in sample_state]
        state["is_translational"] = vote_bool(
            [bool(item.get("is_translational", True)) for item in matches],
            bool(state.get("is_translational", True)),
        )
        state["is_rotational"] = vote_bool(
            [bool(item.get("is_rotational", True)) for item in matches],
            bool(state.get("is_rotational", True)),
        )
        voted_states.append(state)

    return voted_states


def aggregate_samples(
    samples: list[dict[str, Any]],
    expected_objects: list[str],
    majority_threshold: int,
) -> dict[str, Any]:
    base = json.loads(json.dumps(samples[0]))
    voted_edges = vote_edges(samples, majority_threshold)

    required_object_nodes: dict[str, str] = {}
    required_body_nodes: dict[str, str] = {}
    for edge in voted_edges:
        for node in edge["nodes"]:
            normalized = normalize_node(node)
            entity, _ = split_node(node)
            if entity.lower().startswith("person"):
                required_body_nodes[normalized] = display_node(node)
            else:
                required_object_nodes[normalized] = display_node(node)

    base["object part nodes"] = vote_node_list(
        samples,
        "object part nodes",
        majority_threshold,
        required_object_nodes,
    )
    base["body part nodes"] = vote_node_list(
        samples,
        "body part nodes",
        majority_threshold,
        required_body_nodes,
    )
    base["interaction edges"] = voted_edges
    base["object states"] = vote_object_states(samples, expected_objects)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Part Affordance Graph (PAG) using Ollama.",
    )
    parser.add_argument("--host", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen3.5:27b")
    parser.add_argument("--system-prompt", default="./system_prompt_pag.md")
    parser.add_argument("--input-dir", default="./input_prompts/video_01")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--num-samples", type=int, default=5)
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

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")

    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    input_dir = Path(args.input_dir)
    user_payload = json.loads(
        (input_dir / "input_pag.json").read_text(encoding="utf-8")
    )
    expected_objects = [clean_text(name) for name in user_payload.get("objects", [])]
    majority_threshold = args.num_samples // 2 + 1
    reasoning_effort = None
    if args.reasoning_effort != "none":
        reasoning_effort = args.reasoning_effort

    client = OpenAI(base_url=args.host, api_key="ollama")

    output_dir = Path("./output") / input_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for index in range(1, args.num_samples + 1):
        sample = request_sample(
            client,
            args.model,
            system_prompt,
            user_payload,
            args.temperature,
            reasoning_effort,
        )
        save_sample(index, sample, samples_dir)
        samples.append(sample)
        print(f"Sample {index:02d}: parsed")

    final_pag = aggregate_samples(samples, expected_objects, majority_threshold)
    model_tag = args.model.replace(":", "_").replace("-", "_").replace(".", "_")
    out_path = output_dir / f"output_pag_{model_tag}.json"
    out_path.write_text(
        json.dumps(final_pag, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Aggregation mode: field-vote")
    print(f"Wrote: {out_path}")
    print(f"Saved samples under: {samples_dir}")


if __name__ == "__main__":
    main()
