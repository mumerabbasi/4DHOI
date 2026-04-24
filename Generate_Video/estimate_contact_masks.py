from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BODY_PART_COLOR_MAP: dict[str, tuple[str, str]] = {
    "left hand": ("#FF00FF", "pure magenta"),
    "right hand": ("#00FF00", "pure lime green"),
    "left arm": ("#00FFFF", "bright cyan"),
    "right arm": ("#FF8C00", "bright orange"),
    "left shoulder": ("#FFD700", "bright gold"),
    "right shoulder": ("#1E90FF", "bright dodger blue"),
    "left leg": ("#FF1493", "bright deep pink"),
    "right leg": ("#7FFF00", "bright chartreuse"),
    "left foot": ("#00BFFF", "bright sky blue"),
    "right foot": ("#FF4500", "bright orange red"),
    "head": ("#FFFF00", "pure yellow"),
    "hips": ("#9400D3", "bright violet"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def resolve_pag_path(script_dir: Path, video_name: str, raw_pag: str | None) -> Path:
    if raw_pag:
        return Path(raw_pag).resolve()

    pag_dir = (script_dir.parent / "Generate_PAG" / "output" / video_name).resolve()
    candidates = sorted(pag_dir.glob("output_pag_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No output_pag_*.json found in: {pag_dir}")
    return candidates[0]


def parse_node(node_value: str) -> tuple[str, str]:
    parts = node_value.split(",", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse PAG node: '{node_value}'")
    return parts[0].strip(), parts[1].strip()


def is_human_entity(entity_name: str) -> bool:
    return entity_name.lower().startswith("person")


def normalize_part_name(part_name: str) -> str:
    return " ".join(part_name.strip().lower().split())


def color_for_part(part_name: str) -> tuple[str, str]:
    normalized = normalize_part_name(part_name)
    if normalized not in BODY_PART_COLOR_MAP:
        raise KeyError(
            f"Unsupported human body part in PAG interaction edge: '{part_name}'"
        )
    return BODY_PART_COLOR_MAP[normalized]


def collect_contact_entries(pag_payload: dict[str, Any]) -> list[dict[str, Any]]:
    edges = pag_payload.get("interaction edges", [])
    if not isinstance(edges, list):
        raise ValueError("PAG field 'interaction edges' must be a list.")

    node_color_map: dict[str, tuple[str, str]] = {}
    entries: list[dict[str, Any]] = []

    for edge_idx, edge_payload in enumerate(edges):
        if not isinstance(edge_payload, dict):
            continue
        node_values = edge_payload.get("nodes", [])
        if not isinstance(node_values, list) or len(node_values) != 2:
            continue

        parsed_nodes: list[dict[str, str]] = []
        for raw_node in node_values:
            node_str = str(raw_node)
            entity_name, part_name = parse_node(node_str)
            parsed_nodes.append(
                {
                    "raw_node": node_str,
                    "entity_name": entity_name,
                    "part_name": part_name,
                }
            )

        human_nodes = [node for node in parsed_nodes if is_human_entity(node["entity_name"])]
        object_nodes = [node for node in parsed_nodes if not is_human_entity(node["entity_name"])]
        if len(human_nodes) != 1 or len(object_nodes) != 1:
            continue

        human_node = human_nodes[0]
        object_node = object_nodes[0]
        node_key = human_node["raw_node"]

        if node_key not in node_color_map:
            hex_code, color_name = color_for_part(part_name=human_node["part_name"])
            node_color_map[node_key] = (hex_code, color_name)

        assigned_hex, assigned_name = node_color_map[node_key]
        entries.append(
            {
                "edge_index": edge_idx,
                "human_node": human_node["raw_node"],
                "human_entity": human_node["entity_name"],
                "human_part": human_node["part_name"],
                "object_node": object_node["raw_node"],
                "color_hex": assigned_hex,
                "color_name": assigned_name,
            }
        )

    return entries


def unique_contact_nodes(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    ordered: list[dict[str, str]] = []
    for entry in entries:
        node = str(entry["human_node"])
        if node in seen:
            continue
        seen.add(node)
        ordered.append(
            {
                "human_node": node,
                "human_part": str(entry["human_part"]),
                "color_hex": str(entry["color_hex"]),
                "color_name": str(entry["color_name"]),
            }
        )
    return ordered


def build_region_lines(contact_nodes: list[dict[str, str]]) -> str:
    lines = ["  - black background everywhere else (`#000000`)"]
    for node in contact_nodes:
        part = node["human_part"].replace(" ", "-")
        color_name = node["color_name"]
        hex_code = node["color_hex"]
        lines.append(
            f"  - {part} contact region in {color_name} (`{hex_code}`)"
        )
    return "\n".join(lines)


def build_prompt(contact_nodes: list[dict[str, str]]) -> str:
    region_lines = build_region_lines(contact_nodes)
    return (
        "You are generating a segmentation-style contact mask from an image.\n\n"
        "Image is a scene with a person touching the target object, and you need "
        "to infer human-object contact.\n\n"
        "Task\n\n"
        "- Remove the person entirely.\n"
        "- Detect the object-side contact regions where the person’s body parts "
        "touch the target object.\n"
        "- Output a mask-only image:\n"
        f"{region_lines}\n\n"
        "Rules\n\n"
        "- Preserve the original image aspect ratio exactly. The output mask must "
        "have the same aspect ratio as Image A.\n"
        "- Do not crop, pad, stretch, or reframe.\n"
        "- Do not render the scene, furniture, object texture, or the person.\n"
        "- Show only the contact regions as solid filled areas on a black background.\n"
        "- The mask must represent only the object-side touch footprint, never the "
        "human-side surface.\n"
        "- Do not place any mask pixels on the human body or in free space.\n"
        "- Use crisp edges and exact solid colors.\n"
        "- Keep the regions conservative and physically plausible.\n"
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Estimate contact-mask color assignments from PAG interaction edges and "
            "generate a contact-mask prompt."
        )
    )
    parser.add_argument("--video_name", default="video_01")
    parser.add_argument("--pag", default=None, help="Path to output_pag_*.json")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Defaults to Generate_Video/output/<video_name>.",
    )
    args = parser.parse_args()

    pag_path = resolve_pag_path(script_dir, args.video_name, args.pag)
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else (script_dir / "output" / args.video_name).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    raw_pag_payload = load_json(pag_path)
    pag_payload = raw_pag_payload.get("pag", raw_pag_payload)
    if not isinstance(pag_payload, dict):
        raise ValueError("PAG payload must be a JSON object.")

    contact_entries = collect_contact_entries(pag_payload)
    if not contact_entries:
        raise ValueError("No human-object interaction edges found in PAG.")

    contact_nodes = unique_contact_nodes(contact_entries)
    prompt = build_prompt(contact_nodes)

    payload = {
        "video_name": args.video_name,
        "pag_json": str(pag_path),
        "contact_nodes": contact_nodes,
        "prompt": prompt,
    }

    json_path = output_root / "contact_mask_prompt_payload.json"
    prompt_path = output_root / "contact_mask_prompt.md"
    save_json(json_path, payload)
    save_text(prompt_path, prompt)

    print(f"PAG file: {pag_path}")
    print(f"Saved contact prompt payload: {json_path}")
    print(f"Saved contact prompt markdown: {prompt_path}")


if __name__ == "__main__":
    main()
