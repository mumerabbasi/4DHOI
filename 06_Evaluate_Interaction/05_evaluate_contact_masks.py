from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SETTING_MASK_DIRS = {
    "round1": Path("rounds") / "round_01" / "contact_masks",
    "final": Path("contact_masks"),
}
BILATERAL_SWAP_MIN_IMPROVEMENT_NORMALIZED = 0.02

EDGE_CSV_FIELDNAMES = [
    "setting",
    "interaction_name",
    "contact_type",
    "scene_target",
    "body_part",
    "gt_body_part",
    "bilateral_swapped",
    "status",
    "gt_mask_path",
    "predicted_mask_path",
    "width",
    "height",
    "image_diagonal_px",
    "gt_pixels",
    "predicted_pixels",
    "intersection_pixels",
    "union_pixels",
    "gt_centroid_x",
    "gt_centroid_y",
    "predicted_centroid_x",
    "predicted_centroid_y",
    "containment",
    "centroid_distance_px",
    "centroid_distance_normalized",
    "iou",
]
INTERACTION_CSV_FIELDNAMES = [
    "setting",
    "interaction_name",
    "num_required_edges",
    "num_evaluated_edges",
    "num_skipped_gt_edges",
    "num_misses",
    "num_bilateral_swapped_edges",
    "num_centroid_evaluated_edges",
    "mean_containment",
    "mean_centroid_distance_px",
    "mean_centroid_distance_normalized",
    "mean_iou",
]
SETTING_CSV_FIELDNAMES = [
    "setting",
    "num_interactions_discovered",
    "num_interactions_aggregated",
    "num_required_edges",
    "num_evaluated_edges",
    "num_skipped_gt_edges",
    "num_misses",
    "num_bilateral_swapped_edges",
    "num_centroid_evaluated_edges",
    "mean_containment",
    "mean_centroid_distance_px",
    "mean_centroid_distance_normalized",
    "mean_iou",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate round-1 and final agentic contact masks against manually "
            "annotated ground-truth masks."
        )
    )
    parser.add_argument(
        "--gt-output-dir",
        "--gt_output_dir",
        dest="gt_output_dir",
        default=str(PROJECT_DIR / "00_Annotate_GT_Contact" / "output"),
        help="Root containing GT interaction_*/contact_masks_gt directories.",
    )
    parser.add_argument(
        "--agentic-output-dir",
        "--agentic_output_dir",
        dest="agentic_output_dir",
        default=str(PROJECT_DIR / "03_Estimate_Contact_Agentic" / "output"),
        help="Root containing round-1 and final agentic masks.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=str(SCRIPT_DIR / "output_contact_masks"),
        help="Directory in which evaluation CSV and JSON files are written.",
    )
    parser.add_argument(
        "--interaction-name",
        "--interaction_name",
        dest="interaction_name",
        default=None,
        help="Evaluate one interaction (for example interaction_02).",
    )
    return parser.parse_args(argv)


def normalize_label(value: str) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def slugify(value: str) -> str:
    return normalize_label(value).replace(" ", "_")


def interaction_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    suffix = name.removeprefix("interaction_")
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"WARNING: {message}", file=sys.stderr)


def scene_targets_by_body_part(sig_path: Path) -> dict[str, str]:
    if not sig_path.exists():
        raise FileNotFoundError(f"SIG file not found for bilateral matching: {sig_path}")
    payload = load_json(sig_path)
    interactions = payload.get("interaction_edges", [])
    if not isinstance(interactions, list):
        raise ValueError(f"Expected list field 'interaction_edges' in {sig_path}")

    targets: dict[str, str] = {}
    for interaction in interactions:
        if not isinstance(interaction, dict):
            continue
        body_part = normalize_label(str(interaction.get("human_part", "")))
        scene_target = normalize_label(str(interaction.get("scene_element", "")))
        if not body_part or not scene_target:
            continue
        if body_part in targets and targets[body_part] != scene_target:
            raise ValueError(
                f"Body part '{body_part}' has multiple scene targets in {sig_path}"
            )
        targets[body_part] = scene_target
    return targets


def required_edges(
    metadata: dict[str, Any],
    metadata_path: Path,
    sig_path: Path,
) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    seen: set[str] = set()
    scene_targets = scene_targets_by_body_part(sig_path)
    for key, contact_type in (("human_parts", "object"), ("floor_parts", "floor")):
        values = metadata.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"Expected list field '{key}' in {metadata_path}")
        for raw_part in values:
            body_part = normalize_label(str(raw_part))
            if not body_part:
                raise ValueError(f"Empty body-part identity in {metadata_path}:{key}")
            if body_part in seen:
                raise ValueError(
                    f"Duplicate required body part '{body_part}' in {metadata_path}"
                )
            seen.add(body_part)
            scene_target = scene_targets.get(body_part)
            if scene_target is None:
                raise ValueError(
                    f"Required body part '{body_part}' has no SIG scene target in "
                    f"{sig_path}"
                )
            edges.append(
                {
                    "body_part": body_part,
                    "contact_type": contact_type,
                    "scene_target": scene_target,
                }
            )
    if not edges:
        raise ValueError(f"No required contact edges found in {metadata_path}")
    return edges


def palette_colors(metadata: dict[str, Any], metadata_path: Path) -> dict[str, tuple[int, int, int]]:
    palette = metadata.get("palette", [])
    if palette is None:
        return {}
    if not isinstance(palette, list):
        raise ValueError(f"Expected list field 'palette' in {metadata_path}")

    colors: dict[str, tuple[int, int, int]] = {}
    for entry in palette:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed palette entry in {metadata_path}")
        body_part = normalize_label(str(entry.get("part", "")))
        rgb = entry.get("rgb")
        if not body_part or not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError(f"Malformed palette identity in {metadata_path}: {entry}")
        color = tuple(int(channel) for channel in rgb)
        if any(channel < 0 or channel > 255 for channel in color):
            raise ValueError(f"Invalid RGB color in {metadata_path}: {entry}")
        if body_part in colors and colors[body_part] != color:
            raise ValueError(
                f"Conflicting colors for body part '{body_part}' in {metadata_path}"
            )
        colors[body_part] = color
    return colors


def read_binary_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if mask is None else mask > 127


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("Cannot compute the centroid of an empty mask")
    return float(np.mean(xs)), float(np.mean(ys))


def bilateral_gt_assignments(
    setting: str,
    interaction_name: str,
    edges: list[dict[str, str]],
    gt_mask_dir: Path,
    predicted_mask_dir: Path,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    assignments = {edge["body_part"]: edge["body_part"] for edge in edges}
    edge_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for edge in edges:
        part_tokens = edge["body_part"].split()
        if len(part_tokens) < 2 or part_tokens[0] not in {"left", "right"}:
            continue
        side = part_tokens[0]
        base_part = " ".join(part_tokens[1:])
        edge_by_key[(edge["scene_target"], base_part, side)] = edge

    swaps: list[dict[str, Any]] = []
    for scene_target, base_part, side in list(edge_by_key):
        if side != "left":
            continue
        left_edge = edge_by_key.get((scene_target, base_part, "left"))
        right_edge = edge_by_key.get((scene_target, base_part, "right"))
        if left_edge is None or right_edge is None:
            continue

        left_part = left_edge["body_part"]
        right_part = right_edge["body_part"]
        paths = {
            "predicted_left": predicted_mask_dir / f"{slugify(left_part)}.png",
            "predicted_right": predicted_mask_dir / f"{slugify(right_part)}.png",
            "gt_left": gt_mask_dir / f"{slugify(left_part)}.png",
            "gt_right": gt_mask_dir / f"{slugify(right_part)}.png",
        }
        masks: dict[str, np.ndarray] = {}
        pair_is_usable = True
        for key, path in paths.items():
            if not path.exists():
                pair_is_usable = False
                break
            mask = read_binary_mask(path)
            if mask is None or not np.any(mask):
                pair_is_usable = False
                break
            masks[key] = mask
        if not pair_is_usable:
            continue
        shapes = {mask.shape for mask in masks.values()}
        if len(shapes) != 1:
            raise ValueError(
                f"Bilateral mask shape mismatch for {setting}/{interaction_name}/"
                f"{base_part}: {sorted(shapes)}"
            )

        height, width = next(iter(shapes))
        diagonal = float(math.hypot(width, height))
        predicted_left = np.asarray(mask_centroid(masks["predicted_left"]))
        predicted_right = np.asarray(mask_centroid(masks["predicted_right"]))
        gt_left = np.asarray(mask_centroid(masks["gt_left"]))
        gt_right = np.asarray(mask_centroid(masks["gt_right"]))
        current_cost = float(
            (
                np.linalg.norm(predicted_left - gt_left)
                + np.linalg.norm(predicted_right - gt_right)
            )
            / diagonal
        )
        swapped_cost = float(
            (
                np.linalg.norm(predicted_left - gt_right)
                + np.linalg.norm(predicted_right - gt_left)
            )
            / diagonal
        )
        if (
            swapped_cost + BILATERAL_SWAP_MIN_IMPROVEMENT_NORMALIZED
            >= current_cost
        ):
            continue

        assignments[left_part] = right_part
        assignments[right_part] = left_part
        swap = {
            "setting": setting,
            "interaction_name": interaction_name,
            "scene_target": scene_target,
            "base_part": base_part,
            "left_body_part": left_part,
            "right_body_part": right_part,
            "current_cost_normalized": current_cost,
            "swapped_cost_normalized": swapped_cost,
            "minimum_improvement_normalized": (
                BILATERAL_SWAP_MIN_IMPROVEMENT_NORMALIZED
            ),
        }
        swaps.append(swap)
        print(
            "  spatially swapped bilateral contact masks for "
            f"{setting}/{interaction_name}/{base_part} -> {scene_target}: "
            f"current_cost={current_cost:.4f}, swapped_cost={swapped_cost:.4f}"
        )
    return assignments, swaps


def blank_edge_row(
    setting: str,
    interaction_name: str,
    contact_type: str,
    scene_target: str,
    body_part: str,
    gt_body_part: str,
    gt_mask_path: Path,
    predicted_mask_path: Path,
) -> dict[str, Any]:
    return {
        field: None
        for field in EDGE_CSV_FIELDNAMES
    } | {
        "setting": setting,
        "interaction_name": interaction_name,
        "contact_type": contact_type,
        "scene_target": scene_target,
        "body_part": body_part,
        "gt_body_part": gt_body_part,
        "bilateral_swapped": body_part != gt_body_part,
        "gt_mask_path": str(gt_mask_path),
        "predicted_mask_path": str(predicted_mask_path),
    }


def skipped_gt_row(
    row: dict[str, Any],
    status: str,
    message: str,
    warnings: list[str],
) -> dict[str, Any]:
    row["status"] = status
    warn(message, warnings)
    return row


def missed_prediction_row(
    row: dict[str, Any],
    status: str,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    height, width = gt_mask.shape
    gt_pixels = int(np.count_nonzero(gt_mask))
    gt_centroid_x, gt_centroid_y = mask_centroid(gt_mask)
    diagonal = float(math.hypot(width, height))

    # Convention: an omitted required contact remains a containment/IoU miss,
    # while centroid distance is left undefined and excluded from distance means.
    row.update(
        {
            "status": status,
            "width": int(width),
            "height": int(height),
            "image_diagonal_px": diagonal,
            "gt_pixels": gt_pixels,
            "predicted_pixels": 0,
            "intersection_pixels": 0,
            "union_pixels": gt_pixels,
            "gt_centroid_x": gt_centroid_x,
            "gt_centroid_y": gt_centroid_y,
            "containment": 0.0,
            "centroid_distance_px": None,
            "centroid_distance_normalized": None,
            "iou": 0.0,
        }
    )
    return row


def evaluate_edge(
    setting: str,
    interaction_name: str,
    edge: dict[str, str],
    gt_body_part: str,
    gt_mask_dir: Path,
    predicted_mask_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    body_part = edge["body_part"]
    gt_mask_path = gt_mask_dir / f"{slugify(gt_body_part)}.png"
    predicted_mask_path = predicted_mask_dir / f"{slugify(body_part)}.png"
    row = blank_edge_row(
        setting=setting,
        interaction_name=interaction_name,
        contact_type=edge["contact_type"],
        scene_target=edge["scene_target"],
        body_part=body_part,
        gt_body_part=gt_body_part,
        gt_mask_path=gt_mask_path,
        predicted_mask_path=predicted_mask_path,
    )

    if not gt_mask_path.exists():
        return skipped_gt_row(
            row,
            "missing_gt_skipped",
            f"{setting}/{interaction_name}/{body_part}: missing GT mask; edge skipped: "
            f"{gt_mask_path}",
            warnings,
        )
    gt_mask = read_binary_mask(gt_mask_path)
    if gt_mask is None:
        return skipped_gt_row(
            row,
            "unreadable_gt_skipped",
            f"{setting}/{interaction_name}/{body_part}: unreadable GT mask; edge "
            f"skipped: {gt_mask_path}",
            warnings,
        )
    if not np.any(gt_mask):
        return skipped_gt_row(
            row,
            "empty_gt_skipped",
            f"{setting}/{interaction_name}/{body_part}: empty GT mask; edge skipped: "
            f"{gt_mask_path}",
            warnings,
        )

    if not predicted_mask_path.exists():
        warn(
            f"{setting}/{interaction_name}/{body_part}: predicted mask is missing; "
            "counted as a miss",
            warnings,
        )
        return missed_prediction_row(row, "missing_prediction_miss", gt_mask)
    predicted_mask = read_binary_mask(predicted_mask_path)
    if predicted_mask is None:
        warn(
            f"{setting}/{interaction_name}/{body_part}: predicted mask is unreadable; "
            "counted as a miss",
            warnings,
        )
        return missed_prediction_row(row, "unreadable_prediction_miss", gt_mask)
    if predicted_mask.shape != gt_mask.shape:
        raise ValueError(
            f"Mask shape mismatch for {setting}/{interaction_name}/{body_part}: "
            f"prediction {predicted_mask.shape[::-1]} at {predicted_mask_path}, "
            f"GT {gt_mask.shape[::-1]} at {gt_mask_path}. Masks are not resized "
            "during evaluation."
        )
    if not np.any(predicted_mask):
        warn(
            f"{setting}/{interaction_name}/{body_part}: predicted mask is empty; "
            "counted as a miss",
            warnings,
        )
        return missed_prediction_row(row, "empty_prediction_miss", gt_mask)

    height, width = gt_mask.shape
    diagonal = float(math.hypot(width, height))
    gt_pixels = int(np.count_nonzero(gt_mask))
    predicted_pixels = int(np.count_nonzero(predicted_mask))
    intersection_pixels = int(np.count_nonzero(predicted_mask & gt_mask))
    union_pixels = int(np.count_nonzero(predicted_mask | gt_mask))
    gt_centroid_x, gt_centroid_y = mask_centroid(gt_mask)
    predicted_centroid_x, predicted_centroid_y = mask_centroid(predicted_mask)
    centroid_distance_px = float(
        math.hypot(
            predicted_centroid_x - gt_centroid_x,
            predicted_centroid_y - gt_centroid_y,
        )
    )
    row.update(
        {
            "status": "ok",
            "width": int(width),
            "height": int(height),
            "image_diagonal_px": diagonal,
            "gt_pixels": gt_pixels,
            "predicted_pixels": predicted_pixels,
            "intersection_pixels": intersection_pixels,
            "union_pixels": union_pixels,
            "gt_centroid_x": gt_centroid_x,
            "gt_centroid_y": gt_centroid_y,
            "predicted_centroid_x": predicted_centroid_x,
            "predicted_centroid_y": predicted_centroid_y,
            "containment": float(intersection_pixels / predicted_pixels),
            "centroid_distance_px": centroid_distance_px,
            "centroid_distance_normalized": float(centroid_distance_px / diagonal),
            "iou": float(intersection_pixels / union_pixels),
        }
    )
    return row


def is_skipped_gt(row: dict[str, Any]) -> bool:
    return str(row["status"]).endswith("_gt_skipped")


def is_miss(row: dict[str, Any]) -> bool:
    return str(row["status"]).endswith("_miss")


def mean_field(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def summarize_interaction(
    setting: str,
    interaction_name: str,
    rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    evaluated_rows = [row for row in rows if not is_skipped_gt(row)]
    centroid_rows = [
        row for row in evaluated_rows if row["centroid_distance_px"] is not None
    ]
    skipped_count = len(rows) - len(evaluated_rows)
    summary: dict[str, Any] = {
        "setting": setting,
        "interaction_name": interaction_name,
        "num_required_edges": int(len(rows)),
        "num_evaluated_edges": int(len(evaluated_rows)),
        "num_skipped_gt_edges": int(skipped_count),
        "num_misses": int(sum(is_miss(row) for row in evaluated_rows)),
        "num_bilateral_swapped_edges": int(
            sum(bool(row["bilateral_swapped"]) for row in evaluated_rows)
        ),
        "num_centroid_evaluated_edges": int(len(centroid_rows)),
        "mean_containment": None,
        "mean_centroid_distance_px": None,
        "mean_centroid_distance_normalized": None,
        "mean_iou": None,
    }
    if not evaluated_rows:
        warn(
            f"{setting}/{interaction_name}: every GT edge was skipped; interaction "
            "excluded from the setting mean",
            warnings,
        )
        return summary

    summary.update(
        {
            "mean_containment": mean_field(evaluated_rows, "containment"),
            "mean_centroid_distance_px": (
                mean_field(centroid_rows, "centroid_distance_px")
                if centroid_rows
                else None
            ),
            "mean_centroid_distance_normalized": (
                mean_field(centroid_rows, "centroid_distance_normalized")
                if centroid_rows
                else None
            ),
            "mean_iou": mean_field(evaluated_rows, "iou"),
        }
    )
    return summary


def summarize_setting(
    setting: str,
    interaction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregated_rows = [
        row for row in interaction_rows if int(row["num_evaluated_edges"]) > 0
    ]
    if not aggregated_rows:
        raise RuntimeError(f"No interactions with valid GT masks for setting '{setting}'")
    centroid_interaction_rows = [
        row
        for row in aggregated_rows
        if int(row["num_centroid_evaluated_edges"]) > 0
    ]
    return {
        "setting": setting,
        "num_interactions_discovered": int(len(interaction_rows)),
        "num_interactions_aggregated": int(len(aggregated_rows)),
        "num_required_edges": int(
            sum(int(row["num_required_edges"]) for row in interaction_rows)
        ),
        "num_evaluated_edges": int(
            sum(int(row["num_evaluated_edges"]) for row in interaction_rows)
        ),
        "num_skipped_gt_edges": int(
            sum(int(row["num_skipped_gt_edges"]) for row in interaction_rows)
        ),
        "num_misses": int(sum(int(row["num_misses"]) for row in interaction_rows)),
        "num_bilateral_swapped_edges": int(
            sum(int(row["num_bilateral_swapped_edges"]) for row in interaction_rows)
        ),
        "num_centroid_evaluated_edges": int(
            sum(int(row["num_centroid_evaluated_edges"]) for row in interaction_rows)
        ),
        # Dataset means are deliberately means of per-interaction means, not
        # pooled edge means, so every interaction receives equal weight.
        "mean_containment": mean_field(aggregated_rows, "mean_containment"),
        "mean_centroid_distance_px": (
            mean_field(centroid_interaction_rows, "mean_centroid_distance_px")
            if centroid_interaction_rows
            else None
        ),
        "mean_centroid_distance_normalized": (
            mean_field(
                centroid_interaction_rows, "mean_centroid_distance_normalized"
            )
            if centroid_interaction_rows
            else None
        ),
        "mean_iou": mean_field(aggregated_rows, "mean_iou"),
    }


def load_optional_prediction_metadata(
    path: Path,
    setting: str,
    interaction_name: str,
    warnings: list[str],
) -> dict[str, Any]:
    if not path.exists():
        warn(
            f"{setting}/{interaction_name}: prediction metadata is missing; "
            "matching will use body-part filenames only: " + str(path),
            warnings,
        )
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        warn(
            f"{setting}/{interaction_name}: prediction metadata is unreadable; "
            f"matching will use body-part filenames only: {path} ({exc})",
            warnings,
        )
        return {}
    if not isinstance(payload, dict):
        warn(
            f"{setting}/{interaction_name}: prediction metadata is not an object; "
            "matching will use body-part filenames only: " + str(path),
            warnings,
        )
        return {}
    return payload


def validate_palette_identity(
    setting: str,
    interaction_name: str,
    object_parts: list[str],
    gt_metadata: dict[str, Any],
    gt_metadata_path: Path,
    predicted_metadata: dict[str, Any],
    predicted_metadata_path: Path,
    warnings: list[str],
) -> None:
    gt_colors = palette_colors(gt_metadata, gt_metadata_path)
    predicted_colors = (
        palette_colors(predicted_metadata, predicted_metadata_path)
        if predicted_metadata
        else {}
    )
    for body_part in object_parts:
        gt_color = gt_colors.get(body_part)
        predicted_color = predicted_colors.get(body_part)
        if gt_color is None:
            warn(
                f"{setting}/{interaction_name}/{body_part}: GT palette has no color "
                "entry; identity falls back to the body-part filename",
                warnings,
            )
        if predicted_metadata and predicted_color is None:
            warn(
                f"{setting}/{interaction_name}/{body_part}: prediction palette has no "
                "color entry; identity falls back to the body-part filename",
                warnings,
            )
        if gt_color is not None and predicted_color is not None and gt_color != predicted_color:
            raise ValueError(
                f"Palette identity mismatch for {setting}/{interaction_name}/{body_part}: "
                f"prediction RGB {predicted_color}, GT RGB {gt_color}."
            )


def discover_gt_interactions(
    gt_output_dir: Path,
    interaction_name: str | None,
) -> list[Path]:
    if interaction_name:
        interaction_dir = gt_output_dir / interaction_name
        if not interaction_dir.is_dir():
            raise FileNotFoundError(f"GT interaction directory not found: {interaction_dir}")
        return [interaction_dir]
    if not gt_output_dir.is_dir():
        raise FileNotFoundError(f"GT output directory not found: {gt_output_dir}")
    interaction_dirs = sorted(
        (path for path in gt_output_dir.glob("interaction_*") if path.is_dir()),
        key=interaction_sort_key,
    )
    if not interaction_dirs:
        raise RuntimeError(f"No interaction_* directories found under {gt_output_dir}")
    return interaction_dirs


def evaluate(
    gt_output_dir: Path,
    agentic_output_dir: Path,
    interaction_name: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    interaction_dirs = discover_gt_interactions(gt_output_dir, interaction_name)
    warnings: list[str] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_interaction_rows: list[dict[str, Any]] = []
    all_setting_rows: list[dict[str, Any]] = []

    interaction_inputs: list[tuple[str, Path, dict[str, Any], list[dict[str, str]]]] = []
    for interaction_dir in interaction_dirs:
        gt_metadata_path = interaction_dir / "contact_masks_gt" / "metadata.json"
        if not gt_metadata_path.exists():
            raise FileNotFoundError(f"GT metadata not found: {gt_metadata_path}")
        gt_metadata = load_json(gt_metadata_path)
        if not isinstance(gt_metadata, dict):
            raise ValueError(f"Expected JSON object in {gt_metadata_path}")
        sig_path = (
            PROJECT_DIR
            / "01_Generate_SIG"
            / "output"
            / interaction_dir.name
            / "sig.json"
        )
        interaction_inputs.append(
            (
                interaction_dir.name,
                interaction_dir,
                gt_metadata,
                required_edges(gt_metadata, gt_metadata_path, sig_path),
            )
        )

    for setting, relative_mask_dir in SETTING_MASK_DIRS.items():
        setting_interaction_rows: list[dict[str, Any]] = []
        for (
            current_interaction_name,
            gt_interaction_dir,
            gt_metadata,
            edges,
        ) in interaction_inputs:
            gt_mask_dir = gt_interaction_dir / "contact_masks_gt"
            predicted_mask_dir = (
                agentic_output_dir / current_interaction_name / relative_mask_dir
            )
            predicted_metadata_path = predicted_mask_dir / "metadata.json"
            predicted_metadata = load_optional_prediction_metadata(
                predicted_metadata_path,
                setting,
                current_interaction_name,
                warnings,
            )
            validate_palette_identity(
                setting=setting,
                interaction_name=current_interaction_name,
                object_parts=[
                    edge["body_part"]
                    for edge in edges
                    if edge["contact_type"] == "object"
                ],
                gt_metadata=gt_metadata,
                gt_metadata_path=gt_mask_dir / "metadata.json",
                predicted_metadata=predicted_metadata,
                predicted_metadata_path=predicted_metadata_path,
                warnings=warnings,
            )

            required_slugs = {slugify(edge["body_part"]) for edge in edges}
            if predicted_mask_dir.is_dir():
                extra_masks = sorted(
                    path
                    for path in predicted_mask_dir.glob("*.png")
                    if path.stem not in required_slugs
                )
                for extra_mask in extra_masks:
                    warn(
                        f"{setting}/{current_interaction_name}: predicted mask has no "
                        f"required GT identity and is ignored: {extra_mask}",
                        warnings,
                    )

            gt_assignments, _swaps = bilateral_gt_assignments(
                setting=setting,
                interaction_name=current_interaction_name,
                edges=edges,
                gt_mask_dir=gt_mask_dir,
                predicted_mask_dir=predicted_mask_dir,
            )

            interaction_edge_rows = [
                evaluate_edge(
                    setting=setting,
                    interaction_name=current_interaction_name,
                    edge=edge,
                    gt_body_part=gt_assignments[edge["body_part"]],
                    gt_mask_dir=gt_mask_dir,
                    predicted_mask_dir=predicted_mask_dir,
                    warnings=warnings,
                )
                for edge in edges
            ]
            all_edge_rows.extend(interaction_edge_rows)
            interaction_summary = summarize_interaction(
                setting=setting,
                interaction_name=current_interaction_name,
                rows=interaction_edge_rows,
                warnings=warnings,
            )
            setting_interaction_rows.append(interaction_summary)
            all_interaction_rows.append(interaction_summary)

        all_setting_rows.append(summarize_setting(setting, setting_interaction_rows))

    return all_edge_rows, all_interaction_rows, all_setting_rows, warnings


def build_json_payload(
    gt_output_dir: Path,
    agentic_output_dir: Path,
    edge_rows: list[dict[str, Any]],
    interaction_rows: list[dict[str, Any]],
    setting_rows: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for setting, relative_mask_dir in SETTING_MASK_DIRS.items():
        settings[setting] = {
            "prediction_mask_relative_path": str(relative_mask_dir),
            "summary": next(row for row in setting_rows if row["setting"] == setting),
            "interactions": [
                {
                    "summary": interaction_row,
                    "edges": [
                        edge_row
                        for edge_row in edge_rows
                        if edge_row["setting"] == setting
                        and edge_row["interaction_name"]
                        == interaction_row["interaction_name"]
                    ],
                }
                for interaction_row in interaction_rows
                if interaction_row["setting"] == setting
            ],
        }
    return {
        "schema_version": 2,
        "sources": {
            "gt_output_dir": str(gt_output_dir),
            "agentic_output_dir": str(agentic_output_dir),
        },
        "metric_definitions": {
            "containment": "|prediction ∩ GT| / |prediction|; higher is better",
            "centroid_distance_px": (
                "Euclidean distance between prediction and GT centroids in pixel-index "
                "coordinates; lower is better"
            ),
            "centroid_distance_normalized": (
                "centroid_distance_px / sqrt(width^2 + height^2); lower is better"
            ),
            "iou": "|prediction ∩ GT| / |prediction ∪ GT|; secondary metric",
            "aggregation": (
                "edge metrics are averaged per interaction, then interaction means are "
                "averaged per setting; missing predictions remain containment/IoU "
                "misses but are excluded from centroid-distance means"
            ),
            "bilateral_assignment": (
                "for left/right contacts with the same base part and SIG scene target, "
                "swap GT assignments when total normalized centroid distance improves "
                f"by more than {BILATERAL_SWAP_MIN_IMPROVEMENT_NORMALIZED}"
            ),
        },
        "empty_prediction_convention": {
            "containment": 0.0,
            "iou": 0.0,
            "centroid_distance_px": None,
            "centroid_distance_normalized": None,
            "excluded_from_centroid_means": True,
            "counted_as_miss": True,
        },
        "bilateral_swaps": [
            {
                "setting": row["setting"],
                "interaction_name": row["interaction_name"],
                "scene_target": row["scene_target"],
                "body_part": row["body_part"],
                "gt_body_part": row["gt_body_part"],
            }
            for row in edge_rows
            if bool(row["bilateral_swapped"])
        ],
        "warnings": warnings,
        "settings": settings,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    gt_output_dir = Path(args.gt_output_dir).resolve()
    agentic_output_dir = Path(args.agentic_output_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    edge_rows, interaction_rows, setting_rows, warnings = evaluate(
        gt_output_dir=gt_output_dir,
        agentic_output_dir=agentic_output_dir,
        interaction_name=args.interaction_name,
    )
    save_csv_rows(
        output_dir / "contact_mask_metrics.csv",
        setting_rows,
        SETTING_CSV_FIELDNAMES,
    )
    save_csv_rows(
        output_dir / "interaction_metrics.csv",
        interaction_rows,
        INTERACTION_CSV_FIELDNAMES,
    )
    save_csv_rows(output_dir / "edge_metrics.csv", edge_rows, EDGE_CSV_FIELDNAMES)
    save_json(
        output_dir / "metrics.json",
        build_json_payload(
            gt_output_dir=gt_output_dir,
            agentic_output_dir=agentic_output_dir,
            edge_rows=edge_rows,
            interaction_rows=interaction_rows,
            setting_rows=setting_rows,
            warnings=warnings,
        ),
    )

    print("Contact-mask evaluation")
    for row in setting_rows:
        print(
            f"  {row['setting']}: interactions="
            f"{row['num_interactions_aggregated']}/"
            f"{row['num_interactions_discovered']} "
            f"edges={row['num_evaluated_edges']}/{row['num_required_edges']} "
            f"misses={row['num_misses']} "
            f"swapped_edges={row['num_bilateral_swapped_edges']} "
            f"centroid_edges={row['num_centroid_evaluated_edges']} "
            f"containment={row['mean_containment']:.6f} "
            f"centroid_px={row['mean_centroid_distance_px']:.6f} "
            f"centroid_normalized="
            f"{row['mean_centroid_distance_normalized']:.6f}"
        )
    print(f"  warnings={len(warnings)}")
    for filename in (
        "contact_mask_metrics.csv",
        "interaction_metrics.csv",
        "edge_metrics.csv",
        "metrics.json",
    ):
        print(f"Wrote artifact: {output_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
