#!/usr/bin/env python3
"""Select ScanNet++ scenes for diverse human-scene interaction examples.

This script is intentionally independent of the low-clutter chair/table scene
ranking. It computes its own visual usability metrics from DSLR images and its
own affordance-diversity metrics from scans/segments_anno.json labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
LABEL_RE = re.compile(rb'"label"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"')

AFFORDANCE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "sit_table": re.compile(r"\b(chair|stool|bench|sofa|couch|armchair|table|desk)\b"),
    "kitchen_cook": re.compile(
        r"\b(kitchen cabinet|kitchen counter|countertop|counter top|stove|stovetop|oven|"
        r"microwave|fridge|refrigerator|dishwasher|cooktop|kettle|toaster|coffee maker|"
        r"coffee machine|sink|pan|pot|knife|cutting board|chopping board)\b"
    ),
    "bathroom_wash": re.compile(
        r"\b(toilet|shower|bathtub|bath tub|sink|washbasin|towel|mirror|soap|faucet|"
        r"toilet paper|shower curtain|shower head|shower hose|toothbrush|toothpaste)\b"
    ),
    "bed_lie": re.compile(r"\b(bed|mattress|pillow|blanket|duvet|bedsheet|headboard|nightstand|bedside)\b"),
    "storage_open": re.compile(r"\b(cabinet|drawer|cupboard|wardrobe|closet|locker|shelf|shelving)\b"),
    "screen_work": re.compile(
        r"\b(computer|monitor|keyboard|mouse|laptop|printer|copier|scanner|tv|television|screen|projector)\b"
    ),
    "whiteboard_present": re.compile(
        r"\b(whiteboard|blackboard|chalkboard|marker board|bulletin board|podium|lectern|projector screen)\b"
    ),
    "clean_laundry": re.compile(
        r"\b(washing machine|washer|dryer|laundry|mop|broom|vacuum|cleaner|trash can|"
        r"trash bin|bucket|hamper|detergent|spray bottle|dustpan)\b"
    ),
    "play_exercise_music": re.compile(
        r"\b(treadmill|kettlebell|dumbbell|weight|weights|weight plate|exercise|gym|bike|"
        r"bicycle|ping pong|table tennis|billiard|pool table|foosball|guitar|piano|"
        r"speaker|stereo|drum)\b"
    ),
    "door_transition": re.compile(r"\b(door|stair|stairs|staircase|elevator|ramp|railing|corridor|hallway)\b"),
    "eat_drink": re.compile(r"\b(plate|bowl|cup|mug|glass|bottle|coffee|dining|kettle|toaster|pitcher)\b"),
}

CORE_CATEGORIES = {
    "kitchen_cook",
    "bathroom_wash",
    "bed_lie",
    "screen_work",
    "whiteboard_present",
    "clean_laundry",
    "play_exercise_music",
}

CATEGORY_WEIGHTS = {
    "play_exercise_music": 1.55,
    "whiteboard_present": 1.40,
    "kitchen_cook": 1.35,
    "bathroom_wash": 1.30,
    "bed_lie": 1.25,
    "clean_laundry": 1.25,
    "screen_work": 1.20,
    "storage_open": 0.85,
    "door_transition": 0.70,
    "eat_drink": 0.60,
    "sit_table": 0.35,
}

CATEGORY_IDEAS = {
    "kitchen_cook": "cook / wash dishes / use appliance",
    "bathroom_wash": "wash hands / shower / towel / toilet",
    "bed_lie": "lie down / make bed / pack near bed",
    "storage_open": "open cabinet / shelf / closet",
    "screen_work": "type / use monitor / print",
    "whiteboard_present": "write / point / present",
    "clean_laundry": "clean / mop / vacuum / laundry",
    "play_exercise_music": "exercise / play music / game",
    "door_transition": "enter / exit / open door",
    "eat_drink": "drink / eat / handle containers",
    "sit_table": "sit / reach at table",
}

CLUTTER_LABELS = {
    "object",
    "objects",
    "box",
    "cardboard box",
    "paper",
    "papers",
    "bag",
    "backpack",
    "clothes",
    "cable",
    "cables",
}

IGNORED_LABELS = {"remove", "unannotated", "wall", "floor", "ceiling", "split", "object", "objects"}

SCENE_TYPE_NOVELTY = {
    "kitchen": 0.10,
    "bathroom": 0.10,
    "bedroom / hotel": 0.08,
    "storage / basement / garage": 0.08,
    "gym": 0.08,
    "laundry room": 0.08,
    "hallway": 0.05,
    "common area": 0.05,
    "table tennis room": 0.10,
    "pingpong room": 0.10,
    "living room / lounge": 0.04,
    "lounge": 0.04,
    "office": 0.03,
    "classroom": 0.03,
    "conference room": -0.04,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select affordance-diverse HSI scenes.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data", help="Dataset root")
    parser.add_argument(
        "--scene-types",
        type=Path,
        default=PROJECT_ROOT / "metadata" / "scene_types.json",
        help="Scene type metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hsi_diverse_scene_selection",
        help="Directory for the ranking CSV and selected_scenes_export/",
    )
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--top-k", type=int, default=20, help="Number of balanced recommendations to export")
    parser.add_argument("--sample-images", type=int, default=8, help="DSLR images sampled per scene")
    parser.add_argument("--max-side", type=int, default=640, help="Resize sampled images to this max side")
    parser.add_argument("--min-open-ratio", type=float, default=0.62, help="Minimum usable open-space ratio")
    parser.add_argument("--max-visual-clutter", type=float, default=0.70, help="Maximum allowed visual clutter")
    parser.add_argument(
        "--min-core-categories",
        type=int,
        default=1,
        help="Minimum core non sit/table interaction categories",
    )
    parser.add_argument("--max-per-scene-type", type=int, default=3, help="Balanced export scene-type quota")
    parser.add_argument("--max-per-primary-category", type=int, default=3, help="Balanced export category quota")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 8) // 2),
        help="Parallel workers for image scoring",
    )
    return parser.parse_args()


def output_csv_path(args: argparse.Namespace) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_dir / "hsi_diverse_scene_ranking.csv"


def load_scene_types(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {str(k): str(v) for k, v in data.items()}


def list_images(img_dir: Path) -> List[Path]:
    files: List[Path] = []
    for pattern in IMAGE_EXTS:
        files.extend(img_dir.glob(pattern))
    return sorted(files)


def sample_uniform(items: List[Path], k: int) -> List[Path]:
    if not items:
        return []
    k = min(k, len(items))
    if k == len(items):
        return items
    idx = np.linspace(0, len(items) - 1, num=k)
    picked = sorted({int(round(i)) for i in idx})
    while len(picked) < k:
        picked.append(min(len(items) - 1, picked[-1] + 1 if picked else 0))
        picked = sorted(set(picked))
    return [items[i] for i in picked[:k]]


def load_gray(path: Path, max_side: int) -> np.ndarray | None:
    if cv2 is not None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            img = cv2.resize(
                img,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 255.0

    if Image is None:
        return None

    try:
        with Image.open(path) as pil:
            pil = pil.convert("L")
            w, h = pil.size
            if max(w, h) > max_side:
                scale = max_side / float(max(w, h))
                pil = pil.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.Resampling.BILINEAR,
                )
            return np.asarray(pil, dtype=np.float32) / 255.0
    except Exception:
        return None


def image_metrics(gray: np.ndarray) -> Dict[str, float]:
    gray_u8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)

    if cv2 is not None:
        med = float(np.median(gray_u8))
        low = int(max(0.0, 0.66 * med))
        high = int(min(255.0, 1.33 * med))
        if high <= low:
            high = min(255, low + 1)
        edges = cv2.Canny(gray_u8, threshold1=low, threshold2=high)
        edge_density = float(np.mean(edges > 0))
        lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_32F).var())
    else:
        gx = np.diff(gray, axis=1, append=gray[:, -1:])
        gy = np.diff(gray, axis=0, append=gray[-1:, :])
        grad = np.hypot(gx, gy)
        edge_density = float(np.mean(grad > 0.07))
        lap = np.diff(gray, n=2, axis=0, prepend=gray[:1, :], append=gray[-1:, :])
        lap_var = float(np.var(lap)) * (255.0 * 255.0)

    h, w = gray.shape
    lower = gray[int(h * 0.45) :, :] if h > 10 else gray
    center = gray[int(h * 0.25) : int(h * 0.85), int(w * 0.18) : int(w * 0.82)] if h > 10 and w > 10 else gray

    lower_gx = np.diff(lower, axis=1, append=lower[:, -1:])
    lower_gy = np.diff(lower, axis=0, append=lower[-1:, :])
    lower_grad = np.hypot(lower_gx, lower_gy)
    open_ratio = float(np.mean(lower_grad < 0.03))

    center_gx = np.diff(center, axis=1, append=center[:, -1:])
    center_gy = np.diff(center, axis=0, append=center[-1:, :])
    center_grad = np.hypot(center_gx, center_gy)
    center_detail = float(np.mean(center_grad > 0.035))

    brightness = float(np.mean(gray))
    exposure_balance = 1.0 - min(abs(brightness - 0.50) / 0.50, 1.0)

    return {
        "edge_density": edge_density,
        "laplacian_var": lap_var,
        "open_ratio": open_ratio,
        "center_detail": center_detail,
        "exposure_balance": exposure_balance,
    }


def discover_scenes(data_root: Path) -> List[Tuple[str, Path, List[Path]]]:
    scenes: List[Tuple[str, Path, List[Path]]] = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        img_dir = scene_dir / "dslr" / "resized_undistorted_images"
        if not img_dir.is_dir():
            continue
        images = list_images(img_dir)
        if images:
            scenes.append((scene_dir.name, scene_dir, images))
    return scenes


def aggregate_scene_visual(
    scene_id: str,
    image_paths: List[Path],
    sample_images: int,
    max_side: int,
) -> Dict[str, object] | None:
    sampled = sample_uniform(image_paths, sample_images)
    stats: List[Dict[str, float]] = []
    paths: List[Path] = []
    for path in sampled:
        gray = load_gray(path, max_side=max_side)
        if gray is None:
            continue
        stats.append(image_metrics(gray))
        paths.append(path)
    if not stats:
        return None

    edge_vals = np.array([s["edge_density"] for s in stats], dtype=np.float32)
    lap_vals = np.array([s["laplacian_var"] for s in stats], dtype=np.float32)
    open_vals = np.array([s["open_ratio"] for s in stats], dtype=np.float32)
    detail_vals = np.array([s["center_detail"] for s in stats], dtype=np.float32)
    exposure_vals = np.array([s["exposure_balance"] for s in stats], dtype=np.float32)

    view_scores = []
    for s in stats:
        detail_preference = 1.0 - min(abs(s["center_detail"] - 0.07) / 0.12, 1.0)
        open_preference = 1.0 - min(abs(s["open_ratio"] - 0.78) / 0.35, 1.0)
        edge_preference = 1.0 - min(s["edge_density"] / 0.18, 1.0)
        view_scores.append(0.38 * open_preference + 0.32 * detail_preference + 0.18 * edge_preference + 0.12 * s["exposure_balance"])
    representative_image = paths[int(np.argmax(np.array(view_scores, dtype=np.float32)))]

    return {
        "scene_id": scene_id,
        "images_total": len(image_paths),
        "images_sampled": len(stats),
        "edge_density": float(np.median(edge_vals)),
        "laplacian_var": float(np.median(lap_vals)),
        "open_ratio": float(np.median(open_vals)),
        "center_detail": float(np.median(detail_vals)),
        "exposure_balance": float(np.median(exposure_vals)),
        "representative_image": representative_image.as_posix(),
    }


def robust_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo + 1e-12:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def stream_labels(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[str]:
    tail = b""
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                for match in LABEL_RE.finditer(tail):
                    yield match.group(1).decode("utf-8", errors="ignore").strip().lower()
                return

            data = tail + chunk
            cutoff = max(0, len(data) - 256)
            for match in LABEL_RE.finditer(data[:cutoff]):
                yield match.group(1).decode("utf-8", errors="ignore").strip().lower()
            tail = data[cutoff:]


def scan_scene_labels(scene_dir: Path) -> Counter[str]:
    anno = scene_dir / "scans" / "segments_anno.json"
    if not anno.is_file():
        return Counter()
    return Counter(stream_labels(anno))


def label_categories(labels: Counter[str]) -> Counter[str]:
    categories: Counter[str] = Counter()
    for label, count in labels.items():
        if label in {"remove", "unannotated"}:
            continue
        for category, pattern in AFFORDANCE_PATTERNS.items():
            if pattern.search(label):
                categories[category] += count
    return categories


def weighted_category_score(categories: Counter[str]) -> float:
    score = 0.0
    max_score = 0.0
    for category, weight in CATEGORY_WEIGHTS.items():
        max_score += weight
        count = categories.get(category, 0)
        if count > 0:
            score += weight * min(math.log1p(count) / math.log1p(12), 1.0)
    return score / max_score if max_score > 0 else 0.0


def primary_category(categories: Counter[str]) -> str:
    scored = []
    for category, count in categories.items():
        if count <= 0:
            continue
        scored.append((CATEGORY_WEIGHTS.get(category, 1.0) * min(count, 12), count, category))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def sorted_categories(categories: Counter[str]) -> List[str]:
    scored = []
    for category, count in categories.items():
        if count <= 0:
            continue
        scored.append((CATEGORY_WEIGHTS.get(category, 1.0) * min(count, 12), count, category))
    scored.sort(reverse=True)
    return [category for _, _, category in scored]


def category_summary(categories: Counter[str]) -> str:
    return ";".join(f"{category}:{count}" for category, count in categories.most_common() if count > 0)


def label_summary(labels: Counter[str], limit: int = 18) -> str:
    useful = [(label, count) for label, count in labels.items() if label not in IGNORED_LABELS]
    useful.sort(key=lambda item: (-item[1], item[0]))
    return ";".join(f"{label}:{count}" for label, count in useful[:limit])


def scene_type_bonus(scene_type: str) -> float:
    key = scene_type.strip().lower()
    return SCENE_TYPE_NOVELTY.get(key, 0.0)


def make_affordance_metrics(labels: Counter[str]) -> Dict[str, object]:
    categories = label_categories(labels)
    core_count = sum(1 for category in CORE_CATEGORIES if categories.get(category, 0) > 0)
    non_table_count = sum(1 for category, count in categories.items() if category != "sit_table" and count > 0)
    total_affordance_count = sum(categories.values())
    sit_count = categories.get("sit_table", 0)
    sit_dominance = sit_count / total_affordance_count if total_affordance_count else 0.0
    clutter_count = sum(count for label, count in labels.items() if label in CLUTTER_LABELS)
    annotated_count = sum(labels.values())
    clutter_ratio = clutter_count / annotated_count if annotated_count else 0.0

    richness = weighted_category_score(categories)
    breadth = min(core_count / 4.0, 1.0)
    non_table_breadth = min(non_table_count / 6.0, 1.0)
    category_score = 0.52 * richness + 0.30 * breadth + 0.18 * non_table_breadth
    category_score -= 0.18 * max(sit_dominance - 0.45, 0.0)
    category_score -= 0.20 * min(clutter_ratio / 0.45, 1.0)
    category_score = float(np.clip(category_score, 0.0, 1.0))

    ordered = sorted_categories(categories)
    ideas = [CATEGORY_IDEAS[category] for category in ordered if category in CATEGORY_IDEAS and category != "sit_table"]

    return {
        "category_score": category_score,
        "core_category_count": core_count,
        "non_table_category_count": non_table_count,
        "primary_category": primary_category(categories),
        "category_summary": category_summary(categories),
        "interaction_ideas": "; ".join(ideas[:5]),
        "useful_labels": label_summary(labels),
        "annotated_object_count": annotated_count,
        "clutter_label_count": clutter_count,
        "clutter_label_ratio": clutter_ratio,
        "sit_table_dominance": sit_dominance,
    }


def build_records(args: argparse.Namespace) -> List[Dict[str, object]]:
    scene_types = load_scene_types(args.scene_types)
    scenes = discover_scenes(args.data_root)
    if not scenes:
        raise SystemExit(f"No scenes with dslr/resized_undistorted_images found in {args.data_root}")

    print(f"Discovered {len(scenes)} scenes with undistorted DSLR images.")

    records: List[Dict[str, object]] = []
    total = len(scenes)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(aggregate_scene_visual, scene_id, images, args.sample_images, args.max_side): (scene_id, scene_dir)
            for scene_id, scene_dir, images in scenes
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            visual = future.result()
            if visual is not None:
                records.append(visual)
            if idx % 80 == 0 or idx == total:
                print(f"Visual scoring progress: {idx}/{total}")

    if not records:
        raise SystemExit("Failed to compute visual metrics.")

    edges = np.array([float(r["edge_density"]) for r in records], dtype=np.float32)
    laps = np.array([float(r["laplacian_var"]) for r in records], dtype=np.float32)
    opens = np.array([float(r["open_ratio"]) for r in records], dtype=np.float32)
    details = np.array([float(r["center_detail"]) for r in records], dtype=np.float32)

    edge_n = robust_norm(edges)
    lap_n = robust_norm(laps)
    open_n = robust_norm(opens)
    detail_n = robust_norm(details)

    for i, record in enumerate(records):
        visual_clutter = 0.46 * edge_n[i] + 0.34 * lap_n[i] + 0.20 * (1.0 - open_n[i])
        detail_quality = 1.0 - abs(float(detail_n[i]) - 0.45) / 0.55
        detail_quality = float(np.clip(detail_quality, 0.0, 1.0))
        visual_usability = (
            0.45 * (1.0 - float(visual_clutter))
            + 0.35 * float(open_n[i])
            + 0.12 * detail_quality
            + 0.08 * float(record["exposure_balance"])
        )
        record["visual_clutter"] = float(visual_clutter)
        record["visual_usability"] = float(np.clip(visual_usability, 0.0, 1.0))
        record["scene_type"] = scene_types.get(str(record["scene_id"]), "unknown")

    scene_dirs = {scene_id: scene_dir for scene_id, scene_dir, _ in scenes}
    for idx, record in enumerate(records, start=1):
        scene_id = str(record["scene_id"])
        labels = scan_scene_labels(scene_dirs[scene_id])
        record.update(make_affordance_metrics(labels))
        stype_bonus = scene_type_bonus(str(record["scene_type"]))
        record["scene_type_bonus"] = stype_bonus
        record["diverse_score"] = float(
            np.clip(
                0.58 * float(record["category_score"])
                + 0.32 * float(record["visual_usability"])
                + 0.10 * stype_bonus,
                0.0,
                1.0,
            )
        )
        if idx % 120 == 0 or idx == len(records):
            print(f"Label/affordance scoring progress: {idx}/{len(records)}")

    records.sort(
        key=lambda r: (
            float(r["diverse_score"]),
            int(r["core_category_count"]),
            float(r["visual_usability"]),
            float(r["open_ratio"]),
        ),
        reverse=True,
    )
    return records


def is_usable(record: Dict[str, object], args: argparse.Namespace) -> bool:
    return (
        float(record["open_ratio"]) >= args.min_open_ratio
        and float(record["visual_clutter"]) <= args.max_visual_clutter
        and int(record["core_category_count"]) >= args.min_core_categories
        and str(record["primary_category"]) != "sit_table"
    )


def balanced_top(records: List[Dict[str, object]], args: argparse.Namespace) -> List[Dict[str, object]]:
    picked: List[Dict[str, object]] = []
    by_scene_type: Counter[str] = Counter()
    by_category: Counter[str] = Counter()
    for record in records:
        if not is_usable(record, args):
            continue
        scene_type = str(record["scene_type"]).strip().lower()
        category = str(record["primary_category"])
        if by_scene_type[scene_type] >= args.max_per_scene_type:
            continue
        if category and by_category[category] >= args.max_per_primary_category:
            continue
        picked.append(record)
        by_scene_type[scene_type] += 1
        by_category[category] += 1
        if len(picked) >= args.top_k:
            break
    return picked


def resolve_image_path(image_path: object, data_root: Path) -> Path | None:
    p = Path(str(image_path))
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(data_root.parent / p)
        if p.parts and p.parts[0] == data_root.name:
            candidates.append(data_root.parent / p)
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    return None


def write_selected_export(
    selected: List[Dict[str, object]],
    fields: List[str],
    output_dir: Path,
    data_root: Path,
) -> Path:
    export_dir = output_dir / "selected_scenes_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    image_dir = export_dir / "representative_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = export_dir / "selected_scene_candidates.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({field: row.get(field, "") for field in fields})

    for row in selected:
        scene_id = str(row.get("scene_id", "scene"))
        src = resolve_image_path(row.get("representative_image", ""), data_root)
        if src is None:
            continue
        suffix = src.suffix if src.suffix else ".jpg"
        shutil.copy2(src, image_dir / f"{scene_id}{suffix}")

    return export_dir


def write_outputs(records: List[Dict[str, object]], selected: List[Dict[str, object]], args: argparse.Namespace) -> Tuple[Path, Path]:
    output_csv = output_csv_path(args)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_id",
        "scene_type",
        "diverse_score",
        "category_score",
        "visual_usability",
        "scene_type_bonus",
        "primary_category",
        "core_category_count",
        "non_table_category_count",
        "sit_table_dominance",
        "clutter_label_ratio",
        "annotated_object_count",
        "clutter_label_count",
        "open_ratio",
        "visual_clutter",
        "edge_density",
        "laplacian_var",
        "center_detail",
        "exposure_balance",
        "images_total",
        "images_sampled",
        "category_summary",
        "interaction_ideas",
        "useful_labels",
        "representative_image",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})

    export_dir = write_selected_export(selected, fields, output_csv.parent, args.data_root)
    return output_csv, export_dir


def main() -> None:
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"data root not found: {args.data_root}")
    if Image is None and cv2 is None:
        raise SystemExit("Need either Pillow or OpenCV to read images.")

    records = build_records(args)
    selected = balanced_top(records, args)
    output_csv, export_dir = write_outputs(records, selected, args)

    print(f"\nSaved diverse ranking CSV to: {output_csv}")
    print(f"Saved selected export to: {export_dir}")
    print("\nBalanced diverse recommendations:")
    for idx, record in enumerate(selected, start=1):
        print(
            f"{idx}. {record['scene_id']} | type={record['scene_type']} | "
            f"primary={record['primary_category']} | score={float(record['diverse_score']):.3f} | "
            f"category={float(record['category_score']):.3f} | visual={float(record['visual_usability']):.3f}"
        )
        print(f"   ideas: {record['interaction_ideas']}")
        print(f"   labels: {record['useful_labels']}")
        print(f"   img: {record['representative_image']}")


if __name__ == "__main__":
    main()
