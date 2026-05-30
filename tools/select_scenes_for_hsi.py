#!/usr/bin/env python3
"""Rank ScanNet++ scenes for human-scene interaction experiments.

The ranking combines:
1) Visual unclutteredness from undistorted DSLR images.
2) Scene-type priors from metadata/scene_types.json.
3) Interaction suitability from labeled objects in scans/segments_anno.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
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


IMAGE_EXTS = ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG")
LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"')
PROJECT_ROOT = Path(__file__).resolve().parents[1]


CHAIR_LABELS = {
    "chair",
    "office chair",
    "rolling chair",
    "dining chair",
    "office visitor chair",
    "armchair",
    "arm chair",
    "stool",
    "bench",
    "lounge chair",
    "sofa chair",
}

INTERACTION_LABELS = CHAIR_LABELS | {
    "table",
    "office table",
    "conference table",
    "desk",
    "office desk",
    "coffee table",
    "sofa",
    "couch",
    "armchair",
    "arm chair",
}

SMALL_CLUTTER_LABELS = {
    "object",
    "objects",
    "book",
    "books",
    "paper",
    "papers",
    "folder",
    "bag",
    "backpack",
    "bottle",
    "bottles",
    "box",
    "cardboard box",
    "trash can",
    "trash bin",
    "container",
    "cable",
    "cables",
    "mug",
    "clothes",
}

SCENE_TYPE_BONUS = {
    "living room / lounge": 0.08,
    "conference room": 0.08,
    "office": 0.06,
    "apartment": 0.06,
    "classroom": 0.05,
    "kitchen": 0.02,
    "copy / mail room": 0.02,
    "lab": -0.03,
    "storage / basement / garage": -0.08,
    "machine": -0.10,
    "bathroom": -0.15,
    "laundry room": -0.05,
    "gym": -0.02,
    "bedroom / hotel": 0.02,
    "server room": -0.08,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select low-clutter HSI-friendly scenes.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data", help="Dataset root with scene directories")
    parser.add_argument(
        "--scene-types",
        type=Path,
        default=PROJECT_ROOT / "metadata" / "scene_types.json",
        help="Scene type metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "hsi_scene_selection",
        help="Directory for the ranking CSV and selected_scenes_export/",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to output-dir/hsi_scene_ranking.csv",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of scenes to return")
    parser.add_argument(
        "--sample-images",
        type=int,
        default=8,
        help="Undistorted DSLR images sampled per scene",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=640,
        help="Resize images so max(H, W) <= max-side for faster metrics",
    )
    parser.add_argument(
        "--metadata-candidates",
        type=int,
        default=140,
        help="Top pre-ranked scenes to scan with segments_anno labels",
    )
    parser.add_argument(
        "--min-open-ratio",
        type=float,
        default=0.38,
        help="Preferred minimum open-space ratio (fallback if too strict)",
    )
    parser.add_argument(
        "--min-chairs",
        type=int,
        default=1,
        help="Preferred minimum chair-like objects (fallback if too strict)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 8) // 2),
        help="Parallel workers for image scoring",
    )
    return parser.parse_args()


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
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                pil = pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32) / 255.0
            return arr
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

    h = gray.shape[0]
    lower = gray[int(h * 0.45) :, :] if h > 10 else gray
    if lower.size == 0:
        lower = gray
    gx = np.diff(lower, axis=1, append=lower[:, -1:])
    gy = np.diff(lower, axis=0, append=lower[-1:, :])
    grad = np.hypot(gx, gy)
    open_ratio = float(np.mean(grad < 0.03))

    return {
        "edge_density": edge_density,
        "laplacian_var": lap_var,
        "open_ratio": open_ratio,
    }


def aggregate_scene_visual(
    scene_id: str,
    image_paths: List[Path],
    data_root: Path,
    sample_images: int,
    max_side: int,
) -> Dict[str, object] | None:
    sampled = sample_uniform(image_paths, sample_images)
    if not sampled:
        return None

    stats: List[Dict[str, float]] = []
    for p in sampled:
        gray = load_gray(p, max_side=max_side)
        if gray is None:
            continue
        stats.append(image_metrics(gray))

    if not stats:
        return None

    edge_vals = np.array([s["edge_density"] for s in stats], dtype=np.float32)
    lap_vals = np.array([s["laplacian_var"] for s in stats], dtype=np.float32)
    open_vals = np.array([s["open_ratio"] for s in stats], dtype=np.float32)

    representative_image = sampled[int(np.argmin(edge_vals))]

    return {
        "scene_id": scene_id,
        "images_total": len(image_paths),
        "images_sampled": len(stats),
        "edge_density": float(np.median(edge_vals)),
        "laplacian_var": float(np.median(lap_vals)),
        "open_ratio": float(np.median(open_vals)),
        "representative_image": representative_image.as_posix(),
        "data_root": data_root.as_posix(),
    }


def robust_norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo, hi = np.percentile(values, [5, 95])
    if hi <= lo + 1e-12:
        return np.zeros_like(values)
    out = (values - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def scene_type_bonus(scene_type: str) -> float:
    return SCENE_TYPE_BONUS.get(scene_type.lower(), 0.0)


def load_scene_types(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def scan_interaction_labels(scene_dir: Path) -> Dict[str, float]:
    anno = scene_dir / "scans" / "segments_anno.json"
    if not anno.is_file():
        return {
            "chair_count": 0,
            "interaction_count": 0,
            "small_clutter_count": 0,
            "generic_object_count": 0,
            "annotated_object_count": 0,
            "meta_bonus": 0.0,
        }

    try:
        text = anno.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {
            "chair_count": 0,
            "interaction_count": 0,
            "small_clutter_count": 0,
            "generic_object_count": 0,
            "annotated_object_count": 0,
            "meta_bonus": 0.0,
        }

    labels = [lbl.strip().lower() for lbl in LABEL_RE.findall(text)]
    annotated_count = len(labels)
    if annotated_count == 0:
        return {
            "chair_count": 0,
            "interaction_count": 0,
            "small_clutter_count": 0,
            "generic_object_count": 0,
            "annotated_object_count": 0,
            "meta_bonus": 0.0,
        }

    chair_count = sum(1 for l in labels if l in CHAIR_LABELS)
    interaction_count = sum(1 for l in labels if l in INTERACTION_LABELS)
    small_clutter_count = sum(1 for l in labels if l in SMALL_CLUTTER_LABELS)
    generic_object_count = sum(1 for l in labels if l in {"object", "objects"})

    chair_bonus = min(chair_count / 6.0, 1.0) * 0.14
    interaction_bonus = min(interaction_count / 15.0, 1.0) * 0.06
    small_clutter_penalty = min(small_clutter_count / 90.0, 1.0) * 0.09
    dense_scene_penalty = min(max(annotated_count - 220, 0) / 320.0, 1.0) * 0.12
    generic_penalty = min(generic_object_count / 40.0, 1.0) * 0.08

    meta_bonus = chair_bonus + interaction_bonus - small_clutter_penalty - dense_scene_penalty - generic_penalty

    return {
        "chair_count": chair_count,
        "interaction_count": interaction_count,
        "small_clutter_count": small_clutter_count,
        "generic_object_count": generic_object_count,
        "annotated_object_count": annotated_count,
        "meta_bonus": float(meta_bonus),
    }


def discover_scenes(data_root: Path) -> List[Tuple[str, Path, List[Path]]]:
    scenes: List[Tuple[str, Path, List[Path]]] = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        img_dir = scene_dir / "dslr" / "resized_undistorted_images"
        if not img_dir.is_dir():
            continue
        image_paths = list_images(img_dir)
        if not image_paths:
            continue
        scenes.append((scene_dir.name, scene_dir, image_paths))
    return scenes


def output_csv_path(args: argparse.Namespace) -> Path:
    if args.output_csv is not None:
        return args.output_csv
    return args.output_dir / "hsi_scene_ranking.csv"


def resolve_image_path(image_path: object, data_root: Path) -> Path | None:
    p = Path(str(image_path))
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        if p.parts and p.parts[0] == data_root.name:
            candidates.append(data_root.parent / p)
        if p.parts and p.parts[0] == "data":
            candidates.append(data_root / Path(*p.parts[1:]))

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
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k, "") for k in fields})

    for row in selected:
        scene_id = str(row.get("scene_id", "scene"))
        src = resolve_image_path(row.get("representative_image", ""), data_root)
        if src is None:
            continue
        suffix = src.suffix if src.suffix else ".jpg"
        shutil.copy2(src, image_dir / f"{scene_id}{suffix}")

    return export_dir


def main() -> None:
    args = parse_args()
    output_csv = output_csv_path(args)

    if not args.data_root.is_dir():
        raise SystemExit(f"data root not found: {args.data_root}")

    scene_types = load_scene_types(args.scene_types)
    scenes = discover_scenes(args.data_root)
    if not scenes:
        raise SystemExit("No scenes with dslr/resized_undistorted_images found.")

    print(f"Discovered {len(scenes)} scenes with undistorted DSLR images.")

    records: List[Dict[str, object]] = []
    total = len(scenes)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(
                aggregate_scene_visual,
                scene_id,
                image_paths,
                args.data_root,
                args.sample_images,
                args.max_side,
            ): (scene_id, scene_dir)
            for scene_id, scene_dir, image_paths in scenes
        }
        for idx, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            if rec is not None:
                records.append(rec)
            if idx % 80 == 0 or idx == total:
                print(f"Visual scoring progress: {idx}/{total}")

    if not records:
        raise SystemExit("Failed to compute visual metrics for all scenes.")

    edges = np.array([float(r["edge_density"]) for r in records], dtype=np.float32)
    laps = np.array([float(r["laplacian_var"]) for r in records], dtype=np.float32)
    opens = np.array([float(r["open_ratio"]) for r in records], dtype=np.float32)

    edge_n = robust_norm(edges)
    lap_n = robust_norm(laps)
    open_n = robust_norm(opens)

    for i, rec in enumerate(records):
        visual_clutter = 0.50 * edge_n[i] + 0.35 * lap_n[i] + 0.15 * (1.0 - open_n[i])
        rec["visual_clutter"] = float(visual_clutter)
        rec["base_score"] = float(1.0 - visual_clutter)

        scene_id = str(rec["scene_id"])
        stype = scene_types.get(scene_id, "unknown")
        rec["scene_type"] = stype
        rec["type_bonus"] = scene_type_bonus(stype)
        rec["pre_meta_score"] = float(rec["base_score"]) + float(rec["type_bonus"])

    pre_ranked = sorted(records, key=lambda r: float(r["pre_meta_score"]), reverse=True)
    candidate_n = min(len(pre_ranked), max(args.metadata_candidates, args.top_k))
    candidate_ids = {str(r["scene_id"]) for r in pre_ranked[:candidate_n]}

    print(f"Running label-based suitability scan for {len(candidate_ids)} candidate scenes...")

    for rec in records:
        scene_id = str(rec["scene_id"])
        if scene_id in candidate_ids:
            meta = scan_interaction_labels(args.data_root / scene_id)
        else:
            meta = {
                "chair_count": 0,
                "interaction_count": 0,
                "small_clutter_count": 0,
                "generic_object_count": 0,
                "annotated_object_count": 0,
                "meta_bonus": 0.0,
            }

        rec.update(meta)
        rec["final_score"] = (
            float(rec["base_score"])
            + float(rec["type_bonus"])
            + float(rec["meta_bonus"])
        )

    ranked = sorted(
        records,
        key=lambda r: (
            float(r["final_score"]),
            float(r["open_ratio"]),
            -float(r["visual_clutter"]),
        ),
        reverse=True,
    )

    preferred = [
        r
        for r in ranked
        if float(r["open_ratio"]) >= args.min_open_ratio and int(r["chair_count"]) >= args.min_chairs
    ]
    if len(preferred) < args.top_k:
        preferred = [r for r in ranked if float(r["open_ratio"]) >= args.min_open_ratio]
    if len(preferred) < args.top_k:
        preferred = ranked

    top = preferred[: args.top_k]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene_id",
        "scene_type",
        "images_total",
        "images_sampled",
        "edge_density",
        "laplacian_var",
        "open_ratio",
        "visual_clutter",
        "base_score",
        "type_bonus",
        "chair_count",
        "interaction_count",
        "small_clutter_count",
        "generic_object_count",
        "annotated_object_count",
        "meta_bonus",
        "final_score",
        "representative_image",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ranked:
            row = {k: r.get(k, "") for k in fields}
            writer.writerow(row)

    export_dir = write_selected_export(top, fields, output_csv.parent, args.data_root)

    print("\nTop scene recommendations:")
    for i, r in enumerate(top, start=1):
        print(
            f"{i}. {r['scene_id']} (type={r['scene_type']}, "
            f"score={float(r['final_score']):.4f}, "
            f"clutter={float(r['visual_clutter']):.4f}, "
            f"open={float(r['open_ratio']):.4f}, chairs={int(r['chair_count'])})"
        )

    print(f"\nSaved ranking CSV to: {output_csv}")
    print(f"Saved selected export to: {export_dir}")


if __name__ == "__main__":
    main()
