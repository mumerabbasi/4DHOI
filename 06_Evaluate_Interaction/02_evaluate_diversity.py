from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OUTPUT_MODES = ("output", "output_round1", "output_init")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_path(raw_path: str | None, default_path: Path) -> Path:
    return default_path.resolve() if raw_path is None else Path(raw_path).resolve()


def discover_optimized_param_paths(optimization_output_root: Path) -> list[tuple[str, Path]]:
    if not optimization_output_root.is_dir():
        raise FileNotFoundError(
            f"Optimization output directory not found: {optimization_output_root}"
        )

    items: list[tuple[str, Path]] = []
    for interaction_dir in sorted(optimization_output_root.iterdir()):
        if not interaction_dir.is_dir():
            continue
        params_path = interaction_dir / "debug" / "params" / "optimized_frame_0000.pt"
        if params_path.exists():
            items.append((interaction_dir.name, params_path))
    if not items:
        raise RuntimeError(
            f"No optimized_frame_0000.pt files found under {optimization_output_root}."
        )
    return items


def discover_initial_param_paths(human_pose_output_root: Path) -> list[tuple[str, Path]]:
    if not human_pose_output_root.is_dir():
        raise FileNotFoundError(
            f"Human pose output directory not found: {human_pose_output_root}"
        )

    items: list[tuple[str, Path]] = []
    for interaction_dir in sorted(human_pose_output_root.iterdir()):
        if not interaction_dir.is_dir():
            continue
        params_path = interaction_dir / "hmr4d_results.pt"
        mesh_path = interaction_dir / "first_frame_smplx_world.ply"
        if params_path.exists() and mesh_path.exists():
            items.append((interaction_dir.name, params_path))
    if not items:
        raise RuntimeError(
            f"No hmr4d_results.pt files found under {human_pose_output_root}."
        )
    return items


def flatten_numeric(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    return array.reshape(-1)


def load_smplx_parameter_vector(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = ("transl", "global_orient", "body_pose", "betas", "scale")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    parts = [flatten_numeric(payload[key]) for key in required_keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


def load_initial_smplx_parameter_vector(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    params = payload.get("smpl_params_incam")
    if not isinstance(params, dict):
        raise KeyError(f"Missing smpl_params_incam in {path}")
    required_keys = ("transl", "global_orient", "body_pose", "betas")
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    parts = [flatten_numeric(params[key][0]) for key in required_keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def initialize_centers(features: np.ndarray, num_clusters: int) -> np.ndarray:
    # Deterministic farthest-point initialization avoids random result drift.
    centers = [features[0]]
    min_sq_dists = np.sum((features - centers[0][None]) ** 2, axis=1)
    for _ in range(1, num_clusters):
        next_index = int(np.argmax(min_sq_dists))
        centers.append(features[next_index])
        sq_dists = np.sum((features - features[next_index][None]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
    return np.stack(centers, axis=0).astype(np.float32)


def run_kmeans(
    features: np.ndarray,
    num_clusters: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.ndim != 2:
        raise ValueError(f"Expected features with shape [N, D], got {features.shape}.")
    if not 1 <= int(num_clusters) <= features.shape[0]:
        raise ValueError(
            f"num_clusters must be in [1, {features.shape[0]}], got {num_clusters}."
        )

    centers = initialize_centers(features, int(num_clusters))
    labels = np.zeros((features.shape[0],), dtype=np.int64)
    for _ in range(int(max_iters)):
        sq_dists = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(sq_dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

        new_centers = centers.copy()
        for cluster_id in range(int(num_clusters)):
            members = features[labels == cluster_id]
            if members.shape[0] > 0:
                new_centers[cluster_id] = members.mean(axis=0)
        centers = new_centers

    sq_dists = np.sum((features - centers[labels]) ** 2, axis=1)
    distances = np.sqrt(np.maximum(sq_dists, 0.0))
    return labels, centers, distances


def compute_entropy(labels: np.ndarray, num_clusters: int) -> float:
    counts = np.bincount(labels.astype(np.int64), minlength=int(num_clusters))
    probs = counts.astype(np.float64) / float(labels.shape[0])
    nonzero = probs > 0.0
    return float(-np.sum(probs[nonzero] * np.log(probs[nonzero])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate interaction diversity by clustering optimized SMPL-X "
            "parameter vectors, following the GenZI entropy/cluster-size metrics."
        )
    )
    parser.add_argument(
        "--output_mode",
        choices=OUTPUT_MODES,
        default="output",
        help=(
            "Choose the matching optimization/evaluation output set. "
            "'output' uses 05_Optimize_Static_Scene/output and writes to "
            "06_Evaluate_Interaction/output by default; 'output_round1' "
            "uses/writes the output_round1 ablation folders; 'output_init' "
            "clusters module-04 first-frame SMPL-X parameters and writes to "
            "06_Evaluate_Interaction/output_init."
        ),
    )
    parser.add_argument(
        "--optimization_output_root",
        type=str,
        default=None,
        help=(
            "Defaults to 05_Optimize_Static_Scene/<output_mode>, or "
            "04_Estimate_Human_Pose/output for output_init."
        ),
    )
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_clusters", type=int, default=15)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument(
        "--no_standardize",
        action="store_true",
        help="Cluster raw parameter vectors instead of z-scored dimensions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_source_root = (
        PROJECT_DIR / "04_Estimate_Human_Pose" / "output"
        if args.output_mode == "output_init"
        else PROJECT_DIR / "05_Optimize_Static_Scene" / args.output_mode
    )
    optimization_output_root = resolve_path(
        args.optimization_output_root,
        default_source_root,
    )
    output_root = ensure_dir(
        resolve_path(args.output_root, SCRIPT_DIR / args.output_mode)
    )

    param_items = (
        discover_initial_param_paths(optimization_output_root)
        if args.output_mode == "output_init"
        else discover_optimized_param_paths(optimization_output_root)
    )
    interaction_names = [name for name, _path in param_items]
    param_vectors = np.stack(
        [
            (
                load_initial_smplx_parameter_vector(path)
                if args.output_mode == "output_init"
                else load_smplx_parameter_vector(path)
            )
            for _name, path in param_items
        ],
        axis=0,
    )
    features = param_vectors if args.no_standardize else standardize_features(param_vectors)
    num_clusters = min(int(args.num_clusters), features.shape[0])
    labels, centers, distances = run_kmeans(
        features=features,
        num_clusters=num_clusters,
        max_iters=int(args.kmeans_iters),
    )

    entropy = compute_entropy(labels, num_clusters=num_clusters)
    cluster_size = float(np.mean(distances))
    counts = np.bincount(labels.astype(np.int64), minlength=num_clusters)

    csv_path = output_root / "diversity.csv"
    json_path = output_root / "diversity.json"
    rows = [
        {
            "num_samples": int(features.shape[0]),
            "num_clusters": int(num_clusters),
            "entropy": entropy,
            "cluster_size": cluster_size,
        }
    ]
    save_csv_rows(
        csv_path,
        rows,
        fieldnames=["num_samples", "num_clusters", "entropy", "cluster_size"],
    )
    save_json(
        json_path,
        {
            "num_samples": int(features.shape[0]),
            "num_clusters": int(num_clusters),
            "entropy": entropy,
            "cluster_size": cluster_size,
            "standardized": not bool(args.no_standardize),
            "cluster_counts": counts.astype(int).tolist(),
            "assignments": [
                {
                    "interaction_name": interaction_name,
                    "cluster_id": int(cluster_id),
                    "distance_to_cluster_center": float(distance),
                }
                for interaction_name, cluster_id, distance in zip(
                    interaction_names,
                    labels.tolist(),
                    distances.tolist(),
                )
            ],
        },
    )

    print("Diversity metrics")
    print(f"  num_samples={features.shape[0]}")
    print(f"  num_clusters={num_clusters}")
    print(f"  entropy={entropy:.6f}")
    print(f"  cluster_size={cluster_size:.6f}")
    print(f"  csv={csv_path}")
    print(f"  json={json_path}")


if __name__ == "__main__":
    main()
