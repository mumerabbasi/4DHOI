#!/usr/bin/env python3
"""Evaluate diversity over optimized PROX SMPL-X parameters."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from prox_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    discover_prox_interactions,
    ensure_dir,
    prox_eval_root,
    prox_output_root,
    save_csv_rows,
    save_json,
)


def flatten_numeric(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_smplx_parameter_vector(path: Path) -> np.ndarray:
    with path.open("rb") as file_obj:
        payload = pickle.load(file_obj, encoding="latin1")
    required = ("transl", "global_orient", "body_pose", "betas")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    # Match PhySIC's feature definition exactly; PROX has a fixed unit scale.
    parts = [flatten_numeric(payload[key]) for key in required]
    parts.append(np.asarray([1.0], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / np.maximum(std, 1e-8)


def initialize_centers(features: np.ndarray, num_clusters: int) -> np.ndarray:
    centers = [features[0]]
    min_sq_dists = np.sum((features - centers[0][None]) ** 2, axis=1)
    for _ in range(1, num_clusters):
        next_index = int(np.argmax(min_sq_dists))
        centers.append(features[next_index])
        sq_dists = np.sum((features - features[next_index][None]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, sq_dists)
    return np.stack(centers).astype(np.float32)


def run_kmeans(
    features: np.ndarray,
    num_clusters: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 1 <= num_clusters <= features.shape[0]:
        raise ValueError(f"Invalid cluster count {num_clusters} for {features.shape[0]} samples.")
    centers = initialize_centers(features, num_clusters)
    labels = np.zeros(features.shape[0], dtype=np.int64)
    for _ in range(max_iters):
        sq_dists = np.sum((features[:, None] - centers[None]) ** 2, axis=2)
        new_labels = np.argmin(sq_dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for cluster_id in range(num_clusters):
            members = features[labels == cluster_id]
            if members.shape[0]:
                centers[cluster_id] = members.mean(axis=0)
    sq_dists = np.sum((features - centers[labels]) ** 2, axis=1)
    return labels, centers, np.sqrt(np.maximum(sq_dists, 0.0))


def compute_entropy(labels: np.ndarray, num_clusters: int) -> float:
    counts = np.bincount(labels, minlength=num_clusters)
    probabilities = counts.astype(np.float64) / labels.shape[0]
    nonzero = probabilities > 0
    return float(-np.sum(probabilities[nonzero] * np.log(probabilities[nonzero])))


def discover_param_paths(output_mode: str) -> list[tuple[str, Path]]:
    root = prox_output_root(output_mode)
    items = [
        (name, root / name / "result.pkl")
        for name in discover_prox_interactions(output_mode)
        if (root / name / "result.pkl").is_file()
    ]
    if not items:
        raise RuntimeError(f"No PROX result.pkl files found under {root}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PROX interaction diversity from saved SMPL-X parameters."
    )
    parser.add_argument("--output_mode", default=DEFAULT_OUTPUT_MODE)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_clusters", type=int, default=15)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--no_standardize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = discover_param_paths(args.output_mode)
    names = [name for name, _ in items]
    raw = np.stack([load_smplx_parameter_vector(path) for _, path in items])
    features = raw if args.no_standardize else standardize_features(raw)
    num_clusters = min(int(args.num_clusters), features.shape[0])
    labels, _centers, distances = run_kmeans(
        features, num_clusters, int(args.kmeans_iters)
    )
    counts = np.bincount(labels, minlength=num_clusters)
    summary = {
        "num_samples": int(features.shape[0]),
        "num_clusters": int(num_clusters),
        "entropy": compute_entropy(labels, num_clusters),
        "cluster_size": float(np.mean(distances)),
    }
    output_root = ensure_dir(
        Path(args.output_root).resolve() if args.output_root else prox_eval_root(args.output_mode)
    )
    save_json(
        output_root / "diversity.json",
        {
            **summary,
            "standardized": not args.no_standardize,
            "feature_definition": ["transl", "global_orient", "body_pose", "betas", "scale=1"],
            "cluster_counts": counts.astype(int).tolist(),
            "assignments": [
                {
                    "interaction_name": name,
                    "cluster_id": int(label),
                    "distance_to_cluster_center": float(distance),
                }
                for name, label, distance in zip(names, labels, distances)
            ],
        },
    )
    save_csv_rows(
        output_root / "diversity.csv",
        [summary],
        ["num_samples", "num_clusters", "entropy", "cluster_size"],
    )
    print(
        f"PROX diversity: samples={summary['num_samples']} "
        f"clusters={summary['num_clusters']} entropy={summary['entropy']:.6f} "
        f"cluster_size={summary['cluster_size']:.6f}"
    )


if __name__ == "__main__":
    main()
