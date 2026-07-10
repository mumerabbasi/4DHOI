#!/usr/bin/env python3
"""Evaluate diversity over PhySIC SMPL-X parameter artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from physic_eval_utils import (
    DEFAULT_OUTPUT_MODE,
    discover_physic_interactions,
    ensure_dir,
    physic_eval_root,
    physic_output_root,
    save_csv_rows,
    save_json,
)


def flatten_numeric(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_smplx_parameter_vector(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = ("transl", "global_orient", "body_pose", "betas", "scale")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")
    parts = [flatten_numeric(payload[key]) for key in required_keys]
    return np.concatenate(parts, axis=0).astype(np.float32)


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
    return np.stack(centers, axis=0).astype(np.float32)


def run_kmeans(
    features: np.ndarray,
    num_clusters: int,
    max_iters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        for cluster_id in range(int(num_clusters)):
            members = features[labels == cluster_id]
            if members.shape[0] > 0:
                centers[cluster_id] = members.mean(axis=0)
    sq_dists = np.sum((features - centers[labels]) ** 2, axis=1)
    return labels, centers, np.sqrt(np.maximum(sq_dists, 0.0))


def compute_entropy(labels: np.ndarray, num_clusters: int) -> float:
    counts = np.bincount(labels.astype(np.int64), minlength=int(num_clusters))
    probs = counts.astype(np.float64) / float(labels.shape[0])
    nonzero = probs > 0.0
    return float(-np.sum(probs[nonzero] * np.log(probs[nonzero])))


def discover_param_paths(output_mode: str) -> list[tuple[str, Path]]:
    items: list[tuple[str, Path]] = []
    for interaction_name in discover_physic_interactions(output_mode):
        params_path = (
            physic_output_root(output_mode)
            / interaction_name
            / "debug"
            / "params"
            / "optimized_frame_0000.pt"
        )
        if params_path.exists():
            items.append((interaction_name, params_path))
    if not items:
        raise RuntimeError(f"No PhySIC optimized params found under {physic_output_root(output_mode)}")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PhySIC interaction diversity from saved SMPL-X params."
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
    names = [name for name, _path in items]
    features_raw = np.stack(
        [load_smplx_parameter_vector(path) for _name, path in items],
        axis=0,
    )
    features = features_raw if args.no_standardize else standardize_features(features_raw)
    num_clusters = min(int(args.num_clusters), features.shape[0])
    labels, centers, distances = run_kmeans(
        features=features,
        num_clusters=num_clusters,
        max_iters=int(args.kmeans_iters),
    )
    counts = np.bincount(labels, minlength=num_clusters)
    entropy = compute_entropy(labels, num_clusters)
    rows = [
        {
            "interaction_name": name,
            "cluster_id": int(label),
            "distance_to_center": float(distance),
        }
        for name, label, distance in zip(names, labels, distances)
    ]
    summary = {
        "num_interactions": int(features.shape[0]),
        "num_clusters": int(num_clusters),
        "entropy": float(entropy),
        "mean_cluster_size": float(np.mean(counts)),
        "min_cluster_size": int(np.min(counts)),
        "max_cluster_size": int(np.max(counts)),
        "standardized": not bool(args.no_standardize),
    }
    output_root = ensure_dir(
        Path(args.output_root).resolve()
        if args.output_root
        else physic_eval_root(args.output_mode) / "diversity"
    )
    save_csv_rows(
        output_root / "assignments.csv",
        rows,
        ["interaction_name", "cluster_id", "distance_to_center"],
    )
    save_json(
        output_root / "metrics.json",
        {
            **summary,
            "cluster_sizes": counts.astype(int).tolist(),
            "centers": centers.astype(float).tolist(),
            "assignments": rows,
        },
    )
    save_csv_rows(
        output_root / "metrics.csv",
        [summary],
        [
            "num_interactions",
            "num_clusters",
            "entropy",
            "mean_cluster_size",
            "min_cluster_size",
            "max_cluster_size",
            "standardized",
        ],
    )
    print(
        f"PhySIC diversity: interactions={summary['num_interactions']} "
        f"clusters={summary['num_clusters']} entropy={summary['entropy']:.6f}"
    )


if __name__ == "__main__":
    main()
