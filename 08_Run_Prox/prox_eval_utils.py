#!/usr/bin/env python3
"""Path and serialization helpers for the thin PROX evaluation wrappers."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_MODE = "output"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def save_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_python_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prox_output_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / output_mode


def prox_interaction_root(
    interaction_name: str,
    output_mode: str = DEFAULT_OUTPUT_MODE,
) -> Path:
    return prox_output_root(output_mode) / interaction_name


def prox_eval_root(output_mode: str = DEFAULT_OUTPUT_MODE) -> Path:
    return SCRIPT_DIR / "evaluation" / output_mode


def interaction_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 10**9, name


def discover_prox_interactions(output_mode: str = DEFAULT_OUTPUT_MODE) -> list[str]:
    root = prox_output_root(output_mode)
    names = [
        path.name
        for path in root.glob("interaction_*")
        if path.is_dir()
        and (path / "final_smplx_world.ply").is_file()
        and (path / "result.pkl").is_file()
    ]
    names.sort(key=interaction_sort_key)
    if not names:
        raise RuntimeError(f"No completed PROX interactions found under {root}.")
    return names
