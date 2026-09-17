"""Debug logging and visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from losses import get_scaled_loss_terms
from models import (
    DiagnosticLossResult,
    FRAME_DIAGNOSTIC_TERM_KEYS,
    LossResult,
    LOSS_TERM_KEYS,
)
from utils import ensure_dir


def build_loss_row(
    iteration: int,
    result: LossResult,
) -> dict[str, Any]:
    scaled_terms = get_scaled_loss_terms(result)
    row: dict[str, Any] = {
        "iter": iteration,
        "total": float(result.total.item()),
    }
    for key in LOSS_TERM_KEYS:
        row[f"{key}_weight"] = float(result.weights[key])
        row[f"{key}_raw"] = float(getattr(result, key).item())
        row[f"{key}_scaled"] = float(scaled_terms[key].item())
    return row


def format_loss_log(
    iteration: int,
    total_iterations: int,
    result: LossResult,
) -> list[str]:
    scaled_terms = get_scaled_loss_terms(result)
    weights_fmt = "  ".join(
        f"{key}={result.weights[key]:.4g}" for key in LOSS_TERM_KEYS
    )
    raw_fmt = "  ".join(
        f"{key}={getattr(result, key).item():.5f}" for key in LOSS_TERM_KEYS
    )
    scaled_fmt = "  ".join(
        f"{key}={scaled_terms[key].item():.5f}" for key in LOSS_TERM_KEYS
    )
    return [
        f"  [{iteration:4d}/{total_iterations}] total={result.total.item():.5f}",
        f"      weights: {weights_fmt}",
        f"      raw:     {raw_fmt}",
        f"      scaled:  {scaled_fmt}",
    ]


def build_frame_loss_rows(
    frame_offset: int,
    diagnostic: DiagnosticLossResult,
) -> list[dict[str, Any]]:
    if not diagnostic.per_frame_raw:
        return []

    num_frames = next(iter(diagnostic.per_frame_raw.values())).shape[0]
    per_frame_scaled = {
        key: diagnostic.sequence.weights[key] * values
        for key, values in diagnostic.per_frame_raw.items()
    }
    device = next(iter(diagnostic.per_frame_raw.values())).device
    total_raw = torch.zeros(num_frames, device=device)
    total_scaled = torch.zeros_like(total_raw)
    for key in FRAME_DIAGNOSTIC_TERM_KEYS:
        total_raw = total_raw + diagnostic.per_frame_raw[key]
        total_scaled = total_scaled + per_frame_scaled[key]

    rows: list[dict[str, Any]] = []
    for t in range(num_frames):
        row: dict[str, Any] = {"frame_idx": frame_offset + t}
        for key in FRAME_DIAGNOSTIC_TERM_KEYS:
            row[f"{key}_weight"] = float(diagnostic.sequence.weights[key])
            row[f"{key}_raw"] = float(diagnostic.per_frame_raw[key][t].item())
            row[f"{key}_scaled"] = float(per_frame_scaled[key][t].item())
        row["total_raw_local"] = float(total_raw[t].item())
        row["total_scaled_local"] = float(total_scaled[t].item())
        rows.append(row)
    return rows


def build_final_loss_summary_row(
    best_iter: int,
    result: LossResult,
) -> dict[str, Any]:
    scaled_terms = get_scaled_loss_terms(result)
    row: dict[str, Any] = {
        "best_iter": best_iter,
        "best_total_loss": float(result.total.item()),
        "total_scaled": float(result.total.item()),
        "frame_loss_semantics": "diagnostic_local",
    }
    for key in LOSS_TERM_KEYS:
        row[f"{key}_weight"] = float(result.weights[key])
        row[f"{key}_raw"] = float(getattr(result, key).item())
        row[f"{key}_scaled"] = float(scaled_terms[key].item())
    return row


def save_loss_plot_tree(
    plot_dir: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    total_key: str,
    term_keys: tuple[str, ...] | list[str],
    x_label: str,
    title_prefix: str,
) -> None:
    if not rows:
        return

    ensure_dir(plot_dir)
    raw_dir = plot_dir / "raw"
    scaled_dir = plot_dir / "scaled"
    ensure_dir(raw_dir)
    ensure_dir(scaled_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    xs = [float(r[x_key]) for r in rows]

    def _save_plot(
        keys: list[str],
        labels: list[str],
        title: str,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, label in zip(keys, labels):
            vals = [float(r.get(key, 0.0)) for r in rows]
            ax.plot(xs, vals, linewidth=1.3, label=label)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Loss")
        ax.set_title(title)
        if labels:
            ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=140)
        plt.close(fig)

    raw_keys = [f"{key}_raw" for key in term_keys]
    scaled_keys = [f"{key}_scaled" for key in term_keys]

    _save_plot(
        [total_key],
        ["total"],
        f"{title_prefix} Total Loss",
        plot_dir / "loss_total.png",
    )
    if raw_keys:
        _save_plot(
            raw_keys,
            list(term_keys),
            f"{title_prefix} Raw Loss Terms",
            raw_dir / "loss_all_terms.png",
        )
    if scaled_keys:
        _save_plot(
            scaled_keys,
            list(term_keys),
            f"{title_prefix} Scaled Loss Terms",
            scaled_dir / "loss_all_terms.png",
        )

    for key in term_keys:
        _save_plot(
            [f"{key}_raw"],
            [key],
            f"{title_prefix} Raw: {key}",
            raw_dir / f"{key}.png",
        )
        _save_plot(
            [f"{key}_scaled"],
            [key],
            f"{title_prefix} Scaled: {key}",
            scaled_dir / f"{key}.png",
        )
