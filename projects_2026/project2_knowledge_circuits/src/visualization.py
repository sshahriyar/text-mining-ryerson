"""Plotting helpers for head and MLP importance arrays."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_head_heatmap(
    importance: np.ndarray,
    title: str = "Attention Head Importance (Δ logit diff when ablated)",
):
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = float(np.abs(importance).max()) if importance.size else 1.0
    im = ax.imshow(importance, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(importance.shape[1]))
    ax.set_yticks(range(importance.shape[0]))
    fig.colorbar(im, ax=ax, label="baseline LD − ablated LD")
    fig.tight_layout()
    return fig


def plot_mlp_bars(
    importance: np.ndarray,
    title: str = "MLP Layer Importance (Δ logit diff when ablated)",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    layers = np.arange(len(importance))
    ax.bar(layers, importance, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("baseline LD − ablated LD")
    ax.set_title(title)
    ax.set_xticks(layers)
    fig.tight_layout()
    return fig


def plot_cumulative_topk_faithfulness(
    k_values: np.ndarray,
    mean_logit_diffs: np.ndarray,
    title: str = "Mean LD vs jointly ablated top-K heads (faithfulness)",
):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, mean_logit_diffs, marker="o", color="darkgreen")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("K (top-K heads removed together)")
    ax.set_ylabel("Mean logit diff (correct − counterfactual)")
    ax.set_title(title)
    ax.set_xticks(k_values)
    fig.tight_layout()
    return fig


def export_top_heads_csv(
    path: str | Path,
    head_mean: np.ndarray,
    head_std: np.ndarray | None = None,
    top_n: int = 25,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_h = head_mean.shape[1]
    flat_m = head_mean.ravel()
    order = np.argsort(-flat_m)
    if head_std is not None:
        flat_s = head_std.ravel()
    rows: list[dict[str, int | float]] = []
    for rank, flat_idx in enumerate(order[:top_n], start=1):
        flat_idx = int(flat_idx)
        L = flat_idx // n_h
        H = flat_idx % n_h
        row: dict[str, int | float] = {
            "rank": rank,
            "layer": int(L),
            "head": int(H),
            "mean_importance": float(flat_m[flat_idx]),
        }
        if head_std is not None:
            row["std_importance"] = float(flat_s[flat_idx])
        rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else ["rank", "layer", "head", "mean_importance"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
