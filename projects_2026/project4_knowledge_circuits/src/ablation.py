"""Zero-ablation hooks and sweep loops for attention heads and MLP layers."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from transformer_lens import HookedTransformer

from src.model_utils import Fact, _single_token_id, logit_diff


def make_head_zero_hook(head_idx: int) -> Callable:
    """Zero the slice of attention `z` corresponding to one head.

    z shape: [batch, seq, n_heads, d_head].
    """
    def hook_fn(z: torch.Tensor, hook=None) -> torch.Tensor:
        z[..., head_idx, :] = 0.0
        return z
    return hook_fn


def make_mlp_zero_hook() -> Callable:
    """Zero an MLP layer's output. Shape: [batch, seq, d_model]."""
    def hook_fn(mlp_out: torch.Tensor, hook=None) -> torch.Tensor:
        mlp_out[...] = 0.0
        return mlp_out
    return hook_fn


@torch.no_grad()
def _ablated_logit_diff(
    model: HookedTransformer,
    prompt: str,
    correct: str,
    counterfactual: str,
    hook_name: str,
    hook_fn: Callable,
) -> float:
    correct_id = _single_token_id(model, correct)
    cf_id = _single_token_id(model, counterfactual)
    tokens = model.to_tokens(prompt)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    final = logits[0, -1]
    return float((final[correct_id] - final[cf_id]).item())


def _make_multi_head_z_hook(head_indices: list[int]) -> Callable:
    """Zero several heads' slices in the same layer's hook_z."""
    heads = sorted({int(h) for h in head_indices})

    def hook_fn(z: torch.Tensor, hook=None) -> torch.Tensor:
        for h in heads:
            z[..., h, :] = 0.0
        return z

    return hook_fn


def _logit_diff_from_logits(
    model: HookedTransformer,
    logits: torch.Tensor,
    correct: str,
    counterfactual: str,
) -> float:
    correct_id = _single_token_id(model, correct)
    cf_id = _single_token_id(model, counterfactual)
    final = logits[0, -1]
    return float((final[correct_id] - final[cf_id]).item())


@torch.no_grad()
def mean_logit_diff_joint_head_ablation(
    model: HookedTransformer,
    facts: list[Fact],
    ablated_heads: set[tuple[int, int]],
) -> float:
    """Mean final-token logit diff when jointly zeroing every (layer, head) in ablated_heads."""
    if not ablated_heads:
        return float(
            np.mean(
                [
                    logit_diff(model, f["prompt"], f["correct"], f["counterfactual"]).item()
                    for f in facts
                ]
            )
        )
    by_layer: dict[int, list[int]] = defaultdict(list)
    for L, H in ablated_heads:
        by_layer[int(L)].append(int(H))
    fwd_hooks = [
        (f"blocks.{L}.attn.hook_z", _make_multi_head_z_hook(hs))
        for L, hs in sorted(by_layer.items())
    ]
    vals: list[float] = []
    for f in facts:
        tokens = model.to_tokens(f["prompt"])
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
        vals.append(
            _logit_diff_from_logits(
                model, logits, f["correct"], f["counterfactual"]
            )
        )
    return float(np.mean(vals))


@torch.no_grad()
def cumulative_topk_head_ablation_curve(
    model: HookedTransformer,
    facts: list[Fact],
    head_importance: np.ndarray,
    max_k: int = 25,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean logit diff after jointly ablating the K highest single-head-importance heads.

    Ranks heads by ``head_importance`` descending (same ordering as the per-head sweep).
    Returns ``(k_values, mean_ld)`` for K = 0, 1, ..., max_k (faithfulness).
    """
    n_layers, n_heads = head_importance.shape
    flat = head_importance.ravel()
    order = np.argsort(-flat)
    ranked: list[tuple[int, int]] = []
    for idx in order:
        idx = int(idx)
        L = idx // n_heads
        H = idx % n_heads
        ranked.append((L, H))

    max_k = min(max_k, len(ranked))
    k_values = np.arange(0, max_k + 1, dtype=np.int32)
    mean_lds = np.zeros(max_k + 1, dtype=np.float32)
    ablated: set[tuple[int, int]] = set()
    for k in range(max_k + 1):
        if k > 0:
            ablated.add(ranked[k - 1])
        mean_lds[k] = mean_logit_diff_joint_head_ablation(model, facts, ablated)
        if verbose:
            print(f"joint top-{k} ablation: mean LD = {mean_lds[k]:.4f}")
    return k_values, mean_lds


@torch.no_grad()
def head_importance_sweep(
    model: HookedTransformer,
    facts: list[Fact],
    verbose: bool = True,
    return_std: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Mean importance per (layer, head): baseline_LD - ablated_LD averaged over facts.

    If ``return_std`` is True, also returns per-cell std across prompts.
    """
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    importance = np.zeros((n_layers, n_heads), dtype=np.float32)
    std = np.zeros((n_layers, n_heads), dtype=np.float32)

    baselines = [
        logit_diff(model, f["prompt"], f["correct"], f["counterfactual"]).item()
        for f in facts
    ]

    for L in range(n_layers):
        if verbose:
            print(f"Head sweep: layer {L}/{n_layers - 1}")
        for H in range(n_heads):
            hook_fn = make_head_zero_hook(H)
            hook_name = f"blocks.{L}.attn.hook_z"
            drops = [
                baseline
                - _ablated_logit_diff(
                    model,
                    f["prompt"],
                    f["correct"],
                    f["counterfactual"],
                    hook_name,
                    hook_fn,
                )
                for f, baseline in zip(facts, baselines)
            ]
            importance[L, H] = float(np.mean(drops))
            std[L, H] = float(np.std(drops))
    if return_std:
        return importance, std
    return importance


@torch.no_grad()
def mlp_importance_sweep(
    model: HookedTransformer,
    facts: list[Fact],
    verbose: bool = True,
) -> np.ndarray:
    """Mean importance per MLP layer."""
    n_layers = model.cfg.n_layers
    importance = np.zeros(n_layers, dtype=np.float32)

    baselines = [
        logit_diff(model, f["prompt"], f["correct"], f["counterfactual"]).item()
        for f in facts
    ]

    hook_fn = make_mlp_zero_hook()
    for L in range(n_layers):
        if verbose:
            print(f"MLP sweep: layer {L}/{n_layers - 1}")
        hook_name = f"blocks.{L}.hook_mlp_out"
        drops = [
            baseline - _ablated_logit_diff(
                model, f["prompt"], f["correct"], f["counterfactual"],
                hook_name, hook_fn,
            )
            for f, baseline in zip(facts, baselines)
        ]
        importance[L] = float(np.mean(drops))
    return importance
