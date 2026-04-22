"""Model loading, logit-difference metric, and fact filtering for GPT-2 small."""
from __future__ import annotations

from typing import TypedDict

import torch
from transformer_lens import HookedTransformer


class Fact(TypedDict):
    prompt: str
    correct: str
    counterfactual: str


def load_model(device: str | None = None) -> HookedTransformer:
    """Load GPT-2 small via transformer_lens in eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    return model


def _single_token_id(model: HookedTransformer, text: str) -> int:
    """Return the single token id for `text`. Raises if it is not one token."""
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        raise ValueError(f"Expected single token for {text!r}, got {ids.shape[0]}")
    return int(ids[0].item())


@torch.no_grad()
def logit_diff(
    model: HookedTransformer,
    prompt: str,
    correct: str,
    counterfactual: str,
) -> torch.Tensor:
    """logit[correct] - logit[counterfactual] at the final position."""
    correct_id = _single_token_id(model, correct)
    cf_id = _single_token_id(model, counterfactual)
    tokens = model.to_tokens(prompt)
    logits = model(tokens)  # [1, seq, vocab]
    final = logits[0, -1]
    return final[correct_id] - final[cf_id]


@torch.no_grad()
def filter_known_facts(
    model: HookedTransformer,
    facts: list[Fact],
    min_logit_diff: float = 0.0,
) -> list[Fact]:
    """Keep facts where baseline logit_diff(correct, counterfactual) > min_logit_diff.

    This is looser than "top-1 must be the correct token": for small models like
    GPT-2, many factual prompts have distractors like " now" / " the" / " a"
    that outrank the factual answer. What matters for ablation is whether the
    model's final logits already prefer the correct answer over the
    counterfactual — if so, ablating a relevant component should shrink that
    preference.
    """
    kept: list[Fact] = []
    for f in facts:
        try:
            correct_id = _single_token_id(model, f["correct"])
            cf_id = _single_token_id(model, f["counterfactual"])
        except ValueError:
            continue
        tokens = model.to_tokens(f["prompt"])
        logits = model(tokens)[0, -1]
        ld = (logits[correct_id] - logits[cf_id]).item()
        if ld > min_logit_diff:
            kept.append(f)
    return kept
