import numpy as np
import pytest
import torch

from src.ablation import (
    cumulative_topk_head_ablation_curve,
    head_importance_sweep,
    make_head_zero_hook,
    make_mlp_zero_hook,
    mean_logit_diff_joint_head_ablation,
    mlp_importance_sweep,
)
from src.model_utils import load_model


@pytest.fixture(scope="module")
def model():
    return load_model(device="cpu")


def test_head_zero_hook_zeroes_correct_slice():
    hook = make_head_zero_hook(head_idx=3)
    z = torch.randn(1, 4, 12, 64)
    out = hook(z.clone())
    assert torch.all(out[..., 3, :] == 0)
    assert torch.any(out[..., 0, :] != 0)
    assert torch.any(out[..., 11, :] != 0)


def test_mlp_zero_hook_zeroes_entire_output():
    hook = make_mlp_zero_hook()
    x = torch.randn(1, 5, 768)
    out = hook(x.clone())
    assert torch.all(out == 0)


def test_head_sweep_shape_and_nonzero(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp = head_importance_sweep(model, facts, verbose=False)
    assert imp.shape == (12, 12)
    assert imp.any()


def test_head_sweep_return_std(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp, std = head_importance_sweep(model, facts, verbose=False, return_std=True)
    assert imp.shape == (12, 12) and std.shape == (12, 12)
    assert np.all(std >= 0)


def test_joint_ablation_baseline_matches_no_ablation(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    m_joint = mean_logit_diff_joint_head_ablation(model, facts, set())
    from src.model_utils import logit_diff

    m_direct = float(
        logit_diff(model, facts[0]["prompt"], facts[0]["correct"], facts[0]["counterfactual"]).item()
    )
    assert abs(m_joint - m_direct) < 1e-4


def test_cumulative_curve_shape(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp = head_importance_sweep(model, facts, verbose=False)
    k, ld = cumulative_topk_head_ablation_curve(model, facts, imp, max_k=3, verbose=False)
    assert len(k) == 4 and len(ld) == 4
    assert k[0] == 0 and k[-1] == 3


def test_export_top_heads_csv(tmp_path):
    from src.visualization import export_top_heads_csv

    mean = np.arange(12, dtype=np.float32).reshape(3, 4)
    std = np.ones_like(mean)
    out = tmp_path / "top_heads.csv"
    export_top_heads_csv(out, mean, std, top_n=5)
    text = out.read_text()
    assert "rank" in text and "mean_importance" in text and "std_importance" in text


def test_mlp_sweep_shape_and_nonzero(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
    ]
    imp = mlp_importance_sweep(model, facts, verbose=False)
    assert imp.shape == (12,)
    assert imp.any()
