import pytest
import torch

from src.model_utils import filter_known_facts, load_model, logit_diff


@pytest.fixture(scope="module")
def model():
    return load_model(device="cpu")


def test_load_model_returns_gpt2_small(model):
    assert model.cfg.n_layers == 12
    assert model.cfg.n_heads == 12
    assert model.cfg.d_model == 768


def test_logit_diff_returns_scalar(model):
    ld = logit_diff(model, "The capital of France is", " Paris", " Berlin")
    assert ld.ndim == 0
    assert isinstance(ld.item(), float)


def test_logit_diff_sign_on_known_fact(model):
    ld = logit_diff(model, "The capital of France is", " Paris", " Berlin")
    assert ld.item() > 0


def test_logit_diff_sign_flips_when_swapped(model):
    a = logit_diff(model, "The capital of France is", " Paris", " Berlin").item()
    b = logit_diff(model, "The capital of France is", " Berlin", " Paris").item()
    assert a == pytest.approx(-b, abs=1e-4)


def test_filter_keeps_known_and_drops_unknown(model):
    facts = [
        {"prompt": "The capital of France is", "correct": " Paris", "counterfactual": " Berlin"},
        {"prompt": "The capital of Zzzzland is", "correct": " Xyzzy", "counterfactual": " Qwerty"},
    ]
    kept = filter_known_facts(model, facts)
    assert len(kept) == 1
    assert kept[0]["correct"] == " Paris"
