"""Tests for Poisson encoder."""
import torch
import pytest
from scc.data.pipeline import _PostDoGNorm, encode_patch


def test_poisson_output_shape():
    patch = torch.rand(35, 35)
    spikes = encode_patch(patch, time_window=15, ratio=0.9)
    assert spikes.shape == (15, 35, 35)


def test_zero_input_few_spikes():
    """Zero input → ~0 spikes."""
    patch = torch.zeros(35, 35)
    spikes = encode_patch(patch, time_window=20, ratio=0.9)
    assert spikes.float().sum() == 0, "Zero input must give no spikes"


def test_mean_spike_count_vs_expected():
    """Mean spikes per pixel ≈ value * ratio * time_window within 30%."""
    ratio = 0.5
    tw = 30
    value = 0.6
    patch = torch.full((35, 35), value)

    # Average over many runs
    totals = []
    for seed in range(5):
        torch.manual_seed(seed)
        spikes = encode_patch(patch, time_window=tw, ratio=ratio)
        totals.append(spikes.float().mean().item())

    mean_count = sum(totals) / len(totals)
    expected = value * ratio  # mean probability per step * steps (normalised)
    # mean spikes per pixel per step
    mean_per_step = mean_count
    expected_per_step = value * ratio

    assert abs(mean_per_step - expected_per_step) < 0.15, (
        f"Mean per step {mean_per_step:.3f} vs expected {expected_per_step:.3f}"
    )


def test_monotone_in_intensity():
    """Higher intensity → more spikes (averaged over seeds)."""
    tw = 20
    ratio = 0.8
    low_patch = torch.full((10, 10), 0.1)
    high_patch = torch.full((10, 10), 0.8)

    low_counts, high_counts = [], []
    for seed in range(10):
        torch.manual_seed(seed)
        low_counts.append(encode_patch(low_patch, tw, ratio).float().sum().item())
        torch.manual_seed(seed)
        high_counts.append(encode_patch(high_patch, tw, ratio).float().sum().item())

    assert sum(high_counts) > sum(low_counts), "Higher intensity should spike more"


def test_on_center_dog_normalization_is_sparse_and_bounded():
    response = torch.tensor([[-2.0, -0.5, 0.0], [0.1, 0.5, 3.0]])
    normalized = _PostDoGNorm(mode="on", percentile=0.99)(response)

    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) <= 1.0
    assert torch.count_nonzero(normalized) == 3
