"""Shared pytest fixtures."""
import pytest
import torch
import numpy as np

from scc.config import Config
from scc.utils import seed_everything


@pytest.fixture(autouse=True)
def fixed_seed():
    seed_everything(42)
    yield


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def synthetic_image():
    """80×80 grayscale image in [0, 1]."""
    seed_everything(42)
    return torch.rand(80, 80)


@pytest.fixture
def tiny_image():
    """35×35 patch already normalised."""
    seed_everything(42)
    return torch.rand(35, 35)


@pytest.fixture
def two_bump_positions():
    """Two distinct (row, col) positions on the 28×28 sheet."""
    return [(7, 7), (21, 21)]


@pytest.fixture
def kclass_spike_dataset(k: int = 10):
    """Linearly-separable K-class spike dataset.

    Returns (X, y) where X is (n_samples, n_features) float in [0,1]
    and y is (n_samples,) int in [0, K-1].
    """
    k = 10
    n_per_class = 20
    n_features = 25   # matches n_features in L2/3 (one per L4 map)
    rng = torch.Generator().manual_seed(42)

    samples, labels = [], []
    for cls_id in range(k):
        base = torch.zeros(n_features)
        base[cls_id % n_features] = 1.0  # one dominant feature per class
        noise = torch.rand(n_per_class, n_features, generator=rng) * 0.2
        x = base.unsqueeze(0).expand(n_per_class, -1) + noise
        samples.append(x.clamp(0, 1))
        labels.append(torch.full((n_per_class,), cls_id, dtype=torch.long))

    return torch.cat(samples), torch.cat(labels)
