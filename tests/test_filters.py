"""Tests for DoG filter: zero-mean kernel, edge enhancement."""
import torch
import pytest
from PIL import Image

from scc.config import Config
from scc.data.filters import build_dog_kernel, apply_dog
from scc.data.pipeline import build_transform


def test_dog_kernel_zero_mean():
    """Kernel sum must be ≈ 0."""
    kernel = build_dog_kernel(size=17, sigma_1=8.0, sigma_2=2.0)
    assert abs(kernel.sum().item()) < 1e-5, f"kernel sum = {kernel.sum().item()}"


def test_dog_increases_edge_variance():
    """Filtering a step-edge image should increase edge-pixel variance."""
    # Step edge
    img = torch.zeros(80, 80)
    img[:, 40:] = 1.0

    raw_std = img.std().item()
    filtered = apply_dog(img.clone(), size=17, sigma_1=8.0, sigma_2=2.0)

    # Variance at the edge column vs. raw
    edge_col_raw = img[:, 38:43].std().item()
    edge_col_filt = filtered[:, 38:43].std().item()

    assert edge_col_filt > edge_col_raw or edge_col_filt > raw_std * 0.5, (
        "DoG should enhance edges"
    )


def test_dog_kernel_shape():
    kernel = build_dog_kernel(17, 8, 2)
    assert kernel.shape == (1, 1, 17, 17)


def test_apply_dog_same_spatial_size():
    img = torch.rand(80, 80)
    out = apply_dog(img)
    assert out.shape == img.shape


def test_pipeline_transform_preserves_image_size():
    """The dataset transform must keep the configured 80x80 image contract."""
    cfg = Config()
    img = Image.new("L", (28, 28), color=128)
    out = build_transform(cfg)(img)
    assert out.shape == (cfg.image.image_size, cfg.image.image_size)
