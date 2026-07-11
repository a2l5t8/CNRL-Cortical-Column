"""Tests for overlapping saccade generator."""
import torch
import pytest
from scc.data.saccades import OverlappingSaccadeGenerator
from scc.config import Config


def test_exactly_five_saccades(synthetic_image):
    gen = OverlappingSaccadeGenerator()
    saccades = gen(synthetic_image)
    assert len(saccades) == 5, f"Expected 5, got {len(saccades)}"


def test_patch_shape(synthetic_image):
    gen = OverlappingSaccadeGenerator()
    for patch, _ in gen(synthetic_image):
        assert patch.shape == (35, 35), f"Bad patch shape {patch.shape}"


def test_patches_overlap(synthetic_image):
    """Adjacent patches must share some pixels (overlapping)."""
    gen = OverlappingSaccadeGenerator()
    saccades = gen(synthetic_image)
    patches = [p for p, _ in saccades]
    found_overlap = False
    for i in range(len(patches) - 1):
        # Simple heuristic: same pixel value appears in both
        common = set(patches[i].reshape(-1).tolist()) & set(patches[i+1].reshape(-1).tolist())
        if len(common) > 5:
            found_overlap = True
            break
    assert found_overlap, "Adjacent patches should overlap"


def test_movement_vectors(synthetic_image):
    """v_0 == 0; v_i == L*(c_i - c_{i-1})."""
    L = 1.0
    gen = OverlappingSaccadeGenerator(location_scale=L)
    saccades = gen(synthetic_image)
    centers = gen.centers

    vels = [v for _, v in saccades]
    # First velocity must be zero
    assert vels[0].abs().max().item() == 0.0, "First velocity must be zero"

    # Subsequent velocities: v_i = L*(c_i - c_{i-1})
    for i in range(1, len(centers)):
        expected = torch.tensor([
            centers[i][0] - centers[i-1][0],
            centers[i][1] - centers[i-1][1],
        ], dtype=torch.float32) * L
        assert torch.allclose(vels[i], expected), (
            f"Velocity mismatch at saccade {i}: {vels[i]} vs {expected}"
        )


def test_patches_in_bounds(synthetic_image):
    """All patch pixels must be in-bounds (no clamp artifacts)."""
    gen = OverlappingSaccadeGenerator()
    for patch, _ in gen(synthetic_image):
        assert patch.shape == (35, 35)
        # Values should be in [0,1] range (image was rand [0,1])
        assert patch.min() >= 0.0
        assert patch.max() <= 1.0 + 1e-6
