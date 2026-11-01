"""Post-DoG normalisation of the retina transform."""
import torch

from scc.data.pipeline import _PostDoGNorm


def test_on_center_dog_normalization_is_sparse_and_bounded():
    response = torch.tensor([[-2.0, -0.5, 0.0], [0.1, 0.5, 3.0]])
    normalized = _PostDoGNorm(mode="on", percentile=0.99)(response)

    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) <= 1.0
    assert torch.count_nonzero(normalized) == 3
