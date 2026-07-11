"""Overlapping saccade generator: 5 fixation crops + movement vectors."""
from __future__ import annotations
from typing import List, Tuple

import torch
from torchvision.transforms.functional import crop


class OverlappingSaccadeGenerator:
    """Generate 5 overlapping 35×35 crops from an 80×80 image.

    Parameters
    ----------
    image_size : int
        Square image side (default 80).
    patch : int
        Square crop side (default 35).
    centers : list of [row, col]
        Fixation centres in (row, col) pixel coordinates.  Default is a
        quincunx pattern: image centre + 4 corners offset by 22 px.
    location_scale : float
        Scalar *L* multiplied into the movement vector ``v = L*(c_i - c_{i-1})``.
    """

    def __init__(
        self,
        image_size: int = 80,
        patch: int = 35,
        centers: List[List[int]] | None = None,
        location_scale: float = 1.0,
    ) -> None:
        self.image_size = image_size
        self.patch = patch
        self.location_scale = location_scale
        half = patch // 2

        if centers is None:
            offset = 22
            cx, cy = image_size // 2, image_size // 2
            centers = [
                [cx, cy],
                [cx - offset, cy - offset],
                [cx - offset, cy + offset],
                [cx + offset, cy - offset],
                [cx + offset, cy + offset],
            ]

        # Clamp so every patch stays in-bounds
        lo, hi = half, image_size - half - 1
        self.centers = [
            [max(lo, min(hi, r)), max(lo, min(hi, c))]
            for r, c in centers
        ]

    # ------------------------------------------------------------------ #

    def __call__(
        self, image: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract patches and movement vectors for all saccades.

        Parameters
        ----------
        image : torch.Tensor
            Shape ``(H, W)`` — single-channel 2-D image.

        Returns
        -------
        list of (patch, v) tuples
            ``patch`` shape ``(patch, patch)``;
            ``v`` shape ``(2,)`` — (row_delta, col_delta) movement vector.
        """
        results = []
        prev_center = None

        for r, c in self.centers:
            # top-left corner for torchvision.crop (top, left)
            top = r - self.patch // 2
            left = c - self.patch // 2

            patch = crop(image.unsqueeze(0), top=top, left=left,
                         height=self.patch, width=self.patch).squeeze(0)

            if prev_center is None:
                v = torch.zeros(2, dtype=torch.float32)
            else:
                dr = r - prev_center[0]
                dc = c - prev_center[1]
                v = torch.tensor([dr, dc], dtype=torch.float32) * self.location_scale

            results.append((patch, v))
            prev_center = [r, c]

        return results

    @property
    def n_saccades(self) -> int:
        return len(self.centers)
