"""DoG filter construction and application."""
from __future__ import annotations
import torch
from conex.helpers.filters import DoGFilter


def build_dog_kernel(size: int = 17, sigma_1: float = 8.0, sigma_2: float = 2.0) -> torch.Tensor:
    """Return a 4-D DoG kernel (1,1,size,size) ready for Conv2d.

    Parameters
    ----------
    size : int
        Kernel side length (must be odd for symmetric DoG).
    sigma_1 : float
        Outer (broad) Gaussian std — creates the surround.
    sigma_2 : float
        Inner (narrow) Gaussian std — creates the centre-ON response.

    Returns
    -------
    torch.Tensor
        Shape ``(1, 1, size, size)``.  Sum ≈ 0 (zero_mean=True).
    """
    kernel = DoGFilter(size=size, sigma_1=sigma_1, sigma_2=sigma_2,
                       zero_mean=True, one_sum=False)
    return kernel.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)


def apply_dog(tensor: torch.Tensor, kernel: torch.Tensor | None = None,
              size: int = 17, sigma_1: float = 8.0, sigma_2: float = 2.0) -> torch.Tensor:
    """Apply DoG filter to a 2-D image tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Grayscale image in [0, 1] with shape ``(H, W)`` or ``(1, H, W)``.
    kernel : torch.Tensor, optional
        Pre-built ``(1,1,kH,kW)`` kernel.  Built on the fly if None.

    Returns
    -------
    torch.Tensor
        Filtered image, same spatial shape as input.
    """
    import torch.nn.functional as F

    if kernel is None:
        kernel = build_dog_kernel(size, sigma_1, sigma_2)

    was_2d = tensor.dim() == 2
    if was_2d:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)               # (1,1,H,W)

    kernel = kernel.to(tensor.device, tensor.dtype)
    pad = kernel.shape[-1] // 2
    out = F.conv2d(tensor, kernel, padding=pad)

    if was_2d:
        out = out.squeeze(0).squeeze(0)
    elif out.dim() == 4:
        out = out.squeeze(0)
    return out
