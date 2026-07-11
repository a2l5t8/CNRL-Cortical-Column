"""Dataset and DataLoader factories for MNIST and Caltech-101."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from conex.helpers.transforms.misc import SqueezeTransform
from conex.helpers.transforms.encoders import SimplePoisson

from scc.config import Config
from scc.data.filters import apply_dog, build_dog_kernel

log = logging.getLogger("scc.pipeline")


# ── Normalisation helpers ──────────────────────────────────────────────────

class _ScaleToUnit(torch.nn.Module):
    """Map values to [0, 1] then multiply by ``scale``."""
    def __init__(self, scale: float = 5.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.min()
        rng = x.max()
        if rng > 0:
            x = x / rng
        return x * self.scale


class _PostDoGNorm(torch.nn.Module):
    """Convert a signed DoG response to a sparse non-negative spike rate."""

    def __init__(self, mode: str = "on", percentile: float = 0.99):
        super().__init__()
        if mode not in {"on", "magnitude", "shift"}:
            raise ValueError(f"Unknown DoG output mode: {mode!r}")
        self.mode = mode
        self.percentile = float(percentile)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "shift":
            x = x - x.min()
            return x / x.max().clamp(min=1e-9)

        x = x.abs() if self.mode == "magnitude" else x.clamp(min=0)
        scale = torch.quantile(x.flatten(), self.percentile).clamp(min=1e-9)
        return (x / scale).clamp(0, 1)


class _ApplyDoG(torch.nn.Module):
    """Apply the padded SCC DoG filter while preserving spatial size."""

    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_dog(x, kernel=self.kernel)


# ── Transform pipeline ────────────────────────────────────────────────────

def build_transform(cfg: Config) -> Callable:
    """Return a torchvision Compose that: resize → grayscale → scale×5 → DoG → [0,1]."""
    dog_kernel = build_dog_kernel(
        cfg.image.dog_size, cfg.image.dog_sigma_1, cfg.image.dog_sigma_2
    )
    return transforms.Compose([
        transforms.Resize((cfg.image.image_size, cfg.image.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        _ScaleToUnit(scale=5.0),
        _ApplyDoG(dog_kernel),
        _PostDoGNorm(
            mode=cfg.image.dog_output,
            percentile=cfg.image.dog_percentile,
        ),
        SqueezeTransform(dim=0),   # remove channel → (H, W)
    ])


# ── Poisson encoder ───────────────────────────────────────────────────────

def encode_patch(patch: torch.Tensor, time_window: int, ratio: float) -> torch.Tensor:
    """Poisson-encode a 2-D patch to a binary spike train.

    Parameters
    ----------
    patch : torch.Tensor
        Shape ``(H, W)`` in [0, 1].
    time_window : int
        Number of timesteps.
    ratio : float
        Firing-probability scaling factor.

    Returns
    -------
    torch.Tensor
        Bool tensor shape ``(time_window, H, W)``.
    """
    encoder = SimplePoisson(time_window=time_window, ratio=ratio)
    return encoder(patch)


# ── Caltech-101 subset ────────────────────────────────────────────────────

def _find_caltech_category_dir(root: Path) -> Path | None:
    """Return the directory that contains Caltech category folders, if present."""
    candidates = [
        root / "101_ObjectCategories",
        root / "caltech101" / "101_ObjectCategories",
        root / "caltech-101" / "101_ObjectCategories",
        root / "caltech-101" / "caltech101" / "101_ObjectCategories",
    ]
    for candidate in candidates:
        if (candidate / "Motorbikes").exists() and (
            (candidate / "Faces_easy").exists() or (candidate / "Faces").exists()
        ):
            return candidate
    return None


class CaltechBinaryDataset(Dataset):
    """200 faces + 200 motorcycles from Caltech-101, shuffled, no leakage."""

    FACE_LABEL = 0
    MOTOR_LABEL = 1

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        n_per_class: int = 200,
        seed: int = 42,
        download: bool = True,
    ):
        self.transform = transform
        rng = torch.Generator()
        rng.manual_seed(seed)

        root_path = Path(root)
        category_dir = _find_caltech_category_dir(root_path)
        if category_dir is None and download:
            log.info("Caltech-101 not found under %s; downloading with torchvision.", root_path)
            torchvision.datasets.Caltech101(root=str(root_path), download=True)
            category_dir = _find_caltech_category_dir(root_path)

        if category_dir is None:
            raise FileNotFoundError(
                "Caltech-101 category directory not found under "
                f"{root_path}. Expected 101_ObjectCategories with Faces/Faces_easy "
                "and Motorbikes, or pass download=True."
            )

        face_dir = category_dir / "Faces_easy"
        motor_dir = category_dir / "Motorbikes"

        if not face_dir.exists():
            face_dir = category_dir / "Faces"
        if not face_dir.exists():
            raise FileNotFoundError(f"Caltech face directory not found under {category_dir}")
        if not motor_dir.exists():
            raise FileNotFoundError(f"Caltech motor directory not found under {category_dir}")

        face_files = sorted(face_dir.glob("*.jpg"))[:n_per_class]
        motor_files = sorted(motor_dir.glob("*.jpg"))[:n_per_class]
        if not face_files or not motor_files:
            raise FileNotFoundError(
                f"Caltech binary subset is empty under {category_dir}: "
                f"{len(face_files)} face images, {len(motor_files)} motorbike images."
            )

        self.samples: list[Tuple[Path, int]] = (
            [(f, self.FACE_LABEL) for f in face_files] +
            [(f, self.MOTOR_LABEL) for f in motor_files]
        )
        # Shuffle deterministically
        idx = torch.randperm(len(self.samples), generator=rng).tolist()
        self.samples = [self.samples[i] for i in idx]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Factory functions ─────────────────────────────────────────────────────

def make_mnist_loaders(
    cfg: Config,
    batch_size: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for MNIST."""
    tfm = build_transform(cfg)
    root = cfg.paths.mnist_path
    full = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tfm)
    n_train = int(0.8 * len(full))
    n_val = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info("MNIST: %d train / %d val", n_train, n_val)
    return train_loader, val_loader


def make_caltech_loaders(
    cfg: Config,
    batch_size: int = 1,
    seed: int = 42,
    n_per_class: int = 200,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for 2-class Caltech-101."""
    tfm = build_transform(cfg)
    ds = CaltechBinaryDataset(cfg.paths.caltech_path, transform=tfm,
                              n_per_class=n_per_class, seed=seed,
                              download=download)
    n_train = int(cfg.training.caltech_train_split * len(ds))
    n_val = len(ds) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    log.info("Caltech: %d train / %d val", n_train, n_val)
    return train_loader, val_loader
