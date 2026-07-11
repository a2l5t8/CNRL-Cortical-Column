"""Dataset and preprocessing configuration for the fully spiking column."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ImageConfig:
    """Image preprocessing and foveal patch geometry."""

    image_size: int = 80
    dog_size: int = 17
    dog_sigma_1: float = 8.0
    dog_sigma_2: float = 2.0
    patch_size: int = 35
    n_saccades: int = 5
    location_scale: float = 1.0
    dog_output: str = "on"  # "on", "magnitude", or legacy dense "shift"
    dog_percentile: float = 0.99


@dataclass
class SaccadeConfig:
    """Default five-saccade quincunx used by small diagnostics."""

    centers: List[List[int]] = field(
        default_factory=lambda: [
            [40, 40],
            [22, 22],
            [22, 58],
            [58, 22],
            [58, 58],
        ]
    )


@dataclass
class TrainingConfig:
    """Dataset split settings shared by experiment scripts."""

    caltech_train_split: float = 0.8
    n_seeds: int = 5
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])


@dataclass
class PathConfig:
    """Local dataset and output roots."""

    dataset_path: str = "./data"
    caltech_path: str = "./data/caltech"
    mnist_path: str = "./data/mnist"
    output_dir: str = "./outputs"
    figures_dir: str = "./outputs/figures"
    checkpoints_dir: str = "./outputs/checkpoints"


@dataclass
class Config:
    """Master config for data and experiment bookkeeping.

    Fully spiking network dynamics live in
    :class:`scc.fully_spiking.FullySpikingConfig`. This class intentionally no
    longer exposes the old gradient/readout column hyperparameters.
    """

    image: ImageConfig = field(default_factory=ImageConfig)
    saccade: SaccadeConfig = field(default_factory=SaccadeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    n_classes: int = 10
    device: str = "auto"

    @classmethod
    def mnist(cls) -> "Config":
        """Config preset for 10-class MNIST."""
        cfg = cls()
        cfg.n_classes = 10
        return cfg

    @classmethod
    def caltech(cls) -> "Config":
        """Config preset for 2-class Caltech-101 faces vs. motorbikes."""
        cfg = cls()
        cfg.n_classes = 2
        return cfg
