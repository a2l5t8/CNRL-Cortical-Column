"""Dataset and retina configuration, independent of neuronal dynamics."""

from dataclasses import dataclass, field


@dataclass
class ImageConfig:
    image_size: int = 80
    dog_size: int = 17
    dog_sigma_1: float = 8.0
    dog_sigma_2: float = 2.0
    dog_output: str = "on"  # "on", "magnitude", or legacy dense "shift"
    dog_percentile: float = 0.99


@dataclass
class TrainingConfig:
    caltech_train_split: float = 0.8


@dataclass
class PathConfig:
    caltech_path: str = "./data/caltech"
    mnist_path: str = "./data/mnist"


@dataclass
class Config:
    image: ImageConfig = field(default_factory=ImageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def mnist(cls) -> "Config":
        return cls()

    @classmethod
    def caltech(cls) -> "Config":
        return cls()
