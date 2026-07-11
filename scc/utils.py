"""Seeding, device detection, checkpoint utilities, logging setup."""
from __future__ import annotations
import logging
import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("scc")


log = setup_logging()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Resolve device string: 'auto' picks CUDA if available, else CPU."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def ensure_dirs(*paths: Union[str, Path]) -> None:
    """Create directories if they do not exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


# ── Checkpointing ────────────────────────────────────────────────────────────

def save_weights(weights: torch.Tensor, path: Union[str, Path]) -> None:
    """Save a raw weight tensor to disk."""
    path = Path(path)
    ensure_dirs(path.parent)
    torch.save(weights, path)
    log.info("Saved weights → %s", path)


def load_weights(path: Union[str, Path], device: torch.device = None) -> torch.Tensor:
    """Load a weight tensor from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    w = torch.load(path, map_location=device or torch.device("cpu"))
    log.info("Loaded weights ← %s", path)
    return w


def save_structure_json(net, path: Union[str, Path]) -> None:
    """Persist a CoNeX network structure to JSON using conex replication utils."""
    try:
        from conex.nn.utils.replication import save_structure_dict_to_json
        path = Path(path)
        ensure_dirs(path.parent)
        save_structure_dict_to_json(net, str(path))
        log.info("Saved network structure → %s", path)
    except Exception as exc:
        log.warning("save_structure_json failed (%s); skipping.", exc)


def load_structure_json(path: Union[str, Path], net):
    """Load a CoNeX network structure from JSON."""
    try:
        from conex.nn.utils.replication import load_structure_dict_from_json
        return load_structure_dict_from_json(str(path), net)
    except Exception as exc:
        log.warning("load_structure_json failed (%s); returning None.", exc)
        return None
