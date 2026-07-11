"""Caltech raw-image online learning with the pretrained L4 frozen."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scc.live_training import run_cli


if __name__ == "__main__":
    run_cli(
        "caltech",
        False,
        "outputs/live_online/caltech_frozen_l4",
    )
