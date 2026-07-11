"""MNIST online L4 STDP followed by raw-image cortical learning."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scc.live_training import run_cli


if __name__ == "__main__":
    run_cli(
        "mnist",
        True,
        "outputs/live_online/mnist_trained_l4",
    )
